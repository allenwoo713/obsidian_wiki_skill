"""Fail-closed held-out FLAT/SQ ANN comparison used only by evaluation jobs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import statistics
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa


HERE = Path(__file__).resolve().parent
SKILL_ROOT = HERE.parent
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from obsidian_wiki.application.index_build_service import BENCHMARK_MAX_PROBES  # noqa: E402
from obsidian_wiki.domain.index_models import VectorIndexConfig  # noqa: E402
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository  # noqa: E402


EVIDENCE_SCHEMA_VERSION = 5
DECISION_EF_GRID = (30, 50, 75, 100, 150, 200)
CANDIDATES = ("ivf-hnsw-flat", "ivf-hnsw-sq")
_REPOSITORY_TYPES = {"ivf-hnsw-flat": "hnsw_flat", "ivf-hnsw-sq": "hnsw_sq"}
DEFAULT_MAX_EXACT_SECONDS = 10.0
DEFAULT_MAX_WALL_SECONDS = 60.0
CORPUS_SEED = 41001
QUERY_SEED = 41002
_LOCKED_PACKAGES = {"lancedb", "numpy", "pyarrow"}
CALIBRATION_RULE_VERSION = "omp-median-mad-v1"


def _vectors(rows: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((rows, dimensions), dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), np.finfo(np.float32).eps)
    return values.astype("<f4", copy=False)


def _matrix_digest(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _row_hashes(values: np.ndarray) -> set[str]:
    return {hashlib.sha256(np.asarray(row, dtype="<f4").tobytes()).hexdigest() for row in values}


def _arrow_table(vectors: np.ndarray, chunk_ids: list[str]) -> pa.Table:
    column = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()), int(vectors.shape[1])
    )
    return pa.table({"chunk_id": pa.array(chunk_ids), "vector": column})


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _percentile(samples: list[float], percentile: int) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    return float(ordered[round((len(ordered) - 1) * percentile / 100)])


def _config_digest(candidate: str, *, rows: int, dimensions: int) -> str:
    value = {
        "candidate": candidate,
        "metric": "cosine",
        "rows": rows,
        "dimensions": dimensions,
        "m": 16,
        "ef_construction": 300,
        "num_partitions": 1,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _locked_runtime_identity() -> dict[str, str]:
    """Read the three approval-critical versions from the checked-in lock."""
    versions: dict[str, str] = {}
    for line in (SKILL_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        name, separator, version = line.partition("==")
        if separator and name in _LOCKED_PACKAGES:
            versions[name] = version.strip()
    if set(versions) != _LOCKED_PACKAGES:
        raise ValueError("locked runtime identity")
    return versions


def _runtime_identity() -> dict[str, str]:
    return {
        "lancedb": importlib.metadata.version("lancedb"),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
    }


def _head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SKILL_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("source head") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _spawn_worker_schedule() -> dict[str, Any]:
    return {
        "kind": "bounded_process_pool",
        "max_workers": 2,
        "configured_workers": 2,
        "effective_workers": 2,
        "candidate_concurrency": 2,
        "start_method": "spawn",
    }


def _candidate_assignment() -> dict[str, str]:
    return {candidate: f"candidate-run::{candidate}" for candidate in CANDIDATES}


def build_calibration_record(
    *, head_sha: str, lock_identity: dict[str, str], configuration: dict[str, Any],
    repetitions: dict[int, list[float]],
) -> dict[str, Any]:
    """Derive a cap solely from five complete per-query matrices per setting."""
    expected = {1, 2}
    if set(repetitions) != expected or any(len(repetitions[omp]) != 5 for omp in expected):
        raise ValueError("calibration requires five complete repetitions for OMP 1 and OMP 2")
    if any(not _finite(value) or float(value) < 0 for runs in repetitions.values() for value in runs):
        raise ValueError("calibration timing")
    summaries: dict[int, dict[str, float]] = {}
    for omp, runs in repetitions.items():
        values = [float(value) for value in runs]
        median = float(statistics.median(values))
        mad = float(statistics.median([abs(value - median) for value in values]))
        summaries[omp] = {"median_seconds": median, "max_seconds": max(values), "mad_seconds": mad}
    selected = min(expected, key=lambda omp: (summaries[omp]["median_seconds"], summaries[omp]["max_seconds"], omp))
    summary = summaries[selected]
    calculated_cap = math.ceil(max(
        summary["max_seconds"], summary["median_seconds"] + 3 * 1.4826 * summary["mad_seconds"]
    ))
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "non_accepting_calibration",
        "rule_version": CALIBRATION_RULE_VERSION,
        "head_sha": head_sha,
        "lock_identity": lock_identity,
        "configuration": configuration,
        "repetitions": {str(omp): list(repetitions[omp]) for omp in sorted(expected)},
        "summaries": {str(omp): summaries[omp] for omp in sorted(expected)},
        "selected_omp_threads": selected,
        "selection": summary,
        "calculated_cap_seconds": calculated_cap,
    }
    return finalize_calibration_record(record)


def finalize_calibration_record(record: dict[str, Any]) -> dict[str, Any]:
    """Seal the complete calibration record after all diagnostic fields exist."""
    record.pop("sha256", None)
    record["sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return record


def run_multivector_batch_spike(
    table: Any, vectors: list[list[float]], *, metric: str, ef: int, limit: int,
    individual_result_ids: list[list[str]],
) -> dict[str, Any]:
    """Observe LanceDB multi-vector output without changing the acceptance path.

    This is deliberately not called by the per-query comparator.  Until query
    mapping, IDs, recall, and the separate p50/p95 latency contract are proven
    equivalent, its output is diagnostic-only.
    """
    started = time.perf_counter()
    rows = table.search(vectors).distance_type(metric).ef(ef).limit(limit).to_list()
    elapsed_ms = (time.perf_counter() - started) * 1000
    returned_ids_by_query: dict[int, list[str]] = {index: [] for index in range(len(vectors))}
    invalid_row_count = 0
    for row in rows:
        query_index = row.get("query_index") if isinstance(row, dict) else None
        if not isinstance(query_index, int) or not 0 <= query_index < len(vectors):
            invalid_row_count += 1
            continue
        returned_ids_by_query[query_index].append(str(row.get("chunk_id", "")))
    observations = []
    for query_index, expected in enumerate(individual_result_ids):
        ids = returned_ids_by_query[query_index]
        observations.append({
            "query_index": query_index, "returned_ids": ids,
            "individual_ids": expected, "ids_identical": ids == expected,
            "recall": len(set(ids) & set(expected)) / len(expected) if expected else 0.0,
        })
    return {
        "status": "observational_only",
        "method": "lancedb_multi_vector_search",
        "metric": metric, "ef": ef, "limit": limit, "elapsed_ms": elapsed_ms,
        "latency_contract_validated": False,
        "can_substitute_per_query_acceptance": False,
        "returned_row_count": len(rows),
        "invalid_query_index_row_count": invalid_row_count,
        "observations": observations,
    }


def _error_output_path(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "error_output", None) or Path(args.work_dir) / "index-benchmark-error.json")


def _write_rejected_error(
    args: argparse.Namespace,
    exc: Exception,
    *,
    failure_phase: str,
    matrix_started: float | None,
    exact_time_ms: float | None = None,
    wall_seconds: float | None = None,
    raw_staging_path: Path | None = None,
    candidate_runs: list[dict[str, Any]] | None = None,
) -> Path:
    """Persist any rejected comparison without manufacturing approval evidence."""
    output = _error_output_path(args)
    payload = {
        "status": "reject-evidence",
        "error_schema_version": 1,
        "source": {"head_sha": _head_sha(), "lock_identity": _locked_runtime_identity()},
        "runtime": _runtime_identity(),
        "worker_schedule": _spawn_worker_schedule(),
        "candidate_assignment": _candidate_assignment(),
        "matrix_started_monotonic": matrix_started,
        "candidate_processes": [
            {"candidate": run.get("candidate"), "worker": run.get("worker")}
            for run in (candidate_runs or [])
        ],
        "observed": {
            "benchmark_wall_seconds": wall_seconds,
            "exact_time_ms": exact_time_ms,
        },
        "failure_phase": failure_phase,
        "raw_staging_path": str(raw_staging_path) if raw_staging_path else None,
        "error": {
            "class": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        "failed_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def validate_evidence(
    payload: dict[str, Any], *, aggregate_cap_seconds: float | None = None,
    allow_calibration: bool = False,
) -> dict[str, Any]:
    """Reject partial, mixed, self-query, or policy-bearing decision evidence."""
    _require(payload.get("evidence_schema_version") == EVIDENCE_SCHEMA_VERSION, "schema version")
    _require(payload.get("benchmark_intent") == "held_out_ann_comparator", "benchmark intent")
    config = payload.get("configuration")
    _require(isinstance(config, dict), "configuration")
    rows, dimensions, probes = config.get("rows"), config.get("dimensions"), config.get("max_probes")
    _require(isinstance(rows, int) and rows > 0 and isinstance(dimensions, int) and dimensions > 0, "dimensions")
    _require(isinstance(probes, int) and 0 < probes <= BENCHMARK_MAX_PROBES, "max probes")
    _require(config.get("ef_grid") == list(DECISION_EF_GRID), "ef grid")
    _require(config.get("candidates") == list(CANDIDATES), "candidate identity")
    corpus, queries = payload.get("corpus"), payload.get("queries")
    _require(isinstance(corpus, dict) and isinstance(queries, dict), "corpus/query metadata")
    for name, item, count in (("corpus", corpus, rows), ("queries", queries, probes)):
        _require(item.get("count") == count and item.get("dimensions") == dimensions, f"{name} dimensions/counts")
        _require(isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64, f"{name} digest")
        _require(isinstance(item.get("seed"), str) and item["seed"], f"{name} seed")
    _require(queries.get("zero_overlap_count") == 0, "self-query overlap")
    records = payload.get("records")
    _require(isinstance(records, list) and len(records) == len(CANDIDATES) * len(DECISION_EF_GRID), "records")
    source = payload.get("source")
    _require(isinstance(source, dict) and isinstance(source.get("head_sha"), str) and len(source["head_sha"]) == 40, "source head")
    _require(source.get("lock_identity") == _locked_runtime_identity(), "lock identity")
    environment = payload.get("environment")
    _require(isinstance(environment, dict), "environment")
    _require(environment.get("runtime") == _locked_runtime_identity(), "runtime identity")
    _require(isinstance(environment.get("cpu_count"), int) and environment["cpu_count"] > 0, "cpu count")
    _require(isinstance(environment.get("worker_schedule"), dict), "worker schedule")
    schedule = environment["worker_schedule"]
    _require(schedule == _spawn_worker_schedule(), "worker schedule")

    matrix_timing = payload.get("matrix_timing")
    _require(isinstance(matrix_timing, dict), "matrix timing")
    matrix_start, matrix_end = matrix_timing.get("start_monotonic"), matrix_timing.get("end_monotonic")
    _require(_finite(matrix_start) and _finite(matrix_end) and matrix_end >= matrix_start, "matrix timing")
    wall_seconds = payload.get("benchmark_wall_seconds")
    _require(_finite(wall_seconds) and wall_seconds >= 0, "wall-time cap")
    _require(math.isclose(matrix_end - matrix_start, float(wall_seconds), abs_tol=0.01), "matrix timing")
    acceptance = payload.get("acceptance", {})
    _require(isinstance(acceptance, dict), "acceptance")
    if acceptance.get("mode") == "calibration_non_accepting":
        _require(allow_calibration, "calibration cannot authorize evidence")
    if acceptance.get("mode") == "calibrated_acceptance":
        _require(
            isinstance(acceptance.get("calibration_sha256"), str)
            and len(acceptance["calibration_sha256"]) == 64
            and acceptance.get("calibration_rule_version") == CALIBRATION_RULE_VERSION
            and acceptance.get("selected_omp_threads") in {1, 2},
            "calibration binding",
        )
    if acceptance.get("mode") == "calibration_non_accepting" and allow_calibration:
        cap = None
    else:
        cap = aggregate_cap_seconds if aggregate_cap_seconds is not None else acceptance.get(
            "aggregate_cap_seconds", DEFAULT_MAX_WALL_SECONDS
        )
        _require(_finite(cap) and float(cap) >= 0 and float(wall_seconds) <= float(cap), "wall-time cap")
    accounting = payload.get("matrix_accounting")
    _require(isinstance(accounting, dict), "matrix accounting")
    accounting_start, accounting_end = accounting.get("start_monotonic"), accounting.get("end_monotonic")
    _require(
        _finite(accounting_start) and _finite(accounting_end)
        and matrix_start <= accounting_start <= accounting_end <= matrix_end,
        "matrix accounting",
    )
    candidate_runs = payload.get("candidate_runs")
    _require(isinstance(candidate_runs, list) and len(candidate_runs) == len(CANDIDATES), "candidate runs")
    runs_by_id: dict[str, dict[str, Any]] = {}
    for run in candidate_runs:
        _require(isinstance(run, dict), "candidate run")
        candidate, run_id = run.get("candidate"), run.get("candidate_run_id")
        _require(candidate in CANDIDATES and isinstance(run_id, str) and run_id and run_id not in runs_by_id, "candidate run")
        _require(run.get("normal_ann_request_count") == len(DECISION_EF_GRID) * probes, "normal ANN request count")
        _require(run.get("dense_table_open_count") == 1, "dense table reuse")
        worker = run.get("worker")
        _require(isinstance(worker, dict) and isinstance(worker.get("pid"), int), "worker identity")
        _require(worker.get("start_method") == "spawn" and worker.get("effective_workers") == 2, "worker identity")
        for key in (
            "candidate_start_monotonic", "table_create_start_monotonic", "table_create_end_monotonic",
            "index_build_start_monotonic", "index_build_end_monotonic", "candidate_query_start_monotonic",
            "candidate_query_end_monotonic", "candidate_end_monotonic", "table_create_ms", "index_build_ms",
            "candidate_query_wall_ms", "candidate_wall_ms", "total_bytes", "index_delta_bytes",
        ):
            _require(key in run and _finite(run[key]), f"candidate run {key}")
        _require(
            matrix_start <= run["candidate_start_monotonic"] <= run["table_create_start_monotonic"]
            <= run["table_create_end_monotonic"] <= run["index_build_start_monotonic"]
            <= run["index_build_end_monotonic"] <= run["candidate_query_start_monotonic"]
            <= run["candidate_query_end_monotonic"] <= run["candidate_end_monotonic"] <= matrix_end,
            "candidate interval containment",
        )
        runs_by_id[run_id] = run
    _require({run["candidate"] for run in candidate_runs} == set(CANDIDATES), "candidate run identity")
    expected = {(candidate, ef) for candidate in CANDIDATES for ef in DECISION_EF_GRID}
    seen: set[tuple[str, int]] = set()
    for record in records:
        _require(isinstance(record, dict), "record type")
        candidate, ef = record.get("candidate"), record.get("query_ef")
        _require((candidate, ef) in expected and (candidate, ef) not in seen, "candidate/grid binding")
        seen.add((candidate, ef))
        candidate_run_id = record.get("candidate_run_id")
        _require(candidate_run_id in runs_by_id and runs_by_id[candidate_run_id]["candidate"] == candidate, "candidate run reference")
        _require(record.get("config_sha256") == _config_digest(candidate, rows=rows, dimensions=dimensions), "config digest")
        _require(record.get("unindexed_dense_rows") == 0, "unindexed rows")
        for key in ("build_time_ms", "exact_time_ms", "total_bytes", "index_delta_bytes", "latency_p50_ms", "latency_p95_ms", "recall_at_10", "recall_at_20", "query_group_wall_ms", "result_id_assembly_ms"):
            _require(key in record and _finite(record[key]), f"non-finite {key}")
        group_start, group_end = record.get("query_group_start_monotonic"), record.get("query_group_end_monotonic")
        assembly_start, assembly_end = record.get("result_id_assembly_start_monotonic"), record.get("result_id_assembly_end_monotonic")
        run = runs_by_id[candidate_run_id]
        _require(
            _finite(group_start) and _finite(group_end) and _finite(assembly_start) and _finite(assembly_end)
            and run["candidate_query_start_monotonic"] <= group_start <= assembly_start <= assembly_end <= group_end <= run["candidate_query_end_monotonic"],
            "query group timing",
        )
        samples = record.get("queries")
        _require(isinstance(samples, list) and len(samples) == probes, "query evidence")
        for sample in samples:
            _require(isinstance(sample, dict), "query record")
            for key, count in (("exact_top_10", 10), ("exact_top_20", 20), ("candidate_top_10", 10), ("candidate_top_20", 20)):
                values = sample.get(key)
                _require(isinstance(values, list) and len(values) == count and all(isinstance(v, str) and v for v in values), f"incomplete {key}")
            _require(_finite(sample.get("recall_at_10")) and _finite(sample.get("recall_at_20")), "non-finite recall")
    exact = payload.get("exact")
    _require(isinstance(exact, dict) and _finite(exact.get("time_ms")), "exact evidence")
    _require(float(exact["time_ms"]) / 1000 <= DEFAULT_MAX_EXACT_SECONDS, "exact-time cap")
    _require(seen == expected and _finite(payload), "incomplete evidence")
    _require("selected_candidate" not in payload and "recall_floor" not in payload, "policy decision")
    return payload


def _candidate_worker(
    candidate: str,
    root: str,
    rows: int,
    dimensions: int,
    probes: int,
    ef_grid: tuple[int, ...],
    exact_ids: tuple[tuple[str, ...], ...],
    exact_time_ms: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build and query one isolated candidate in a bounded worker process.

    The worker regenerates the deterministic matrices instead of receiving a
    multi-hundred-megabyte corpus over IPC.  Its only query path is the public
    ``search_dense`` ANN operation, whose cached table handle is shared across
    all six groups.
    """
    candidate_start = time.perf_counter()
    corpus = _vectors(rows, dimensions, CORPUS_SEED)
    queries = _vectors(probes, dimensions, QUERY_SEED)
    chunk_ids = [f"synthetic::{index:016x}" for index in range(rows)]
    lance_dir = Path(root) / candidate
    table_create_start = time.perf_counter()
    db = lancedb.connect(str(lance_dir))
    db.create_table("dense_chunks", data=_arrow_table(corpus, chunk_ids))
    table_create_end = time.perf_counter()
    pre_index_bytes = _directory_bytes(lance_dir)
    repository = LanceDbIndexRepository(lance_dir)
    index_build_start = time.perf_counter()
    stats = repository.create_vector_index(VectorIndexConfig(
        index_type=_REPOSITORY_TYPES[candidate], metric="cosine", num_partitions=1,
        m=16, ef_construction=300, dense_chunks_count=rows,
    ))
    index_build_end = time.perf_counter()
    total_bytes = _directory_bytes(lance_dir)
    candidate_query_start = time.perf_counter()
    records: list[dict[str, Any]] = []
    run_id = f"candidate-run::{candidate}"
    for ef in ef_grid:
        group_start = time.perf_counter()
        samples, latencies = [], []
        for index, vector in enumerate(queries):
            request_started = time.perf_counter()
            result = repository.search_dense(vector.tolist(), metric="cosine", limit=20, ef=ef)
            latencies.append((time.perf_counter() - request_started) * 1000)
            assembly_start = time.perf_counter()
            candidate_20 = [str(row.get("chunk_id", "")) for row in result]
            candidate_10, truth_20 = candidate_20[:10], list(exact_ids[index])
            truth_10 = truth_20[:10]
            samples.append({
                "query_index": index, "exact_top_10": truth_10, "exact_top_20": truth_20,
                "candidate_top_10": candidate_10, "candidate_top_20": candidate_20,
                "recall_at_10": len(set(truth_10) & set(candidate_10)) / 10,
                "recall_at_20": len(set(truth_20) & set(candidate_20)) / 20,
            })
            # The group-level assembly interval includes every request's ID work.
            if index == 0:
                result_assembly_start = assembly_start
        result_assembly_end = time.perf_counter()
        group_end = time.perf_counter()
        records.append({
            "candidate": candidate, "candidate_run_id": run_id, "query_ef": ef,
            "config_sha256": _config_digest(candidate, rows=rows, dimensions=dimensions),
            "build_time_ms": (index_build_end - index_build_start) * 1000,
            "exact_time_ms": exact_time_ms, "total_bytes": total_bytes,
            "index_delta_bytes": total_bytes - pre_index_bytes,
            "unindexed_dense_rows": stats.unindexed_dense_rows,
            "latency_p50_ms": _percentile(latencies, 50), "latency_p95_ms": _percentile(latencies, 95),
            "recall_at_10": sum(s["recall_at_10"] for s in samples) / len(samples),
            "recall_at_20": sum(s["recall_at_20"] for s in samples) / len(samples), "queries": samples,
            "query_group_start_monotonic": group_start, "query_group_end_monotonic": group_end,
            "query_group_wall_ms": (group_end - group_start) * 1000,
            "result_id_assembly_start_monotonic": result_assembly_start,
            "result_id_assembly_end_monotonic": result_assembly_end,
            "result_id_assembly_ms": (result_assembly_end - result_assembly_start) * 1000,
        })
    candidate_query_end = time.perf_counter()
    candidate_end = time.perf_counter()
    return {
        "candidate": candidate, "candidate_run_id": run_id,
        "candidate_start_monotonic": candidate_start,
        "table_create_start_monotonic": table_create_start, "table_create_end_monotonic": table_create_end,
        "index_build_start_monotonic": index_build_start, "index_build_end_monotonic": index_build_end,
        "candidate_query_start_monotonic": candidate_query_start, "candidate_query_end_monotonic": candidate_query_end,
        "candidate_end_monotonic": candidate_end,
        "table_create_ms": (table_create_end - table_create_start) * 1000,
        "index_build_ms": (index_build_end - index_build_start) * 1000,
        "candidate_query_wall_ms": (candidate_query_end - candidate_query_start) * 1000,
        "candidate_wall_ms": (candidate_end - candidate_start) * 1000,
        "total_bytes": total_bytes, "index_delta_bytes": total_bytes - pre_index_bytes,
        "normal_ann_request_count": len(ef_grid) * probes,
        "dense_table_open_count": repository.dense_table_open_count,
        "worker": {
            "pid": os.getpid(), "cpu_count": os.cpu_count() or 1,
            "start_method": "spawn", "effective_workers": 2,
            "candidate_assignment": _candidate_assignment()[candidate],
        },
        "concurrency": _spawn_worker_schedule(),
    }, records


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.rows <= args.max_probes or args.dimensions <= 0 or not 0 < args.max_probes <= BENCHMARK_MAX_PROBES:
        raise ValueError("rows must exceed probes, dimensions must be positive, and probes must be bounded")
    if tuple(args.ef_grid) != DECISION_EF_GRID or tuple(args.candidates) != CANDIDATES:
        raise ValueError("the decision comparator has a fixed candidate grid")
    corpus, queries = _vectors(args.rows, args.dimensions, CORPUS_SEED), _vectors(args.max_probes, args.dimensions, QUERY_SEED)
    overlap = len(_row_hashes(corpus) & _row_hashes(queries))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    matrix_started: float | None = None
    exact_time_ms: float | None = None
    wall_seconds: float | None = None
    candidate_runs: list[dict[str, Any]] = []
    raw_staging_path = Path(args.work_dir) / "index-benchmark.raw.json"
    failure_phase = "initialization"
    try:
        with tempfile.TemporaryDirectory(prefix="held-out-ann-", dir=args.work_dir) as directory:
            root = Path(directory)
            chunk_ids = [f"synthetic::{index:016x}" for index in range(args.rows)]
            truth_repo_dir = root / "truth"
            truth_db = lancedb.connect(str(truth_repo_dir))
            truth_db.create_table("dense_chunks", data=_arrow_table(corpus, chunk_ids))
            truth_repository = LanceDbIndexRepository(truth_repo_dir)
            exact_started = time.perf_counter()
            exact = truth_repository.search_dense_exact_batch(queries.tolist(), metric="cosine", limit=20, row_batch_size=args.row_batch_size, query_batch_size=args.query_batch_size)
            exact_time_ms = (time.perf_counter() - exact_started) * 1000
            matrix_started = time.perf_counter()
            records: list[dict[str, Any]] = []
            failure_phase = "candidate_execution"
            # Spawn is explicit: children never inherit the parent process's LanceDB
            # runtime, connection, table handle, or Arrow state.
            with ProcessPoolExecutor(
                max_workers=2, mp_context=multiprocessing.get_context("spawn")
            ) as executor:
                futures = [
                    executor.submit(
                        _candidate_worker, candidate, str(root), args.rows, args.dimensions,
                        args.max_probes, tuple(args.ef_grid), exact.result_ids, exact_time_ms,
                    )
                    for candidate in args.candidates
                ]
                for future in futures:
                    candidate_run, candidate_records = future.result()
                    candidate_runs.append(candidate_run)
                    records.extend(candidate_records)
            accounting_started = time.perf_counter()
            candidate_runs.sort(key=lambda record: record["candidate"])
            records.sort(key=lambda record: (record["candidate"], record["query_ef"]))
            _require(sum(run["normal_ann_request_count"] for run in candidate_runs) == len(args.candidates) * len(args.ef_grid) * args.max_probes, "matrix accounting")
            accounting_ended = time.perf_counter()
            matrix_ended = time.perf_counter()
            wall_seconds = matrix_ended - matrix_started
        calibration_mode = bool(getattr(args, "calibration_mode", False))
        has_calibration_reference = bool(getattr(args, "calibration_reference", None))
        acceptance = {
            "mode": "calibration_non_accepting" if calibration_mode else (
                "calibrated_acceptance" if has_calibration_reference else "acceptance"
            ),
            "aggregate_cap_seconds": None if calibration_mode else args.max_seconds,
        }
        if not calibration_mode and has_calibration_reference:
            acceptance.update(getattr(args, "calibration_reference"))
        payload: dict[str, Any] = {
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION, "benchmark_intent": "held_out_ann_comparator",
            "configuration": {"rows": args.rows, "dimensions": args.dimensions, "max_probes": args.max_probes, "ef_grid": list(args.ef_grid), "candidates": list(args.candidates), "row_batch_size": args.row_batch_size, "query_batch_size": args.query_batch_size},
            "corpus": {"count": args.rows, "dimensions": args.dimensions, "seed": "corpus-v1", "sha256": _matrix_digest(corpus)},
            "queries": {"count": args.max_probes, "dimensions": args.dimensions, "seed": "queries-v1", "sha256": _matrix_digest(queries), "zero_overlap_count": overlap},
            "source": {"head_sha": _head_sha(), "lock_identity": _locked_runtime_identity()},
            "exact": {"method": exact.method, "scan_rows": exact.scan_rows, "scan_batches": exact.scan_batches, "time_ms": round(exact_time_ms, 3)},
            "environment": {
                "python": platform.python_version(), "os": platform.platform(), "runtime": _runtime_identity(),
                "cpu_count": os.cpu_count() or 1, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                "runner_name": os.environ.get("RUNNER_NAME"), "runner_os": os.environ.get("RUNNER_OS"),
                "worker_schedule": _spawn_worker_schedule(),
            },
            "candidate_runs": candidate_runs, "records": records,
            "matrix_timing": {"start_monotonic": matrix_started, "end_monotonic": matrix_ended},
            "matrix_accounting": {"start_monotonic": accounting_started, "end_monotonic": accounting_ended},
            "benchmark_wall_seconds": wall_seconds,
            "acceptance": acceptance,
        }
        payload["benchmark_payload_bytes"] = len(json.dumps(payload, sort_keys=True).encode())
        raw_staging_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        failure_phase = "post_run_validation"
        failures: list[str] = []
        try:
            validate_evidence(
                payload, aggregate_cap_seconds=args.max_seconds,
                allow_calibration=calibration_mode,
            )
        except ValueError as exc:
            failures.append(str(exc))
        if exact_time_ms / 1000 > args.max_exact_seconds:
            failures.append("exact-time cap")
        if wall_seconds > args.max_seconds:
            failures.append("wall-time cap")
        if payload["benchmark_payload_bytes"] > args.max_evidence_bytes:
            failures.append("evidence-size cap")
        if failures:
            raise RuntimeError("; ".join(dict.fromkeys(failures)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        raw_staging_path.replace(args.output)
        return payload
    except Exception as exc:
        _write_rejected_error(
            args, exc, failure_phase=failure_phase, matrix_started=matrix_started,
            exact_time_ms=exact_time_ms, wall_seconds=wall_seconds,
            raw_staging_path=raw_staging_path if raw_staging_path.exists() else None,
            candidate_runs=candidate_runs,
        )
        raise


def _with_omp(args: argparse.Namespace, omp_threads: int, operation: Any) -> Any:
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    original = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = str(omp_threads)
        return operation(args)
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_with_omp(args: argparse.Namespace, omp_threads: int) -> dict[str, Any]:
    return _with_omp(args, omp_threads, run)


def _run_actual_batch_spike(args: argparse.Namespace, omp_threads: int) -> dict[str, Any]:
    """Persist one real batch observation without changing per-query evidence.

    This intentionally creates its own diagnostic-only table.  It is neither a
    candidate-matrix record nor an acceptance shortcut: the individual requests
    are retained solely as the comparison reference for the batch output.
    """
    corpus = _vectors(args.rows, args.dimensions, CORPUS_SEED)
    queries = _vectors(args.max_probes, args.dimensions, QUERY_SEED)
    chunk_ids = [f"synthetic::{index:016x}" for index in range(args.rows)]
    with tempfile.TemporaryDirectory(prefix=f"ann-batch-omp-{omp_threads}-", dir=args.work_dir) as directory:
        lance_dir = Path(directory) / "diagnostic"
        lancedb.connect(str(lance_dir)).create_table("dense_chunks", data=_arrow_table(corpus, chunk_ids))
        repository = LanceDbIndexRepository(lance_dir)
        candidate = CANDIDATES[0]
        repository.create_vector_index(VectorIndexConfig(
            index_type=_REPOSITORY_TYPES[candidate], metric="cosine", num_partitions=1,
            m=16, ef_construction=300, dense_chunks_count=args.rows,
        ))
        ef, limit = DECISION_EF_GRID[0], 20
        individual_result_ids = [
            [str(row.get("chunk_id", "")) for row in repository.search_dense(vector.tolist(), metric="cosine", limit=limit, ef=ef)]
            for vector in queries
        ]
        spike = run_multivector_batch_spike(
            repository._dense_table(), queries.tolist(), metric="cosine", ef=ef, limit=limit,
            individual_result_ids=individual_result_ids,
        )
    spike.update({
        "omp_threads": omp_threads,
        "candidate": candidate,
        "individual_query_count": args.max_probes,
        "query_count": args.max_probes,
        "normal_per_query_contract_preserved": True,
    })
    return spike


def run_manual_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Run only non-authorizing diagnostics; never run an acceptance matrix."""
    if args.calibration_output is None or args.calibration_batch_output is None:
        raise ValueError("calibration output and batch output")
    trials: dict[int, list[float]] = {1: [], 2: []}
    trial_records: dict[str, list[dict[str, Any]]] = {"1": [], "2": []}
    batch_spikes: list[dict[str, Any]] = []
    try:
        for omp_threads in (1, 2):
            for repetition in range(5):
                trial = argparse.Namespace(**vars(args))
                trial.calibration_mode = True
                trial.max_seconds = float("inf")
                trial.output = Path(args.work_dir) / "calibration" / f"omp-{omp_threads}-{repetition}.json"
                trial.error_output = Path(args.work_dir) / "index-benchmark-error.json"
                payload = _run_with_omp(trial, omp_threads)
                trials[omp_threads].append(float(payload["benchmark_wall_seconds"]))
                trial_records[str(omp_threads)].append({
                    "repetition": repetition,
                    "wall_seconds": payload["benchmark_wall_seconds"],
                    "exact_time_ms": payload["exact"]["time_ms"],
                    "path": str(trial.output),
                    "sha256": hashlib.sha256(trial.output.read_bytes()).hexdigest(),
                })
            batch_spikes.append(
                _with_omp(args, omp_threads, lambda _: _run_actual_batch_spike(args, omp_threads))
            )
        calibration = build_calibration_record(
            head_sha=_head_sha(), lock_identity=_locked_runtime_identity(),
            configuration={
                "cpu_count": os.cpu_count() or 1,
                "worker_schedule": _spawn_worker_schedule(),
                "candidates": list(CANDIDATES), "ef_grid": list(DECISION_EF_GRID),
                "probes": args.max_probes,
            },
            repetitions=trials,
        )
        calibration["trial_records"] = trial_records
        calibration["batch_spike_path"] = str(args.calibration_batch_output)
        finalize_calibration_record(calibration)
        args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_output.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
        batch_payload = {
            "status": "observational_only",
            "head_sha": _head_sha(),
            "lock_identity": _locked_runtime_identity(),
            "spikes": batch_spikes,
            "can_authorize_acceptance": False,
        }
        batch_payload["sha256"] = hashlib.sha256(json.dumps(batch_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        args.calibration_batch_output.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_batch_output.write_text(json.dumps(batch_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return calibration
    except Exception as exc:
        _write_rejected_error(args, exc, failure_phase="manual_calibration", matrix_started=None)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=77_348)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--max-probes", type=int, default=BENCHMARK_MAX_PROBES)
    parser.add_argument("--ef-grid", default=",".join(map(str, DECISION_EF_GRID)))
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_WALL_SECONDS)
    parser.add_argument("--max-exact-seconds", type=float, default=DEFAULT_MAX_EXACT_SECONDS)
    parser.add_argument("--row-batch-size", type=int, default=8192)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--max-evidence-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--work-dir", type=Path, default=SKILL_ROOT / ".review-tmp" / "held-out-ann")
    parser.add_argument("--output", type=Path, default=HERE / "index-benchmark.json")
    parser.add_argument("--error-output", type=Path, help="Rejected worker/pool diagnostics path")
    parser.add_argument("--calibrate", action="store_true", help="Run five OMP 1/2 diagnostics only; never acceptance")
    parser.add_argument("--calibration-output", type=Path, help="Non-accepting calibration artifact")
    parser.add_argument("--calibration-batch-output", type=Path, help="Observational-only multi-vector batch artifact")
    parser.add_argument("--approved-static-cap", type=float, help="Root/user-approved committed PR acceptance cap")
    parser.add_argument("--approved-calibration-sha256", help="Root/user-approved committed calibration digest")
    parser.add_argument("--approved-calibration-rule-version", help="Rule version bound to the approved digest")
    parser.add_argument("--approved-omp-threads", type=int, choices=(1, 2), help="OMP setting bound to the approved digest")
    parser.add_argument(
        "--validate-evidence", type=Path,
        help="Validate an existing comparator artifact without running a benchmark.",
    )
    args = parser.parse_args()
    if args.validate_evidence is not None:
        try:
            validate_evidence(json.loads(args.validate_evidence.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[FAIL] held-out ANN evidence: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print("[PASS] held-out ANN evidence", file=sys.stderr)
        return 0
    args.ef_grid = tuple(int(value) for value in args.ef_grid.split(",") if value)
    args.candidates = tuple(value for value in args.candidates.split(",") if value)
    approval_values = (
        args.approved_static_cap, args.approved_calibration_sha256,
        args.approved_calibration_rule_version, args.approved_omp_threads,
    )
    if any(value is not None for value in approval_values):
        if args.calibrate or any(value is None for value in approval_values):
            parser.error("approved static acceptance inputs must be complete and cannot calibrate")
        if (
            args.approved_static_cap <= 0
            or args.approved_static_cap > DEFAULT_MAX_WALL_SECONDS
            or len(args.approved_calibration_sha256) != 64
        ):
            parser.error("approved static acceptance inputs")
        args.max_seconds = args.approved_static_cap
        args.calibration_reference = {
            "calibration_sha256": args.approved_calibration_sha256,
            "calibration_rule_version": args.approved_calibration_rule_version,
            "selected_omp_threads": args.approved_omp_threads,
        }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        runner = run_manual_calibration if args.calibrate else run
        print(json.dumps(runner(args), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[FAIL] held-out ANN comparator: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("[PASS] held-out ANN comparator", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
