"""Fail-closed held-out FLAT/SQ ANN comparison used only by evaluation jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
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


EVIDENCE_SCHEMA_VERSION = 3
DECISION_EF_GRID = (30, 50, 75, 100, 150, 200)
CANDIDATES = ("ivf-hnsw-flat", "ivf-hnsw-sq")
_REPOSITORY_TYPES = {"ivf-hnsw-flat": "hnsw_flat", "ivf-hnsw-sq": "hnsw_sq"}
DEFAULT_MAX_EXACT_SECONDS = 10.0
DEFAULT_MAX_WALL_SECONDS = 60.0
CORPUS_SEED = 41001
QUERY_SEED = 41002


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_evidence(payload: dict[str, Any]) -> dict[str, Any]:
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
    expected = {(candidate, ef) for candidate in CANDIDATES for ef in DECISION_EF_GRID}
    seen: set[tuple[str, int]] = set()
    for record in records:
        _require(isinstance(record, dict), "record type")
        candidate, ef = record.get("candidate"), record.get("query_ef")
        _require((candidate, ef) in expected and (candidate, ef) not in seen, "candidate/grid binding")
        seen.add((candidate, ef))
        _require(record.get("config_sha256") == _config_digest(candidate, rows=rows, dimensions=dimensions), "config digest")
        _require(record.get("unindexed_dense_rows") == 0, "unindexed rows")
        for key in ("build_time_ms", "exact_time_ms", "total_bytes", "index_delta_bytes", "latency_p50_ms", "latency_p95_ms", "recall_at_10", "recall_at_20"):
            _require(key in record and _finite(record[key]), f"non-finite {key}")
        samples = record.get("queries")
        _require(isinstance(samples, list) and len(samples) == probes, "query evidence")
        for sample in samples:
            _require(isinstance(sample, dict), "query record")
            for key, count in (("exact_top_10", 10), ("exact_top_20", 20), ("candidate_top_10", 10), ("candidate_top_20", 20)):
                values = sample.get(key)
                _require(isinstance(values, list) and len(values) == count and all(isinstance(v, str) and v for v in values), f"incomplete {key}")
            _require(_finite(sample.get("recall_at_10")) and _finite(sample.get("recall_at_20")), "non-finite recall")
    _require(seen == expected and _finite(payload), "incomplete evidence")
    _require("selected_candidate" not in payload and "recall_floor" not in payload, "policy decision")
    return payload


def _candidate_record(
    *, candidate: str, root: Path, corpus: np.ndarray, queries: np.ndarray, chunk_ids: list[str], args: argparse.Namespace, exact_ids: tuple[tuple[str, ...], ...], exact_time_ms: float,
) -> list[dict[str, Any]]:
    lance_dir = root / candidate
    db = lancedb.connect(str(lance_dir))
    db.create_table("dense_chunks", data=_arrow_table(corpus, chunk_ids))
    pre_index_bytes = _directory_bytes(lance_dir)
    repository = LanceDbIndexRepository(lance_dir)
    started = time.perf_counter()
    stats = repository.create_vector_index(VectorIndexConfig(index_type=_REPOSITORY_TYPES[candidate], metric="cosine", num_partitions=1, m=16, ef_construction=300, dense_chunks_count=args.rows))
    build_time_ms = (time.perf_counter() - started) * 1000
    total_bytes = _directory_bytes(lance_dir)
    records: list[dict[str, Any]] = []
    for ef in args.ef_grid:
        samples, latencies = [], []
        for index, vector in enumerate(queries):
            started = time.perf_counter()
            result = repository.search_dense(vector.tolist(), metric="cosine", limit=20, ef=ef)
            latencies.append((time.perf_counter() - started) * 1000)
            candidate_20 = [str(row.get("chunk_id", "")) for row in result]
            candidate_10, truth_20 = candidate_20[:10], list(exact_ids[index])
            truth_10 = truth_20[:10]
            samples.append({
                "query_index": index, "exact_top_10": truth_10, "exact_top_20": truth_20,
                "candidate_top_10": candidate_10, "candidate_top_20": candidate_20,
                "recall_at_10": len(set(truth_10) & set(candidate_10)) / 10,
                "recall_at_20": len(set(truth_20) & set(candidate_20)) / 20,
            })
        records.append({
            "candidate": candidate, "query_ef": ef, "config_sha256": _config_digest(candidate, rows=args.rows, dimensions=args.dimensions),
            "build_time_ms": round(build_time_ms, 3), "exact_time_ms": round(exact_time_ms, 3),
            "total_bytes": total_bytes, "index_delta_bytes": total_bytes - pre_index_bytes,
            "unindexed_dense_rows": stats.unindexed_dense_rows,
            "latency_p50_ms": round(_percentile(latencies, 50), 3), "latency_p95_ms": round(_percentile(latencies, 95), 3),
            "recall_at_10": sum(s["recall_at_10"] for s in samples) / len(samples),
            "recall_at_20": sum(s["recall_at_20"] for s in samples) / len(samples), "queries": samples,
        })
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.rows <= args.max_probes or args.dimensions <= 0 or not 0 < args.max_probes <= BENCHMARK_MAX_PROBES:
        raise ValueError("rows must exceed probes, dimensions must be positive, and probes must be bounded")
    if tuple(args.ef_grid) != DECISION_EF_GRID or tuple(args.candidates) != CANDIDATES:
        raise ValueError("the decision comparator has a fixed candidate grid")
    corpus, queries = _vectors(args.rows, args.dimensions, CORPUS_SEED), _vectors(args.max_probes, args.dimensions, QUERY_SEED)
    overlap = len(_row_hashes(corpus) & _row_hashes(queries))
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
        wall_started = time.perf_counter()
        records = []
        for candidate in args.candidates:
            records.extend(_candidate_record(candidate=candidate, root=root, corpus=corpus, queries=queries, chunk_ids=chunk_ids, args=args, exact_ids=exact.result_ids, exact_time_ms=exact_time_ms))
        wall_seconds = time.perf_counter() - wall_started
    payload: dict[str, Any] = {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION, "benchmark_intent": "held_out_ann_comparator",
        "configuration": {"rows": args.rows, "dimensions": args.dimensions, "max_probes": args.max_probes, "ef_grid": list(args.ef_grid), "candidates": list(args.candidates), "row_batch_size": args.row_batch_size, "query_batch_size": args.query_batch_size},
        "corpus": {"count": args.rows, "dimensions": args.dimensions, "seed": "corpus-v1", "sha256": _matrix_digest(corpus)},
        "queries": {"count": args.max_probes, "dimensions": args.dimensions, "seed": "queries-v1", "sha256": _matrix_digest(queries), "zero_overlap_count": overlap},
        "exact": {"method": exact.method, "scan_rows": exact.scan_rows, "scan_batches": exact.scan_batches, "time_ms": round(exact_time_ms, 3)},
        "environment": {"python": platform.python_version(), "os": platform.platform(), "numpy": np.__version__, "lancedb": lancedb.__version__, "pyarrow": pa.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS")},
        "records": records, "benchmark_wall_seconds": round(wall_seconds, 6),
    }
    payload["benchmark_payload_bytes"] = len(json.dumps(payload, sort_keys=True).encode())
    failures: list[str] = []
    try:
        validate_evidence(payload)
    except ValueError as exc:
        failures.append(str(exc))
    if exact_time_ms / 1000 > args.max_exact_seconds:
        failures.append("exact-time cap")
    if wall_seconds > args.max_seconds:
        failures.append("wall-time cap")
    if payload["benchmark_payload_bytes"] > args.max_evidence_bytes:
        failures.append("evidence-size cap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError("; ".join(failures))
    return payload


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
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[FAIL] held-out ANN comparator: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("[PASS] held-out ANN comparator", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
