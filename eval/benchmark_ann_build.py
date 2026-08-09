"""Reproducible issue #41 build-time ANN benchmark (performance-only gate).

The script generates vectors at runtime, creates a real LanceDB HNSW index, and
executes the production IndexBuildService benchmark path.  No large fixture is
stored in git.  A non-zero exit means the bounded-probe performance/evidence
contract is not satisfied.

This is a PERFORMANCE-ONLY gate: it asserts the benchmark scans the dense corpus
exactly once (streamed batch exact) and stays within wall-clock/evidence budgets.
ANN publication quality (recall, promote-to-ann) is validated separately by
eval/run_eval.py against a real-model fixture, not by this random-vector scale.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import lancedb
import numpy as np
import pyarrow as pa


HERE = Path(__file__).resolve().parent
SKILL_ROOT = HERE.parent
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

DEFAULT_MAX_EXACT_SECONDS = 10.0
# Coarse end-to-end guard only. The stage-specific exact SLO above is the
# deterministic #41 algorithmic regression gate; scalar ANN latency varies on
# shared runners and remains fully measured in the evidence payload.
DEFAULT_MAX_WALL_SECONDS = 60.0

import obsidian_wiki.application.index_build_service as build_service_module  # noqa: E402
from obsidian_wiki.application.index_build_service import IndexBuildService  # noqa: E402
from obsidian_wiki.domain.index_models import VectorIndexConfig  # noqa: E402
from obsidian_wiki.domain.index_policy import select_vector_policy  # noqa: E402
from obsidian_wiki.infrastructure.lancedb_index_repository import (  # noqa: E402
    LanceDbIndexRepository,
)


class _VectorView(Sequence[float]):
    """List-compatible view without duplicating the full vector matrix in Python."""

    __slots__ = ("_matrix", "_index")

    def __init__(self, matrix: np.ndarray, index: int) -> None:
        self._matrix = matrix
        self._index = index

    def __len__(self) -> int:
        return int(self._matrix.shape[1])

    def __bool__(self) -> bool:
        return True

    def __getitem__(self, index):
        return self._matrix[self._index][index]

    def __iter__(self) -> Iterator[float]:
        return (float(value) for value in self._matrix[self._index])


@dataclass(slots=True)
class _DenseProbe:
    chunk_id: str
    page_id: str
    path: str
    chunk_kind: str
    chunk_index: int
    continuation_index: int
    content_hash: str
    vector: Sequence[float]


class _NoopDependency:
    pass


def _require_issue41_api() -> int:
    if not hasattr(build_service_module, "BENCHMARK_MAX_PROBES"):
        raise RuntimeError("Issue #41 contract missing: BENCHMARK_MAX_PROBES")
    signature = inspect.signature(IndexBuildService)
    if "benchmark_max_probes" not in signature.parameters:
        raise RuntimeError("Issue #41 contract missing: benchmark_max_probes constructor input")
    benchmark_signature = inspect.signature(IndexBuildService._benchmark)
    if "wiki_dir" not in benchmark_signature.parameters:
        raise RuntimeError("Issue #41 contract missing: _benchmark(..., wiki_dir=...) portable sampling root")
    return int(build_service_module.BENCHMARK_MAX_PROBES)


def _vectors(rows: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((rows, dimensions), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, np.finfo(np.float32).eps)
    return vectors


def _arrow_table(vectors: np.ndarray, chunk_ids: list[str]) -> pa.Table:
    dimensions = int(vectors.shape[1])
    vector_column = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()), dimensions
    )
    return pa.table({"chunk_id": pa.array(chunk_ids), "vector": vector_column})


def run(args: argparse.Namespace) -> dict:
    default_cap = _require_issue41_api()
    if args.max_probes != default_cap:
        raise ValueError(
            f"scale gate must exercise production cap {default_cap}, got {args.max_probes}"
        )
    if args.rows <= args.max_probes or args.dimensions <= 0:
        raise ValueError("rows must exceed max_probes and dimensions must be positive")

    vectors = _vectors(args.rows, args.dimensions, args.seed)
    with tempfile.TemporaryDirectory(prefix="issue41-scale-", dir=args.work_dir) as temp_dir:
        root = Path(temp_dir)
        wiki_dir = root / "Wiki"
        wiki_dir.mkdir()
        lance_dir = root / "lance_db"
        chunk_ids = [
            f"{(wiki_dir / 'synthetic' / f'page-{index // 4:06d}.md').resolve()}::"
            f"{index:016x}"
            for index in range(args.rows)
        ]
        db = lancedb.connect(str(lance_dir))
        db.create_table("dense_chunks", data=_arrow_table(vectors, chunk_ids))
        repository = LanceDbIndexRepository(lance_dir)

        index_started = time.perf_counter()
        stats = repository.create_vector_index(
            VectorIndexConfig(
                index_type="hnsw_flat",
                metric="cosine",
                num_partitions=1,
                m=16,
                ef_construction=300,
                dense_chunks_count=args.rows,
            )
        )
        index_build_ms = (time.perf_counter() - index_started) * 1000
        probes = tuple(
            _DenseProbe(
                chunk_id=chunk_ids[index],
                page_id=chunk_ids[index].rsplit("::", 1)[0],
                path=chunk_ids[index].rsplit("::", 1)[0],
                chunk_kind="dense",
                chunk_index=index,
                continuation_index=-1,
                content_hash=f"{index:064x}",
                vector=_VectorView(vectors, index),
            )
            for index in range(args.rows)
        )
        service = IndexBuildService(
            _NoopDependency(),
            reopen_storage=lambda _path: _NoopDependency(),
            manifest_store=_NoopDependency(),
            post_commit_journal=_NoopDependency(),
            benchmark_max_probes=args.max_probes,
        )

        benchmark_started = time.perf_counter()
        observation, evidence = service._benchmark(
            repository,
            probes,
            stats,
            build_time_ms=index_build_ms,
            disk_bytes=sum(path.stat().st_size for path in lance_dir.rglob("*") if path.is_file()),
            wiki_dir=wiki_dir,
            row_batch_size=args.row_batch_size,
            query_batch_size=args.query_batch_size,
        )
        measured_seconds = time.perf_counter() - benchmark_started
        decision = select_vector_policy(observation, stats, evidence=evidence)
        payload = {
            "benchmark_intent": "performance_only",
            "quality_gate": "eval/run_eval.py",
            "configuration": {
                "rows": args.rows,
                "dimensions": args.dimensions,
                "max_probes": args.max_probes,
                "seed": args.seed,
                "max_seconds": args.max_seconds,
                "max_exact_seconds": args.max_exact_seconds,
                "row_batch_size": args.row_batch_size,
                "query_batch_size": args.query_batch_size,
                "max_evidence_bytes": args.max_evidence_bytes,
            },
            "index_build_ms": round(index_build_ms, 3),
            "benchmark_wall_seconds": round(measured_seconds, 6),
            "benchmark": {**observation.to_json(), **evidence},
            "policy": decision.to_json(),
            "diagnostics": {
                "cpu_count": os.cpu_count(),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "lancedb_version": lancedb.__version__,
                "pyarrow_version": pa.__version__,
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            },
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        payload["benchmark_payload_bytes"] = len(encoded)

    benchmark = payload["benchmark"]
    failures = []
    if benchmark.get("probe_scope") != "sampled":
        failures.append(f"probe_scope={benchmark.get('probe_scope')!r}, expected sampled")
    if benchmark.get("probe_count") != args.max_probes:
        failures.append(
            f"probe_count={benchmark.get('probe_count')!r}, expected {args.max_probes}"
        )
    if benchmark.get("probe_total") != args.rows:
        failures.append(f"probe_total={benchmark.get('probe_total')!r}, expected {args.rows}")
    if len(benchmark.get("exact_result_ids", [])) != args.max_probes:
        failures.append("exact_result_ids is not bounded by max_probes")
    if len(benchmark.get("candidate_result_ids", [])) != args.max_probes:
        failures.append("candidate_result_ids is not bounded by max_probes")

    # #41 batch-exact structure gates. Performance-only: ANN promotion quality is
    # validated by eval/run_eval.py, not by this random-vector scale fixture.
    if benchmark.get("evidence_schema_version") != 2:
        failures.append("expected benchmark evidence schema v2")
    if benchmark.get("exact_method") != "streamed_numpy_cosine_v1":
        failures.append(f"unexpected exact_method={benchmark.get('exact_method')!r}")
    if benchmark.get("exact_scan_rows") != args.rows:
        failures.append(
            f"exact_scan_rows={benchmark.get('exact_scan_rows')!r}, expected {args.rows}"
        )
    if not isinstance(benchmark.get("exact_scan_batches"), int) \
            or benchmark["exact_scan_batches"] <= 0:
        failures.append("exact_scan_batches must be positive")
    if benchmark.get("ann_query_count") != args.max_probes:
        failures.append(
            f"ann_query_count={benchmark.get('ann_query_count')!r}, expected {args.max_probes}"
        )
    exact_seconds = float(benchmark.get("exact_verification_ms", float("inf"))) / 1000
    if exact_seconds > args.max_exact_seconds:
        failures.append(
            f"exact_verification_seconds={exact_seconds:.3f} > "
            f"SLO {args.max_exact_seconds:.3f}"
        )

    if measured_seconds > args.max_seconds:
        failures.append(
            f"benchmark_wall_seconds={measured_seconds:.3f} > SLO {args.max_seconds:.3f}"
        )
    if payload["benchmark_payload_bytes"] > args.max_evidence_bytes:
        failures.append(
            f"benchmark_payload_bytes={payload['benchmark_payload_bytes']} > "
            f"budget {args.max_evidence_bytes}"
        )
    if not math.isfinite(float(benchmark.get("benchmark_duration_ms", float("nan")))):
        failures.append("benchmark_duration_ms is absent or non-finite")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError("; ".join(failures))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=77_348)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--max-probes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_WALL_SECONDS)
    parser.add_argument(
        "--max-exact-seconds", type=float, default=DEFAULT_MAX_EXACT_SECONDS
    )
    parser.add_argument("--row-batch-size", type=int, default=8192)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--max-evidence-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--work-dir", type=Path, default=SKILL_ROOT / ".review-tmp" / "issue41-scale")
    parser.add_argument("--output", type=Path, default=HERE / "index-benchmark.json")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = run(args)
    except Exception as exc:
        print(f"[FAIL] issue #41 ANN build benchmark: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("[PASS] issue #41 bounded ANN benchmark", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
