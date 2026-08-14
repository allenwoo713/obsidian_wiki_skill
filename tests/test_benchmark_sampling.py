"""Issue #41 acceptance gates for bounded, auditable ANN build probes.

These tests intentionally describe the required production contract before the
implementation lands.  They must fail on the unbounded master implementation
and pass without weakening the existing recall thresholds.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from eval import benchmark_ann_build
from obsidian_wiki.application.index_build_service import IndexBuildService
from obsidian_wiki.domain.index_models import (
    BenchmarkObservation,
    DenseChunk,
    ExactBatchResult,
    IndexStats,
    SparseChunk,
)
from obsidian_wiki.domain.index_policy import select_vector_policy
from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest
from obsidian_wiki.infrastructure.filesystem_post_commit_journal import FilesystemPostCommitJournal
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository


REQUIRED_EVIDENCE_FIELDS = {
    "evidence_schema_version",
    "evidence_source",
    "probe_scope",
    "sampling_method",
    "sampling_key_schema",
    "probe_keys",
    "probe_selection_sha256",
    "probe_count",
    "probe_total",
    "probe_coverage",
    "result_limit",
    "recall_aggregation",
    "benchmark_duration_ms",
    "probe_selection_ms",
    "exact_verification_ms",
    "ann_verification_ms",
    "recall_assembly_ms",
    "exact_method",
    "exact_scan_rows",
    "exact_scan_batches",
    "ann_query_count",
    "exact_result_ids",
    "candidate_result_ids",
}


def test_decision_comparator_contract_is_held_out_and_fail_closed() -> None:
    """The scale gate must not accept self-query or partial A/B evidence."""
    assert benchmark_ann_build.DECISION_EF_GRID == (30, 50, 75, 100, 150, 200)
    payload = {
        "evidence_schema_version": benchmark_ann_build.EVIDENCE_SCHEMA_VERSION,
        "benchmark_intent": "held_out_ann_comparator",
        "configuration": {
            "rows": 513,
            "dimensions": 32,
            "max_probes": 256,
            "ef_grid": list(benchmark_ann_build.DECISION_EF_GRID),
            "candidates": ["ivf-hnsw-flat", "ivf-hnsw-sq"],
        },
        "corpus": {"count": 513, "dimensions": 32, "sha256": "a" * 64, "seed": "corpus-v1"},
        "queries": {"count": 256, "dimensions": 32, "sha256": "b" * 64, "seed": "queries-v1", "zero_overlap_count": 0},
        "records": [],
    }
    with pytest.raises(ValueError, match="records"):
        benchmark_ann_build.validate_evidence(payload)


def test_issue41_scale_gate_separates_exact_slo_from_coarse_wall_ceiling() -> None:
    """Runner-sensitive ANN latency must not crowd the exact-path regression gate."""
    assert benchmark_ann_build.DEFAULT_MAX_EXACT_SECONDS == 10.0
    assert benchmark_ann_build.DEFAULT_MAX_WALL_SECONDS == 60.0

    workflow = (Path(__file__).parents[1] / ".github/workflows/eval.yml").read_text(
        encoding="utf-8"
    )
    scale_job = workflow.split("issue41-scale-benchmark:", 1)[1]
    assert "--max-exact-seconds 10" in scale_job
    assert "--max-seconds 60" in scale_job


def _stats(total: int) -> IndexStats:
    return IndexStats(index_name="dense_hnsw", indexed_rows=total, unindexed_dense_rows=0)


def _observation(*, recall_at_20: float = 1.0) -> BenchmarkObservation:
    return BenchmarkObservation(
        recall_at_10=1.0,
        recall_at_20=recall_at_20,
        latency_p50_ms=0.1,
        latency_p95_ms=0.2,
        build_time_ms=1.0,
        disk_bytes=1,
    )


def _chunks(root: Path, total: int) -> tuple[DenseChunk, ...]:
    wiki = root / "Wiki"
    rows = []
    for index in range(total):
        relative = Path("concepts") / f"page-{index // 4:05d}.md"
        path = (wiki / relative).resolve()
        suffix = hashlib.sha256(f"dense-{index}".encode()).hexdigest()[:16]
        rows.append(
            DenseChunk(
                chunk_id=f"{path}::{suffix}",
                page_id=str(path),
                path=str(path),
                title=f"Page {index // 4}",
                text=f"dense body {index}",
                vector=(float(index + 1), 1.0, 0.5, 0.25),
                chunk_kind="dense",
                chunk_index=index,
                content_hash=hashlib.sha256(f"body-{index}".encode()).hexdigest(),
                continuation_index=-1,
            )
        )
    return tuple(rows)


class _NoopDependency:
    pass


class _CountingRepository:
    def __init__(self, chunk_ids: Sequence[str], *, miss_at_20: bool = False) -> None:
        self._chunk_ids = list(chunk_ids)
        self._truth = [{"chunk_id": chunk_id} for chunk_id in self._chunk_ids[:20]]
        self._miss_at_20 = miss_at_20
        self.exact_batch_calls = 0
        self.exact_scalar_calls = 0
        self.ann_calls = 0

    def search_dense_exact(self, *args, **kwargs):
        self.exact_scalar_calls += 1
        raise AssertionError("#41 benchmark must not use scalar exact queries")

    def search_dense_exact_batch(
        self,
        vectors,
        *,
        metric,
        limit=20,
        row_batch_size=8192,
        query_batch_size=32,
    ) -> ExactBatchResult:
        self.exact_batch_calls += 1
        ids = tuple(str(row["chunk_id"]) for row in self._truth[:limit])
        return ExactBatchResult(
            result_ids=tuple(ids for _ in vectors),
            elapsed_ms=0.1,
            scan_rows=len(self._chunk_ids),
            scan_batches=1,
            method="streamed_numpy_cosine_v1",
        )

    def search_dense(
        self, vector: Sequence[float], *, metric: str, limit: int = 10,
        where: str | None = None, ef: int | None = None,
    ) -> list[Mapping[str, object]]:
        self.ann_calls += 1
        rows = list(self._truth[:limit])
        if self._miss_at_20 and self.ann_calls == 1 and len(rows) >= 20:
            rows[-1] = {"chunk_id": "not-in-exact-result"}
        return rows


def _service(max_probes: int, *, observer=None) -> IndexBuildService:
    signature = inspect.signature(IndexBuildService)
    assert "benchmark_max_probes" in signature.parameters, (
        "Issue #41 requires IndexBuildService(..., benchmark_max_probes=...)"
    )
    return IndexBuildService(
        _NoopDependency(),
        reopen_storage=lambda _path: _NoopDependency(),
        manifest_store=_NoopDependency(),
        post_commit_journal=_NoopDependency(),
        benchmark_observer=observer,
        benchmark_max_probes=max_probes,
    )


def _run_benchmark(
    root: Path,
    chunks: Sequence[DenseChunk],
    *,
    max_probes: int,
    repository=None,
    observer=None,
):
    service = _service(max_probes, observer=observer)
    repository = repository or _CountingRepository([chunk.chunk_id for chunk in chunks])
    observation, evidence = service._benchmark(
        repository,
        chunks,
        _stats(len(chunks)),
        build_time_ms=1.0,
        disk_bytes=1,
        wiki_dir=root / "Wiki",
    )
    return observation, evidence, repository


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_benchmark_probe_cap_rejects_invalid_constructor_values(value) -> None:
    with pytest.raises((TypeError, ValueError), match="benchmark_max_probes"):
        _service(value)  # type: ignore[arg-type]


def test_small_corpus_uses_full_probe_scope(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path, 8)
    observation, evidence, repository = _run_benchmark(tmp_path, chunks, max_probes=8)

    assert observation.recall_at_10 == observation.recall_at_20 == 1.0
    assert repository.exact_batch_calls == 1
    assert repository.exact_scalar_calls == 0
    assert repository.ann_calls == 8
    assert REQUIRED_EVIDENCE_FIELDS <= set(evidence)
    assert evidence["evidence_schema_version"] == 2
    assert evidence["evidence_source"] == "measured"
    assert evidence["probe_scope"] == "full"
    assert evidence["probe_count"] == evidence["probe_total"] == 8
    assert evidence["probe_coverage"] == 1.0
    assert evidence["sampling_method"] == "full"
    assert evidence["exact_method"] == "streamed_numpy_cosine_v1"
    assert evidence["exact_scan_rows"] == 8
    assert evidence["ann_query_count"] == 8


def test_large_corpus_caps_calls_and_emits_auditable_sample(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path, 500)
    observation, evidence, repository = _run_benchmark(tmp_path, chunks, max_probes=32)

    assert observation.recall_at_10 == observation.recall_at_20 == 1.0
    assert repository.exact_batch_calls == 1
    assert repository.exact_scalar_calls == 0
    assert repository.ann_calls == 32
    assert REQUIRED_EVIDENCE_FIELDS <= set(evidence)
    assert evidence["evidence_schema_version"] == 2
    assert evidence["evidence_source"] == "measured"
    assert evidence["probe_scope"] == "sampled"
    assert evidence["sampling_method"] == "bottom_k_sha256_v1"
    assert evidence["sampling_key_schema"] == "wiki_relative_path+chunk_suffix+kind+chunk_index:v1"
    assert evidence["probe_count"] == 32
    assert evidence["probe_total"] == 500
    assert evidence["probe_coverage"] == pytest.approx(32 / 500)
    assert evidence["result_limit"] == 20
    assert evidence["recall_aggregation"] == "minimum"
    assert evidence["exact_method"] == "streamed_numpy_cosine_v1"
    assert evidence["exact_scan_rows"] == 500
    assert evidence["ann_query_count"] == 32
    assert len(evidence["probe_keys"]) == 32
    assert len(set(evidence["probe_keys"])) == 32
    assert len(evidence["exact_result_ids"]) == 32
    assert len(evidence["candidate_result_ids"]) == 32
    expected_digest = hashlib.sha256("\n".join(evidence["probe_keys"]).encode()).hexdigest()
    assert evidence["probe_selection_sha256"] == expected_digest
    assert evidence["benchmark_duration_ms"] >= 0


def test_sampling_is_order_and_checkout_root_independent(tmp_path: Path) -> None:
    first_root = tmp_path / "mac-checkout"
    second_root = tmp_path / "windows-checkout"
    first_chunks = _chunks(first_root, 500)
    second_chunks = tuple(reversed(_chunks(second_root, 500)))

    _, first, _ = _run_benchmark(first_root, first_chunks, max_probes=32)
    _, second, _ = _run_benchmark(second_root, second_chunks, max_probes=32)

    assert first["probe_keys"] == second["probe_keys"]
    assert first["probe_selection_sha256"] == second["probe_selection_sha256"]


def test_observer_emits_complete_explicit_synthetic_evidence(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path, 32)

    class _NoQueries:
        def __getattr__(self, name):
            raise AssertionError(f"observer path must not query repository: {name}")

    observation, evidence, _ = _run_benchmark(
        tmp_path,
        chunks,
        max_probes=16,
        repository=_NoQueries(),
        observer=lambda _stats: _observation(recall_at_20=0.9),
    )

    assert observation.recall_at_20 == 0.9
    assert REQUIRED_EVIDENCE_FIELDS <= set(evidence)
    assert evidence["evidence_source"] == "observer"
    assert evidence["probe_scope"] == "synthetic"
    assert evidence["probe_count"] == 0
    assert evidence["probe_total"] == 32
    assert evidence["probe_coverage"] == 0.0
    assert evidence["exact_method"] == "observer"
    assert evidence["exact_scan_rows"] == 0
    assert evidence["exact_scan_batches"] == 0
    assert evidence["ann_query_count"] == 0
    assert evidence["probe_keys"] == []
    assert evidence["exact_result_ids"] == []
    assert evidence["candidate_result_ids"] == []


def test_sampled_miss_falls_back_and_policy_reports_scope(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path, 500)
    repository = _CountingRepository([chunk.chunk_id for chunk in chunks], miss_at_20=True)
    observation, evidence, _ = _run_benchmark(
        tmp_path, chunks, max_probes=32, repository=repository
    )

    decision = select_vector_policy(observation, _stats(len(chunks)), evidence=evidence)
    payload = decision.to_json()
    assert decision.selected_mode == "exact"
    assert "sampled" in decision.reason
    assert "32/500" in decision.reason
    assert payload["benchmark_scope"] == "sampled"
    assert payload["benchmark_probe_count"] == 32
    assert payload["benchmark_probe_total"] == 500


def test_eval_manifest_contract_rejects_unversioned_or_inconsistent_evidence() -> None:
    from eval.run_eval import validate_benchmark_contract

    valid = {
        "format_version": 5,
        "benchmark": {
            **{field: None for field in REQUIRED_EVIDENCE_FIELDS},
            "evidence_schema_version": 2,
            "evidence_source": "measured",
            "probe_scope": "sampled",
            "sampling_method": "bottom_k_sha256_v1",
            "sampling_key_schema": "wiki_relative_path+chunk_suffix+kind+chunk_index:v1",
            "probe_keys": ["concepts/a.md::abc::dense::0"],
            "probe_selection_sha256": hashlib.sha256(
                b"concepts/a.md::abc::dense::0"
            ).hexdigest(),
            "probe_count": 1,
            "probe_total": 2,
            "probe_coverage": 0.5,
            "result_limit": 20,
            "recall_aggregation": "minimum",
            "benchmark_duration_ms": 1.0,
            "probe_selection_ms": 1.0,
            "exact_verification_ms": 1.0,
            "ann_verification_ms": 1.0,
            "recall_assembly_ms": 1.0,
            "exact_method": "streamed_numpy_cosine_v1",
            "exact_scan_rows": 2,
            "exact_scan_batches": 1,
            "ann_query_count": 1,
            "exact_result_ids": [["a"]],
            "candidate_result_ids": [["a"]],
        },
        "policy": {
            "selected_mode": "ann",
            "benchmark_scope": "sampled",
            "benchmark_probe_count": 1,
            "benchmark_probe_total": 2,
        },
    }
    assert validate_benchmark_contract(valid)["probe_scope"] == "sampled"

    stale = json.loads(json.dumps(valid))
    stale["format_version"] = 4
    with pytest.raises(ValueError, match="format_version"):
        validate_benchmark_contract(stale)

    inconsistent = json.loads(json.dumps(valid))
    inconsistent["policy"]["benchmark_probe_count"] = 2
    with pytest.raises(ValueError, match="policy"):
        validate_benchmark_contract(inconsistent)


def test_real_build_over_cap_publishes_bounded_v5_evidence(tmp_path: Path) -> None:
    wiki = tmp_path / "Wiki"
    wiki.mkdir()
    index_dir = tmp_path / ".index"
    total = 257
    # issue #47 strict two-table contract: a canonical build must carry both
    # lexical (sparse) and vector (dense) chunks. This test only exercises the
    # dense/vector benchmark-sampling path, but the production build invariant
    # still applies, so we emit a matching sparse chunk per page too.
    dense_chunks = tuple(
        SparseChunk(
            chunk_id=f"{(wiki / f'page-{index:04d}.md').resolve()}::dense::{index:016x}",
            page_id=str((wiki / f"page-{index:04d}.md").resolve()),
            path=str((wiki / f"page-{index:04d}.md").resolve()),
            title=f"Page {index}",
            text=f"dense text {index}",
            fts_text=f"DENSETERM{index}",
            chunk_kind="dense",
            chunk_index=index,
            content_hash=hashlib.sha256(f"dense text {index}".encode()).hexdigest(),
        )
        for index in range(total)
    )
    sparse_chunks = tuple(
        SparseChunk(
            chunk_id=f"{(wiki / f'page-{index:04d}.md').resolve()}::sparse::{index:016x}",
            page_id=str((wiki / f"page-{index:04d}.md").resolve()),
            path=str((wiki / f"page-{index:04d}.md").resolve()),
            title=f"Page {index}",
            text=f"sparse text {index}",
            fts_text=f"SPARSETERM{index}",
            chunk_kind="sparse",
            chunk_index=index,
            content_hash=hashlib.sha256(f"sparse text {index}".encode()).hexdigest(),
        )
        for index in range(total)
    )
    chunks = dense_chunks + sparse_chunks

    service = IndexBuildService(
        LanceDbIndexRepository(index_dir),
        reopen_storage=LanceDbIndexRepository,
        manifest_store=FilesystemIndexManifest(),
        post_commit_journal=FilesystemPostCommitJournal(index_dir),
        benchmark_max_probes=256,
    )

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return [
            [
                math.cos(index / total * math.tau),
                math.sin(index / total * math.tau),
                math.cos(index / total * math.tau * 3),
                math.sin(index / total * math.tau * 3),
            ]
            for index, _text in enumerate(texts)
        ]

    artifact = service.build(wiki, index_dir, embed=embed, sparse_chunks=chunks)
    manifest_bytes = artifact.manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    benchmark = manifest["benchmark"]
    policy = manifest["policy"]

    assert manifest["format_version"] == 5
    assert benchmark["probe_scope"] == "sampled"
    assert benchmark["probe_count"] == 256
    assert benchmark["probe_total"] == total
    assert len(benchmark["exact_result_ids"]) == 256
    assert len(benchmark["candidate_result_ids"]) == 256
    assert policy["benchmark_scope"] == "sampled"
    assert policy["benchmark_probe_count"] == 256
    assert policy["benchmark_probe_total"] == total
    assert len(manifest_bytes) <= 10 * 1024 * 1024
    assert (index_dir / "ACTIVE_INDEX").is_file()


def _reduced_comparator_args(tmp_path: Path) -> Namespace:
    return Namespace(
        rows=257,
        dimensions=8,
        max_probes=256,
        ef_grid=benchmark_ann_build.DECISION_EF_GRID,
        candidates=benchmark_ann_build.CANDIDATES,
        max_seconds=60.0,
        max_exact_seconds=10.0,
        row_batch_size=64,
        query_batch_size=32,
        max_evidence_bytes=10 * 1024 * 1024,
        work_dir=tmp_path / "work",
        output=tmp_path / "index-benchmark.json",
        error_output=tmp_path / "index-benchmark-error.json",
    )


class _InlineFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _InlineTwoWorkerSchedule:
    """Exercise real LanceDB records where the sandbox disallows semaphores."""
    def __init__(self, *, max_workers: int, mp_context):
        assert max_workers == 2
        assert mp_context.get_start_method() == "spawn"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def submit(self, function, *args):
        return _InlineFuture(function(*args))


class _WorkerStartupFailure:
    def __init__(self, *, max_workers: int, mp_context):
        assert max_workers == 2
        assert mp_context.get_start_method() == "spawn"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def submit(self, *_args):
        raise RuntimeError("simulated spawn worker startup failure")


def test_reduced_real_matrix_uses_one_cached_table_per_candidate_and_complete_normal_queries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The performance seam must retain every real normal ANN request."""
    monkeypatch.setattr(
        benchmark_ann_build,
        "_runtime_identity",
        lambda: {"lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0"},
    )
    monkeypatch.setattr(benchmark_ann_build, "ProcessPoolExecutor", _InlineTwoWorkerSchedule)

    payload = benchmark_ann_build.run(_reduced_comparator_args(tmp_path))

    assert len(payload["candidate_runs"]) == 2
    assert len(payload["records"]) == 2 * 6
    assert payload["exact"]["method"] == "streamed_numpy_cosine_v1"
    assert payload["queries"]["zero_overlap_count"] == 0
    assert all(len(record["queries"]) == 256 for record in payload["records"])
    assert all(record["candidate_run_id"] for record in payload["records"])
    assert all(record["query_group_wall_ms"] >= 0 for record in payload["records"])
    assert {run["dense_table_open_count"] for run in payload["candidate_runs"]} == {1}
    assert {run["normal_ann_request_count"] for run in payload["candidate_runs"]} == {6 * 256}
    assert payload["environment"]["worker_schedule"]["start_method"] == "spawn"
    assert payload["environment"]["worker_schedule"]["configured_workers"] == 2
    assert payload["environment"]["worker_schedule"]["effective_workers"] == 2
    assert {run["worker"]["start_method"] for run in payload["candidate_runs"]} == {"spawn"}
    assert payload["matrix_timing"]["end_monotonic"] - payload["matrix_timing"]["start_monotonic"] \
        == pytest.approx(payload["benchmark_wall_seconds"], abs=0.01)


def test_complete_matrix_timer_and_candidate_run_references_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        benchmark_ann_build,
        "_runtime_identity",
        lambda: {"lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0"},
    )
    monkeypatch.setattr(benchmark_ann_build, "ProcessPoolExecutor", _InlineTwoWorkerSchedule)
    payload = benchmark_ann_build.run(_reduced_comparator_args(tmp_path))
    assert benchmark_ann_build.validate_evidence(payload) is payload

    for mutation in (
        lambda value: value.pop("candidate_runs"),
        lambda value: value["candidate_runs"].pop(),
        lambda value: value["records"].pop(),
        lambda value: value["records"][0].pop("candidate_run_id"),
        lambda value: value["matrix_timing"].update(start_monotonic=value["records"][0]["query_group_end_monotonic"]),
        lambda value: value["candidate_runs"][0].update(index_build_end_monotonic=float("nan")),
    ):
        broken = deepcopy(payload)
        mutation(broken)
        with pytest.raises(ValueError):
            benchmark_ann_build.validate_evidence(broken)

    boundary = deepcopy(payload)
    boundary["benchmark_wall_seconds"] = 60.0
    boundary["matrix_timing"]["end_monotonic"] = (
        boundary["matrix_timing"]["start_monotonic"] + 60.0
    )
    assert benchmark_ann_build.validate_evidence(boundary) is boundary
    boundary["benchmark_wall_seconds"] = 60.000001
    boundary["matrix_timing"]["end_monotonic"] += 0.000001
    with pytest.raises(ValueError, match="wall-time cap"):
        benchmark_ann_build.validate_evidence(boundary)


def test_spawn_worker_failure_writes_rejected_error_evidence_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _reduced_comparator_args(tmp_path)
    monkeypatch.setattr(benchmark_ann_build, "ProcessPoolExecutor", _WorkerStartupFailure)

    with pytest.raises(RuntimeError, match="simulated spawn worker startup failure"):
        benchmark_ann_build.run(args)

    assert not args.output.exists()
    error = json.loads(args.error_output.read_text(encoding="utf-8"))
    assert error["status"] == "reject-evidence"
    assert error["worker_schedule"]["start_method"] == "spawn"
    assert error["worker_schedule"]["configured_workers"] == 2
    assert set(error["candidate_assignment"]) == set(benchmark_ann_build.CANDIDATES)
    assert error["error"]["class"] == "RuntimeError"
    assert "simulated spawn worker startup failure" in error["error"]["message"]
    assert error["error"]["traceback"]
    with pytest.raises(ValueError):
        benchmark_ann_build.validate_evidence(error)


def test_post_run_wall_cap_rejection_withholds_scale_evidence_and_persists_timing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _reduced_comparator_args(tmp_path)
    args.max_seconds = 0.0
    monkeypatch.setattr(
        benchmark_ann_build,
        "_runtime_identity",
        lambda: {"lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0"},
    )
    monkeypatch.setattr(benchmark_ann_build, "ProcessPoolExecutor", _InlineTwoWorkerSchedule)

    with pytest.raises(RuntimeError, match="wall-time cap"):
        benchmark_ann_build.run(args)

    assert not args.output.exists()
    error = json.loads(args.error_output.read_text(encoding="utf-8"))
    assert error["status"] == "reject-evidence"
    assert error["failure_phase"] == "post_run_validation"
    assert error["observed"]["benchmark_wall_seconds"] > 0
    assert error["observed"]["exact_time_ms"] >= 0
    assert error["error"]["class"] == "RuntimeError"
    assert "wall-time cap" in error["error"]["message"]
    assert error["raw_staging_path"]
    assert Path(error["raw_staging_path"]).is_file()
    with pytest.raises(ValueError):
        benchmark_ann_build.validate_evidence(error)


def test_calibration_selects_omp_and_derives_cap_from_five_complete_runs() -> None:
    calibration = benchmark_ann_build.build_calibration_record(
        head_sha="a" * 40,
        lock_identity={"lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0"},
        configuration={"cpu_count": 4, "worker_schedule": {"start_method": "spawn"}},
        repetitions={
            1: [101.0, 100.0, 102.0, 99.0, 100.0],
            2: [98.0, 97.0, 99.0, 98.0, 100.0],
        },
    )

    assert calibration["status"] == "non_accepting_calibration"
    assert calibration["selected_omp_threads"] == 2
    assert calibration["selection"]["median_seconds"] == 98.0
    assert calibration["calculated_cap_seconds"] == 103
    assert calibration["rule_version"] == benchmark_ann_build.CALIBRATION_RULE_VERSION
    assert len(calibration["repetitions"]["1"]) == 5
    assert len(calibration["repetitions"]["2"]) == 5

    tied = benchmark_ann_build.build_calibration_record(
        head_sha="a" * 40,
        lock_identity={"lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0"},
        configuration={"cpu_count": 4, "worker_schedule": {"start_method": "spawn"}},
        repetitions={1: [100.0] * 5, 2: [100.0] * 5},
    )
    assert tied["selected_omp_threads"] == 1

    with pytest.raises(ValueError, match="five"):
        benchmark_ann_build.build_calibration_record(
            head_sha="a" * 40,
            lock_identity={"lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0"},
            configuration={"cpu_count": 4, "worker_schedule": {"start_method": "spawn"}},
            repetitions={1: [100.0] * 5, 2: [100.0] * 4},
        )
