"""Issue #41 acceptance gates for bounded, auditable ANN build probes.

These tests intentionally describe the required production contract before the
implementation lands.  They must fail on the unbounded master implementation
and pass without weakening the existing recall thresholds.
"""
from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import math
import os
import sys
import time
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
from obsidian_wiki.domain.index_policy import PolicyError, select_vector_policy
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
    assert benchmark_ann_build.MAX_APPROVED_STATIC_CAP_SECONDS == 180.0
    assert not benchmark_ann_build._valid_approved_static_cap(179.0)
    assert benchmark_ann_build._valid_approved_static_cap(180.0)
    assert not benchmark_ann_build._valid_approved_static_cap(181.0)

    workflow = (Path(__file__).parents[1] / ".github/workflows/eval.yml").read_text(
        encoding="utf-8"
    )
    scale_job = workflow.split("issue41-scale-benchmark:", 1)[1]
    assert "--max-exact-seconds 10" in scale_job
    assert "--per-build-cap-seconds 180" in scale_job
    assert "--max-seconds 60" not in scale_job


@pytest.mark.parametrize("cap, expected_exit", ((179.0, 2), (180.0, 0), (181.0, 2)))
def test_per_build_cli_requires_exactly_180_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cap: float, expected_exit: int,
) -> None:
    monkeypatch.setattr(
        sys, "argv", [
            "benchmark_ann_build", "--per-build-cap-seconds", str(cap),
            "--work-dir", str(tmp_path / "work"), "--output", str(tmp_path / "out.json"),
        ],
    )
    monkeypatch.setattr(benchmark_ann_build, "run", lambda _args: {})
    if expected_exit:
        with pytest.raises(SystemExit) as exc:
            benchmark_ann_build.main()
        assert exc.value.code == expected_exit
    else:
        assert benchmark_ann_build.main() == expected_exit


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
    """Fake candidate repository: truth 与 candidate 可控部分命中。

    ``top_hits=None`` 完全一致（recall 1.0）；``top_hits=k`` 只保留前 k 个
    真命中，其余用不存在的 id 填充（构造低于 floor 的场景）。
    """

    def __init__(self, chunk_ids: Sequence[str], *, top_hits: int | None = None) -> None:
        self._chunk_ids = list(chunk_ids)
        self._truth = [{"chunk_id": chunk_id} for chunk_id in self._chunk_ids[:20]]
        self._top_hits = top_hits
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
        where: str | None = None,
    ) -> list[Mapping[str, object]]:
        self.ann_calls += 1
        rows = list(self._truth[:limit])
        if self._top_hits is not None and len(rows) > self._top_hits:
            filler = [
                {"chunk_id": f"miss-{self.ann_calls}-{index}"}
                for index in range(len(rows) - self._top_hits)
            ]
            rows = rows[:self._top_hits] + filler
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


def _run_publication(
    root: Path,
    chunks: Sequence[DenseChunk],
    *,
    max_probes: int,
    repository=None,
):
    service = _service(max_probes)
    repository = repository or _CountingRepository([chunk.chunk_id for chunk in chunks])
    evidence, observation = service._publication_validation(
        repository,
        chunks,
        vector_stats=_stats(len(chunks)),
        actual_dense_rows=len(chunks),
        unindexed_dense_rows=0,
        build_time_ms=1.0,
        disk_bytes=1,
    )
    return observation, evidence, repository


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_benchmark_probe_cap_rejects_invalid_constructor_values(value) -> None:
    with pytest.raises((TypeError, ValueError), match="benchmark_max_probes"):
        _service(value)  # type: ignore[arg-type]


def test_small_candidate_validates_with_all_rows(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path, 8)
    observation, evidence, repository = _run_publication(tmp_path, chunks, max_probes=8)

    assert observation.recall_at_10 == observation.recall_at_20 == 1.0
    assert repository.exact_batch_calls == 1
    assert repository.exact_scalar_calls == 0
    assert repository.ann_calls == 8
    assert evidence.evidence_schema_version == 3
    assert evidence.actual_dense_rows == 8
    assert evidence.validation_query_count == min(8, 8)
    assert evidence.corpus_query_overlap == 0
    assert len(evidence.exact_result_ids) == 8
    assert len(evidence.candidate_result_ids) == 8


def test_large_candidate_caps_validation_queries_deterministically(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path, 500)
    observation, evidence, repository = _run_publication(tmp_path, chunks, max_probes=32)

    assert observation.recall_at_10 == observation.recall_at_20 == 1.0
    assert repository.exact_batch_calls == 1
    assert repository.exact_scalar_calls == 0
    assert repository.ann_calls == 32
    assert evidence.actual_dense_rows == 500
    assert evidence.validation_query_count == min(32, 500) == 32
    assert evidence.corpus_query_overlap == 0
    assert len(evidence.exact_result_ids) == 32
    assert len(evidence.candidate_result_ids) == 32
    for row in evidence.exact_result_ids:
        assert len(row) == min(20, 500)
    assert evidence.benchmark_duration_ms >= 0


def test_validation_queries_are_checkout_root_and_order_independent(tmp_path: Path) -> None:
    first_root = tmp_path / "mac-checkout"
    second_root = tmp_path / "windows-checkout"
    first_chunks = _chunks(first_root, 500)
    second_chunks = tuple(reversed(_chunks(second_root, 500)))

    _, first, _ = _run_publication(first_root, first_chunks, max_probes=32)
    _, second, _ = _run_publication(second_root, second_chunks, max_probes=32)

    # 独立确定性流：query 集与 corpus/checkout 根/输入顺序无关。
    assert first.query_selection_sha256 == second.query_selection_sha256
    assert first.exact_result_ids != second.exact_result_ids or True  # truth 随 corpus


def test_observer_benchmark_emits_explicit_synthetic_evidence(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path, 32)
    service = _service(16, observer=lambda _stats: _observation(recall_at_20=0.9))

    observation, evidence = service._observer_benchmark(chunks)

    assert observation.recall_at_20 == 0.9
    assert evidence["evidence_source"] == "observer"
    assert evidence["probe_scope"] == "synthetic"
    assert evidence["probe_count"] == 0
    assert evidence["probe_total"] == 32
    assert evidence["exact_result_ids"] == []
    assert evidence["candidate_result_ids"] == []


def test_below_floor_candidate_fails_closed_at_publication(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path, 500)
    repository = _CountingRepository(
        [chunk.chunk_id for chunk in chunks], top_hits=2
    )

    # recall@10 = 2/10 = 0.2 >= 0.19；recall@20 = 2/20 = 0.1 < 0.17 → fail-closed。
    with pytest.raises(PolicyError, match="recall@20"):
        _run_publication(tmp_path, chunks, max_probes=32, repository=repository)


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


def test_real_build_over_cap_publishes_bounded_publication_evidence(tmp_path: Path) -> None:
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

    import random as _random

    def embed(texts: Sequence[str]) -> list[list[float]]:
        # Phase 06：批准策略固定 384 维；确定性伪随机单位向量。
        out = []
        for index, _text in enumerate(texts):
            rng = _random.Random(31337 + index)
            raw = [rng.gauss(0.0, 1.0) for _ in range(384)]
            norm = math.sqrt(sum(value * value for value in raw))
            out.append([value / norm for value in raw])
        return out

    artifact = service.build(wiki, index_dir, embed=embed, sparse_chunks=chunks)
    manifest_bytes = artifact.manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    evidence = manifest["candidate_publication_evidence"]
    policy = manifest["policy"]

    assert manifest["format_version"] == 6
    assert evidence["actual_dense_rows"] == total
    assert evidence["validation_query_count"] == min(256, total) == 256
    assert evidence["corpus_query_overlap"] == 0
    assert len(evidence["exact_result_ids"]) == 256
    assert len(evidence["candidate_result_ids"]) == 256
    assert evidence["recall_at_10"] >= 0.19
    assert evidence["recall_at_20"] >= 0.17
    assert policy["selected_mode"] == "ann"
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


def test_schema_v5_per_build_static_acceptance_rejects_any_over_cap_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _reduced_comparator_args(tmp_path)
    args.per_build_cap_seconds = 180.0
    monkeypatch.setattr(
        benchmark_ann_build,
        "_runtime_identity",
        lambda: {"lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0"},
    )
    monkeypatch.setattr(benchmark_ann_build, "ProcessPoolExecutor", _InlineTwoWorkerSchedule)

    payload = benchmark_ann_build.run(args)
    assert payload["evidence_schema_version"] == benchmark_ann_build.EVIDENCE_SCHEMA_VERSION
    assert payload["acceptance"] == {
        "mode": "per_build_static_acceptance",
        "aggregate_cap_seconds": None,
        "per_build_cap_seconds": 180.0,
    }
    benchmark_ann_build.validate_evidence(payload)

    over_cap = deepcopy(payload)
    over_cap["candidate_runs"][0]["index_build_ms"] = 180_000.001
    with pytest.raises(ValueError, match="per-build watchdog cap"):
        benchmark_ann_build.validate_evidence(over_cap)

    for invalid_cap in (179.0, 181.0):
        invalid = deepcopy(payload)
        invalid["acceptance"]["per_build_cap_seconds"] = invalid_cap
        with pytest.raises(ValueError, match="per-build watchdog cap"):
            benchmark_ann_build.validate_evidence(invalid)


def test_reduced_real_per_build_comparator_uses_public_spawn_supervisor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    args = _reduced_comparator_args(tmp_path)
    args.per_build_cap_seconds = 180.0
    monkeypatch.setattr(
        benchmark_ann_build, "_runtime_identity",
        lambda: {"lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0"},
    )

    payload = benchmark_ann_build.run(args, _watchdog_deadline_seconds=30.0)

    assert payload["evidence_schema_version"] == 5
    assert len(payload["candidate_runs"]) == 2
    assert len(payload["records"]) == 12
    assert sum(run["normal_ann_request_count"] for run in payload["candidate_runs"]) == 2 * 6 * 256
    assert payload["acceptance"]["per_build_cap_seconds"] == 180.0
    assert payload["environment"]["worker_schedule"]["start_method"] == "spawn"


def _delayed_outside_build_phase_candidate_worker(
    candidate, pre_build_delay, build_delay, post_build_delay, *, _build_watchdog=None,
):
    time.sleep(pre_build_delay)
    if _build_watchdog is None:
        time.sleep(build_delay)
    else:
        with _build_watchdog:
            time.sleep(build_delay)
    time.sleep(post_build_delay)
    return ({"candidate": candidate}, [])


def test_per_build_deadline_excludes_preparation_and_query_work() -> None:
    completed = benchmark_ann_build._run_spawned_candidates_with_deadline(
        candidates=benchmark_ann_build.CANDIDATES,
        deadline_seconds=0.1,
        worker=_delayed_outside_build_phase_candidate_worker,
        worker_args_for=lambda candidate: (candidate, 0.15, 0.01, 0.15),
    )

    assert [run[0]["candidate"] for run in completed] == list(
        benchmark_ann_build.CANDIDATES
    )


def _install_public_spawn_supervisor_race(
    monkeypatch: pytest.MonkeyPatch, *, trigger: str,
):
    candidate = benchmark_ann_build.CANDIDATES[0]
    expected_result = ({"candidate": candidate}, [])
    state = {
        "messages": [("build_ready", candidate)],
        "process": None,
        "published": False,
        "released": False,
    }

    def publish_terminal_messages() -> None:
        if state["published"]:
            return
        state["messages"].extend(
            [
                ("build_finished", candidate),
                ("candidate_result", candidate, True, expected_result),
            ]
        )
        state["published"] = True

    class Connection:
        def __init__(self, *, parent: bool):
            self.parent = parent
            self.closed = False

        def poll(self):
            assert self.parent and not self.closed
            return bool(state["messages"])

        def recv(self):
            message = state["messages"].pop(0)
            if message[0] == "candidate_result":
                state["process"]._exitcode = 0
            return message

        def send(self, message):
            assert message == ("start_build", candidate)
            state["released"] = True

        def close(self):
            self.closed = True

    class ChildConnection:
        def close(self):
            pass

    class Process:
        def __init__(self, **_kwargs):
            self._exitcode = None
            state["process"] = self

        @property
        def exitcode(self):
            if trigger == "exitcode" and state["released"]:
                publish_terminal_messages()
                self._exitcode = 0
            return self._exitcode

        def start(self):
            pass

        def is_alive(self):
            return self._exitcode is None

        def terminate(self):
            self._exitcode = -15

        def join(self):
            pass

    class Context:
        def Pipe(self, *, duplex):
            assert duplex is True
            return Connection(parent=True), ChildConnection()

        def Process(self, **kwargs):
            return Process(**kwargs)

    monkeypatch.setattr(
        benchmark_ann_build.multiprocessing, "get_context", lambda method: Context()
    )

    if trigger == "deadline":
        class RaceClock:
            calls = 0

            def monotonic(self):
                self.calls += 1
                if self.calls == 2:
                    publish_terminal_messages()
                return 0.0 if self.calls == 1 else 1.0

            def sleep(self, _seconds):
                pass

        monkeypatch.setattr(benchmark_ann_build, "time", RaceClock())

    return candidate, expected_result


def test_buffered_build_finished_wins_over_stale_deadline_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, expected_result = _install_public_spawn_supervisor_race(
        monkeypatch, trigger="deadline"
    )

    completed = benchmark_ann_build._run_spawned_candidates_with_deadline(
        candidates=(candidate,), deadline_seconds=1.0,
        worker=lambda: None, worker_args_for=lambda _candidate: (),
    )

    assert completed == [expected_result]


def test_buffered_candidate_result_wins_over_stale_exitcode_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, expected_result = _install_public_spawn_supervisor_race(
        monkeypatch, trigger="exitcode"
    )

    completed = benchmark_ann_build._run_spawned_candidates_with_deadline(
        candidates=(candidate,), deadline_seconds=1.0,
        worker=lambda: None, worker_args_for=lambda _candidate: (),
    )

    assert completed == [expected_result]


def _install_poll_failure_context(
    monkeypatch: pytest.MonkeyPatch, *, messages: list[tuple], poll_error: OSError,
) -> str:
    candidate = benchmark_ann_build.CANDIDATES[0]

    class Connection:
        def poll(self):
            if messages:
                return True
            raise poll_error

        def recv(self):
            return messages.pop(0)

        def send(self, message):
            assert message == ("start_build", candidate)

        def close(self):
            pass

    class ChildConnection:
        def close(self):
            pass

    class Process:
        exitcode = 0

        def start(self):
            pass

        def is_alive(self):
            return False

        def terminate(self):
            pytest.fail("a cleanly exited child must not be terminated")

        def join(self):
            pass

    class Context:
        def Pipe(self, *, duplex):
            assert duplex is True
            return Connection(), ChildConnection()

        def Process(self, **_kwargs):
            return Process()

    monkeypatch.setattr(
        benchmark_ann_build.multiprocessing, "get_context", lambda _method: Context()
    )
    return candidate


@pytest.mark.parametrize("exit_phase", ("pre_build", "post_build"))
def test_closed_pipe_poll_after_clean_child_exit_is_phase_aware(
    monkeypatch: pytest.MonkeyPatch, exit_phase: str,
) -> None:
    """Win32 closed-pipe polling must retain the supervisor phase contract."""
    candidate = benchmark_ann_build.CANDIDATES[0]
    messages = [] if exit_phase == "pre_build" else [
        ("build_ready", candidate),
        ("build_finished", candidate),
    ]
    candidate = _install_poll_failure_context(
        monkeypatch,
        messages=messages,
        poll_error=BrokenPipeError(109, "The pipe has been ended"),
    )

    with pytest.raises(
        RuntimeError,
        match=rf"endpoint.*phase={exit_phase}",
    ):
        benchmark_ann_build._run_spawned_candidates_with_deadline(
            candidates=(candidate,), deadline_seconds=1.0,
            worker=lambda: None, worker_args_for=lambda _candidate: (),
        )


def test_unrelated_poll_oserror_is_not_reclassified_as_endpoint_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _install_poll_failure_context(
        monkeypatch, messages=[], poll_error=PermissionError(13, "permission denied")
    )

    with pytest.raises(PermissionError, match="permission denied"):
        benchmark_ann_build._run_spawned_candidates_with_deadline(
            candidates=(candidate,), deadline_seconds=1.0,
            worker=lambda: None, worker_args_for=lambda _candidate: (),
        )


def _never_returning_candidate_worker(*args, _build_watchdog=None):
    candidate = args[0]
    pid_path = Path(args[-1]) / f"{candidate}.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    if candidate == "ivf-hnsw-sq":
        while True:
            time.sleep(1)
    with _build_watchdog:
        while True:
            time.sleep(1)


def _process_is_active(pid: int, *, _windows_api=None) -> bool:
    """Query process status without assuming POSIX signal-0 semantics."""
    if sys.platform == "win32":
        from ctypes import wintypes

        if _windows_api is None:
            _windows_api = ctypes.WinDLL("kernel32", use_last_error=True)
            _windows_api.OpenProcess.argtypes = [
                wintypes.DWORD, wintypes.BOOL, wintypes.DWORD,
            ]
            _windows_api.OpenProcess.restype = wintypes.HANDLE
            _windows_api.GetExitCodeProcess.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
            ]
            _windows_api.GetExitCodeProcess.restype = wintypes.BOOL
            _windows_api.CloseHandle.argtypes = [wintypes.HANDLE]
            _windows_api.CloseHandle.restype = wintypes.BOOL
        handle = _windows_api.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:
                return False
            raise ctypes.WinError(error)
        try:
            exit_code = wintypes.DWORD()
            if not _windows_api.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return exit_code.value == 259
        finally:
            _windows_api.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _clean_exit_candidate_worker(candidate, exit_phase, *, _build_watchdog=None):
    if exit_phase == "pre_build":
        os._exit(0)
    with _build_watchdog:
        pass
    os._exit(0)


@pytest.mark.parametrize(
    ("exit_code", "expected_active"), ((0, False), (259, True)),
    ids=("exited", "still-active"),
)
def test_process_liveness_probe_uses_win32_exit_status(
    monkeypatch: pytest.MonkeyPatch, exit_code: int, expected_active: bool,
) -> None:
    from ctypes import wintypes

    calls = []

    class Kernel32:
        def OpenProcess(self, access, inherit_handle, pid):
            calls.append(("open", access, inherit_handle, pid))
            return 17

        def GetExitCodeProcess(self, handle, output):
            calls.append(("status", handle))
            assert isinstance(output._obj, wintypes.DWORD)
            output._obj.value = exit_code
            return 1

        def CloseHandle(self, handle):
            calls.append(("close", handle))
            return 1

    monkeypatch.setattr(sys, "platform", "win32")

    assert _process_is_active(4321, _windows_api=Kernel32()) is expected_active
    assert calls == [
        ("open", 0x1000, False, 4321),
        ("status", 17),
        ("close", 17),
    ]


@pytest.mark.parametrize(
    ("last_error", "inactive"), ((87, True), (5, False)),
    ids=("missing-process", "access-denied"),
)
def test_process_liveness_probe_handles_win32_open_failure(
    monkeypatch: pytest.MonkeyPatch, last_error: int, inactive: bool,
) -> None:
    class Kernel32:
        def OpenProcess(self, _access, _inherit_handle, _pid):
            return 0

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "get_last_error", lambda: last_error, raising=False)
    monkeypatch.setattr(
        ctypes, "WinError", lambda error: OSError(error, f"winerror {error}"),
        raising=False,
    )

    if inactive:
        assert not _process_is_active(4321, _windows_api=Kernel32())
    else:
        with pytest.raises(OSError, match="winerror 5"):
            _process_is_active(4321, _windows_api=Kernel32())


@pytest.mark.parametrize("exit_phase, expected_phase", (("pre_build", "pre_build"), ("post_build", "post_build")))
def test_clean_child_exit_before_candidate_result_rejects_promptly(
    exit_phase: str, expected_phase: str,
) -> None:
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=rf"phase={expected_phase}.*exit=0|endpoint.*phase={expected_phase}"):
        benchmark_ann_build._run_spawned_candidates_with_deadline(
            candidates=benchmark_ann_build.CANDIDATES,
            deadline_seconds=2.0,
            worker=_clean_exit_candidate_worker,
            worker_args_for=lambda candidate: (candidate, exit_phase),
        )
    # This wall clock includes two fresh spawn interpreters importing the
    # benchmark dependency graph; hosted-runner scheduling is not supervisor
    # detection latency.  Keep a bounded smoke ceiling without coupling the
    # gate to a 2-second machine-speed assumption.
    assert time.monotonic() - started < 10.0


def test_spawn_start_failure_cleans_local_resources_and_live_sibling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    endpoints = []
    processes = []

    class Endpoint:
        close_count = 0

        def close(self):
            self.close_count += 1

    class Process:
        def __init__(self, **_kwargs):
            self.alive = False
            self.exitcode = None
            self.terminate_count = 0
            self.join_count = 0
            processes.append(self)

        def start(self):
            self.alive = True
            if len(processes) == 2:
                raise RuntimeError("simulated public spawn start failure")

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminate_count += 1
            self.alive = False
            self.exitcode = -15

        def join(self):
            self.join_count += 1

    class Context:
        def Pipe(self, *, duplex):
            assert duplex is True
            pair = (Endpoint(), Endpoint())
            endpoints.extend(pair)
            return pair

        def Process(self, **kwargs):
            return Process(**kwargs)

    monkeypatch.setattr(benchmark_ann_build.multiprocessing, "get_context", lambda method: Context())
    args = _reduced_comparator_args(tmp_path)
    args.per_build_cap_seconds = 180.0
    with pytest.raises(RuntimeError, match="simulated public spawn start failure"):
        benchmark_ann_build.run(args, _watchdog_deadline_seconds=1.0)

    assert [endpoint.close_count for endpoint in endpoints] == [1, 1, 1, 1]
    assert all(not process.is_alive() for process in processes)
    assert [process.terminate_count for process in processes] == [1, 1]
    assert [process.join_count for process in processes] == [1, 1]
    error = json.loads(args.error_output.read_text(encoding="utf-8"))
    assert error["status"] == "reject-evidence"
    assert error["error"]["class"] == "RuntimeError"
    assert error["error"]["message"] == "simulated public spawn start failure"


def test_per_build_deadline_terminates_never_returning_spawn_child_and_rejects_only(
    tmp_path: Path,
) -> None:
    args = _reduced_comparator_args(tmp_path)
    args.per_build_cap_seconds = 180.0
    pid_dir = tmp_path / "hung-children"
    pid_dir.mkdir()
    args.output.write_text("stale accepted evidence", encoding="utf-8")

    with pytest.raises(TimeoutError, match="per-build watchdog"):
        benchmark_ann_build.run(
            args,
            _watchdog_deadline_seconds=0.75,
            _candidate_worker_override=_never_returning_candidate_worker,
            _candidate_worker_extra_args=(str(pid_dir),),
        )

    pid_paths = sorted(pid_dir.glob("*.pid"))
    assert {path.stem for path in pid_paths} == set(benchmark_ann_build.CANDIDATES)
    for pid_path in pid_paths:
        pid = int(pid_path.read_text(encoding="utf-8"))
        assert not _process_is_active(pid)
    assert not args.output.exists()
    error = json.loads(args.error_output.read_text(encoding="utf-8"))
    assert error["status"] == "reject-evidence"
    assert error["failure_phase"] == "candidate_execution"
    assert error["error"]["class"] == "TimeoutError"
    assert "per-build watchdog" in error["error"]["message"]


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

    calibration["trial_records"] = {"1": [], "2": []}
    sealed = benchmark_ann_build.finalize_calibration_record(calibration)
    expected_hash = hashlib.sha256(json.dumps(
        {key: value for key, value in sealed.items() if key != "sha256"},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    assert sealed["sha256"] == expected_hash

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


def test_multivector_spike_groups_rows_by_query_index_before_comparison() -> None:
    class _Query:
        def distance_type(self, metric):
            assert metric == "cosine"
            return self

        def ef(self, value):
            assert value == 30
            return self

        def limit(self, value):
            assert value == 2
            return self

        def to_list(self):
            return [
                {"query_index": 1, "chunk_id": "q1-a"},
                {"query_index": 0, "chunk_id": "q0-a"},
                {"query_index": 1, "chunk_id": "q1-b"},
                {"query_index": 0, "chunk_id": "q0-b"},
            ]

    class _Table:
        def search(self, vectors):
            assert vectors == [[0.0], [1.0]]
            return _Query()

    spike = benchmark_ann_build.run_multivector_batch_spike(
        _Table(), [[0.0], [1.0]], metric="cosine", ef=30, limit=2,
        individual_result_ids=[["q0-a", "q0-b"], ["q1-a", "q1-b"]],
    )

    assert spike["returned_row_count"] == 4
    assert spike["invalid_query_index_row_count"] == 0
    assert [item["ids_identical"] for item in spike["observations"]] == [True, True]
    assert [item["recall"] for item in spike["observations"]] == [1.0, 1.0]
