"""Pure D-02/D-03 ANN-promotion contract tests."""
from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from obsidian_wiki.domain.index_models import (  # noqa: E402
    AnnDecisionEvidence,
    ANN_DECISION_EVIDENCE_SCHEMA_VERSION,
    BenchmarkObservation,
    CandidatePublicationEvidence,
    CANDIDATE_PUBLICATION_EVIDENCE_SCHEMA_VERSION,
    FtsIndexConfig,
    IndexStats,
    ProductionAnnPolicy,
    VectorIndexConfig,
)
from obsidian_wiki.domain.index_policy import (  # noqa: E402
    PolicyError,
    load_ann_policy_record,
    production_policy_sha256,
    select_vector_policy,
    validate_ann_decision_evidence,
    validate_candidate_publication_evidence,
)
from obsidian_wiki.application.index_build_service import IndexBuildService  # noqa: E402
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository  # noqa: E402
from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest  # noqa: E402
from obsidian_wiki.infrastructure.filesystem_post_commit_journal import (  # noqa: E402
    FilesystemPostCommitJournal,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _benchmark(*, recall_at_10: float = 1.0, recall_at_20: float = 1.0) -> BenchmarkObservation:
    return BenchmarkObservation(
        recall_at_10=recall_at_10,
        recall_at_20=recall_at_20,
        latency_p50_ms=0.1,
        latency_p95_ms=1.5,
        build_time_ms=4.0,
        disk_bytes=99,
    )


def _stats(*, unindexed: int = 0) -> IndexStats:
    return IndexStats(index_name="dense_hnsw", indexed_rows=20, unindexed_dense_rows=unindexed)


def test_domain_configurations_are_immutable_and_json_safe() -> None:
    config = VectorIndexConfig(
        index_type="hnsw_flat",
        metric="cosine",
        num_partitions=2,
        m=16,
        ef_construction=300,
        dense_chunks_count=20,
    )

    assert config.to_json() == {
        "index_type": "hnsw_flat",
        "metric": "cosine",
        "num_partitions": 2,
        "m": 16,
        "ef_construction": 300,
        "dense_chunks_count": 20,
        "index_name": "dense_hnsw",
    }
    assert FtsIndexConfig().to_json()["base_tokenizer"] == "whitespace"
    with pytest.raises(AttributeError):
        config.metric = "l2"  # type: ignore[misc]


def test_auto_promotes_only_complete_recall_and_coverage() -> None:
    decision = select_vector_policy(_benchmark(), _stats())

    assert decision.selected_mode == "ann"
    assert decision.reason == "candidate meets recall and dense-index coverage requirements"
    assert decision.benchmark == _benchmark()
    assert decision.index_stats == _stats()


@pytest.mark.parametrize(
    ("benchmark", "stats", "reason"),
    [
        (_benchmark(recall_at_10=0.99), _stats(), "recall@10 was 0.99, requires 1.00"),
        (_benchmark(recall_at_20=0.95), _stats(), "recall@20 was 0.95, requires 1.00"),
        (_benchmark(), _stats(unindexed=1), "1 dense row remains unindexed"),
    ],
)
def test_auto_fails_closed_for_valid_candidate_misses(
    benchmark: BenchmarkObservation, stats: IndexStats, reason: str
) -> None:
    # Phase 06（issue #49）：exact 不再是可发布结果——未达标必须 fail-closed。
    with pytest.raises(PolicyError, match=reason):
        select_vector_policy(benchmark, stats)


@pytest.mark.parametrize(
    "benchmark, stats",
    [
        (None, _stats()),
        (_benchmark(), None),
        (_benchmark(recall_at_10=float("nan")), _stats()),
        (_benchmark(), IndexStats(index_name="dense_hnsw", indexed_rows=20, unindexed_dense_rows=-1)),
    ],
)
def test_auto_rejects_missing_or_malformed_observations(
    benchmark: BenchmarkObservation | None, stats: IndexStats | None
) -> None:
    with pytest.raises(PolicyError):
        select_vector_policy(benchmark, stats)  # type: ignore[arg-type]


def test_vector_config_rejects_non_dense_population_inputs() -> None:
    with pytest.raises(ValueError):
        VectorIndexConfig(
            index_type="hnsw_flat", metric="cosine", num_partitions=0,
            m=16, ef_construction=300, dense_chunks_count=20,
        )


def test_service_records_publication_evidence_and_fixed_ann_policy(tmp_path: Path) -> None:
    """Phase 06：真实构建发布固定 SQ 策略 + held-out 发布证据。"""
    import math
    import random as _random

    wiki = tmp_path / "Wiki"
    for index in range(20):
        page = wiki / "concepts" / f"page-{index:02d}.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(f"# Page {index}\n\nUNIQUE{index:02d}\n", encoding="utf-8")

    def embed(texts):
        out = []
        for row, _text in enumerate(texts):
            rng = _random.Random(4242 + row)
            raw = [rng.gauss(0.0, 1.0) for _ in range(384)]
            norm = math.sqrt(sum(value * value for value in raw))
            out.append([value / norm for value in raw])
        return out

    index_dir = tmp_path / ".index"
    artifact = IndexBuildService(
        LanceDbIndexRepository(index_dir),
        reopen_storage=LanceDbIndexRepository,
        manifest_store=FilesystemIndexManifest(),
        post_commit_journal=FilesystemPostCommitJournal(index_dir),
    ).build(wiki, index_dir, embed=embed)

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    benchmark = manifest["benchmark"]
    assert benchmark["recall_at_10"] >= 0.19
    assert benchmark["recall_at_20"] >= 0.17
    evidence = manifest["candidate_publication_evidence"]
    assert evidence["actual_dense_rows"] == 20
    assert evidence["validation_query_count"] == min(256, 20)
    assert len(evidence["exact_result_ids"]) == 20
    assert evidence["query_source"] == "deterministic_disjoint_unit_v1"
    assert evidence["corpus_query_overlap"] == 0
    assert manifest["vector_config"]["index_type"] == "hnsw_sq"
    assert manifest["vector_config"]["num_partitions"] == 1
    assert manifest["ann_policy"]["query_ef"] == 100
    assert manifest["policy"]["selected_mode"] == "ann"


def test_service_probe_cap_changes_only_validation_query_count(tmp_path: Path) -> None:
    """BENCHMARK_MAX_PROBES 只改验证 query 数，不改类型/ef/成功路径。"""
    import math
    import random as _random

    def embed(texts):
        out = []
        for row, _text in enumerate(texts):
            rng = _random.Random(9100 + row)
            raw = [rng.gauss(0.0, 1.0) for _ in range(384)]
            norm = math.sqrt(sum(value * value for value in raw))
            out.append([value / norm for value in raw])
        return out

    results = {}
    for probes in (256, 37):
        wiki = tmp_path / f"Wiki{probes}"
        for index in range(8):
            page = wiki / "concepts" / f"page-{index:02d}.md"
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(f"# Page {index}\n\nCAP{probes}UNIQUE{index:02d}\n", encoding="utf-8")
        index_dir = tmp_path / f".index{probes}"
        IndexBuildService(
            LanceDbIndexRepository(index_dir),
            reopen_storage=LanceDbIndexRepository,
            manifest_store=FilesystemIndexManifest(),
            post_commit_journal=FilesystemPostCommitJournal(index_dir),
            benchmark_max_probes=probes,
        ).build(wiki, index_dir, embed=embed)
        manifest = json.loads(
            next((index_dir / "builds").glob("build_*/manifest.json")).read_text(encoding="utf-8")
        )
        results[probes] = manifest

    for probes, manifest in results.items():
        evidence = manifest["candidate_publication_evidence"]
        assert evidence["benchmark_max_probes"] == probes
        assert evidence["validation_query_count"] == min(probes, 8)
    assert results[256]["vector_config"]["index_type"] == results[37]["vector_config"]["index_type"] == "hnsw_sq"
    assert results[256]["ann_policy"]["query_ef"] == results[37]["ann_policy"]["query_ef"] == 100


# ---- Phase 06（issue #49）：固定生产 ANN 契约 ---------------------------------


def _approved_policy() -> ProductionAnnPolicy:
    return load_ann_policy_record(
        json.loads((_REPO_ROOT / "eval" / "ann-policy.json").read_text(encoding="utf-8"))
    )


def _decision_evidence(policy: ProductionAnnPolicy) -> AnnDecisionEvidence:
    return AnnDecisionEvidence(
        evidence_schema_version=ANN_DECISION_EVIDENCE_SCHEMA_VERSION,
        corpus_rows=77348,
        dimensions=policy.dimensions,
        held_out_queries=256,
        candidates=("ivf-hnsw-flat", "ivf-hnsw-sq"),
        ef_grid=(30, 50, 75, 100, 150, 200),
        approved_index_type=policy.selected_index_type,
        approved_query_ef=policy.query_ef,
        approved_recall_at_10_floor=policy.recall_at_10_floor,
        approved_recall_at_20_floor=policy.recall_at_20_floor,
        comparator_sha256=policy.comparator_sha256,
        candidate_hybrid_sha256=policy.candidate_hybrid_sha256,
        reconciliation_sha256=policy.reconciliation_sha256,
        evidence_run_url=policy.evidence_run_url,
        approved_by="root/user (Derek)",
        approved_at="2026-08-17T06:40:12Z",
    )


def _publication_evidence(
    policy: ProductionAnnPolicy,
    *,
    rows: int = 513,
    probes: int = 256,
    top10_hits: int | None = None,
    top20_hits: int | None = None,
    index_type: str | None = None,
    query_ef: int | None = None,
    policy_sha256: str | None = None,
    decision_sha256: str | None = None,
    validation_query_count: int | None = None,
    corpus_query_overlap: int = 0,
    recall_at_10: float | None = None,
    recall_at_20: float | None = None,
    unindexed: int = 0,
    exact_verification_ms: float = 1.0,
) -> CandidatePublicationEvidence:
    count = min(probes, rows)
    card = min(20, rows)
    if top10_hits is None:
        top10_hits = card if card < 10 else 10
    if top20_hits is None:
        top20_hits = top10_hits
    exact_ids, candidate_ids = [], []
    for query in range(count):
        truth = tuple(f"e{query:04d}::{i:02d}" for i in range(card))
        observed = list(truth[:top10_hits])
        observed.extend(f"c{query:04d}::{i:02d}" for i in range(card - len(observed)))
        # 控制第 11..20 名的命中数（top20_hits >= top10_hits）
        extra_hits = max(0, top20_hits - top10_hits)
        tail = list(truth[top10_hits:top10_hits + extra_hits])
        tail.extend(
            f"c{query:04d}::t{i:02d}"
            for i in range(card - top10_hits - extra_hits)
        )
        observed = observed[:top10_hits] + tail
        exact_ids.append(truth)
        candidate_ids.append(tuple(observed[:card]))

    def aggregate(limit: int) -> float:
        hits = total = 0
        for truth, observed in zip(exact_ids, candidate_ids):
            truth_prefix = set(truth[:limit])
            hits += len(truth_prefix & set(observed[:limit]))
            total += len(truth_prefix)
        return hits / total

    return CandidatePublicationEvidence(
        evidence_schema_version=CANDIDATE_PUBLICATION_EVIDENCE_SCHEMA_VERSION,
        actual_dense_rows=rows,
        dimensions=policy.dimensions,
        metric=policy.metric,
        index_type=index_type if index_type is not None else policy.selected_index_type,
        query_ef=query_ef if query_ef is not None else policy.query_ef,
        policy_sha256=policy_sha256 if policy_sha256 is not None else production_policy_sha256(policy),
        decision_evidence_sha256=decision_sha256 if decision_sha256 is not None else policy.comparator_sha256,
        benchmark_max_probes=probes,
        validation_query_count=(
            validation_query_count if validation_query_count is not None else count
        ),
        query_source="deterministic_disjoint_unit_v1",
        query_selection_sha256="a" * 64,
        corpus_query_overlap=corpus_query_overlap,
        exact_result_ids=tuple(exact_ids),
        candidate_result_ids=tuple(candidate_ids),
        recall_at_10=recall_at_10 if recall_at_10 is not None else aggregate(10),
        recall_at_20=recall_at_20 if recall_at_20 is not None else aggregate(20),
        unindexed_dense_rows=unindexed,
        exact_verification_ms=exact_verification_ms,
        ann_verification_ms=2.0,
        benchmark_duration_ms=3.0,
    )


def test_tracked_policy_record_loads_and_binds_the_approved_decision() -> None:
    policy = _approved_policy()

    assert policy.policy_schema_version == 2
    assert policy.selected_index_type == "ivf-hnsw-sq"
    assert policy.lancedb_index_type == "hnsw_sq"
    assert policy.metric == "cosine"
    assert policy.dimensions == 384
    assert policy.num_partitions == 1
    assert policy.query_ef == 100
    assert policy.recall_at_10_floor == 0.19
    assert policy.recall_at_20_floor == 0.17
    assert policy.retention_days == 90
    for digest in (
        policy.comparator_sha256, policy.candidate_hybrid_sha256,
        policy.reconciliation_sha256,
    ):
        assert len(digest) == 64

    authorized = validate_ann_decision_evidence(_decision_evidence(policy), policy)
    assert authorized is policy
    assert authorized.to_json() == policy.to_json()  # stable JSON


def test_decision_evidence_rejects_unlocked_scale_or_binding_mismatch() -> None:
    policy = _approved_policy()

    def mutated(**overrides):
        payload = _decision_evidence(policy).to_json()
        payload["candidates"] = list(payload["candidates"])
        payload["ef_grid"] = list(payload["ef_grid"])
        payload.update(overrides)
        return AnnDecisionEvidence(**payload)

    cases = [
        mutated(corpus_rows=77347),
        mutated(dimensions=383),
        mutated(held_out_queries=255),
        mutated(candidates=["ivf-hnsw-sq"]),
        mutated(ef_grid=[30, 50, 75, 100, 150]),
        mutated(comparator_sha256="f" * 64),
        mutated(approved_index_type="ivf-hnsw-flat"),
        mutated(approved_query_ef=150),
        mutated(approved_recall_at_10_floor=0.5),
        mutated(approved_by=""),
    ]
    for evidence in cases:
        with pytest.raises(PolicyError):
            validate_ann_decision_evidence(evidence, policy)


def test_publication_evidence_accepts_non_decision_corpus_and_probe_caps() -> None:
    policy = _approved_policy()

    for probes in (256, 37):
        evidence = _publication_evidence(policy, rows=513, probes=probes)
        assert evidence.validation_query_count == min(probes, 513)
        authorized = validate_candidate_publication_evidence(evidence, policy)
        # 返回的是固定批准策略——probe cap 不能选择或修改生产值。
        assert authorized is policy
        assert authorized.selected_index_type == "ivf-hnsw-sq"
        assert authorized.query_ef == 100


def test_publication_evidence_rejects_every_named_mutation() -> None:
    policy = _approved_policy()
    cases = [
        _publication_evidence(policy, index_type="ivf-hnsw-flat"),
        _publication_evidence(policy, query_ef=30),
        _publication_evidence(policy, policy_sha256="0" * 64),
        _publication_evidence(policy, decision_sha256="0" * 64),
        _publication_evidence(policy, validation_query_count=36, probes=37),
        _publication_evidence(policy, corpus_query_overlap=1),
        _publication_evidence(policy, unindexed=1),
        _publication_evidence(policy, top10_hits=0),          # recall@10 = 0 < 0.19
        _publication_evidence(policy, top10_hits=2),          # @10=0.2 ok, @20=0.1 < 0.17
        _publication_evidence(policy, recall_at_10=0.5),      # declared != IDs
        _publication_evidence(policy, exact_verification_ms=float("nan")),
        _publication_evidence(policy, exact_verification_ms=-1.0),
    ]
    for evidence in cases:
        with pytest.raises(PolicyError):
            validate_candidate_publication_evidence(evidence, policy)


def test_publication_evidence_requires_complete_top_k_id_rows() -> None:
    policy = _approved_policy()
    evidence = _publication_evidence(policy, rows=513, probes=256)
    truncated = evidence.to_json()
    truncated["exact_result_ids"] = truncated["exact_result_ids"][:-1]
    truncated["candidate_result_ids"] = truncated["candidate_result_ids"][:-1]
    with pytest.raises(PolicyError):
        validate_candidate_publication_evidence(
            CandidatePublicationEvidence(**truncated), policy
        )

    # exact 行必须满额：flat 扫描保证恰好 min(20, rows) 个 ID。
    short_exact = evidence.to_json()
    short_exact["exact_result_ids"] = [row[:-1] for row in short_exact["exact_result_ids"]]
    with pytest.raises(PolicyError):
        validate_candidate_publication_evidence(
            CandidatePublicationEvidence(**short_exact), policy
        )

    # ANN 行允许不足额（Linux LanceDB 实测会返回少于 limit 的行），
    # 但空行 = 查询失败，必须拒绝。截掉的尾行元素非命中项，recall 声明保持一致。
    short_candidate = evidence.to_json()
    short_candidate["candidate_result_ids"] = [
        row[:-1] for row in short_candidate["candidate_result_ids"]
    ]
    validate_candidate_publication_evidence(
        CandidatePublicationEvidence(**short_candidate), policy
    )
    empty_candidate = evidence.to_json()
    empty_candidate["candidate_result_ids"] = [
        [] for _ in empty_candidate["candidate_result_ids"]
    ]
    with pytest.raises(PolicyError):
        validate_candidate_publication_evidence(
            CandidatePublicationEvidence(**empty_candidate), policy
        )


def test_evidence_records_are_not_interchangeable() -> None:
    policy = _approved_policy()
    with pytest.raises(PolicyError):
        validate_ann_decision_evidence(_publication_evidence(policy), policy)
    with pytest.raises(PolicyError):
        validate_candidate_publication_evidence(_decision_evidence(policy), policy)
