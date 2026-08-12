"""Pure D-02/D-03 ANN-promotion contract tests."""
from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from obsidian_wiki.domain.index_models import (  # noqa: E402
    BenchmarkObservation,
    FtsIndexConfig,
    IndexStats,
    VectorIndexConfig,
)
from obsidian_wiki.domain.index_policy import PolicyError, select_vector_policy  # noqa: E402
from obsidian_wiki.application.index_build_service import IndexBuildService  # noqa: E402
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository  # noqa: E402
from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest  # noqa: E402
from obsidian_wiki.infrastructure.filesystem_post_commit_journal import (  # noqa: E402
    FilesystemPostCommitJournal,
)


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
def test_auto_falls_back_to_exact_for_valid_candidate_misses(
    benchmark: BenchmarkObservation, stats: IndexStats, reason: str
) -> None:
    decision = select_vector_policy(benchmark, stats)

    assert decision.selected_mode == "exact"
    assert decision.reason == reason


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


def test_service_records_exact_and_candidate_id_sets_before_ann_promotion(tmp_path: Path) -> None:
    """D-02: promotion evidence is persisted, not inferred from timing."""
    wiki = tmp_path / "Wiki"
    for index in range(20):
        page = wiki / "concepts" / f"page-{index:02d}.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(f"# Page {index}\n\nUNIQUE{index:02d}\n", encoding="utf-8")

    index_dir = tmp_path / ".index"
    artifact = IndexBuildService(
        LanceDbIndexRepository(index_dir),
        reopen_storage=LanceDbIndexRepository,
        manifest_store=FilesystemIndexManifest(),
        post_commit_journal=FilesystemPostCommitJournal(index_dir),
    ).build(
        wiki,
        index_dir,
        # Tie-free by construction: a shared strictly-decreasing tail makes every
        # non-self cosine distinct, so exact top-k has ONE valid ordering. Plain
        # one-hot vectors leave a 19-way tie at score 0.0, and the two engines
        # (numpy batch exact vs lancedb HNSW) break that tie differently, which
        # made recall@10 platform-dependent noise rather than a real signal.
        embed=lambda texts: [
            [1.0 if column == row else 1.0 / (2 ** (column + 2)) for column in range(20)]
            for row, _text in enumerate(texts)
        ],
    )

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    benchmark = manifest["benchmark"]
    assert benchmark["recall_at_10"] == 1.0
    assert benchmark["recall_at_20"] == 1.0
    assert benchmark["exact_result_ids"]
    assert benchmark["candidate_result_ids"]
    assert manifest["vector_config"]["index_type"] == "ivf_flat"
    assert manifest["vector_config"]["num_partitions"] == 1
    assert manifest["policy"]["selected_mode"] == "ann"
