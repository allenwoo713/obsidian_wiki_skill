"""Persisted D-01/D-04 contract tests for the first storage tracer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lancedb
import pytest


sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_index import build_storage_contract  # noqa: E402
from obsidian_wiki.domain.index_models import RebuildRequiredError  # noqa: E402
from obsidian_wiki.domain.index_models import (  # noqa: E402
    BenchmarkObservation,
    DenseChunk,
    VectorIndexConfig,
)
from obsidian_wiki.application.index_build_service import IndexBuildService  # noqa: E402
from obsidian_wiki.infrastructure import lancedb_index_repository as repository_module  # noqa: E402
from obsidian_wiki.infrastructure.lancedb_index_repository import (  # noqa: E402
    LanceDbIndexRepository,
)


def _write_page(wiki: Path, body: str) -> None:
    page = wiki / "concepts" / "storage.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        "---\n"
        "type: concept\n"
        "title: Storage contract\n"
        "sources: []\n"
        "tags: []\n"
        "related: []\n"
        "---\n\n"
        + body,
        encoding="utf-8",
    )


def test_wrapper_builds_two_physical_tables_and_explicit_fts(tmp_path: Path) -> None:
    """The direct script wrapper crosses service/port/adapter into LanceDB."""
    long_term = "D01ExactTerm" + "abc123" * 29
    wiki = tmp_path / "Wiki"
    _write_page(wiki, f"# Contract\n\nThe exact storage token is\n{long_term}\n")

    artifact = build_storage_contract(
        wiki,
        tmp_path / ".index",
        embed=lambda texts: [[float(len(text)), 1.0] for text in texts],
    )

    db = lancedb.connect(str(artifact.lance_dir))
    assert set(db.table_names()) == {"sparse_chunks", "dense_chunks"}
    sparse = db.open_table("sparse_chunks")
    dense = db.open_table("dense_chunks")
    assert "vector" not in sparse.schema.names
    assert "vector" in dense.schema.names
    assert sparse.count_rows() > 0
    assert dense.count_rows() > 0
    assert "fts_text_idx" in {index.name for index in sparse.list_indices()}

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["layout"] == "sparse_chunks+dense_chunks"
    assert manifest["fts_config"] == {
        "column": "fts_text",
        "base_tokenizer": "whitespace",
        "lower_case": False,
        "stem": False,
        "remove_stop_words": False,
        "ascii_folding": False,
        "max_token_length": 256,
    }
    assert LanceDbIndexRepository(artifact.lance_dir).search_sparse(long_term)


def test_legacy_manifest_requires_rebuild(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"layout": "chunks"}), encoding="utf-8")

    with pytest.raises(RebuildRequiredError, match="rebuild"):
        LanceDbIndexRepository.require_current_layout(manifest)


class _QuerySpy:
    def __init__(self) -> None:
        self.exact_bypass = False
        self.metric = None
        self.predicate = None
        self.result_limit = None

    def distance_type(self, metric: str) -> "_QuerySpy":
        self.metric = metric
        return self

    def bypass_vector_index(self) -> "_QuerySpy":
        self.exact_bypass = True
        return self

    def where(self, predicate: str) -> "_QuerySpy":
        self.predicate = predicate
        return self

    def limit(self, value: int) -> "_QuerySpy":
        self.result_limit = value
        return self

    def to_list(self) -> list[dict[str, str]]:
        return [{"chunk_id": "dense:1"}]


class _DenseTableSpy:
    def __init__(self) -> None:
        self.index_call = None
        self.queries: list[_QuerySpy] = []

    def create_index(self, column: str, **kwargs: object) -> None:
        self.index_call = (column, kwargs)

    def search(self, vector: list[float]) -> _QuerySpy:
        query = _QuerySpy()
        self.queries.append(query)
        return query

    def count_rows(self) -> int:
        return 20

    def index_stats(self, index_name: str):
        return type("Stats", (), {"num_indexed_rows": 20, "num_unindexed_rows": 0})()


class _DatabaseSpy:
    def __init__(self, dense_table: _DenseTableSpy) -> None:
        self.dense_table = dense_table

    def open_table(self, name: str) -> _DenseTableSpy:
        assert name == "dense_chunks"
        return self.dense_table


def test_candidate_hnsw_and_exact_bypass_stay_in_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    table = _DenseTableSpy()
    monkeypatch.setattr(repository_module.lancedb, "connect", lambda _: _DatabaseSpy(table))
    repository = LanceDbIndexRepository(tmp_path / "lance")
    config = VectorIndexConfig(
        index_type="hnsw_flat", metric="cosine", num_partitions=2,
        m=16, ef_construction=300, dense_chunks_count=20,
    )

    stats = repository.create_vector_index(config)
    ann = repository.search_dense([1.0, 0.0], metric="cosine", limit=20, where="page_id = 'safe'")
    exact = repository.search_dense_exact([1.0, 0.0], metric="cosine", limit=20, where="page_id = 'safe'")

    column, kwargs = table.index_call
    hnsw = kwargs["config"]
    assert column == "vector"
    assert kwargs["name"] == "dense_hnsw"
    assert hnsw.distance_type == "cosine"
    assert hnsw.num_partitions == 2
    assert hnsw.m == 16
    assert hnsw.ef_construction == 300
    assert stats.unindexed_dense_rows == 0
    assert ann == exact == [{"chunk_id": "dense:1"}]
    assert table.queries[0].exact_bypass is False
    assert table.queries[1].exact_bypass is True
    assert table.queries[0].metric == table.queries[1].metric == "cosine"
    assert table.queries[0].predicate == table.queries[1].predicate == "page_id = 'safe'"
    assert table.queries[0].result_limit == table.queries[1].result_limit == 20


def test_adapter_rejects_duplicate_or_nonfinite_dense_vectors(tmp_path: Path) -> None:
    row = DenseChunk("dense:1", "page", "page.md", "Page", "text", (1.0, 0.0))
    repository = LanceDbIndexRepository(tmp_path / "lance")

    with pytest.raises(ValueError, match="duplicate"):
        repository.validate_dense_chunks((row, row))
    with pytest.raises(ValueError, match="finite"):
        repository.validate_dense_chunks((
            DenseChunk("dense:2", "page", "page.md", "Page", "text", (float("nan"), 0.0)),
        ))


def test_adapter_escapes_quoted_page_identifiers() -> None:
    assert LanceDbIndexRepository.page_predicate("O'Reilly") == "page_id = 'O''Reilly'"


def test_reopened_artifact_has_validation_evidence_and_publishes(tmp_path: Path) -> None:
    """D-01: only a newly reopened, fully validated two-table build can publish."""
    wiki = tmp_path / "Wiki"
    _write_page(wiki, "# Contract\n\nThe standalone exact token is VALIDATIONTERM\n")
    index_dir = tmp_path / ".index"

    artifact = build_storage_contract(
        wiki, index_dir, embed=lambda texts: [[1.0, float(index + 1)] for index, _ in enumerate(texts)]
    )

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert (index_dir / "ACTIVE_INDEX").exists()
    assert manifest["format_version"] >= 4
    assert manifest["validation"]["schema_counts"] == {"sparse_chunks_count": 1, "dense_chunks_count": 1}
    assert manifest["validation"]["exact_term_validated"] is True
    assert manifest["config_hashes"]["fts_config"]
    assert manifest["sdk_versions"]["lancedb"]
    assert manifest["policy"]["selected_mode"] in {"ann", "exact"}


def test_complete_non_promoting_candidate_publishes_exact_policy(tmp_path: Path) -> None:
    wiki = tmp_path / "Wiki"
    _write_page(wiki, "# Contract\n\nThe standalone exact token is FALLBACKTERM\n")
    index_dir = tmp_path / ".index"
    service = IndexBuildService(
        LanceDbIndexRepository(index_dir),
        benchmark_observer=lambda _stats: BenchmarkObservation(
            recall_at_10=0.9, recall_at_20=1.0, latency_p50_ms=1.0,
            latency_p95_ms=2.0, build_time_ms=3.0, disk_bytes=4,
        ),
    )

    artifact = service.build(
        wiki, index_dir, embed=lambda texts: [[1.0, float(index + 1)] for index, _ in enumerate(texts)]
    )

    assert (index_dir / "ACTIVE_INDEX").exists()
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["policy"]["selected_mode"] == "exact"
    assert manifest["policy"]["reason"].startswith("recall@10")
