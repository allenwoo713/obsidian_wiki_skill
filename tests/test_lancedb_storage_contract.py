"""Persisted D-01/D-04 contract tests for the first storage tracer."""
from __future__ import annotations

import inspect
import json
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lancedb
import pytest


sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_index import WikiIndex, build_storage_contract  # noqa: E402
import build_index as build_index_module  # noqa: E402
from obsidian_wiki.domain.index_models import (  # noqa: E402
    FtsIndexConfig,
    RebuildRequiredError,
    SparseChunk,
)
from obsidian_wiki.domain.index_models import (  # noqa: E402
    BenchmarkObservation,
    DenseChunk,
    VectorIndexConfig,
)
from obsidian_wiki.application.index_build_service import IndexBuildService  # noqa: E402
from obsidian_wiki.application.active_index_pointer import resolve_active_lance_dir  # noqa: E402
from obsidian_wiki.infrastructure import lancedb_index_repository as repository_module  # noqa: E402
from obsidian_wiki.infrastructure.lancedb_index_repository import (  # noqa: E402
    LanceDbIndexRepository,
)
from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest  # noqa: E402
from obsidian_wiki.infrastructure.filesystem_post_commit_journal import (  # noqa: E402
    FilesystemPostCommitJournal,
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


def _write_pages(wiki: Path, count: int, *, body_prefix: str = "PHASE06PAGE") -> None:
    for index in range(count):
        page = wiki / "concepts" / f"page-{index:04d}.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(
            f"# Page {index}\n\n{body_prefix}{index:04d} unique token body.\n",
            encoding="utf-8",
        )


def _embed384(seed: int = 0):
    """Deterministic pseudo-random unit vectors at the approved 384 dimensions."""
    import math
    import random as _random

    def embed(texts):
        out = []
        for row, _text in enumerate(texts):
            rng = _random.Random(9173 + seed * 100003 + row)
            raw = [rng.gauss(0.0, 1.0) for _ in range(384)]
            norm = math.sqrt(sum(value * value for value in raw))
            out.append([value / norm for value in raw])
        return out

    return embed


def _phase07_test_corpus_identity(wiki: Path) -> dict[str, object]:
    """The only small-corpus seam is an explicit pytest-only identity."""
    from eval.run_eval import expected_phase07_expanded_corpus_identity

    return expected_phase07_expanded_corpus_identity(
        fixture_root=wiki,
        target_size=sum(1 for path in wiki.rglob("*.md") if path.is_file()),
        test_only=True,
    )


class _FacadeEmbedder:
    """Small deterministic embedder that still exercises WikiIndex's public build path."""

    def __init__(self) -> None:
        self._encode = _embed384()
        self.tokenizer = lambda text, **_kwargs: {"input_ids": list(range(max(1, len(text) // 4)))}

    def get_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts, **_kwargs):
        return self._encode(texts)


def test_wrapper_builds_two_physical_tables_and_explicit_fts(tmp_path: Path) -> None:
    """The direct script wrapper crosses service/port/adapter into LanceDB."""
    long_term = "D01ExactTerm" + "abc123" * 29
    wiki = tmp_path / "Wiki"
    _write_page(wiki, f"# Contract\n\nThe exact storage token is\n{long_term}\n")

    artifact = build_storage_contract(
        wiki,
        tmp_path / ".index",
        embed=_embed384(),
    )

    db = lancedb.connect(str(artifact.artifact.lance_dir))
    assert set(db.table_names()) == {"sparse_chunks", "dense_chunks"}
    sparse = db.open_table("sparse_chunks")
    dense = db.open_table("dense_chunks")
    assert "vector" not in sparse.schema.names
    assert "vector" in dense.schema.names
    assert sparse.count_rows() > 0
    assert dense.count_rows() > 0
    assert "fts_text_idx" in {index.name for index in sparse.list_indices()}

    manifest = json.loads(artifact.artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["layout"] == "sparse_chunks+dense_chunks"
    # Phase 06：manifest 绑定批准策略（不再有 requested_vector_index_mode）。
    assert manifest["format_version"] == 6
    assert manifest["ann_policy"]["selected_index_type"] == "ivf-hnsw-sq"
    assert manifest["ann_policy"]["query_ef"] == 100
    assert "requested_vector_index_mode" not in manifest
    assert manifest["fts_config"] == {
        "column": "fts_text",
        "base_tokenizer": "whitespace",
        "lower_case": False,
        "stem": False,
        "remove_stop_words": False,
        "ascii_folding": False,
        "max_token_length": 256,
    }
    assert LanceDbIndexRepository(artifact.artifact.lance_dir).search_sparse(long_term)


def test_runtime_mode_selection_is_removed_from_public_surfaces() -> None:
    """Phase 06（issue #49）：facade/CLI 不暴露 auto/exact/FLAT/SQ 运行时选择。"""
    facade_params = inspect.signature(build_storage_contract).parameters
    build_params = inspect.signature(WikiIndex.build).parameters
    assert "vector_index_mode" not in facade_params
    assert "vector_index_mode" not in build_params
    # eval candidate 绑定仍显式存在（仅 eval comparator 使用）。
    assert "candidate_query_policy" in facade_params
    assert "candidate_query_policy" in build_params


def test_normal_production_build_constructs_only_the_approved_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """唯一可构建的 production candidate 是批准的 hnsw_sq。"""
    wiki = tmp_path / "Wiki"
    _write_page(wiki, "# Forced HNSW\n\nEXPLICITHNSWTERM\n")
    observed_configs: list[VectorIndexConfig] = []
    real_create = LanceDbIndexRepository.create_vector_index

    def capture_create(self, config):
        observed_configs.append(config)
        return real_create(self, config)

    monkeypatch.setattr(LanceDbIndexRepository, "create_vector_index", capture_create)
    outcome = build_storage_contract(
        wiki,
        tmp_path / ".index",
        embed=_embed384(),
    )
    manifest = json.loads(outcome.artifact.manifest_path.read_text(encoding="utf-8"))

    assert [config.index_type for config in observed_configs] == ["hnsw_sq"]
    assert manifest["vector_config"]["index_type"] == "hnsw_sq"
    assert manifest["policy"]["selected_mode"] == "ann"
    assert manifest["candidate_publication_evidence"]["index_type"] == "ivf-hnsw-sq"


def test_real_lancedb_has_no_storage_mutation_after_final_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#36：真实 LanceDB 的 persist + vector-index 写入必须都早于最终 seal。"""
    wiki = tmp_path / "Wiki"
    index_dir = tmp_path / ".index"
    _write_page(wiki, "# Contract\n\nThe standalone exact token is SEALORDERTERM\n")
    events: list[str] = []
    real_persist = LanceDbIndexRepository.persist
    real_create = LanceDbIndexRepository.create_vector_index
    real_seal = LanceDbIndexRepository.seal

    def _persist(self, *args, **kwargs):
        events.append("persist")
        return real_persist(self, *args, **kwargs)

    def _create(self, *args, **kwargs):
        events.append("create_vector_index")
        return real_create(self, *args, **kwargs)

    def _seal(self, *args, **kwargs):
        events.append("seal")
        return real_seal(self, *args, **kwargs)

    monkeypatch.setattr(LanceDbIndexRepository, "persist", _persist)
    monkeypatch.setattr(LanceDbIndexRepository, "create_vector_index", _create)
    monkeypatch.setattr(LanceDbIndexRepository, "seal", _seal)

    build_storage_contract(
        wiki,
        index_dir,
        embed=_embed384(),
    )

    assert "seal" in events
    final_seal = max(index for index, event in enumerate(events) if event == "seal")
    final_mutation = max(
        index for index, event in enumerate(events)
        if event in {"persist", "create_vector_index"}
    )
    assert final_mutation < final_seal, f"storage 在最终 seal 后仍被修改: {events}"


def test_wikiindex_post_publication_marker_failure_preserves_published_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final timing write can reject the caller without relabelling a published build failed."""
    wiki = tmp_path / "Wiki"
    index_dir = tmp_path / ".index"
    _write_page(wiki, "# Timing\n\nPOSTPUBLICATIONMARKERTERM\n")
    index = WikiIndex(index_dir)
    index._embedder = _FacadeEmbedder()
    import build_index as build_index_module

    real_storage_contract = build_index_module.build_storage_contract
    forwarded_sinks: list[object] = []

    def capture_storage_contract(*args, **kwargs):
        forwarded_sinks.append(kwargs.get("progress_sink"))
        return real_storage_contract(*args, **kwargs)

    monkeypatch.setattr(build_index_module, "build_storage_contract", capture_storage_contract)
    stages: list[str] = []

    def fail_only_after_publication(stage: str) -> None:
        stages.append(stage)
        if stage == "validation_seal_publication":
            raise BrokenPipeError("timing marker transport closed")

    with pytest.raises(BrokenPipeError, match="timing marker transport closed"):
        index.build(wiki, full_rebuild=True, progress_sink=fail_only_after_publication)

    assert forwarded_sinks == [fail_only_after_publication]
    assert stages == [
        "scan_chunk", "dense_embedding", "lance_fts_persist",
        "hnsw_create_index", "validation_seal_publication",
    ]
    active_lance = resolve_active_lance_dir(index_dir)
    published_build = active_lance.parent
    assert published_build.parent == index_dir / "builds"
    assert not (published_build / ".failed").exists()
    assert not list((index_dir / "builds").glob("*/.failed"))
    assert not (index_dir / "campaign-success.json").exists()
    assert not (index_dir / "authorization.json").exists()


def test_wikiindex_hnsw_failure_emits_no_later_timing_markers_or_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HNSW failures stay pre-publication and cannot emit a complete timing boundary."""
    wiki = tmp_path / "Wiki"
    index_dir = tmp_path / ".index"
    _write_page(wiki, "# Timing\n\nHNSWMUTATIONMARKERTERM\n")
    index = WikiIndex(index_dir)
    index._embedder = _FacadeEmbedder()
    stages: list[str] = []

    def fail_hnsw(*_args, **_kwargs):
        raise RuntimeError("forced HNSW mutation failure")

    monkeypatch.setattr(LanceDbIndexRepository, "create_vector_index", fail_hnsw)
    with pytest.raises(RuntimeError, match="forced HNSW mutation failure"):
        index.build(wiki, full_rebuild=True, progress_sink=stages.append)

    assert "lance_fts_persist" in stages
    assert "hnsw_create_index" not in stages
    assert "validation_seal_publication" not in stages
    assert not (index_dir / "ACTIVE_INDEX").exists()


def test_legacy_manifest_requires_rebuild(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"layout": "chunks"}), encoding="utf-8")

    with pytest.raises(RebuildRequiredError, match="rebuild"):
        LanceDbIndexRepository.require_current_layout(manifest)


def test_wikiindex_routes_sparse_and_dense_queries_to_split_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The compatibility facade must not reopen the retired ``chunks`` table."""
    calls: list[tuple[str, object]] = []

    class Repository:
        def search_sparse(self, query: str, *, limit: int):
            calls.append(("sparse", (query, limit)))
            return [{"chunk_id": "s", "page_id": "p", "path": "p.md", "title": "P", "text": "text"}]

        def search_dense(self, vector, *, metric: str, limit: int):
            calls.append(("dense", (list(vector), metric, limit)))
            return [{"chunk_id": "d", "page_id": "p", "path": "p.md", "title": "P", "text": "text", "_distance": 0.1}]

    class Embedder:
        def encode(self, *_args, **_kwargs):
            return [[1.0, 0.0]]

    index = WikiIndex(tmp_path / ".index")
    monkeypatch.setattr(index, "_get_repository", lambda: Repository())
    monkeypatch.setattr(index, "_get_embedder", lambda: Embedder())
    monkeypatch.setattr(index, "_get_lance_table", lambda *args, **kwargs: pytest.fail("legacy chunks opened"))

    assert index.search_fts("needle", k=2)[0].chunk_id == "s"
    assert index.search_vector("meaning", k=2)[0].chunk_id == "d"
    assert [name for name, _ in calls] == ["sparse", "dense"]


def test_native_fts_creation_failure_is_fatal_and_preserves_active_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wiki = tmp_path / "Wiki"
    _write_page(wiki, "# Contract\n\nThe exact token is FTSFAILTERM\n")
    index_dir = tmp_path / ".index"
    embed = _embed384()
    build_storage_contract(wiki, index_dir, embed=embed)
    original_pointer = (index_dir / "ACTIVE_INDEX").read_bytes()

    real_connect = repository_module.lancedb.connect

    class SparseTableWithBrokenFts:
        def __init__(self, table):
            self._table = table

        def create_fts_index(self, *_args, **_kwargs):
            raise RuntimeError("native FTS unavailable")

        def __getattr__(self, name):
            return getattr(self._table, name)

    class DatabaseWithBrokenFts:
        def __init__(self, database):
            self._database = database

        def create_table(self, name, *args, **kwargs):
            table = self._database.create_table(name, *args, **kwargs)
            return SparseTableWithBrokenFts(table) if name == "sparse_chunks" else table

        def __getattr__(self, name):
            return getattr(self._database, name)

    monkeypatch.setattr(
        repository_module.lancedb, "connect",
        lambda *args, **kwargs: DatabaseWithBrokenFts(real_connect(*args, **kwargs)),
    )
    with pytest.raises(RuntimeError, match="native FTS unavailable"):
        build_storage_contract(wiki, index_dir, embed=embed)

    assert (index_dir / "ACTIVE_INDEX").read_bytes() == original_pointer
    assert list((index_dir / "builds").glob("build_*/.failed"))


def test_context_reads_use_persisted_sparse_metadata_after_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Context expansion must never fall back to the retired chunks accessor."""
    wiki = tmp_path / "Wiki"
    _write_page(wiki, "# Install\n\nplaceholder page body\n")
    index_dir = tmp_path / ".index"
    records = [
        SparseChunk(
            "before", "page-a", "Wiki/page-a.md", "Page A", "before text", "before text",
            page_type="procedure", section_path='["Install"]', heading="Install",
            chunk_kind="dense", chunk_index=0, content_hash="before", end_char=11,
        ),
        SparseChunk(
            "anchor", "page-a", "Wiki/page-a.md", "Page A", "anchor text", "anchor text",
            page_type="procedure", section_path='["Install"]', heading="Install",
            chunk_kind="dense", chunk_index=1, content_hash="anchor", start_char=12, end_char=23,
        ),
        SparseChunk(
            "after", "page-a", "Wiki/page-a.md", "Page A", "after text", "after text",
            page_type="procedure", section_path='["Install"]', heading="Install",
            chunk_kind="dense", chunk_index=2, content_hash="after", start_char=24, end_char=34,
        ),
        SparseChunk(
            "section", "page-a", "Wiki/page-a.md", "Page A", "section text", "section text",
            page_type="procedure", section_path='["Install"]', heading="Install",
            chunk_kind="sparse", chunk_index=3, content_hash="section", start_char=0, end_char=34,
        ),
    ]
    build_storage_contract(
        wiki, index_dir, sparse_chunks=records,
        embed=_embed384(),
    )

    index = WikiIndex(index_dir)
    index.load()
    monkeypatch.setattr(index, "_get_lance_table", lambda *args, **kwargs: pytest.fail("legacy chunks opened"))

    anchor = index.get_chunk("anchor")
    assert anchor is not None
    assert (anchor.text, anchor.heading, anchor.chunk_index, anchor.page_type) == (
        "anchor text", "Install", 1, "procedure",
    )
    assert [hit.chunk_id for hit in index.get_neighbors("anchor")] == ["before", "after"]
    assert [hit.chunk_id for hit in index.get_parent_section("anchor")] == [
        "before", "anchor", "after", "section",
    ]


class _QuerySpy:
    def __init__(self) -> None:
        self.exact_bypass = False
        self.metric = None
        self.predicate = None
        self.result_limit = None
        self.ef_value = None

    def distance_type(self, metric: str) -> "_QuerySpy":
        self.metric = metric
        return self

    def bypass_vector_index(self) -> "_QuerySpy":
        self.exact_bypass = True
        return self

    def ef(self, value: int) -> "_QuerySpy":
        self.ef_value = value
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
    def __init__(self, row_count: int = 20) -> None:
        self.index_call = None
        self.queries: list[_QuerySpy] = []
        self.row_count = row_count

    def create_index(self, column: str, **kwargs: object) -> None:
        self.index_call = (column, kwargs)

    def search(self, vector: list[float]) -> _QuerySpy:
        query = _QuerySpy()
        self.queries.append(query)
        return query

    def count_rows(self) -> int:
        return self.row_count

    def index_stats(self, index_name: str):
        return type("Stats", (), {"num_indexed_rows": 20, "num_unindexed_rows": 0})()


class _DatabaseSpy:
    def __init__(self, dense_table: _DenseTableSpy) -> None:
        self.dense_table = dense_table

    def open_table(self, name: str) -> _DenseTableSpy:
        assert name == "dense_chunks"
        return self.dense_table


def test_bound_adapter_rejects_runtime_type_selection_and_fixes_ef(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Phase 06：生产绑定只建批准类型；普通查询固定批准 ef、绝不 bypass。"""
    table = _DenseTableSpy()
    monkeypatch.setattr(repository_module.lancedb, "connect", lambda _: _DatabaseSpy(table))
    repository = LanceDbIndexRepository(tmp_path / "lance")  # 未注入 → 默认批准策略
    unapproved = VectorIndexConfig(
        index_type="hnsw_flat", metric="cosine", num_partitions=2,
        m=16, ef_construction=300, dense_chunks_count=20,
    )
    with pytest.raises(ValueError, match="bound to index type"):
        repository.create_vector_index(unapproved)

    approved = VectorIndexConfig(
        index_type="hnsw_sq", metric="cosine", num_partitions=2,
        m=16, ef_construction=300, dense_chunks_count=20,
    )
    repository.create_vector_index(approved)

    ann = repository.search_dense([1.0, 0.0], metric="cosine", limit=20, where="page_id = 'safe'")
    exact = repository.search_dense_exact([1.0, 0.0], metric="cosine", limit=20, where="page_id = 'safe'")

    _, kwargs = table.index_call
    hnsw = kwargs["config"]
    assert type(hnsw).__name__ == "HnswSq"
    assert hnsw.distance_type == "cosine"
    assert hnsw.num_partitions == 2
    assert hnsw.m == 16
    assert hnsw.ef_construction == 300
    assert ann == exact == [{"chunk_id": "dense:1"}]
    assert table.queries[0].exact_bypass is False
    assert table.queries[1].exact_bypass is True
    assert table.queries[0].metric == table.queries[1].metric == "cosine"
    assert table.queries[0].predicate == table.queries[1].predicate == "page_id = 'safe'"
    assert table.queries[0].result_limit == table.queries[1].result_limit == 20
    # 固定批准 ef=100（与 limit 无关；不再有 max(100, 1.5*limit) 启发式）。
    assert table.queries[0].ef_value == 100
    assert table.queries[1].ef_value is None  # exact path bypasses ef


def test_eval_candidate_binding_builds_its_candidate_and_applies_grid_ef(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """显式 eval candidate 绑定：FLAT 可建、声明 ef 生效——仅 eval seam。"""
    from obsidian_wiki.domain.index_models import CandidateQueryPolicy

    table = _DenseTableSpy()
    monkeypatch.setattr(repository_module.lancedb, "connect", lambda _: _DatabaseSpy(table))
    repository = LanceDbIndexRepository(
        tmp_path / "lance",
        eval_candidate_policy=CandidateQueryPolicy(candidate="ivf-hnsw-flat", query_ef=30),
    )
    config = VectorIndexConfig(
        index_type="hnsw_flat", metric="cosine", num_partitions=1,
        m=16, ef_construction=300, dense_chunks_count=20,
    )
    repository.create_vector_index(config)
    repository.search_dense([1.0, 0.0], metric="cosine", limit=20)
    repository.search_dense_eval([1.0, 0.0], metric="cosine", limit=20, ef=157)

    _, kwargs = table.index_call
    assert type(kwargs["config"]).__name__ == "HnswFlat"
    assert kwargs["config"].num_partitions == 1
    assert table.queries[0].ef_value == 30   # 绑定的 candidate ef
    assert table.queries[1].ef_value == 157  # 显式 eval grid ef
    # eval 绑定也拒绝其它类型（SQ 配置进 FLAT 绑定仓库）。
    with pytest.raises(ValueError, match="bound to index type"):
        repository.create_vector_index(VectorIndexConfig(
            index_type="hnsw_sq", metric="cosine", num_partitions=1,
            m=16, ef_construction=300, dense_chunks_count=20,
        ))


def test_eval_seam_still_builds_ivf_flat_comparator_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """显式 named eval seam 可构建 comparator 声明的 IVF-FLAT（不经过生产端口）。"""
    table = _DenseTableSpy()
    monkeypatch.setattr(repository_module.lancedb, "connect", lambda _: _DatabaseSpy(table))
    repository = LanceDbIndexRepository(tmp_path / "lance")
    config = VectorIndexConfig(
        index_type="ivf_flat", metric="cosine", num_partitions=1,
        m=16, ef_construction=300, dense_chunks_count=20,
    )

    repository.create_eval_candidate_index(config)

    _, kwargs = table.index_call
    ivf = kwargs["config"]
    assert type(ivf).__name__ == "IvfFlat"
    assert ivf.distance_type == "cosine"
    assert ivf.num_partitions == 1


def test_candidate_query_policy_is_immutable_and_applies_only_at_build_and_dense_query_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    policy_type = getattr(build_index_module, "CandidateQueryPolicy")
    policy = policy_type(candidate="ivf-hnsw-sq", query_ef=75)
    with pytest.raises((AttributeError, TypeError)):
        policy.query_ef = 100

    wiki = tmp_path / "Wiki"
    _write_page(wiki, "# Candidate policy\n\nCANDIDATEPOLICYTERM\n")
    outcome = build_storage_contract(
        wiki, tmp_path / ".index",
        embed=_embed384(),
        candidate_query_policy=policy,
    )
    manifest = json.loads(outcome.artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidate_query_policy"] == {
        "candidate": "ivf-hnsw-sq", "query_ef": 75,
    }
    assert manifest["vector_config"]["index_type"] == "hnsw_sq"
    # eval candidate 构建不经过生产发布门禁，policy 记录 eval 语义。
    assert manifest["policy"]["selected_mode"] == "ann"
    assert "candidate_publication_evidence" not in manifest

    class Repository:
        def search_dense(self, vector, *, metric, limit):
            # Phase 06：facade 不再传 ef——eval ef 由仓库绑定策略决定。
            return [{"chunk_id": "d", "page_id": "p", "path": "p.md", "title": "P", "text": "text", "_distance": 0.1}]

    class Embedder:
        def encode(self, *_args, **_kwargs):
            return [[1.0, 0.0]]

    index = WikiIndex(tmp_path / ".query-index")
    index._candidate_query_policy = policy
    monkeypatch.setattr(index, "_get_repository", lambda: Repository())
    monkeypatch.setattr(index, "_get_embedder", lambda: Embedder())
    assert index.search_vector("meaning", k=2)[0].chunk_id == "d"
    assert "candidate_query_policy" in inspect.signature(WikiIndex.build).parameters


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
        wiki, index_dir, embed=_embed384()
    )

    manifest = json.loads(artifact.artifact.manifest_path.read_text(encoding="utf-8"))
    assert (index_dir / "ACTIVE_INDEX").exists()
    assert manifest["format_version"] == 6
    assert manifest["validation"]["schema_counts"] == {"sparse_chunks_count": 1, "dense_chunks_count": 1}
    assert manifest["validation"]["exact_term_validated"] is True
    assert manifest["config_hashes"]["fts_config"]
    assert manifest["sdk_versions"]["lancedb"]
    assert manifest["policy"]["selected_mode"] == "ann"
    # Phase 06：发布证据绑定实际行数与动态验证 query 数。
    evidence = manifest["candidate_publication_evidence"]
    assert evidence["actual_dense_rows"] == 1
    assert evidence["validation_query_count"] == min(256, 1)
    assert evidence["query_ef"] == 100
    assert evidence["corpus_query_overlap"] == 0


def test_publication_gate_failure_preserves_active_pointer(tmp_path: Path) -> None:
    """Phase 06：发布门禁失败 → .failed 标记 + 旧 ACTIVE_INDEX 字节不变。"""
    from obsidian_wiki.domain.index_policy import PolicyError

    wiki = tmp_path / "Wiki"
    _write_pages(wiki, 4, body_prefix="GATEFAIL")
    index_dir = tmp_path / ".index"
    build_storage_contract(wiki, index_dir, embed=_embed384())
    original_pointer = (index_dir / "ACTIVE_INDEX").read_bytes()

    service = IndexBuildService(
        LanceDbIndexRepository(index_dir),
        reopen_storage=LanceDbIndexRepository,
        manifest_store=FilesystemIndexManifest(),
        post_commit_journal=FilesystemPostCommitJournal(index_dir),
    )

    def _rejecting_validation(*_args, **_kwargs):
        raise PolicyError("staged candidate recall@10 0.1200 is below the approved floor 0.19")

    real_validation = IndexBuildService._publication_validation
    IndexBuildService._publication_validation = _rejecting_validation
    try:
        with pytest.raises(PolicyError, match="below the approved floor"):
            service.build(wiki, index_dir, embed=_embed384(seed=1))
    finally:
        IndexBuildService._publication_validation = real_validation

    assert (index_dir / "ACTIVE_INDEX").read_bytes() == original_pointer
    failed_markers = list((index_dir / "builds").glob("build_*/.failed"))
    assert failed_markers
    marker_text = failed_markers[0].read_text(encoding="utf-8")
    assert "below the approved floor" in marker_text


# --- Issue #47 C/D — strict two-table persistence + fail-closed kind purity. ---
# The lexical FTS corpus must never again be polluted by dense rows (the #47 P0
# contamination), and the two physical tables stay strictly separated. These tests
# exercise the real LanceDB adapter with tiny synthetic chunks (no embedding model).


def _sparse(chunk_id: str, *, kind: str = "sparse") -> SparseChunk:
    return SparseChunk(
        chunk_id=chunk_id, page_id="page", path="page.md", title="Page",
        text="lexical body text", fts_text="lexical body text",
        chunk_kind=kind, chunk_index=0,
    )


def _dense(chunk_id: str, *, kind: str = "dense") -> DenseChunk:
    return DenseChunk(
        chunk_id=chunk_id, page_id="page", path="page.md", title="Page",
        text="vector leaf text", vector=(1.0, 0.0),
        chunk_kind=kind, chunk_index=0,
    )


def test_persist_rejects_dense_row_inside_sparse_table(tmp_path: Path):
    repo = LanceDbIndexRepository(tmp_path / "lance")
    sparse = [_sparse("s1"), _sparse("s2", kind="dense")]  # mixed!
    dense = [_dense("d1")]
    with pytest.raises(ValueError, match="sparse"):
        repo.persist(tmp_path / "lance", sparse, dense, FtsIndexConfig())


def test_persist_rejects_sparse_row_inside_dense_table(tmp_path: Path):
    repo = LanceDbIndexRepository(tmp_path / "lance")
    sparse = [_sparse("s1")]
    dense = [_dense("d1"), _dense("d2", kind="sparse")]  # mixed!
    with pytest.raises(ValueError, match="dense"):
        repo.persist(tmp_path / "lance", sparse, dense, FtsIndexConfig())


def test_context_rows_unions_both_tables_and_dense_wins_on_collision(tmp_path: Path):
    """When a chunk_id exists in both tables, context_rows keeps one row and the
    dense (vector-bearing) copy wins (issue #47 D)."""
    repo = LanceDbIndexRepository(tmp_path / "lance")
    # Same chunk_id in both tables to exercise the dedup path.
    repo.persist(
        tmp_path / "lance",
        [_sparse("shared")],
        [_dense("shared")],
        FtsIndexConfig(),
    )
    rows = repo.context_rows("page_id = 'page'")
    assert len(rows) == 1
    row = rows[0]
    # Dense copy wins -> carries the vector column, labelled dense.
    assert row["chunk_kind"] == "dense"
    assert "vector" in row


def test_context_rows_returns_union_without_collision(tmp_path: Path):
    repo = LanceDbIndexRepository(tmp_path / "lance")
    repo.persist(
        tmp_path / "lance",
        [_sparse("s1"), _sparse("s2")],
        [_dense("d1")],
        FtsIndexConfig(),
    )
    rows = repo.context_rows("page_id = 'page'")
    kinds = {r["chunk_kind"] for r in rows}
    assert kinds == {"sparse", "dense"}
    assert len(rows) == 3  # no spurious dedup when ids differ


def test_require_current_layout_rejects_stale_layout_version(tmp_path: Path):
    """Fail-closed migration guard (issue #47 F + Phase 06)：旧布局与旧
    format-5 mode-ambiguous manifest 都必须被拒绝并要求重建。"""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "layout": "sparse_chunks+dense_chunks",
        "index_layout_version": 5,  # older than current
    }), encoding="utf-8")
    with pytest.raises(RebuildRequiredError):
        LanceDbIndexRepository.require_current_layout(manifest)

    # Phase 06（issue #49）：format-5 / 缺 format_version 的 mode-ambiguous
    # manifest（requested_vector_index_mode / exact 回退时代）同样拒绝。
    manifest.write_text(json.dumps({
        "layout": "sparse_chunks+dense_chunks",
        "index_layout_version": 6,
        "format_version": 5,
    }), encoding="utf-8")
    with pytest.raises(RebuildRequiredError, match="mode-ambiguous"):
        LanceDbIndexRepository.require_current_layout(manifest)

    manifest.write_text(json.dumps({
        "layout": "sparse_chunks+dense_chunks",
        "index_layout_version": 6,  # current
        "format_version": 6,
    }), encoding="utf-8")
    # Current version is accepted without raising.
    LanceDbIndexRepository.require_current_layout(manifest)


def test_phase07_frozen_base_prepares_real_tables_and_private_hnsw_roles(tmp_path: Path) -> None:
    """The frozen path is a real Lance prepare/clone boundary, never a mock cache."""
    from eval.phase07_frozen_base import (  # noqa: PLC0415
        finalize_private_role,
        prepare_frozen_base,
        validate_frozen_base,
    )
    from obsidian_wiki.domain.index_models import CandidateBuildPolicy, CandidateQueryPolicy  # noqa: PLC0415

    wiki = tmp_path / ".review-tmp" / "phase07" / "frozen-corpus" / "Wiki"
    _write_page(wiki, "# Frozen base\n\nFROZENBASE exact retrieval content.")
    frozen = tmp_path / "prepared"
    descriptor = prepare_frozen_base(
        wiki_dir=wiki, frozen_dir=frozen, embed=_embed384(), tokenizer=_FacadeEmbedder().tokenizer,
        expected_corpus_identity=_phase07_test_corpus_identity(wiki),
    )

    assert descriptor["schema_version"] == 1
    assert not (frozen / "ACTIVE_INDEX").exists()
    assert not list((frozen / "lance_db").rglob("*hnsw*"))
    source_digest = validate_frozen_base(frozen, expected_wiki_root=frozen / "Wiki")
    source = lancedb.connect(str(frozen / "lance_db"))
    assert set(source.table_names()) == {"sparse_chunks", "dense_chunks"}
    assert {index.name for index in source.open_table("sparse_chunks").list_indices()} == {"fts_text_idx"}
    assert not source.open_table("dense_chunks").list_indices()

    for m, ef in ((16, 100), (20, 300), (32, 300)):
        policy = CandidateQueryPolicy(
            candidate="ivf-hnsw-sq", query_ef=ef,
            build_policy=CandidateBuildPolicy(candidate="ivf-hnsw-sq", m=m, ef_construction=300),
        )
        target = tmp_path / f"role-m{m}"
        finalized = finalize_private_role(
            frozen_dir=frozen, target_dir=target, expected_wiki_root=frozen / "Wiki", candidate_query_policy=policy,
        )
        assert finalized["source_tree_sha256"] == source_digest
        target_db = lancedb.connect(str(target / "lance_db"))
        assert len(target_db.open_table("dense_chunks").list_indices()) == 1
        assert {index.name for index in target_db.open_table("sparse_chunks").list_indices()} == {"fts_text_idx"}
        assert validate_frozen_base(frozen, expected_wiki_root=frozen / "Wiki") == source_digest


def test_phase07_private_clone_publishes_and_loads_only_after_validation(tmp_path: Path) -> None:
    """The role clone follows the same ACTIVE_INDEX commit lifecycle as production."""
    from eval.phase07_frozen_base import finalize_private_role, prepare_frozen_base  # noqa: PLC0415
    from obsidian_wiki.domain.index_models import CandidateBuildPolicy, CandidateQueryPolicy  # noqa: PLC0415

    wiki = tmp_path / ".review-tmp" / "phase07" / "frozen-corpus-source" / "Wiki"
    _write_page(wiki, "# Frozen lifecycle\n\nFROZENLIFECYCLE exact retrieval content.")
    frozen = tmp_path / "frozen"
    prepare_frozen_base(
        wiki_dir=wiki, frozen_dir=frozen, embed=_embed384(), tokenizer=_FacadeEmbedder().tokenizer,
        expected_corpus_identity=_phase07_test_corpus_identity(wiki),
    )
    policy = CandidateQueryPolicy(
        candidate="ivf-hnsw-sq", query_ef=300,
        build_policy=CandidateBuildPolicy(candidate="ivf-hnsw-sq", m=20, ef_construction=300),
    )
    private_root = tmp_path / "private"
    published = finalize_private_role(
        frozen_dir=frozen, target_dir=tmp_path / "clone", expected_wiki_root=frozen / "Wiki",
        candidate_query_policy=policy, publish_index_dir=private_root,
    )
    index = WikiIndex(private_root)
    index.load()
    assert (private_root / "ACTIVE_INDEX").is_file()
    assert Path(published["manifest_path"]).is_file()
    assert not list((private_root / "builds").glob("*/.failed"))


def test_phase07_frozen_base_rejects_unsafe_archive_members(tmp_path: Path) -> None:
    from eval.phase07_frozen_base import FrozenBaseError, safe_extract_frozen_base  # noqa: PLC0415

    archive = tmp_path / "unsafe.tar"
    payload = tmp_path / "payload.txt"
    payload.write_text("not allowed", encoding="utf-8")
    with tarfile.open(archive, "w") as handle:
        handle.add(payload, arcname="../escape.txt")
    with pytest.raises(FrozenBaseError, match="archive member"):
        safe_extract_frozen_base(archive, tmp_path / "extract")


def test_phase07_frozen_prepare_uses_the_final_canonical_root_and_rejects_relocated_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixed absolute page IDs make a relocated base invalid before clone/HNSW."""
    from eval.phase07_frozen_base import (  # noqa: PLC0415
        FrozenBaseError,
        finalize_private_role,
        prepare_frozen_base,
        validate_frozen_base,
    )
    from obsidian_wiki.domain.index_models import CandidateBuildPolicy, CandidateQueryPolicy  # noqa: PLC0415

    root_a = tmp_path / ".review-tmp" / "phase07" / "frozen-corpus"
    wiki = root_a / "Wiki"
    _write_page(wiki, "# Canonical root\n\nCANONICALROOTTERM\n")
    prepare_frozen_base(
        wiki_dir=wiki, frozen_dir=root_a, embed=_embed384(), tokenizer=_FacadeEmbedder().tokenizer,
        expected_corpus_identity=_phase07_test_corpus_identity(wiki),
    )
    assert validate_frozen_base(root_a, expected_wiki_root=root_a / "Wiki")

    root_b = tmp_path / "relocated-frozen-corpus"
    import shutil
    shutil.copytree(root_a, root_b)
    policy = CandidateQueryPolicy(
        candidate="ivf-hnsw-sq", query_ef=100,
        build_policy=CandidateBuildPolicy(candidate="ivf-hnsw-sq", m=16, ef_construction=300),
    )
    monkeypatch.setattr(
        LanceDbIndexRepository, "clone_tables",
        lambda *_args, **_kwargs: pytest.fail("relocated frozen source reached clone"),
    )
    with pytest.raises(FrozenBaseError, match="resolved root"):
        finalize_private_role(
            frozen_dir=root_b, target_dir=tmp_path / "should-not-exist",
            expected_wiki_root=root_b / "Wiki", candidate_query_policy=policy,
        )
    assert not (tmp_path / "should-not-exist").exists()


def test_phase07_frozen_prepare_loads_and_validates_model_before_creating_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI must not use a target-local probe or create a failed target."""
    from eval import phase07_frozen_base as frozen  # noqa: PLC0415

    wiki = tmp_path / "Wiki"
    _write_page(wiki, "# CLI\n\nMODELBEFORETARGETTERM\n")
    bundle = tmp_path / "prepare-bundle.json"
    bundle.write_text("{}", encoding="utf-8")
    target = tmp_path / ".review-tmp" / "phase07" / "frozen-corpus"
    calls: list[Path] = []

    monkeypatch.setattr(frozen, "validate_frozen_prepare_bundle", lambda *_args, **_kwargs: {}, raising=False)
    def fail_model(model_dir: Path):
        calls.append(Path(model_dir))
        raise frozen.FrozenBaseError("verified model unavailable")
    monkeypatch.setattr(frozen, "load_verified_frozen_embedder", fail_model)

    with pytest.raises(frozen.FrozenBaseError, match="verified model unavailable"):
        frozen.main([
            "prepare", "--wiki-dir", str(wiki), "--frozen-dir", str(target),
            "--prepare-bundle", str(bundle), "--model-dir", str(tmp_path / "models" / "missing"),
        ])
    assert calls == [tmp_path / "models" / "missing"]
    assert not target.exists()
    assert not list(tmp_path.rglob(".embedder-probe"))


def test_phase07_frozen_prepare_cli_accepts_an_external_verified_model_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful CLI prepare still builds at the already-materialized canonical root."""
    from eval import phase07_frozen_base as frozen  # noqa: PLC0415

    target = tmp_path / ".review-tmp" / "phase07" / "frozen-corpus"
    _write_page(target / "Wiki", "# CLI success\n\nEXTERNALMODELTERM\n")
    bundle = tmp_path / "prepare-bundle.json"
    bundle.write_text("{}", encoding="utf-8")

    class VerifiedEmbedder:
        def __init__(self) -> None:
            self.tokenizer = _FacadeEmbedder().tokenizer

        def embed(self, texts):
            return _embed384()(texts)

    monkeypatch.setattr(frozen, "validate_frozen_prepare_bundle", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(frozen, "load_verified_frozen_embedder", lambda _model_dir: VerifiedEmbedder())
    assert frozen.main([
        "prepare", "--wiki-dir", str(target / "Wiki"), "--frozen-dir", str(target),
        "--prepare-bundle", str(bundle), "--model-dir", str(tmp_path / "external-model"),
    ]) == 0
    assert (target / "frozen-base.json").is_file()
    assert not list(target.rglob(".embedder-probe"))


def test_phase07_frozen_archive_rejects_noncanonical_aliases_and_extracted_tree_is_revalidated(
    tmp_path: Path,
) -> None:
    """Archive member spelling is part of the sealed POSIX inventory."""
    from eval.phase07_frozen_base import FrozenBaseError, safe_extract_frozen_base  # noqa: PLC0415

    archive = tmp_path / "aliases.tar"
    payload = tmp_path / "payload.txt"
    payload.write_text("x", encoding="utf-8")
    with tarfile.open(archive, "w") as handle:
        handle.add(payload, arcname="x")
        handle.add(payload, arcname="./x")
    with pytest.raises(FrozenBaseError, match="archive member"):
        safe_extract_frozen_base(archive, tmp_path / "extract")


def test_phase07_private_finalizer_uses_durable_manifest_collaborator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private publication must preserve the production pre/post-pointer fault boundary."""
    from eval.phase07_frozen_base import finalize_private_role, prepare_frozen_base  # noqa: PLC0415
    from obsidian_wiki.domain.index_models import CandidateBuildPolicy, CandidateQueryPolicy  # noqa: PLC0415

    frozen = tmp_path / "frozen"
    _write_page(frozen / "Wiki", "# Finalizer\n\nDURABLEMANIFESTTERM\n")
    prepare_frozen_base(
        wiki_dir=frozen / "Wiki", frozen_dir=frozen,
        embed=_embed384(), tokenizer=_FacadeEmbedder().tokenizer,
        expected_corpus_identity=_phase07_test_corpus_identity(frozen / "Wiki"),
    )
    policy = CandidateQueryPolicy(
        candidate="ivf-hnsw-sq", query_ef=100,
        build_policy=CandidateBuildPolicy(candidate="ivf-hnsw-sq", m=16, ef_construction=300),
    )
    writes: list[Path] = []
    real_write = FilesystemIndexManifest.write
    def capture_write(self, path: Path, manifest):
        writes.append(path)
        return real_write(self, path, manifest)
    monkeypatch.setattr(FilesystemIndexManifest, "write", capture_write)

    index_dir = tmp_path / "private"
    result = finalize_private_role(
        frozen_dir=frozen, target_dir=tmp_path / "clone", expected_wiki_root=frozen / "Wiki",
        candidate_query_policy=policy, publish_index_dir=index_dir,
    )
    assert writes == [Path(result["manifest_path"])]
    assert (index_dir / "ACTIVE_INDEX").is_file()


def test_phase07_frozen_base_requires_canonical_corpus_identity_and_rejects_nested_sidecars_before_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#6: a descriptor may not bless a tiny/self-reported source or hidden policy file."""
    from eval.phase07_frozen_base import FrozenBaseError, finalize_private_role, prepare_frozen_base  # noqa: PLC0415
    from obsidian_wiki.domain.index_models import CandidateBuildPolicy, CandidateQueryPolicy  # noqa: PLC0415

    wiki = tmp_path / "source" / "Wiki"
    _write_page(wiki, "# Canonical corpus\n\nCANONICALCORPUSIDENTITYTERM\n")
    with pytest.raises(FrozenBaseError, match="corpus identity"):
        prepare_frozen_base(
            wiki_dir=wiki, frozen_dir=tmp_path / "default-rejects-tiny",
            embed=_embed384(), tokenizer=_FacadeEmbedder().tokenizer,
        )

    frozen = tmp_path / "frozen"
    descriptor = prepare_frozen_base(
        wiki_dir=wiki, frozen_dir=frozen, embed=_embed384(), tokenizer=_FacadeEmbedder().tokenizer,
        expected_corpus_identity=_phase07_test_corpus_identity(wiki),
    )
    assert descriptor["expected_corpus_identity"] == _phase07_test_corpus_identity(wiki)
    policy = CandidateQueryPolicy(
        candidate="ivf-hnsw-sq", query_ef=100,
        build_policy=CandidateBuildPolicy(candidate="ivf-hnsw-sq", m=16, ef_construction=300),
    )
    (frozen / "Wiki" / "nested" / "candidate-policy.json").parent.mkdir(parents=True)
    (frozen / "Wiki" / "nested" / "candidate-policy.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        LanceDbIndexRepository, "clone_tables",
        lambda *_args, **_kwargs: pytest.fail("untrusted frozen tree reached clone"),
    )
    with pytest.raises(FrozenBaseError):
        finalize_private_role(
            frozen_dir=frozen, target_dir=tmp_path / "clone", expected_wiki_root=frozen / "Wiki",
            candidate_query_policy=policy,
        )
    assert not (tmp_path / "clone").exists()


def test_phase07_private_finalizer_marks_every_pre_pointer_clone_failure_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record/build/clone/HNSW window has one pre-commit failure outcome."""
    from eval.phase07_frozen_base import finalize_private_role, prepare_frozen_base  # noqa: PLC0415
    from obsidian_wiki.domain.index_models import CandidateBuildPolicy, CandidateQueryPolicy  # noqa: PLC0415

    wiki = tmp_path / "source" / "Wiki"
    _write_page(wiki, "# Clone fault\n\nCLONEFAULTTERM\n")
    frozen = tmp_path / "frozen"
    prepare_frozen_base(
        wiki_dir=wiki, frozen_dir=frozen, embed=_embed384(), tokenizer=_FacadeEmbedder().tokenizer,
        expected_corpus_identity=_phase07_test_corpus_identity(wiki),
    )
    policy = CandidateQueryPolicy(
        candidate="ivf-hnsw-sq", query_ef=100,
        build_policy=CandidateBuildPolicy(candidate="ivf-hnsw-sq", m=16, ef_construction=300),
    )
    monkeypatch.setattr(LanceDbIndexRepository, "clone_tables", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("clone boom")))
    target = tmp_path / "clone"
    private = tmp_path / "private"
    with pytest.raises(RuntimeError, match="clone boom"):
        finalize_private_role(
            frozen_dir=frozen, target_dir=target, expected_wiki_root=frozen / "Wiki",
            candidate_query_policy=policy, publish_index_dir=private,
        )
    builds = list((private / "builds").glob("*"))
    assert len(builds) == 1
    assert (builds[0] / ".failed").is_file()
    assert not (private / "ACTIVE_INDEX").exists()


def test_phase07_frozen_prepare_identity_shape_rejects_future_or_noncanonical_collector_data() -> None:
    """The operator shares one fail-closed, API-collector-shaped prepare identity."""
    from eval.phase07_frozen_base import (  # noqa: PLC0415
        FROZEN_PREPARE_IDENTITY_FIELDS,
        FrozenBaseError,
        validate_frozen_prepare_identity_shape,
    )

    now = datetime(2026, 8, 27, tzinfo=timezone.utc)
    head_sha = "b" * 40
    digest = "a" * 64
    identity = {
        "repository": "example/obsidian-wiki-skill", "head_sha": head_sha,
        "run_id": 11, "run_attempt": 1, "job_id": 12, "artifact_id": 13,
        "artifact_name": "phase07-frozen-base-11-1", "archive_sha256": digest,
        "archive_size_bytes": 99, "descriptor_self_sha256": digest,
        "base_tree_sha256": digest, "model_manifest_sha256": digest,
        "corpus_manifest_sha256": digest, "generator_recipe_sha256": digest,
        "runtime": {"python": "3.13"},
        "artifact_created_at": (now - timedelta(minutes=1)).isoformat(),
        "artifact_expires_at": (now + timedelta(days=89, hours=23, minutes=59)).isoformat(),
        "retention_days": 90, "replacement_for_run_id": None, "status": "success",
    }
    assert set(identity) == FROZEN_PREPARE_IDENTITY_FIELDS
    assert validate_frozen_prepare_identity_shape(
        identity, expected_repository=identity["repository"], expected_head=head_sha, now=now,
    ) == identity
    for key, value in (("run_attempt", 2), ("status", "completed"),
                       ("artifact_name", "latest"),
                       ("artifact_created_at", (now + timedelta(seconds=1)).isoformat()),
                       ("replacement_for_run_id", 7)):
        invalid = dict(identity); invalid[key] = value
        with pytest.raises(FrozenBaseError):
            validate_frozen_prepare_identity_shape(
                invalid, expected_repository=identity["repository"], expected_head=head_sha, now=now,
            )
