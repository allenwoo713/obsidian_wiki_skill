"""Real-LanceDB gates for the staged online incremental path."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_index import build_storage_contract  # noqa: E402
from obsidian_wiki.domain.index_models import SparseChunk  # noqa: E402


def _embed(texts):
    """Small deterministic, approved-dimension test embedder."""
    vectors = []
    for offset, _text in enumerate(texts):
        raw = [float((offset + index) % 17 + 1) for index in range(384)]
        norm = math.sqrt(sum(value * value for value in raw))
        vectors.append([value / norm for value in raw])
    return vectors


def _chunks(*, text: str) -> tuple[SparseChunk, ...]:
    page_id = "concepts/online.md"
    common = dict(
        page_id=page_id,
        path="concepts/online.md", title="Online", text=text,
        content_hash=text, end_char=len(text),
    )


def _page_chunks(page: str, *, text: str, suffix: str = "stable") -> tuple[SparseChunk, ...]:
    page_id = f"concepts/{page}.md"
    common = dict(
        page_id=page_id, path=f"concepts/{page}.md", title=page.title(), text=text,
        content_hash=text, end_char=len(text),
    )
    return (
        SparseChunk(**common, chunk_id=f"{page_id}::sparse-{suffix}", fts_text=text, chunk_kind="sparse"),
        SparseChunk(**common, chunk_id=f"{page_id}::dense-{suffix}", fts_text=text, chunk_kind="dense"),
    )
    return (
        SparseChunk(**common, chunk_id=f"{page_id}::sparse-stable", fts_text=text, chunk_kind="sparse"),
        SparseChunk(**common, chunk_id=f"{page_id}::dense-stable", fts_text=text, chunk_kind="dense"),
    )


def _pointer(index_dir: Path) -> dict[str, object]:
    return json.loads((index_dir / "ACTIVE_INDEX").read_text(encoding="utf-8"))


def test_single_page_edit_uses_staged_shallow_clone_and_atomic_pointer(tmp_path, monkeypatch):
    """An edit mutates only a cloned generation before the sole pointer commit."""
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService
    import obsidian_wiki.application.incremental_index_service as service_module
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    original = _chunks(text="original unique payload")
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=original)
    old_pointer_bytes = (index_dir / "ACTIVE_INDEX").read_bytes()
    old_pointer = _pointer(index_dir)
    old_lance = index_dir / str(old_pointer["active_lance"])
    old_repository = LanceDbIndexRepository(old_lance)
    old_sparse_rows = old_repository.context_rows("page_id = 'concepts/online.md'")

    publish_calls: list[tuple[Path, Path]] = []
    real_publish = service_module.publish_pointer

    def _publish_after_all_validation(index_path, build_path, **kwargs):
        # Every source observation remains unchanged until the actual commit point.
        assert (index_dir / "ACTIVE_INDEX").read_bytes() == old_pointer_bytes
        assert LanceDbIndexRepository(old_lance).context_rows(
            "page_id = 'concepts/online.md'"
        ) == old_sparse_rows
        publish_calls.append((index_path, build_path))
        return real_publish(index_path, build_path, **kwargs)

    monkeypatch.setattr(service_module, "publish_pointer", _publish_after_all_validation)
    outcome = IncrementalIndexService().build(
        wiki_dir, index_dir, canonical_chunks=_chunks(text="edited unique payload"), embed=_embed,
    )

    assert len(publish_calls) == 1
    assert outcome.artifact.lance_dir != old_lance
    active = _pointer(index_dir)
    assert active["active_lance"] == str(outcome.artifact.lance_dir.relative_to(index_dir))
    manifest = json.loads(outcome.artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["layout"] == "sparse_chunks+dense_chunks"
    assert manifest["ann_policy"]["selected_index_type"] == "ivf-hnsw-sq"
    assert manifest["ann_policy"]["query_ef"] == 100


def test_unchanged_population_has_no_mutation_rows(tmp_path):
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    plan = _page_chunks("steady", text="unchanged alpha") + _page_chunks("also_steady", text="unchanged beta")
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=plan)

    outcome = IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=plan, embed=_embed)

    for delta in (outcome.sparse_delta, outcome.dense_delta):
        assert not delta.added_ids and not delta.updated_ids and not delta.deleted_ids
        assert len(delta.unchanged_ids) == 2
        assert not delta.physically_written_ids


def test_split_merge_retains_stable_ids_and_reports_only_new_rows(tmp_path):
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    retained = _page_chunks("split", text="retained block")
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=retained)
    expanded = retained + _page_chunks("split", text="new split block", suffix="new")

    outcome = IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=expanded, embed=_embed)

    assert len(outcome.sparse_delta.unchanged_ids) == 1
    assert len(outcome.dense_delta.unchanged_ids) == 1
    assert len(outcome.sparse_delta.added_ids) == 1
    assert len(outcome.dense_delta.added_ids) == 1
    assert not outcome.sparse_delta.updated_ids and not outcome.dense_delta.updated_ids


def test_page_deletion_removes_all_table_and_manifest_residue(tmp_path):
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    keeper = _page_chunks("keeper", text="retain this content")
    doomed = _page_chunks("doomed", text="unique deleted-token-zqxwp")
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=keeper + doomed)

    outcome = IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=keeper, embed=_embed)
    repository = LanceDbIndexRepository(outcome.artifact.lance_dir)
    for table_name in ("sparse_chunks", "dense_chunks"):
        assert not [row for row in repository.table_rows(table_name) if row["page_id"] == "concepts/doomed.md"]
    manifest = json.loads(outcome.artifact.manifest_path.read_text(encoding="utf-8"))
    assert all(page["page_id"] != "concepts/doomed.md" for page in manifest["pages"])
    assert len(outcome.sparse_delta.deleted_ids) == len(outcome.dense_delta.deleted_ids) == 1


def test_row_accounting_reports_only_physical_upserts(tmp_path):
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    original = _page_chunks("accounted", text="original accounting")
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=original)
    changed = _page_chunks("accounted", text="changed accounting")

    outcome = IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=changed, embed=_embed)

    for delta, mutation in ((outcome.sparse_delta, outcome.sparse_mutation), (outcome.dense_delta, outcome.dense_mutation)):
        assert mutation.inserted == len(delta.added_ids)
        assert mutation.updated == len(delta.updated_ids)
        assert mutation.deleted == len(delta.deleted_ids)
        assert mutation.physically_written == len(delta.physically_written_ids)
