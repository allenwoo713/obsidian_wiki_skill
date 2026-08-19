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
