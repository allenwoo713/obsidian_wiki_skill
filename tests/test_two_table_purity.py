"""Issue #47 C/D — strict two-table persistence + fail-closed kind purity.

The lexical FTS corpus must never again be polluted by dense rows (the #47 P0
contamination), and the two physical tables stay strictly separated. These tests
exercise the real LanceDB adapter with tiny synthetic chunks (no embedding model).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from obsidian_wiki.domain.index_models import (
    DenseChunk,
    FtsIndexConfig,
    RebuildRequiredError,
    SparseChunk,
)
from obsidian_wiki.infrastructure.lancedb_index_repository import (
    LanceDbIndexRepository,
)


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
    """Fail-closed migration guard (issue #47 F): an index built under an older
    layout version must be rejected, forcing a rebuild, not served from a
    contaminated/old schema."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "layout": "sparse_chunks+dense_chunks",
        "index_layout_version": 5,  # older than current
    }), encoding="utf-8")
    with pytest.raises(RebuildRequiredError):
        LanceDbIndexRepository.require_current_layout(manifest)

    manifest.write_text(json.dumps({
        "layout": "sparse_chunks+dense_chunks",
        "index_layout_version": 6,  # current
    }), encoding="utf-8")
    # Current version is accepted without raising.
    LanceDbIndexRepository.require_current_layout(manifest)
