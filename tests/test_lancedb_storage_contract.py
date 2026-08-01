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
    long_term = "x" * 180
    wiki = tmp_path / "Wiki"
    _write_page(wiki, f"# Contract\n\nThe exact storage token is {long_term}.")

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

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["layout"] == "sparse_chunks+dense_chunks"
    assert manifest["fts_config"] == {
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
