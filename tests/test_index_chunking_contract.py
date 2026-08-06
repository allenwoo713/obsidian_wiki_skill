"""Issue #39: build pipeline must use the tokenizer-aware ``chunk_page`` path.

Regression guard: a large page must be split into multiple token-bounded dense
leaves (``<= DENSE_HARD_MAX_TOKENS``) instead of being stored as one whole-page
dense chunk that the embedding model truncates at its 128-token window.
Model-free: a fake tokenizer keeps the test fast and platform-independent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import chunking
import lancedb
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_index import build_storage_contract, plan_sparse_chunks  # noqa: E402


def _write_big_page(wiki: Path, body: str) -> None:
    page = wiki / "big.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        "---\n"
        "type: concept\n"
        "title: Big doc\n"
        "sources: []\n"
        "tags: []\n"
        "related: []\n"
        "---\n\n"
        + body,
        encoding="utf-8",
    )


def _fake_tokenizer(text: str) -> int:
    # ~10 chars/token: deterministic, no heavy model load.
    return max(1, len(text) // 10)


def test_plan_sparse_chunks_splits_large_page_into_token_bounded_leaves(tmp_path: Path) -> None:
    wiki = tmp_path / "Wiki"
    body = "# Overview\n" + ("Radar calibration procedure with many details. " * 200) + \
           "\n## Mounting\n" + ("Installation torque and alignment guidance. " * 200)
    _write_big_page(wiki, body)

    chunks = plan_sparse_chunks(wiki, tmp_path, tokenizer=_fake_tokenizer, lexicon={})
    dense = [c for c in chunks if c.chunk_kind == "dense"]
    sparse = [c for c in chunks if c.chunk_kind == "sparse"]

    assert sparse and dense, "plan must contain both sparse and dense chunk kinds"
    assert len(dense) > 1, "large page must split into multiple dense leaves"
    for d in dense:
        assert d.token_count <= chunking.DENSE_HARD_MAX_TOKENS, (
            f"dense leaf {d.chunk_id} exceeds cap: {d.token_count}"
        )
        assert d.text != body, "dense leaf must not be the whole page body"


def test_storage_contract_honors_token_bounded_dense_chunks(tmp_path: Path) -> None:
    wiki = tmp_path / "Wiki"
    body = "# Overview\n" + ("Calibration procedure detail line. " * 300) + \
           "\n## Torque\n" + ("Installation torque specification note. " * 300)
    _write_big_page(wiki, body)

    artifact = build_storage_contract(
        wiki,
        tmp_path / ".index",
        embed=lambda texts: [[1.0, float(i + 1)] for i, _ in enumerate(texts)],
        tokenizer=_fake_tokenizer,
        lexicon={},
    )

    db = lancedb.connect(str(artifact.artifact.lance_dir))
    dense = db.open_table("dense_chunks")
    token_counts = dense.to_arrow().to_pydict()["token_count"]

    assert len(token_counts) > 1, "should persist multiple dense leaves, not one whole-page block"
    assert max(token_counts) <= chunking.DENSE_HARD_MAX_TOKENS, (
        f"stored dense leaf exceeds cap: {max(token_counts)}"
    )

    manifest = json.loads(artifact.artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["validation"]["schema_counts"]["dense_chunks_count"] > 1
