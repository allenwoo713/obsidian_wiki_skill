"""Real-model Phase 07 frozen-base acceptance gates.

This module is intentionally absent from the model-free CI matrix.  The Eval
workflow runs it explicitly after the immutable embedding model is available.
"""
from __future__ import annotations

from pathlib import Path


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


def _corpus_identity(wiki: Path) -> dict[str, object]:
    from eval.ann_corpus_manifest import canonical_content_tree_sha256

    return {
        "expanded_content_tree_sha256": canonical_content_tree_sha256(wiki),
        "expanded_member_count": sum(1 for path in wiki.rglob("*.md") if path.is_file()),
    }


def test_phase07_frozen_base_uses_the_manifest_verified_local_model_tokenizer(tmp_path: Path) -> None:
    """Exercise the production tokenizer identity without a skip or fake embedder."""
    from eval.phase07_frozen_base import (
        load_verified_frozen_embedder,
        prepare_frozen_base,
        validate_frozen_base,
    )

    model_dir = Path(__file__).parent.parent / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
    embedder = load_verified_frozen_embedder(model_dir)
    wiki = tmp_path / "source" / "Wiki"
    _write_page(wiki, "# Model tokenizer\n\nMODEL_TOKENIZER_FROZEN_IDENTITY\n")
    identity = _corpus_identity(wiki)
    frozen = tmp_path / "frozen"
    prepare_frozen_base(
        wiki_dir=wiki,
        frozen_dir=frozen,
        embed=lambda texts: embedder.embed(list(texts)),
        tokenizer=embedder.tokenizer,
        expected_corpus_identity=identity,
    )
    assert validate_frozen_base(
        frozen,
        expected_wiki_root=frozen / "Wiki",
        tokenizer=embedder.tokenizer,
        expected_corpus_identity=identity,
    )
