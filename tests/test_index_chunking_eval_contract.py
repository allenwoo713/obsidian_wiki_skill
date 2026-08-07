"""Real-model production-path gate for issue #39.

This suite belongs in the Ubuntu Eval job after the local model bootstrap.  It
must not be replaced by the fast fake-tokenizer architecture tests: the original
defect existed only in the production ``main()`` orchestration path.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import lancedb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_index  # noqa: E402
import chunking  # noqa: E402
from chunking import EmbeddingTokenizer  # noqa: E402
from obsidian_wiki.application.active_index_pointer import resolve_active_lance_dir  # noqa: E402
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository  # noqa: E402
from obsidian_wiki.infrastructure.sentence_transformer_embedder import (  # noqa: E402
    SentenceTransformerEmbedder,
)


def _write_page(wiki: Path, name: str, title: str, body: str) -> Path:
    path = wiki / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "type: concept\n"
        f"title: {json.dumps(title)}\n"
        "sources: []\n"
        "tags: []\n"
        "related: []\n"
        "---\n\n"
        + body,
        encoding="utf-8",
    )
    return path


def test_production_cli_real_model_recalls_fact_beyond_token_500(tmp_path, monkeypatch):
    """Build through ``main`` and retrieve a tail-only fact from persisted dense leaves."""
    project = tmp_path / "project"
    wiki = project / "Wiki"
    filler = (
        "Routine inspection records cover ordinary fasteners, labels, housings, and visual checks. "
        * 140
    )
    gold = (
        "The named emergency mode is Zephyr Isolation Protocol. "
        "It disconnects the torque bus when the inertial controller reports runaway drift."
    )
    target_body = "# Routine service history\n" + filler + "\n## Emergency control\n" + gold
    target = _write_page(wiki, "tail_target.md", "Tail target", target_body)
    for index in range(6):
        _write_page(
            wiki,
            f"distractor_{index:02d}.md",
            f"Drift response catalog {index}",
            "# Incident catalog\n"
            "Emergency inertial drift response, torque bus isolation, and named stabilization "
            "procedures are discussed here, but this catalog contains no approved mode name.",
        )

    model_path = Path(
        os.environ.get("WIKI_EMBEDDER_LOCAL_PATH") or build_index.SKILL_EMBEDDER_DIR
    )
    assert (model_path / "model.safetensors").is_file(), (
        f"Eval must bootstrap the repository-local embedding model first: {model_path}"
    )
    embedder = SentenceTransformerEmbedder(model_path)
    token_counter = EmbeddingTokenizer(embedder.tokenizer).count
    prefix = target_body.split(gold, 1)[0]
    assert token_counter(prefix) > 500, "gold evidence must be beyond the model's head window"

    monkeypatch.setenv("WIKI_EMBEDDER_LOCAL_PATH", str(model_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_index.py", str(project), "--full-rebuild"],
    )
    build_index.main()

    index_dir = project / ".index"
    active_lance = resolve_active_lance_dir(index_dir)
    dense_rows = lancedb.connect(str(active_lance)).open_table("dense_chunks").to_arrow().to_pylist()
    target_rows = [row for row in dense_rows if row["page_id"] == str(target.resolve())]
    assert len(target_rows) > 1
    assert max(row["token_count"] for row in target_rows) <= chunking.DENSE_HARD_MAX_TOKENS
    assert all(row["text"] != target_body for row in target_rows)

    query = "What named emergency protocol disconnects the torque bus during runaway inertial drift?"
    query_vector = embedder.embed([query])[0]
    hits = LanceDbIndexRepository(active_lance).search_dense_exact(
        query_vector, metric="cosine", limit=10,
    )
    assert any("Zephyr Isolation Protocol" in str(hit.get("text", "")) for hit in hits), (
        "tail evidence must be recalled from a persisted dense leaf within rank 10"
    )

    pointer = json.loads((index_dir / "ACTIVE_INDEX").read_text(encoding="utf-8"))
    manifest = json.loads(
        (index_dir / "builds" / pointer["build_id"] / "manifest.json").read_text(encoding="utf-8")
    )
    page_ids = [page["page_id"] for page in manifest["pages"]]
    assert len(page_ids) == len(set(page_ids)) == 7
