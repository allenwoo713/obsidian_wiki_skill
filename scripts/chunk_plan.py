"""Shared page→SparseChunk planning (issue #39).

Single chunking contract reused by both the ``WikiIndex`` facade and the
direct ``build_storage_contract`` path in ``build_index.py``. Reuses the
tokenizer-aware ``chunk_page`` from ``chunking`` plus the ``fts_text``
assembly from ``lexical_tokenizer`` so the two build paths stop diverging
(the divergence stored every page as one whole-page dense chunk, truncating
large docs at the embedding model's 128-token window).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Sequence

import chunking
from chunking import ChunkBuildError, chunk_page
from lexical_tokenizer import extract_exact_terms, fts_terms, load_lexicon
from obsidian_wiki.domain.index_models import SparseChunk


def chunk_records_to_sparse(records: Sequence[chunking.ChunkRecord], lexicon) -> List[SparseChunk]:
    """Map ``chunk_page`` ChunkRecords → ``SparseChunk`` rows (with ``fts_text``)."""
    chunks: List[SparseChunk] = []
    for r in records:
        chunks.append(SparseChunk(
            chunk_id=r.chunk_id, page_id=r.page_id, path=str(r.path),
            title=r.title, text=r.text,
            fts_text=" ".join(fts_terms(r.text, lexicon) + extract_exact_terms(r.text)),
            page_type=r.page_type,
            section_path=json.dumps(r.section_path, ensure_ascii=False),
            heading=r.heading, chunk_kind=r.chunk_kind,
            chunk_index=r.chunk_index, parent_section_id=r.parent_section_id or "",
            token_count=r.token_count, content_hash=r.content_hash,
            forced_split=r.forced_split, continuation_index=r.continuation_index,
            start_char=r.start_char, end_char=r.end_char,
        ))
    return chunks


def plan_sparse_chunks(wiki_dir: Path, project_root: Path, *, tokenizer, lexicon) -> tuple[SparseChunk, ...]:
    """Token-bounded sparse+dense plan for every canonical .md under ``wiki_dir``.

    ``tokenizer`` is ``callable[[str], int]`` (e.g. ``EmbeddingTokenizer(...).count``).
    Dense leaves are bounded by ``chunking.DENSE_HARD_MAX_TOKENS`` in tokenizer
    units, so large docs are split instead of embedded only at their head.
    """
    chunks: List[SparseChunk] = []
    for path in sorted(Path(wiki_dir).rglob("*.md")):
        if ".graph" in path.parts:
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        front_matter, body = ("", raw.strip())
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            front_matter, body = parts[1], parts[-1].strip()
        if not body:
            continue
        title = path.stem
        page_type = "concept"
        for line in front_matter.splitlines():
            if line.startswith("title:"):
                title = line.partition(":")[2].strip().strip("\"'") or title
            elif line.startswith("type:"):
                page_type = line.partition(":")[2].strip().strip("\"'") or "concept"
        page_id = str(path.resolve())
        try:
            records = list(chunk_page(
                page_id=page_id, path=path, title=title, page_type=page_type,
                content=body, tokenizer=tokenizer,
            ))
        except Exception as exc:
            raise ChunkBuildError(f"chunk_page 失败: page_id={page_id}, path={path}") from exc
        kinds = {r.chunk_kind for r in records}
        if body.strip() and (not records or kinds != {"dense", "sparse"}):
            raise ChunkBuildError(
                f"索引完整性校验失败：非空页面 {page_id} 的 retrieval kinds="
                f"{sorted(kinds)}，期望 ['dense', 'sparse']"
            )
        chunks.extend(chunk_records_to_sparse(records, lexicon))
    return tuple(chunks)
