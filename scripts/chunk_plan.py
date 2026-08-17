"""Shared page→SparseChunk planning (issue #39).

Single chunking contract reused by both the ``WikiIndex`` facade and the
direct ``build_storage_contract`` path in ``build_index.py``. Reuses the
tokenizer-aware ``chunk_page`` from ``chunking`` plus the ``fts_text``
assembly from ``lexical_tokenizer`` so the two build paths stop diverging
(the divergence stored every page as one whole-page dense chunk, truncating
large docs at the embedding model's 128-token window).

Issue #39 review findings:

* Planning MUST accept/reject pages exactly like ``WikiIndex`` — i.e. through
  the canonical ``scan_wiki`` / ``parse_wiki_page`` front-matter parser — so a
  file without valid front matter is skipped, not turned into a hard error.
* The manifest must carry one logical page per canonical source (with the
  full-file SHA-256), not one row per chunk. ``plan_pages_and_chunks`` returns
  the canonical ``WikiPage`` list alongside the chunks so the orchestration
  layer can build page metadata from the same snapshot it chunked.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import chunking
from chunking import ChunkBuildError, chunk_page
from lexical_tokenizer import extract_exact_terms, fts_terms
from models import WikiPage
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
            structure_kind=r.structure_kind,
            table_header_text=r.table_header_text,
            table_header_start_char=r.table_header_start_char,
            table_header_end_char=r.table_header_end_char,
        ))
    return chunks


def _chunks_for_page(page: WikiPage, *, tokenizer, lexicon) -> List[SparseChunk]:
    """Token-bounded sparse+dense chunks for a single canonical page."""
    page_id = str(Path(page.path).resolve())
    body = (page.content or "").strip()
    if not body:
        return []
    try:
        records = list(chunk_page(
            page_id=page_id, path=Path(page.path), title=page.title,
            page_type=page.page_type, content=body, tokenizer=tokenizer,
        ))
    except Exception as exc:  # pragma: no cover - defensive wrap
        raise ChunkBuildError(
            f"chunk_page 失败: page_id={page_id}, path={page.path}"
        ) from exc
    kinds = {r.chunk_kind for r in records}
    if not records or kinds != {"dense", "sparse"}:
        raise ChunkBuildError(
            f"索引完整性校验失败：非空页面 {page_id} 的 retrieval kinds="
            f"{sorted(kinds)}，期望 ['dense', 'sparse']"
        )
    return chunk_records_to_sparse(records, lexicon)


def plan_pages_and_chunks(
    wiki_dir: Path, project_root: Path, *, tokenizer, lexicon
) -> Tuple[Tuple[WikiPage, ...], Tuple[SparseChunk, ...]]:
    """Canonical pages + token-bounded plan from the *same* Wiki snapshot.

    Pages are parsed through ``scan_wiki`` (the canonical ``parse_wiki_page``
    front-matter contract) so acceptance/rejection and title/type/body
    semantics match ``WikiIndex`` exactly. Files without valid front matter are
    skipped, not treated as hard chunking errors.

    Returns ``(pages, chunks)`` where ``pages`` has one entry per canonical
    source file (each carries the full-file ``sha256``) and ``chunks`` is the
    flattened token-bounded sparse+dense plan for those pages.
    """
    # Imported lazily to avoid a build_index <-> chunk_plan import cycle.
    from build_index import scan_wiki

    pages = tuple(scan_wiki(Path(wiki_dir), Path(project_root)))
    chunks: List[SparseChunk] = []
    for page in pages:
        chunks.extend(_chunks_for_page(page, tokenizer=tokenizer, lexicon=lexicon))
    return pages, tuple(chunks)


def plan_sparse_chunks(wiki_dir: Path, project_root: Path, *, tokenizer, lexicon) -> Tuple[SparseChunk, ...]:
    """Token-bounded sparse+dense plan for every canonical page under ``wiki_dir``.

    ``tokenizer`` is ``callable[[str], int]`` (e.g. ``EmbeddingTokenizer(...).count``).
    Dense leaves are bounded by ``chunking.DENSE_HARD_MAX_TOKENS`` in tokenizer
    units, so large docs are split instead of embedded only at their head. Pages
    are accepted/rejected through the canonical ``scan_wiki`` parser.
    """
    _pages, chunks = plan_pages_and_chunks(
        wiki_dir, project_root, tokenizer=tokenizer, lexicon=lexicon
    )
    return chunks


def page_metadata_from_pages(pages: Sequence[WikiPage]) -> List[dict]:
    """One manifest page entry per canonical source (full-file SHA-256).

    Mirrors the ``WikiIndex`` page metadata contract so both build paths write
    identical logical-page manifests (one row per source, not per chunk).
    """
    return [
        {
            "page_id": str(Path(page.path).resolve()),
            "path": str(page.path),
            "title": page.title,
            "page_type": page.page_type,
            "sources": list(page.sources),
            "links": list(page.links),
            "aliases": list(page.aliases),
            "sha256": page.sha256,
        }
        for page in pages
    ]
