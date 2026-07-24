"""Hierarchical, tokenizer-aware chunking (GitHub issue #1).

Replaces the legacy char-based ``split_into_chunks`` in ``build_index.py``
with a persistent :class:`ChunkRecord` model. Chunks preserve document
structure (headings, paragraphs, lists, tables, fenced code, blockquotes,
wikilinks) and measure length with the **actual** embedding tokenizer
rather than a character count.

Three-layer model (issue #1):

    Page
    └── Parent section   (one per heading; carries section_path)
        ├── Sparse section chunk   (full section text  → FTS/BM25)
        └── Dense leaf chunks      (token-bounded       → vector index)

The tokenizer is **injected** (``Tokenizer`` callable) so the module is
testable without loading the heavy ``sentence-transformers`` model.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

# --------------------------------------------------------------------------
# Config (issue #1: default parameters)
# --------------------------------------------------------------------------
DENSE_TARGET_TOKENS = 96
DENSE_HARD_MAX_TOKENS = 112
DENSE_OVERLAP_TOKENS = 20

SPARSE_TARGET_CHARS = 650
SPARSE_HARD_MAX_CHARS = 1000
SPARSE_OVERLAP_CHARS = 100

CHUNK_SCHEMA_VERSION = 2

Tokenizer = Callable[[str], int]


def _default_tokenizer(text: str) -> int:
    """Fallback estimator (~4 chars/token mixed zh/en).

    Production injects the real embedding tokenizer so dense chunks never
    exceed ``DENSE_HARD_MAX_TOKENS``. This keeps chunking importable and
    testable without the model.
    """
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class ChunkRecord:
    chunk_id: str
    page_id: str
    path: Path
    title: str
    page_type: str
    chunk_kind: str      # 'sparse' (section context) | 'dense' (retrieval leaf)
    section_path: List[str]
    heading: str
    chunk_index: int
    parent_section_id: Optional[str]
    text: str
    start_char: int
    end_char: int
    token_count: int
    content_hash: str


@dataclass
class Block:
    """A structural block inside a section."""
    kind: str            # heading|paragraph|list|table|code|quote
    text: str
    level: int = 0
    start_char: int = 0
    end_char: int = 0


@dataclass
class Section:
    """A parent section: blocks accumulated under one heading."""
    heading: str
    level: int
    section_path: List[str]
    blocks: List[Block] = field(default_factory=list)
    start_char: int = 0


# --------------------------------------------------------------------------
# Block parsing
# --------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^(```|~~~)")
_HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
_QUOTE_RE = re.compile(r"^\s*>\s?")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]+\|?\s*$")


def parse_blocks(text: str) -> List[Block]:
    """Split markdown text into structural :class:`Block` objects."""
    lines = text.split("\n")
    blocks: List[Block] = []
    i = 0
    n = len(lines)
    offset = 0  # running char offset for start/end accounting

    def _push(kind, buf_lines, level=0, start=0):
        body = "\n".join(buf_lines).strip()
        if body:
            blocks.append(Block(kind=kind, text=body, level=level,
                                start_char=start, end_char=start + len("\n".join(buf_lines))))

    para: List[str] = []
    para_start = 0

    def _flush_para():
        nonlocal para
        if para:
            _push("paragraph", para, start=para_start)
            para = []

    while i < n:
        line = lines[i]
        line_start = offset
        # fenced code
        fm = _FENCE_RE.match(line)
        if fm:
            _flush_para()
            fence = fm.group(1)
            buf = [line]
            j = i + 1
            while j < n and not lines[j].strip().startswith(fence):
                buf.append(lines[j])
                j += 1
            if j < n:
                buf.append(lines[j])  # closing fence
            _push("code", buf, start=line_start)
            i = j + 1
            offset = (offset + sum(len(l) + 1 for l in lines[i - len(buf):i])) if False else _advance(offset, lines, i - len(buf), i)
            continue
        # heading
        hm = _HEAD_RE.match(line)
        if hm:
            _flush_para()
            _push("heading", [line], level=len(hm.group(1)), start=line_start)
            i += 1
            offset += len(line) + 1
            continue
        # table (header + separator)
        if line.strip().startswith("|") and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            _flush_para()
            buf = [line, lines[i + 1]]
            j = i + 2
            while j < n and lines[j].strip().startswith("|"):
                buf.append(lines[j])
                j += 1
            _push("table", buf, start=line_start)
            i = j
            offset += sum(len(l) + 1 for l in buf)
            continue
        # blockquote
        if _QUOTE_RE.match(line):
            _flush_para()
            buf = [line]
            j = i + 1
            while j < n and _QUOTE_RE.match(lines[j]):
                buf.append(lines[j])
                j += 1
            _push("quote", buf, start=line_start)
            i = j
            offset += sum(len(l) + 1 for l in buf)
            continue
        # list item
        if _LIST_RE.match(line):
            _flush_para()
            buf = [line]
            j = i + 1
            while j < n and (_LIST_RE.match(lines[j]) or (lines[j].strip() and lines[j].startswith((" ", "\t")))):
                buf.append(lines[j])
                j += 1
            _push("list", buf, start=line_start)
            i = j
            offset += sum(len(l) + 1 for l in buf)
            continue
        # blank line ends paragraph
        if not line.strip():
            _flush_para()
            i += 1
            offset += len(line) + 1
            continue
        # ordinary line → paragraph accumulator
        if not para:
            para_start = line_start
        para.append(line)
        i += 1
        offset += len(line) + 1
    _flush_para()
    return blocks


def _advance(offset, lines, a, b):
    return offset + sum(len(lines[k]) + 1 for k in range(a, b))


# --------------------------------------------------------------------------
# Sentence splitting (wikilink-safe)
# --------------------------------------------------------------------------
_SENT_RE = re.compile(r"[^。！？!?；;]+[。！？!?；]?|\n|$")
_WLINK_RE = re.compile(r"\[\[.*?\]\]")
_WLINK_SPLIT = re.compile(r"(\[\[.*?\]\])")


def split_sentences(text: str) -> List[str]:
    """Split into sentences without breaking a ``[[wikilink]]`` in half."""
    raw = [s.strip() for s in _SENT_RE.findall(text) if s.strip()]
    out: List[str] = []
    for s in raw:
        if out and out[-1].count("[[") > out[-1].count("]]"):
            out[-1] = out[-1] + " " + s  # merge unbalanced wikilink tail
        else:
            out.append(s)
    return out


def _force_split_tokens(text: str, tokenizer: Tokenizer, max_tokens: int,
                        overlap_tokens: int) -> List[str]:
    """Fallback for a single over-long block/sentence: split into atomic
    pieces (whitespace + CJK chars) and pack by token budget with overlap.

    Never uses ``text[-n:]`` character slicing.
    """
    atoms: List[str] = []
    for part in _WLINK_SPLIT.split(text):
        if not part:
            continue
        if part.startswith("[[") and part.endswith("]]"):
            atoms.append(part)            # keep wikilinks whole
            continue
        for piece in re.split(r"(\s+)", part):
            if not piece:
                continue
            if re.fullmatch(r"[一-鿿]+", piece):
                atoms.extend(piece)       # each CJK char is an atom
            else:
                atoms.append(piece)
    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for a in atoms:
        t = tokenizer(a)
        if cur and cur_tok + t > max_tokens:
            chunks.append("".join(cur).strip())
            tail_tok = 0
            kept: List[str] = []
            for s in reversed(cur):
                st = tokenizer(s)
                if tail_tok + st > overlap_tokens:
                    break
                kept.insert(0, s)
                tail_tok += st
            cur = kept
            cur_tok = tail_tok
        cur.append(a)
        cur_tok += t
    if cur:
        chunks.append("".join(cur).strip())
    return [c for c in chunks if c]


# --------------------------------------------------------------------------
# IDs
# --------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _make_chunk_id(page_id: str, section_path: List[str], text: str,
                   occurrence: int) -> str:
    payload = f"{page_id}|{'/'.join(section_path)}|{_norm(text)}|{occurrence}|v{CHUNK_SCHEMA_VERSION}"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{page_id}::{h}"


def _section_id(page_id: str, section_path: List[str]) -> str:
    payload = f"{page_id}|{'/'.join(section_path)}"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{page_id}::sec::{h}"


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def chunk_page(page_id: str, path: Path, title: str, page_type: str,
               content: str,
               tokenizer: Optional[Tokenizer] = None) -> List[ChunkRecord]:
    """Chunk a wiki page into sparse section + dense leaf ChunkRecords.

    ``tokenizer`` defaults to :func:`_default_tokenizer`. Production passes
    the real embedding tokenizer so dense chunks respect ``DENSE_HARD_MAX_TOKENS``.
    """
    tok = tokenizer or _default_tokenizer
    blocks = parse_blocks(content)

    # group into sections by heading levels. section_path follows the
    # markdown heading hierarchy (resets on a higher/same-level heading),
    # not literal append — so a new H1 starts a fresh top-level section.
    sections: List[Section] = []
    heading_stack: List[tuple] = []  # (level, heading)
    cur: Optional[Section] = None
    for b in blocks:
        if b.kind == "heading":
            heading_text = re.sub(r"^#{1,6}\s+", "", b.text).strip()
            lvl = b.level
            while heading_stack and heading_stack[-1][0] >= lvl:
                heading_stack.pop()
            heading_stack.append((lvl, heading_text))
            if cur is not None:
                sections.append(cur)
            cur = Section(heading=heading_text, level=lvl,
                          section_path=[h for _, h in heading_stack])
            cur.start_char = b.start_char
        else:
            if cur is None:
                # content before any heading → synthetic top section
                cur = Section(heading="", level=0, section_path=[])
            cur.blocks.append(b)
    if cur is not None:
        sections.append(cur)

    records: List[ChunkRecord] = []
    idx = 0
    for sec in sections:
        prefix = (" / ".join(sec.section_path) + "\n") if sec.section_path else ""
        sec_text = prefix + "\n".join(b.text for b in sec.blocks)
        psec_id = _section_id(page_id, sec.section_path)

        # --- sparse section chunk(s) ---
        for occ, sp_text in enumerate(_split_sparse(sec_text)):
            cid = _make_chunk_id(page_id, sec.section_path, sp_text, occ)
            records.append(ChunkRecord(
                chunk_id=cid, page_id=page_id, path=Path(path), title=title,
                page_type=page_type, chunk_kind="sparse",
                section_path=list(sec.section_path),
                heading=sec.heading, chunk_index=idx,
                parent_section_id=None, text=sp_text,
                start_char=sec.start_char, end_char=sec.start_char + len(sec_text),
                token_count=len(sp_text) // 4,  # sparse is char-budgeted
                content_hash=hashlib.sha256(sp_text.encode("utf-8")).hexdigest()[:16],
            ))
            idx += 1

        # --- dense leaf chunks ---
        segs: List[str] = []
        for b in sec.blocks:
            if b.kind in ("paragraph", "quote", "list"):
                for s in split_sentences(b.text):
                    if s.strip():
                        segs.append(s)
            else:  # table / code → one segment, force-split if too big
                if tok(b.text) > DENSE_HARD_MAX_TOKENS:
                    segs.extend(_force_split_tokens(b.text, tok,
                                                    DENSE_TARGET_TOKENS,
                                                    DENSE_OVERLAP_TOKENS))
                else:
                    segs.append(b.text)
        for occ, dtext in enumerate(_pack_dense(prefix, segs, tok)):
            cid = _make_chunk_id(page_id, sec.section_path, dtext, occ)
            records.append(ChunkRecord(
                chunk_id=cid, page_id=page_id, path=Path(path), title=title,
                page_type=page_type, chunk_kind="dense",
                section_path=list(sec.section_path),
                heading=sec.heading, chunk_index=idx,
                parent_section_id=psec_id, text=dtext,
                start_char=sec.start_char, end_char=sec.start_char + len(dtext),
                token_count=tok(dtext),
                content_hash=hashlib.sha256(dtext.encode("utf-8")).hexdigest()[:16],
            ))
            idx += 1
    return records


def _split_sparse(text: str) -> List[str]:
    """Section text → one or more char-budgeted sparse chunks."""
    if len(text) <= SPARSE_HARD_MAX_CHARS:
        return [text] if text.strip() else []
    out: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + SPARSE_TARGET_CHARS, len(text))
        out.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - SPARSE_OVERLAP_CHARS, start + 1)
    return out


def _pack_dense(prefix: str, segs: List[str], tok: Tokenizer) -> List[str]:
    """Pack sentences into ≤DENSE_HARD_MAX_TOKENS dense chunks with
    sentence/block overlap (never ``text[-n:]``)."""
    out: List[str] = []
    cur: List[str] = []
    cur_tok = tok(prefix) if prefix else 0
    for raw in segs:
        # a single over-long sentence/segment → force-split by tokens
        if tok(raw) > DENSE_HARD_MAX_TOKENS:
            sub_segs = _force_split_tokens(raw, tok, DENSE_TARGET_TOKENS,
                                           DENSE_OVERLAP_TOKENS)
        else:
            sub_segs = [raw]
        for seg in sub_segs:
            t = tok(seg)
            if cur and cur_tok + t > DENSE_TARGET_TOKENS:
                out.append(prefix + " ".join(cur))
                # overlap: keep trailing complete segments up to budget
                tail_tok = 0
                kept: List[str] = []
                for s in reversed(cur):
                    st = tok(s)
                    if tail_tok + st > DENSE_OVERLAP_TOKENS:
                        break
                    kept.insert(0, s)
                    tail_tok += st
                cur = kept
                cur_tok = (tok(prefix) if prefix else 0) + tail_tok
            cur.append(seg)
            cur_tok += t
            if cur_tok > DENSE_HARD_MAX_TOKENS and len(cur) > 1:
                # hard cap guard: drop leading segment(s)
                while cur_tok > DENSE_HARD_MAX_TOKENS and len(cur) > 1:
                    dropped = cur.pop(0)
                    cur_tok -= tok(dropped)
    if cur:
        out.append(prefix + " ".join(cur))
    return out
