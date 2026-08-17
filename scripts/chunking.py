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
from typing import Callable, Iterator, List, NamedTuple, Optional

# --------------------------------------------------------------------------
# Config (issue #1: default parameters)
# --------------------------------------------------------------------------
DENSE_TARGET_TOKENS = 96
DENSE_HARD_MAX_TOKENS = 112
DENSE_OVERLAP_TOKENS = 20

SPARSE_TARGET_CHARS = 650
SPARSE_HARD_MAX_CHARS = 1000
SPARSE_OVERLAP_CHARS = 100
# #13 review：sparse 正文最小保留预算。当 section 前缀（标题路径）自身超过硬上限时，
# 压缩前缀而非吞掉正文，保证 body_budget >= MIN_BODY_RESERVE。
MIN_BODY_RESERVE = 200

# issue #47：sparse / dense 分表后，schema 版本必须独立，使 sparse-only 改动
# （如本 issue 的 block-first 重构）不会使未变的 dense ID / 向量缓存失效。
# 注意 dense 仍用 v3（与历史向量缓存兼容），sparse 升到 v4。
SPARSE_CHUNK_SCHEMA_VERSION = 4
DENSE_CHUNK_SCHEMA_VERSION = 3

# #13：chunk_id 契约升级。v2 把 ID 编成 `schema:page_id:kind:index`，
# 前方插入/删除 chunk 会导致后续 ID 全漂移；v3 改为内容哈希
# （page_id::{sha256(kind|body|occurrence)}），与位置无关，插入无关
# section 后未修改 chunk 的 ID 保持不变。vec_cache namespace / checkpoint
# signature / eval 归一化均随之失效旧缓存、强制全量重编码。
# 默认版本保留为 dense（历史兼容）；sparse 调用方显式传 SPARSE_CHUNK_SCHEMA_VERSION。
CHUNK_SCHEMA_VERSION = DENSE_CHUNK_SCHEMA_VERSION

Tokenizer = Callable[[str], int]


def count_token_ids(hf_tokenizer, text: str) -> int:
    """Count the model input IDs without invoking model-length warnings.

    ``PreTrainedTokenizer.encode`` warns when a *counting* request is longer
    than the model sequence window, although no inference is about to happen.
    The callable tokenizer accepts ``verbose=False`` and an explicit
    ``truncation=False``: this both suppresses that inference-oriented warning
    and avoids its backend's configured max-length truncation. Include special
    tokens so the count remains identical to the former ``tokenizer.encode``
    path.
    """
    try:
        encoded = hf_tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )
        return len(encoded["input_ids"])
    except (TypeError, KeyError):
        # Kept for lightweight test doubles and non-fast tokenizer adapters.
        backend = getattr(hf_tokenizer, "backend_tokenizer", None)
        if backend is not None:
            return len(backend.encode(text, add_special_tokens=True).ids)
        return len(hf_tokenizer.encode(text))


class ChunkBuildError(RuntimeError):
    """领域异常：分块/索引构建失败（fail-fast）。

    #13 review：生产构建中任一页面分块异常都必须让 staging build 失败并保留旧
    活动索引，不得静默漏页。``_chunk_rows_for_page`` 捕获 ``chunk_page`` 异常后
    包装为本异常（携带 page_id/path），``_build_chunks`` 直接上抛，``build()``
    不翻转 ACTIVE_INDEX。sparse 硬上限被违反时同样抛出，作为不可绕过的最终守卫。
    """



def _default_tokenizer(text: str) -> int:
    """Fallback estimator (~4 chars/token mixed zh/en).

    Production injects the real embedding tokenizer so dense chunks never
    exceed ``DENSE_HARD_MAX_TOKENS``. This keeps chunking importable and
    testable without the model. The build path must NOT use this — it is only
    a safety net for offline unit tests that pass an explicit tokenizer.
    """
    return max(1, len(text) // 4)


class EmbeddingTokenizer:
    """Lightweight adapter around an HF tokenizer for accurate token counting.

    #13：生产构建只初始化一次 embedding 模型的 tokenizer，并显式传入
    ``chunk_page(..., tokenizer=token_counter.count)``。模型 tokenizer 加载
    失败时必须终止构建（保留旧活动索引），不得静默回退字符估算，故构造时
    若拿到 ``None`` 直接抛错。
    """

    def __init__(self, hf_tokenizer):
        if hf_tokenizer is None:
            raise ValueError(
                "EmbeddingTokenizer 需要真实 HF tokenizer；禁止回退字符估算")
        self._hf = hf_tokenizer

    def count(self, text: str) -> int:
        """Return the exact token count the embedding model would produce."""
        return count_token_ids(self._hf, text)



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
    forced_split: bool = False          # #13 review：sparse 超长强制切片时为 True
    continuation_index: int = -1        # #13 review：强制切片的续片序号（从 0）
    # issue #47：结构感知 provenance（原样保留真实 source span，不伪造连续切片）。
    structure_kind: str = "paragraph"   # paragraph|quote|list|code|table
    table_header_text: str = ""         # 大表窗口重复 header 的真实文本
    table_header_start_char: int = -1   # header 真实起始 offset
    table_header_end_char: int = -1     # header 真实结束 offset


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
        # Retain the exact source slice.  Trimming here makes the stored text
        # disagree with its advertised offsets whenever a block has leading or
        # trailing whitespace.
        body = "\n".join(buf_lines)
        if body.strip():
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


class _Seg:
    """A text piece with its original span in the source document.

    ``start_char``/``end_char`` are absolute offsets into the page content,
    so a chunk's span maps back to the real original text (issue #13).
    """
    __slots__ = ("text", "start_char", "end_char")

    def __init__(self, text: str, start_char: int, end_char: int):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char


def _force_split_with_spans(text: str, start_char: int, tokenizer: Tokenizer,
                            max_tokens: int, overlap_tokens: int):
    """Split one over-long piece into atomic, wikilink-safe pieces with exact
    original spans. Never uses ``text[a:b]`` mid-token / mid-wikilink slicing.

    Returns a list of ``(text, start_char, end_char)`` whose spans are derived
    from each atom's real offset within ``text`` (offset by ``start_char``).
    """
    # 1) atomize, tracking each atom's offset within `text`
    atoms: List[tuple] = []  # (atom_text, offset)
    offset = 0
    for part in _WLINK_SPLIT.split(text):
        if not part:
            continue
        if part.startswith("[[") and part.endswith("]]"):
            atoms.append((part, offset))          # keep wikilinks whole
            offset += len(part)
            continue
        for piece in re.split(r"(\s+)", part):
            if not piece:
                continue
            if re.fullmatch(r"[一-鿿]+", piece):
                for ch in piece:                  # each CJK char is an atom
                    atoms.append((ch, offset))
                    offset += 1
            else:
                atoms.append((piece, offset))
                offset += len(piece)
    def _split_oversize_atom(atom: str, atom_off: int) -> List[tuple]:
        """Split only an atom that itself violates the hard limit.

        Best-effort atom boundaries (including Wiki Links) are preserved until
        one atom is too large.  At that point the hard token contract wins;
        binary-search a character prefix with the *real* tokenizer so every
        emitted fragment fits and retains an exact source offset.
        """
        if tokenizer(atom) <= max_tokens:
            return [(atom, atom_off)]
        pieces: List[tuple] = []
        pos = 0
        while pos < len(atom):
            lo, hi, best = 1, len(atom) - pos, 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if tokenizer(atom[pos:pos + mid]) <= max_tokens:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best == 0:
                raise ChunkBuildError(
                    f"单字符已超过 dense hard limit ({max_tokens} tokens)")
            pieces.append((atom[pos:pos + best], atom_off + pos))
            pos += best
        return pieces

    # 2) pack atoms by token budget with overlap tail
    out: List[tuple] = []
    cur: List[tuple] = []
    cur_tok = 0
    expanded_atoms: List[tuple] = []
    for a, a_off in atoms:
        expanded_atoms.extend(_split_oversize_atom(a, a_off))
    for a, a_off in expanded_atoms:
        t = tokenizer(a)
        if cur and cur_tok + t > max_tokens:
            s0 = cur[0][1]
            s1 = cur[-1][1] + len(cur[-1][0])
            out.append(("".join(x[0] for x in cur),
                        start_char + s0, start_char + s1))
            tail_tok = 0
            kept: List[tuple] = []
            for x in reversed(cur):
                xt = tokenizer(x[0])
                if tail_tok + xt > overlap_tokens:
                    break
                kept.insert(0, x)
                tail_tok += xt
            cur = kept
            cur_tok = tail_tok
        cur.append((a, a_off))
        cur_tok += t
    if cur:
        s0 = cur[0][1]
        s1 = cur[-1][1] + len(cur[-1][0])
        out.append(("".join(x[0] for x in cur),
                    start_char + s0, start_char + s1))
    return out


def _seg_block(b: "Block", tokenizer: Tokenizer) -> List["_Seg"]:
    """Segment a single block into ``_Seg`` pieces, each within the dense
    budget. Splits on sentence/token boundaries (never mid-block char slice).
    """
    if b.kind in ("paragraph", "quote", "list"):
        segs: List[_Seg] = []
        offset = 0
        for s in split_sentences(b.text):
            if not s.strip():
                offset += len(s)
                continue
            st = b.text.find(s, offset)
            if st < 0:
                st = offset
            segs.append(_Seg(s, b.start_char + st, b.start_char + st + len(s)))
            offset = st + len(s)
        return segs
    # table / code → one piece; if over the dense hard max, force-split
    if tokenizer(b.text) > DENSE_HARD_MAX_TOKENS:
        return [_Seg(t, s, e) for (t, s, e) in
                _force_split_with_spans(b.text, b.start_char, tokenizer,
                                       DENSE_TARGET_TOKENS, DENSE_OVERLAP_TOKENS)]
    return [_Seg(b.text, b.start_char, b.end_char)]


# --------------------------------------------------------------------------
# IDs
# --------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _make_chunk_id(page_id: str, chunk_kind: str, body_text: str,
                   occurrence: int, schema_version: int = CHUNK_SCHEMA_VERSION) -> str:
    """Position-stable chunk ID (issue #13).

    Derived from ``(chunk_kind, normalized body, occurrence)`` — NOT from
    section_path or in-page position. Therefore inserting an unrelated
    section at the top of a page does not change the IDs of unchanged chunks:
    their body text is identical and their occurrence count (per
    ``(kind, body)`` in document order) is unchanged.

    ``page_id`` is kept as a namespace prefix so two pages with identical body
    text still get distinct, unique IDs in the LanceDB table.

    issue #47：``schema_version`` 按 kind 区分（sparse v4 / dense v3），使
    sparse-only 改动不会使未变的 dense ID / 向量缓存失效。
    """
    payload = f"{chunk_kind}|{_norm(body_text)}|{occurrence}|v{schema_version}"
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

    ``tokenizer`` defaults to :func:`_default_tokenizer` (offline unit tests
    only). Production passes the real embedding tokenizer via
    :class:`EmbeddingTokenizer` so dense chunks respect ``DENSE_HARD_MAX_TOKENS``
    and ``start_char``/``end_char`` map back to the original text.
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
    seen: Dict[tuple, int] = {}  # (kind, norm_body) -> occurrence counter
    stats: dict = {"forced_sparse_splits": 0}

    def _occ(kind: str, body: str) -> int:
        # occurrence is counted per (kind, normalized body) in document order,
        # so inserting an unrelated section elsewhere leaves unchanged chunks'
        # IDs stable (issue #13).
        key = (kind, _norm(body))
        n = seen.get(key, 0)
        seen[key] = n + 1
        return n

    for sec in sections:
        prefix = (" / ".join(sec.section_path) + "\n") if sec.section_path else ""
        psec_id = _section_id(page_id, sec.section_path)

        # --- sparse section chunk(s): block-first 结构感知 + 真·硬上限 (issue #47) ---
        for (sp_text, sp_prefix, sp_start, sp_end, forced, cont_idx,
             structure_kind, th_text, th_start, th_end) in \
                _sparse_chunks_for_section(sec, prefix, tok, stats):
            body = sp_text[len(sp_prefix):] if sp_prefix and sp_text.startswith(sp_prefix) else sp_text
            cid = _make_chunk_id(page_id, "sparse", body, _occ("sparse", body),
                                 SPARSE_CHUNK_SCHEMA_VERSION)
            records.append(ChunkRecord(
                chunk_id=cid, page_id=page_id, path=Path(path), title=title,
                page_type=page_type, chunk_kind="sparse",
                section_path=list(sec.section_path),
                heading=sec.heading, chunk_index=idx,
                parent_section_id=None, text=sp_text,
                start_char=sp_start, end_char=sp_end,
                token_count=len(sp_text) // 4,  # sparse is char-budgeted (no vector)
                content_hash=hashlib.sha256(sp_text.encode("utf-8")).hexdigest()[:16],
                forced_split=forced, continuation_index=cont_idx,
                structure_kind=structure_kind,
                table_header_text=th_text,
                table_header_start_char=th_start,
                table_header_end_char=th_end,
            ))
            idx += 1

        # --- dense leaf chunks: real spans from _Seg pieces ---
        segs: List[_Seg] = []
        for b in sec.blocks:
            segs.extend(_seg_block(b, tok))
        for dtext, dstart, dend in _pack_dense(prefix, segs, tok):
            body = dtext[len(prefix):] if dtext.startswith(prefix) else dtext
            cid = _make_chunk_id(page_id, "dense", body, _occ("dense", body),
                                 DENSE_CHUNK_SCHEMA_VERSION)
            records.append(ChunkRecord(
                chunk_id=cid, page_id=page_id, path=Path(path), title=title,
                page_type=page_type, chunk_kind="dense",
                section_path=list(sec.section_path),
                heading=sec.heading, chunk_index=idx,
                parent_section_id=psec_id, text=dtext,
                start_char=dstart, end_char=dend,
                token_count=tok(dtext),
                content_hash=hashlib.sha256(dtext.encode("utf-8")).hexdigest()[:16],
            ))
            idx += 1
    return records


class _SparseYield(NamedTuple):
    """Block-first sparse 规划的一次产出（issue #47）。"""
    text: str
    start: int
    end: int
    forced: bool
    cont_idx: int
    structure_kind: str
    table_header_text: str
    table_header_start_char: int
    table_header_end_char: int


def _sparse_chunks_for_section(sec: "Section", prefix: str,
                               tokenizer: Tokenizer,
                               stats: Optional[dict] = None):
    """Yield 10-tuple ``(full_text, eff_prefix, start, end, forced, cont_idx,
    structure_kind, table_header_text, table_header_start_char, table_header_end_char)``.

    issue #47：block-first 结构感知。每个 Markdown block 默认成为 1 个 exact-span
    sparse chunk（段落/引用/列表/代码/小表整体），仅当超过硬上限（SPARSE_HARD_MAX_CHARS
    减 section 前缀）时才 fallback 切片；大表按 header-aware 连续行窗口切分，每个数据
    行恰好覆盖一次。绝不把多个 block 拼成 1 个 chunk 再谎称是 1 段连续 source span。
    """
    eff_prefix = _effective_prefix(prefix)
    prefix_len = len(eff_prefix)
    body_budget = SPARSE_HARD_MAX_CHARS - prefix_len
    if body_budget < MIN_BODY_RESERVE:
        body_budget = MIN_BODY_RESERVE

    for y in _sparse_blocks_for_section(sec, body_budget, tokenizer, stats):
        full = (eff_prefix + y.text) if eff_prefix else y.text
        if len(full) > SPARSE_HARD_MAX_CHARS:
            raise ChunkBuildError(
                f"sparse 硬上限被违反: len={len(full)} > {SPARSE_HARD_MAX_CHARS}")
        yield (full, eff_prefix, y.start, y.end, y.forced, y.cont_idx,
               y.structure_kind, y.table_header_text,
               y.table_header_start_char, y.table_header_end_char)


def _sparse_blocks_for_section(sec: "Section", body_budget: int,
                               tokenizer: Tokenizer,
                               stats: Optional[dict] = None) -> Iterator["_SparseYield"]:
    """Block-first sparse planning (issue #47).

    一个 Markdown block → 一个 exact-span sparse chunk；仅当该 block 单独超限时才
    fallback。大表按 header-aware 连续行窗口切分。
    """
    for block in sec.blocks:
        if not block.text.strip():
            continue
        if block.kind == "table" and len(block.text) > body_budget:
            yield from _split_large_table_windows(block, body_budget)
            continue
        if len(block.text) <= body_budget:
            yield _SparseYield(
                text=block.text, start=block.start_char, end=block.end_char,
                forced=False, cont_idx=-1, structure_kind=block.kind,
                table_header_text="", table_header_start_char=-1,
                table_header_end_char=-1,
            )
            continue
        # 单 block 超限（段落/引用/列表/代码）→ 边界安全 fallback 切片。
        for piece_text, ps, pe, cidx in _force_split_sparse_unit(block, body_budget):
            yield _SparseYield(
                text=piece_text, start=ps, end=pe, forced=True, cont_idx=cidx,
                structure_kind=block.kind, table_header_text="",
                table_header_start_char=-1, table_header_end_char=-1,
            )
            if stats is not None:
                stats["forced_sparse_splits"] += 1


def _lines_with_absolute_spans(block: "Block") -> List[tuple]:
    """表格/代码 block 的逐行 (text, start_char, end_char)；滚动 cursor 保留真实 offset。"""
    out: List[tuple] = []
    cursor = 0
    for raw in block.text.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        if line.strip():
            out.append((line, block.start_char + cursor, block.start_char + cursor + len(line)))
        cursor += len(raw)
    return out


def _pack_complete_rows(rows: List[tuple], budget: int) -> List[List[tuple]]:
    """把完整数据行贪心打包进窗口；绝不切断行。单行长于 budget 时独占一个窗口。"""
    windows: List[List[tuple]] = []
    cur: List[tuple] = []
    cur_len = 0
    for row in rows:
        row_len = len(row[0])
        if cur and cur_len + 1 + row_len > budget:
            windows.append(cur)
            cur, cur_len = [], 0
        cur.append(row)
        cur_len += row_len + (1 if len(cur) > 1 else 0)
    if cur:
        windows.append(cur)
    return windows


def _split_large_table_windows(block: "Block", body_budget: int) -> Iterator["_SparseYield"]:
    """大表按 header-aware 连续行窗口切分（issue #47）。

    header = 第 1 行(表头) + 第 2 行(分隔符)；每个窗口覆盖若干完整数据行，且每个
    数据行恰好出现一次。``text`` 为行窗口原文（真实 span），``table_header_*``
    为真实 header span —— 绝不伪造单一连续 source slice。

    超长表头自身也被强制切片（保证其源内容被真实 span 覆盖，不静默丢弃，issue #13）；
    超长数据行按 cell 感知字符切片（真实 span，不丢单元格内容）。
    """
    lines = _lines_with_absolute_spans(block)
    if len(lines) < 3:  # 仅表头+分隔符，无数据行 → 整表一个 chunk
        yield _SparseYield(
            text=block.text, start=block.start_char, end=block.end_char,
            forced=False, cont_idx=-1, structure_kind="table",
            table_header_text="", table_header_start_char=-1,
            table_header_end_char=-1,
        )
        return
    header = lines[:2]
    rows = lines[2:]
    header_text = "\n".join(line[0] for line in header)
    header_start = header[0][1]
    header_end = header[-1][2]

    # issue #13 硬上限 + #47：超长表头自身也必须被真实 span 覆盖（强制切片），
    # 否则表头内容在 block-first 规划下会整段丢失。
    if len(header_text) > body_budget:
        for piece, ps, pe in _char_slice_text(header_text, body_budget, header_start):
            yield _SparseYield(
                text=piece, start=ps, end=pe, forced=True, cont_idx=-1,
                structure_kind="table",
                table_header_text=header_text,
                table_header_start_char=header_start,
                table_header_end_char=header_end,
            )

    # 数据行打包成窗口；单条超长行自身按 cell/字符强制切片（真实 span，不丢内容）。
    for window in _pack_complete_rows(rows, body_budget):
        overlong = [w for w in window if len(w[0]) > body_budget]
        if overlong:
            for w in window:
                if len(w[0]) <= body_budget:
                    yield _SparseYield(
                        text=w[0], start=w[1], end=w[2], forced=False, cont_idx=-1,
                        structure_kind="table",
                        table_header_text=header_text,
                        table_header_start_char=header_start,
                        table_header_end_char=header_end,
                    )
                else:
                    for piece, ps, pe in _split_table_row(w[0], w[1], body_budget):
                        yield _SparseYield(
                            text=piece, start=ps, end=pe, forced=True, cont_idx=-1,
                            structure_kind="table",
                            table_header_text=header_text,
                            table_header_start_char=header_start,
                            table_header_end_char=header_end,
                        )
        else:
            start = window[0][1]
            end = window[-1][2]
            yield _SparseYield(
                text=block.text[start - block.start_char: end - block.start_char],
                start=start, end=end, forced=False, cont_idx=-1,
                structure_kind="table",
                table_header_text=header_text,
                table_header_start_char=header_start,
                table_header_end_char=header_end,
            )


def _force_split_sparse_unit(unit: "Block", body_budget: int):
    """单个超限原子的强制切片 fallback。返回 ``(text, start, end, cont_idx)`` 列表，
    每段 ``len(text) <= body_budget``（最终拼接 prefix 后 <= 硬上限）。保留真实 span，
    续片序号从 0 起。"""
    if unit.kind == "table":
        frags = _split_table_row(unit.text, unit.start_char, body_budget)
    else:
        frags = _char_slice_text(unit.text, body_budget, unit.start_char)
    return [(t, s, e, i) for i, (t, s, e) in enumerate(frags)]


def _table_cells_row(text: str):
    """将表格行按单元格切成 ``(cell_content, rel_start, rel_end)``（内容为两管道间文本，
    不含管道本身）。``rel_*`` 是相对本行 ``text`` 的偏移，调用方需叠加 ``row_unit.start_char``。
    """
    cells: List[tuple] = []
    n = len(text)
    i = 1 if (n > 0 and text[0] == "|") else 0
    end = n - 1 if (n > 0 and text[n - 1] == "|") else n
    seg = text[i:end]
    pos = i
    for part in seg.split("|"):
        cells.append((part, pos, pos + len(part)))
        pos += len(part) + 1  # +1 跳过被 split 吃掉的分隔管道
    return cells


def _split_table_row(row_text: str, row_start: int, body_budget: int):
    """超长表格行按 cell 强制切片；单个 cell 仍超限时按字符切片。返回
    ``(text, start, end)`` 片段，每段 ``len(text) <= body_budget``，且保留真实 span
    （偏移叠加 ``row_start``，span 与片段文本逐字符一致）。绝不丢弃单元格内容。

    span 约定：片段文本为 ``|`` + cell 内容 + ``|``（cell 感知切片的天然边界），
    单 cell 字符切片片段同理取 ``[ps, pe]`` 的源区间。
    """
    # Fragment exact source slices rather than synthesising leading/trailing
    # pipes.  This handles optional trailing pipes and escaped/duplicate cell
    # text without claiming an out-of-range or fabricated span.  Prefer table
    # separators as cut points; an overlong cell falls through to the mandatory
    # character fallback.
    text = row_text
    cuts = [0]
    escaped = False
    for i, ch in enumerate(text):
        if ch == "|" and not escaped and i > 0:
            cuts.append(i + 1)
        escaped = (ch == "\\" and not escaped)
        if ch != "\\":
            escaped = False
    cuts.append(len(text))
    cuts = sorted(set(cuts))
    frags: List[tuple] = []
    start = 0
    while start < len(text):
        end = min(start + body_budget, len(text))
        candidates = [c for c in cuts if start < c <= end]
        if candidates:
            end = max(candidates)
        if end <= start:
            end = min(start + body_budget, len(text))
        piece = text[start:end]
        frags.append((piece, row_start + start, row_start + end))
        start = end
    return frags


def _char_slice_text(text: str, body_budget: int, base_start: int):
    """字符预算感知的强制切片 fallback（最终硬限制）。绝不切断 ``[[wikilink]]``，
    返回 ``(piece, start, end)``，每段 <= body_budget，span 映射回原文。

    若整段都是单个超长 ``[[wikilink]]``/identifier（原子本身 > 硬上限），硬限制优先，
    必须执行最终切分（metadata 由调用方标记 forced_split）。
    """
    if len(text) <= body_budget:
        return [(text, base_start, base_start + len(text))]
    out: List[tuple] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + body_budget, n)
        # 避免从 [[ ... ]] 中间切断 wikilink（除非整段都是 wikilink，则硬限制优先）
        while j > i + 1 and "]]" in text[i:j] and "[[" in text[i:j]:
            k = text.rfind("[[", i, j)
            if k <= i:
                break
            j = k
        piece = text[i:j]
        out.append((piece, base_start + i, base_start + j))
        if j >= n:
            break
        i = j
    return out


def _effective_prefix(prefix: str) -> str:
    """展示前缀（section path）若自身超过硬上限，做可追溯压缩：保留头尾、中段用
    省略标记，确保正文预算 >= MIN_BODY_RESERVE（正文不得因前缀过长而消失）。

    返回长度严格 <= ``cap = SPARSE_HARD_MAX_CHARS - MIN_BODY_RESERVE``（=800），
    故 ``body_budget = HARD - len(prefix) >= MIN_BODY_RESERVE`` 恒成立。
    """
    cap = SPARSE_HARD_MAX_CHARS - MIN_BODY_RESERVE  # 800
    if len(prefix) <= cap:
        return prefix
    marker = " …(section path truncated)… "
    budget = cap - len(marker)
    half = budget // 2
    return prefix[:half] + marker + prefix[-(budget - half):]


def _mk_block(text: str, start_char: int, end_char: int) -> "Block":
    """Minimal ``Block`` carrying only the fields read by the sparse splitter."""
    return Block(kind="paragraph", text=text, start_char=start_char, end_char=end_char)


def _dense_prefix_with_budget(prefix: str, tok: Tokenizer) -> str:
    """Return a display prefix that leaves room for at least one body token."""
    if not prefix or tok(prefix) < DENSE_HARD_MAX_TOKENS:
        return prefix
    cap = DENSE_HARD_MAX_TOKENS - 1
    lo, hi, best = 0, len(prefix), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if tok(prefix[:mid]) <= cap:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best == 0:
        raise ChunkBuildError("dense section prefix exceeds the token hard limit")
    return prefix[:best]


def _pack_dense(prefix: str, segs: List["_Seg"], tok: Tokenizer):
    """Pack ``_Seg`` pieces into dense chunks ≤DENSE_HARD_MAX_TOKENS.

    Returns ``(text, start_char, end_char)`` tuples. A chunk's span is the
    real min/max of its contributing segments' original offsets (issue #13),
    and the budget is prefix-aware so the stored token count (prefix + body)
    never exceeds the hard max even for nested section paths.
    """
    prefix = _dense_prefix_with_budget(prefix, tok)
    prefix_tok = tok(prefix) if prefix else 0
    hard = DENSE_HARD_MAX_TOKENS - prefix_tok
    target = max(1, min(DENSE_TARGET_TOKENS - prefix_tok, hard))

    out: List[tuple] = []
    cur: List[_Seg] = []
    cur_tok = prefix_tok

    def _flush():
        if not cur:
            return
        body = " ".join(s.text for s in cur)
        full = prefix + body
        if tok(full) > DENSE_HARD_MAX_TOKENS:
            raise ChunkBuildError(
                f"dense 硬上限被违反: tokens={tok(full)} > {DENSE_HARD_MAX_TOKENS}")
        out.append((full, cur[0].start_char, cur[-1].end_char))

    def _split_for_prefix(seg: _Seg) -> List[_Seg]:
        """Ensure each body piece fits when its synthetic prefix is included."""
        if tok(prefix + seg.text) <= DENSE_HARD_MAX_TOKENS:
            return [seg]
        pieces: List[_Seg] = []
        pos = 0
        while pos < len(seg.text):
            lo, hi, best = 1, len(seg.text) - pos, 0
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = seg.text[pos:pos + mid]
                if tok(prefix + candidate) <= DENSE_HARD_MAX_TOKENS:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best == 0:
                raise ChunkBuildError("dense prefix leaves no token budget for a body character")
            pieces.append(_Seg(seg.text[pos:pos + best], seg.start_char + pos,
                               seg.start_char + pos + best))
            pos += best
        return pieces

    for raw in segs:
        # a single over-long segment → force-split by tokens (preserves spans)
        if tok(prefix + raw.text) > DENSE_HARD_MAX_TOKENS:
            sub = _force_split_with_spans(raw.text, raw.start_char, tok,
                                         hard, DENSE_OVERLAP_TOKENS)
            sub_segs = [_Seg(t, s, e) for (t, s, e) in sub]
        else:
            sub_segs = [raw]
        prefixed_sub_segs: List[_Seg] = []
        for sub_seg in sub_segs:
            prefixed_sub_segs.extend(_split_for_prefix(sub_seg))
        for seg in prefixed_sub_segs:
            t = tok(seg.text)
            candidate = prefix + " ".join(s.text for s in cur + [seg])
            if cur and tok(candidate) > target:
                _flush()
                # overlap: keep trailing complete segments up to budget
                tail_tok = 0
                kept: List[_Seg] = []
                for s in reversed(cur):
                    st = tok(s.text)
                    if tail_tok + st > DENSE_OVERLAP_TOKENS:
                        break
                    kept.insert(0, s)
                    tail_tok += st
                cur = kept
                cur_tok = prefix_tok + tail_tok
                # Joining pieces adds separator tokens with some tokenizers.
                # Drop overlap until the next full record fits the hard cap.
                while cur and tok(prefix + " ".join(s.text for s in cur + [seg])) > DENSE_HARD_MAX_TOKENS:
                    cur.pop(0)
            cur.append(seg)
            cur_tok = tok(prefix + " ".join(s.text for s in cur))
            if cur_tok > DENSE_HARD_MAX_TOKENS:
                # A single segment that cannot fit even after forced splitting
                # is a build error, never a silently truncated embedding.
                raise ChunkBuildError(
                    f"dense segment exceeds hard limit: tokens={cur_tok}")
    _flush()
    return out
