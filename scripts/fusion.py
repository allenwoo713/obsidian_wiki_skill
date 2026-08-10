"""检索融合与上下文装配（GitHub issues #3/#4）。

- ``page_level_rrf``：把 chunk 级 FTS / 向量命中按 page_id 归并，做 page-level
  RRF 融合。图谱信号不在主 RRF 内（由 query.py 作为独立扩展通道追加）。
- ``assemble_context``：按 token 预算（默认 60/20/5/15 四路）把 PageCandidate
  装配成直接可喂 LLM 的 ContextBundle。

放在独立模块，避免 build_index ↔ query 的循环依赖。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

from models import (
    ChunkHit, PageCandidate, ContextItem, ContextBundle, EvidenceHit, GraphPath,
)

_FM_RE = re.compile(r"^---\n.*?\n---\n(.*)$", re.DOTALL)


def _read_full_content(path: Path, max_chars: Optional[int] = None) -> str:
    """读取 wiki 页面完整内容（去 frontmatter）。

    ``max_chars`` is retained only for backwards-compatible callers. Context
    assembly always passes ``None`` and applies an explicit token-aware
    truncation policy instead of silently changing the source content here.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = _FM_RE.match(raw)
    content = m.group(1).strip() if m else raw
    if max_chars is not None and len(content) > max_chars:
        content = content[:max_chars] + "\n...[截断，完整内容见原文件]"
    return content


def page_level_rrf(fts_hits: List[ChunkHit], vector_hits: List[ChunkHit],
                   k: int = 5, k_rrf: int = 60) -> List[PageCandidate]:
    """Page-level RRF 融合。

    FTS 与向量两路各自按 score 排序（rank 从 1），命中按 page_id 归并，
    每页的 RRF 分 = Σ 1/(k_rrf + rank)。每页保留各路最佳 chunk 作为证据。
    图谱不在主 RRF 内——由调用方作为独立通道追加，避免噪声挤占 top。
    """
    # 归并：保留每一路的所有命中证据；呈现文本随后去重，provenance 不去重。
    pages: dict = {}
    for channel_hits in (fts_hits, vector_hits):
        for rank, h in enumerate(channel_hits, 1):
            pid = h.page_id
            entry = pages.get(pid)
            if entry is None:
                entry = {
                    "page_id": pid, "path": h.path, "title": h.title,
                    "page_type": h.page_type, "fts_rank": None, "vec_rank": None,
                    "fts_hits": [], "vec_hits": [], "rrf": 0.0,
                }
                pages[pid] = entry
            entry["rrf"] += 1.0 / (k_rrf + rank)
            if channel_hits is fts_hits:
                entry["fts_hits"].append((rank, h))
                if entry["fts_rank"] is None or rank < entry["fts_rank"]:
                    entry["fts_rank"] = rank
            else:
                entry["vec_hits"].append((rank, h))
                if entry["vec_rank"] is None or rank < entry["vec_rank"]:
                    entry["vec_rank"] = rank

    candidates: List[PageCandidate] = []
    for pid, e in pages.items():
        sparse_ev: List[EvidenceHit] = []
        dense_ev: List[EvidenceHit] = []
        for rank, h in e["fts_hits"]:
            sparse_ev.append(EvidenceHit(
                chunk_id=h.chunk_id, channel="sparse", rank=rank,
                raw_score=h.score, text=h.text, section_path=h.section_path))
        for rank, h in e["vec_hits"]:
            dense_ev.append(EvidenceHit(
                chunk_id=h.chunk_id, channel="dense", rank=rank,
                raw_score=h.score, text=h.text, section_path=h.section_path))
        candidates.append(PageCandidate(
            page_id=pid, path=Path(e["path"]), title=e["title"],
            rrf_score=e["rrf"], sparse_rank=e["fts_rank"], dense_rank=e["vec_rank"],
            sparse_evidence=sparse_ev, dense_evidence=dense_ev,
        ))
    candidates.sort(key=lambda c: -c.rrf_score)
    return candidates[:k]


def _all_evidence(c: PageCandidate) -> List[EvidenceHit]:
    """Return deterministic full evidence provenance from both channels."""
    return sorted(c.sparse_evidence + c.dense_evidence,
                  key=lambda e: (e.rank, e.channel, e.chunk_id))


def _unique_text(hits: List[EvidenceHit]) -> str:
    """Deduplicate rendered text only; every source EvidenceHit remains intact."""
    seen, parts = set(), []
    for hit in hits:
        text = hit.text.strip()
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
    return "\n\n".join(parts)


class _FallbackContextRepository:
    """Compatibility adapter for callers that have not supplied WikiIndex yet."""
    def __init__(self, candidates: List[PageCandidate]):
        self._paths = {c.page_id: c.path for c in candidates}

    def get_chunk(self, chunk_id):
        return None

    def get_neighbors(self, chunk_id):
        return []

    def get_parent_section(self, chunk_id):
        return []

    def get_page_sources(self, page_id):
        return []

    def read_page(self, page_id):
        path = self._paths.get(page_id)
        return _read_full_content(path) if path else ""


def _repository_text(repository, candidate: PageCandidate, evidence: List[EvidenceHit], scope: str) -> str:
    """Read selected context through the narrow repository port only."""
    if scope in ("full_page", "full_source"):
        return (getattr(repository, "read_page", lambda _pid: "")(candidate.page_id) or "").strip()
    if not evidence:
        return ""
    anchor = evidence[0]
    chunks = []
    get_chunk = getattr(repository, "get_chunk", None)
    if get_chunk:
        stored = get_chunk(anchor.chunk_id)
        if stored:
            chunks.append(stored)
    if not chunks:
        chunks = [anchor]
    if scope == "adjacent":
        chunks = list(getattr(repository, "get_neighbors", lambda _id: [])(anchor.chunk_id) or []) + chunks
    elif scope in ("section", "parent_section"):
        chunks = list(getattr(repository, "get_parent_section", lambda _id: [])(anchor.chunk_id) or []) or chunks
    elif scope == "multiple_sections":
        chunks = []
        seen_ids = set()
        for hit in evidence:
            for part in getattr(repository, "get_parent_section", lambda _id: [])(hit.chunk_id) or [hit]:
                key = getattr(part, "chunk_id", "")
                if key not in seen_ids:
                    seen_ids.add(key)
                    chunks.append(part)
    # Repository chunks are ChunkHit-compatible. Their displayed text is still
    # deduplicated without dropping the original evidence metadata.
    texts, seen = [], set()
    for chunk in chunks:
        text = getattr(chunk, "text", "").strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return "\n\n".join(texts)


def _truncate_to_budget(text: str, budget: int, token_counter):
    """Return token-counted prefix plus explicit omitted-range metadata."""
    total = token_counter(text)
    if total <= budget:
        return text, False, []
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if token_counter(text[:mid]) <= budget:
            low = mid
        else:
            high = mid - 1
    prefix = text[:low].rstrip()
    return prefix, True, [{"start_char": len(prefix), "end_char": len(text), "reason": "token_limit"}]


def _citation_path(path, wiki_root: Optional[Path]) -> str:
    """Canonicalise a page path into the ``Wiki/<...>.md`` citation contract.

    Presentation-only boundary (issue #43): storage identity (``page_id``,
    stored index rows, ``PageCandidate.path`` used for disk reads) is never
    rewritten here — only what the agent finally cites.

    ``wiki_root`` is the *authoritative* vault root supplied by the caller.
    Rightmost-``Wiki``-component matching is deliberately NOT used: for a vault
    at ``/project/Wiki`` a page at ``/project/Wiki/archive/Wiki/page.md`` must
    cite ``Wiki/archive/Wiki/page.md``, not ``Wiki/page.md``.

    Fails closed: an absolute candidate outside ``wiki_root`` — or a
    non-canonical path when no root is known — raises ``ValueError`` rather
    than emitting a citation the reader cannot resolve.
    """
    raw = str(path).replace("\\", "/")
    pure = PurePosixPath(raw)
    if wiki_root is None:
        # No authoritative root: only an already-canonical citation is safe.
        if not pure.is_absolute() and pure.parts and pure.parts[0] == "Wiki" \
                and ".." not in pure.parts:
            return pure.as_posix()
        raise ValueError(
            f"absolute/non-canonical citation path requires wiki_root: {raw}")

    candidate = Path(path)
    root = Path(wiki_root)
    if candidate.is_absolute():
        try:
            relative = candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
        except ValueError as exc:
            raise ValueError(f"citation path is outside Wiki root: {candidate}") from exc
    else:
        relative = PurePosixPath(raw)
        if relative.parts and relative.parts[0] == "Wiki":
            relative = PurePosixPath(*relative.parts[1:])
    parts = tuple(relative.parts)
    if not parts or ".." in parts:
        raise ValueError(f"invalid Wiki-relative citation path: {relative}")
    return PurePosixPath("Wiki", *parts).as_posix()


def assemble_context(
    candidates: List[PageCandidate],
    wi=None,
    *,
    mode: str = "snippet",
    repository=None,
    scope: Optional[str] = None,
    max_tokens: int = 4096,
    token_counter=None,
    citation_root: Optional[Path] = None,
) -> ContextBundle:
    """把 PageCandidate 按 token 预算装配为 ContextBundle。

    四路预算分配（issue #3）：dense 片段 60% / 整页 20% / 图片 5% / 图谱 15%。
    mode 控制非图片项的展开粒度：
      - "summary"：仅页标题 + 证据片段前 200 字（极省 token）
      - "snippet"：证据 chunk 全文（默认）
      - "full"   ：读取命中页面完整内容（问数值/流程/对比时用）

    Args:
        token_counter: 可选 callable(text)->int；缺省用 char//4 估计。
        citation_root: 权威 vault 根目录（生产链路即 ``hybrid_search(wiki_dir=)``）。
            仅用于把 ``ContextItem.path`` 规范化为 ``Wiki/xxx.md`` 引用形态；
            读盘仍走 ``PageCandidate.path`` 原值。
    """
    bundle = ContextBundle(query="", mode=mode, max_context_tokens=max_tokens)
    if token_counter is None:
        def token_counter(t: str) -> int:
            return max(1, len(t) // 4)

    repository = repository or wi or _FallbackContextRepository(candidates)
    requested_scope = scope or ("full_page" if mode == "full" else "chunk")
    # Reservations protect active channels, while capacity belonging to absent
    # channels immediately rejoins the shared pool.
    fractions = {"dense": .60, "page": .20, "image": .05, "graph": .15}
    category_by_id: Dict[str, str] = {}
    for c in candidates:
        path = str(c.path).replace("\\", "/").lower()
        category_by_id[c.page_id] = ("image" if "assets/" in path or path.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
                                      else "graph" if c.graph_paths and c.rrf_score <= 0
                                      else "page" if requested_scope in ("full_page", "full_source")
                                      else "dense")
    active = set(category_by_id.values())
    reserved = {kind: int(max_tokens * fractions[kind]) if kind in active else 0 for kind in fractions}
    used = 0
    used_by_type = {kind: 0 for kind in fractions}
    deferred = []
    included_order = {}

    def admit(order, c, type_key, item_scope, evidence, text, item_sources, allowed):
        nonlocal used
        selected_text, truncated, omitted_ranges = _truncate_to_budget(
            text, allowed, token_counter)
        tc = token_counter(selected_text) if selected_text else 0
        if not selected_text or tc <= 0:
            return False
        sources = list(getattr(repository, "get_page_sources", lambda _pid: [])(c.page_id) or [])
        item = ContextItem(
            page_id=c.page_id, path=_citation_path(c.path, citation_root), title=c.title,
            inclusion_reason=("graph_expansion" if type_key == "graph"
                              else ("image" if type_key == "image" else "rrf")),
            scope=item_scope, evidence=evidence,
            text=selected_text, sources=(item_sources or sources), graph_paths=c.graph_paths,
            token_count=tc,
            truncated=truncated,
            truncation_reason=("full_page_token_limit"
                               if truncated and item_scope in ("full_page", "full_source")
                               else "token_limit" if truncated else None),
            omitted_ranges=omitted_ranges,
        )
        bundle.items.append(item)
        included_order[id(item)] = order
        used += tc
        used_by_type[type_key] += tc
        return True

    for order, c in enumerate(candidates):
        type_key = category_by_id[c.page_id]
        is_image = type_key == "image"
        is_graph = type_key == "graph"
        item_scope = "chunk" if type_key in ("image", "graph") else requested_scope
        evidence = _all_evidence(c)
        text = _repository_text(repository, c, evidence, item_scope)
        if not text:
            text = _unique_text(evidence)
        if mode == "summary" and item_scope == "chunk":
            text = text[:200]

        # issue #12 多模态：图片命中回溯父文档/页码/section/附近正文
        item_sources: List[str] = []
        if is_image and repository is not None:
            _meta = getattr(repository, "get_image_meta", lambda p: None)(c.path)
            if _meta:
                _src_line = f"[来源: {_meta.get('source_doc') or '?'}"
                if _meta.get("source_page") is not None:
                    _src_line += f", 页 {_meta['source_page']}"
                _sec = _meta.get("source_section")
                if _sec:
                    _src_line += (f", section {'/'.join(_sec)}"
                                  if isinstance(_sec, (list, tuple)) else f", section {_sec}")
                _src_line += "]"
                _nearby = (_meta.get("nearby_text") or "").strip()
                if _nearby:
                    text = (text or "") + f"\n\n{_src_line}\n[附近正文] {_nearby}"
                else:
                    text = (text or "") + f"\n\n{_src_line}\n[注: 该图片附近无可用正文上下文]"
                if _meta.get("source_doc"):
                    item_sources = [_meta["source_doc"]]
            else:
                text = (text or "") + "\n\n[注: 该图片无父文档元数据]"

        tc = token_counter(text)
        # A channel may consume its own reservation and then the capacity of
        # inactive channels. Never borrow a reservation that an active later
        # channel still needs.
        protected_remaining = sum(max(0, reserved[k] - used_by_type[k]) for k in active if k != type_key)
        channel_limit = max_tokens if active == {type_key} else reserved[type_key]
        allowed = min(max(0, channel_limit - used_by_type[type_key]),
                      max(0, max_tokens - used - protected_remaining))
        if allowed <= 0 or tc > allowed:
            deferred.append((order, c, type_key, item_scope, evidence, text, item_sources))
            continue
        admit(order, c, type_key, item_scope, evidence, text, item_sources, allowed)

    # Once every active channel has had its reserved opportunity, all remaining
    # global capacity is shared. Revisit candidates that did not fit their
    # minimum reservation without weakening the hard cap.
    for order, c, type_key, item_scope, evidence, text, item_sources in deferred:
        allowed = max(0, max_tokens - used)
        if allowed and admit(order, c, type_key, item_scope, evidence, text, item_sources, allowed):
            continue
        bundle.omitted_items.append({
            "page_id": c.page_id, "title": c.title,
            "reason": f"{type_key}_budget_exhausted",
        })

    bundle.items.sort(key=lambda item: included_order[id(item)])
    bundle.token_count = used
    bundle.context_text = render_context_markdown(bundle)
    return bundle


def render_context_markdown(bundle: ContextBundle) -> str:
    """把 ContextBundle 渲染为 agent 可读的 markdown（取代旧 format_for_agent）。"""
    if not bundle.items:
        return "[无检索结果]"
    label = {"summary": "概要", "snippet": "片段", "full": "全文"}.get(bundle.mode, bundle.mode)
    lines = [f"## 检索结果（hybrid FTS+RAG，{label}模式，{bundle.token_count}/{bundle.max_context_tokens} tokens）\n"]
    for i, item in enumerate(bundle.items, 1):
        lines.append(f"### [{i}] {item.title}")
        lines.append(f"- 路径: {item.path}")
        lines.append(f"- 引用: [来源: {item.path}]")
        lines.append(f"- 纳入原因: {item.inclusion_reason} | 范围: {item.scope} | tokens: {item.token_count}")
        lines.append(f"- 页面 ID: {item.page_id}")
        lines.append(f"- 来源: {', '.join(item.sources) if item.sources else '无'}")
        lines.append(f"- Evidence IDs: {', '.join(e.chunk_id for e in item.evidence) if item.evidence else '无'}")
        if item.truncated:
            lines.append(f"- 截断: {item.truncation_reason}; omitted={item.omitted_ranges}")
        if item.graph_paths:
            g = "; ".join(f"{p.edge_type}({p.weight:.2f}) signals={p.edge_signals}" for p in item.graph_paths)
            lines.append(f"- 图谱路径: {g}")
        lines.append(f"- 内容:\n```\n{item.text}\n```")
        lines.append("")
    if bundle.omitted_items:
        lines.append(f"### 已省略（预算耗尽）: {len(bundle.omitted_items)} 项")
    return "\n".join(lines)
