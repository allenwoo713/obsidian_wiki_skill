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
from pathlib import Path
from typing import List, Optional

from models import (
    ChunkHit, PageCandidate, ContextItem, ContextBundle, EvidenceHit, GraphPath,
)

_FM_RE = re.compile(r"^---\n.*?\n---\n(.*)$", re.DOTALL)


def _read_full_content(path: Path, max_chars: int = 8000) -> str:
    """读取 wiki 页面完整内容（去 frontmatter），截断到 max_chars。"""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = _FM_RE.match(raw)
    content = m.group(1).strip() if m else raw
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[截断，完整内容见原文件]"
    return content


def page_level_rrf(fts_hits: List[ChunkHit], vector_hits: List[ChunkHit],
                   k: int = 5, k_rrf: int = 60) -> List[PageCandidate]:
    """Page-level RRF 融合。

    FTS 与向量两路各自按 score 排序（rank 从 1），命中按 page_id 归并，
    每页的 RRF 分 = Σ 1/(k_rrf + rank)。每页保留各路最佳 chunk 作为证据。
    图谱不在主 RRF 内——由调用方作为独立通道追加，避免噪声挤占 top。
    """
    # 归并：收集每个 page 在两路中的最佳 rank + 最佳 chunk 文本
    pages: dict = {}
    for channel_hits in (fts_hits, vector_hits):
        for rank, h in enumerate(channel_hits, 1):
            pid = h.page_id
            entry = pages.get(pid)
            if entry is None:
                entry = {
                    "page_id": pid, "path": h.path, "title": h.title,
                    "page_type": h.page_type, "fts_rank": None, "vec_rank": None,
                    "fts_hit": None, "vec_hit": None, "rrf": 0.0,
                }
                pages[pid] = entry
            entry["rrf"] += 1.0 / (k_rrf + rank)
            if channel_hits is fts_hits:
                if entry["fts_rank"] is None or rank < entry["fts_rank"]:
                    entry["fts_rank"] = rank
                    entry["fts_hit"] = h
            else:
                if entry["vec_rank"] is None or rank < entry["vec_rank"]:
                    entry["vec_rank"] = rank
                    entry["vec_hit"] = h

    candidates: List[PageCandidate] = []
    for pid, e in pages.items():
        sparse_ev: List[EvidenceHit] = []
        dense_ev: List[EvidenceHit] = []
        if e["fts_hit"]:
            h = e["fts_hit"]
            sparse_ev.append(EvidenceHit(
                chunk_id=h.chunk_id, channel="sparse", rank=e["fts_rank"],
                raw_score=h.score, text=h.text, section_path=h.section_path))
        if e["vec_hit"]:
            h = e["vec_hit"]
            dense_ev.append(EvidenceHit(
                chunk_id=h.chunk_id, channel="dense", rank=e["vec_rank"],
                raw_score=h.score, text=h.text, section_path=h.section_path))
        candidates.append(PageCandidate(
            page_id=pid, path=Path(e["path"]), title=e["title"],
            rrf_score=e["rrf"], sparse_rank=e["fts_rank"], dense_rank=e["vec_rank"],
            sparse_evidence=sparse_ev, dense_evidence=dense_ev,
        ))
    candidates.sort(key=lambda c: -c.rrf_score)
    return candidates[:k]


def _best_evidence_text(c: PageCandidate, max_chars: int = 1200) -> str:
    """取该页最佳 dense 证据 chunk 文本（无则 sparse）。"""
    ev = c.dense_evidence or c.sparse_evidence
    if not ev:
        return ""
    # 多证据按 rank 升序拼接（已在 candidate 内按 rank 排），控制总长度
    parts = []
    used = 0
    for e in sorted(ev, key=lambda x: x.rank or 0):
        t = e.text.strip()
        if used + len(t) > max_chars:
            t = t[: max(0, max_chars - used)]
        if not t:
            break
        parts.append(t)
        used += len(t)
        if used >= max_chars:
            break
    return "\n\n".join(parts)


def assemble_context(
    candidates: List[PageCandidate],
    wi=None,
    *,
    mode: str = "snippet",
    max_tokens: int = 4096,
    token_counter=None,
) -> ContextBundle:
    """把 PageCandidate 按 token 预算装配为 ContextBundle。

    四路预算分配（issue #3）：dense 片段 60% / 整页 20% / 图片 5% / 图谱 15%。
    mode 控制非图片项的展开粒度：
      - "summary"：仅页标题 + 证据片段前 200 字（极省 token）
      - "snippet"：证据 chunk 全文（默认）
      - "full"   ：读取命中页面完整内容（问数值/流程/对比时用）

    Args:
        token_counter: 可选 callable(text)->int；缺省用 char//4 估计。
    """
    bundle = ContextBundle(query="", mode=mode, max_context_tokens=max_tokens)
    if token_counter is None:
        def token_counter(t: str) -> int:
            return max(1, len(t) // 4)

    dense_budget = int(max_tokens * 0.60)
    page_budget = int(max_tokens * 0.20)
    image_budget = int(max_tokens * 0.05)
    graph_budget = max(0, max_tokens - dense_budget - page_budget - image_budget)

    used = 0
    used_by_type = {"dense": 0, "page": 0, "image": 0, "graph": 0}
    for c in candidates:
        _p = str(c.path).replace("\\", "/").lower()
        is_image = ("assets/" in _p) or _p.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
        is_graph = bool(c.graph_paths)
        if is_image:
            budget, scope, type_key = image_budget, "chunk", "image"
        elif is_graph:
            budget, scope, type_key = graph_budget, "chunk", "graph"
        elif mode == "full":
            budget, scope, type_key = page_budget, "full_page", "page"
        else:
            budget, scope, type_key = dense_budget, "chunk", "dense"

        # 渲染文本
        if scope == "full_page":
            text = _read_full_content(c.path)
        else:
            text = _best_evidence_text(c)
            if mode == "summary":
                # 仅保留前 200 字作为概要
                text = text[:200]

        # issue #12 多模态：图片命中回溯父文档/页码/section/附近正文
        item_sources: List[str] = []
        if is_image and wi is not None:
            _meta = getattr(wi, "get_image_meta", lambda p: None)(c.path)
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
        # 四路预算强制（issue #3）：每类有独立预算，超出该类预算即省略，
        # 避免某一路（如 dense）挤占 image/graph/page 的配额。
        if used_by_type[type_key] + tc > budget:
            bundle.omitted_items.append({
                "page_id": c.page_id, "title": c.title,
                "reason": f"{type_key}_budget_exhausted",
            })
            continue
        # 全局硬上限（安全兜底；正常情况下四路预算之和 = max_tokens 已覆盖）
        if used + tc > max_tokens:
            bundle.omitted_items.append({
                "page_id": c.page_id, "title": c.title,
                "reason": "budget_exhausted",
            })
            continue
        item = ContextItem(
            page_id=c.page_id, path=str(c.path), title=c.title,
            inclusion_reason=("graph_expansion" if is_graph else ("image" if is_image else "rrf")),
            scope=scope, evidence=c.dense_evidence,
            text=text, sources=item_sources, graph_paths=c.graph_paths,
            token_count=tc,
        )
        bundle.items.append(item)
        used += tc
        used_by_type[type_key] += tc

    bundle.token_count = used
    bundle.context_text = "\n\n".join(
        f"### {i.title}\n{i.text}" for i in bundle.items
    )
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
        lines.append(f"- 纳入原因: {item.inclusion_reason} | 范围: {item.scope} | tokens: {item.token_count}")
        if item.graph_paths:
            g = "; ".join(f"{p.edge_type}({p.weight:.2f})" for p in item.graph_paths)
            lines.append(f"- 图谱路径: {g}")
        lines.append(f"- 内容:\n```\n{item.text}\n```")
        lines.append("")
    if bundle.omitted_items:
        lines.append(f"### 已省略（预算耗尽）: {len(bundle.omitted_items)} 项")
    return "\n".join(lines)
