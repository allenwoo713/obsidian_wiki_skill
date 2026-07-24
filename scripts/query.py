"""Hybrid FTS+RAG 检索（Retrieval v2，GitHub issues #3/#4/#5）。

流程：chunk 级 FTS + 向量 → page-level RRF 融合 → 图谱 1-hop 独立扩展 →
按 token 预算装配 ContextBundle（直接可喂 LLM）。答案须标注出处 [来源: ...]。

用法：
    python query.py <project_root> "<query>" [--k 5] [--max-tokens 4096]
                     [--mode snippet|summary|full] [--json] [--out FILE]
"""
from __future__ import annotations
import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import _config  # noqa: F401  # 加载 <skill_dir>/.env（ISSUE-01）

from models import PageCandidate, ContextBundle, ContextItem, GraphPath
from fusion import assemble_context, render_context_markdown

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    query: str
    bundle: ContextBundle
    candidates: List[PageCandidate] = field(default_factory=list)
    text_items: List[ContextItem] = field(default_factory=list)
    image_items: List[ContextItem] = field(default_factory=list)


def _split_text_image(items: List[ContextItem]):
    text, images = [], []
    for it in items:
        ps = str(it.path).replace("\\", "/")
        if "assets/" in ps or it.page_id.endswith(".png") or it.page_id.endswith(".jpg"):
            images.append(it)
        else:
            text.append(it)
    return text, images


def graph_expand(wi, top_page_ids: List[str], wiki_dir: Path, k: int = 10,
                 hop: int = 1) -> List[PageCandidate]:
    """图谱 1-hop 扩展（issue #5）：从 top 命中的 page_id 出发找邻居。

    节点用 page_id（规范化绝对路径）精确匹配；默认 1-hop（hop=1）。图谱结果
    作为独立通道返回 PageCandidate（带 graph_paths），由上层合并，不进主 RRF，
    避免噪声挤占 top（ISSUE-16 图谱降权思路的演进：彻底移出主 RRF）。
    """
    idx_file = wiki_dir.parent / ".index" / "graph.json"
    if not idx_file.exists():
        return []
    try:
        data = json.loads(idx_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("graph_expand: graph.json 解析失败: %s", e)
        return []
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    edges = data.get("edges", [])
    neighbors: Dict[str, List[tuple]] = {}
    for e in edges:
        s, t = e.get("source"), e.get("target")
        w = float(e.get("weight", 1.0))
        etype = e.get("signal") or e.get("type") or "unknown"
        is_inf = "inferred" in str(etype).lower() or etype in ("adamic_adar", "type_affinity")
        neighbors.setdefault(s, []).append((t, w, etype, is_inf))
        neighbors.setdefault(t, []).append((s, w, etype, is_inf))

    seeds = set(top_page_ids)
    expanded: List[PageCandidate] = []
    seen = set(seeds)
    frontier = list(seeds)
    for h in range(hop):
        nxt = []
        for pid in frontier:
            for (nbr, w, etype, is_inf) in neighbors.get(pid, []):
                if nbr in seen:
                    continue
                seen.add(nbr)
                n = nodes.get(nbr, {})
                gp = GraphPath(source_id=pid, target_id=nbr, edge_type=etype,
                                is_inferred=is_inf, weight=w, hop=h + 1)
                expanded.append(PageCandidate(
                    page_id=nbr,
                    path=Path(n.get("path", nbr)),
                    title=n.get("title", nbr),
                    rrf_score=0.0,
                    sparse_rank=None,
                    dense_rank=None,
                    graph_paths=[gp],
                ))
                nxt.append(nbr)
                if len(expanded) >= k:
                    break
            if len(expanded) >= k:
                break
        frontier = nxt
        if len(expanded) >= k:
            break
    return expanded[:k]


def hybrid_search(wi, query: str, k: int = 5, max_tokens: int = 4096,
                  wiki_dir: Path = None, mode: str = "snippet") -> HybridResult:
    # 1) chunk 级 FTS + 向量 → page-level RRF
    candidates = wi.search(query, k=k * 3)
    top_ids = [c.page_id for c in candidates[:5]]

    # 2) 图谱 1-hop 独立扩展（不在主 RRF 内）
    graph_cands: List[PageCandidate] = []
    if wiki_dir:
        graph_cands = graph_expand(wi, top_ids, wiki_dir, k=10, hop=1)

    # 3) 合并（图谱候选去重追加）
    by_id = {c.page_id: c for c in candidates}
    for gc in graph_cands:
        if gc.page_id not in by_id:
            by_id[gc.page_id] = gc
    merged = list(by_id.values())
    # 保持 RRF 候选在前、图谱候选在后
    merged.sort(key=lambda c: (c.rrf_score <= 0, -c.rrf_score))

    # 4) 按 token 预算装配 ContextBundle
    bundle = assemble_context(merged, wi, mode=mode, max_tokens=max_tokens,
                              token_counter=wi.count_tokens)
    text_items, image_items = _split_text_image(bundle.items)
    return HybridResult(query=query, bundle=bundle, candidates=merged,
                        text_items=text_items, image_items=image_items)


def format_for_agent(result: HybridResult) -> str:
    """markdown 渲染（替代旧 format_for_agent）。"""
    return render_context_markdown(result.bundle)


def _rrf_score_by_id(candidates: List[PageCandidate]) -> Dict[str, float]:
    return {c.page_id: c.rrf_score for c in candidates}


def result_to_json(result: HybridResult) -> dict:
    rrf = _rrf_score_by_id(result.candidates)

    def item_entry(it: ContextItem):
        ps = str(it.path).replace("\\", "/")
        return {
            "page_id": it.page_id,
            "path": str(it.path),
            "title": it.title,
            "score": round(rrf.get(it.page_id, 0.0), 6),
            "snippet": it.text,
            "sources": it.sources,
            "method": it.inclusion_reason,
            "scope": it.scope,
            "tokens": it.token_count,
            "embed": f"![[{Path(it.path).name}]]" if "assets/" in ps else None,
        }

    return {
        "query": result.query,
        "mode": result.bundle.mode,
        "token_count": result.bundle.token_count,
        "max_context_tokens": result.bundle.max_context_tokens,
        "text": [item_entry(it) for it in result.text_items],
        "images": [item_entry(it) for it in result.image_items],
        "omitted": result.bundle.omitted_items,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="query.py",
        description="Hybrid FTS+RAG 检索：分层分块 + LanceDB FTS + 自适应向量 + 图谱 → RRF → ContextBundle",
    )
    p.add_argument("project_root", help="知识库项目根目录（含 Wiki/ 与 .index/）")
    p.add_argument("query", help="检索查询串（建议先做关键词提取与中英互译扩展）")
    p.add_argument("--k", type=int, default=5, help="返回 top-K 页面（默认 5）")
    p.add_argument("--max-tokens", type=int, default=4096, help="ContextBundle token 预算上限（默认 4096）")
    p.add_argument("--mode", choices=["summary", "snippet", "full"], default="snippet",
                   help="展开粒度：summary=概要前200字 / snippet=证据chunk（默认） / full=命中页全文")
    p.add_argument("--json", dest="as_json", action="store_true", help="输出 JSON 格式（默认输出 markdown）")
    p.add_argument("--out", dest="out_path", default=None, help="输出落盘路径（大输出必须用，绕开沙箱 stdout 拦截段错误）")
    return p


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()
    proj = Path(args.project_root)
    from build_index import WikiIndex
    wi = WikiIndex(proj / ".index")
    wi.load()
    result = hybrid_search(wi, args.query, k=args.k, max_tokens=args.max_tokens,
                           wiki_dir=proj / "Wiki", mode=args.mode)
    payload = (json.dumps(result_to_json(result), ensure_ascii=False, indent=2)
               if args.as_json else format_for_agent(result))
    if args.out_path:
        op = Path(args.out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(payload, encoding="utf-8")
        print(f"wrote {op} ({len(payload)} bytes)")
    else:
        print(payload)


if __name__ == "__main__":
    main()
