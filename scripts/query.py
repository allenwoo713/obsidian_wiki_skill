"""Hybrid FTS+RAG 检索（Retrieval v2，GitHub issues #3/#4/#5/#6）。

流程：原始问题 → Query Planner（issue #6 独立模块）生成通道专用 QueryPlan
→ chunk 级 FTS（lexical+exact 词项）+ 向量（多 semantic query 融合）
→ page-level RRF 融合 → 图谱 1-hop 实体扩展 → 按 token 预算装配 ContextBundle。

关键原则（issue #6）：
- 位置参数始终接收用户**原始问题**，永不在 agent 层手工改写；
- 最终回答 LLM 收到 original_query + QueryPlan + ContextBundle；
- hook / agent 不得自行构造增强查询。
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

from models import PageCandidate, ContextBundle, ContextItem, GraphPath, ChunkHit
from fusion import page_level_rrf, assemble_context, render_context_markdown
from query_planner import DefaultQueryPlanner
from query_plan_models import (
    QueryPlan, PlannerContext, RetrievalFeedback, QueryIntent,
)

logger = logging.getLogger(__name__)

# planner.context_mode → (ContextBundle mode, token 预算倍数)
_CONTEXT_MODE_MAP = {
    "section": ("snippet", 1.0),
    "parent_section": ("snippet", 1.0),
    "multiple_sections": ("snippet", 1.4),
    "evidence": ("snippet", 1.0),
    "chunk": ("snippet", 1.0),
    "global": ("summary", 1.0),
}


@dataclass
class HybridResult:
    query: str
    bundle: ContextBundle
    plan: QueryPlan
    candidates: List[PageCandidate] = field(default_factory=list)
    text_items: List[ContextItem] = field(default_factory=list)
    image_items: List[ContextItem] = field(default_factory=list)


def _split_text_image(items: List[ContextItem]):
    text, images = [], []
    for it in items:
        ps = str(it.path).replace("\\", "/")
        if "assets/" in ps or it.page_id.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
            images.append(it)
        else:
            text.append(it)
    return text, images


def _dedup_chunk_hits(hits: List[ChunkHit]) -> List[ChunkHit]:
    """按 chunk_id 合并多语义 query 的向量命中，保留最佳 score。"""
    best: Dict[str, ChunkHit] = {}
    for h in hits:
        cur = best.get(h.chunk_id)
        if cur is None or h.score > cur.score:
            best[h.chunk_id] = h
    return list(best.values())


def _resolve_entity_seeds(data: dict, entities: tuple) -> List[str]:
    """把 Query Planner 的实体字符串解析为图谱节点 page_id。"""
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    seeds: List[str] = []
    for ent in entities:
        el = ent.lower()
        # 精确匹配 id / title
        for nid, n in nodes.items():
            title = str(n.get("title", "")).lower()
            if nid.lower() == el or title == el:
                seeds.append(nid)
                break
        else:
            # 子串匹配（保守，避免误命中）
            for nid, n in nodes.items():
                title = str(n.get("title", "")).lower()
                if el and (el in nid.lower() or el in title):
                    seeds.append(nid)
                    break
    return seeds


def graph_expand(wi, seed_page_ids: List[str], wiki_dir: Path, k: int = 10,
                 hop: int = 1) -> List[PageCandidate]:
    """图谱 1-hop 扩展（issue #5）：从 seed page_id 出发找邻居。

    节点用 page_id（规范化绝对路径）精确匹配；默认 1-hop。图谱结果作为独立通道
    返回 PageCandidate（带 graph_paths），由上层合并，不进主 RRF，避免噪声挤占 top。
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

    expanded: List[PageCandidate] = []
    seen = set(seed_page_ids)
    frontier = list(seed_page_ids)
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


def _global_retrieve(wi, plan: QueryPlan, k: int, max_tokens: int) -> Optional[HybridResult]:
    """issue #10 占位：global intent 路由到 community reports（若已构建）。"""
    cr_file = wi.index_dir / "community_reports.jsonl"
    if not cr_file.exists():
        return None
    try:
        lines = [json.loads(l) for l in cr_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (json.JSONDecodeError, OSError):
        return None
    items = []
    used = 0
    for rep in lines[:k]:
        text = rep.get("summary") or rep.get("content") or ""
        title = rep.get("title") or rep.get("id") or "community_report"
        tc = max(1, len(text) // 4)
        if used + tc > max_tokens:
            break
        items.append(ContextItem(
            page_id=rep.get("id", title), path=str(rep.get("id", title)), title=title,
            inclusion_reason="global_community_report", scope="full_page",
            evidence=[], text=text, sources=[], graph_paths=[], token_count=tc))
        used += tc
    bundle = ContextBundle(query=plan.original_query, mode="summary",
                           max_context_tokens=max_tokens, items=items, token_count=used)
    bundle.context_text = "\n\n".join(f"### {i.title}\n{i.text}" for i in items)
    return HybridResult(query=plan.original_query, bundle=bundle, plan=plan,
                        text_items=items, image_items=[])


def hybrid_search(wi, original_query: str, planner: DefaultQueryPlanner,
                  context: Optional[PlannerContext] = None,
                  k: int = 5, max_tokens: int = 4096, wiki_dir: Optional[Path] = None,
                  intent_override: str = "auto", rewrite_override: str = "auto",
                  mode_override: Optional[str] = None) -> HybridResult:
    ctx = context or PlannerContext()
    plan = planner.plan(original_query, ctx)
    if intent_override not in (None, "auto"):
        from dataclasses import replace
        plan = replace(plan, intent=intent_override,
                       routing_reason=plan.routing_reason + f"|override={intent_override}")
    if rewrite_override not in (None, "auto"):
        # 仅在 off/force 语义与当前不同步时提示；实际 rewrite 已在 plan() 内按 config 决定
        logger.info("rewrite_override=%s (effective config: planner.config['rewrite']=%s)",
                    rewrite_override, planner.config["rewrite"])

    # global intent → issue #10 路由（占位）
    if plan.intent == QueryIntent.GLOBAL.value:
        gr = _global_retrieve(wi, plan, k, max_tokens)
        if gr is not None:
            return gr
        logger.warning("global intent 但 community reports 未构建 (#10)；回退本地检索")

    # 1) FTS：planner 通道专用词项
    fts_hits = wi.search_fts_terms(plan.lexical_terms, plan.exact_terms, k=20)
    # 2) 向量：多 semantic query 融合
    vec_hits: List[ChunkHit] = []
    for sq in plan.semantic_queries:
        vec_hits.extend(wi.search_vector(sq, k=20))
    vec_hits = _dedup_chunk_hits(vec_hits)

    # 3) page-level RRF
    candidates = page_level_rrf(fts_hits, vec_hits, k=k * 3)
    top_ids = [c.page_id for c in candidates[:5]]

    # 4) 图谱：由 planner 的 entities 解析 seed 后 1-hop 扩展（不在主 RRF 内）
    graph_cands: List[PageCandidate] = []
    if wiki_dir and plan.entities:
        try:
            import json as _json
            gdata = _json.loads((wiki_dir.parent / ".index" / "graph.json").read_text(encoding="utf-8"))
            seeds = _resolve_entity_seeds(gdata, plan.entities)
            if seeds:
                graph_cands = graph_expand(wi, seeds, wiki_dir, k=10, hop=1)
        except (OSError, _json.JSONDecodeError) as e:
            logger.warning("图谱实体扩展失败: %s", e)

    # 5) 合并（图谱候选去重追加）
    by_id = {c.page_id: c for c in candidates}
    for gc in graph_cands:
        if gc.page_id not in by_id:
            by_id[gc.page_id] = gc
    merged = list(by_id.values())
    merged.sort(key=lambda c: (c.rrf_score <= 0, -c.rrf_score))

    # 6) 低召回重试（最多 1 次，issue #6）
    sparse_n, dense_n, ev_n = len(fts_hits), len(vec_hits), len(candidates)
    if (sparse_n == 0 and dense_n == 0) or ev_n == 0:
        feedback = RetrievalFeedback(sparse_hit_count=sparse_n, dense_hit_count=dense_n,
                                     top_score_gap=None, evidence_count=ev_n,
                                     failure_reason="low_recall")
        plan2 = planner.plan_retry(plan, feedback, ctx)
        if plan2 is not None:
            fts_hits2 = wi.search_fts_terms(plan2.lexical_terms, plan2.exact_terms, k=20)
            vec_hits2: List[ChunkHit] = []
            for sq in plan2.semantic_queries:
                vec_hits2.extend(wi.search_vector(sq, k=20))
            vec_hits2 = _dedup_chunk_hits(vec_hits2)
            candidates2 = page_level_rrf(fts_hits2, vec_hits2, k=k * 3)
            by_id2 = {c.page_id: c for c in candidates2}
            for gc in graph_cands:
                if gc.page_id not in by_id2:
                    by_id2[gc.page_id] = gc
            merged = list(by_id2.values())
            merged.sort(key=lambda c: (c.rrf_score <= 0, -c.rrf_score))
            plan = plan2

    # 7) 按 token 预算装配 ContextBundle（context_mode → mode/倍数）
    mode, mult = _CONTEXT_MODE_MAP.get(plan.context_mode, ("snippet", 1.0))
    if mode_override:
        mode = mode_override
        mult = 1.0
    eff_tokens = int(max_tokens * mult)
    bundle = assemble_context(merged, wi, mode=mode, max_tokens=eff_tokens,
                              token_counter=wi.count_tokens)
    text_items, image_items = _split_text_image(bundle.items)
    return HybridResult(query=original_query, bundle=bundle, plan=plan,
                        candidates=merged, text_items=text_items, image_items=image_items)


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
        "query_plan": result.plan.to_json(),
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
        description="Hybrid FTS+RAG 检索：Query Planner → 分层分块 + LanceDB FTS + 自适应向量 + 图谱 → RRF → ContextBundle",
    )
    p.add_argument("project_root", help="知识库项目根目录（含 Wiki/ 与 .index/）")
    p.add_argument("query", help="用户本轮原始问题（原样传入，禁止调用前改写/拼接关键词）")
    p.add_argument("--k", type=int, default=5, help="返回 top-K 页面（默认 5）")
    p.add_argument("--max-tokens", type=int, default=4096, help="ContextBundle token 预算上限（默认 4096）")
    p.add_argument("--mode", choices=["summary", "snippet", "full"], default=None,
                   help="展开粒度覆盖（默认由 QueryPlanner.context_mode 决定）：summary/snippet/full")
    p.add_argument("--intent", default="auto",
                   choices=["auto", "lookup", "procedure", "comparison", "relation", "global"],
                   help="意图覆盖（默认 auto=由 Planner 识别）")
    p.add_argument("--rewrite", default="auto", choices=["auto", "off", "force"],
                   help="LLM rewrite 策略覆盖（默认 auto）")
    p.add_argument("--conversation-context", default=None,
                   help="多轮对话的最小必要上下文（只消解指代，不替代原始问题）")
    p.add_argument("--conversation-context-file", default=None,
                   help="上下文文件（JSON 或纯文本）；JSON 可含 conversation_text/known_entities 等")
    p.add_argument("--json", dest="as_json", action="store_true", help="输出 JSON 格式（含完整 query_plan）")
    p.add_argument("--out", dest="out_path", default=None, help="输出落盘路径（大输出必须用，绕开沙箱 stdout 拦截段错误）")
    return p


def _load_context(args) -> PlannerContext:
    if args.conversation_context_file:
        fp = Path(args.conversation_context_file)
        text = fp.read_text(encoding="utf-8", errors="replace")
        try:
            d = json.loads(text)
            return PlannerContext(
                conversation_text=d.get("conversation_text"),
                domain_terms=tuple(d.get("domain_terms", [])),
                known_entities=tuple(d.get("known_entities", [])),
                page_types=tuple(d.get("page_types", [])),
                language_hints=tuple(d.get("language_hints", [])),
            )
        except (json.JSONDecodeError, TypeError):
            return PlannerContext(conversation_text=text)
    if args.conversation_context:
        return PlannerContext(conversation_text=args.conversation_context)
    return PlannerContext()


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()
    proj = Path(args.project_root)

    # rewrite 策略覆盖：注入到 planner 的 config（仅当显式给出时）
    planner_config = {}
    if args.rewrite != "auto":
        planner_config["rewrite"] = args.rewrite

    from build_index import WikiIndex
    wi = WikiIndex(proj / ".index")
    wi.load()
    planner = DefaultQueryPlanner(project_root=proj, config=planner_config or None)
    ctx = _load_context(args)

    result = hybrid_search(
        wi, args.query, planner, ctx,
        k=args.k, max_tokens=args.max_tokens,
        wiki_dir=proj / "Wiki",
        intent_override=args.intent,
        rewrite_override=args.rewrite,
        mode_override=args.mode,
    )
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
