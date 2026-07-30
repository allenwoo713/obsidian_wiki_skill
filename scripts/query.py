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
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# 0xC0000005 修复（与 build_index.py 同源）：torch 必须在任何可能拉起后台 asyncio
# 事件循环线程的导入（如经 build_index 间接引入的 lancedb）之前完成原生模块加载，
# 否则在 PowerShell 启动的 managed-python 下，torch 原生加载会与宿主注入的后台事件
# 循环线程时序 race → 段错误。pyarrow 也须先于 torch（ISSUE-16）。
import pyarrow  # noqa: F401
import torch
torch.set_num_threads(int(os.environ.get("WIKI_TORCH_THREADS", "1") or "1"))
torch.set_grad_enabled(False)

import _config  # noqa: F401  # 加载 <skill_dir>/.env（ISSUE-01）

from models import PageCandidate, ContextBundle, ContextItem, EvidenceHit, GraphPath, ChunkHit
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
    graph_validated_count: int = 0


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
        signals = list(dict.fromkeys(e.get("signals") or [e.get("signal") or e.get("type") or "unknown"]))
        etype = e.get("signal") or e.get("type") or signals[0]
        # A mixed explicit/inferred edge remains explicit. It is inferred-only
        # only when every serialized signal is an inference signal.
        inferred_signals = {"adamic_adar", "type_affinity"}
        is_inf = bool(signals) and all(str(signal).lower() in inferred_signals or "inferred" in str(signal).lower()
                                       for signal in signals)
        neighbors.setdefault(s, []).append((t, w, etype, is_inf, signals))
        neighbors.setdefault(t, []).append((s, w, etype, is_inf, signals))

    expanded: List[PageCandidate] = []
    seen = set(seed_page_ids)
    frontier = list(seed_page_ids)
    for h in range(hop):
        nxt = []
        for pid in frontier:
            for (nbr, w, etype, is_inf, signals) in neighbors.get(pid, []):
                if nbr in seen:
                    continue
                seen.add(nbr)
                n = nodes.get(nbr, {})
                gp = GraphPath(source_id=pid, target_id=nbr, edge_type=etype,
                                is_inferred=is_inf, weight=w, hop=h + 1,
                                edge_signals=signals)
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


def _validate_graph_candidates(wi, graph_candidates: List[PageCandidate],
                               plan: QueryPlan) -> List[PageCandidate]:
    """Admit graph recommendations only after same-page textual validation."""
    validated: List[PageCandidate] = []
    search_page = getattr(wi, "search_page", None)
    if search_page is None:
        logger.warning("graph validation skipped: index lacks restricted retrieval adapter")
        return []
    for candidate in graph_candidates:
        hits = [hit for hit in search_page(candidate.page_id, plan) if hit.page_id == candidate.page_id and hit.text.strip()]
        if not hits:
            continue
        sparse, dense = [], []
        for rank, hit in enumerate(hits, 1):
            evidence = EvidenceHit(hit.chunk_id, "sparse" if hit.channel == "fts" else "dense",
                                  rank, hit.score, hit.text, hit.section_path)
            (sparse if hit.channel == "fts" else dense).append(evidence)
        # Relation answers still retain explicit graph signals, but no signal
        # (including an explicit one) substitutes for supporting page text.
        candidate.sparse_evidence = sparse
        candidate.dense_evidence = dense
        validated.append(candidate)
    return validated


def _retrieve_for_plan(wi, plan: QueryPlan, k: int, wiki_dir: Optional[Path]):
    """One complete retrieval/graph-validation pass, reusable for retries."""
    fts_hits = wi.search_fts_terms(plan.lexical_terms, plan.exact_terms, k=20)
    vec_hits: List[ChunkHit] = []
    for semantic_query in plan.semantic_queries:
        vec_hits.extend(wi.search_vector(semantic_query, k=20))
    vec_hits = _dedup_chunk_hits(vec_hits)
    # Keep enough direct RRF hits to seed graph expansion, while preserving the
    # requested direct-result count as a separate post-RRF channel.
    direct_candidates = page_level_rrf(fts_hits, vec_hits, k=max(k, 5))
    candidates = direct_candidates[:k]

    graph_candidates: List[PageCandidate] = []
    if wiki_dir:
        try:
            graph_file = wiki_dir.parent / ".index" / "graph.json"
            graph_data = json.loads(graph_file.read_text(encoding="utf-8"))
            direct_seeds = [candidate.page_id for candidate in direct_candidates[:5]]
            entity_seeds = _resolve_entity_seeds(graph_data, plan.entities)
            seeds = list(dict.fromkeys(direct_seeds + entity_seeds))
            if seeds:
                graph_candidates = _validate_graph_candidates(
                    wi, graph_expand(wi, seeds, wiki_dir, k=10, hop=1), plan)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("图谱扩展/验证失败: %s", exc)

    by_id = {candidate.page_id: candidate for candidate in candidates}
    for candidate in graph_candidates:
        if candidate.page_id not in by_id:
            by_id[candidate.page_id] = candidate
    merged = list(by_id.values())
    merged.sort(key=lambda candidate: (candidate.rrf_score <= 0, -candidate.rrf_score))
    return fts_hits, vec_hits, candidates, merged, len(graph_candidates)


def _global_retrieve(wi, plan: QueryPlan, k: int, max_tokens: int) -> Optional[HybridResult]:
    """issue #10 GraphRAG global：路由到 community reports（与 local retrieval 分离）。

    map 阶段：按 query 词项与报告文本（summary+key_entities+title）的相关性检索
    相关社区报告；reduce 阶段：按 token 预算聚合为 ContextBundle。
    无 LLM 时为结构化聚合（非自然语言 map/reduce 摘要）；社区报告由
    ``build_community_reports.py`` 构建。
    """
    cr_file = wi.index_dir / "community_reports.jsonl"
    if not cr_file.exists():
        return None
    try:
        lines = [json.loads(l) for l in cr_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (json.JSONDecodeError, OSError):
        return None
    # map：检索相关社区报告（query 词项与报告文本重叠度排序）
    qtokens = set()
    for t in list(plan.lexical_terms) + list(plan.exact_terms) + list(plan.entities):
        if t:
            qtokens.add(t.lower())
    for w in (plan.original_query or "").lower().split():
        if w:
            qtokens.add(w)

    def _rep_score(rep):
        text = ((rep.get("summary") or "") + " "
                + " ".join(rep.get("key_entities") or [])
                + " " + (rep.get("title") or "")).lower()
        return sum(1 for t in qtokens if t in text)

    ranked = sorted(lines, key=lambda r: (-_rep_score(r), r.get("community_id", 0)))
    items = []
    used = 0
    for rep in ranked[:k]:
        text = rep.get("summary") or rep.get("content") or ""
        title = rep.get("title") or rep.get("id") or "community_report"
        tc = max(1, len(text) // 4)
        if used + tc > max_tokens:
            break
        items.append(ContextItem(
            page_id=str(rep.get("community_id", title)), path=str(rep.get("community_id", title)),
            title=title, inclusion_reason="global_community_report", scope="full_page",
            evidence=[], text=text, sources=rep.get("source_pages", []),
            graph_paths=[], token_count=tc))
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

    fts_hits, vec_hits, candidates, merged, graph_validated_count = _retrieve_for_plan(wi, plan, k, wiki_dir)

    # 6) 低召回重试（最多 1 次，issue #6）
    sparse_n, dense_n, ev_n = len(fts_hits), len(vec_hits), len(candidates)
    if (sparse_n == 0 and dense_n == 0) or ev_n == 0:
        feedback = RetrievalFeedback(sparse_hit_count=sparse_n, dense_hit_count=dense_n,
                                     top_score_gap=None, evidence_count=ev_n,
                                     failure_reason="low_recall")
        plan2 = planner.plan_retry(plan, feedback, ctx)
        if plan2 is not None:
            fts_hits, vec_hits, candidates, merged, graph_validated_count = _retrieve_for_plan(wi, plan2, k, wiki_dir)
            plan = plan2

    # 7) 按 token 预算装配 ContextBundle（context_mode → mode/倍数）
    mode, mult = _CONTEXT_MODE_MAP.get(plan.context_mode, ("snippet", 1.0))
    if mode_override:
        mode = mode_override
        mult = 1.0
    eff_tokens = int(max_tokens * mult)
    bundle = assemble_context(merged, repository=wi, mode=mode,
                              scope=("full_page" if mode == "full" else plan.context_mode), max_tokens=eff_tokens,
                              token_counter=wi.count_tokens)
    text_items, image_items = _split_text_image(bundle.items)
    return HybridResult(query=original_query, bundle=bundle, plan=plan,
                        candidates=merged, text_items=text_items, image_items=image_items,
                        graph_validated_count=graph_validated_count)


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
            "evidence": [{
                "chunk_id": hit.chunk_id, "channel": hit.channel, "rank": hit.rank,
                "raw_score": hit.raw_score, "section_path": hit.section_path,
            } for hit in it.evidence],
            "graph_paths": [{
                "source_id": path.source_id, "target_id": path.target_id,
                "edge_type": path.edge_type, "edge_signals": path.edge_signals,
                "is_inferred": path.is_inferred, "weight": path.weight, "hop": path.hop,
            } for path in it.graph_paths],
            "method": it.inclusion_reason,
            "scope": it.scope,
            "tokens": it.token_count,
            "truncated": it.truncated,
            "truncation_reason": it.truncation_reason,
            "omitted_ranges": it.omitted_ranges,
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
