"""obsidian_wiki_skill 检索评测（issue #9）。

用法：
    python eval/run_eval.py                              # 用 tests/fixtures 构建并评测，对比 baselines.json
    python eval/run_eval.py --work-dir D:/tmp/eval       # 指定构建临时目录（避免 C: 沙箱虚拟化）
    python eval/run_eval.py --init-baseline              # 首次/契约变更后：把当前指标写入 baselines.json（含 chunk_schema_version）并退出 0
    python eval/run_eval.py --force-compare              # 忽略 chunk_schema_version 不匹配，强制对比旧基线（仅本地调试，CI 禁用）

指标：
    质量：Page Recall@5, Evidence Recall@10, Exact lookup Hit@3, MRR@10,
          ANN Recall@10, Context overflow count, Graph-only unsupported evidence count
    性能：全量构建时间, 单页增量时间, embedding 数, 索引磁盘大小,
          P50/P95/P99 查询延迟, peak memory, ContextBundle token 数, exact vs ANN 差异

退出码：0=通过；1=回归（Recall 下降>阈值 / overflow>0 / graph-only>0 / ANN<0.98）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL_ROOT = HERE.parent
SCRIPTS = SKILL_ROOT / "scripts"
FIXTURES_WIKI = SKILL_ROOT / "tests" / "fixtures" / "wiki"

sys.path.insert(0, str(SCRIPTS))

from build_index import WikiIndex  # noqa: E402
from query_planner import DefaultQueryPlanner  # noqa: E402
from query import hybrid_search  # noqa: E402
import build_graph as _bg  # noqa: E402
from chunking import CHUNK_SCHEMA_VERSION  # noqa: E402


def _page_hit(candidate, gold_pages):
    pid = candidate.page_id.lower().replace("\\", "/")
    return any(g.lower() in pid for g in gold_pages)


def _stage_project(project_root: Path, wiki_src: Path):
    """把 fixture wiki 复制进 project_root/Wiki，使 .index 与 Wiki 同父目录。"""
    wiki = project_root / "Wiki"
    if project_root.exists():
        shutil.rmtree(project_root)
    shutil.copytree(str(wiki_src), str(wiki))
    return wiki


def _build(project_root: Path, wiki_src: Path, vector_index_mode: str, full_rebuild: bool):
    wiki = _stage_project(project_root, wiki_src)
    idx = project_root / ".index"
    wi = WikiIndex(idx)
    t0 = time.perf_counter()
    wi.build(wiki, full_rebuild=full_rebuild, vector_index_mode=vector_index_mode)
    dt = time.perf_counter() - t0
    wi.load()
    # 生成图谱（供 hybrid_search 的 relations 扩展通道使用）
    try:
        G = _bg.build_graph(wiki)
        stats = _bg.compute_4_signals(G)
        comms = _bg.detect_communities(G)
        graph_json = {
            "nodes": [{"id": n, **{k: v for k, v in d.items() if k != "signals"}}
                      for n, d in G.nodes(data=True)],
            "edges": [{"source": u, "target": v,
                       "weight": round(d.get("weight", 1.0), 4),
                       "signal": sorted(d.get("signals", set()))[0] if d.get("signals") else "unknown",
                       "signals": sorted(d.get("signals", set()))}
                      for u, v, d in G.edges(data=True)],
            "signals": stats,
            "communities": comms,
        }
        (idx / "graph.json").write_text(
            json.dumps(graph_json, ensure_ascii=False, indent=2, default=list), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 图谱生成失败（不影响主评测）: {e}", file=sys.stderr)
    return wi, wiki, dt


def _norm_chunk_key(cid: str) -> str:
    """#13：chunk_id 现为 `page_id::{content_hash}`（content_hash 由正文内容决定，
    与位置/路径无关）。跨 project（不同根路径、相同内容）content_hash 一致，故归一化
    只需取 `::` 之后的 hash 部分即可对齐 exact 与 ANN 两个独立 project 的结果。"""
    try:
        return cid.split("::", 1)[-1]
    except Exception:
        return cid


def _chunk_ids(wi: WikiIndex, query: str, k: int = 10):
    hits = wi.search_vector(query, k=k)
    return [_norm_chunk_key(h.chunk_id) for h in hits[:k]]


def run_evaluation(wiki_src: Path, queries: list, work_dir: Path, max_tokens: int,
                   build_ann: bool, regression_pp: float):
    work_dir.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()

    # 1) 主索引（exact，小库等价精确）
    main_root = work_dir / "main"
    main_wi, main_wiki, build_time = _build(main_root, wiki_src, "exact", full_rebuild=True)
    planner = DefaultQueryPlanner(project_root=main_root)

    # 2) ANN 索引（独立 project，复用语义内容；向量层走 IVF_HNSW_FLAT）
    ann_wi = None
    ann_build_time = None
    if build_ann:
        ann_root = work_dir / "ann"
        ann_wi, _, ann_build_time = _build(ann_root, wiki_src, "ivf-hnsw-flat", full_rebuild=True)

    # 3) 逐查询评测
    page_recalls, evid_recalls, mrrs, exact_hits, ann_recalls = [], [], [], [], []
    latencies = []
    context_overflow = 0
    graph_only_unsupported = 0
    graph_trigger_queries = 0
    graph_trigger_validated = 0
    bundle_tokens = []
    detail = []

    for q in queries:
        gold = q["relevant_pages"]
        plan = planner.plan(q["query"])
        t0 = time.perf_counter()
        res = hybrid_search(main_wi, q["query"], planner, k=10,
                            max_tokens=max_tokens, wiki_dir=main_wiki, intent_override="auto")
        latencies.append(time.perf_counter() - t0)

        cands = res.candidates
        top5 = cands[:5]
        top10 = cands[:10]

        hits5 = sum(1 for c in top5 if _page_hit(c, gold))
        page_recalls.append(min(1.0, hits5 / max(1, len(gold))))

        ctx = res.bundle.context_text or ""
        facts = q.get("required_facts") or []
        found = sum(1 for f in facts if f and f in ctx)
        evid_recalls.append((found / len(facts)) if facts else 1.0)

        rank = None
        for i, c in enumerate(top10, 1):
            if _page_hit(c, gold):
                rank = i
                break
        mrrs.append(1.0 / rank if rank else 0.0)

        if q["intent"] == "lookup":
            h3 = any(_page_hit(c, gold) for c in top10[:3])
            exact_hits.append(1.0 if h3 else 0.0)

        if ann_wi is not None:
            exact_ids = set(_chunk_ids(main_wi, q["query"], 10))
            ann_ids = set(_chunk_ids(ann_wi, q["query"], 10))
            ann_recalls.append(len(exact_ids & ann_ids) / len(exact_ids) if exact_ids else 1.0)

        # Context overflow：最终上下文超出 max_tokens（assemble_context 已截断 → 应恒 0）
        if res.bundle.token_count > max_tokens:
            context_overflow += 1
        if q.get("graph_trigger"):
            graph_trigger_queries += 1
            graph_trigger_validated += sum(
                1 for item in res.bundle.items if item.inclusion_reason == "graph_expansion"
            )
        for it in res.bundle.items:
            if it.inclusion_reason == "graph_expansion" and not it.evidence:
                graph_only_unsupported += 1
        bundle_tokens.append(res.bundle.token_count)

        # 明细（便于调优 gold 标签）
        detail.append({
            "query": q["query"], "intent": q["intent"],
            "gold": gold,
            "top5_hit": [_page_hit(c, gold) for c in top5],
            "page_recall": min(1.0, hits5 / max(1, len(gold))),
            "evidence_recall": (found / len(facts)) if facts else 1.0,
            "mrr": (1.0 / rank) if rank else 0.0,
            "facts_found": [f for f in facts if f and f in ctx],
            "facts_missing": [f for f in facts if f and f not in ctx],
            "top5_pages": [Path(c.page_id).name for c in top5],
        })

    # 4) 性能：增量时间（复用同一 project，单页修改触发增量）
    inc_page = sorted(main_wiki.glob("*.md"))[0]
    original = inc_page.read_text(encoding="utf-8")
    try:
        inc_page.write_text(original + "\n<!-- 增量评测追加一行 -->\n", encoding="utf-8")
        t0 = time.perf_counter()
        main_wi.build(main_wiki, full_rebuild=False)
        inc_time = time.perf_counter() - t0
    finally:
        inc_page.write_text(original, encoding="utf-8")
        # 恢复索引到完整态（重跑确保主索引干净）
        main_wi.build(main_wiki, full_rebuild=False)

    vec_cache = main_root / ".index" / "vec_cache"
    emb_count = sum(1 for _ in vec_cache.rglob("*.npy")) if vec_cache.exists() else 0
    disk_bytes = sum(f.stat().st_size for f in (main_root / ".index").rglob("*") if f.is_file())
    disk_mb = disk_bytes / (1024 * 1024)
    peak_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    def pct(xs, p):
        if not xs:
            return 0.0
        s = sorted(xs)
        k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
        return s[k]

    metrics = {
        "n_queries": len(queries),
        "quality": {
            "page_recall_at_5": round(statistics.mean(page_recalls), 4),
            "evidence_recall_at_10": round(statistics.mean(evid_recalls), 4),
            "exact_lookup_hit_at_3": round(statistics.mean(exact_hits), 4) if exact_hits else None,
            "mrr_at_10": round(statistics.mean(mrrs), 4),
            "ann_recall_at_10": round(statistics.mean(ann_recalls), 4) if ann_recalls else None,
            "context_overflow_count": context_overflow,
            "graph_only_unsupported_count": graph_only_unsupported,
            "graph_trigger_query_count": graph_trigger_queries,
            "graph_trigger_validated_count": graph_trigger_validated,
        },
        "performance": {
            "full_build_time_s": round(build_time, 2),
            "incremental_build_time_s": round(inc_time, 2),
            "embedding_count": emb_count,
            "index_disk_mb": round(disk_mb, 2),
            "latency_p50_s": round(pct(latencies, 50), 4),
            "latency_p95_s": round(pct(latencies, 95), 4),
            "latency_p99_s": round(pct(latencies, 99), 4),
            "peak_memory_mb": round(peak_mb, 1),
            "mean_contextbundle_tokens": round(statistics.mean(bundle_tokens), 1),
            "ann_build_time_s": round(ann_build_time, 2) if ann_build_time else None,
        },
    }
    return metrics, detail


def load_queries(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", type=Path, default=FIXTURES_WIKI)
    ap.add_argument("--queries", type=Path, default=HERE / "queries.jsonl")
    ap.add_argument("--baselines", type=Path, default=HERE / "baselines.json")
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="构建临时目录（默认系统临时目录；本地建议 D: 盘避免 C: 虚拟化）")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--no-ann", action="store_true", help="跳过 ANN 索引构建")
    ap.add_argument("--regression-pp", type=float, default=2.0, help="Recall 下降百分点阈值")
    ap.add_argument("--init-baseline", action="store_true", help="把当前指标写为 baselines.json 并退出 0")
    ap.add_argument("--force-compare", action="store_true",
                    help="忽略 chunk_schema_version 不匹配，强制对比旧基线（仅本地调试用，CI 不应使用）")
    args = ap.parse_args()

    if not args.wiki.exists():
        print(f"[ERROR] fixture wiki 不存在: {args.wiki}", file=sys.stderr)
        return 2
    queries = load_queries(args.queries)
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="obs_wiki_eval_"))
    print(f"[info] work-dir = {work_dir}", file=sys.stderr)

    metrics, detail = run_evaluation(args.wiki, queries, work_dir, args.max_tokens,
                             build_ann=not args.no_ann, regression_pp=args.regression_pp)

    (HERE / "results.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    (HERE / "detail.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in detail), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.init_baseline:
        baseline_obj = dict(metrics)
        baseline_obj["meta"] = {
            "chunk_schema_version": CHUNK_SCHEMA_VERSION,
            "note": "由 run_eval.py --init-baseline 生成。chunking 契约（chunk_id 格式 / 分块策略）"
                    "变更导致指标含义或分布变化时，须重新执行 --init-baseline 重置本基线，"
                    "并在对应 issue / CHANGELOG 说明重置原因；CI 会在 schema 版本不匹配时直接标红。",
        }
        args.baselines.write_text(json.dumps(baseline_obj, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"[info] baselines.json 已初始化: {args.baselines} "
              f"(chunk_schema_version={CHUNK_SCHEMA_VERSION})", file=sys.stderr)
        return 0

    if not args.baselines.exists():
        print("[WARN] baselines.json 不存在，跳过回归检查（用 --init-baseline 初始化）",
              file=sys.stderr)
        return 0

    base = json.loads(args.baselines.read_text(encoding="utf-8"))
    base_meta = base.get("meta", {})
    base_schema = base_meta.get("chunk_schema_version")
    if not args.force_compare and base_schema != CHUNK_SCHEMA_VERSION:
        print(
            f"[FAIL] baselines.json 契约不匹配：基线生成于 chunk_schema_version="
            f"{base_schema}，当前代码为 {CHUNK_SCHEMA_VERSION}。\n"
            f"       chunking 契约已变更，请先执行 "
            f"`python eval/run_eval.py --init-baseline` 重新建立基线\n"
            f"       （并在对应 issue / CHANGELOG 说明重置原因），或加 --force-compare 强制对比旧基线。",
            file=sys.stderr,
        )
        return 1
    bq, bp = base["quality"], base["performance"]
    mq = metrics["quality"]
    failures = []

    def check_drop(name, cur, ref):
        if cur is None or ref is None:
            return
        drop = (ref - cur) * 100
        if drop > args.regression_pp:
            failures.append(f"{name}: {cur:.4f} 较基线 {ref:.4f} 下降 {drop:.2f}pp > {args.regression_pp}pp")

    check_drop("page_recall_at_5", mq["page_recall_at_5"], bq["page_recall_at_5"])
    check_drop("evidence_recall_at_10", mq["evidence_recall_at_10"], bq["evidence_recall_at_10"])
    check_drop("exact_lookup_hit_at_3", mq["exact_lookup_hit_at_3"], bq["exact_lookup_hit_at_3"])
    check_drop("mrr_at_10", mq["mrr_at_10"], bq["mrr_at_10"])
    check_drop("ann_recall_at_10", mq["ann_recall_at_10"], bq["ann_recall_at_10"])

    if mq["context_overflow_count"] > 0:
        failures.append(f"context_overflow_count={mq['context_overflow_count']} > 0")
    if mq["graph_only_unsupported_count"] > 0:
        failures.append(f"graph_only_unsupported_count={mq['graph_only_unsupported_count']} > 0")
    if mq.get("graph_trigger_query_count", 0) <= 0 or mq.get("graph_trigger_validated_count", 0) <= 0:
        failures.append("graph trigger fixture did not produce a validated graph result")
    if mq["ann_recall_at_10"] is not None and mq["ann_recall_at_10"] < 0.98:
        failures.append(f"ann_recall_at_10={mq['ann_recall_at_10']:.4f} < 0.98")

    if failures:
        print("\n[FAIL] 评测未通过：", file=sys.stderr)
        for f in failures:
            print("  - " + f, file=sys.stderr)
        return 1
    print("\n[PASS] 评测通过，无回归。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
