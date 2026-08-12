"""obsidian_wiki_skill 检索评测（issue #9）。

用法：
    python eval/run_eval.py                              # 用 tests/fixtures 构建并评测，对比 baselines.json
    python eval/run_eval.py --work-dir D:/tmp/eval       # 指定构建临时目录（避免 C: 沙箱虚拟化）
    python eval/run_eval.py --init-baseline              # 首次/契约变更后：把当前指标写入 baselines.json（含 chunk_schema_version）并退出 0
    python eval/run_eval.py --force-compare              # 忽略 chunk_schema_version 不匹配，强制对比旧基线（仅本地调试，CI 禁用）

指标：
    质量：Page Recall@5, Evidence Recall@10, Exact lookup Hit@3, MRR@10,
          ANN Recall@10, Context overflow count, Graph-only unsupported evidence count,
          Budget contract violation count
    性能：全量构建时间, 单页增量时间, embedding 数, 索引磁盘大小,
          P50/P95/P99 查询延迟, peak memory, ContextBundle token 数, exact vs ANN 差异
    预算：base/effective 预算分布（按意图）、扩张查询数、最大有效预算

Overflow 判定口径：以 ``bundle.effective_budget_tokens``（hybrid_search 实际分配的
预算）为准，而不是 CLI 传入的基础预算——预算策略由 query.hybrid_search 唯一决定，
本脚本不得复制 ``_CONTEXT_MODE_MAP``。

退出码：0=通过；1=回归（Recall 下降>阈值 / overflow>0 / graph-only>0 / ANN<0.98 /
预算契约漂移）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path, PurePosixPath, PureWindowsPath

HERE = Path(__file__).resolve().parent
SKILL_ROOT = HERE.parent
SCRIPTS = SKILL_ROOT / "scripts"
FIXTURES_WIKI = SKILL_ROOT / "tests" / "fixtures" / "wiki"
GRAPH_CONTRACT_WIKI = SKILL_ROOT / "tests" / "fixtures" / "graph_contract" / "wiki"
GRAPH_CONTRACT_QUERIES = HERE / "graph_queries.jsonl"

sys.path.insert(0, str(SCRIPTS))

from build_index import WikiIndex  # noqa: E402
from query_planner import DefaultQueryPlanner  # noqa: E402
from query import hybrid_search, BUDGET_POLICY as _BUDGET_POLICY  # noqa: E402
import build_graph as _bg  # noqa: E402
from chunking import CHUNK_SCHEMA_VERSION  # noqa: E402


def _citation_violations(bundle):
    """Zero-tolerance check of the ``[来源: Wiki/xxx.md]`` citation contract.

    Community-report rows never map to a Wiki page and are exempt; every other
    ContextItem must expose a forward-slash, non-absolute ``Wiki/``-rooted path
    that literally appears as a citation token inside ``context_text``.
    """
    found = []
    for item in getattr(bundle, "items", ()) or ():
        if item.inclusion_reason == "global_community_report":
            continue
        path = str(item.path)
        reasons = []
        if "\\" in path:
            reasons.append("backslash_separator")
        posix_path = PurePosixPath(path)
        if posix_path.is_absolute() or PureWindowsPath(path).is_absolute():
            reasons.append("absolute_path")
        if not path.startswith("Wiki/"):
            reasons.append("not_wiki_rooted")
        # ``PurePosixPath`` normalizes duplicate separators and ``.``; reject
        # every input whose literal spelling is not its canonical serialized
        # form, and reject traversal segments explicitly.  A valid citation is
        # both Wiki-rooted *and* a stable, publishable identity.
        if any(part in {".", ".."} for part in path.split("/")):
            reasons.append("dot_or_traversal_segment")
        if path != posix_path.as_posix():
            reasons.append("noncanonical_posix_path")
        if f"[来源: {path}]" not in (getattr(bundle, "context_text", "") or ""):
            reasons.append("citation_token_missing_from_context_text")
        if reasons:
            found.append({"path": path, "page_id": item.page_id, "reasons": reasons})
    return found


def _citation_contract_failures(metrics: dict) -> list[str]:
    """Return baseline-independent citation failures for every eval mode.

    Citation paths are publication safety data rather than a relative quality
    metric.  Consequently an invalid run must not be accepted merely because
    it initializes, lacks, or cannot compare a historical baseline.
    """
    count = metrics.get("quality", {}).get("citation_path_contract_violation_count", 0)
    if count > 0:
        return [
            f"citation_path_contract_violation_count={count} > 0"
            "（判定口径：ContextItem.path 必须为 Wiki/ 起始的规范相对 posix 路径，"
            "且 context_text 内含 [来源: <path>] 字面；样本见 metrics.citation_paths）"
        ]
    return []


BENCHMARK_EVIDENCE_FIELDS = frozenset({
    "evidence_schema_version",
    "evidence_source",
    "probe_scope",
    "sampling_method",
    "sampling_key_schema",
    "probe_keys",
    "probe_selection_sha256",
    "probe_count",
    "probe_total",
    "probe_coverage",
    "result_limit",
    "recall_aggregation",
    "benchmark_duration_ms",
    "probe_selection_ms",
    "exact_verification_ms",
    "ann_verification_ms",
    "recall_assembly_ms",
    "exact_method",
    "exact_scan_rows",
    "exact_scan_batches",
    "ann_query_count",
    "exact_result_ids",
    "candidate_result_ids",
})


def validate_benchmark_contract(manifest: dict) -> dict:
    """Fail closed when build-time ANN evidence cannot support its policy claim.

    Issue #41 changes recall from a full-corpus minimum to a bounded sampled
    minimum.  Eval must therefore validate the evidence schema and ensure the
    policy repeats the same scope/counts instead of silently interpreting a
    sampled 1.0 as a corpus-wide proof.
    """
    if manifest.get("format_version") != 5:
        raise ValueError("ANN benchmark evidence requires manifest format_version=5")
    benchmark = manifest.get("benchmark")
    policy = manifest.get("policy")
    if not isinstance(benchmark, dict) or not isinstance(policy, dict):
        raise ValueError("manifest benchmark and policy must be objects")
    missing = sorted(BENCHMARK_EVIDENCE_FIELDS - set(benchmark))
    if missing:
        raise ValueError(f"benchmark evidence fields missing: {missing}")

    source = benchmark["evidence_source"]
    scope = benchmark["probe_scope"]
    count = benchmark["probe_count"]
    total = benchmark["probe_total"]
    coverage = benchmark["probe_coverage"]
    probe_keys = benchmark["probe_keys"]
    exact_ids = benchmark["exact_result_ids"]
    candidate_ids = benchmark["candidate_result_ids"]
    if benchmark["evidence_schema_version"] != 2:
        raise ValueError("unsupported benchmark evidence_schema_version")
    if source not in {"measured", "observer"}:
        raise ValueError("benchmark evidence_source must be measured or observer")
    if scope not in {"full", "sampled", "synthetic"}:
        raise ValueError("benchmark probe_scope must be full, sampled, or synthetic")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise ValueError("benchmark probe_total must be a positive integer")
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= total:
        raise ValueError("benchmark probe_count must be an integer within probe_total")
    if not isinstance(coverage, (int, float)) or not math.isclose(coverage, count / total):
        raise ValueError("benchmark probe_coverage is inconsistent with count/total")
    if not isinstance(probe_keys, list) or len(probe_keys) != count:
        raise ValueError("benchmark probe_keys length must equal probe_count")
    if len(set(probe_keys)) != len(probe_keys):
        raise ValueError("benchmark probe_keys must be unique")
    selection_digest = hashlib.sha256("\n".join(probe_keys).encode()).hexdigest()
    if benchmark["probe_selection_sha256"] != selection_digest:
        raise ValueError("benchmark probe_selection_sha256 does not match probe_keys")
    if benchmark["result_limit"] != 20 or benchmark["recall_aggregation"] != "minimum":
        raise ValueError("benchmark result_limit/recall_aggregation contract changed")
    if not isinstance(benchmark["benchmark_duration_ms"], (int, float)) \
            or benchmark["benchmark_duration_ms"] < 0:
        raise ValueError("benchmark_duration_ms must be non-negative")

    if scope == "full":
        if count != total or coverage != 1.0 or benchmark["sampling_method"] != "full":
            raise ValueError("full benchmark scope must cover every probe")
    elif scope == "sampled":
        if not 0 < count < total or benchmark["sampling_method"] != "bottom_k_sha256_v1":
            raise ValueError("sampled benchmark scope must be bounded bottom-k SHA-256")
    else:
        if source != "observer" or count != 0 or coverage != 0.0:
            raise ValueError("synthetic benchmark evidence must be an explicit zero-probe observer")

    expected_rows = count if source == "measured" else 0
    if not isinstance(exact_ids, list) or not isinstance(candidate_ids, list) \
            or len(exact_ids) != expected_rows or len(candidate_ids) != expected_rows:
        raise ValueError("benchmark result evidence lengths do not match measured probe_count")

    if source == "measured":
        if benchmark["exact_method"] != "streamed_numpy_cosine_v1":
            raise ValueError("measured benchmark must use streamed batch exact verification")
        if benchmark["exact_scan_rows"] != total:
            raise ValueError("exact batch verification must scan the dense corpus exactly once")
        if not isinstance(benchmark["exact_scan_batches"], int) \
                or benchmark["exact_scan_batches"] <= 0:
            raise ValueError("exact_scan_batches must be positive")
        if benchmark["ann_query_count"] != count:
            raise ValueError("ann_query_count must equal measured probe_count")

    for field in (
        "benchmark_duration_ms",
        "probe_selection_ms",
        "exact_verification_ms",
        "ann_verification_ms",
        "recall_assembly_ms",
    ):
        value = benchmark[field]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be a finite non-negative number")

    if source == "observer":
        if benchmark["exact_method"] != "observer":
            raise ValueError("observer evidence must identify exact_method=observer")
        if any(benchmark[field] != 0 for field in (
            "exact_scan_rows", "exact_scan_batches", "ann_query_count"
        )):
            raise ValueError("observer evidence cannot report measured scan/query counters")

    if policy.get("benchmark_scope") != scope \
            or policy.get("benchmark_probe_count") != count \
            or policy.get("benchmark_probe_total") != total:
        raise ValueError("policy benchmark scope/counts are inconsistent with benchmark evidence")
    if policy.get("selected_mode") not in {"ann", "exact"}:
        raise ValueError("policy selected_mode must be ann or exact")
    return {
        "selected_mode": policy["selected_mode"],
        "probe_scope": scope,
        "probe_count": count,
        "probe_total": total,
        "probe_coverage": coverage,
        "probe_selection_sha256": benchmark["probe_selection_sha256"],
        "benchmark_duration_ms": benchmark["benchmark_duration_ms"],
    }


def _active_benchmark_contract(wi: WikiIndex) -> dict:
    manifest = json.loads(wi._resolve_active_manifest().read_text(encoding="utf-8"))
    return validate_benchmark_contract(manifest)


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
                   build_ann: bool, regression_pp: float,
                   hard_max_tokens: int | None = None):
    work_dir.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()

    # 1) 主索引（exact，小库等价精确）
    main_root = work_dir / "main"
    main_wi, main_wiki, build_time = _build(main_root, wiki_src, "exact", full_rebuild=True)
    main_benchmark = _active_benchmark_contract(main_wi)
    planner = DefaultQueryPlanner(project_root=main_root)

    # 2) ANN 索引（独立 project，复用语义内容；向量层走 IVF_HNSW_FLAT）
    ann_wi = None
    ann_build_time = None
    ann_benchmark = None
    if build_ann:
        ann_root = work_dir / "ann"
        ann_wi, _, ann_build_time = _build(ann_root, wiki_src, "ivf-hnsw-flat", full_rebuild=True)
        ann_benchmark = _active_benchmark_contract(ann_wi)

    # 3) 逐查询评测
    page_recalls, evid_recalls, mrrs, exact_hits, ann_recalls = [], [], [], [], []
    latencies = []
    context_overflow = 0
    budget_violations = 0
    budget_violation_samples = []
    citation_path_violations = 0
    citation_path_violation_samples = []
    budget_by_intent = {}          # intent → {"base": Counter, "effective": Counter}
    expanded_queries = 0           # effective > base（意图倍率生效）
    max_effective_budget = 0
    graph_only_unsupported = 0
    bundle_tokens = []
    detail = []

    for q in queries:
        gold = q["relevant_pages"]
        plan = planner.plan(q["query"])
        t0 = time.perf_counter()
        res = hybrid_search(main_wi, q["query"], planner, k=10,
                            max_tokens=max_tokens, hard_max_tokens=hard_max_tokens,
                            wiki_dir=main_wiki, intent_override="auto",
                            allow_local_fallback=True)
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

        # Context overflow：以本次实际分配的预算为准（effective = base × 意图倍率，
        # 再受 hard cap 限制）。assemble_context 按 effective 截断 → 应恒 0。
        bundle = res.bundle
        if bundle.token_count > bundle.effective_budget_tokens:
            context_overflow += 1
        # Citation contract (issue #43): every cited page must be Wiki-relative
        # posix. Zero tolerance — one violation fails publication.
        cite_bad = _citation_violations(bundle)
        if cite_bad:
            citation_path_violations += len(cite_bad)
            if len(citation_path_violation_samples) < 5:
                citation_path_violation_samples.append(
                    {"query": q["query"], "violations": cite_bad[:3]})
        # 防契约漂移：max_context_tokens 必须等于 effective；hard cap 必须被尊重
        violations = bundle.budget_contract_violations()
        if violations:
            budget_violations += 1
            if len(budget_violation_samples) < 5:
                budget_violation_samples.append({"query": q["query"], "violations": violations})
        slot = budget_by_intent.setdefault(res.plan.intent, {"base": {}, "effective": {}})
        slot["base"][str(bundle.requested_base_budget_tokens)] = \
            slot["base"].get(str(bundle.requested_base_budget_tokens), 0) + 1
        slot["effective"][str(bundle.effective_budget_tokens)] = \
            slot["effective"].get(str(bundle.effective_budget_tokens), 0) + 1
        if bundle.effective_budget_tokens > bundle.requested_base_budget_tokens:
            expanded_queries += 1
        max_effective_budget = max(max_effective_budget, bundle.effective_budget_tokens)
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
            "context_mode": res.plan.context_mode,
            "budget": {**bundle.budget_to_json(), "token_count": bundle.token_count},
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
            "budget_contract_violation_count": budget_violations,
            "citation_path_contract_violation_count": citation_path_violations,
            "graph_only_unsupported_count": graph_only_unsupported,
        },
        "budget": {
            "policy": _BUDGET_POLICY,
            "requested_base_budget_tokens": max_tokens,
            "hard_max_tokens": hard_max_tokens,
            "expanded_query_count": expanded_queries,
            "max_effective_budget_tokens": max_effective_budget,
            "by_intent": budget_by_intent,
            "violation_samples": budget_violation_samples,
        },
        "citation_paths": {
            "contract": "Wiki/<relative>.md (posix, non-absolute)",
            "violation_samples": citation_path_violation_samples,
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
        "index_benchmark": {
            "main": main_benchmark,
            "ann": ann_benchmark,
        },
    }
    return metrics, detail


def _recall_at_5(candidates, gold_pages):
    hits = sum(1 for candidate in candidates[:5] if _page_hit(candidate, gold_pages))
    return min(1.0, hits / max(1, len(gold_pages)))


def _evidence_recall(bundle, required_facts):
    if not required_facts:
        return 1.0
    context = bundle.context_text or ""
    return sum(fact in context for fact in required_facts) / len(required_facts)


def run_graph_contract_evaluation(wiki_src: Path, queries: list, work_dir: Path,
                                  max_tokens: int, hard_max_tokens: int | None = None):
    """Measure graph lift with paired production retrieval on an explicit-link corpus."""
    root = work_dir / "graph-contract"
    wi, wiki, _ = _build(root, wiki_src, "exact", full_rebuild=True)
    planner = DefaultQueryPlanner(project_root=root)
    failures, detail = [], []
    validated_count = unsupported_count = incremental_gold_pages = 0
    off_recalls, on_recalls = [], []

    for query in queries:
        k = int(query.get("k", 5))
        common = {"k": k, "max_tokens": max_tokens,
                  "hard_max_tokens": hard_max_tokens, "wiki_dir": wiki}
        without_graph = hybrid_search(wi, query["query"], planner,
                                      enable_graph=False, **common)
        with_graph = hybrid_search(wi, query["query"], planner,
                                   enable_graph=True, **common)
        gold = query["relevant_pages"]
        off_recall = _recall_at_5(without_graph.candidates, gold)
        on_recall = _recall_at_5(with_graph.candidates, gold)
        off_recalls.append(off_recall)
        on_recalls.append(on_recall)
        off_gold = {c.page_id for c in without_graph.candidates[:5] if _page_hit(c, gold)}
        on_gold = {c.page_id for c in with_graph.candidates[:5] if _page_hit(c, gold)}
        added = len(on_gold - off_gold)
        incremental_gold_pages += added
        validated_count += with_graph.graph_validated_count
        graph_items = [item for item in with_graph.bundle.items
                       if item.inclusion_reason == "graph_expansion"]
        unsupported_count += sum(1 for item in graph_items if not item.evidence)
        required_gain = int(query.get("min_incremental_gold_pages", 0))
        if added < required_gain:
            failures.append(f"{query['query']}: incremental_gold_pages={added} < required {required_gain}")
        if on_recall < off_recall:
            failures.append(f"{query['query']}: graph recall regressed {off_recall:.4f}->{on_recall:.4f}")
        detail.append({
            "query": query["query"], "k": k,
            "page_recall_without_graph": off_recall,
            "page_recall_with_graph": on_recall,
            "evidence_recall_without_graph": _evidence_recall(without_graph.bundle, query.get("required_facts", [])),
            "evidence_recall_with_graph": _evidence_recall(with_graph.bundle, query.get("required_facts", [])),
            "incremental_gold_pages": added,
            "graph_validated_count": with_graph.graph_validated_count,
            "direct_top5": [Path(c.page_id).name for c in without_graph.candidates[:5]],
            "graph_top5": [Path(c.page_id).name for c in with_graph.candidates[:5]],
            "graph_evidence": [{"page": Path(item.page_id).name,
                                "evidence_chunks": [ev.chunk_id for ev in item.evidence],
                                "paths": [{"source": Path(path.source_id).name,
                                           "target": Path(path.target_id).name,
                                           "signals": path.edge_signals}
                                          for path in item.graph_paths]}
                               for item in graph_items],
        })
    if unsupported_count:
        failures.append(f"graph_only_unsupported_count={unsupported_count} > 0")
    if validated_count <= 0:
        failures.append("graph contract fixture did not produce a validated graph result")
    return {
        "n_queries": len(queries),
        "page_recall_at_5_without_graph": round(statistics.mean(off_recalls), 4),
        "page_recall_at_5_with_graph": round(statistics.mean(on_recalls), 4),
        "incremental_gold_pages": incremental_gold_pages,
        "graph_validated_count": validated_count,
        "graph_only_unsupported_count": unsupported_count,
        "failures": failures,
    }, detail


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
    ap.add_argument("--graph-wiki", type=Path, default=GRAPH_CONTRACT_WIKI)
    ap.add_argument("--graph-queries", type=Path, default=GRAPH_CONTRACT_QUERIES)
    ap.add_argument("--baselines", type=Path, default=HERE / "baselines.json")
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="构建临时目录（默认系统临时目录；本地建议 D: 盘避免 C: 虚拟化）")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="基础 token 预算；实际上限 = 基础预算 × 意图倍率（由 hybrid_search 决定）")
    ap.add_argument("--hard-max-tokens", type=int, default=None,
                    help="硬上限：effective budget 与 token_count 均不得超过该值")
    ap.add_argument("--no-ann", action="store_true", help="跳过 ANN 索引构建")
    ap.add_argument("--regression-pp", type=float, default=2.0, help="Recall 下降百分点阈值")
    ap.add_argument("--init-baseline", action="store_true", help="把当前指标写为 baselines.json 并退出 0")
    ap.add_argument("--force-compare", action="store_true",
                    help="忽略 chunk_schema_version 不匹配，强制对比旧基线（仅本地调试用，CI 不应使用）")
    args = ap.parse_args()

    if not args.wiki.exists():
        print(f"[ERROR] fixture wiki 不存在: {args.wiki}", file=sys.stderr)
        return 2
    if not args.graph_wiki.exists() or not args.graph_queries.exists():
        print("[ERROR] graph contract fixture or queries missing", file=sys.stderr)
        return 2
    queries = load_queries(args.queries)
    graph_queries = load_queries(args.graph_queries)
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="obs_wiki_eval_"))
    print(f"[info] work-dir = {work_dir}", file=sys.stderr)

    metrics, detail = run_evaluation(args.wiki, queries, work_dir, args.max_tokens,
                             build_ann=not args.no_ann, regression_pp=args.regression_pp,
                             hard_max_tokens=args.hard_max_tokens)

    # This is a publication contract, not a baseline-relative quality metric.
    # Enforce it at the primary evaluation boundary: unsafe evidence must not
    # trigger unrelated graph/model work or reach any baseline write path.
    citation_failures = _citation_contract_failures(metrics)
    if citation_failures:
        print("\n[FAIL] citation 路径契约未通过：", file=sys.stderr)
        for failure in citation_failures:
            print("  - " + failure, file=sys.stderr)
        return 1

    graph_metrics, graph_detail = run_graph_contract_evaluation(
        args.graph_wiki, graph_queries, work_dir, args.max_tokens,
        hard_max_tokens=args.hard_max_tokens)
    metrics["graph_contract"] = graph_metrics

    (HERE / "results.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    (HERE / "detail.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in detail), encoding="utf-8")
    (HERE / "graph_detail.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in graph_detail), encoding="utf-8")
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
        failures.append(
            f"context_overflow_count={mq['context_overflow_count']} > 0"
            f"（判定口径：token_count > bundle.effective_budget_tokens）")
    if mq.get("budget_contract_violation_count", 0) > 0:
        failures.append(
            f"budget_contract_violation_count={mq['budget_contract_violation_count']} > 0"
            f"：{metrics['budget']['violation_samples']}")
    if mq["graph_only_unsupported_count"] > 0:
        failures.append(f"graph_only_unsupported_count={mq['graph_only_unsupported_count']} > 0")
    failures.extend("graph contract: " + failure
                    for failure in metrics["graph_contract"]["failures"])
    if mq["ann_recall_at_10"] is not None and mq["ann_recall_at_10"] < 0.98:
        failures.append(f"ann_recall_at_10={mq['ann_recall_at_10']:.4f} < 0.98")
    ann_benchmark = metrics.get("index_benchmark", {}).get("ann")
    if ann_benchmark is not None and ann_benchmark.get("selected_mode") != "ann":
        failures.append(
            "ANN evaluation build was not promoted: "
            f"selected_mode={ann_benchmark.get('selected_mode')} "
            f"scope={ann_benchmark.get('probe_scope')} "
            f"probes={ann_benchmark.get('probe_count')}/{ann_benchmark.get('probe_total')}"
        )

    if failures:
        print("\n[FAIL] 评测未通过：", file=sys.stderr)
        for f in failures:
            print("  - " + f, file=sys.stderr)
        return 1
    print("\n[PASS] 评测通过，无回归。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
