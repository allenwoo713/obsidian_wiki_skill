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
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from types import SimpleNamespace
from pathlib import Path, PurePosixPath, PureWindowsPath

HERE = Path(__file__).resolve().parent
SKILL_ROOT = HERE.parent
if str(SKILL_ROOT) not in sys.path:
    # Direct-file entry points start with eval/ on sys.path; package imports
    # below require the repository root before they are resolved.
    sys.path.insert(0, str(SKILL_ROOT))
SCRIPTS = SKILL_ROOT / "scripts"
FIXTURES_WIKI = SKILL_ROOT / "tests" / "fixtures" / "wiki"
GRAPH_CONTRACT_WIKI = SKILL_ROOT / "tests" / "fixtures" / "graph_contract" / "wiki"
GRAPH_CONTRACT_QUERIES = HERE / "graph_queries.jsonl"

sys.path.insert(0, str(SCRIPTS))

from build_index import CandidateQueryPolicy, WikiIndex  # noqa: E402
from obsidian_wiki.domain.index_models import CandidateBuildPolicy  # noqa: E402
from eval.ann_corpus_manifest import (  # noqa: E402
    canonical_corpus_file_bytes,
    canonical_content_tree_sha256,
    PHASE07_CURRENT_BASELINE,
    public_distractor_bytes,
    validate_indexed_query_digest_separation,
)
from query_planner import DefaultQueryPlanner  # noqa: E402
from query import hybrid_search, BUDGET_POLICY as _BUDGET_POLICY  # noqa: E402
import build_graph as _bg  # noqa: E402
from chunking import CHUNK_SCHEMA_VERSION  # noqa: E402
try:  # ``python eval/run_eval.py`` has eval/ rather than repo root on sys.path.
    from eval.benchmark_ann_build import (  # noqa: E402
        CANDIDATES, DECISION_EF_GRID, EVIDENCE_SCHEMA_VERSION, validate_evidence,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI entry point
    from benchmark_ann_build import (  # noqa: E402
        CANDIDATES, DECISION_EF_GRID, EVIDENCE_SCHEMA_VERSION, validate_evidence,
    )


FUNCTIONAL_FINAL_RETRIEVAL_METRIC = "functional_final_retrieval_ann_overlap_at_10"
FINAL_RETRIEVAL_DECISION_SCHEMA_VERSION = 2
_CANDIDATE_OBSERVATION_FIELDS = {
    "retrieval", "page", "evidence", "mrr", "latency", "context",
    "citation", "graph", "non_regression",
}


def _head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SKILL_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _decision_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "lancedb": importlib.metadata.version("lancedb"),
        "numpy": importlib.metadata.version("numpy"),
        "pyarrow": importlib.metadata.version("pyarrow"),
    }


def _stable_json_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_identities() -> tuple[str, str]:
    checkout = os.environ.get("GITHUB_SHA") or _head_sha()
    pr_head = os.environ.get("GITHUB_PR_HEAD_SHA") or checkout
    return pr_head, checkout


def _all_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return False


def validate_candidate_decision_records(
    packet: dict, comparator_evidence: dict
) -> dict:
    """Fail closed before a model-backed packet can be considered for approval."""
    if comparator_evidence.get("evidence_schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("comparator evidence schema")
    if packet.get("schema_version") != FINAL_RETRIEVAL_DECISION_SCHEMA_VERSION:
        raise ValueError("decision schema")
    if packet.get("comparator_schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("comparator binding schema")
    records = packet.get("records")
    expected = {(candidate, ef) for candidate in CANDIDATES for ef in DECISION_EF_GRID}
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("candidate records")
    if not packet.get("pr_head_sha") or not packet.get("actions_merge_checkout_sha"):
        raise ValueError("source identities")
    environment = packet.get("environment")
    if not isinstance(environment, dict) or {
        "lancedb": environment.get("lancedb"),
        "numpy": environment.get("numpy"),
        "pyarrow": environment.get("pyarrow"),
    } != {"lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0"}:
        raise ValueError("locked candidate environment")
    seen = set()
    run_ids, index_ids, invocation_ids, result_digests, payload_digests = set(), set(), set(), set(), set()
    for record in records:
        if not isinstance(record, dict) or (record.get("candidate"), record.get("query_ef")) not in expected:
            raise ValueError("candidate/grid binding")
        binding = (record["candidate"], record["query_ef"])
        if binding in seen:
            raise ValueError("duplicate candidate record")
        seen.add(binding)
        if record.get("head_sha") != packet.get("head_sha") or record.get("environment") != packet.get("environment"):
            raise ValueError("mixed head or environment")
        if record.get("pr_head_sha") != packet.get("pr_head_sha") \
                or record.get("actions_merge_checkout_sha") != packet.get("actions_merge_checkout_sha"):
            raise ValueError("mixed source identities")
        if record.get("corpus_sha256") != comparator_evidence.get("corpus", {}).get("sha256") \
                or record.get("queries_sha256") != comparator_evidence.get("queries", {}).get("sha256"):
            raise ValueError("mixed comparator evidence")
        policy = record.get("applied_policy")
        if policy != {"candidate": record["candidate"], "query_ef": record["query_ef"]}:
            raise ValueError("candidate policy binding")
        run_id = record.get("candidate_run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("candidate run identity")
        if run_id in run_ids:
            raise ValueError("candidate run identity")
        run_ids.add(run_id)
        index = record.get("candidate_index")
        if not isinstance(index, dict) or not all(index.get(key) for key in (
            "build_id", "manifest_sha256", "root_identity",
        )):
            raise ValueError("candidate index identity")
        index_id = (index["build_id"], index["manifest_sha256"], index["root_identity"])
        if index_id in index_ids:
            raise ValueError("candidate index identity")
        index_ids.add(index_id)
        invocation = record.get("hybrid_invocation")
        if not isinstance(invocation, dict) or invocation.get("entrypoint") != "query.hybrid_search" \
                or not invocation.get("trace_id") or not invocation.get("digest") \
                or invocation.get("query_count", 0) <= 0:
            raise ValueError("hybrid invocation identity")
        invocation_id = (invocation["trace_id"], invocation["digest"])
        if invocation_id in invocation_ids:
            raise ValueError("hybrid invocation identity")
        invocation_ids.add(invocation_id)
        traces = invocation.get("traces")
        if not isinstance(traces, list) or len(traces) != invocation["query_count"]:
            raise ValueError("hybrid invocation identity")
        expected_run_id = _stable_json_digest({
            "candidate": record["candidate"], "query_ef": record["query_ef"],
            "build_id": index["build_id"], "traces": traces,
        })
        if run_id != expected_run_id:
            raise ValueError("candidate run identity")
        if invocation["digest"] != _stable_json_digest({"run_id": run_id, "traces": traces}) \
                or invocation["trace_id"] != _stable_json_digest({
                    "run_id": run_id, "entrypoint": "query.hybrid_search",
                }):
            raise ValueError("hybrid invocation identity")
        result_digest = record.get("result_sha256")
        if not isinstance(result_digest, str) or len(result_digest) != 64 \
                or result_digest in result_digests:
            raise ValueError("candidate result digest")
        result_digests.add(result_digest)
        metrics = record.get("final_retrieval")
        if not isinstance(metrics, dict) or set(metrics) != _CANDIDATE_OBSERVATION_FIELDS:
            raise ValueError("candidate-specific final retrieval")
        retrieval = metrics.get("retrieval")
        if not isinstance(retrieval, dict) or FUNCTIONAL_FINAL_RETRIEVAL_METRIC not in retrieval \
                or not isinstance(retrieval.get("result_payload"), list):
            raise ValueError("candidate-specific final retrieval")
        latency = metrics.get("latency")
        if not isinstance(latency, dict) or not isinstance(latency.get("samples_s"), list) \
                or not latency["samples_s"]:
            raise ValueError("candidate-specific final retrieval")
        non_regression = metrics.get("non_regression")
        if not isinstance(non_regression, dict) or non_regression.get("baseline_refresh") is not False \
                or non_regression.get("failures") != []:
            raise ValueError("candidate-specific final retrieval")
        if not _all_finite(metrics):
            raise ValueError("non-finite candidate metric")
        payload_digest = _stable_json_digest(metrics)
        if payload_digest in payload_digests:
            raise ValueError("candidate result payload")
        payload_digests.add(payload_digest)
        if result_digest != _stable_json_digest({
            "run_id": run_id, "final_retrieval": metrics,
        }):
            raise ValueError("candidate result digest")
        binding_input = record.get("input_binding")
        if not isinstance(binding_input, dict) or not all(binding_input.get(key) for key in (
            "fixture_sha256", "evaluation_queries_sha256", "comparator_corpus_sha256",
            "comparator_queries_sha256",
        )):
            raise ValueError("immutable input binding")
        if binding_input["comparator_corpus_sha256"] != comparator_evidence["corpus"]["sha256"] \
                or binding_input["comparator_queries_sha256"] != comparator_evidence["queries"]["sha256"]:
            raise ValueError("immutable input binding")
    if seen != expected:
        raise ValueError("incomplete candidate records")
    return packet


def _fixture_digest(wiki_src: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(wiki_src).rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(wiki_src).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _candidate_index_identity(*, wi: WikiIndex, root: Path, policy: CandidateQueryPolicy) -> dict:
    manifest_path = wi._resolve_active_manifest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate_query_policy") != policy.to_json():
        raise ValueError("candidate policy was not applied to the built index")
    return {
        "build_id": manifest["build_id"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "root_identity": _stable_json_digest({
            "root": root.name, "build_id": manifest["build_id"], "policy": policy.to_json(),
        }),
    }


def _candidate_result_payload(result) -> dict:
    bundle = result.bundle
    return {
        "query": result.query,
        "plan": result.plan.to_json(),
        "pages": [candidate.page_id for candidate in result.candidates[:10]],
        "items": [
            {
                "page_id": item.page_id,
                "path": str(item.path),
                "scope": item.scope,
                "inclusion_reason": item.inclusion_reason,
                "evidence": [hit.chunk_id for hit in item.evidence],
                "graph_paths": [
                    {
                        "source": path.source_id, "target": path.target_id,
                        "signals": list(path.edge_signals),
                    }
                    for path in item.graph_paths
                ],
            }
            for item in bundle.items
        ],
        "context_sha256": hashlib.sha256((bundle.context_text or "").encode("utf-8")).hexdigest(),
        "context_text": bundle.context_text or "",
        "token_count": bundle.token_count,
        "budget": bundle.budget_to_json(),
    }


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"finite {name}")
    return float(value)


def _is_lookup_specification(specification: dict) -> bool:
    intent = specification.get("intent", specification.get("query_type", "lookup"))
    return intent == "lookup"


def _page_id_hit(page_id: str, gold_pages: object) -> bool:
    """Serialized page identities use exactly the live ``_page_hit`` rules."""
    if not isinstance(page_id, str):
        raise ValueError("serialized page identity")
    return _page_hit(SimpleNamespace(page_id=page_id), gold_pages)


def _metrics_from_result_pairs(*, specifications: list[dict], results: list) -> dict[str, float | int]:
    if len(specifications) != len(results) or not specifications:
        raise ValueError("complete hybrid live results")
    page, evidence, exact_lookup, mrr, functional = [], [], [], [], []
    citation = overflow = budget = graph_unsupported = 0
    for specification, result in zip(specifications, results, strict=True):
        bundle = result.bundle
        candidates = list(result.candidates[:10])
        gold = set(specification.get("relevant_pages", []))
        hits = [position for position, candidate in enumerate(candidates, 1) if _page_hit(candidate, gold)]
        page.append(min(1.0, sum(_page_hit(candidate, gold) for candidate in candidates[:5]) / max(1, len(gold))))
        functional.append(min(1.0, sum(_page_hit(candidate, gold) for candidate in candidates) / max(1, len(gold))))
        required = specification.get("required_facts") or []
        context = bundle.context_text or ""
        evidence.append(sum(fact in context for fact in required) / len(required) if required else 1.0)
        if _is_lookup_specification(specification):
            exact_lookup.append(1.0 if any(position <= 3 for position in hits) else 0.0)
        mrr.append(1.0 / hits[0] if hits else 0.0)
        citation += len(_citation_violations(bundle))
        overflow += int(bundle.token_count > bundle.effective_budget_tokens)
        budget += int(bool(bundle.budget_contract_violations()))
        graph_unsupported += sum(1 for item in bundle.items
                                 if item.inclusion_reason == "graph_expansion" and not item.evidence)
    values: dict[str, float | int] = {
        FUNCTIONAL_FINAL_RETRIEVAL_METRIC: statistics.mean(functional),
        "page_recall_at_5": statistics.mean(page),
        "evidence_recall_at_10": statistics.mean(evidence),
        "exact_lookup_hit_at_3": statistics.mean(exact_lookup) if exact_lookup else 1.0,
        "mrr_at_10": statistics.mean(mrr),
        "citation_violation_count": citation, "context_overflow_count": overflow,
        "budget_violation_count": budget, "graph_unsupported_count": graph_unsupported,
    }
    for name, value in values.items():
        _finite_number(name, value)
    return values


def aggregate_hybrid_result_metrics(*, specifications: list[dict], baseline_results: list,
                                    candidate_results: list) -> dict[str, dict[str, float | int]]:
    """Aggregate live public ``HybridResult`` objects without metric aliases."""
    return {
        "baseline": _metrics_from_result_pairs(specifications=specifications, results=baseline_results),
        "candidate": _metrics_from_result_pairs(specifications=specifications, results=candidate_results),
    }


def aggregate_hybrid_serialized_metrics(*, specifications: list[dict], observations: list[dict]) -> dict[str, dict[str, float | int]]:
    """Recompute gates from serialized query evidence in an uploaded raw tree."""
    if len(specifications) != len(observations) or not observations:
        raise ValueError("complete serialized hybrid observations")
    def metrics(role: str) -> dict[str, float | int]:
        page = []; evidence = []; exact = []; mrr = []; functional = []
        citation = overflow = budget = graph = 0
        for specification, row in zip(specifications, observations, strict=True):
            observation = row.get(role)
            if not isinstance(observation, dict) or set(observation) != {"result", "duration_ms"}:
                raise ValueError("complete serialized hybrid observation")
            payload = observation["result"]
            duration = observation["duration_ms"]
            if not isinstance(payload, dict) or isinstance(duration, bool) \
                    or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or duration < 0:
                raise ValueError("serialized hybrid evidence")
            required_payload = {"query", "plan", "pages", "items", "context_text", "context_sha256", "token_count", "budget"}
            if set(payload) != required_payload:
                raise ValueError("serialized hybrid payload schema")
            context = payload["context_text"]
            if not isinstance(context, str) or hashlib.sha256(context.encode("utf-8")).hexdigest() != payload.get("context_sha256"):
                raise ValueError("serialized hybrid context")
            raw = payload["budget"]
            budget_fields = {"requested_base_budget_tokens", "budget_multiplier", "effective_budget_tokens", "hard_max_tokens", "budget_policy", "max_context_tokens"}
            if not isinstance(raw, dict) or set(raw) != budget_fields:
                raise ValueError("serialized hybrid raw budget")
            token_count = payload.get("token_count")
            effective, maximum, hard = raw["effective_budget_tokens"], raw["max_context_tokens"], raw["hard_max_tokens"]
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (token_count, effective, maximum)) \
                    or (hard is not None and (isinstance(hard, bool) or not isinstance(hard, int) or hard < 0)) \
                    or maximum != effective or token_count > effective or (hard is not None and effective > hard):
                budget += 1
            overflow += int(isinstance(token_count, int) and isinstance(effective, int) and token_count > effective)
            gold = set(specification.get("relevant_pages", [])); pages = payload.get("pages")
            if not isinstance(pages, list) or not all(isinstance(item, str) for item in pages):
                raise ValueError("serialized hybrid pages")
            hits = [position for position, page_id in enumerate(pages[:10], 1) if _page_id_hit(page_id, gold)]
            page.append(min(1.0, sum(_page_id_hit(page_id, gold) for page_id in pages[:5]) / max(1, len(gold))))
            functional.append(min(1.0, sum(_page_id_hit(page_id, gold) for page_id in pages[:10]) / max(1, len(gold))))
            required = specification.get("required_facts") or []
            evidence.append(sum(fact in context for fact in required) / len(required) if required else 1.0)
            if _is_lookup_specification(specification): exact.append(1.0 if any(hit <= 3 for hit in hits) else 0.0)
            mrr.append(1.0 / hits[0] if hits else 0.0)
            items = payload.get("items", [])
            if not isinstance(items, list):
                raise ValueError("serialized hybrid items")
            citation += len(_citation_violations(SimpleNamespace(
                context_text=context,
                items=[SimpleNamespace(page_id=item.get("page_id"), path=item.get("path"),
                                       inclusion_reason=item.get("inclusion_reason", "dense_retrieval"))
                       for item in items if isinstance(item, dict)],
            )))
            for item in items:
                if not isinstance(item, dict): raise ValueError("serialized hybrid item")
                reason, path, item_evidence = item.get("inclusion_reason"), item.get("path"), item.get("evidence")
                if reason == "graph_expansion" and not item_evidence:
                    graph += 1
        return {FUNCTIONAL_FINAL_RETRIEVAL_METRIC: statistics.mean(functional),
                "page_recall_at_5": statistics.mean(page), "evidence_recall_at_10": statistics.mean(evidence),
                "exact_lookup_hit_at_3": statistics.mean(exact) if exact else 1.0, "mrr_at_10": statistics.mean(mrr),
                "citation_violation_count": citation, "context_overflow_count": overflow,
                "budget_violation_count": budget, "graph_unsupported_count": graph}
    return {"baseline": metrics("baseline"), "candidate": metrics("candidate")}


def aggregate_hybrid_serialized_scale_diagnostics(*, observations: list[dict]) -> dict[str, object]:
    """Reduce expanded hybrid evidence to latency diagnostics without gold labels."""
    if not observations:
        raise ValueError("complete serialized hybrid scale observations")

    def diagnostics(role: str) -> dict[str, float | int]:
        durations: list[float] = []
        for row in observations:
            observation = row.get(role) if isinstance(row, dict) else None
            if not isinstance(observation, dict) or set(observation) != {"result", "duration_ms"} \
                    or not isinstance(observation["result"], dict):
                raise ValueError("complete serialized hybrid scale observation")
            duration = observation["duration_ms"]
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) \
                    or not math.isfinite(float(duration)) or duration < 0:
                raise ValueError("serialized hybrid scale duration")
            durations.append(float(duration))
        return {"sample_count": len(durations),
                "duration_p50_ms": _percentile(durations, 50),
                "duration_p95_ms": _percentile(durations, 95)}

    return {"stratum": "expanded_30k_scale",
            "baseline": diagnostics("baseline"), "candidate": diagnostics("candidate")}


def expected_phase07_expanded_corpus_identity(*, fixture_root: Path, target_size: int,
                                              test_only: bool = False) -> dict[str, object]:
    """Compute deterministic public-corpus identity from fixed inputs, not a label.

    This mirrors ``canonical_content_tree_sha256`` without needing a retained
    30k tree.  The small-target escape hatch is intentionally pytest-only.
    """
    if target_size != 30000 and not (test_only and os.environ.get("PYTEST_CURRENT_TEST")):
        raise ValueError("phase07 expanded target must be 30000")
    root = Path(fixture_root)
    sources = sorted((path for path in root.rglob("*.md") if path.is_file()),
                     key=lambda path: path.relative_to(root).as_posix())
    if not sources or target_size < len(sources):
        raise ValueError("phase07 deterministic corpus source")
    source_members = sorted(
        (path.relative_to(root).as_posix(), path)
        for path in root.rglob("*") if path.is_file()
    )
    source_by_name = dict(source_members)
    synthetic_names = [f"phase07_distractors/hybrid-{ordinal:05d}.md"
                       for ordinal in range(target_size - len(sources))]
    digest = hashlib.sha256()
    for name in sorted([*source_by_name, *synthetic_names]):
        if name in source_by_name:
            contents = canonical_corpus_file_bytes(source_by_name[name])
        else:
            ordinal = int(name.rsplit("-", 1)[1].removesuffix(".md"))
            contents = public_distractor_bytes(
                canonical_corpus_file_bytes(sources[ordinal % len(sources)]), ordinal,
            )
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(contents); digest.update(b"\0")
    return {"expanded_content_tree_sha256": digest.hexdigest(),
            "expanded_member_count": len(source_by_name) + len(synthetic_names)}


def _materialize_phase07_expanded_corpus(*, fixture_root: Path, output_root: Path, target_size: int,
                                         test_only: bool) -> dict[str, object]:
    shutil.copytree(fixture_root, output_root)
    sources = sorted(output_root.rglob("*.md"))
    expected = expected_phase07_expanded_corpus_identity(
        fixture_root=fixture_root, target_size=target_size, test_only=test_only,
    )
    for ordinal in range(target_size - len(sources)):
        target = output_root / "phase07_distractors" / f"hybrid-{ordinal:05d}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(public_distractor_bytes(
            canonical_corpus_file_bytes(sources[ordinal % len(sources)]), ordinal,
        ))
    actual = {"expanded_content_tree_sha256": canonical_content_tree_sha256(output_root),
              "expanded_member_count": sum(1 for item in output_root.rglob("*") if item.is_file())}
    if actual != expected:
        raise ValueError("phase07 expanded corpus materialization identity")
    return actual


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, int(round(
        (percentile / 100) * (len(ordered) - 1)
    ))))
    return ordered[position]


def _candidate_result_observation(*, query: list[dict], results: list,
                                  latencies: list[float], baseline_quality: dict,
                                  binding: dict[str, object]) -> dict:
    page_recalls, evidence_recalls, reciprocal_ranks = [], [], []
    context_overflow = budget_violations = citation_violations = 0
    graph_validated = graph_unsupported = 0
    payload = []
    for specification, result in zip(query, results):
        gold = specification.get("relevant_pages", [])
        pages = result.candidates[:10]
        top5_hits = sum(1 for candidate in pages[:5] if _page_hit(candidate, gold))
        page_recalls.append(min(1.0, top5_hits / max(1, len(gold))))
        required = specification.get("required_facts") or []
        context = result.bundle.context_text or ""
        evidence_recalls.append(
            sum(fact in context for fact in required) / len(required) if required else 1.0
        )
        rank = next((ordinal for ordinal, candidate in enumerate(pages, 1)
                     if _page_hit(candidate, gold)), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        context_overflow += int(result.bundle.token_count > result.bundle.effective_budget_tokens)
        budget_violations += int(bool(result.bundle.budget_contract_violations()))
        citation_violations += len(_citation_violations(result.bundle))
        graph_validated += result.graph_validated_count
        graph_unsupported += sum(
            1 for item in result.bundle.items
            if item.inclusion_reason == "graph_expansion" and not item.evidence
        )
        payload.append({
            "binding": dict(binding),
            "observation": _candidate_result_payload(result),
        })
    page_recall = statistics.mean(page_recalls)
    evidence_recall = statistics.mean(evidence_recalls)
    mrr = statistics.mean(reciprocal_ranks)
    functional = statistics.mean(
        min(1.0, sum(_page_hit(candidate, spec.get("relevant_pages", []))
                     for candidate in result.candidates[:10]) / max(1, len(spec.get("relevant_pages", []))))
        for spec, result in zip(query, results)
    )
    failures = []
    for name, current in (
        ("page_recall_at_5", page_recall),
        ("evidence_recall_at_10", evidence_recall),
        ("mrr_at_10", mrr),
        (FUNCTIONAL_FINAL_RETRIEVAL_METRIC, functional),
    ):
        reference = baseline_quality.get(name)
        if reference is not None and current < float(reference) - 0.02:
            failures.append(name)
    if context_overflow or budget_violations or citation_violations or graph_unsupported:
        failures.append("zero-tolerance contract")
    return {
        "retrieval": {
            FUNCTIONAL_FINAL_RETRIEVAL_METRIC: round(functional, 4),
            "result_payload": payload,
        },
        "page": {"page_recall_at_5": round(page_recall, 4)},
        "evidence": {"evidence_recall_at_10": round(evidence_recall, 4)},
        "mrr": {"mrr_at_10": round(mrr, 4)},
        "latency": {
            "samples_s": [round(value, 6) for value in latencies],
            "p50_s": round(_percentile(latencies, 50), 6),
            "p95_s": round(_percentile(latencies, 95), 6),
        },
        "context": {"overflow_count": context_overflow, "budget_violation_count": budget_violations},
        "citation": {"violation_count": citation_violations},
        "graph": {"validated_count": graph_validated, "unsupported_count": graph_unsupported},
        "non_regression": {"baseline_refresh": False, "failures": failures},
    }


def run_candidate_hybrid_evaluation(
    wiki_src: Path, queries: list[dict], work_dir: Path, max_tokens: int,
    comparator_evidence: dict, *, baseline_quality: dict,
    hard_max_tokens: int | None = None,
) -> dict:
    """Build and measure every FLAT/SQ × ef binding through ``hybrid_search``."""
    validate_evidence(comparator_evidence)
    if not queries:
        raise ValueError("candidate hybrid evaluation requires queries")
    head, environment = _head_sha(), _decision_environment()
    pr_head, checkout = _source_identities()
    fixture_sha = _fixture_digest(wiki_src)
    evaluation_queries_sha = _stable_json_digest(queries)
    records = []
    for candidate in CANDIDATES:
        for query_ef in DECISION_EF_GRID:
            policy = CandidateQueryPolicy(candidate=candidate, query_ef=query_ef)
            root = work_dir / f"{candidate}-ef-{query_ef}"
            wi, wiki, build_time = _build(
                root, wiki_src, True, candidate_query_policy=policy,
            )
            planner = DefaultQueryPlanner(project_root=root)
            results, latencies, traces = [], [], []
            for ordinal, specification in enumerate(queries):
                started = time.perf_counter()
                result = hybrid_search(
                    wi, specification["query"], planner, k=10, max_tokens=max_tokens,
                    hard_max_tokens=hard_max_tokens, wiki_dir=wiki,
                    intent_override="auto", allow_local_fallback=True,
                )
                latency = time.perf_counter() - started
                results.append(result)
                latencies.append(latency)
                traces.append({
                    "ordinal": ordinal,
                    "query_sha256": hashlib.sha256(specification["query"].encode("utf-8")).hexdigest(),
                    "result_sha256": _stable_json_digest(_candidate_result_payload(result)),
                    "latency_s": round(latency, 6),
                })
            final_retrieval = _candidate_result_observation(
                query=queries, results=results, latencies=latencies,
                baseline_quality=baseline_quality,
                binding=policy.to_json(),
            )
            index_identity = _candidate_index_identity(wi=wi, root=root, policy=policy)
            run_id = _stable_json_digest({
                "candidate": candidate, "query_ef": query_ef,
                "build_id": index_identity["build_id"], "traces": traces,
            })
            invocation_digest = _stable_json_digest({"run_id": run_id, "traces": traces})
            result_digest = _stable_json_digest({"run_id": run_id, "final_retrieval": final_retrieval})
            records.append({
                "candidate": candidate,
                "query_ef": query_ef,
                "candidate_run_id": run_id,
                "candidate_index": {**index_identity, "build_time_s": round(build_time, 4)},
                "applied_policy": policy.to_json(),
                "hybrid_invocation": {
                    "entrypoint": "query.hybrid_search",
                    "trace_id": _stable_json_digest({"run_id": run_id, "entrypoint": "query.hybrid_search"}),
                    "digest": invocation_digest,
                    "query_count": len(queries),
                    "traces": traces,
                },
                "result_sha256": result_digest,
                "final_retrieval": final_retrieval,
                "input_binding": {
                    "fixture_sha256": fixture_sha,
                    "evaluation_queries_sha256": evaluation_queries_sha,
                    "comparator_corpus_sha256": comparator_evidence["corpus"]["sha256"],
                    "comparator_queries_sha256": comparator_evidence["queries"]["sha256"],
                },
                "head_sha": head,
                "pr_head_sha": pr_head,
                "actions_merge_checkout_sha": checkout,
                "environment": environment,
                "corpus_sha256": comparator_evidence["corpus"]["sha256"],
                "queries_sha256": comparator_evidence["queries"]["sha256"],
            })
    packet = {
        "schema_version": FINAL_RETRIEVAL_DECISION_SCHEMA_VERSION,
        "head_sha": head,
        "pr_head_sha": pr_head,
        "actions_merge_checkout_sha": checkout,
        "environment": environment,
        "comparator_schema_version": EVIDENCE_SCHEMA_VERSION,
        "records": records,
    }
    return validate_candidate_decision_records(packet, comparator_evidence)


def run_phase07_representative_campaign(
    *, mode: str, size: int, baseline: dict | None = None, finalist: dict | None = None,
    work_dir: Path, authorization: str, embed=None, query_limit: int | None = None,
) -> dict:
    """Exercise the public build/load/``hybrid_search`` path for one pinned scale.

    ``embed`` is an in-process dependency seam used only by tiny integration
    tests.  Requests never reach it; normal campaign execution therefore uses
    the established model-backed ``WikiIndex.build`` facade.  The generated
    corpus stays in the temporary work directory and is never an artifact.
    """
    if mode not in {"representative_ann", "hybrid_non_regression"} or size not in {1000, 10000, 30000}:
        raise ValueError("pinned representative campaign")
    root = Path(work_dir) / f"representative-{mode}-{size}"
    if root.exists():
        shutil.rmtree(root)
    original_wiki = root / "original" / "Wiki"
    expanded_wiki = root / "expanded" / "Wiki"
    shutil.copytree(FIXTURES_WIKI, original_wiki)
    shutil.copytree(FIXTURES_WIKI, expanded_wiki)
    # Public deterministic distractors preserve fixture-language structure;
    # the query is never written into this corpus.
    source_pages = sorted(expanded_wiki.rglob("*.md"))
    if not source_pages:
        raise ValueError("representative public fixture unavailable")
    for ordinal in range(max(0, size - len(source_pages))):
        target = expanded_wiki / "phase07_distractors" / f"public-{ordinal:05d}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(public_distractor_bytes(
            canonical_corpus_file_bytes(source_pages[ordinal % len(source_pages)]), ordinal,
        ))
    baseline = baseline or dict(PHASE07_CURRENT_BASELINE)
    finalist = finalist or {"candidate": "ivf-hnsw-sq", "m": 16, "ef_construction": 300, "query_ef": 200, "refine_factor": None}
    def build_candidate(name, config, wiki):
        index_dir = root / name / ".index"
        policy = CandidateQueryPolicy(
            candidate=config["candidate"], query_ef=config["query_ef"],
            build_policy=CandidateBuildPolicy(candidate=config["candidate"], m=config["m"], ef_construction=config["ef_construction"]),
        )
        if embed is None:
            wi = WikiIndex(index_dir); wi.build(wiki, full_rebuild=True, candidate_query_policy=policy)
        else:
            from build_index import build_storage_contract
            build_storage_contract(wiki, index_dir, embed=embed, candidate_query_policy=policy)
            wi = WikiIndex(index_dir)
        wi.load()
        return wi
    original_baseline = build_candidate("original-baseline", baseline, original_wiki)
    baseline_wi = build_candidate("expanded-baseline", baseline, expanded_wiki)
    finalist_wi = build_candidate("expanded-finalist", finalist, expanded_wiki)
    if embed is not None:
        # Query embedding is the matching half of the explicitly injected test
        # encoder.  Storage, build, load, fusion and citation/context paths are
        # still the public production implementations.
        class _InjectedEncoder:
            def encode(self, texts, **_kwargs):
                return embed(texts)
        original_baseline._embedder = _InjectedEncoder(); baseline_wi._embedder = _InjectedEncoder(); finalist_wi._embedder = _InjectedEncoder()
    planner = DefaultQueryPlanner(project_root=root)
    original_queries = load_queries(HERE / "queries.jsonl")
    if len(original_queries) != 105: raise ValueError("pinned original hybrid query manifest")
    queries = original_queries[:query_limit] if query_limit is not None else original_queries
    original_observations, expanded_observations = [], []
    for ordinal, item in enumerate(queries):
        query = item["query"]; started = time.perf_counter()
        original_result = hybrid_search(original_baseline, query, planner, k=10, max_tokens=4096, wiki_dir=original_wiki, intent_override="auto", allow_local_fallback=True)
        original_observations.append({"ordinal": ordinal, "query_sha256": hashlib.sha256(query.encode()).hexdigest(), "baseline": _candidate_result_payload(original_result), "duration_ms": (time.perf_counter() - started) * 1000})
        baseline_result = hybrid_search(baseline_wi, query, planner, k=10, max_tokens=4096, wiki_dir=expanded_wiki, intent_override="auto", allow_local_fallback=True)
        finalist_result = hybrid_search(finalist_wi, query, planner, k=10, max_tokens=4096, wiki_dir=expanded_wiki, intent_override="auto", allow_local_fallback=True)
        expanded_observations.append({"ordinal": ordinal, "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                             "baseline": _candidate_result_payload(baseline_result), "finalist": _candidate_result_payload(finalist_result),
                             "duration_ms": (time.perf_counter() - started) * 1000})
    # Separate natural-language ANN exact stratum: pinned query rows are never
    # written to the corpus and use normal candidate ANN calls only.
    ann_queries = load_phase07_personal_wiki_ann_queries(HERE / "personal-wiki-ann-queries.jsonl")
    ann_queries = ann_queries[:query_limit] if query_limit is not None else ann_queries
    ann_rows = []
    for item in ann_queries:
        query_text = item["query"]
        baseline_vector = baseline_wi._get_embedder().encode([query_text])[0]
        finalist_vector = finalist_wi._get_embedder().encode([query_text])[0]
        baseline_repo, finalist_repo = baseline_wi._get_repository(), finalist_wi._get_repository()
        exact_ids = [str(row.get("chunk_id", "")) for row in baseline_repo.search_dense_exact(baseline_vector, metric="cosine", limit=20)]
        finalist_exact_ids = [str(row.get("chunk_id", "")) for row in finalist_repo.search_dense_exact(finalist_vector, metric="cosine", limit=20)]
        baseline_ids = [str(row.get("chunk_id", "")) for row in baseline_repo.search_dense_eval(baseline_vector, metric="cosine", limit=20, ef=baseline["query_ef"])]
        finalist_ids = [str(row.get("chunk_id", "")) for row in finalist_repo.search_dense_eval(finalist_vector, metric="cosine", limit=20, ef=finalist["query_ef"])]
        ann_rows.append({"query_id": item["query_id"], "baseline_exact_top_20": exact_ids, "finalist_exact_top_20": finalist_exact_ids, "baseline_top_20": baseline_ids,
                         "finalist_top_20": finalist_ids,
                         "baseline_recall_at_10": len(set(exact_ids[:10]) & set(baseline_ids[:10])) / 10, "baseline_recall_at_20": len(set(exact_ids) & set(baseline_ids)) / 20,
                         "finalist_recall_at_10": len(set(finalist_exact_ids[:10]) & set(finalist_ids[:10])) / 10, "finalist_recall_at_20": len(set(finalist_exact_ids) & set(finalist_ids)) / 20})
    separation = _representative_indexed_query_separation(baseline_repo, [item["query"] for item in ann_queries])
    return {
        "mode": mode, "scale": size, "candidate_configs": {"baseline": baseline, "finalist": finalist},
        "original_fixture": {"query_count": len(queries), "corpus_sha256": canonical_content_tree_sha256(original_wiki), "absolute_baseline": original_observations},
        "expanded": {"size": size, "corpus_sha256": canonical_content_tree_sha256(expanded_wiki), "paired_observations": expanded_observations},
        "personal_wiki_ann_exact": {"query_count": len(ann_rows), **separation, "rows": ann_rows},
        "hybrid_invocation": {"entrypoint": "query.hybrid_search", "original_baseline_calls": len(original_observations), "baseline_calls": len(expanded_observations), "finalist_calls": len(expanded_observations)},
        "authorization": authorization,
    }


def _load_phase07_frozen_runner_embedder():
    """Load the verified frozen-source model behind the private runner boundary."""
    from eval.phase07_frozen_base import load_verified_frozen_embedder
    from build_index import SKILL_EMBEDDER_DIR

    return load_verified_frozen_embedder(SKILL_EMBEDDER_DIR)


def _run_phase07_hybrid_campaign_with_capability(*, capability: object, work_dir: Path, embed=None,
                                                  query_limit: int | None = None,
                                                  progress_sink=None,
                                                  frozen_dir: Path | None = None,
                                                  tokenizer: object | None = None,
                                                  _expected_frozen_corpus_identity: dict | None = None) -> dict:
    """Run one sealed hybrid *role* with an operator-minted capability.

    A role is either the shared baseline or one candidate.  It builds exactly
    two independent indexes (the committed fixture and its expanded corpus),
    then makes one normal public ``hybrid_search`` call per query against each.
    Pairing, metric aggregation, and gate decisions happen only after the
    three sealed role artifacts are reconciled; this runner never has both
    sides of a comparison in memory.  ``embed`` plus ``query_limit`` is a
    pytest-only finite integration seam; normal execution materializes the
    sealed 30k corpus and evaluates all 105 committed queries.  The
    underscored frozen-corpus identity is an internal test seam only: neither
    dispatch nor the CLI accepts it, so default production validation remains
    bound to the committed 30k identity.
    """
    from eval.phase07_operator_gate import _consume_hybrid_execution_capability

    consumed = _consume_hybrid_execution_capability(capability)
    if isinstance(consumed, dict) and set(consumed) == {"member", "frozen_prepare"}:
        bundle = consumed["member"]
        frozen_prepare = consumed["frozen_prepare"]
    else:
        # Legacy non-frozen PR topology intentionally remains member-only.
        bundle = consumed
        frozen_prepare = None
    injected_embed = embed is not None
    if injected_embed and not os.environ.get("PYTEST_CURRENT_TEST"):
        raise ValueError("injected hybrid encoder is pytest-only")
    if tokenizer is not None and (not injected_embed or not os.environ.get("PYTEST_CURRENT_TEST")):
        raise ValueError("injected hybrid tokenizer is pytest-only")
    if query_limit is not None and (not injected_embed or not os.environ.get("PYTEST_CURRENT_TEST")
                                    or not isinstance(query_limit, int) or not 1 <= query_limit < 105):
        raise ValueError("hybrid query limit is pytest-only")
    frozen_tokenizer = tokenizer
    frozen_query_embedder = None
    if frozen_dir is not None and injected_embed and frozen_tokenizer is None:
        raise ValueError("injected frozen tokenizer is required")
    if frozen_dir is not None and frozen_tokenizer is None:
        # The source validator and private clone share this one verified
        # tokenizer.  Keep the public ``embed`` argument untouched: its None
        # value is what makes the original side use ``WikiIndex.build``.
        frozen_query_embedder = _load_phase07_frozen_runner_embedder()
        frozen_tokenizer = frozen_query_embedder.tokenizer
    if frozen_dir is not None and frozen_tokenizer is None:
        raise ValueError("frozen tokenizer is required")
    role, config = bundle["role"], bundle["config"]
    frozen_source_evidence: dict[str, object] = {}
    root = Path(work_dir) / f"hybrid-{role}-m{config['m']}"
    if root.exists():
        shutil.rmtree(root)
    original_wiki = root / "original" / "Wiki"
    shutil.copytree(FIXTURES_WIKI, original_wiki)
    # A local test exercises the production topology but not a 30k benchmark.
    source_count = sum(1 for page in FIXTURES_WIKI.rglob("*.md") if page.is_file())
    target_size = source_count + 2 if injected_embed else bundle["scale"]
    if frozen_dir is None:
        expanded_wiki = root / "expanded" / "Wiki"
        expanded_identity = _materialize_phase07_expanded_corpus(
            fixture_root=FIXTURES_WIKI, output_root=expanded_wiki, target_size=target_size,
            test_only=embed is not None,
        )
    else:
        # The downloaded tree is validated before a clone can be created.  It
        # remains read-only evidence; only ``root/expanded-private`` receives
        # an HNSW, manifest, generation record and ACTIVE_INDEX pointer.
        from eval.phase07_frozen_base import validate_frozen_base
        frozen_dir = Path(frozen_dir).resolve()
        expanded_wiki = frozen_dir / "Wiki"
        base_tree_sha256 = validate_frozen_base(
            frozen_dir, expected_wiki_root=expanded_wiki,
            tokenizer=frozen_tokenizer,
            expected_corpus_identity=_expected_frozen_corpus_identity,
        )
        if not isinstance(frozen_prepare, dict):
            raise ValueError("frozen source requires collector-bound prepare provenance")
        descriptor = json.loads((frozen_dir / "frozen-base.json").read_text(encoding="utf-8"))
        if descriptor.get("record_self_sha256") != frozen_prepare.get("descriptor_self_sha256") \
                or base_tree_sha256 != frozen_prepare.get("base_tree_sha256") \
                or descriptor.get("model_manifest_sha256") != frozen_prepare.get("model_manifest_sha256") \
                or descriptor.get("corpus_manifest_sha256") != frozen_prepare.get("corpus_manifest_sha256") \
                or descriptor.get("generator_recipe_sha256") != frozen_prepare.get("generator_recipe_sha256") \
                or descriptor.get("runtime") != frozen_prepare.get("runtime"):
            raise ValueError("frozen source collector provenance mismatch")
        expanded_identity = {
            "expanded_content_tree_sha256": canonical_content_tree_sha256(expanded_wiki),
            "expanded_member_count": sum(1 for path in expanded_wiki.rglob("*.md") if path.is_file()),
        }
        frozen_source_evidence = {
            "frozen_prepare": frozen_prepare,
            "source_before_sha256": base_tree_sha256,
            "source_after_sha256": base_tree_sha256,
        }

    def build_candidate(name: str, config: dict, wiki: Path):
        corpus_started = time.perf_counter()

        def emit_stage(stage: str) -> None:
            marker = {
                "role": role,
                "corpus": name,
                "stage": stage,
                "state": "complete",
                "elapsed_ms": round((time.perf_counter() - corpus_started) * 1000, 6),
            }
            if progress_sink is None:
                print(f"[phase07-hybrid-progress] {json.dumps(marker, sort_keys=True)}", flush=True)
            else:
                progress_sink(marker)

        policy = CandidateQueryPolicy(
            candidate="ivf-hnsw-sq", query_ef=config["query_ef"],
            build_policy=CandidateBuildPolicy(candidate="ivf-hnsw-sq", m=config["m"], ef_construction=config["ef_construction"]),
        )
        index_dir = root / name / ".index"
        if name == "expanded" and frozen_dir is not None:
            from eval.phase07_frozen_base import finalize_private_role

            finalized = finalize_private_role(
                frozen_dir=frozen_dir,
                target_dir=root / "expanded-private-clone",
                expected_wiki_root=wiki,
                candidate_query_policy=policy,
                tokenizer=frozen_tokenizer,
                publish_index_dir=index_dir,
                expected_corpus_identity=_expected_frozen_corpus_identity,
            )
            index = WikiIndex(Path(finalized["index_dir"]))
            index.load()
            emit_stage("private_clone_hnsw_publish_load")
            return index
        if embed is None:
            index = WikiIndex(index_dir)
            index.build(
                wiki, full_rebuild=True, candidate_query_policy=policy,
                progress_sink=emit_stage,
            )
        else:
            from build_index import build_storage_contract
            build_storage_contract(
                wiki, index_dir, embed=embed, candidate_query_policy=policy,
                progress_sink=emit_stage,
            )
            index = WikiIndex(index_dir)
        # ``hybrid_search`` consumes the graph sidecar through the public
        # index path; all four independently built indexes require one.
        write_graph_artifact(wiki, index_dir)
        emit_stage("graph_artifact")
        index.load()
        emit_stage("load")
        return index

    original_index = build_candidate("original", config, original_wiki)
    print(f"[phase07-hybrid] role={role} build=original complete", flush=True)
    expanded_index = build_candidate("expanded", config, expanded_wiki)
    print(f"[phase07-hybrid] role={role} build=expanded complete", flush=True)
    if embed is not None:
        class _InjectedEncoder:
            def encode(self, texts, **_kwargs):
                return embed(texts)
        for index in (original_index, expanded_index):
            index._embedder = _InjectedEncoder()
    elif frozen_query_embedder is not None:
        class _FrozenSourceEncoder:
            def encode(self, texts, **_kwargs):
                return frozen_query_embedder.embed(list(texts))

        # The private clone has no model object of its own.  Its vectors were
        # sealed with the verified frozen-source model above; query through the
        # same dependency without converting this production run into an
        # injected-embedder/test build.
        expanded_index._embedder = _FrozenSourceEncoder()
    queries = load_queries(HERE / "queries.jsonl")
    if len(queries) != bundle["query_count"]:
        raise ValueError("sealed hybrid original query manifest")
    queries = queries[:query_limit] if query_limit is not None else queries
    planner = DefaultQueryPlanner(project_root=root)

    def observe(index, query: str, wiki: Path) -> tuple[dict, object]:
        started = time.perf_counter()
        result = hybrid_search(index, query, planner, k=10, max_tokens=4096, wiki_dir=wiki,
                               intent_override="auto", allow_local_fallback=True)
        payload = _candidate_result_payload(result)
        return ({"result": payload, "duration_ms": round((time.perf_counter() - started) * 1000, 6)}, result)

    original, expanded = [], []
    for ordinal, item in enumerate(queries):
        query = item["query"]
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        original_row, _ = observe(original_index, query, original_wiki)
        expanded_row, _ = observe(expanded_index, query, expanded_wiki)
        original.append({"ordinal": ordinal, "query_sha256": query_sha256, "observation": original_row})
        expanded.append({"ordinal": ordinal, "query_sha256": query_sha256, "observation": expanded_row})
        completed = ordinal + 1
        if completed % 15 == 0 or completed == len(queries):
            print(f"[phase07-hybrid] role={role} original_queries={completed}/{len(queries)} complete", flush=True)
            print(f"[phase07-hybrid] role={role} expanded_queries={completed}/{len(queries)} complete", flush=True)
    if frozen_dir is not None:
        from eval.phase07_frozen_base import validate_frozen_base
        source_after_sha256 = validate_frozen_base(
            frozen_dir, expected_wiki_root=expanded_wiki,
            tokenizer=frozen_tokenizer,
            expected_corpus_identity=_expected_frozen_corpus_identity,
        )
        if source_after_sha256 != base_tree_sha256:
            raise ValueError("frozen source mutated during private hybrid lifecycle")
        frozen_source_evidence["source_after_sha256"] = source_after_sha256
    return {
        "schema_version": 1, "campaign_stage": "hybrid", "bundle_sha256": bundle["record_self_sha256"],
        "role": role, "config": config,
        "planned_scale": bundle["scale"], "executed_scale": target_size,
        **expanded_identity, **frozen_source_evidence,
        "query_count": len(queries), "authorization": "none",
        "original_observations": original,
        "expanded_observations": expanded,
        "hybrid_invocation": {
            "entrypoint": "query.hybrid_search", "candidate_aware_public_arguments": False,
            "original_calls": len(original), "expanded_calls": len(expanded),
        },
        "campaign_progress": {
            "role": role, "original_completed": len(original), "expanded_completed": len(expanded),
        },
    }


def _representative_indexed_query_separation(repository, queries: list[str]) -> dict[str, object]:
    """Validate actual dense-table text against unindexed representative queries."""
    indexed = {hashlib.sha256(str(row["text"]).encode()).hexdigest() for row in repository._dense_table().to_arrow().to_pylist()}
    query_digests = [hashlib.sha256(query.encode()).hexdigest() for query in queries]
    return validate_indexed_query_digest_separation(indexed_row_digests=indexed, query_row_digests=query_digests)


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
    if manifest.get("format_version") == 6:
        # Phase 06：固定策略 manifest——生产构建必须携带通过域验证器的
        # candidate_publication_evidence；eval candidate 构建除外。
        if isinstance(manifest.get("candidate_query_policy"), dict):
            return manifest.get("benchmark") or {}
        evidence = manifest.get("candidate_publication_evidence")
        if not isinstance(evidence, dict):
            raise ValueError(
                "format-6 production manifests require candidate_publication_evidence"
            )
        from obsidian_wiki.domain.index_models import CandidatePublicationEvidence
        from obsidian_wiki.domain.index_policy import (
            load_ann_policy_file,
            validate_candidate_publication_evidence,
        )
        try:
            record = CandidatePublicationEvidence(**evidence)
        except TypeError as exc:
            raise ValueError(f"publication evidence fields invalid: {exc}") from exc
        validate_candidate_publication_evidence(record, load_ann_policy_file())
        return manifest.get("benchmark") or {}
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


def write_graph_artifact(wiki: Path, index_dir: Path) -> None:
    """Build the production graph sidecar used by both evaluation entry points."""
    try:
        graph = _bg.build_graph(wiki)
        stats = _bg.compute_4_signals(graph)
        communities = _bg.detect_communities(graph)
        graph_json = {
            "nodes": [{"id": node, **{key: value for key, value in data.items()
                                        if key != "signals"}}
                      for node, data in graph.nodes(data=True)],
            "edges": [{"source": source, "target": target,
                       "weight": round(data.get("weight", 1.0), 4),
                       "signal": sorted(data.get("signals", set()))[0]
                       if data.get("signals") else "unknown",
                       "signals": sorted(data.get("signals", set()))}
                      for source, target, data in graph.edges(data=True)],
            "signals": stats,
            "communities": communities,
        }
        (index_dir / "graph.json").write_text(
            json.dumps(graph_json, ensure_ascii=False, indent=2, default=list),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 图谱生成失败（不影响主评测）: {exc}", file=sys.stderr)


def _build(project_root: Path, wiki_src: Path, full_rebuild: bool,
           *, candidate_query_policy: CandidateQueryPolicy | None = None):
    # Phase 06：普通评测构建走固定生产路径（批准 SQ/ef）；eval candidate
    # 构建显式传 candidate_query_policy。
    wiki = _stage_project(project_root, wiki_src)
    idx = project_root / ".index"
    wi = WikiIndex(idx)
    t0 = time.perf_counter()
    wi.build(
        wiki, full_rebuild=full_rebuild,
        candidate_query_policy=candidate_query_policy,
    )
    dt = time.perf_counter() - t0
    wi.load()
    # 生成图谱（供 hybrid_search 的 relations 扩展通道使用）
    write_graph_artifact(wiki, idx)
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
    main_wi, main_wiki, build_time = _build(main_root, wiki_src, full_rebuild=True)
    main_benchmark = _active_benchmark_contract(main_wi)
    planner = DefaultQueryPlanner(project_root=main_root)

    # 2) ANN candidate（独立 project，复用语义内容；auto 选择受自检的索引策略）
    ann_wi = None
    ann_build_time = None
    ann_benchmark = None
    if build_ann:
        ann_root = work_dir / "ann"
        ann_wi, _, ann_build_time = _build(ann_root, wiki_src, full_rebuild=True)
        ann_benchmark = _active_benchmark_contract(ann_wi)

    # 3) 逐查询评测
    page_recalls, evid_recalls, mrrs, exact_hits, functional_ann_overlaps = [], [], [], [], []
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
            functional_ann_overlaps.append(
                len(exact_ids & ann_ids) / len(exact_ids) if exact_ids else 1.0
            )

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
            # This 157-row fixture verifies the real model-backed retrieval path;
            # it is explicitly not a large-scale ANN-quality KPI.
            FUNCTIONAL_FINAL_RETRIEVAL_METRIC: (
                round(statistics.mean(functional_ann_overlaps), 4)
                if functional_ann_overlaps else None
            ),
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
    wi, wiki, _ = _build(root, wiki_src, full_rebuild=True)
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


def load_phase07_personal_wiki_ann_queries(path: Path) -> list[dict]:
    """Load the separately pinned 256 natural-language ANN truth queries.

    These records are intentionally only an evaluation input; passing them to
    ``hybrid_search`` or indexing their text is not part of this boundary.
    """
    queries = load_queries(path)
    if len(queries) != 256 or any(
        item.get("schema_version") != 1
        or item.get("stratum") != "natural_language_ann_exact"
        or not isinstance(item.get("query_id"), str)
        or not isinstance(item.get("query"), str)
        for item in queries
    ):
        raise ValueError("Phase 7 natural-language ANN query manifest")
    if len({item["query_id"] for item in queries}) != 256:
        raise ValueError("duplicate Phase 7 ANN query ID")
    return queries


def _configure_cli_text_streams() -> None:
    """Use UTF-8 for CLI output when the host exposes reconfigurable streams."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass


def main():
    _configure_cli_text_streams()
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
    ap.add_argument("--decision-evidence", type=Path,
                    help="Validated held-out comparator artifact used only to bind decision records")
    ap.add_argument("--decision-output", type=Path, default=HERE / "model-ann-decision.json",
                    help="Machine-readable model-backed decision records (never a production policy)")
    ap.add_argument("--validate-candidate-hybrid-evidence", type=Path,
                    help="Validate an existing candidate-specific hybrid packet and exit")
    ap.add_argument("--validate-ann-policy", action="store_true",
                    help="Validate the tracked eval/ann-policy.json against the approved "
                         "Phase 06 contract (type/ef/floors/digests) and exit")
    args = ap.parse_args()

    if args.validate_ann_policy:
        # Phase 06：把源码控制的批准策略作为身份门禁——任何字段被篡改即失败。
        from obsidian_wiki.domain.index_policy import (
            PolicyError,
            load_ann_policy_file,
            production_policy_sha256,
        )
        try:
            policy = load_ann_policy_file()
        except PolicyError as exc:
            print(f"[ERROR] ann policy record invalid: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({
            "valid": True,
            "selected_index_type": policy.selected_index_type,
            "lancedb_index_type": policy.lancedb_index_type,
            "query_ef": policy.query_ef,
            "recall_at_10_floor": policy.recall_at_10_floor,
            "recall_at_20_floor": policy.recall_at_20_floor,
            "policy_sha256": production_policy_sha256(policy),
            "comparator_sha256": policy.comparator_sha256,
        }, sort_keys=True))
        return 0

    if args.validate_candidate_hybrid_evidence is not None:
        if args.decision_evidence is None:
            print("[ERROR] --decision-evidence is required for candidate validation", file=sys.stderr)
            return 2
        comparator_evidence = json.loads(args.decision_evidence.read_text(encoding="utf-8"))
        validate_evidence(comparator_evidence)
        packet = json.loads(args.validate_candidate_hybrid_evidence.read_text(encoding="utf-8"))
        validate_candidate_decision_records(packet, comparator_evidence)
        print(json.dumps({"valid": True, "records": len(packet["records"])}, sort_keys=True))
        return 0

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

    if args.decision_evidence is not None:
        comparator_evidence = json.loads(args.decision_evidence.read_text(encoding="utf-8"))
        baseline_quality = {}
        if args.baselines.exists():
            baseline_quality = json.loads(
                args.baselines.read_text(encoding="utf-8")
            ).get("quality", {})
        packet = run_candidate_hybrid_evaluation(
            args.wiki, queries, work_dir / "candidate-hybrid", args.max_tokens,
            comparator_evidence, baseline_quality=baseline_quality,
            hard_max_tokens=args.hard_max_tokens,
        )
        args.decision_output.parent.mkdir(parents=True, exist_ok=True)
        args.decision_output.write_text(
            json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
        )

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
    if failures:
        print("\n[FAIL] 评测未通过：", file=sys.stderr)
        for f in failures:
            print("  - " + f, file=sys.stderr)
        return 1
    print("\n[PASS] 评测通过，无回归。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
