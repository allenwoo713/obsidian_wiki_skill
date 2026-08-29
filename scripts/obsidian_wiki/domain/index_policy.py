"""Pure, deterministic D-02/D-03 candidate ANN selection policy."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from obsidian_wiki.domain.index_models import (
    AnnDecisionEvidence,
    ANN_DECISION_EVIDENCE_SCHEMA_VERSION,
    ANN_POLICY_SCHEMA_VERSION,
    BenchmarkObservation,
    CandidatePublicationEvidence,
    CANDIDATE_PUBLICATION_EVIDENCE_SCHEMA_VERSION,
    IndexStats,
    ProductionAnnPolicy,
    VectorPolicyDecision,
)


class PolicyError(ValueError):
    """Raised for missing or malformed observations, never as a valid fallback."""


# ---- Phase 06（issue #49）：固定生产 ANN 契约 ---------------------------------
#
# 决策授权（AnnDecisionEvidence）与发布证据（CandidatePublicationEvidence）是
# 两种不可互换的记录：前者只绑定锁定规模的 A/B 决策工件，后者只绑定一次 staged
# candidate 的实际行数与质量观测。两者互相喂给对方的验证器都会 fail-closed。

_DECISION_CORPUS_ROWS = 77348
_DECISION_HELD_OUT_QUERIES = 256
_DECISION_CANDIDATES: tuple[str, ...] = ("ivf-hnsw-flat", "ivf-hnsw-sq")
_DECISION_EF_GRID: tuple[int, ...] = (30, 50, 75, 100, 150, 200)
_PRODUCTION_RESULT_LIMIT = 20
_VALIDATION_QUERY_SOURCE = "deterministic_disjoint_unit_v1"

_POLICY_DIGEST_FIELDS = (
    "policy_schema_version", "selected_index_type", "lancedb_index_type", "metric",
    "dimensions", "num_partitions", "m", "ef_construction", "query_ef",
    "recall_at_10_floor", "recall_at_20_floor", "comparator_sha256",
    "candidate_hybrid_sha256", "reconciliation_sha256", "evidence_run_url",
    "retention_days",
)

_PHASE6_DECISION_FIELDS = frozenset({
    "approval_scope", "configuration", "corpus_rows", "held_out_queries",
    "candidates", "ef_grid", "pr_head_sha", "approved_by", "approved_at",
    "comparator_sha256", "candidate_hybrid_sha256", "reconciliation_sha256",
    "evidence_run_url",
})
_PHASE7_DECISION_FIELDS = frozenset({
    "configuration", "approved_by", "approval_reference", "approved_at",
    "sealed_dense_ledger_sha256", "dense_evidence_head_sha",
})
_PHASE6_CONFIGURATION = {"m": 16, "ef_construction": 300, "query_ef": 100}
_PHASE7_CONFIGURATION = {"m": 20, "ef_construction": 300, "query_ef": 300}
_PHASE6_APPROVAL_SCOPE = (
    "Inherited Phase 6 recall-floor and hybrid-baseline provenance only; it does not "
    "approve the Phase 7 m=20 selection."
)
_PHASE6_PR_HEAD_SHA = "2e1de3c8e30794bf67b160dbe37e7fca41889bea"
_PHASE6_APPROVED_BY = "root/user (Derek), recorded in issue #49 comment 5312709866"
_PHASE6_APPROVED_AT = "2026-08-17T06:40:12Z"
_PHASE7_APPROVED_BY = "repository owner/user, recorded in D-28 / Issue #50"
_PHASE7_APPROVAL_REFERENCE = "D-28 / Issue #50"
_PHASE7_APPROVED_AT = "2026-08-28"
_SEALED_DENSE_LEDGER_SHA256 = (
    "71335b6bfa03f24368414ae56a22fd8896d4479c6bfbe871c36a14b26e3b211b"
)
_DENSE_EVIDENCE_HEAD_SHA = "2f15d6a4fef54dda9b0f4a258e78898e2ef6ea57"


def _require_exact_mapping(
    name: str, value: object, expected_fields: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise PolicyError(f"ann decision {name} has an incompatible shape")
    return value


def _require_exact_configuration(
    name: str, value: object, expected: Mapping[str, int],
) -> None:
    configuration = _require_exact_mapping(name, value, frozenset(expected))
    for field_name, expected_value in expected.items():
        if type(configuration[field_name]) is not int or configuration[field_name] != expected_value:
            raise PolicyError(f"ann decision {name}.{field_name} does not match the fixed policy")


def _require_exact_value(name: str, value: object, expected: object) -> None:
    if value != expected or type(value) is not type(expected):
        raise PolicyError(f"ann decision {name} does not match the recorded provenance")


def _validate_informational_decision(
    decision: object, policy: ProductionAnnPolicy,
) -> None:
    """Fail closed on the tracked, human-readable Phase 6/7 policy provenance."""
    record = _require_exact_mapping(
        "record", decision,
        frozenset({"phase6_inherited_provenance", "phase7_selection"}),
    )
    phase6 = _require_exact_mapping(
        "phase6_inherited_provenance", record["phase6_inherited_provenance"],
        _PHASE6_DECISION_FIELDS,
    )
    phase7 = _require_exact_mapping(
        "phase7_selection", record["phase7_selection"], _PHASE7_DECISION_FIELDS,
    )
    _require_exact_configuration(
        "phase6_inherited_provenance.configuration", phase6["configuration"],
        _PHASE6_CONFIGURATION,
    )
    _require_exact_configuration(
        "phase7_selection.configuration", phase7["configuration"],
        _PHASE7_CONFIGURATION,
    )
    if (
        policy.m, policy.ef_construction, policy.query_ef
    ) != (
        _PHASE7_CONFIGURATION["m"], _PHASE7_CONFIGURATION["ef_construction"],
        _PHASE7_CONFIGURATION["query_ef"],
    ):
        raise PolicyError("top-level ann policy does not match the fixed Phase 7 selection")
    for field_name, expected in (
        ("approval_scope", _PHASE6_APPROVAL_SCOPE),
        ("corpus_rows", _DECISION_CORPUS_ROWS),
        ("held_out_queries", _DECISION_HELD_OUT_QUERIES),
        ("candidates", list(_DECISION_CANDIDATES)),
        ("ef_grid", list(_DECISION_EF_GRID)),
        ("pr_head_sha", _PHASE6_PR_HEAD_SHA),
        ("approved_by", _PHASE6_APPROVED_BY),
        ("approved_at", _PHASE6_APPROVED_AT),
        ("comparator_sha256", policy.comparator_sha256),
        ("candidate_hybrid_sha256", policy.candidate_hybrid_sha256),
        ("reconciliation_sha256", policy.reconciliation_sha256),
        ("evidence_run_url", policy.evidence_run_url),
    ):
        _require_exact_value(f"phase6_inherited_provenance.{field_name}", phase6[field_name], expected)
    for field_name, expected in (
        ("approved_by", _PHASE7_APPROVED_BY),
        ("approval_reference", _PHASE7_APPROVAL_REFERENCE),
        ("approved_at", _PHASE7_APPROVED_AT),
        ("sealed_dense_ledger_sha256", _SEALED_DENSE_LEDGER_SHA256),
        ("dense_evidence_head_sha", _DENSE_EVIDENCE_HEAD_SHA),
    ):
        _require_exact_value(f"phase7_selection.{field_name}", phase7[field_name], expected)


def production_policy_sha256(policy: ProductionAnnPolicy) -> str:
    """策略记录的稳定 digest——发布证据据此绑定它实际验证的策略。"""
    payload = {name: getattr(policy, name) for name in _POLICY_DIGEST_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_ann_policy_record(data: Mapping) -> ProductionAnnPolicy:
    """从 ``eval/ann-policy.json`` 的映射构造并校验批准策略（fail-closed）。"""
    if not isinstance(data, Mapping):
        raise PolicyError("ann policy record must be a mapping")
    required = (set(_POLICY_DIGEST_FIELDS) - {"policy_schema_version"}) | {
        "schema_version", "decision",
    }
    missing = required - set(data)
    if missing:
        raise PolicyError(f"ann policy record missing fields: {sorted(missing)}")
    if set(data) != required:
        raise PolicyError("ann policy record contains unknown fields")
    try:
        policy = ProductionAnnPolicy(
            policy_schema_version=data["schema_version"],
            selected_index_type=data["selected_index_type"],
            lancedb_index_type=data["lancedb_index_type"],
            metric=data["metric"],
            dimensions=data["dimensions"],
            num_partitions=data["num_partitions"],
            m=data["m"],
            ef_construction=data["ef_construction"],
            query_ef=data["query_ef"],
            recall_at_10_floor=data["recall_at_10_floor"],
            recall_at_20_floor=data["recall_at_20_floor"],
            comparator_sha256=data["comparator_sha256"],
            candidate_hybrid_sha256=data["candidate_hybrid_sha256"],
            reconciliation_sha256=data["reconciliation_sha256"],
            evidence_run_url=data["evidence_run_url"],
            retention_days=data["retention_days"],
        )
    except (TypeError, ValueError) as exc:
        raise PolicyError(f"ann policy record is invalid: {exc}") from exc
    _validate_informational_decision(data["decision"], policy)
    return policy


def default_ann_policy_path() -> Path:
    """源码控制的批准策略记录位置（scripts/obsidian_wiki/domain → 仓库根）。"""
    return Path(__file__).resolve().parents[3] / "eval" / "ann-policy.json"


_ANN_POLICY_CACHE: dict[str, ProductionAnnPolicy] = {}


def load_ann_policy_file(path: Path | None = None) -> ProductionAnnPolicy:
    """加载并缓存默认批准策略；记录缺失/非法时 fail-closed。"""
    resolved = Path(path) if path is not None else default_ann_policy_path()
    key = str(resolved)
    cached = _ANN_POLICY_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"approved ann policy record cannot be read: {exc}") from exc
    policy = load_ann_policy_record(data)
    _ANN_POLICY_CACHE[key] = policy
    return policy


def validate_ann_decision_evidence(
    evidence: object, policy: ProductionAnnPolicy
) -> ProductionAnnPolicy:
    """只接受锁定规模 A/B 决策工件的授权；返回唯一批准的生产策略。"""
    if isinstance(evidence, CandidatePublicationEvidence):
        raise PolicyError("candidate publication evidence cannot authorize ann policy")
    if not isinstance(evidence, AnnDecisionEvidence):
        raise PolicyError("decision authorization requires AnnDecisionEvidence")
    if not isinstance(policy, ProductionAnnPolicy):
        raise PolicyError("decision authorization requires a ProductionAnnPolicy")
    if evidence.evidence_schema_version != ANN_DECISION_EVIDENCE_SCHEMA_VERSION:
        raise PolicyError("decision evidence schema version is not current")
    if evidence.corpus_rows != _DECISION_CORPUS_ROWS:
        raise PolicyError(
            f"decision evidence corpus must be {_DECISION_CORPUS_ROWS} rows, "
            f"got {evidence.corpus_rows!r}"
        )
    if evidence.dimensions != policy.dimensions:
        raise PolicyError("decision evidence dimensions do not match the approved policy")
    if evidence.held_out_queries != _DECISION_HELD_OUT_QUERIES:
        raise PolicyError(
            f"decision evidence must bind exactly {_DECISION_HELD_OUT_QUERIES} "
            f"held-out queries, got {evidence.held_out_queries!r}"
        )
    if tuple(evidence.candidates) != _DECISION_CANDIDATES:
        raise PolicyError("decision evidence must contain the complete two-candidate grid")
    if tuple(evidence.ef_grid) != _DECISION_EF_GRID:
        raise PolicyError("decision evidence must contain the complete declared ef grid")
    if evidence.approved_index_type != policy.selected_index_type:
        raise PolicyError("decision evidence approves a different index type")
    if evidence.approved_query_ef != policy.query_ef:
        raise PolicyError("decision evidence approves a different query ef")
    if evidence.approved_recall_at_10_floor != policy.recall_at_10_floor:
        raise PolicyError("decision evidence approves a different recall@10 floor")
    if evidence.approved_recall_at_20_floor != policy.recall_at_20_floor:
        raise PolicyError("decision evidence approves a different recall@20 floor")
    for field_name in (
        "comparator_sha256", "candidate_hybrid_sha256", "reconciliation_sha256",
        "evidence_run_url",
    ):
        if getattr(evidence, field_name) != getattr(policy, field_name):
            raise PolicyError(f"decision evidence {field_name} does not match the policy")
    if not evidence.approved_by or not evidence.approved_at:
        raise PolicyError("decision evidence requires approver identity and timestamp")
    return policy


def _finite_non_negative(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) \
            or not math.isfinite(value) or value < 0:
        raise PolicyError(f"{name} must be finite and non-negative")
    return float(value)


def _aggregate_recall(
    exact_ids: tuple[tuple[str, ...], ...],
    candidate_ids: tuple[tuple[str, ...], ...],
    recall_limit: int,
) -> float:
    hits = total = 0
    for truth, observed in zip(exact_ids, candidate_ids, strict=True):
        truth_prefix = set(truth[:recall_limit])
        hits += len(truth_prefix & set(observed[:recall_limit]))
        total += len(truth_prefix)
    if total <= 0:
        raise PolicyError("publication evidence contains no truth IDs to score")
    return hits / total


def validate_candidate_publication_evidence(
    evidence: object, policy: ProductionAnnPolicy
) -> ProductionAnnPolicy:
    """校验一次 staged candidate 的发布证据；返回固定批准策略（不可重选）。

    ``benchmark_max_probes`` 只决定验证 query 数
    ``min(benchmark_max_probes, actual_dense_rows)``——它不能授权、选择或
    修改任何生产策略值。
    """
    if isinstance(evidence, AnnDecisionEvidence):
        raise PolicyError("decision evidence cannot authorize staged publication")
    if not isinstance(evidence, CandidatePublicationEvidence):
        raise PolicyError("staged publication requires CandidatePublicationEvidence")
    if not isinstance(policy, ProductionAnnPolicy):
        raise PolicyError("staged publication requires a ProductionAnnPolicy")
    if evidence.evidence_schema_version != CANDIDATE_PUBLICATION_EVIDENCE_SCHEMA_VERSION:
        raise PolicyError("publication evidence schema version is not current")
    if evidence.index_type != policy.selected_index_type:
        raise PolicyError("staged candidate index type is not the approved type")
    if evidence.query_ef != policy.query_ef:
        raise PolicyError("staged candidate query ef is not the approved ef")
    if evidence.metric != policy.metric:
        raise PolicyError("staged candidate metric is not the approved metric")
    if evidence.dimensions != policy.dimensions:
        raise PolicyError("staged candidate dimensions do not match the approved policy")
    if evidence.policy_sha256 != production_policy_sha256(policy):
        raise PolicyError("publication evidence is bound to a different policy digest")
    if evidence.decision_evidence_sha256 != policy.comparator_sha256:
        raise PolicyError("publication evidence is bound to a different decision evidence")
    if not isinstance(evidence.actual_dense_rows, int) or isinstance(evidence.actual_dense_rows, bool) \
            or evidence.actual_dense_rows <= 0:
        raise PolicyError("actual_dense_rows must be a positive integer")
    if not isinstance(evidence.benchmark_max_probes, int) or isinstance(evidence.benchmark_max_probes, bool) \
            or evidence.benchmark_max_probes <= 0:
        raise PolicyError("benchmark_max_probes must be a positive integer")
    expected_queries = min(evidence.benchmark_max_probes, evidence.actual_dense_rows)
    if evidence.validation_query_count != expected_queries:
        raise PolicyError(
            f"validation_query_count must equal min(benchmark_max_probes, "
            f"actual_dense_rows) = {expected_queries}, got {evidence.validation_query_count!r}"
        )
    if evidence.query_source != _VALIDATION_QUERY_SOURCE:
        raise PolicyError("validation queries must come from the deterministic disjoint stream")
    if evidence.corpus_query_overlap != 0:
        raise PolicyError("validation queries must not overlap indexed corpus rows")
    if not isinstance(evidence.query_selection_sha256, str) or len(evidence.query_selection_sha256) != 64:
        raise PolicyError("query_selection_sha256 must be a SHA-256 hex digest")
    expected_cardinality = min(_PRODUCTION_RESULT_LIMIT, evidence.actual_dense_rows)
    for name, rows in (
        ("exact_result_ids", evidence.exact_result_ids),
        ("candidate_result_ids", evidence.candidate_result_ids),
    ):
        if len(rows) != evidence.validation_query_count:
            raise PolicyError(f"{name} must contain exactly validation_query_count rows")
    # Flat 精确扫描保证满额 top-k；近似检索只保证"至多满额"——部分 LanceDB
    # 原生实现（Linux x86_64，CI 实测 lancedb 0.34.0）会返回少于 limit 的行，
    # 空行才是查询失败。发布证据按各自的真实契约校验。
    for row in evidence.exact_result_ids:
        if len(row) != expected_cardinality:
            raise PolicyError(
                f"exact_result_ids rows must each contain min(20, actual_dense_rows) = "
                f"{expected_cardinality} IDs"
            )
    for row in evidence.candidate_result_ids:
        if not 1 <= len(row) <= expected_cardinality:
            raise PolicyError(
                f"candidate_result_ids rows must contain 1..min(20, actual_dense_rows) = "
                f"{expected_cardinality} IDs each"
            )
    recall_10 = _aggregate_recall(
        evidence.exact_result_ids, evidence.candidate_result_ids, 10
    )
    recall_20 = _aggregate_recall(
        evidence.exact_result_ids, evidence.candidate_result_ids, 20
    )
    if not math.isclose(evidence.recall_at_10, recall_10, rel_tol=0.0, abs_tol=1e-9):
        raise PolicyError("declared recall@10 does not match the recorded ID lists")
    if not math.isclose(evidence.recall_at_20, recall_20, rel_tol=0.0, abs_tol=1e-9):
        raise PolicyError("declared recall@20 does not match the recorded ID lists")
    if evidence.recall_at_10 < policy.recall_at_10_floor:
        raise PolicyError(
            f"staged candidate recall@10 {evidence.recall_at_10:.4f} is below the "
            f"approved floor {policy.recall_at_10_floor}"
        )
    if evidence.recall_at_20 < policy.recall_at_20_floor:
        raise PolicyError(
            f"staged candidate recall@20 {evidence.recall_at_20:.4f} is below the "
            f"approved floor {policy.recall_at_20_floor}"
        )
    if evidence.unindexed_dense_rows != 0:
        raise PolicyError("staged candidate has unindexed dense rows")
    for name in ("exact_verification_ms", "ann_verification_ms", "benchmark_duration_ms"):
        _finite_non_negative(name, getattr(evidence, name))
    # ponytail: timings/sizes are recorded evidence only — no latency/resource SLO
    # was approved in Plan 02, so none is enforced here.
    return policy


_EVIDENCE_REQUIRED = frozenset({
    "evidence_schema_version", "evidence_source", "probe_scope",
    "sampling_method", "probe_count", "probe_total", "probe_coverage",
    "probe_keys", "probe_selection_sha256", "result_limit",
    "recall_aggregation", "benchmark_duration_ms",
    "exact_result_ids", "candidate_result_ids",
})


def _validated_evidence(evidence: Mapping) -> tuple[str, int, int]:
    """#41 fail-closed：校验 benchmark evidence，返回 (scope, count, total)。

    采样口径缺失、非法或自相矛盾时禁止发布——绝不把 sampled 1.00 静默解读为
    全库 100% 证明。``probe_coverage`` 与 count/total 必须一致。
    """
    missing = _EVIDENCE_REQUIRED - set(evidence)
    if missing:
        raise PolicyError(f"benchmark evidence missing fields: {sorted(missing)}")
    scope = evidence["probe_scope"]
    count = evidence["probe_count"]
    total = evidence["probe_total"]
    if scope not in {"full", "sampled", "synthetic"}:
        raise PolicyError(f"benchmark probe_scope must be full/sampled/synthetic, got {scope!r}")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise PolicyError("benchmark probe_total must be a positive integer")
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= total:
        raise PolicyError("benchmark probe_count must be an integer within probe_total")
    if not math.isclose(evidence["probe_coverage"], count / total):
        raise PolicyError("benchmark probe_coverage is inconsistent with probe_count/probe_total")
    if scope == "sampled" and evidence["sampling_method"] != "bottom_k_sha256_v1":
        raise PolicyError("sampled benchmark scope requires bottom_k_sha256_v1 sampling")
    if scope == "full" and count != total:
        raise PolicyError("full benchmark scope must cover every probe")
    if scope == "synthetic" and evidence["evidence_source"] != "observer":
        raise PolicyError("synthetic benchmark evidence must come from an observer")
    return scope, count, total


def _require_valid_observations(
    benchmark: BenchmarkObservation | None, stats: IndexStats | None
) -> tuple[BenchmarkObservation, IndexStats]:
    if benchmark is None or stats is None:
        raise PolicyError("complete benchmark and vector index observations are required")
    for name, value in (
        ("recall_at_10", benchmark.recall_at_10),
        ("recall_at_20", benchmark.recall_at_20),
        ("latency_p50_ms", benchmark.latency_p50_ms),
        ("latency_p95_ms", benchmark.latency_p95_ms),
        ("build_time_ms", benchmark.build_time_ms),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise PolicyError(f"{name} must be finite")
    if not 0.0 <= benchmark.recall_at_10 <= 1.0 or not 0.0 <= benchmark.recall_at_20 <= 1.0:
        raise PolicyError("recall observations must be between 0.00 and 1.00")
    if benchmark.disk_bytes < 0:
        raise PolicyError("disk_bytes must be non-negative")
    if not stats.index_name or stats.indexed_rows < 0 or stats.unindexed_dense_rows < 0:
        raise PolicyError("vector index coverage observation is invalid")
    return benchmark, stats


def select_vector_policy(
    benchmark: BenchmarkObservation | None, stats: IndexStats | None,
    evidence: Mapping | None = None,
) -> VectorPolicyDecision:
    """Promote ANN only at the fixed recall floor with complete dense coverage.

    #41：``evidence`` 非 None 时按采样口径解读 recall（full=全量最小 /
    sampled=样本最小 / synthetic=observer 观察），reason 与 decision 的
    benchmark_scope/count 字段复述同一口径；evidence 缺失时保持旧的
    「全量最小」语义与文案（兼容未接入 #41 的调用方/测试）。

    Phase 06（issue #49）：recall/coverage 不达标时**抛 PolicyError**，不再
    返回 exact——exact 从不是可发布的成功结果；唯一的成功结果是 ANN。
    """
    benchmark, stats = _require_valid_observations(benchmark, stats)
    scope, count, total = ("full", 0, 0)
    if evidence is not None:
        scope, count, total = _validated_evidence(evidence)

    def decide(mode: str, reason: str) -> VectorPolicyDecision:
        return VectorPolicyDecision(
            selected_mode=mode, reason=reason, benchmark=benchmark,
            index_stats=stats, benchmark_scope=scope,
            benchmark_probe_count=count, benchmark_probe_total=total,
        )

    # Phase 06（issue #49）：exact 不是可发布的成功结果。任何 recall/coverage
    # 失败都 fail-closed 抛 PolicyError——发布门禁是 CandidatePublicationEvidence，
    # 不存在运行时降级到 exact 的路径。
    if benchmark.recall_at_10 != 1.0:
        if scope == "sampled":
            reason = (
                f"sampled recall@10 was {benchmark.recall_at_10:.2f} across "
                f"{count}/{total} probes, requires 1.00"
            )
        else:
            reason = f"recall@10 was {benchmark.recall_at_10:.2f}, requires 1.00"
        raise PolicyError(reason)
    if benchmark.recall_at_20 != 1.0:
        if scope == "sampled":
            reason = (
                f"sampled recall@20 was {benchmark.recall_at_20:.2f} across "
                f"{count}/{total} probes, requires 1.00"
            )
        else:
            reason = f"recall@20 was {benchmark.recall_at_20:.2f}, requires 1.00"
        raise PolicyError(reason)
    if stats.unindexed_dense_rows:
        noun = "row" if stats.unindexed_dense_rows == 1 else "rows"
        raise PolicyError(f"{stats.unindexed_dense_rows} dense {noun} remains unindexed")
    if scope == "sampled":
        reason = f"sampled minimum recall across {count}/{total} probes meets 1.00 floor"
    elif scope == "synthetic":
        reason = "synthetic observer recall meets 1.00 floor (not measured evidence)"
    else:
        reason = "candidate meets recall and dense-index coverage requirements"
    return decide("ann", reason)
