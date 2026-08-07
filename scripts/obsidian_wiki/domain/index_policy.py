"""Pure, deterministic D-02/D-03 candidate ANN selection policy."""
from __future__ import annotations

import math
from typing import Mapping

from obsidian_wiki.domain.index_models import (
    BenchmarkObservation,
    IndexStats,
    VectorPolicyDecision,
)


class PolicyError(ValueError):
    """Raised for missing or malformed observations, never as a valid fallback."""


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

    if benchmark.recall_at_10 != 1.0:
        if scope == "sampled":
            reason = (
                f"sampled recall@10 was {benchmark.recall_at_10:.2f} across "
                f"{count}/{total} probes, requires 1.00"
            )
        else:
            reason = f"recall@10 was {benchmark.recall_at_10:.2f}, requires 1.00"
        return decide("exact", reason)
    if benchmark.recall_at_20 != 1.0:
        if scope == "sampled":
            reason = (
                f"sampled recall@20 was {benchmark.recall_at_20:.2f} across "
                f"{count}/{total} probes, requires 1.00"
            )
        else:
            reason = f"recall@20 was {benchmark.recall_at_20:.2f}, requires 1.00"
        return decide("exact", reason)
    if stats.unindexed_dense_rows:
        noun = "row" if stats.unindexed_dense_rows == 1 else "rows"
        reason = f"{stats.unindexed_dense_rows} dense {noun} remains unindexed"
        return decide("exact", reason)
    if scope == "sampled":
        reason = f"sampled minimum recall across {count}/{total} probes meets 1.00 floor"
    elif scope == "synthetic":
        reason = "synthetic observer recall meets 1.00 floor (not measured evidence)"
    else:
        reason = "candidate meets recall and dense-index coverage requirements"
    return decide("ann", reason)
