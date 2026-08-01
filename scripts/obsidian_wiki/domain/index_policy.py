"""Pure, deterministic D-02/D-03 candidate ANN selection policy."""
from __future__ import annotations

import math

from obsidian_wiki.domain.index_models import (
    BenchmarkObservation,
    IndexStats,
    VectorPolicyDecision,
)


class PolicyError(ValueError):
    """Raised for missing or malformed observations, never as a valid fallback."""


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
    benchmark: BenchmarkObservation | None, stats: IndexStats | None
) -> VectorPolicyDecision:
    """Promote ANN only at the fixed recall floor with complete dense coverage."""
    benchmark, stats = _require_valid_observations(benchmark, stats)
    if benchmark.recall_at_10 != 1.0:
        reason = f"recall@10 was {benchmark.recall_at_10:.2f}, requires 1.00"
    elif benchmark.recall_at_20 != 1.0:
        reason = f"recall@20 was {benchmark.recall_at_20:.2f}, requires 1.00"
    elif stats.unindexed_dense_rows:
        noun = "row" if stats.unindexed_dense_rows == 1 else "rows"
        reason = f"{stats.unindexed_dense_rows} dense {noun} remains unindexed"
    else:
        return VectorPolicyDecision(
            selected_mode="ann",
            reason="candidate meets recall and dense-index coverage requirements",
            benchmark=benchmark,
            index_stats=stats,
        )
    return VectorPolicyDecision(
        selected_mode="exact", reason=reason, benchmark=benchmark, index_stats=stats
    )
