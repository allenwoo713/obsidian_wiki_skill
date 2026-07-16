"""向量检索 metric 与 score 归一化集中管理（ISSUE-15）。

设计目标：建立完整的 metric contract——
    Embedding 配置 → 固定 Metric → 查询显式指定 → Score 转换 → Manifest 持久化

核心原则：
1. 不依赖 LanceDB 默认 metric（默认 L2，版本间可能漂移）。
2. 查询时必须显式指定 metric，不静默回退。
3. score 按 metric 用对应公式转换，不用通用 1/(1+d)。
4. 原始 distance 保留，score 与 distance 分离。
"""
from __future__ import annotations

import math
from typing import Literal

VectorMetric = Literal["cosine", "l2", "dot"]


def normalize_vector_score(
    raw_value: float,
    metric: VectorMetric,
    *,
    vectors_are_unit_normalized: bool = False,
    l2_scale: float | None = None,
) -> float:
    """把 LanceDB 返回的原始 distance 转换为 0~1 的展示相似度 score。

    Args:
        raw_value: LanceDB 返回的 _distance（原始距离）。
        metric: 向量检索使用的距离度量。
        vectors_are_unit_normalized: embedding 是否做了 L2 归一化。
            dot metric 必须为 True；l2 归一化后可用精确公式，否则需 l2_scale。
        l2_scale: 未归一化 L2 的 RBF 校准参数，必须通过验证集确定，不能随意硬编码。

    Returns:
        0.0 ~ 1.0 的展示相似度。1.0 = 完全相同，0.0 = 完全相反。

    Raises:
        ValueError: metric 不支持，或 dot/l2 前置条件不满足时。
    """
    value = float(raw_value)

    if metric == "cosine":
        # LanceDB cosine distance = 1 - cosine_similarity，范围 [0, 2]
        # score = 1 - distance/2，等价于 (1 + cosine_similarity) / 2
        return max(0.0, min(1.0, 1.0 - value / 2.0))

    if metric == "dot":
        if not vectors_are_unit_normalized:
            raise ValueError(
                "Dot score normalization requires unit-normalized vectors. "
                "Either set normalize_embeddings=True or use cosine/l2 metric."
            )
        # dot similarity 范围 [-1, 1]（归一化后），映射到 [0, 1]
        return max(0.0, min(1.0, (value + 1.0) / 2.0))

    if metric == "l2":
        if vectors_are_unit_normalized:
            # 归一化向量 L2 距离范围 [0, 2]，score = 1 - d²/4
            return max(0.0, min(1.0, 1.0 - value ** 2 / 4.0))
        if l2_scale is None or l2_scale <= 0:
            raise ValueError(
                "Non-normalized L2 score requires a calibrated l2_scale > 0. "
                "Determine l2_scale via validation set, not arbitrary hardcoded value."
            )
        # RBF 映射：exp(-d²/(2τ²))
        return math.exp(-(value ** 2) / (2.0 * l2_scale ** 2))

    raise ValueError(f"Unsupported vector metric: {metric}")


def apply_vector_metric(query_builder, metric: VectorMetric):
    """给 LanceDB query builder 显式指定 distance metric。

    兼容不同 LanceDB 版本：优先用 distance_type()，回退 metric()。
    两者都不存在则报错，不静默回退到默认 L2（避免 ISSUE-15 语义不确定性）。

    Args:
        query_builder: table.search(vector) 返回的 query builder。
        metric: "cosine" / "l2" / "dot"。

    Returns:
        已设置 metric 的 query builder。
    """
    if hasattr(query_builder, "distance_type"):
        return query_builder.distance_type(metric)

    if hasattr(query_builder, "metric"):
        return query_builder.metric(metric)

    raise RuntimeError(
        "Current LanceDB version does not expose an explicit metric API. "
        "Upgrade LanceDB to a version supporting distance_type() or metric()."
    )
