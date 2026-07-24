"""Query Planner 数据契约（GitHub issue #6）。

独立的、低耦合的数据层：本模块**不导入** LanceDB / NetworkX / RRF / ContextBundle
实现，只定义 Query Planner 与其消费者（query.py / #4/#3/#5/#10）之间的稳定契约。

``QueryPlan`` 必须可 JSON 序列化，并作为 ``query.py`` 后续检索阶段的唯一查询输入。
``original_query`` 始终等于 ``semantic_queries[0]``，永不被改写替换。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple


class QueryIntent(str, Enum):
    LOOKUP = "lookup"
    PROCEDURE = "procedure"
    COMPARISON = "comparison"
    RELATION = "relation"
    GLOBAL = "global"


@dataclass(frozen=True)
class PlannerContext:
    """多轮对话 / 项目级上下文（仅用于消解指代，不替代原始问题）。"""
    conversation_text: Optional[str] = None
    domain_terms: Tuple[str, ...] = ()
    known_entities: Tuple[str, ...] = ()
    page_types: Tuple[str, ...] = ()
    language_hints: Tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryPlan:
    """确定性规划 + 可选 LLM rewrite 的产物；query.py 的唯一查询输入。"""
    original_query: str
    normalized_query: str

    intent: str
    routing_reason: str

    semantic_queries: Tuple[str, ...]
    lexical_terms: Tuple[str, ...]
    exact_terms: Tuple[str, ...]

    entities: Tuple[str, ...]
    relation_intent: Optional[str]
    filters: Dict[str, Any]
    context_mode: str

    rewrite_used: bool
    rewrite_provider: str
    rewrite_confidence: float
    preserved_constraints: Tuple[str, ...]
    warnings: Tuple[str, ...] = ()

    # ---- 调试 / 可溯源字段（供评测与 --json 输出）----
    planner_schema_version: str = "qp-1"
    tokenizer_hash: str = ""
    lexicon_hash: str = ""
    retry_attempt: int = 0
    rewrite_source: Optional[str] = None
    hook_injected_enhanced: Optional[bool] = None

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalFeedback:
    """首轮检索结果，供低召回重试决策。"""
    sparse_hit_count: int
    dense_hit_count: int
    top_score_gap: Optional[float]
    evidence_count: int
    failure_reason: Optional[str] = None


class QueryPlanner(Protocol):
    def plan(self, query: str, context: PlannerContext) -> QueryPlan: ...
    def plan_retry(
        self,
        previous: QueryPlan,
        feedback: RetrievalFeedback,
        context: PlannerContext,
    ) -> Optional[QueryPlan]: ...


class RewriteProvider(Protocol):
    def rewrite(
        self,
        original_query: str,
        deterministic_plan: QueryPlan,
        context: PlannerContext,
        retry_feedback: Optional[RetrievalFeedback] = None,
    ) -> Dict[str, Any]: ...


class NullRewriteProvider:
    """默认 rewrite provider：不调用任何 LLM，直接返回「未改写」结果。

    保证无 LLM / 无网络时 Query Planner 仍能完成确定性规划与检索。
    """

    name = "null"

    def rewrite(self, original_query, deterministic_plan, context, retry_feedback=None):
        return {
            "semantic_queries": [],
            "rewrite_used": False,
            "confidence": 1.0,
            "preserved_constraints": list(deterministic_plan.preserved_constraints),
            "entities": list(deterministic_plan.entities),
            "relation_intent": deterministic_plan.relation_intent,
        }
