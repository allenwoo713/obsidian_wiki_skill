"""Stable, infrastructure-free query data contracts."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


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
class PlannerWarning:
    """Stable, JSON-safe diagnostic emitted by the planner."""

    code: str
    message: str = ""


@dataclass(frozen=True)
class ResolvedEntity:
    """A deterministic entity match supplied by an injected catalog."""

    matched_text: str
    value: str
    kind: str


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
    warnings: Tuple[PlannerWarning, ...] = ()

    planner_schema_version: str = "qp-1"
    tokenizer_hash: str = ""
    lexicon_hash: str = ""
    retry_attempt: int = 0
    rewrite_source: Optional[str] = None
    rewrite_attempted: bool = False
    rewrite_applied: bool = False
    rewrite_failure_reason: Optional[str] = None
    hook_injected_enhanced: Optional[bool] = None

    def to_json(self) -> Dict[str, Any]:
        """Return the public JSON-safe, list-normalized query-plan contract."""
        return json.loads(json.dumps(asdict(self), ensure_ascii=False))


@dataclass(frozen=True)
class RetrievalFeedback:
    """首轮检索结果，供低召回重试决策。"""

    sparse_hit_count: int
    dense_hit_count: int
    top_score_gap: Optional[float]
    evidence_count: int
    failure_reason: Optional[str] = None
