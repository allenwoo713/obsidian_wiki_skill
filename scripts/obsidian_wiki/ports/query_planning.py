"""Protocol boundaries used by query-planning implementations."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

from obsidian_wiki.domain.query_models import (
    PlannerContext,
    QueryPlan,
    ResolvedEntity,
    RetrievalFeedback,
)


class EntityCatalog(Protocol):
    """Read-only entity matching port; planner never owns index dependencies."""

    def resolve(self, query: str, context: PlannerContext) -> Tuple[ResolvedEntity, ...]: ...


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
