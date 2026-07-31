"""Backward-compatible query-planning contracts for direct script imports.

The canonical pure data contracts and protocol boundaries live in
``obsidian_wiki``. This facade keeps existing bare imports used by scripts and
tests stable during the staged architecture migration.
"""
from __future__ import annotations

from obsidian_wiki.domain.query_models import (
    PlannerContext,
    PlannerWarning,
    QueryIntent,
    QueryPlan,
    ResolvedEntity,
    RetrievalFeedback,
)
from obsidian_wiki.ports.query_planning import (
    EntityCatalog,
    QueryPlanner,
    RewriteProvider,
)

__all__ = [
    "EntityCatalog",
    "NullRewriteProvider",
    "PlannerContext",
    "PlannerWarning",
    "QueryIntent",
    "QueryPlan",
    "QueryPlanner",
    "ResolvedEntity",
    "RetrievalFeedback",
    "RewriteProvider",
]


class NullRewriteProvider:
    """Default rewrite adapter: deterministically decline every rewrite."""

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
