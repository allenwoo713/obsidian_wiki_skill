"""Standard-library regression checks for the query-contract package boundary."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Protocol


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from obsidian_wiki.domain.query_models import (  # noqa: E402
    PlannerContext as PackagePlannerContext,
    PlannerWarning as PackagePlannerWarning,
    QueryIntent as PackageQueryIntent,
    QueryPlan as PackageQueryPlan,
    ResolvedEntity as PackageResolvedEntity,
    RetrievalFeedback as PackageRetrievalFeedback,
)
from obsidian_wiki.ports.query_planning import (  # noqa: E402
    EntityCatalog as PackageEntityCatalog,
    QueryPlanner as PackageQueryPlanner,
    RewriteProvider as PackageRewriteProvider,
)
from query_plan_models import (  # noqa: E402
    EntityCatalog,
    PlannerContext,
    PlannerWarning,
    QueryIntent,
    QueryPlan,
    QueryPlanner,
    ResolvedEntity,
    RetrievalFeedback,
    RewriteProvider,
)


FORBIDDEN_SDK_ROOTS = (
    "lancedb",
    "networkx",
    "pyvis",
    "sentence_transformers",
    "torch",
    "transformers",
)


class ArchitectureFoundationTests(unittest.TestCase):
    def test_legacy_exports_are_canonical_package_objects(self):
        self.assertIs(QueryIntent, PackageQueryIntent)
        self.assertIs(PlannerContext, PackagePlannerContext)
        self.assertIs(PlannerWarning, PackagePlannerWarning)
        self.assertIs(ResolvedEntity, PackageResolvedEntity)
        self.assertIs(QueryPlan, PackageQueryPlan)
        self.assertIs(RetrievalFeedback, PackageRetrievalFeedback)
        self.assertIs(EntityCatalog, PackageEntityCatalog)
        self.assertIs(QueryPlanner, PackageQueryPlanner)
        self.assertIs(RewriteProvider, PackageRewriteProvider)

    def test_query_plan_json_contract_remains_list_normalized(self):
        plan = QueryPlan(
            original_query="calibrate Radar-7",
            normalized_query="calibrate Radar-7",
            intent=QueryIntent.PROCEDURE.value,
            routing_reason="test",
            semantic_queries=("calibrate Radar-7",),
            lexical_terms=("calibrate", "radar-7"),
            exact_terms=("Radar-7",),
            entities=("Radar-7",),
            relation_intent=None,
            filters={"page_type": ["procedure"]},
            context_mode="parent_section",
            rewrite_used=False,
            rewrite_provider="null",
            rewrite_confidence=1.0,
            preserved_constraints=("Radar-7",),
            warnings=(PlannerWarning("test_warning", "kept"),),
            rewrite_attempted=False,
            rewrite_applied=False,
        )

        self.assertEqual(
            plan.to_json(),
            {
                "original_query": "calibrate Radar-7",
                "normalized_query": "calibrate Radar-7",
                "intent": "procedure",
                "routing_reason": "test",
                "semantic_queries": ["calibrate Radar-7"],
                "lexical_terms": ["calibrate", "radar-7"],
                "exact_terms": ["Radar-7"],
                "entities": ["Radar-7"],
                "relation_intent": None,
                "filters": {"page_type": ["procedure"]},
                "context_mode": "parent_section",
                "rewrite_used": False,
                "rewrite_provider": "null",
                "rewrite_confidence": 1.0,
                "preserved_constraints": ["Radar-7"],
                "warnings": [{"code": "test_warning", "message": "kept"}],
                "planner_schema_version": "qp-1",
                "tokenizer_hash": "",
                "lexicon_hash": "",
                "retry_attempt": 0,
                "rewrite_source": None,
                "rewrite_attempted": False,
                "rewrite_applied": False,
                "rewrite_failure_reason": None,
                "hook_injected_enhanced": None,
            },
        )

    def test_ports_are_importable_without_heavy_sdk_imports(self):
        self.assertTrue(issubclass(PackageEntityCatalog, Protocol))
        self.assertTrue(issubclass(PackageQueryPlanner, Protocol))
        self.assertTrue(issubclass(PackageRewriteProvider, Protocol))
        for root in FORBIDDEN_SDK_ROOTS:
            self.assertFalse(
                any(name == root or name.startswith(f"{root}.") for name in sys.modules),
                f"{root} was imported by a core contract module",
            )

    def test_ci_declares_cross_platform_architecture_gate(self):
        workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("  architecture:\n", workflow)
        self.assertIn("os: [ubuntu-latest, windows-latest]", workflow)
        self.assertIn('python-version: ["3.10", "3.13"]', workflow)
        self.assertIn("runs-on: ${{ matrix.os }}", workflow)
        self.assertIn("import-linter==2.13", workflow)
        self.assertIn("working-directory: scripts", workflow)
        self.assertIn("lint-imports --config ../.importlinter --no-cache", workflow)
        self.assertIn("python tests/test_architecture_foundation.py", workflow)


if __name__ == "__main__":
    unittest.main()
