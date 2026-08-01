"""Standard-library regression checks for the query-contract package boundary."""
from __future__ import annotations

import sys
import subprocess
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
from obsidian_wiki.ports.chunk_repository import ChunkRepository  # noqa: E402
from obsidian_wiki.ports.embedding import EmbeddingProvider  # noqa: E402
from obsidian_wiki.ports.index_build import IndexPublisher  # noqa: E402
from obsidian_wiki.ports.index_manifest import IndexManifestStore  # noqa: E402
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
    "pyarrow",
    "pyvis",
    "sentence_transformers",
    "torch",
    "transformers",
)

STORAGE_SDK_CONSTRAINTS = (
    "lancedb==0.34.0",
    "pyarrow==25.0.0",
    "sentence-transformers==5.6.1",
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
        self.assertTrue(issubclass(ChunkRepository, Protocol))
        self.assertTrue(issubclass(EmbeddingProvider, Protocol))
        self.assertTrue(issubclass(IndexManifestStore, Protocol))
        self.assertTrue(issubclass(IndexPublisher, Protocol))
        script = """
import sys

sys.path.insert(0, sys.argv[1])
from obsidian_wiki.domain import query_models  # noqa: F401
from obsidian_wiki.domain import index_models  # noqa: F401
from obsidian_wiki.ports import query_planning  # noqa: F401
from obsidian_wiki.ports import index_storage  # noqa: F401
from obsidian_wiki.ports import chunk_repository  # noqa: F401
from obsidian_wiki.ports import embedding  # noqa: F401
from obsidian_wiki.ports import index_manifest  # noqa: F401
from obsidian_wiki.ports import index_build  # noqa: F401

for root in {forbidden_roots!r}:
    if any(name == root or name.startswith(root + ".") for name in sys.modules):
        raise SystemExit(f"{{root}} was imported by a core contract module")
""".format(forbidden_roots=FORBIDDEN_SDK_ROOTS)
        result = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).resolve().parents[1] / "scripts")],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_import_linter_declares_index_build_layer_direction(self):
        config = (Path(__file__).resolve().parents[1] / ".importlinter").read_text(
            encoding="utf-8"
        )
        self.assertIn("[importlinter:contract:index-build-layers]", config)
        self.assertIn("    obsidian_wiki.infrastructure", config)
        self.assertIn("    obsidian_wiki.application", config)
        self.assertIn("    obsidian_wiki.ports", config)
        self.assertIn("    obsidian_wiki.domain", config)

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

    def test_storage_sdk_resolution_is_pinned_and_reported_by_workflows(self):
        root = Path(__file__).resolve().parents[1]
        requirements = (root / "requirements.in").read_text(encoding="utf-8")
        ci_workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        eval_workflow = (root / ".github/workflows/eval.yml").read_text(encoding="utf-8")

        for constraint in STORAGE_SDK_CONSTRAINTS:
            self.assertIn(constraint, requirements)

        for workflow in (ci_workflow, eval_workflow):
            self.assertIn("Verify dependency lock is current", workflow)
            self.assertIn("Report storage SDK versions", workflow)
            self.assertIn("importlib.metadata", workflow)
            for package in ("lancedb", "pyarrow", "sentence-transformers"):
                self.assertIn(package, workflow)


if __name__ == "__main__":
    unittest.main()
