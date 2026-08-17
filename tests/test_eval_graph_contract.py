"""Regression tests for the graph off/on evaluation contract."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import run_eval  # noqa: E402


def test_graph_contract_fixture_requires_explicit_linked_gold_pages():
    queries = run_eval.load_queries(run_eval.GRAPH_CONTRACT_QUERIES)
    assert len(queries) == 1
    query = queries[0]
    assert query["k"] == 1
    assert query["min_incremental_gold_pages"] == len(query["relevant_pages"])
    assert len(query["required_facts"]) == len(query["relevant_pages"])
