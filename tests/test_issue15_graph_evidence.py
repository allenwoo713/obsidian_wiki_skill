"""Deterministic regression coverage for issue #15's evidence-first graph path."""
import json
from pathlib import Path

from models import ChunkHit, PageCandidate
from query import _validate_graph_candidates, graph_expand, hybrid_search
from query_plan_models import QueryPlan


def _hit(page_id, text="matching page text"):
    return ChunkHit(
        chunk_id=f"{page_id}-chunk", page_id=page_id, path=f"Wiki/{page_id}.md",
        title=page_id, page_type="concept", section_path=["Facts"], heading="Facts",
        chunk_kind="dense", text=text, channel="fts", score=1.0,
    )


def _plan(query="relation", attempt=0):
    return QueryPlan(
        original_query=query, normalized_query=query, intent="relation", routing_reason="test",
        semantic_queries=(query,), lexical_terms=(query,), exact_terms=(), entities=(),
        relation_intent="cause_or_influence", filters={}, context_mode="section",
        rewrite_used=False, rewrite_provider="test", rewrite_confidence=1.0,
        preserved_constraints=(), retry_attempt=attempt,
    )


class _Planner:
    def __init__(self):
        self.calls = []

    def plan(self, query, context):
        self.calls.append("initial")
        return _plan("initial")

    def plan_retry(self, previous, feedback, context):
        self.calls.append("retry")
        return _plan("retry", attempt=1)


class _Index:
    def __init__(self):
        self.page_calls = []

    def search_fts_terms(self, lexical, exact, k=20):
        return [_hit("seed")] if lexical == ("retry",) else []

    def search_vector(self, query, k=20):
        return []

    def search_page(self, page_id, plan, sparse_k=20, dense_k=20):
        self.page_calls.append((page_id, plan.original_query))
        return [_hit(page_id)] if page_id == "supported" else []

    def get_page_sources(self, page_id):
        return [f"Raw/{page_id}.docx"]

    def get_chunk(self, chunk_id):
        return None

    def get_neighbors(self, chunk_id):
        return []

    def get_parent_section(self, chunk_id):
        return []

    def read_page(self, page_id):
        return ""

    def count_tokens(self, text):
        return max(1, len(text.split()))


def _write_graph(wiki):
    graph = {
        "nodes": [
            {"id": "seed", "title": "Seed", "path": "Wiki/seed.md"},
            {"id": "supported", "title": "Supported", "path": "Wiki/supported.md"},
            {"id": "inferred-only", "title": "Inferred", "path": "Wiki/inferred.md"},
        ],
        "edges": [
            {"source": "seed", "target": "supported", "weight": 1.0,
             "signal": "direct_link", "signals": ["direct_link", "source_overlap"]},
            {"source": "seed", "target": "inferred-only", "weight": .4,
             "signal": "adamic_adar", "signals": ["adamic_adar", "type_affinity"]},
        ],
    }
    (wiki.parent / ".index").mkdir(parents=True)
    (wiki.parent / ".index" / "graph.json").write_text(json.dumps(graph), encoding="utf-8")


def test_graph_expand_preserves_all_signals_and_marks_complete_inference(tmp_path):
    wiki = tmp_path / "Wiki"
    wiki.mkdir()
    _write_graph(wiki)
    expanded = graph_expand(_Index(), ["seed"], wiki)
    supported = next(c for c in expanded if c.page_id == "supported")
    inferred = next(c for c in expanded if c.page_id == "inferred-only")
    assert supported.graph_paths[0].edge_signals == ["direct_link", "source_overlap"]
    assert supported.graph_paths[0].is_inferred is False
    assert inferred.graph_paths[0].edge_signals == ["adamic_adar", "type_affinity"]
    assert inferred.graph_paths[0].is_inferred is True


def test_validation_requires_same_page_evidence_and_nonempty_text():
    index = _Index()
    graph_candidates = [
        PageCandidate("supported", Path("Wiki/supported.md"), "Supported", 0, None, None),
        PageCandidate("inferred-only", Path("Wiki/inferred.md"), "Inferred", 0, None, None),
    ]
    graph_candidates[0].graph_paths = graph_expand(index, ["seed"], Path("."))[:1] or []
    validated = _validate_graph_candidates(index, graph_candidates, _plan("relation"))
    assert [candidate.page_id for candidate in validated] == ["supported"]
    assert validated[0].sparse_evidence and validated[0].sparse_evidence[0].chunk_id == "supported-chunk"


def test_retry_rebuilds_direct_page_seeds_and_only_merges_validated_graph_results(tmp_path):
    wiki = tmp_path / "Wiki"
    wiki.mkdir()
    _write_graph(wiki)
    index, planner = _Index(), _Planner()
    result = hybrid_search(index, "unrecognized relation", planner, wiki_dir=wiki, k=5)

    assert planner.calls == ["initial", "retry"]
    assert ("supported", "retry") in index.page_calls
    assert ("inferred-only", "retry") in index.page_calls
    graph_items = [item for item in result.bundle.items if item.inclusion_reason == "graph_expansion"]
    assert [item.page_id for item in graph_items] == ["supported"]
    assert graph_items[0].evidence and graph_items[0].text


def test_wiki_index_restricted_search_never_returns_another_page(tiny_kb):
    wi, _, _ = tiny_kb
    page_id = next(page_id for page_id in wi._page_by_id if "cam_x200" in page_id)
    hits = wi.search_page(page_id, _plan("雷达"))
    assert hits
    assert {hit.page_id for hit in hits} == {page_id}
