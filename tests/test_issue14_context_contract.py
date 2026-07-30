"""Regression coverage for issue #14's repository-backed context contract."""
from pathlib import Path
from types import SimpleNamespace

from fusion import assemble_context, page_level_rrf, render_context_markdown
from models import ChunkHit, EvidenceHit, PageCandidate
from query import HybridResult, result_to_json


def _hit(chunk_id, channel, text, section=("Install",)):
    return ChunkHit(
        chunk_id=chunk_id, page_id="page-a", path="Wiki/page-a.md", title="Page A",
        page_type="concept", section_path=list(section), heading=section[-1],
        chunk_kind="dense", text=text, channel=channel, score=1.0,
    )


class _Repository:
    """Small read-only repository; no filesystem or LanceDB access is permitted."""

    def __init__(self):
        self.chunks = {
            "sparse-1": _hit("sparse-1", "fts", "same evidence"),
            "dense-1": _hit("dense-1", "vector", "same evidence"),
            "before": _hit("before", "fts", "before chunk"),
            "after": _hit("after", "fts", "after chunk"),
        }

    def get_chunk(self, chunk_id):
        return self.chunks.get(chunk_id)

    def get_neighbors(self, chunk_id):
        return [self.chunks["before"], self.chunks["after"]]

    def get_parent_section(self, chunk_id):
        return [self.chunks["before"], self.chunks[chunk_id], self.chunks["after"]]

    def get_page_sources(self, page_id):
        assert page_id == "page-a"
        return ["Raw/sources/a.docx"]

    def read_page(self, page_id):
        assert page_id == "page-a"
        return "# Install\n\n" + ("complete repository page " * 20)


def _candidate():
    return PageCandidate(
        page_id="page-a", path=Path("Wiki/page-a.md"), title="Page A", rrf_score=1.0,
        sparse_rank=1, dense_rank=1,
        sparse_evidence=[EvidenceHit("sparse-1", "sparse", 1, 2.0, "same evidence", ["Install"])],
        dense_evidence=[EvidenceHit("dense-1", "dense", 1, 3.0, "same evidence", ["Install"])],
    )


def test_dual_channel_provenance_and_citations_survive_rendering():
    repo = _Repository()
    bundle = assemble_context([_candidate()], repository=repo, scope="adjacent",
                              max_tokens=200, token_counter=lambda text: len(text.split()))
    item = bundle.items[0]

    assert {ev.channel for ev in item.evidence} == {"sparse", "dense"}
    assert item.text.count("same evidence") == 1
    assert item.sources == ["Raw/sources/a.docx"]
    assert "before chunk" in item.text and "after chunk" in item.text
    for required in ("page-a", "Wiki/page-a.md", "Raw/sources/a.docx", "sparse-1", "dense-1"):
        assert required in bundle.context_text
        assert required in render_context_markdown(bundle)

    payload = result_to_json(HybridResult("q", bundle, SimpleNamespace(to_json=lambda: {}), [_candidate()], [item], []))
    assert {ev["channel"] for ev in payload["text"][0]["evidence"]} == {"sparse", "dense"}
    assert payload["text"][0]["sources"] == ["Raw/sources/a.docx"]


def test_requested_scopes_use_repository_content_and_report_token_aware_fallback():
    repo = _Repository()
    for scope, expected in (("chunk", "same evidence"), ("section", "before chunk"),
                            ("multiple_sections", "after chunk")):
        item = assemble_context([_candidate()], repository=repo, scope=scope,
                                max_tokens=200, token_counter=lambda text: len(text.split())).items[0]
        assert expected in item.text
        assert item.scope == scope

    full = assemble_context([_candidate()], repository=repo, scope="full_page",
                            max_tokens=9, token_counter=lambda text: len(text.split())).items[0]
    assert full.token_count <= 9
    assert full.truncated is True
    assert full.truncation_reason == "full_page_token_limit"
    assert full.omitted_ranges


def test_reservations_reflow_without_exceeding_global_cap():
    candidates = [_candidate() for _ in range(3)]
    for number, candidate in enumerate(candidates):
        candidate.page_id = f"page-{number}"
        candidate.path = Path(f"Wiki/page-{number}.md")
        candidate.title = f"Page {number}"
        for evidence in candidate.sparse_evidence + candidate.dense_evidence:
            evidence.text = "one two three four"
    bundle = assemble_context(candidates, scope="chunk", max_tokens=12,
                              token_counter=lambda text: len(text.split()))
    assert len(bundle.items) == 3, "unused page/image/graph reservations should reflow"
    assert bundle.token_count <= 12


def test_rrf_collects_all_channel_evidence_for_one_page():
    fused = page_level_rrf(
        [_hit("sparse-1", "fts", "sparse first"), _hit("sparse-2", "fts", "sparse second")],
        [_hit("dense-1", "vector", "dense first"), _hit("dense-2", "vector", "dense second")],
    )
    assert [hit.chunk_id for hit in fused[0].sparse_evidence] == ["sparse-1", "sparse-2"]
    assert [hit.chunk_id for hit in fused[0].dense_evidence] == ["dense-1", "dense-2"]
