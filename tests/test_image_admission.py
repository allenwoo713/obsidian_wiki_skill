"""issue #58 model-free reproduction and end-to-end orchestration tests.

Reproduces the exact defect without depending on embedding-model behavior:
fragment-heavy markdown pages outrank a vector-rank-1 image under the legacy
mixed ``page_ranking_score()`` ordering, so the image is cut by the mixed
top-k before the image channel exists. The fix partitions the complete fused
pool into text and quality-gated images *before* the cut.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fusion import page_level_rrf
from models import ChunkHit, ContextBundle, PageCandidate, is_image_page
from query import (
    _finalize_image_diagnostics,
    _select_image_candidates,
    hybrid_search,
    result_to_json,
)
from query_plan_models import QueryPlan


def _hit(
    page_id: str,
    *,
    ordinal: int,
    channel: str,
    path: Path,
    page_type: str = "concept",
    text: str = "evidence",
) -> ChunkHit:
    return ChunkHit(
        chunk_id=f"{channel}:{page_id}:{ordinal}",
        page_id=page_id,
        path=str(path),
        title=page_id,
        page_type=page_type,
        section_path=[f"section-{ordinal}"],
        heading="",
        chunk_kind="sparse" if channel == "fts" else "dense",
        text=text,
        channel=channel,
        score=float(1000 - ordinal),
    )


def _fragment_heavy_hits(
    wiki: Path,
) -> tuple[list[ChunkHit], list[ChunkHit], str]:
    image_id = "image-page"
    image_path = wiki / "assets" / "corner-fov.jpg"
    text_paths = [
        wiki / "concepts" / f"text-{index}.md"
        for index in range(5)
    ]

    # Image is vector rank 1 and FTS rank 3, matching the reported failure.
    vector = [
        _hit(
            image_id,
            ordinal=1,
            channel="vector",
            path=image_path,
            page_type="image_caption",
            text="Corner Radar 方位角 FOV 极坐标图 bumper loss",
        )
    ]
    fts = [
        _hit("text-0", ordinal=1, channel="fts", path=text_paths[0]),
        _hit("text-1", ordinal=2, channel="fts", path=text_paths[1]),
        _hit(
            image_id,
            ordinal=3,
            channel="fts",
            path=image_path,
            page_type="image_caption",
            text="Corner Radar 方位角 FOV 极坐标图 bumper loss",
        ),
    ]

    # Each Markdown page contributes many fragments. Under current legacy
    # page_ranking_score() all five pages outrank the image despite its best hit.
    for page_index, path in enumerate(text_paths):
        for fragment_index in range(5):
            vector.append(_hit(
                f"text-{page_index}",
                ordinal=len(vector) + 1,
                channel="vector",
                path=path,
                text=f"text vector evidence {page_index}-{fragment_index}",
            ))
            fts.append(_hit(
                f"text-{page_index}",
                ordinal=len(fts) + 1,
                channel="fts",
                path=path,
                text=f"text fts evidence {page_index}-{fragment_index}",
            ))
    return fts, vector, image_id


class _Planner:
    def plan(self, original_query, _context):
        return QueryPlan(
            original_query=original_query,
            normalized_query=original_query,
            intent="lookup",
            routing_reason="test",
            semantic_queries=(original_query,),
            lexical_terms=("Corner", "Radar", "FOV"),
            exact_terms=(),
            entities=(),
            relation_intent=None,
            filters={},
            context_mode="chunk",
            rewrite_used=False,
            rewrite_provider="null",
            rewrite_confidence=1.0,
            preserved_constraints=(),
        )

    def plan_retry(self, _plan, _feedback, _context):
        return None


class _ControlledIndex:
    def __init__(
        self,
        wiki: Path,
        fts: list[ChunkHit],
        vector: list[ChunkHit],
    ):
        self.index_dir = wiki.parent / ".index"
        self._fts = fts
        self._vector = vector
        self.fts_calls = 0
        self.vector_calls = 0

    def search_fts_terms(self, *_args, **_kwargs):
        self.fts_calls += 1
        return list(self._fts)

    def search_vector(self, *_args, **_kwargs):
        self.vector_calls += 1
        return list(self._vector)

    @staticmethod
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def get_chunk(_chunk_id):
        return None

    @staticmethod
    def get_neighbors(_chunk_id):
        return []

    @staticmethod
    def get_parent_section(_chunk_id):
        return []

    @staticmethod
    def get_page_sources(_page_id):
        return []

    @staticmethod
    def get_image_meta(path):
        if str(path).replace("\\", "/").endswith("corner-fov.jpg"):
            return {
                "source_doc": "raw/corner-radar.pdf",
                "source_page": 7,
                "source_section": ["FOV"],
                "nearby_text": (
                    "The polar plot shows azimuth field of view."
                ),
            }
        return None


def test_fragment_heavy_pages_cannot_block_qualified_image_admission(
    tmp_path,
):
    wiki = tmp_path / "Wiki"
    fts, vector, image_id = _fragment_heavy_hits(wiki)
    fused_pool = page_level_rrf(fts, vector, k=None)

    image_position = next(
        index for index, candidate in enumerate(fused_pool)
        if candidate.page_id == image_id
    )
    assert image_position >= 5, (
        "fixture must reproduce the legacy mixed-top-k loss"
    )

    image_candidates, diagnostics = _select_image_candidates(fused_pool)
    assert [candidate.page_id for candidate in image_candidates] == [image_id]
    assert diagnostics["fused_image_pages"] == 1
    assert diagnostics["qualified_image_pages"] == 1


def test_hybrid_search_returns_image_without_displacing_text_or_adding_queries(
    tmp_path,
):
    wiki = tmp_path / "Wiki"
    fts, vector, image_id = _fragment_heavy_hits(wiki)
    index = _ControlledIndex(wiki, fts, vector)

    legacy_pool = page_level_rrf(fts, vector, k=None)
    expected_text = [
        candidate.page_id for candidate in legacy_pool
        if candidate.page_type != "image_caption"
    ][:5]

    result = hybrid_search(
        index,
        "Corner Radar 方位角 FOV 极坐标图 bumper loss",
        _Planner(),
        k=5,
        wiki_dir=wiki,
        enable_graph=False,
    )
    payload = result_to_json(result)

    # Text ranking is the old fused order with image pages filtered out.
    assert [
        candidate.page_id for candidate in result.candidates
    ] == expected_text

    # Image is admitted independently and reaches the public JSON output.
    assert [
        candidate.page_id for candidate in result.image_candidates
    ] == [image_id]
    assert len(payload["images"]) == 1
    assert payload["images"][0]["page_id"] == image_id
    assert payload["images"][0]["page_type"] == "image_caption"
    assert payload["images"][0]["score"] > 0
    assert (
        payload["retrieval_diagnostics"]["image_outcome"]
        == "included_in_context_bundle"
    )

    # The hotfix partitions existing results; it does not add an image-only DB query.
    assert index.fts_calls == 1
    assert index.vector_calls == 1


def test_weak_single_channel_image_is_not_forced_into_context(tmp_path):
    weak = PageCandidate(
        page_id="weak-image",
        path=tmp_path / "Wiki" / "assets" / "weak.jpg",
        title="weak",
        rrf_score=1.0 / 72.0,
        sparse_rank=12,
        dense_rank=None,
        page_type="image_caption",
    )
    selected, diagnostics = _select_image_candidates([weak])

    assert selected == []
    assert diagnostics["fused_image_pages"] == 1
    assert diagnostics["qualified_image_pages"] == 0
    assert diagnostics["image_outcome"] == "image_below_rank_gate"


def test_dual_channel_image_with_bounded_ranks_is_admitted(tmp_path):
    corroborated = PageCandidate(
        page_id="corroborated-image",
        path=tmp_path / "Wiki" / "assets" / "corroborated.jpg",
        title="corroborated",
        rrf_score=(1.0 / 68.0) + (1.0 / 69.0),
        sparse_rank=8,
        dense_rank=9,
        page_type="image_caption",
    )
    selected, _diagnostics = _select_image_candidates([corroborated])
    assert selected == [corroborated]


def test_page_type_is_authoritative_and_path_is_only_legacy_fallback():
    assert is_image_page(
        "image_caption", "Wiki/generated/virtual-page.md", "image"
    )
    assert not is_image_page(
        "concept", "Wiki/assets/not-an-image.jpg", "not-an-image.jpg"
    )
    assert is_image_page(
        "unknown", "Wiki/assets/legacy.jpg", "legacy.jpg"
    )


def test_image_budget_failure_is_reported_at_the_actual_stage(tmp_path):
    image = PageCandidate(
        page_id="image-page",
        path=tmp_path / "Wiki" / "assets" / "image.jpg",
        title="image",
        rrf_score=1.0 / 61.0,
        sparse_rank=None,
        dense_rank=1,
        page_type="image_caption",
    )
    bundle = ContextBundle(
        query="q",
        mode="snippet",
        max_context_tokens=10,
        omitted_items=[{
            "page_id": "image-page",
            "title": "image",
            "reason": "image_budget_exhausted",
        }],
    )

    diagnostics = _finalize_image_diagnostics(
        {
            "image_admission_policy": "qualified_separate_channel_v1",
            "admitted_image_pages": 1,
            "image_outcome": "admitted_to_context_candidates",
        },
        [image],
        [],
        bundle,
    )
    assert diagnostics["image_outcome"] == "image_budget_exhausted"
