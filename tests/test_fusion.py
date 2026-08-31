"""Retrieval v2 融合逻辑单元测试（纯逻辑，不依赖 embedding / LanceDB / 临时目录）。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from models import ChunkHit
from fusion import page_level_rrf, assemble_context
from chunking import chunk_page, CHUNK_SCHEMA_VERSION


def _hit(page_id, title, channel, score, chunk_kind="dense", text="x"):
    return ChunkHit(
        chunk_id=f"{CHUNK_SCHEMA_VERSION}:{page_id}:{chunk_kind}:0",
        page_id=page_id, path=f"/wiki/{page_id}.md", title=title,
        page_type="concept", section_path=[], heading="",
        chunk_kind=chunk_kind, text=text, channel=channel, score=score,
    )


def test_page_level_rrf_merges_two_channels():
    # FTS: A(rank1), B(rank3) ; Vector: B(rank1)
    # 期望 B 的 RRF 分高于 A（B 两路都靠前）
    fts = [_hit("A", "A", "fts", 5.0), _hit("B", "B", "fts", 1.0)]
    vec = [_hit("B", "B", "vector", 9.0)]
    out = page_level_rrf(fts, vec, k=5)
    titles = [c.title for c in out]
    assert "B" in titles and "A" in titles
    # B 同时在两路靠前，应排第一
    assert out[0].title == "B"
    # B 应同时携带 dense(sparse 通道 FTS) 与 vector 证据
    assert out[0].dense_evidence and out[0].sparse_evidence


def test_page_level_rrf_respects_k():
    fts = [_hit(f"P{i}", f"P{i}", "fts", float(10 - i)) for i in range(5)]
    vec = [_hit(f"Q{i}", f"Q{i}", "vector", float(10 - i)) for i in range(5)]
    out = page_level_rrf(fts, vec, k=3)
    assert len(out) == 3


def test_assemble_context_budget_splits_text_and_images():
    # 一个文本页 + 一个图片页
    text_cand = type("_C", (), {})  # placeholder; 用真实 PageCandidate
    from models import PageCandidate, EvidenceHit
    tc = PageCandidate(
        page_id="A", path=Path("/wiki/A.md"), title="A", rrf_score=1.0,
        sparse_rank=1, dense_rank=1,
        dense_evidence=[EvidenceHit("c", "dense", 1, 1.0, "雷达工作原理详细描述" * 5, [])],
    )
    ic = PageCandidate(
        page_id="img1.png", path=Path("/wiki/assets/img1.png"), title="图1",
        rrf_score=0.5, sparse_rank=2, dense_rank=None,
        dense_evidence=[EvidenceHit("ci", "dense", 2, 0.5, "方框图示意雷达前端", [])],
    )
    bundle = assemble_context([tc, ic], wi=None, mode="snippet", max_tokens=500,
                              citation_root=Path("/wiki"))
    kinds = {i.inclusion_reason for i in bundle.items}
    assert "image" in kinds
    assert bundle.token_count <= 500
    assert bundle.token_count > 0


def test_assemble_context_full_mode_reads_page():
    from models import PageCandidate, EvidenceHit
    # 用临时文件模拟页面，验证 full 模式读取
    import tempfile, os
    d = tempfile.mkdtemp()
    p = Path(d) / "full.md"
    p.write_text("---\ntitle: Full\n---\n" + ("完整内容 " * 200), encoding="utf-8")
    tc = PageCandidate(
        page_id=str(p.resolve()), path=p, title="Full", rrf_score=1.0,
        sparse_rank=1, dense_rank=1,
        dense_evidence=[EvidenceHit("c", "dense", 1, 1.0, "short", [])],
    )
    bundle = assemble_context([tc], wi=None, mode="full", max_tokens=2000,
                              citation_root=Path(d))
    assert "完整内容" in bundle.items[0].text
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def test_assemble_context_per_type_budget_enforced():
    """Channel minima protect image admission; unused capacity then reflows."""
    from models import PageCandidate, EvidenceHit
    # 20 个 dense 候选，每个 500 字 → ~125 token；max_tokens=2000。
    # Dense initially gets a 1200-token minimum, then may use the shared remainder.
    dense = [
        PageCandidate(
            page_id=f"d{i}", path=Path(f"/wiki/d{i}.md"), title=f"D{i}",
            rrf_score=1.0 - i * 0.01, sparse_rank=1, dense_rank=1,
            dense_evidence=[EvidenceHit(f"c{i}", "dense", 1, 1.0, "x" * 500, [])],
        ) for i in range(20)
    ]
    # 1 个 image 候选：即使 dense 被截断，image 仍应纳入（独立预算）
    img = PageCandidate(
        page_id="img.png", path=Path("/wiki/assets/img.png"), title="图",
        rrf_score=0.1, sparse_rank=None, dense_rank=None,
        dense_evidence=[EvidenceHit("ci", "dense", 1, 0.5, "图注内容", [])],
    )
    bundle = assemble_context(dense + [img], wi=None, mode="snippet", max_tokens=2000,
                              citation_root=Path("/wiki"))
    dense_items = [i for i in bundle.items if i.inclusion_reason == "rrf"]
    image_items = [i for i in bundle.items if i.inclusion_reason == "image"]
    # Dense cannot displace the image minimum, but may use the shared remainder.
    assert len(dense_items) < 20
    assert len(dense_items) > 9
    # image 仍纳入（独立预算，未被 dense 挤占）
    assert len(image_items) == 1
    assert bundle.token_count <= bundle.max_context_tokens
    # 省略原因标明 dense_budget_exhausted
    omitted_reasons = [o["reason"] for o in bundle.omitted_items]
    assert any("dense_budget_exhausted" in r for r in omitted_reasons)


# --- Issue #47 E: page-level RRF contributes once per page per channel ----------
# Before #47 a single page with N FTS fragments received N RRF contributions,
# inflating its score and biasing retrieval toward fragmented pages. The fix
# scores each page once per channel while still retaining every fragment.

def test_one_rrf_contribution_per_page_per_channel():
    fts = [_hit("P", "P", "fts", 5.0 - i, text=f"f{i}") for i in range(3)]
    out = page_level_rrf(fts, [], k=5)
    assert len(out) == 1
    # Exactly 1/(60+1) — not 1/61 + 1/62 + 1/63.
    assert out[0].rrf_score == pytest.approx(1.0 / 61.0)


def test_nondefault_rrf_denominator_applies_to_public_and_ranking_scores():
    from fusion import page_ranking_score

    hits = [_hit("P", "P", "fts", 5.0 - i, text=f"f{i}") for i in range(3)]
    candidate = page_level_rrf(hits, [], k=1, k_rrf=10)[0]

    assert candidate.rrf_score == pytest.approx(1.0 / 11.0)
    assert page_ranking_score(candidate, k_rrf=10) == pytest.approx(
        1.0 / 11.0 + 1.0 / 12.0 + 1.0 / 13.0)


def test_all_fragments_retained_as_evidence():
    fts = [_hit("P", "P", "fts", 5.0 - i, text=f"f{i}") for i in range(3)]
    out = page_level_rrf(fts, [], k=5)
    assert len(out[0].sparse_evidence) == 3


def test_bounded_evidence_strength_reranks_without_changing_rrf_score():
    sparse = [_hit("supported", "Supported", "fts", 5.0 - i, text=f"s{i}")
              for i in range(4)]
    sparse.append(_hit("thin", "Thin", "fts", 4.5, text="thin"))
    out = page_level_rrf(sparse, [], k=2)
    assert [candidate.page_id for candidate in out] == ["supported", "thin"]
    supported = out[0]
    assert supported.rrf_score == pytest.approx(1.0 / 61.0)


def test_evidence_strength_is_capped_against_unbounded_fragment_bias():
    five = [_hit("five", "Five", "fts", 10.0 - i, text=f"f{i}") for i in range(5)]
    many = [_hit("many", "Many", "fts", 5.0 - i * .01, text=f"m{i}") for i in range(12)]
    out = page_level_rrf(five + many, [], k=2)
    assert [candidate.page_id for candidate in out] == ["five", "many"]
    from fusion import page_ranking_score
    from dataclasses import replace
    extras = [replace(hit, rank=hit.rank + 100) for hit in out[1].sparse_evidence]
    many_more = replace(out[1], sparse_evidence=out[1].sparse_evidence + extras)
    assert page_ranking_score(out[1]) == pytest.approx(page_ranking_score(many_more))


def test_two_channels_give_two_contributions():
    fts = [_hit("P", "P", "fts", 5.0, text="a")]
    vec = [_hit("P", "P", "vector", 9.0, text="b")]
    out = page_level_rrf(fts, vec, k=5)
    assert len(out) == 1
    # fts once + vector once = 2/61.
    assert out[0].rrf_score == pytest.approx(2.0 / 61.0)
    assert out[0].sparse_evidence and out[0].dense_evidence


def test_dual_channel_outranks_single_channel():
    dual = page_level_rrf(
        [_hit("P", "P", "fts", 5.0, text="a")],
        [_hit("P", "P", "vector", 9.0, text="b")],
        k=5,
    )[0]
    single = page_level_rrf(
        [_hit("Q", "Q", "fts", 5.0, text="a")],
        [],
        k=5,
    )[0]
    # A page present in both channels scores exactly double a single-channel page.
    assert dual.rrf_score == pytest.approx(2.0 * single.rrf_score)


def test_page_level_rrf_k_none_returns_full_pool_and_preserves_page_type():
    image = ChunkHit(
        chunk_id="image:1",
        page_id="image",
        path="/wiki/assets/image.jpg",
        title="image",
        page_type="image_caption",
        section_path=[],
        heading="",
        chunk_kind="dense",
        text="image caption",
        channel="vector",
        score=1.0,
    )
    text = ChunkHit(
        chunk_id="text:1",
        page_id="text",
        path="/wiki/text.md",
        title="text",
        page_type="concept",
        section_path=[],
        heading="",
        chunk_kind="dense",
        text="text",
        channel="vector",
        score=0.9,
    )

    out = page_level_rrf([], [image, text], k=None)

    assert len(out) == 2
    by_id = {candidate.page_id: candidate for candidate in out}
    assert by_id["image"].page_type == "image_caption"
    assert by_id["text"].page_type == "concept"


def test_page_level_rrf_rejects_inconsistent_page_type_across_channels():
    fts = ChunkHit(
        chunk_id="p:fts",
        page_id="p",
        path="/wiki/p.md",
        title="p",
        page_type="concept",
        section_path=[],
        heading="",
        chunk_kind="sparse",
        text="p",
        channel="fts",
        score=1.0,
    )
    vector = ChunkHit(
        chunk_id="p:vector",
        page_id="p",
        path="/wiki/p.md",
        title="p",
        page_type="image_caption",
        section_path=[],
        heading="",
        chunk_kind="dense",
        text="p",
        channel="vector",
        score=1.0,
    )

    with pytest.raises(ValueError, match="inconsistent page_type"):
        page_level_rrf([fts], [vector], k=None)
