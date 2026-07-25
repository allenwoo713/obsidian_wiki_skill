"""issue #9：ContextBundle 单元测试 —— 覆盖 mode/预算/omitted/render。"""
from pathlib import Path

from models import PageCandidate, EvidenceHit, ContextItem
from fusion import assemble_context, render_context_markdown


def _cand(page_id, title, score=1.0, evidence_text="关键证据文本", graph_paths=None):
    return PageCandidate(
        page_id=page_id, path=Path(page_id), title=title, rrf_score=score,
        sparse_rank=1, dense_rank=1,
        dense_evidence=[EvidenceHit("c1", "dense", 1, 1.0, evidence_text, [])],
        sparse_evidence=[], graph_paths=graph_paths or [],
    )


def test_assemble_summary_mode_truncates(tiny_kb):
    wi, _, _ = tiny_kb
    c = _cand("x", "T", evidence_text="摘要内容 " * 50)
    bundle = assemble_context([c], wi, mode="summary", max_tokens=2000,
                              token_counter=wi.count_tokens)
    assert bundle.items
    assert bundle.token_count <= 2000


def test_assemble_omitted_when_budget_exhausted(tiny_kb):
    wi, _, _ = tiny_kb
    cands = [_cand(f"p{i}", f"T{i}", evidence_text="内容 " * 30) for i in range(20)]
    bundle = assemble_context(cands, wi, mode="snippet", max_tokens=200,
                              token_counter=wi.count_tokens)
    # 预算不足时应省略部分
    assert bundle.omitted_items


def test_assemble_per_type_budget_isolated(tiny_kb):
    """issue #3：dense 不应挤占 image/graph 预算。"""
    wi, _, _ = tiny_kb
    from models import PageCandidate, EvidenceHit
    dense = [_cand(f"d{i}", f"D{i}", evidence_text="密集文本 " * 10) for i in range(20)]
    img = PageCandidate(
        page_id="img1", path=Path("assets/img1.png"), title="图1", rrf_score=1.0,
        sparse_rank=1, dense_rank=1,
        dense_evidence=[EvidenceHit("ic", "dense", 1, 1.0, "图片说明文字", [])],
        sparse_evidence=[], graph_paths=[],
    )
    bundle = assemble_context(dense + [img], wi, mode="snippet", max_tokens=2000,
                              token_counter=wi.count_tokens)
    # image 仍应纳入（其预算独立）
    assert any(it.inclusion_reason == "image" for it in bundle.items)


def test_render_context_markdown(tiny_kb):
    wi, _, _ = tiny_kb
    c = _cand("x", "T", evidence_text="证据")
    bundle = assemble_context([c], wi, mode="snippet", max_tokens=2000,
                              token_counter=wi.count_tokens)
    md = render_context_markdown(bundle)
    assert "T" in md and "证据" in md
