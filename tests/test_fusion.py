"""Retrieval v2 融合逻辑单元测试（纯逻辑，不依赖 embedding / LanceDB / 临时目录）。"""
import sys
from pathlib import Path

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
    bundle = assemble_context([tc, ic], wi=None, mode="snippet", max_tokens=500)
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
    bundle = assemble_context([tc], wi=None, mode="full", max_tokens=2000)
    assert "完整内容" in bundle.items[0].text
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def test_assemble_context_per_type_budget_enforced():
    """issue #3：四路预算独立强制——dense 不应挤占 image/page/graph 配额。"""
    from models import PageCandidate, EvidenceHit
    # 20 个 dense 候选，每个 500 字 → ~125 token；max_tokens=2000 → dense_budget=1200
    # 最多纳入 9 个（9*125=1125 ≤ 1200，第 10 个 1250 > 1200 省略）
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
    bundle = assemble_context(dense + [img], wi=None, mode="snippet", max_tokens=2000)
    dense_items = [i for i in bundle.items if i.inclusion_reason == "rrf"]
    image_items = [i for i in bundle.items if i.inclusion_reason == "image"]
    # dense 被预算截断（远少于 20），且不超过 dense_budget 允许的上限
    assert len(dense_items) < 20
    assert len(dense_items) <= 9
    assert sum(i.token_count for i in dense_items) <= 1200
    # image 仍纳入（独立预算，未被 dense 挤占）
    assert len(image_items) == 1
    # 省略原因标明 dense_budget_exhausted
    omitted_reasons = [o["reason"] for o in bundle.omitted_items]
    assert any("dense_budget_exhausted" in r for r in omitted_reasons)
