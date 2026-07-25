"""query.py 测试（Retrieval v2 API）。

v2 重写后 API 变迁（测试已对齐）：
- ``rrf_fuse`` → ``fusion.page_level_rrf``（吃 ChunkHit，按 page_id 归并）
- ``budget_control`` → 吸进 ``fusion.assemble_context``（四路 token 预算）
- ``read_full_content`` → ``fusion._read_full_content``（私有）
- ``hybrid_search`` 现在必传 ``planner: DefaultQueryPlanner``
- ``split_text_image`` → ``query._split_text_image``（私有，吃 ContextItem）
"""
import json
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from query import hybrid_search, _split_text_image  # noqa: E402
from fusion import page_level_rrf, assemble_context, _read_full_content  # noqa: E402
from models import ChunkHit, PageCandidate, EvidenceHit, ContextItem  # noqa: E402
from query_planner import DefaultQueryPlanner  # noqa: E402


def _write(wiki, subdir, name, title, body, sources=None):
    d = wiki / subdir
    d.mkdir(parents=True, exist_ok=True)
    fm = '---\ntype: concept\ntitle: "%s"\nsources: %s\ntags: []\nrelated: []\nupdated: 2026-06-29\n---\n\n' % (
        title, json.dumps(sources or []))
    (d / name).write_text(fm + body, encoding="utf-8")


def _chunk(pid, title, channel, score, text="正文片段"):
    """构造一个 ChunkHit（最小必填字段）。"""
    return ChunkHit(
        chunk_id=f"{pid}:{channel}", page_id=pid, path=f"Wiki/{pid}.md",
        title=title, page_type="concept", section_path=[], heading="",
        chunk_kind="dense", text=text, channel=channel, score=score,
    )


def test_page_level_rrf():
    """page "B" 同时命中 FTS 与向量 → RRF 分更高，应排第一。"""
    fts = [_chunk("a", "A", "fts", 1.0), _chunk("b", "B", "fts", 0.9)]
    vec = [_chunk("b", "B", "vector", 0.95), _chunk("c", "C", "vector", 0.8)]
    fused = page_level_rrf(fts, vec, k=5)
    titles = [c.title for c in fused]
    assert "B" in titles
    assert titles[0] == "B"


def test_assemble_context_budget():
    """token 预算保护：超出 max_tokens 的候选进入 omitted，used 不超标。"""
    pages = []
    for i in range(20):
        ev = [EvidenceHit(chunk_id=f"c{i}", channel="dense", rank=1,
                          raw_score=1.0, text="x" * 500, section_path=[])]
        pages.append(PageCandidate(
            page_id=f"p{i}", path=Path(f"Wiki/p{i}.md"), title=f"T{i}",
            rrf_score=1.0, sparse_rank=None, dense_rank=1,
            dense_evidence=ev,
        ))
    bundle = assemble_context(pages, max_tokens=2000)
    assert bundle.token_count <= 2000
    assert len(bundle.omitted_items) > 0, "应有候选因预算耗尽被省略"


def test_hybrid_search(tmp_path):
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write(wiki, "concepts", "a.md", "Acme Front Radar",
           "频率 60fps 探测距离 200m FOV ±60度", ["raw/acme.docx"])
    _write(wiki, "concepts", "b.md", "Vega Radar",
           "频率 76GHz Vega 探测距离 150m", ["raw/vega.docx"])
    from build_index import WikiIndex
    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    planner = DefaultQueryPlanner(project_root=tmp_path)
    result = hybrid_search(wi, "Acme 60fps 频率", planner, k=5, wiki_dir=wiki)
    assert len(result.text_items) > 0


def test_read_full_content(tmp_path):
    f = tmp_path / "page.md"
    f.write_text('---\ntitle: "Test"\ntype: concept\n---\n\n这是正文内容，应该被读取。', encoding="utf-8")
    content = _read_full_content(f)
    assert "这是正文内容" in content
    assert "title" not in content


def test_read_full_content_truncation(tmp_path):
    f = tmp_path / "big.md"
    f.write_text('---\ntitle: "Big"\n---\n\n' + "x" * 10000, encoding="utf-8")
    content = _read_full_content(f, max_chars=100)
    assert len(content) <= 200  # 截断 + 提示
    assert "截断" in content


def test_hybrid_search_read_full(tmp_path):
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write(wiki, "concepts", "a.md", "Acme Front Radar",
           "频率 60fps 探测距离 200m FOV ±60度 这是完整正文", ["raw/acme.docx"])
    from build_index import WikiIndex
    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    planner = DefaultQueryPlanner(project_root=tmp_path)
    result = hybrid_search(wi, "Acme", planner, k=1, wiki_dir=wiki, mode_override="full")
    assert len(result.text_items) > 0
    # mode_override="full" 时读取命中页完整正文（远超 200 字符 snippet 限制）
    assert "这是完整正文" in result.text_items[0].text


def test_split_text_image():
    """_split_text_image 按 path 含 assets/ 把 ContextItem 归入 images。"""
    text_item = ContextItem(
        page_id="p", path="Wiki/p.md", title="P",
        inclusion_reason="rrf", scope="chunk", text="正文",
    )
    img_item = ContextItem(
        page_id="img1", path="Wiki/assets/x_img01.png", title="图1",
        inclusion_reason="image", scope="chunk", text="图注",
    )
    tc, ic = _split_text_image([text_item, img_item])
    assert len(tc) == 1
    assert len(ic) == 1
    assert Path(ic[0].path).name == "x_img01.png"
