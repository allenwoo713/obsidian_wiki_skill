"""issue #9：增量索引单元测试 —— 覆盖修改页/删除页后索引一致性。"""
from pathlib import Path

from build_index import WikiIndex


def _write(wiki, name, title, body, sources=None):
    fm = "---\ntitle: " + title + "\n"
    if sources:
        fm += "sources:\n" + "".join(f"  - {s}\n" for s in sources)
    fm += "---\n\n"
    (wiki / name).write_text(fm + body, encoding="utf-8")


def test_incremental_update_makes_new_content_searchable(tmp_path):
    wiki = tmp_path / "Wiki"
    wiki.mkdir()
    _write(wiki, "a.md", "A", "# A\n\n初始内容 Acme VisionCam X200 分辨率 1920×1080。", ["raw/a.docx"])
    idx = tmp_path / ".index"
    wi = WikiIndex(idx)
    wi.build(wiki)
    wi.load()

    # 修改页面，追加新术语
    _write(wiki, "a.md", "A", "# A\n\n初始内容 Acme VisionCam X200 分辨率 1920×1080。\n\n新术语 0xABCD 测试。", ["raw/a.docx"])
    wi.build(wiki, full_rebuild=False)
    wi.load()

    hits = wi.search_fts("0xABCD", k=5)
    assert any("a.md" in h.page_id for h in hits), "增量后新内容应可被检索"


def test_incremental_delete_removes_stale_rows(tmp_path):
    wiki = tmp_path / "Wiki"
    wiki.mkdir()
    _write(wiki, "a.md", "A", "# A\n\n保留页 Acme VisionCam X200。", ["raw/a.docx"])
    _write(wiki, "b.md", "B", "# B\n\n待删除页 独有标记 ZZZ999。", ["raw/b.docx"])
    idx = tmp_path / ".index"
    wi = WikiIndex(idx)
    wi.build(wiki)
    wi.load()
    assert wi.search_fts("ZZZ999", k=5)  # 删除前可查

    (wiki / "b.md").unlink()
    wi.build(wiki, full_rebuild=False)
    wi.load()
    # 删除后该页内容不应残留
    assert not any("b.md" in h.page_id for h in wi.search_fts("ZZZ999", k=5))


def test_incremental_unchanged_page_keeps_searchable(tmp_path):
    wiki = tmp_path / "Wiki"
    wiki.mkdir()
    _write(wiki, "a.md", "A", "# A\n\n稳定内容 CFR-100 探测距离 250 m。", ["raw/a.docx"])
    _write(wiki, "b.md", "B", "# B\n\n另一页。", ["raw/b.docx"])
    idx = tmp_path / ".index"
    wi = WikiIndex(idx)
    wi.build(wiki)
    wi.load()
    # 修改 b 页，a 页不变
    _write(wiki, "b.md", "B", "# B\n\n改后内容。", ["raw/b.docx"])
    wi.build(wiki, full_rebuild=False)
    wi.load()
    hits = wi.search_fts("CFR-100", k=5)
    assert any("a.md" in h.page_id for h in hits), "未变页增量后仍应可查"
