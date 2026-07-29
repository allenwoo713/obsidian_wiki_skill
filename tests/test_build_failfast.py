"""Issue #13 review (Gap 1): fail-fast chunking in the production build path.

Builds a REAL LanceDB index (loads the embedding model), so it is heavier than
the pure-chunking unit tests. Uses ``tempfile.mkdtemp`` (not pytest ``tmp_path``)
so the Windows sandbox safe-delete guard on ``.pytest_tmp`` teardown does not
mask results.

Covers:
- a page whose ``chunk_page`` raises → ``build()`` raises ``ChunkBuildError``,
  the OLD active index is preserved (pointer unchanged, row count unchanged),
  and there is NO loadable half-built new version; the error carries page_id/path.
- a non-empty page that yields 0 chunks → pre-publish completeness check fails.
- ``--allow-partial-index`` degrades the missing page to a warning and still
  publishes the remaining pages (experimental escape hatch only).
"""
from pathlib import Path
from unittest import mock

import pytest

import chunking
from build_index import WikiIndex, ChunkBuildError


def _make_wiki(n=3):
    tmp = Path(__import__("tempfile").mkdtemp())
    wiki = tmp / "Wiki"
    wiki.mkdir()
    for i in range(n):
        (wiki / f"p{i}.md").write_text(
            f"---\ntitle: P{i}\n---\n\n# P{i}\n段落内容编号{i}用于检索与召回测试样本。\n",
            encoding="utf-8")
    return wiki


def _count_rows(wi):
    return wi._get_lance_table().count_rows()


def test_failfast_on_chunk_exception_preserves_old_index():
    wiki = _make_wiki(3)
    idx = wiki.parent / ".index"
    wi = WikiIndex(idx)
    wi.build(wiki)                 # baseline good build
    wi.load()
    old_count = _count_rows(wi)
    old_pointer = (idx / "ACTIVE_INDEX").read_text(encoding="utf-8")

    real_chunk_page = chunking.chunk_page
    p1 = str((wiki / "p1.md").resolve())   # page_id = 解析后的绝对路径

    def _boom(page_id, *a, **k):
        if page_id == p1:
            raise RuntimeError("injected chunk failure on p1")
        return real_chunk_page(page_id, *a, **k)

    with mock.patch("build_index.chunk_page", _boom):
        with pytest.raises(ChunkBuildError) as excinfo:
            wi.build(wiki)

    # 错误信息必须携带失败页 page_id / path
    assert p1 in str(excinfo.value), str(excinfo.value)

    # 旧活动索引仍可查询、行数不变、指针未翻转
    wi2 = WikiIndex(idx)
    wi2.load()
    assert _count_rows(wi2) == old_count
    assert (idx / "ACTIVE_INDEX").read_text(encoding="utf-8") == old_pointer
    # 失败 staging 仅留 .failed 标记，不被 load 当作新版本
    failed_markers = list((idx / "builds").glob("*/.failed"))
    assert failed_markers, "failed staging dir should be marked (.failed)"


def test_completeness_rejects_zero_chunk_page():
    wiki = _make_wiki(3)
    idx = wiki.parent / ".index"
    wi = WikiIndex(idx)

    with mock.patch("build_index.chunk_page", lambda *a, **k: []):
        with pytest.raises(ChunkBuildError):
            wi.build(wiki)


def test_allow_partial_index_degrades_to_publish():
    wiki = _make_wiki(3)
    idx = wiki.parent / ".index"
    wi = WikiIndex(idx)
    real_chunk_page = chunking.chunk_page
    p1 = str((wiki / "p1.md").resolve())

    def _drop_one(page_id, *a, **k):
        if page_id == p1:
            return []          # p1 漏页
        return real_chunk_page(page_id, *a, **k)

    with mock.patch("build_index.chunk_page", _drop_one):
        # allow_partial_index=True：缺页降级 warning，仍发布其余页
        wi.build(wiki, allow_partial_index=True)

    wi.load()
    rows = wi._get_lance_table().to_arrow().to_pylist()
    pids = {r["page_id"] for r in rows}
    assert p1 not in pids
    assert str((wiki / "p0.md").resolve()) in pids
    assert str((wiki / "p2.md").resolve()) in pids
