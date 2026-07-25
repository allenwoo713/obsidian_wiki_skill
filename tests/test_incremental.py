"""#7 增量更新测试（本地保留，不公开发布）。

验证页级向量缓存：
- 编辑一页只重编码该页，未变页命中缓存（不重复 torch 编码）；
- 结果与全量重建一致；
- 删除页在活动索引中无残留（向量/FTS）；
- --full-rebuild 强制忽略缓存全量重编码。
"""
import json
import logging
import re
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_index import WikiIndex  # noqa: E402


def _write_page(wiki: Path, name: str, title: str, body: str, sources=None):
    d = wiki / "concepts"
    d.mkdir(parents=True, exist_ok=True)
    fm = ('---\ntype: concept\ntitle: "%s"\nsources: %s\ntags: []\n'
          'related: []\nupdated: 2026-07-25\n---\n\n'
          % (title, json.dumps(sources or [])))
    (d / name).write_text(fm + body, encoding="utf-8")


def _cache_ns_dir(idx_dir: Path) -> Path:
    root = idx_dir / "vec_cache"
    subs = [p for p in root.glob("*") if p.is_dir()]
    assert subs, "vec_cache 命名空间目录未生成"
    return subs[0]


def _parse_encode_stats(caplog):
    """从日志解析 (命中页, 需编码页)。"""
    for rec in reversed(caplog.records):
        m = re.search(r"命中 (\d+) 页, 需编码 (\d+) 页", rec.getMessage())
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def test_incremental_reuses_cache_for_unchanged_pages(tmp_path, caplog):
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "Acme Front Radar", "频率 60fps 探测距离 200m 前向雷达参数", ["raw/a.docx"])
    _write_page(wiki, "b.md", "Vega Corner Radar", "频率 76GHz 探测距离 150m 角雷达参数", ["raw/b.docx"])
    _write_page(wiki, "c.md", "UDP 接口", "udp 报文 magicWord 0x0102 诊断接口", ["raw/c.docx"])

    wi = WikiIndex(idx_dir)
    with caplog.at_level(logging.INFO):
        wi.build(wiki)
    first = _parse_encode_stats(caplog)
    assert first is not None
    assert first[0] == 0, "首次构建应无缓存命中"
    assert first[1] == 3, "首次构建应编码全部 3 页"

    ns = _cache_ns_dir(idx_dir)
    cache_files_before = {p.name: p.read_bytes() for p in ns.glob("*.npy")}
    assert len(cache_files_before) == 3

    # 仅编辑 a.md，b/c 不变
    _write_page(wiki, "a.md", "Acme Front Radar", "频率 90fps 探测距离 250m 前向雷达升级参数", ["raw/a.docx"])

    caplog.clear()
    wi2 = WikiIndex(idx_dir)
    with caplog.at_level(logging.INFO):
        wi2.build(wiki)
    second = _parse_encode_stats(caplog)
    assert second is not None
    assert second[0] == 2, f"应命中 2 个未变页（b/c），实际 {second}"
    assert second[1] == 1, f"应仅编码 1 个变更页（a），实际 {second}"

    # 未变页缓存文件应字节不变；变更页产生新键、旧键被剪除
    cache_files_after = {p.name: p.read_bytes() for p in ns.glob("*.npy")}
    unchanged = set(cache_files_before) & set(cache_files_after)
    assert len(unchanged) == 2, "b/c 两页缓存键应保留"
    for name in unchanged:
        assert cache_files_before[name] == cache_files_after[name], "未变页向量应字节一致（复用未重编码）"
    assert len(cache_files_after) == 3, "编辑页旧缓存应被剪除，仅留当前 3 页"

    # 检索仍正确（升级参数命中 a）
    res = wi2.search("Acme 90fps 升级", k=3)
    assert res and ("Acme" in res[0].title or "a.md" in res[0].path.name)


def test_deleted_page_leaves_no_residue(tmp_path):
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "Keeper Page", "保留页 常规内容 探测", ["raw/a.docx"])
    _write_page(wiki, "b.md", "Doomed Page", "唯一词 zqxwplofrom 将被删除", ["raw/b.docx"])

    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    hits = wi.search("zqxwplofrom", k=5)
    assert any("Doomed" in h.title or "b.md" in h.path.name for h in hits), "删除前应能检索到 b"

    # 删除 b.md 后重建（增量）
    (wiki / "concepts" / "b.md").unlink()
    wi2 = WikiIndex(idx_dir)
    wi2.build(wiki)

    # 活动索引中不应再有 b 的任何 chunk
    ids = {h.path.name for h in wi2.search("zqxwplofrom", k=5)}
    assert "b.md" not in ids, "删除页在活动索引中仍有残留"
    # FTS 层直接校验
    fts = wi2.search_fts("zqxwplofrom", k=5)
    assert all("b.md" not in h.path for h in fts), "删除页 FTS 残留"


def test_full_rebuild_ignores_cache(tmp_path, caplog):
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "P1", "内容一 探测距离", ["raw/a.docx"])
    _write_page(wiki, "b.md", "P2", "内容二 频率参数", ["raw/b.docx"])

    WikiIndex(idx_dir).build(wiki)  # 建立缓存

    caplog.clear()
    with caplog.at_level(logging.INFO):
        WikiIndex(idx_dir).build(wiki, full_rebuild=True)
    stats = _parse_encode_stats(caplog)
    assert stats is not None
    assert stats[0] == 0, "full_rebuild 应忽略缓存（命中 0）"
    assert stats[1] == 2, "full_rebuild 应重编码全部页"
