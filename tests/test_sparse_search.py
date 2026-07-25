"""issue #9：稀疏检索（FTS）单元测试 —— 覆盖错误码 / 型号 / 中文词项召回。"""
from pathlib import Path


def test_fts_finds_error_code(tiny_kb):
    wi, _, _ = tiny_kb
    hits = wi.search_fts("0x0102", k=5)
    assert any("cam_x200" in h.page_id for h in hits), "错误码 0x0102 应命中相机页"


def test_fts_finds_model(tiny_kb):
    wi, _, _ = tiny_kb
    hits = wi.search_fts("CFR-100", k=5)
    assert any("radar_cfr100" in h.page_id for h in hits)


def test_fts_terms_with_lexical_and_exact(tiny_kb):
    wi, _, _ = tiny_kb
    # 通道专用词项：lexical_terms + exact_terms
    hits = wi.search_fts_terms(["探测距离"], ["250 m"], k=5)
    assert hits and any("radar_cfr100" in h.page_id for h in hits)


def test_fts_terms_empty_query_returns_empty(tiny_kb):
    wi, _, _ = tiny_kb
    assert wi.search_fts_terms([], [], k=5) == []


def test_fts_hit_is_chunk_level(tiny_kb):
    wi, _, _ = tiny_kb
    hits = wi.search_fts("GigE Vision", k=5)
    assert hits
    assert all(h.channel == "fts" for h in hits)
    assert all(h.chunk_id for h in hits)
