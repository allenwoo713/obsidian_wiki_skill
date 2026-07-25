"""issue #9：向量检索单元测试 —— 覆盖语义召回、chunk 级返回、metric。"""
from pathlib import Path


def test_vector_finds_semantic_chinese(tiny_kb):
    wi, _, _ = tiny_kb
    hits = wi.search_vector("相机分辨率是多少", k=5)
    assert any("cam_x200" in h.page_id for h in hits)


def test_vector_finds_radar(tiny_kb):
    wi, _, _ = tiny_kb
    hits = wi.search_vector("雷达探测距离", k=5)
    assert any("radar_cfr100" in h.page_id or "radar_ccr100" in h.page_id for h in hits)


def test_vector_returns_chunk_hits_with_distance(tiny_kb):
    wi, _, _ = tiny_kb
    hits = wi.search_vector("GigE Vision 接口", k=5)
    assert hits
    assert all(h.channel == "vector" for h in hits)
    # cosine 距离应为非 None
    assert all(h.distance is not None for h in hits)


def test_vector_empty_query_does_not_crash(tiny_kb):
    wi, _, _ = tiny_kb
    hits = wi.search_vector("", k=5)
    # 空查询仍返回结果（按距离），不抛异常
    assert isinstance(hits, list)
