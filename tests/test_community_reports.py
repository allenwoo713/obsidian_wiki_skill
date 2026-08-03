"""issue #10 GraphRAG global 测试。

覆盖：
1. build_community_reports 生成结构化报告（必填字段齐全）。
2. global intent 路由到 community reports，返回非空 text_items（与 local 分离）。
3. 无 community reports 时 _global_retrieve 返回 None（不静默退化为 local）。
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_community_reports import build_community_reports, write_reports  # noqa: E402


class _NamedTokenCounter:
    """Deliberately simple test fake; production never uses character estimates."""

    identity = "test-token-counter/v1"

    def count(self, text: str) -> int:
        return len(text.split())


class _Graph:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def read(self):
        return self._snapshot


class _ReportStore:
    def __init__(self):
        self.staged = {}
        self.active = None
        self.activations = []

    def stage(self, build_id, reports, manifest):
        self.staged[build_id] = (tuple(reports), manifest)

    def read_staged(self, build_id):
        return self.staged.get(build_id)

    def activate(self, build_id):
        self.activations.append(build_id)
        self.active = self.staged[build_id]

    def read_active(self):
        return self.active


def _fresh_snapshot():
    from obsidian_wiki.domain.community_report_models import GraphEdge, GraphSnapshotState, PageSnapshot

    return GraphSnapshotState(
        pages=(
            PageSnapshot("Wiki/a.md", "a-full-markdown-hash"),
            PageSnapshot("Wiki/b.md", "b-full-markdown-hash"),
        ),
        edges=(GraphEdge("Wiki/a.md", "Wiki/b.md", ("related", "source"), 0.75),),
        communities=((7, ("Wiki/a.md", "Wiki/b.md")),),
    )


def test_service_builds_and_reads_fresh_v2_report():
    """A staged v2 set activates only after source and counter evidence validate."""
    from obsidian_wiki.application.community_report_service import CommunityReportService
    from obsidian_wiki.domain.community_report_models import COMMUNITY_REPORT_SCHEMA_VERSION

    store = _ReportStore()
    service = CommunityReportService(store, _Graph(_fresh_snapshot()), _NamedTokenCounter())

    manifest = service.build()

    assert manifest.report_schema_version == COMMUNITY_REPORT_SCHEMA_VERSION == 2
    assert store.activations == [manifest.build_id]
    outcome = service.retrieve()
    assert outcome.status.value == "community_reports_fresh"
    assert outcome.reports and outcome.reports[0].member_fingerprint
    assert outcome.reports[0].edge_fingerprint
    assert outcome.reports[0].token_counter_id == _NamedTokenCounter.identity


def test_report_service_rejects_incompatible_contracts_without_text():
    """Missing, stale, and schema-disagreeing sets fail closed before selection."""
    from dataclasses import replace

    from obsidian_wiki.application.community_report_service import CommunityReportService
    from obsidian_wiki.domain.community_report_models import CommunityReportStatus

    store = _ReportStore()
    service = CommunityReportService(store, _Graph(_fresh_snapshot()), _NamedTokenCounter())
    assert service.retrieve().status is CommunityReportStatus.MISSING

    manifest = service.build()
    reports, active_manifest = store.active
    cases = (
        (None, CommunityReportStatus.MISSING),
        ((reports, replace(active_manifest, report_schema_version=1)), CommunityReportStatus.SCHEMA_UNSUPPORTED),
        ((reports, replace(active_manifest, report_schema_version=3)), CommunityReportStatus.SCHEMA_UNSUPPORTED),
        ((reports, replace(active_manifest, stale_reason="graph_published")), CommunityReportStatus.STALE),
        ((reports[:-1], active_manifest), CommunityReportStatus.SCHEMA_UNSUPPORTED),
    )
    for active, expected in cases:
        store.active = active
        outcome = service.retrieve()
        assert outcome.status is expected
        assert outcome.reports == ()
        assert outcome.local_fallback_used is False
        assert outcome.required_action == "build-community-reports"


def _write(wiki, name, title, sources, related, body="正文"):
    d = wiki / "concepts"
    d.mkdir(parents=True, exist_ok=True)
    fm = '---\ntype: concept\ntitle: "%s"\nsources: %s\ntags: []\nrelated: %s\nupdated: 2026-06-29\n---\n\n%s' % (
        title, json.dumps(sources), json.dumps(related), body)
    (d / name).write_text(fm, encoding="utf-8")


def test_build_community_reports(tmp_path):
    wiki = tmp_path / "Wiki"
    # 两个社区：A-B 同源 s1（+wikilink）；C-D 同源 s2（+wikilink）
    _write(wiki, "a.md", "Acme Radar", ["s1"], ["[[Acme Calibration]]"], "频率 探测 radar")
    _write(wiki, "b.md", "Acme Calibration", ["s1"], ["[[Acme Radar]]"], "校准 角度 calibration")
    _write(wiki, "c.md", "Vega Camera", ["s2"], ["[[Vega Optics]]"], "分辨率 帧率 camera")
    _write(wiki, "d.md", "Vega Optics", ["s2"], ["[[Vega Camera]]"], "镜头 光圈 optics")
    reports = build_community_reports(tmp_path)
    out = write_reports(tmp_path, reports)
    assert out.exists()
    assert (tmp_path / ".index" / "communities.json").exists()
    assert len(reports) >= 1
    rep = reports[0]
    for field in ("community_id", "member_page_ids", "title", "summary",
                  "key_entities", "key_relationships", "source_pages",
                  "token_count", "content_hash"):
        assert field in rep, f"报告缺字段 {field}"
    assert "页面" in rep["summary"]
    assert rep["member_page_ids"], "成员页非空"


def test_global_retrieve_uses_community_reports(tmp_path):
    """global intent 路由到 community reports，返回非空 text_items（与 local 分离）。"""
    from build_index import WikiIndex
    from query_planner import DefaultQueryPlanner
    from query import hybrid_search
    wiki = tmp_path / "Wiki"
    _write(wiki, "a.md", "Acme Radar", ["s1"], ["[[Acme Calibration]]"], "频率 探测 radar")
    _write(wiki, "b.md", "Acme Calibration", ["s1"], ["[[Acme Radar]]"], "校准 角度 calibration")
    _write(wiki, "c.md", "Vega Camera", ["s2"], ["[[Vega Optics]]"], "分辨率 帧率 camera")
    _write(wiki, "d.md", "Vega Optics", ["s2"], "[[Vega Camera]]", "镜头 光圈 optics")
    wi = WikiIndex(tmp_path / ".index")
    wi.build(wiki)
    reports = build_community_reports(tmp_path)
    write_reports(tmp_path, reports)
    planner = DefaultQueryPlanner(project_root=tmp_path)
    # 强制 global intent；query 含 Acme Radar 词项以匹配报告
    result = hybrid_search(wi, "全局概述 Acme Radar 知识库主题", planner, k=5,
                           wiki_dir=wiki, intent_override="global")
    assert len(result.text_items) > 0, "global 查询应返回社区报告"
    assert all(it.inclusion_reason == "global_community_report" for it in result.text_items)


def test_global_without_reports_returns_none(tmp_path):
    """无 community reports 时 _global_retrieve 返回 None（不静默退化为 local）。"""
    from build_index import WikiIndex
    from query import _global_retrieve
    from query_planner import DefaultQueryPlanner
    wiki = tmp_path / "Wiki"
    _write(wiki, "a.md", "Acme", ["s1"], [], "正文")
    wi = WikiIndex(tmp_path / ".index")
    wi.build(wiki)
    planner = DefaultQueryPlanner(project_root=tmp_path)
    plan = planner.plan("全局概述")
    # 未构建 community reports → 返回 None
    assert _global_retrieve(wi, plan, k=5, max_tokens=4096) is None
