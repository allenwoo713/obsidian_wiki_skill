"""issue #10 GraphRAG global 测试。

覆盖：
1. build_community_reports 生成结构化报告（必填字段齐全）。
2. global intent 路由到 community reports，返回非空 text_items（与 local 分离）。
3. 无 community reports 时 _global_retrieve 返回 None（不静默退化为 local）。
"""
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


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
    second_report = replace(reports[0], community_id=8)
    cases = (
        (None, CommunityReportStatus.MISSING),
        ((reports, replace(active_manifest, report_schema_version=1)), CommunityReportStatus.SCHEMA_UNSUPPORTED),
        ((reports, replace(active_manifest, report_schema_version=3)), CommunityReportStatus.SCHEMA_UNSUPPORTED),
        ((reports, replace(reports[0], report_schema_version=1)), CommunityReportStatus.SCHEMA_UNSUPPORTED),
        ((reports, replace(reports[0], report_schema_version=3)), CommunityReportStatus.SCHEMA_UNSUPPORTED),
        (
            ((reports[0], replace(second_report, report_schema_version=1)), replace(active_manifest, report_count=2)),
            CommunityReportStatus.SCHEMA_UNSUPPORTED,
        ),
        ((reports, replace(active_manifest, stale_reason="graph_published")), CommunityReportStatus.STALE),
        ((reports[:-1], active_manifest), CommunityReportStatus.SCHEMA_UNSUPPORTED),
        (((object(),), replace(active_manifest, report_count=1)), CommunityReportStatus.SCHEMA_UNSUPPORTED),
        ((reports, object()), CommunityReportStatus.SCHEMA_UNSUPPORTED),
    )
    for active, expected in cases:
        store.active = active
        outcome = service.retrieve()
        assert outcome.status is expected
        assert outcome.reports == ()
        assert outcome.local_fallback_used is False
        assert outcome.required_action == "build-community-reports"


def test_current_fingerprint_gate_rejects_body_frontmatter_and_edge_changes(tmp_path):
    """Current source facts, not manifest metadata, authorize report retrieval."""
    from obsidian_wiki.application.community_report_service import CommunityReportService
    from obsidian_wiki.infrastructure.filesystem_community_reports import FilesystemCommunityReportStore
    from obsidian_wiki.infrastructure.filesystem_graph_snapshot import FilesystemGraphSnapshot

    wiki = tmp_path / "Wiki"
    wiki.mkdir()
    page_a = wiki / "a.md"
    page_b = wiki / "b.md"
    original_a = "---\ntitle: Alpha\ntype: concept\n---\n\nOriginal body"
    original_b = "---\ntitle: Beta\ntype: concept\n---\n\nSecond body"
    page_a.write_text(original_a, encoding="utf-8")
    page_b.write_text(original_b, encoding="utf-8")
    index = tmp_path / ".index"
    index.mkdir()
    graph_path = index / "graph.json"
    page_ids = [str(page_a), str(page_b)]

    def write_graph(*, signals=("direct_link",), weight=1.0):
        graph_path.write_text(json.dumps({
            "nodes": [{"id": page_id} for page_id in page_ids],
            "edges": [{"source": page_ids[0], "target": page_ids[1], "signals": list(signals), "weight": weight}],
            "communities": [page_ids],
        }), encoding="utf-8")

    write_graph()
    service = CommunityReportService(
        FilesystemCommunityReportStore(index), FilesystemGraphSnapshot(tmp_path), _NamedTokenCounter()
    )
    service.build()
    assert service.retrieve().status.value == "community_reports_fresh"

    page_a.write_text(original_a.replace("Original", "Changed"), encoding="utf-8")
    body_outcome = service.retrieve()
    assert body_outcome.status.value == "community_reports_stale"
    assert any("member fingerprint" in reason for reason in body_outcome.stale_reasons)
    page_a.write_text(original_a, encoding="utf-8")

    page_a.write_text(original_a.replace("type: concept", "type: procedure"), encoding="utf-8")
    frontmatter_outcome = service.retrieve()
    assert frontmatter_outcome.status.value == "community_reports_stale"
    assert any("member fingerprint" in reason for reason in frontmatter_outcome.stale_reasons)
    page_a.write_text(original_a, encoding="utf-8")

    write_graph(signals=("direct_link", "source_overlap", "source_overlap"))
    signals_outcome = service.retrieve()
    assert signals_outcome.status.value == "community_reports_stale"
    assert any("edge fingerprint" in reason for reason in signals_outcome.stale_reasons)
    assert not any("member fingerprint" in reason for reason in signals_outcome.stale_reasons)

    write_graph(weight=2.0)
    weight_outcome = service.retrieve()
    assert weight_outcome.status.value == "community_reports_stale"
    assert any("edge fingerprint" in reason for reason in weight_outcome.stale_reasons)
    assert not any("member fingerprint" in reason for reason in weight_outcome.stale_reasons)
