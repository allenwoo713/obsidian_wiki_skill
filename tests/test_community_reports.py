"""issue #10 GraphRAG global 测试。

覆盖：
1. build_community_reports 生成结构化报告（必填字段齐全）。
2. global intent 路由到 community reports，返回非空 text_items（与 local 分离）。
3. 无 community reports 时 _global_retrieve 返回 None（不静默退化为 local）。
"""
from pathlib import Path
import sys
import json
from types import SimpleNamespace

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


def test_current_membership_gate_rejects_members_gained_or_lost_without_source_changes():
    """The current graph partition, not stored report IDs, authorizes a report."""
    from obsidian_wiki.application.community_report_service import CommunityReportService
    from obsidian_wiki.domain.community_report_models import GraphSnapshotState, PageSnapshot

    graph = _Graph(_fresh_snapshot())
    store = _ReportStore()
    service = CommunityReportService(store, graph, _NamedTokenCounter())
    service.build()

    original = graph._snapshot
    gained_page = PageSnapshot("Wiki/c.md", "c-full-markdown-hash")
    graph._snapshot = GraphSnapshotState(
        pages=(*original.pages, gained_page), edges=original.edges,
        communities=((7, ("Wiki/a.md", "Wiki/b.md", "Wiki/c.md")),),
    )
    gained = service.retrieve()
    assert gained.reports == ()
    assert any("membership is stale" in reason for reason in gained.stale_reasons)

    graph._snapshot = GraphSnapshotState(
        pages=original.pages, edges=(), communities=((7, ("Wiki/a.md",)),),
    )
    lost = service.retrieve()
    assert lost.reports == ()
    assert any("membership is stale" in reason for reason in lost.stale_reasons)


def test_graph_snapshot_failure_is_reported_as_stale_not_token_counter_failure():
    """A graph read failure must direct operators to graph/report remediation."""
    from obsidian_wiki.application.community_report_service import CommunityReportService
    from obsidian_wiki.domain.community_report_models import CommunityReportStatus

    class FailingGraph(_Graph):
        def read(self):
            if self.failed:
                raise RuntimeError("graph snapshot is unavailable")
            return super().read()

    graph = FailingGraph(_fresh_snapshot())
    graph.failed = False
    store = _ReportStore()
    service = CommunityReportService(store, graph, _NamedTokenCounter())
    service.build()
    graph.failed = True

    outcome = service.retrieve()
    assert outcome.status is CommunityReportStatus.STALE
    assert outcome.stale_reasons == ("current graph snapshot is unavailable",)


def test_global_report_budget_skips_oversized_report_and_keeps_fitting_lower_rank():
    """An oversized high-ranked report must not prevent a lower-ranked fit."""
    from obsidian_wiki.application.community_report_service import CommunityReportService
    from obsidian_wiki.domain.community_report_models import GraphSnapshotState, PageSnapshot

    class Counter(_NamedTokenCounter):
        def count(self, text):
            return 10 if text.startswith("Community 7") else 2

    snapshot = GraphSnapshotState(
        pages=(PageSnapshot("Wiki/a.md", "a"), PageSnapshot("Wiki/b.md", "b")),
        edges=(), communities=((7, ("Wiki/a.md",)), (8, ("Wiki/b.md",))),
    )
    store = _ReportStore()
    service = CommunityReportService(store, _Graph(snapshot), Counter())
    service.build()

    outcome = service.retrieve(query_terms=("community",), k=2, max_tokens=5)
    assert outcome.status.value == "community_reports_fresh"
    assert [report.community_id for report in outcome.reports] == [8]
    assert outcome.stale_reasons == ("one or more community reports exceeded the effective token budget",)


def test_query_preserves_fresh_no_fit_budget_diagnostic(monkeypatch, tmp_path):
    """A budget omission is fresh-but-empty evidence, never a tokenizer outage."""
    from obsidian_wiki.domain.community_report_models import CommunityReportStatus, GlobalRetrievalOutcome
    from query import _global_retrieve
    from query_planner import DefaultQueryPlanner
    import query

    class BudgetLimitedService:
        def retrieve(self, **kwargs):
            return GlobalRetrievalOutcome(
                status=CommunityReportStatus.FRESH,
                stale_reasons=("one or more community reports exceeded the effective token budget",),
            )

    monkeypatch.setattr(query, "compose_global_report_service", lambda root: BudgetLimitedService())

    class Wiki:
        index_dir = tmp_path / ".index"

    plan = DefaultQueryPlanner().plan("全局概述")
    result = _global_retrieve(Wiki(), plan, k=5, max_tokens=1)

    assert result.status == "community_reports_fresh"
    assert result.community_report_status == "community_reports_fresh"
    assert result.stale_reasons == ["one or more community reports exceeded the effective token budget"]
    assert result.bundle.items == []


def test_fresh_global_route_adapts_validated_reports_to_public_result(monkeypatch, tmp_path):
    """The query wrapper exposes fresh report evidence through its normal JSON shape."""
    from obsidian_wiki.application.community_report_service import CommunityReportService
    from query import hybrid_search, result_to_json
    from query_planner import DefaultQueryPlanner
    import query

    service = CommunityReportService(_ReportStore(), _Graph(_fresh_snapshot()), _NamedTokenCounter())
    service.build()
    monkeypatch.setattr(query, "compose_global_report_service", lambda root: service)

    class Wiki:
        index_dir = tmp_path / ".index"
        pages = ()

    result = hybrid_search(Wiki(), "summarize the whole wiki", DefaultQueryPlanner(),
                           intent_override="global", max_tokens=100)
    payload = result_to_json(result)

    assert payload["status"] == "community_reports_fresh"
    assert payload["mode"] == "summary"
    assert payload["text"] and payload["text"][0]["method"] == "global_community_report"
    # issue #43: reports are not Wiki pages — they keep the community id as path
    # and expose no citation, instead of being forced into a Wiki/... shape.
    assert payload["text"][0]["path"] == str(result.bundle.items[0].page_id)
    assert payload["text"][0]["citation"] is None
    assert result.bundle.budget_contract_violations() == []


def test_rejected_global_routing_requires_explicit_local_fallback(monkeypatch, tmp_path):
    """A typed report rejection cannot reach local retrieval without one explicit flag."""
    from obsidian_wiki.domain.community_report_models import CommunityReportStatus, GlobalRetrievalOutcome
    from models import ContextBundle, ContextItem
    from query import _exit_code_for_result, format_for_agent, hybrid_search, result_to_json
    from query_planner import DefaultQueryPlanner
    import query

    class MissingService:
        def retrieve(self, **kwargs):
            return GlobalRetrievalOutcome(
                status=CommunityReportStatus.MISSING,
                stale_reasons=("active report set is missing",),
            )

    calls = []
    monkeypatch.setattr(query, "compose_global_report_service", lambda root: MissingService())
    candidate = SimpleNamespace(page_id="local", rrf_score=1.0)
    monkeypatch.setattr(query, "_retrieve_for_plan", lambda *args: calls.append(args) or ([object()], [], [candidate], [candidate], 0))
    monkeypatch.setattr(
        query,
        "assemble_context",
        lambda *args, **kwargs: ContextBundle(
            query="q", mode="snippet", max_context_tokens=kwargs["max_tokens"],
            items=[ContextItem("local", "Wiki/local.md", "Local", "rrf", "section", text="local evidence", sources=["Raw/a"], token_count=2)],
            token_count=2,
        ),
    )

    class Wiki:
        index_dir = tmp_path / ".index"
        pages = ()

        @staticmethod
        def count_tokens(text):
            return len(text.split())

    strict = hybrid_search(Wiki(), "summarize globally", DefaultQueryPlanner(), intent_override="global")
    strict_payload = result_to_json(strict)
    assert calls == []
    assert strict_payload["status"] == "community_reports_missing"
    assert strict_payload["local_fallback_used"] is False
    assert strict_payload["required_action"] == "build-community-reports"
    assert _exit_code_for_result(strict) != 0
    strict_rendered = format_for_agent(strict)
    assert "community_reports_missing" not in strict_rendered
    assert "全局报告尚未构建" in strict_rendered
    assert "python scripts/build_community_reports.py <知识库根目录>" in strict_rendered

    fallback = hybrid_search(Wiki(), "summarize globally", DefaultQueryPlanner(),
                             intent_override="global", allow_local_fallback=True)
    fallback_payload = result_to_json(fallback)
    assert len(calls) == 1
    assert fallback_payload["status"] == "local_fallback_used"
    assert fallback_payload["mode"] == "local_fallback"
    assert fallback_payload["community_report_status"] == "community_reports_missing"
    assert fallback_payload["local_fallback_used"] is True
    assert fallback_payload["confidence_warning"]
    assert all(item["method"] != "global_community_report" for item in fallback_payload["text"])
    assert _exit_code_for_result(fallback) == 0
    fallback_rendered = format_for_agent(fallback)
    assert "降级的本地证据" in fallback_rendered
