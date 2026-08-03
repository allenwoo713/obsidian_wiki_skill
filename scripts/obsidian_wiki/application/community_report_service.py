"""Build, validate, and gate community-report sets before publication or use."""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import replace
from typing import Iterable

from obsidian_wiki.domain.community_report_models import (
    COMMUNITY_REPORT_SCHEMA_VERSION,
    REBUILD_ACTION,
    CommunityReport,
    CommunityReportManifest,
    CommunityReportStatus,
    GlobalRetrievalOutcome,
    GraphEdge,
    GraphSnapshotState,
    PageSnapshot,
    edge_fingerprint,
    member_fingerprint,
    utc_now,
)
from obsidian_wiki.ports.community_reports import CommunityReportSetStore
from obsidian_wiki.ports.graph_snapshot import GraphSnapshot
from obsidian_wiki.ports.token_counter import TokenCounter, TokenCounterUnavailable


class CommunityReportService:
    """The policy owner; adapters may only stage, reopen, and activate sets."""

    def __init__(self, store: CommunityReportSetStore, graph: GraphSnapshot, token_counter: TokenCounter):
        self._store = store
        self._graph = graph
        self._token_counter = token_counter

    def build(self) -> CommunityReportManifest:
        snapshot = self._graph.read()
        build_id = f"community_{uuid.uuid4().hex}"
        generated_at = utc_now()
        reports = tuple(self._build_report(snapshot, community_id, members, generated_at)
                        for community_id, members in snapshot.communities)
        manifest = CommunityReportManifest(
            report_schema_version=COMMUNITY_REPORT_SCHEMA_VERSION,
            build_id=build_id,
            report_count=len(reports),
            token_counter_id=self._token_counter.identity,
            generated_at=generated_at,
        )
        self._store.stage(build_id, reports, manifest)
        reopened = self._store.read_staged(build_id)
        if reopened is None:
            self._record_failure(build_id, "staged community-report set could not be reopened")
            raise RuntimeError("staged community-report set could not be reopened")
        valid, reasons = self._validate(*reopened, snapshot)
        if not valid:
            self._record_failure(build_id, "; ".join(reasons))
            raise RuntimeError("staged community-report validation failed: " + "; ".join(reasons))
        self._store.activate(build_id)
        return manifest

    def retrieve(
        self,
        *,
        query_terms: Iterable[str] = (),
        k: int | None = None,
        max_tokens: int | None = None,
    ) -> GlobalRetrievalOutcome:
        """Return only fresh reports selected with this service's strict counter.

        Persisted report token counts validate the artifact, but selection always
        recounts the text admitted to the caller's effective budget.  This keeps
        the trust boundary in the application layer instead of a query wrapper.
        """
        active = self._store.read_active()
        if active is None:
            return self._rejected(CommunityReportStatus.MISSING, "active report set is missing")
        if not isinstance(active, tuple) or len(active) != 2:
            return self._rejected(CommunityReportStatus.SCHEMA_UNSUPPORTED, "active report set contract is malformed")
        reports, manifest = active
        if not isinstance(manifest, CommunityReportManifest):
            return self._rejected(CommunityReportStatus.SCHEMA_UNSUPPORTED, "report manifest contract is malformed")
        if not isinstance(reports, tuple) or any(not isinstance(report, CommunityReport) for report in reports):
            return self._rejected(CommunityReportStatus.SCHEMA_UNSUPPORTED, "report record contract is malformed")
        if manifest.is_stale:
            reason = manifest.stale_reason or "report set is marked stale"
            return self._rejected(CommunityReportStatus.STALE, reason)
        try:
            snapshot = self._graph.read()
        except Exception:
            return self._rejected(CommunityReportStatus.STALE, "current graph snapshot is unavailable")
        try:
            valid, reasons = self._validate(reports, manifest, snapshot)
        except TokenCounterUnavailable:
            return self._rejected(CommunityReportStatus.TOKEN_COUNTER_UNAVAILABLE, "report token counter is unavailable")
        if not valid:
            status = CommunityReportStatus.SCHEMA_UNSUPPORTED if any(
                "schema" in reason or "count" in reason for reason in reasons
            ) else CommunityReportStatus.STALE
            return self._rejected(status, *reasons)
        selected, selection_reason = self._select_reports(
            reports, query_terms=query_terms, k=k, max_tokens=max_tokens
        )
        if selection_reason is not None:
            return GlobalRetrievalOutcome(
                status=CommunityReportStatus.FRESH,
                reports=selected,
                stale_reasons=(selection_reason,),
            )
        return GlobalRetrievalOutcome(status=CommunityReportStatus.FRESH, reports=selected)

    def _select_reports(
        self,
        reports: Iterable[CommunityReport],
        *,
        query_terms: Iterable[str],
        k: int | None,
        max_tokens: int | None,
    ) -> tuple[tuple[CommunityReport, ...], str | None]:
        terms = {str(term).lower() for term in query_terms if str(term).strip()}

        def score(report: CommunityReport) -> tuple[int, int]:
            text = f"{report.title} {report.text}".lower()
            return (-sum(term in text for term in terms), report.community_id)

        ranked = sorted(reports, key=score)
        if k is not None:
            ranked = ranked[:k]
        selected: list[CommunityReport] = []
        used = 0
        for report in ranked:
            try:
                actual_count = self._token_counter.count(report.text)
            except Exception:
                raise
            if max_tokens is not None and used + actual_count > max_tokens:
                continue
            selected.append(replace(report, token_count=actual_count))
            used += actual_count
        if len(selected) != len(ranked):
            return tuple(selected), "one or more community reports exceeded the effective token budget"
        return tuple(selected), None

    def _build_report(
        self, snapshot: GraphSnapshotState, community_id: int, member_ids: tuple[str, ...], generated_at: str
    ) -> CommunityReport:
        pages = self._pages_for(snapshot, member_ids)
        edges = self._edges_for(snapshot.edges, member_ids)
        ordered_ids = tuple(sorted(member_ids))
        title = f"Community {community_id}"
        text = f"{title}: " + ", ".join(ordered_ids)
        return CommunityReport(
            report_schema_version=COMMUNITY_REPORT_SCHEMA_VERSION,
            community_id=community_id,
            member_page_ids=ordered_ids,
            member_fingerprint=member_fingerprint(pages),
            edge_fingerprint=edge_fingerprint(edges),
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            token_count=self._token_counter.count(text),
            token_counter_id=self._token_counter.identity,
            generated_at=generated_at,
            title=title,
            text=text,
        )

    def _validate(
        self, reports: Iterable[CommunityReport], manifest: CommunityReportManifest, snapshot: GraphSnapshotState
    ) -> tuple[bool, tuple[str, ...]]:
        reports = tuple(reports)
        reasons: list[str] = []
        if manifest.report_schema_version != COMMUNITY_REPORT_SCHEMA_VERSION:
            reasons.append("manifest schema is unsupported")
        if len(reports) != manifest.report_count:
            reasons.append("report count disagrees with manifest")
        if manifest.token_counter_id != self._token_counter.identity:
            reasons.append("manifest token counter identity disagrees")
        for report in reports:
            if report.report_schema_version != COMMUNITY_REPORT_SCHEMA_VERSION or report.report_schema_version != manifest.report_schema_version:
                reasons.append("report schema disagrees with manifest")
                continue
            current_ids = snapshot.members_for(report.community_id)
            community_exists = any(
                community_id == report.community_id for community_id, _ in snapshot.communities
            )
            if not community_exists or tuple(sorted(report.member_page_ids)) != current_ids:
                reasons.append(f"community {report.community_id} membership is stale")
                continue
            pages = self._pages_for(snapshot, current_ids)
            if len(pages) != len(current_ids):
                reasons.append(f"community {report.community_id} member source is missing")
            elif member_fingerprint(pages) != report.member_fingerprint:
                reasons.append(f"community {report.community_id} member fingerprint is stale")
            if edge_fingerprint(self._edges_for(snapshot.edges, current_ids)) != report.edge_fingerprint:
                reasons.append(f"community {report.community_id} edge fingerprint is stale")
            if hashlib.sha256(report.text.encode("utf-8")).hexdigest() != report.content_hash:
                reasons.append(f"community {report.community_id} content hash disagrees")
            if report.token_counter_id != self._token_counter.identity:
                reasons.append(f"community {report.community_id} token counter identity disagrees")
            elif self._token_counter.count(report.text) != report.token_count:
                reasons.append(f"community {report.community_id} token budget disagrees")
        return not reasons, tuple(reasons)

    @staticmethod
    def _pages_for(snapshot: GraphSnapshotState, member_ids: Iterable[str]) -> tuple[PageSnapshot, ...]:
        by_id = {page.page_id: page for page in snapshot.pages}
        return tuple(by_id[page_id] for page_id in member_ids if page_id in by_id)

    @staticmethod
    def _edges_for(edges: Iterable[GraphEdge], member_ids: Iterable[str]) -> tuple[GraphEdge, ...]:
        members = set(member_ids)
        return tuple(edge for edge in edges if edge.source in members and edge.target in members)

    @staticmethod
    def _rejected(status: CommunityReportStatus, *reasons: str) -> GlobalRetrievalOutcome:
        return GlobalRetrievalOutcome(
            status=status,
            stale_reasons=tuple(reasons),
            required_action=REBUILD_ACTION,
            local_fallback_used=False,
        )

    def _record_failure(self, build_id: str, reason: str) -> None:
        recorder = getattr(self._store, "record_failure", None)
        if recorder is not None:
            recorder(build_id, reason)
