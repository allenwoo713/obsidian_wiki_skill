"""Persistence boundary for immutable staged community-report sets."""
from __future__ import annotations

from typing import Protocol, Sequence

from obsidian_wiki.domain.community_report_models import CommunityReport, CommunityReportManifest


class CommunityReportSetStore(Protocol):
    def stage(self, build_id: str, reports: Sequence[CommunityReport], manifest: CommunityReportManifest) -> None: ...

    def read_staged(self, build_id: str) -> tuple[tuple[CommunityReport, ...], CommunityReportManifest] | None: ...

    def activate(self, build_id: str) -> None: ...

    def read_active(self) -> tuple[tuple[CommunityReport, ...], CommunityReportManifest] | None: ...
