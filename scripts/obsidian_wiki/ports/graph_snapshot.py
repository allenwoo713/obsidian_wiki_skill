"""SDK-free graph facts consumed by the report service."""
from __future__ import annotations

from typing import Protocol

from obsidian_wiki.domain.community_report_models import GraphSnapshotState


class GraphSnapshot(Protocol):
    def read(self) -> GraphSnapshotState: ...
