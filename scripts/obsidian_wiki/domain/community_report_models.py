"""SDK-free immutable records for the version-2 community-report contract."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable


COMMUNITY_REPORT_SCHEMA_VERSION = 2
REBUILD_ACTION = "build-community-reports"


class _JsonRecord:
    def to_json(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=str, ensure_ascii=False))


def _stable_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PageSnapshot(_JsonRecord):
    page_id: str
    content_hash: str


@dataclass(frozen=True)
class GraphEdge(_JsonRecord):
    source: str
    target: str
    signals: tuple[str, ...]
    weight: float

    def normalized(self) -> dict[str, object]:
        return {
            "source": min(self.source, self.target),
            "target": max(self.source, self.target),
            "signals": sorted(set(self.signals)),
            "weight": self.weight,
        }


@dataclass(frozen=True)
class GraphSnapshotState(_JsonRecord):
    pages: tuple[PageSnapshot, ...]
    edges: tuple[GraphEdge, ...]
    communities: tuple[tuple[int, tuple[str, ...]], ...]

    def members_for(self, community_id: int) -> tuple[str, ...]:
        for candidate_id, members in self.communities:
            if candidate_id == community_id:
                return tuple(sorted(members))
        return ()


def member_fingerprint(members: Iterable[PageSnapshot]) -> str:
    return _stable_sha256([
        {"page_id": member.page_id, "content_hash": member.content_hash}
        for member in sorted(members, key=lambda item: item.page_id)
    ])


def edge_fingerprint(edges: Iterable[GraphEdge]) -> str:
    normalized = [edge.normalized() for edge in edges]
    return _stable_sha256(sorted(
        normalized,
        key=lambda item: (item["source"], item["target"], tuple(item["signals"]), item["weight"]),
    ))


@dataclass(frozen=True)
class CommunityReport(_JsonRecord):
    report_schema_version: int
    community_id: int
    member_page_ids: tuple[str, ...]
    member_fingerprint: str
    edge_fingerprint: str
    content_hash: str
    token_count: int
    token_counter_id: str
    generated_at: str
    title: str
    text: str


@dataclass(frozen=True)
class CommunityReportManifest(_JsonRecord):
    report_schema_version: int
    build_id: str
    report_count: int
    token_counter_id: str
    generated_at: str
    stale_at: str | None = None
    stale_producer: str | None = None
    stale_reason: str | None = None

    @property
    def is_stale(self) -> bool:
        return bool(self.stale_at or self.stale_producer or self.stale_reason)


class CommunityReportStatus(str, Enum):
    FRESH = "community_reports_fresh"
    MISSING = "community_reports_missing"
    SCHEMA_UNSUPPORTED = "community_reports_schema_unsupported"
    STALE = "community_reports_stale"
    TOKEN_COUNTER_UNAVAILABLE = "token_counter_unavailable"


@dataclass(frozen=True)
class GlobalRetrievalOutcome(_JsonRecord):
    status: CommunityReportStatus
    reports: tuple[CommunityReport, ...] = ()
    stale_reasons: tuple[str, ...] = ()
    required_action: str = REBUILD_ACTION
    local_fallback_used: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
