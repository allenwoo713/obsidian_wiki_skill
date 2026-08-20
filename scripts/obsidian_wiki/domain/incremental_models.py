"""SDK-free immutable records for staged online index mutations."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple

from obsidian_wiki.domain.index_models import StorageArtifact, _JsonRecord


@dataclass(frozen=True)
class SourceTableIdentity(_JsonRecord):
    table_name: str
    version: int
    row_count: int

    def __post_init__(self) -> None:
        if self.table_name not in {"sparse_chunks", "dense_chunks"}:
            raise ValueError("source identity must name a canonical chunk table")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("source table version must be a non-negative integer")
        if not isinstance(self.row_count, int) or self.row_count < 0:
            raise ValueError("source table row_count must be a non-negative integer")


@dataclass(frozen=True)
class TableDelta(_JsonRecord):
    """Stable-ID plan for exactly one physical table; no SDK rows escape here."""

    table_name: str
    added_ids: Tuple[str, ...] = ()
    updated_ids: Tuple[str, ...] = ()
    deleted_ids: Tuple[str, ...] = ()
    unchanged_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.table_name not in {"sparse_chunks", "dense_chunks"}:
            raise ValueError("delta must name a canonical chunk table")
        groups = (self.added_ids, self.updated_ids, self.deleted_ids, self.unchanged_ids)
        ids = [item for group in groups for item in group]
        if any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("delta identities must be non-empty strings")
        if len(ids) != len(set(ids)):
            raise ValueError("a stable identity may belong to only one delta set")

    @property
    def physically_written_ids(self) -> Tuple[str, ...]:
        return self.added_ids + self.updated_ids


@dataclass(frozen=True)
class CoverageObservation(_JsonRecord):
    table_name: str
    row_count: int
    indexed_rows: int
    unindexed_rows: int | None

    def __post_init__(self) -> None:
        if self.table_name not in {"sparse_chunks", "dense_chunks"}:
            raise ValueError("coverage must name a canonical chunk table")
        if any(not isinstance(value, int) or value < 0 for value in (self.row_count, self.indexed_rows)):
            raise ValueError("coverage counts must be non-negative integers")
        if self.unindexed_rows is not None and (
            not isinstance(self.unindexed_rows, int) or self.unindexed_rows < 0
        ):
            raise ValueError("unindexed_rows must be a non-negative integer or None")


@dataclass(frozen=True)
class MutationResult(_JsonRecord):
    table_name: str
    inserted: int
    updated: int
    deleted: int

    @property
    def physically_written(self) -> int:
        return self.inserted + self.updated


_BUILD_MODE_POLICY_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUILD_MODE_METRICS = frozenset({
    "snapshot_p95_ms", "peak_staged_disk_bytes", "index_rebuild_ms",
})


class BuildModeContractDriftCode(str, Enum):
    """Typed mismatch identities persisted by the public auto dispatcher."""

    FTS_CONFIG = "fts_config"


def _finite_non_negative(field_name: str, value: object) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{field_name} must be finite and non-negative")


@dataclass(frozen=True)
class BuildTiming(_JsonRecord):
    """Measured monotonic durations at the production build boundaries."""

    scan_parse_ms: float
    chunking_ms: float
    embedding_cache_hit_ms: float
    embedding_cache_miss_ms: float
    serialization_write_ms: float
    fts_catch_up_ms: float
    vector_catch_up_ms: float
    validation_ms: float
    publication_ms: float
    index_rebuild_ms: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            _finite_non_negative(name, value)


@dataclass(frozen=True)
class TableRowCounts(_JsonRecord):
    """Observed per-table write facts; unchanged rows never count as writes."""

    inserted: int
    updated: int
    deleted: int
    unchanged: int
    physically_written: int

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.physically_written != self.inserted + self.updated:
            raise ValueError("physically_written must equal inserted + updated")


@dataclass(frozen=True)
class BuildTelemetry(_JsonRecord):
    """Schema-v1 observation eligible only after a PUBLISHED snapshot is proven."""

    schema_version: int
    observation_id: str
    mode_requested: str
    mode_selected: str
    selection_reason: str
    compatibility_digest: str
    completed_at_epoch_seconds: float
    timings: BuildTiming
    sparse_rows: TableRowCounts
    dense_rows: TableRowCounts
    embedding_cache_hits: int
    embedding_cache_misses: int
    peak_staged_disk_bytes: int
    completed: bool
    build_mode_policy_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("build telemetry schema_version must be 1")
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise ValueError("build telemetry observation_id is required")
        if self.mode_requested not in {"snapshot", "incremental", "auto"}:
            raise ValueError("build telemetry requested mode is invalid")
        if self.mode_selected not in {"snapshot", "incremental"}:
            raise ValueError("build telemetry selected mode is invalid")
        if not isinstance(self.selection_reason, str) or not self.selection_reason:
            raise ValueError("build telemetry selection_reason is required")
        if not isinstance(self.compatibility_digest, str) or not _BUILD_MODE_POLICY_SHA256.fullmatch(self.compatibility_digest):
            raise ValueError("build telemetry compatibility_digest must be sha256")
        _finite_non_negative("completed_at_epoch_seconds", self.completed_at_epoch_seconds)
        for name in ("embedding_cache_hits", "embedding_cache_misses", "peak_staged_disk_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.completed, bool):
            raise ValueError("build telemetry completed must be boolean")
        if self.build_mode_policy_sha256 is not None and (
            not isinstance(self.build_mode_policy_sha256, str)
            or not _BUILD_MODE_POLICY_SHA256.fullmatch(self.build_mode_policy_sha256)
        ):
            raise ValueError("build telemetry policy digest must be sha256 or None")


@dataclass(frozen=True)
class BuildModeCriterion(_JsonRecord):
    metric: str
    operator: str
    threshold: float

    def __post_init__(self) -> None:
        if self.metric not in _BUILD_MODE_METRICS or self.operator != "gte":
            raise ValueError("build-mode criterion is invalid")
        if (isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float))
                or not math.isfinite(self.threshold) or self.threshold <= 0):
            raise ValueError("build-mode criterion threshold must be finite and positive")


@dataclass(frozen=True)
class BuildModePolicy(_JsonRecord):
    schema_version: int
    enabled: bool
    compatibility_digest: str | None = None
    evidence_observation_ids: Tuple[str, ...] = ()
    minimum_compatible_observations: int | None = None
    max_evidence_age_seconds: float | None = None
    match: str | None = None
    criteria: Tuple[BuildModeCriterion, ...] = ()
    compatibility_contract: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not isinstance(self.enabled, bool):
            raise ValueError("build-mode policy schema is invalid")
        if not self.enabled:
            if any(value not in (None, ()) for value in (
                self.compatibility_digest, self.evidence_observation_ids,
                self.minimum_compatible_observations, self.max_evidence_age_seconds,
                self.match, self.criteria, self.compatibility_contract,
            )):
                raise ValueError("disabled policy carries enabled-only fields")
            return
        if not isinstance(self.compatibility_digest, str) or not _BUILD_MODE_POLICY_SHA256.fullmatch(self.compatibility_digest):
            raise ValueError("enabled policy compatibility_digest must be sha256")
        if self.compatibility_contract is not None and not isinstance(self.compatibility_contract, dict):
            raise ValueError("policy compatibility_contract must be an object when supplied")
        if (not self.evidence_observation_ids
                or any(not isinstance(item, str) or not item for item in self.evidence_observation_ids)
                or len(set(self.evidence_observation_ids)) != len(self.evidence_observation_ids)):
            raise ValueError("enabled policy evidence IDs must be unique and non-empty")
        if (isinstance(self.minimum_compatible_observations, bool)
                or not isinstance(self.minimum_compatible_observations, int)
                or self.minimum_compatible_observations <= 0):
            raise ValueError("enabled policy minimum observations must be positive")
        if (isinstance(self.max_evidence_age_seconds, bool)
                or not isinstance(self.max_evidence_age_seconds, (int, float))
                or not math.isfinite(self.max_evidence_age_seconds)
                or self.max_evidence_age_seconds <= 0):
            raise ValueError("enabled policy evidence age must be finite and positive")
        if self.match not in {"any", "all"} or not self.criteria:
            raise ValueError("enabled policy match and criteria are required")


@dataclass(frozen=True)
class BuildModePolicyLoad(_JsonRecord):
    policy: BuildModePolicy | None
    policy_sha256: str | None
    reason: str


@dataclass(frozen=True)
class BuildModeSelection(_JsonRecord):
    selected_mode: str
    reason: str
    policy_sha256: str | None
    compatibility_digest: str
    evidence_observation_ids: Tuple[str, ...]
    calculated_values: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.selected_mode not in {"snapshot", "incremental"}:
            raise ValueError("selected build mode is invalid")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("build-mode selection reason is required")
        if not _BUILD_MODE_POLICY_SHA256.fullmatch(self.compatibility_digest):
            raise ValueError("build-mode selection compatibility digest must be sha256")


@dataclass(frozen=True)
class IncrementalBuildResult(_JsonRecord):
    artifact: StorageArtifact
    source_tables: Tuple[SourceTableIdentity, ...]
    sparse_delta: TableDelta
    dense_delta: TableDelta
    sparse_mutation: MutationResult
    dense_mutation: MutationResult
    sparse_coverage: CoverageObservation
    dense_coverage: CoverageObservation


class IncrementalJournalState(str, Enum):
    """Durable staged-mutation boundaries; only the pointer may make a build visible."""

    PREPARED = "prepared"
    CLONED = "cloned"
    MUTATED = "mutated"
    CAUGHT_UP = "caught_up"
    VALIDATED = "validated"
    PUBLISHED = "published"
    ABORTED = "aborted"


_JOURNAL_BUILD_ID = re.compile(r"^build_\d{8}T\d{12}_[0-9a-f]{32}$")
_JOURNAL_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class IncrementalJournalRecord(_JsonRecord):
    """Strict, identity-bound durable intent for a single staged mutation."""

    schema_version: int
    build_id: str
    generation: int
    state: IncrementalJournalState
    prior_pointer_sha256: str
    source_build_id: str
    source_tables: Tuple[SourceTableIdentity, ...]
    plan_sha256: str
    config_sha256: str
    policy_sha256: str
    target_build: str
    last_completed_boundary: str
    abort_reason: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("incremental journal schema_version must be 1")
        if not _JOURNAL_BUILD_ID.fullmatch(self.build_id):
            raise ValueError("incremental journal build_id is invalid")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("incremental journal generation must be positive")
        if not isinstance(self.state, IncrementalJournalState):
            raise ValueError("incremental journal state is invalid")
        if any(not _JOURNAL_SHA256.fullmatch(value) for value in (
            self.prior_pointer_sha256, self.plan_sha256, self.config_sha256, self.policy_sha256,
        )):
            raise ValueError("incremental journal digests must be sha256 values")
        if not _JOURNAL_BUILD_ID.fullmatch(self.source_build_id):
            raise ValueError("incremental journal source_build_id is invalid")
        if tuple(sorted(identity.table_name for identity in self.source_tables)) != (
            "dense_chunks", "sparse_chunks",
        ):
            raise ValueError("incremental journal requires both source table identities")
        if self.target_build != f"builds/{self.build_id}":
            raise ValueError("incremental journal target_build must be contained build path")
        if not isinstance(self.last_completed_boundary, str) or not self.last_completed_boundary:
            raise ValueError("incremental journal boundary is required")
        if self.state is IncrementalJournalState.ABORTED:
            if not isinstance(self.abort_reason, str) or not self.abort_reason:
                raise ValueError("aborted incremental journal requires a reason")
        elif self.abort_reason is not None:
            raise ValueError("only aborted incremental journal records may contain a reason")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "build_id": self.build_id,
            "generation": self.generation,
            "state": self.state.value,
            "prior_pointer_sha256": self.prior_pointer_sha256,
            "source_build_id": self.source_build_id,
            "source_tables": [identity.to_json() for identity in self.source_tables],
            "plan_sha256": self.plan_sha256,
            "config_sha256": self.config_sha256,
            "policy_sha256": self.policy_sha256,
            "target_build": self.target_build,
            "last_completed_boundary": self.last_completed_boundary,
            "abort_reason": self.abort_reason,
        }

    @classmethod
    def from_json(cls, data: object) -> "IncrementalJournalRecord":
        fields = {
            "schema_version", "build_id", "generation", "state", "prior_pointer_sha256", "source_build_id",
            "source_tables", "plan_sha256", "config_sha256", "policy_sha256", "target_build",
            "last_completed_boundary", "abort_reason",
        }
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("incremental journal record fields are invalid")
        source = data["source_tables"]
        if not isinstance(source, list):
            raise ValueError("incremental journal source_tables must be a list")
        try:
            identities = tuple(SourceTableIdentity(**item) for item in source)
            return cls(
                schema_version=data["schema_version"], build_id=data["build_id"],
                generation=data["generation"], state=IncrementalJournalState(data["state"]),
                prior_pointer_sha256=data["prior_pointer_sha256"], source_build_id=data["source_build_id"], source_tables=identities,
                plan_sha256=data["plan_sha256"], config_sha256=data["config_sha256"],
                policy_sha256=data["policy_sha256"], target_build=data["target_build"],
                last_completed_boundary=data["last_completed_boundary"], abort_reason=data["abort_reason"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("incremental journal record is malformed") from exc
