"""SDK-free immutable records for staged online index mutations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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
