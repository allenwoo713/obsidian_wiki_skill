"""SDK-free collaborators required by staged incremental publication."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, Sequence

from obsidian_wiki.domain.incremental_models import (
    BuildModeContractDriftCode,
    IncrementalBuildResult,
    IncrementalJournalRecord,
    IncrementalJournalState,
)
from obsidian_wiki.domain.index_models import BuildContext, SparseChunk
from obsidian_wiki.ports.chunk_repository import ChunkRepository


class IncrementalFallbackEligible(RuntimeError):
    """A typed, pre-publication signal that auto mode may safely snapshot."""

    def __init__(
        self, reason: str, *, contract_drift: BuildModeContractDriftCode | None = None,
    ) -> None:
        self.reason = reason
        self.contract_drift = contract_drift
        super().__init__(reason)

    @property
    def selection_reason(self) -> str:
        return self.reason if self.contract_drift is None else f"{self.reason}:{self.contract_drift.value}"


class IncrementalJournal(Protocol):
    def load(self, build_id: str) -> IncrementalJournalRecord | None: ...
    def prepare(self, record: IncrementalJournalRecord) -> IncrementalJournalRecord: ...
    def transition(self, build_id: str, target: IncrementalJournalState, *, boundary: str) -> IncrementalJournalRecord: ...
    def checkpoint(self, build_id: str, *, boundary: str) -> IncrementalJournalRecord: ...
    def abort(self, build_id: str, reason: str) -> IncrementalJournalRecord | None: ...
    def nonterminal(self) -> tuple[IncrementalJournalRecord, ...]: ...
    def records(self) -> tuple[IncrementalJournalRecord, ...]: ...
    def has_invalid_records(self) -> bool: ...


ChunkRepositoryFactory = Callable[[Path], ChunkRepository]
IncrementalJournalFactory = Callable[[Path], IncrementalJournal]


class IncrementalBuildExecutor(Protocol):
    def build_staged(
        self,
        wiki_dir: Path,
        index_dir: Path,
        *,
        canonical_chunks: Sequence[SparseChunk],
        embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
        page_metadata: list[dict] | None = None,
        ctx: BuildContext,
        mode_requested: str = "incremental",
        selection_reason: str = "explicit_incremental",
        build_mode_policy_sha256: str | None = None,
        outer_lock_held: bool = False,
    ) -> IncrementalBuildResult: ...


class IncrementalExecutorFactory(Protocol):
    def __call__(self) -> IncrementalBuildExecutor: ...
