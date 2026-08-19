"""Durable identity-bound recovery journal for staged online index mutations."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from obsidian_wiki.application.durable_filesystem import atomic_write_bytes
from obsidian_wiki.domain.incremental_models import IncrementalJournalRecord, IncrementalJournalState


_PREDECESSOR = {
    IncrementalJournalState.CLONED: IncrementalJournalState.PREPARED,
    IncrementalJournalState.MUTATED: IncrementalJournalState.CLONED,
    IncrementalJournalState.CAUGHT_UP: IncrementalJournalState.MUTATED,
    IncrementalJournalState.VALIDATED: IncrementalJournalState.CAUGHT_UP,
    IncrementalJournalState.PUBLISHED: IncrementalJournalState.VALIDATED,
}


class FilesystemIncrementalJournal:
    """One strict JSON record per build under ``.index/incremental_journal``."""

    def __init__(self, index_dir: Path):
        self._dir = Path(index_dir) / "incremental_journal"

    def _path(self, build_id: str) -> Path:
        return self._dir / f"{build_id}.json"

    def _write(self, record: IncrementalJournalRecord) -> None:
        atomic_write_bytes(
            self._path(record.build_id),
            json.dumps(record.to_json(), sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )

    def load(self, build_id: str) -> IncrementalJournalRecord | None:
        path = self._path(build_id)
        if not path.is_file():
            return None
        try:
            return IncrementalJournalRecord.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def prepare(self, record: IncrementalJournalRecord) -> IncrementalJournalRecord:
        if record.state is not IncrementalJournalState.PREPARED:
            raise ValueError("incremental journal must begin prepared")
        current = self.load(record.build_id)
        if current is not None:
            if current != record:
                raise ValueError("incremental journal build identity already exists")
            return current
        self._write(record)
        return record

    def transition(self, build_id: str, target: IncrementalJournalState, *, boundary: str) -> IncrementalJournalRecord:
        current = self.load(build_id)
        if current is None:
            raise ValueError("incremental journal record is unavailable or malformed")
        if current.state is target:
            return current
        if current.state in (IncrementalJournalState.PUBLISHED, IncrementalJournalState.ABORTED) or _PREDECESSOR.get(target) is not current.state:
            raise ValueError(f"illegal incremental journal transition: {current.state.value} -> {target.value}")
        next_record = replace(current, state=target, last_completed_boundary=boundary)
        self._write(next_record)
        return next_record

    def checkpoint(self, build_id: str, *, boundary: str) -> IncrementalJournalRecord:
        """Durably record a sub-boundary without skipping the public state machine."""
        current = self.load(build_id)
        if current is None or current.state in (IncrementalJournalState.PUBLISHED, IncrementalJournalState.ABORTED):
            raise ValueError("incremental journal cannot checkpoint a terminal or malformed record")
        if current.last_completed_boundary == boundary:
            return current
        checked = replace(current, last_completed_boundary=boundary)
        self._write(checked)
        return checked

    def abort(self, build_id: str, reason: str) -> IncrementalJournalRecord | None:
        current = self.load(build_id)
        if current is None:
            return None
        if current.state is IncrementalJournalState.ABORTED:
            return current
        if current.state is IncrementalJournalState.PUBLISHED:
            return current
        aborted = replace(current, state=IncrementalJournalState.ABORTED, abort_reason=reason)
        self._write(aborted)
        return aborted

    def nonterminal(self) -> tuple[IncrementalJournalRecord, ...]:
        if not self._dir.is_dir():
            return ()
        records: list[IncrementalJournalRecord] = []
        for path in sorted(self._dir.glob("*.json")):
            record = self.load(path.stem)
            if record is not None and record.state not in (
                IncrementalJournalState.PUBLISHED, IncrementalJournalState.ABORTED,
            ):
                records.append(record)
        return tuple(records)

    def records(self) -> tuple[IncrementalJournalRecord, ...]:
        if not self._dir.is_dir():
            return ()
        return tuple(
            record for path in sorted(self._dir.glob("*.json"))
            if (record := self.load(path.stem)) is not None
        )

    def has_invalid_records(self) -> bool:
        if not self._dir.is_dir():
            return False
        return any(self.load(path.stem) is None for path in self._dir.glob("*.json"))
