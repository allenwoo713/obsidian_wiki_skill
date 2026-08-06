"""#37 post-commit ports：提交点之后的衍生工作（journal + invalidator）。"""
from __future__ import annotations

from typing import Protocol

from obsidian_wiki.domain.index_models import PostCommitTask


class PostCommitJournal(Protocol):
    def prepare(self, task: PostCommitTask) -> None: ...
    def complete(self, task_id: str) -> None: ...
    def cancel(self, task_id: str) -> None: ...
    def pending(self) -> tuple[PostCommitTask, ...]: ...


class CommunityReportInvalidator(Protocol):
    def mark_stale(self, *, producer: str, reason: str) -> bool: ...
