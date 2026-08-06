"""#37 post-commit 服务：重放 PREPARED 任务（幂等）。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from obsidian_wiki.application.active_index_pointer import read_generation_record
from obsidian_wiki.domain.index_publication_models import GenerationState


@dataclass(frozen=True)
class RetrySummary:
    completed: int
    still_pending: int


def retry_pending(index_dir: Path, *, journal, invalidator) -> RetrySummary:
    """重放 post-commit 任务（#37）；journal 由调用方注入（application 不依赖 infrastructure 实现）。

    只处理 generation 已经 published/superseded 的 task；``mark_stale`` 与
    ``complete`` 幂等，连续调用两次第二次为 no-op。
    """
    completed = 0
    still_pending = 0
    for task in journal.pending():
        build_dir = Path(index_dir) / "builds" / task.build_id
        record = read_generation_record(build_dir)
        if record is None or record.state not in (
            GenerationState.PUBLISHED, GenerationState.SUPERSEDED,
        ):
            # pointer 从未发布（或已回退）：任务不适用，取消。
            journal.cancel(task.task_id)
            continue
        try:
            invalidator.mark_stale(producer="build_index", reason="index_published")
            journal.complete(task.task_id)
            completed += 1
        except Exception:
            still_pending += 1
    return RetrySummary(completed=completed, still_pending=still_pending)
