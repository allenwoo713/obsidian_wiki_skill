"""#37 post-commit 服务：重放 PREPARED 任务（幂等，commit-aware）。

取消一个 PREPARED 任务前必须先做 pointer/record 身份 reconciliation（#37
follow-up）：若 pointer 已提交该 generation（哪怕 record 仍是 VALIDATED），
先修复 record 为 PUBLISHED 再执行任务；reconciliation 不可用则保留 PREPARED，
绝不把匹配已提交 pointer 的任务标为 CANCELLED。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from obsidian_wiki.application.active_index_pointer import (
    read_generation_record,
    reconcile_committed_record,
)
from obsidian_wiki.domain.index_publication_models import GenerationState


@dataclass(frozen=True)
class RetrySummary:
    completed: int
    still_pending: int
    cancelled: int = 0


def retry_pending(index_dir: Path, *, journal, invalidator) -> RetrySummary:
    """重放 post-commit 任务（#37）；journal 由调用方注入（application 不依赖 infrastructure 实现）。

    只处理 generation 已 published/superseded（或经 reconciliation 修复为
    published）的 task；``mark_stale`` 与 ``complete`` 幂等，连续调用两次第二次
    为 no-op。
    """
    completed = 0
    still_pending = 0
    cancelled = 0
    for task in journal.pending():
        build_dir = Path(index_dir) / "builds" / task.build_id
        record = read_generation_record(build_dir)
        if record is None or record.state is GenerationState.BUILDING:
            # 从未生成/从未发布：任务不适用，取消。
            journal.cancel(task.task_id)
            cancelled += 1
            continue
        if record.state not in (GenerationState.PUBLISHED, GenerationState.SUPERSEDED):
            # record 仍是 VALIDATED：先 reconciliation——若 pointer 已提交该
            # generation，修复为 PUBLISHED 并执行；若未提交，任务不适用，取消。
            if not reconcile_committed_record(
                index_dir, build_dir, build_id=task.build_id, generation=task.generation
            ):
                journal.cancel(task.task_id)
                cancelled += 1
                continue
        try:
            invalidator.mark_stale(producer="build_index", reason="index_published")
            journal.complete(task.task_id)
            completed += 1
        except Exception:
            still_pending += 1
    return RetrySummary(completed=completed, still_pending=still_pending, cancelled=cancelled)
