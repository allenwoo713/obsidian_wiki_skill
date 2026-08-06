"""#37 post-commit journal 文件系统实现（durable，禁止裸 write_text）。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from obsidian_wiki.application.durable_filesystem import atomic_write_bytes
from obsidian_wiki.domain.index_models import PostCommitTask, PostCommitTaskState


class FilesystemPostCommitJournal:
    """保存到 ``.index/post_commit_tasks/<task-id>.json``，全部经 #36 durable 写入。"""

    def __init__(self, index_dir: Path):
        self._dir = Path(index_dir) / "post_commit_tasks"

    def _path(self, task_id: str) -> Path:
        return self._dir / f"{task_id}.json"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load(self, task_id: str) -> PostCommitTask | None:
        path = self._path(task_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return PostCommitTask(
                task_id=str(data["task_id"]), task_type=str(data["task_type"]),
                build_id=str(data["build_id"]), generation=int(data["generation"]),
                state=PostCommitTaskState(data["state"]),
                prepared_at=str(data["prepared_at"]),
                completed_at=str(data["completed_at"]) if data.get("completed_at") else None,
            )
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    def _write(self, task: PostCommitTask) -> None:
        atomic_write_bytes(
            self._path(task.task_id),
            json.dumps(task.to_json(), sort_keys=True).encode("utf-8"),
        )

    def prepare(self, task: PostCommitTask) -> None:
        self._write(task)

    def complete(self, task_id: str) -> None:
        current = self._load(task_id)
        if current is None or current.state is not PostCommitTaskState.PREPARED:
            return  # 幂等：仅 PREPARED → COMPLETED
        self._write(PostCommitTask(
            task_id=current.task_id, task_type=current.task_type,
            build_id=current.build_id, generation=current.generation,
            state=PostCommitTaskState.COMPLETED, prepared_at=current.prepared_at,
            completed_at=self._now(),
        ))

    def cancel(self, task_id: str) -> None:
        current = self._load(task_id)
        if current is None or current.state is not PostCommitTaskState.PREPARED:
            return  # 幂等：仅 PREPARED → CANCELLED
        self._write(PostCommitTask(
            task_id=current.task_id, task_type=current.task_type,
            build_id=current.build_id, generation=current.generation,
            state=PostCommitTaskState.CANCELLED, prepared_at=current.prepared_at,
            completed_at=self._now(),
        ))

    def pending(self) -> tuple[PostCommitTask, ...]:
        if not self._dir.is_dir():
            return ()
        result: list[PostCommitTask] = []
        for path in sorted(self._dir.glob("*.json")):
            task = self._load(path.stem)
            if task is not None and task.state is PostCommitTaskState.PREPARED:
                result.append(task)
        return tuple(result)
