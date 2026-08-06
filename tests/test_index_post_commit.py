"""#35/#37 publication lifecycle and durable post-commit acceptance gates.

These tests deliberately cover commit windows that ordinary happy-path builds cannot
exercise: a pointer committed before its lifecycle record, retry of a matching pending
intent, and construction of a publisher without the required journal port.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import obsidian_wiki.application.active_index_pointer as pointer_module  # noqa: E402
from obsidian_wiki.application.active_index_pointer import (  # noqa: E402
    POINTER_NAME,
    publish_pointer,
    read_generation_record,
    record_building,
    record_validated,
    resolve_active_lance_dir,
)
from obsidian_wiki.application.index_build_service import IndexBuildService  # noqa: E402
from obsidian_wiki.application.post_commit_service import retry_pending  # noqa: E402
from obsidian_wiki.domain.index_models import (  # noqa: E402
    PostCommitTask,
    PostCommitTaskState,
)
from obsidian_wiki.domain.index_publication_models import GenerationState  # noqa: E402
from obsidian_wiki.infrastructure.filesystem_post_commit_journal import (  # noqa: E402
    FilesystemPostCommitJournal,
)


def _build_id(tag: str) -> str:
    suffix = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:32]
    return f"build_20260806T000000000000_{suffix}"


def _validated_build(index_dir: Path, *, tag: str, generation: int = 1) -> tuple[Path, str]:
    build_id = _build_id(tag)
    build_dir = index_dir / "builds" / build_id
    (build_dir / "lance_db").mkdir(parents=True)
    record_building(build_dir, build_id=build_id, generation=generation)
    manifest = build_dir / "manifest.json"
    manifest.write_text(
        json.dumps({"build_id": build_id, "generation": generation}, sort_keys=True),
        encoding="utf-8",
    )
    record_validated(
        build_dir,
        build_id=build_id,
        generation=generation,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    return build_dir, build_id


def _fail_first_published_transition(monkeypatch):
    real_transition = pointer_module._transition
    failed = False

    def _flaky(build_dir, target, **kwargs):
        nonlocal failed
        if target is GenerationState.PUBLISHED and not failed:
            failed = True
            raise OSError("simulated lifecycle fsync failure")
        return real_transition(build_dir, target, **kwargs)

    monkeypatch.setattr(pointer_module, "_transition", _flaky)
    return real_transition


def test_building_record_does_not_claim_validation(tmp_path):
    """#35：BUILDING 的 validated_at 必须为空，只有 VALIDATED transition 才赋值。"""
    index_dir = tmp_path / ".index"
    build_id = _build_id("building")
    build_dir = index_dir / "builds" / build_id
    build_dir.mkdir(parents=True)

    record_building(build_dir, build_id=build_id, generation=1)
    building = json.loads((build_dir / ".generation.json").read_text(encoding="utf-8"))
    assert building["state"] == GenerationState.BUILDING.value
    assert building["validated_at"] is None

    manifest = build_dir / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    record_validated(
        build_dir,
        build_id=build_id,
        generation=1,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    validated = json.loads((build_dir / ".generation.json").read_text(encoding="utf-8"))
    assert validated["validated_at"] is not None
    assert datetime.fromisoformat(validated["validated_at"]).astimezone(timezone.utc).utcoffset().total_seconds() == 0


def test_committed_pointer_reconciles_validated_record(tmp_path, monkeypatch):
    """#35/#37：pointer 已提交时，即使 PUBLISHED record 首写失败，新代仍须可恢复。"""
    index_dir = tmp_path / ".index"
    index_dir.mkdir()
    build_dir, build_id = _validated_build(index_dir, tag="reconcile")
    _fail_first_published_transition(monkeypatch)

    publish_pointer(index_dir, build_dir, generation=1, build_id=build_id)

    assert (index_dir / POINTER_NAME).is_file(), "测试必须越过 pointer commit point"
    assert resolve_active_lance_dir(index_dir) == build_dir / "lance_db"
    record = read_generation_record(build_dir)
    assert record is not None and record.state is GenerationState.PUBLISHED


def test_retry_pending_never_cancels_matching_committed_pointer(tmp_path, monkeypatch):
    """#37：匹配已提交 pointer 的 PREPARED intent 必须 reconciliation，不能取消。"""
    index_dir = tmp_path / ".index"
    index_dir.mkdir()
    build_dir, build_id = _validated_build(index_dir, tag="pending")
    journal = FilesystemPostCommitJournal(index_dir)
    task = PostCommitTask(
        task_id="f" * 32,
        task_type="community_report_invalidation",
        build_id=build_id,
        generation=1,
        state=PostCommitTaskState.PREPARED,
        prepared_at=datetime.now(timezone.utc).isoformat(),
    )
    journal.prepare(task)
    real_transition = _fail_first_published_transition(monkeypatch)
    publish_pointer(index_dir, build_dir, generation=1, build_id=build_id)
    monkeypatch.setattr(pointer_module, "_transition", real_transition)

    class _Invalidator:
        def mark_stale(self, *, producer, reason):
            return True

    retry_pending(index_dir, journal=journal, invalidator=_Invalidator())
    persisted = json.loads(
        (index_dir / "post_commit_tasks" / f"{task.task_id}.json").read_text(encoding="utf-8")
    )
    assert persisted["state"] != PostCommitTaskState.CANCELLED.value


def test_index_build_service_requires_post_commit_journal():
    """#37：所有 publication 必须显式选择真实 journal 或 deliberate no-op port。"""
    parameter = inspect.signature(IndexBuildService).parameters["post_commit_journal"]
    assert parameter.default is inspect.Parameter.empty
