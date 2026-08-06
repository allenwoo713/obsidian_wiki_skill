"""#36 durability 原语测试（模型无关，Windows + Linux CI）。

- 短写：``os.write`` 每次只写部分字节，最终文件必须逐字节等于完整 payload。
- pre-replace 失败：tmp 写入/fsync 失败 → target 保持原字节（旧指针保留）。
- CommitUncertain：replace 成功但目录同步失败 → publish 读回验证新 pointer 后视为已发布。
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from obsidian_wiki.application.active_index_pointer import (  # noqa: E402
    POINTER_NAME,
    publish_pointer,
    read_generation_record,
    resolve_active_lance_dir,
)
from obsidian_wiki.application.durable_filesystem import (  # noqa: E402
    CommitUncertainError,
    atomic_write_bytes,
)
from obsidian_wiki.domain.index_publication_models import (  # noqa: E402
    GenerationRecord,
    GenerationState,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _valid_build_id(tag: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    suffix = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:32] if tag else os.urandom(16).hex()
    return f"build_{ts}_{suffix}"


def _fake_build(idx: Path, build_id: str, generation: int = 1, state: str = "validated") -> Path:
    build = idx / "builds" / build_id
    (build / "lance_db").mkdir(parents=True, exist_ok=True)
    (build / "manifest.json").write_text(
        json.dumps({"layout": "sparse_chunks+dense_chunks", "generation": generation,
                    "build_id": build_id}), encoding="utf-8")
    digest = hashlib.sha256((build / "manifest.json").read_bytes()).hexdigest()
    record = GenerationRecord(
        generation=generation, build_id=build_id, state=GenerationState(state),
        manifest_sha256=digest, validated_at="2026-08-06T00:00:00+00:00",
        published_at="2026-08-06T00:00:00+00:00" if state == "published" else None,
    )
    (build / ".generation.json").write_text(
        json.dumps(record.to_json(), sort_keys=True), encoding="utf-8")
    return build


def test_atomic_write_retries_short_os_write(tmp_path, monkeypatch):
    """#36：os.write 每次只写 1 字节（短写）→ 最终文件逐字节等于完整 payload。"""
    real_write = os.write
    attempts: list[int] = []

    def _short_write(fd, data):
        view = memoryview(data)
        n = real_write(fd, view[:1])
        attempts.append(n)
        return n

    monkeypatch.setattr(os, "write", _short_write)
    target = tmp_path / "payload.bin"
    payload = b'{"a": 1, "b": [1, 2, 3]}' * 20
    atomic_write_bytes(target, payload)
    assert target.read_bytes() == payload, "短写循环后文件必须逐字节一致"
    assert len(attempts) == len(payload), "每次短写都应被循环补偿"


def test_pre_replace_failure_keeps_old_target(tmp_path, monkeypatch):
    """#36：tmp fsync 失败（replace 前）→ target 不改变（旧指针保留）。"""
    import obsidian_wiki.application.durable_filesystem as df

    target = tmp_path / "pointer.json"
    target.write_bytes(b"old-bytes")
    real_fsync = os.fsync

    def _boom_fsync(fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", _boom_fsync)
    with pytest.raises(OSError, match="fsync failure"):
        atomic_write_bytes(target, b"new-bytes")
    assert target.read_bytes() == b"old-bytes", "replace 前失败 → 旧 target 原样保留"
    monkeypatch.setattr(os, "fsync", real_fsync)
    assert not list(tmp_path.glob(".pointer.json.tmp")), "失败后 tmp 不残留"


def test_atomic_write_short_write_failure_propagates(tmp_path, monkeypatch):
    """#36：os.write 返回 0（短写死循环信号）→ 必须抛错而非静默截断。"""
    real_write = os.write

    def _zero_write(fd, data):
        return 0  # 无法推进 → 视为短写错误

    monkeypatch.setattr(os, "write", _zero_write)
    target = tmp_path / "f.bin"
    with pytest.raises(OSError, match="short write"):
        atomic_write_bytes(target, b"data")
    assert not target.exists()


def test_publish_commit_uncertain_reconciles_new_pointer(tmp_path, monkeypatch):
    """#36：replace 成功但目录同步失败（CommitUncertain）→ publish 读回验证后视为已发布。"""
    import obsidian_wiki.application.durable_filesystem as df

    idx = tmp_path / ".index"
    idx.mkdir(parents=True)
    build_id = _valid_build_id("a")
    build = _fake_build(idx, build_id, generation=1)
    calls: list[str] = []

    def _boom_fsync_dir(path):
        calls.append(str(path))
        raise OSError("dir fsync failed")

    monkeypatch.setattr(df, "_fsync_dir", _boom_fsync_dir)
    # 发布不应抛错：CommitUncertain 后读回验证新 pointer 有效 → 视为已发布（reconciliation）
    publish_pointer(idx, build, generation=1, build_id=build_id)
    assert calls, "目录 fsync 应被调用（并失败）"
    assert (idx / POINTER_NAME).is_file()
    assert resolve_active_lance_dir(idx) == build / "lance_db"
    record = read_generation_record(build)
    assert record is not None and record.state == GenerationState.PUBLISHED
