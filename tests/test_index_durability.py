"""#36 durability 原语测试（模型无关，Windows + Linux CI）。

- 短写：``os.write`` 每次只写部分字节，最终文件必须逐字节等于完整 payload。
- pre-replace 失败：tmp 写入/fsync 失败 → target 保持原字节（旧指针保留）。
- CommitUncertain：replace 成功但目录同步失败 → 必须返回非成功，禁止用缓存读回冒充耐久确认。
- storage seal：任一存储文件无法打开/fsync 都必须使发布失败。
- Windows：MoveFileExW 必须使用 REPLACE_EXISTING | WRITE_THROUGH 并传播失败。
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
    _replace_win,
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


def test_publish_directory_fsync_failure_is_not_success(tmp_path, monkeypatch):
    """#36：replace 后目录 fsync 失败必须是非成功，缓存读回不是耐久证据。"""
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
    with pytest.raises(CommitUncertainError, match="目录同步失败|dir fsync failed"):
        publish_pointer(idx, build, generation=1, build_id=build_id)
    assert calls, "目录 fsync 应被调用（并失败）"
    # replace 可能已发生，但调用方绝不能收到成功；生命周期也不能伪装成 PUBLISHED。
    assert (idx / POINTER_NAME).is_file(), "commit-uncertain 允许新 pointer 已可见"
    record = read_generation_record(build)
    assert record is not None and record.state == GenerationState.VALIDATED


def test_storage_seal_propagates_file_open_failure(tmp_path, monkeypatch):
    """#36：seal 不能跳过打不开的存储文件，否则发布物不在已证明的耐久边界内。"""
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    lance_dir = tmp_path / "lance_db"
    lance_dir.mkdir()
    data_file = lance_dir / "data.lance"
    data_file.write_bytes(b"storage")
    real_open = os.open

    def _open(path, flags, *args):
        if Path(path) == data_file:
            raise PermissionError("simulated sharing violation")
        return real_open(path, flags, *args)

    monkeypatch.setattr(os, "open", _open)
    with pytest.raises(PermissionError, match="sharing violation"):
        LanceDbIndexRepository(lance_dir).seal(lance_dir)


@pytest.mark.skipif(os.name != "nt", reason="Windows MoveFileExW contract")
def test_windows_replace_uses_write_through_and_propagates_failure(tmp_path, monkeypatch):
    """#36：Windows 原子替换必须携带 WRITE_THROUGH，系统调用失败必须上抛。"""
    import ctypes
    from types import SimpleNamespace

    observed: dict[str, object] = {}

    def _move_file(source, target, flags):
        observed.update(source=source, target=target, flags=flags)
        return 0

    monkeypatch.setattr(
        ctypes,
        "windll",
        SimpleNamespace(kernel32=SimpleNamespace(MoveFileExW=_move_file)),
    )
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5)

    with pytest.raises(OSError, match="MoveFileExW failed"):
        _replace_win(tmp_path / "source", tmp_path / "target")
    assert observed["flags"] == 0x1 | 0x8


@pytest.mark.skipif(os.name != "nt", reason="Windows MoveFileExW contract")
def test_windows_replace_retries_on_sharing_violation_then_succeeds(tmp_path, monkeypatch):
    """#36：首次共享冲突(32)重试后应成功，不误判发布失败。"""
    import ctypes
    from types import SimpleNamespace

    observed: dict[str, object] = {}
    calls = {"n": 0}

    def _move_file(source, target, flags):
        observed.update(flags=flags)
        calls["n"] += 1
        # 前两次模拟共享冲突，第三次成功
        return 0 if calls["n"] <= 2 else 1

    monkeypatch.setattr(
        ctypes,
        "windll",
        SimpleNamespace(kernel32=SimpleNamespace(MoveFileExW=_move_file)),
    )
    monkeypatch.setattr(ctypes, "GetLastError", lambda: 32)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)  # 跳过退避等待

    _replace_win(tmp_path / "source", tmp_path / "target")
    assert calls["n"] == 3, "应在第 3 次重试成功"
    assert observed["flags"] == 0x1 | 0x8, "重试仍须携带 WRITE_THROUGH"


@pytest.mark.skipif(os.name != "nt", reason="Windows MoveFileExW contract")
def test_windows_replace_exhausts_retries_on_persistent_lock(tmp_path, monkeypatch):
    """#36：持续锁冲突(33)超过重试上限(5)必须上抛 OSError。"""
    import ctypes
    from types import SimpleNamespace

    calls = {"n": 0}

    def _move_file(source, target, flags):
        calls["n"] += 1
        return 0  # 始终失败

    monkeypatch.setattr(
        ctypes,
        "windll",
        SimpleNamespace(kernel32=SimpleNamespace(MoveFileExW=_move_file)),
    )
    monkeypatch.setattr(ctypes, "GetLastError", lambda: 33)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)  # 跳过退避等待

    with pytest.raises(OSError, match="MoveFileExW failed"):
        _replace_win(tmp_path / "source", tmp_path / "target")
    assert calls["n"] == 5, "重试上限应为 range(5)=5 次"
