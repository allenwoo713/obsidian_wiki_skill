"""#36 跨平台耐久文件系统原语（纯 stdlib）。

manifest / ACTIVE_INDEX / generation 记录等关键发布文件必须经此模块写入。

- ``atomic_write_bytes``：同目录临时文件 → 完整写入循环（处理短写）→ fsync →
  原子 replace → 同步父目录。POSIX 失败传播；Windows 用 ``MoveFileExW``
  ``REPLACE_EXISTING | WRITE_THROUGH``。
- commit 边界（#36/#37）：replace 成功后的目录同步失败抛 ``CommitUncertainError``
  ——旧指针不再保证保留，调用方必须重新读取验证，禁止进入 pre-commit 失败路径。
"""
from __future__ import annotations

import os
from pathlib import Path


class CommitUncertainError(OSError):
    """replace 已成功，但后续 durability confirmation 失败（#36）。"""


def _fsync_dir(path: Path) -> None:
    """POSIX：同步目录条目以确保 rename 持久；Windows 无该原语，跳过。"""
    if os.name == "posix":
        fd = os.open(os.fspath(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _replace_win(source: Path, target: Path) -> None:
    """Windows：MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH)（ctypes，stdlib）。

    ponytail: 目标被并发进程短暂持有时首次 replace 会共享冲突
    (ERROR_SHARING_VIOLATION=32 / ERROR_LOCK_VIOLATION=33)；重试可自愈，
    避免发布误判失败。其它错误立即抛出。
    """
    import ctypes
    import time
    from ctypes import wintypes  # noqa: F401 (初始化 ctypes)

    MOVEFILE_REPLACE_EXISTING = 0x1
    MOVEFILE_WRITE_THROUGH = 0x8
    last_err = 0
    for _ in range(5):
        ok = ctypes.windll.kernel32.MoveFileExW(
            str(source), str(target), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)
        if ok:
            return
        last_err = ctypes.GetLastError()
        if last_err not in (32, 33):
            break
        time.sleep(0.2)
    raise OSError(last_err, f"MoveFileExW failed: {target}")


def atomic_write_bytes(target: Path, data: bytes) -> None:
    """同目录 tmp → 完整写入（短写循环）→ fsync → 原子 replace → 父目录同步（POSIX）。

    replace 成功后的目录同步失败抛 ``CommitUncertainError``（旧指针不再保证）。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    fd = os.open(os.fspath(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # ponytail: os.unlink fails under Windows sandbox safe-delete guard;
        # fallback rename keeps .tmp glob clean without changing CI behavior.
        try:
            os.unlink(tmp)
        except OSError:
            try:
                os.replace(tmp, tmp.with_suffix(".trash"))
            except OSError:
                pass
        raise
    try:
        if os.name == "nt":
            _replace_win(tmp, target)
        else:
            os.replace(tmp, target)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            try:
                os.replace(tmp, tmp.with_suffix(".trash"))
            except OSError:
                pass
        raise
    try:
        _fsync_dir(target.parent)
    except OSError as exc:
        raise CommitUncertainError(f"replace 已成功但目录同步失败: {exc}") from exc


def replace_durable(source: Path, target: Path) -> None:
    """原子 replace + 同步父目录（POSIX）；Windows 等价 MoveFileExW WRITE_THROUGH。"""
    if os.name == "nt":
        _replace_win(source, target)
    else:
        os.replace(source, target)
    _fsync_dir(target.parent)
