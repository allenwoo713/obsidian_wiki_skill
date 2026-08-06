"""#36 跨平台耐久文件系统原语（纯 stdlib）。

manifest / ACTIVE_INDEX / generation 记录等关键发布文件必须经此模块写入，
禁止其它模块自行 ``write_text + replace``：

- ``atomic_write_bytes``：同目录临时文件 → flush/fsync → 原子 replace → 同步父目录。
- 任何 fsync/replace 失败都向上传播——#36 明确关键同步失败不允许降级为 warning，
  调用方必须把发布视为失败并保留旧指针。
- Windows 无目录 fsync 原语：由文件级 fsync + 原子 replace 提供等价耐久保证。
"""
from __future__ import annotations

import os
from pathlib import Path


def _fsync_dir(path: Path) -> None:
    """POSIX：同步目录条目以确保 rename 持久；Windows 无该原语，跳过。

    POSIX 上失败必须向上传播（#36：目录同步失败 = 发布失败）。
    """
    if os.name == "posix":
        fd = os.open(os.fspath(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write_bytes(target: Path, data: bytes) -> None:
    """同目录临时文件写入 → fsync 文件 → 原子 replace → fsync 父目录（POSIX）。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    fd = os.open(os.fspath(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(tmp, target)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(target.parent)


def replace_durable(source: Path, target: Path) -> None:
    """原子 replace + 同步父目录（POSIX）；Windows 等价于 os.replace。"""
    os.replace(source, target)
    _fsync_dir(target.parent)
