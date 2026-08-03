"""#21 单写者构建锁：.index/BUILD.lock（跨平台，纯 stdlib）。

约定：
- 获取：``os.open(O_CREAT|O_EXCL)`` 原子创建锁文件，元数据含 pid/hostname/started_at/build_id。
- 释放：仅持有者（同 pid）删除锁文件；进程内同路径可重入（facade + service 双层获取）。
- stale：pid 已死视为 stale，自动回收并重新获取；存活进程的锁绝不抢占。
- 失败：默认立即抛 ``BuildLockHeldError``；``wait=True`` 时轮询等待至 timeout。
- 生命周期映射：building（持锁）→ validated（manifest 落盘）→ published（指针翻转）→ superseded（下一次构建）。
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_LOCK_NAME = "BUILD.lock"
_held: dict[str, int] = {}  # 进程内重入计数：lock path -> depth


class BuildLockHeldError(RuntimeError):
    """另一存活进程正持有构建锁。"""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # Windows 上 os.kill(pid, 0) 会发 CTRL_C_EVENT（0 == CTRL_C_EVENT），
        # 可能干扰目标进程；改用 OpenProcess 只查存在性。
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 存在但属其他用户
    except OSError:
        return False
    return True


class BuildLock:
    """单写者构建锁。stale 判定依赖 pid 存活；pid 复用理论上可能误判，频率可忽略。"""

    def __init__(self, index_dir: Path, build_id: str = ""):
        self.path = Path(index_dir) / _LOCK_NAME
        self.build_id = build_id
        self._depth = 0

    def acquire(self, wait: bool = False, timeout: float = 300.0) -> None:
        key = os.fspath(self.path)
        if key in _held:  # 同进程重入（WikiIndex.build + service.build 双层）
            _held[key] += 1
            self._depth += 1
            return
        deadline = time.monotonic() + max(0.0, timeout)
        self.path.parent.mkdir(parents=True, exist_ok=True)  # 首次构建时 .index 可能尚不存在
        while True:
            try:
                fd = os.open(key, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                holder = self._read()
                if holder is None or not _pid_alive(holder.get("pid", -1)):
                    log.warning(
                        "reclaiming stale %s held by pid=%s", _LOCK_NAME, (holder or {}).get("pid")
                    )
                    try:
                        os.unlink(key)
                    except FileNotFoundError:
                        pass
                    continue
                if wait and time.monotonic() < deadline:
                    time.sleep(0.5)
                    continue
                raise BuildLockHeldError(
                    f"另一构建正持有 {self.path}（pid={holder.get('pid')}, "
                    f"hostname={holder.get('hostname')}, started_at={holder.get('started_at')}）。"
                    f"如确认该进程已死可删除锁文件后重试。"
                ) from None
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "hostname": socket.gethostname(),
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "build_id": self.build_id,
                        "tool": "build_index",
                    },
                    fh,
                    sort_keys=True,
                )
            _held[key] = 1
            self._depth = 1
            return

    def release(self) -> None:
        key = os.fspath(self.path)
        if key not in _held or self._depth <= 0:
            return
        _held[key] -= 1
        self._depth -= 1
        if _held[key] > 0:
            return
        holder = self._read()
        if holder is None or holder.get("pid") == os.getpid():
            try:
                os.unlink(key)
            except FileNotFoundError:
                pass
        _held.pop(key, None)

    def _read(self) -> dict | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None
