"""#21 单写者构建锁：owner-scoped OS advisory lock（跨平台，纯 stdlib）。

设计（review 修复，PR1）：
- 跨进程互斥：持久 lockfile 上的 OS 级 exclusive lock（POSIX ``fcntl.flock`` /
  Windows ``msvcrt.locking``）。OS lock 在进程崩溃/退出时**自动释放**，
  无需 pid 存活判定，也不做"读 metadata → 判 stale → unlink"删除式回收。
- 进程内线程互斥：每路径 ``threading.RLock``（同线程嵌套可重入，独立线程必须等待）。
  review 明确推荐"每路径 threading.RLock 加 OS 级锁"方案——独立线程不得借由
  同进程绕过文件锁。
- owner nonce（hostname+pid+thread+uuid）写入 lockfile metadata **仅供诊断**；
  unreadable/fresh/foreign-host metadata 一律视为"可能被占用"，但**不据此抢占**——
  抢占决策只看 OS lock 真实状态（获取成功即持有，失败即被占用）。因此不会因本机
  PID 不存在而抢占其它 hostname 的锁。
- build_id：UTC microseconds + 完整 UUID，最外层生成一次（#34 完整 UUID 约定）。
- release：最后一层只 close fd（解除 OS lock）→ RLock.release()。BUILD.lock 是
  稳定 pathname/inode，绝不在 release 时 rename/unlink（#34：避免 unlock→删除
  窗口内第三进程抢锁与并发写入）。
- 生命周期映射：building（持锁）→ validated（manifest 落盘）→ published（指针翻转）
  → superseded（下一代发布）。
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from obsidian_wiki.domain.index_models import BuildContext

log = logging.getLogger(__name__)

_LOCK_NAME = "BUILD.lock"
_OS_LOCK_POS = 65536  # Windows mandatory lock 锁此高位字节，避开 metadata 内容区

try:
    import fcntl as _fcntl  # POSIX

    _PLATFORM = "posix"
except ImportError:  # Windows
    import msvcrt as _msvcrt

    _PLATFORM = "msvcrt"


class _LockHolder:
    """per-path 共享状态：RLock（线程级重入）+ OS lock fd + 深度。

    不同 BuildLock 实例（facade / service 双层）在同一路径上共享同一个 holder，
    使同线程嵌套 acquire 可重入、不同线程必须等待 RLock。
    """

    __slots__ = ("rlock", "fd", "depth")

    def __init__(self) -> None:
        self.rlock = threading.RLock()
        self.fd: int | None = None
        self.depth = 0


_holders: dict[str, _LockHolder] = {}
_holders_guard = threading.Lock()


def _holder_for(key: str) -> _LockHolder:
    with _holders_guard:
        holder = _holders.get(key)
        if holder is None:
            holder = _LockHolder()
            _holders[key] = holder
        return holder


def new_build_context() -> BuildContext:
    """创建不可变构建上下文：#34 要求 build_id = UTC 微秒 + 完整随机 UUID。"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return BuildContext(
        build_id=f"build_{ts}_{uuid.uuid4().hex}",
        started_at=datetime.now(timezone.utc).isoformat(),
        owner_nonce=(
            f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}"
        ),
    )


def _try_os_lock(fd: int) -> None:
    """非阻塞获取 OS exclusive lock；被占用时抛 ``OSError``。

    Windows 上 ``msvcrt.locking`` 是 mandatory lock（强制锁）——锁定的字节区域
    会被阻止读写。为避免阻止 metadata 诊断读取，Windows 锁定高位字节（65536），
    该位置远超 metadata 长度，不影响 metadata 内容区的读写。POSIX ``flock`` 是
    advisory，不阻止读写，锁整个文件即可。
    """
    if _PLATFORM == "posix":
        _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    else:
        os.lseek(fd, _OS_LOCK_POS, os.SEEK_SET)
        _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)


def _release_os_lock(fd: int) -> None:
    if _PLATFORM == "posix":
        try:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
        except OSError:
            pass
    else:
        try:
            os.lseek(fd, _OS_LOCK_POS, os.SEEK_SET)
            _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
        except OSError:
            pass


class BuildLockHeldError(RuntimeError):
    """另一存活进程/线程正持有构建锁。"""


class BuildLock:
    """owner-scoped 单写者构建锁。

    跨进程靠 OS advisory lock（崩溃自动释放），进程内靠 ``threading.RLock``
    （同线程嵌套可重入，独立线程互斥）。不做 pid 存活判定与删除式回收——
    OS lock 是最可靠的 stale 指示：进程活着则锁持有，进程死了则锁自动释放。
    """

    def __init__(self, index_dir: Path, ctx: BuildContext):
        self.path = Path(index_dir) / _LOCK_NAME
        self.ctx = ctx

    def acquire(self, wait: bool = False, timeout: float = 300.0) -> None:
        key = os.fspath(self.path)
        holder = _holder_for(key)
        deadline = time.monotonic() + max(0.0, timeout)
        # 线程级：RLock 同线程可重入，不同线程阻塞/失败
        if wait:
            got = holder.rlock.acquire(timeout=timeout)
        else:
            got = holder.rlock.acquire(blocking=False)
        if not got:
            raise BuildLockHeldError(
                f"同进程另一线程正持有 {self.path}。"
            )
        # RLock 已持有（内部计数 +1）。检查 OS lock 是否已由本线程持有（重入）
        if holder.depth > 0:
            holder.depth += 1
            return
        # depth == 0：需获取 OS lock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            fd = os.open(key, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                _try_os_lock(fd)
            except OSError:
                os.close(fd)
                if wait and time.monotonic() < deadline:
                    time.sleep(0.5)
                    continue
                holder.rlock.release()
                raise BuildLockHeldError(
                    f"另一构建正持有 {self.path}（OS lock 被占用）。"
                    f"OS lock 会随持有进程退出/崩溃自动释放；如确认该进程已死，"
                    f"稍后重试即可，无需手动删除锁文件。"
                ) from None
            # OS lock 获取成功——写 metadata（仅供诊断，不影响锁语义）
            try:
                payload = json.dumps(
                    {
                        "pid": os.getpid(),
                        "hostname": socket.gethostname(),
                        "started_at": self.ctx.started_at,
                        "build_id": self.ctx.build_id,
                        "owner_nonce": self.ctx.owner_nonce,
                        "tool": "build_index",
                    },
                    sort_keys=True,
                ).encode("utf-8")
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(fd, payload)
                os.ftruncate(fd, len(payload))
                os.fsync(fd)
            except OSError as exc:
                log.warning("写入 lock metadata 失败（不影响锁持有）：%s", exc)
            holder.fd = fd
            holder.depth = 1
            return

    def release(self) -> None:
        key = os.fspath(self.path)
        holder = _holder_for(key)
        if holder.depth <= 0:
            return  # 未持有，no-op
        holder.depth -= 1
        if holder.depth > 0:
            holder.rlock.release()  # 内层重入释放
            return
        # depth == 0（#34）：仅释放自己持有的 descriptor + RLock。BUILD.lock 是
        # 稳定 pathname/inode，绝不在 release 时 rename/unlink——否则会在
        # unlock 与删除之间的窗口让第三进程抢到原 pathname 并与新 owner 并发写入。
        fd = holder.fd
        holder.fd = None
        if fd is not None:
            _release_os_lock(fd)
            try:
                os.close(fd)
            except OSError:
                pass
        holder.rlock.release()
