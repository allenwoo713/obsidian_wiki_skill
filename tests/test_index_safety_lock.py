"""#21 Index Safety：单写者构建锁 + ACTIVE_INDEX 指针契约（模型无关）。

PR1（Lock + build context）覆盖验收点：
- 并发构建只有一个写者：barrier 控制的两个独立线程、两个独立进程竞争同一项目，
  恰好一个 writer；同进程非嵌套（不同线程）调用不得 bypass 文件锁。
- OS advisory lock 持有期间（即使 metadata 损坏/foreign-host）其它进程不得 reclaim；
  OS lock 未持有时 foreign-host metadata 不阻止获取（不因本机 PID 不存在而抢占）。
- build ID 同微秒/并发唯一性；真实 facade lock metadata 完整性。
- release rename/unlink 失败不卡住新构建。

PR2（Generation publication + recovery）将替换指针测试。

纯 stdlib + 文件系统，可在 ci.yml architecture job（Windows + Linux）运行。
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from obsidian_wiki.application.active_index_pointer import (  # noqa: E402
    POINTER_NAME,
    publish_pointer,
    resolve_active_lance_dir,
)
from obsidian_wiki.application.build_lock import (  # noqa: E402
    BuildLock,
    BuildLockHeldError,
    new_build_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_NAME = "BUILD.lock"


def _index(tmp_path: Path) -> Path:
    idx = tmp_path / ".index"
    idx.mkdir(parents=True)
    return idx


def _fake_build(idx: Path, name: str, body: str = "ok") -> Path:
    """构造一份"已验证"构建产物：manifest.json + lance_db 目录（无需 LanceDB）。"""
    build = idx / "builds" / name
    (build / "lance_db").mkdir(parents=True, exist_ok=True)
    (build / "manifest.json").write_text(
        json.dumps({"layout": "sparse_chunks+dense_chunks", "body": body}), encoding="utf-8"
    )
    return build


def _pointer_data(idx: Path) -> dict:
    return json.loads((idx / POINTER_NAME).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1) owner-scoped 单写者构建锁（PR1）
# ---------------------------------------------------------------------------
def test_lock_acquire_release_writes_metadata(tmp_path):
    """acquire 后 BUILD.lock 含完整诊断 metadata；release 后 canonical path 释放。"""
    idx = _index(tmp_path)
    lock = BuildLock(idx, build_id="build_x")
    lock.acquire()
    data = json.loads((idx / LOCK_NAME).read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    assert data["hostname"] == socket.gethostname()
    assert data["started_at"]
    assert data["build_id"] == "build_x"
    assert data["owner_nonce"]
    lock.release()
    assert not (idx / LOCK_NAME).exists(), "release 后 canonical path 应释放"


def test_lock_reentrant_same_thread(tmp_path):
    """同线程嵌套 acquire（facade + service 双层）可重入；内层 release 不释放外层。"""
    idx = _index(tmp_path)
    outer = BuildLock(idx)
    outer.acquire()
    inner = BuildLock(idx)  # facade + service 双层获取
    inner.acquire()
    assert (idx / LOCK_NAME).exists()
    inner.release()
    assert (idx / LOCK_NAME).exists(), "内层 release 后外层仍持有"
    outer.release()
    assert not (idx / LOCK_NAME).exists()


def test_lock_blocks_different_thread(tmp_path):
    """同进程不同线程（非嵌套）不得绕过文件锁——review 核心要求。"""
    idx = _index(tmp_path)
    outer = BuildLock(idx)
    outer.acquire()
    errors = []

    def _try_inner():
        try:
            BuildLock(idx).acquire(wait=False)
        except BuildLockHeldError:
            errors.append("blocked")

    t = threading.Thread(target=_try_inner)
    t.start()
    t.join(timeout=5)
    assert errors == ["blocked"], "不同线程必须被 RLock 阻止，不得 bypass 文件锁"
    outer.release()


def test_concurrent_threads_one_writer(tmp_path):
    """barrier 控制的两个独立线程同时竞争：恰好一个 writer。"""
    idx = _index(tmp_path)
    barrier = threading.Barrier(2)
    results: list[str] = []

    def _worker():
        barrier.wait()
        lock = BuildLock(idx, build_id="t")
        try:
            lock.acquire(wait=False)
            results.append("acquired")
            time.sleep(0.3)
            lock.release()
        except BuildLockHeldError:
            results.append("blocked")

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert results.count("acquired") == 1, f"恰好一个 writer，got {results}"
    assert results.count("blocked") == 1, f"另一个被阻止，got {results}"


def test_concurrent_processes_one_writer(tmp_path):
    """barrier 控制的两个独立进程同时竞争同一项目：恰好一个 writer。"""
    idx = _index(tmp_path)
    go = tmp_path / "go"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    scripts_dir = REPO_ROOT / "scripts"
    code = (
        "import sys, time, os\n"
        f"sys.path.insert(0, {json.dumps(str(scripts_dir))})\n"
        "from pathlib import Path\n"
        "from obsidian_wiki.application.build_lock import BuildLock, BuildLockHeldError\n"
        f"idx = Path({json.dumps(str(idx))})\n"
        f"go = Path({json.dumps(str(go))})\n"
        f"results = Path({json.dumps(str(results_dir))})\n"
        "for _ in range(200):\n"
        "    if go.exists(): break\n"
        "    time.sleep(0.05)\n"
        "pid = os.getpid()\n"
        "try:\n"
        "    lock = BuildLock(idx, build_id='proc')\n"
        "    lock.acquire(wait=False)\n"
        "    (results / f'acquired_{pid}').write_text('ok', encoding='utf-8')\n"
        "    time.sleep(1.0)\n"
        "    lock.release()\n"
        "except BuildLockHeldError:\n"
        "    (results / f'blocked_{pid}').write_text('blocked', encoding='utf-8')\n"
    )
    p1 = subprocess.Popen([sys.executable, "-c", code], cwd=REPO_ROOT)
    p2 = subprocess.Popen([sys.executable, "-c", code], cwd=REPO_ROOT)
    try:
        time.sleep(0.6)  # 等两个进程都就绪
        go.write_text("go", encoding="utf-8")
        p1.wait(timeout=20)
        p2.wait(timeout=20)
    finally:
        for p in (p1, p2):
            if p.poll() is None:
                p.kill()
                p.wait(timeout=5)
    acquired = list(results_dir.glob("acquired_*"))
    blocked = list(results_dir.glob("blocked_*"))
    assert len(acquired) == 1, f"恰好一个进程获取锁，got acquired={len(acquired)} blocked={len(blocked)}"
    assert len(blocked) == 1, f"另一个被阻止，got acquired={len(acquired)} blocked={len(blocked)}"


def test_os_lock_held_blocks_reclaim_even_with_corrupt_metadata(tmp_path):
    """OS lock 持有期间，即使 metadata 损坏，其它进程也不得 reclaim。"""
    idx = _index(tmp_path)
    ready = tmp_path / "ready"
    scripts_dir = REPO_ROOT / "scripts"
    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {json.dumps(str(scripts_dir))})\n"
        "from pathlib import Path\n"
        "from obsidian_wiki.application.build_lock import BuildLock\n"
        f"lock = BuildLock(Path({json.dumps(str(idx))}), build_id='holder')\n"
        "lock.acquire()\n"
        f"Path({json.dumps(str(ready))}).write_text('ready', encoding='utf-8')\n"
        "time.sleep(15)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=REPO_ROOT)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.1)
        else:
            if not ready.exists():
                raise AssertionError("子进程未能在 15s 内取得锁")
        # 损坏 metadata（但 OS lock 仍被子进程持有）
        (idx / LOCK_NAME).write_bytes(b"corrupt-metadata")
        # 主进程尝试 acquire → 必须失败（OS lock 持有，不看 metadata）
        with pytest.raises(BuildLockHeldError):
            BuildLock(idx).acquire(wait=False)
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_foreign_host_metadata_reclaimable_when_os_unlocked(tmp_path):
    """foreign-host metadata 但 OS lock 未持有 → 可获取（不因 PID 不存在而抢占）。"""
    idx = _index(tmp_path)
    (idx / LOCK_NAME).write_text(
        json.dumps({
            "pid": 2 ** 31 - 1, "hostname": "other-host",
            "started_at": "x", "build_id": "foreign",
        }),
        encoding="utf-8",
    )
    # OS lock 未被持有 → 新进程应成功获取（metadata foreign-host 不阻止）
    lock = BuildLock(idx, build_id="mine")
    lock.acquire()  # 不抛异常
    data = json.loads((idx / LOCK_NAME).read_text(encoding="utf-8"))
    assert data["build_id"] == "mine"
    assert data["hostname"] == socket.gethostname()
    lock.release()


def test_release_rename_failure_does_not_block_new_build(tmp_path, monkeypatch):
    """release 时 rename tombstone 失败不应抛异常，且不卡住新构建。"""
    idx = _index(tmp_path)
    lock = BuildLock(idx, build_id="first")
    lock.acquire()
    real_replace = os.replace

    def _fail_replace(*_args, **_kwargs):
        raise PermissionError("simulated rename failure")

    monkeypatch.setattr(os, "replace", _fail_replace)
    lock.release()  # 不应抛异常
    monkeypatch.setattr(os, "replace", real_replace)
    # OS lock 已释放（fd closed）→ 新构建应能获取
    lock2 = BuildLock(idx, build_id="second")
    lock2.acquire()
    lock2.release()


def test_build_id_concurrent_uniqueness():
    """并发生成 build_id 全部唯一（UTC microseconds + UUID）。"""
    ids: set[str] = set()
    barrier = threading.Barrier(8)

    def _gen():
        barrier.wait()
        for _ in range(100):
            ids.add(new_build_id())

    threads = [threading.Thread(target=_gen) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(ids) == 800, f"800 个并发 build_id 应全部唯一，got {len(ids)}"


# ---------------------------------------------------------------------------
# 2) ACTIVE_INDEX 指针契约（PR2 将重写为严格 generation/schema 测试）
# ---------------------------------------------------------------------------
def test_publish_includes_checksum_and_resolves(tmp_path):
    idx = _index(tmp_path)
    build = _fake_build(idx, "build_a")
    publish_pointer(idx, build)
    data = _pointer_data(idx)
    assert Path(data["active_lance"]) == Path("builds/build_a/lance_db")
    assert data["schema_version"] == 3
    assert data["manifest_sha256"]
    assert resolve_active_lance_dir(idx) == build / "lance_db"


def test_legacy_pointer_without_checksum_accepted(tmp_path):
    idx = _index(tmp_path)
    build = _fake_build(idx, "build_a")
    (idx / POINTER_NAME).write_text(
        json.dumps({"active_lance": "builds/build_a/lance_db", "schema_version": 2}),
        encoding="utf-8",
    )
    assert resolve_active_lance_dir(idx) == build / "lance_db"


def test_torn_pointer_falls_back_to_newest_validated_build(tmp_path):
    idx = _index(tmp_path)
    _fake_build(idx, "build_old")
    newest = _fake_build(idx, "build_new")
    (idx / POINTER_NAME).write_bytes(b"{torn-json")  # 进程中断留下的截断指针
    assert resolve_active_lance_dir(idx) == newest / "lance_db"


def test_checksum_mismatch_falls_back_to_previous(tmp_path):
    idx = _index(tmp_path)
    old = _fake_build(idx, "build_old")
    new = _fake_build(idx, "build_new")
    publish_pointer(idx, old)
    publish_pointer(idx, new)
    (new / "manifest.json").write_text(json.dumps({"body": "tampered"}), encoding="utf-8")
    # 指针校验失败 → 回退上一份已验证索引，且不重选被拒的 new
    assert resolve_active_lance_dir(idx) == old / "lance_db"


def test_pointer_target_dir_missing_falls_back(tmp_path):
    idx = _index(tmp_path)
    build = _fake_build(idx, "build_a")
    (idx / POINTER_NAME).write_text(
        json.dumps({"active_lance": "builds/gone/lance_db",
                    "schema_version": 3, "manifest_sha256": "x"}),
        encoding="utf-8",
    )
    assert resolve_active_lance_dir(idx) == build / "lance_db"


def test_publish_failure_keeps_old_pointer(tmp_path, monkeypatch):
    idx = _index(tmp_path)
    old = _fake_build(idx, "build_a")
    publish_pointer(idx, old)
    before = (idx / POINTER_NAME).read_bytes()

    def _locked(*_args, **_kwargs):
        raise PermissionError("simulated pointer held by Obsidian")

    monkeypatch.setattr(os, "replace", _locked)
    with pytest.raises(RuntimeError, match="ACTIVE_INDEX 发布失败"):
        publish_pointer(idx, _fake_build(idx, "build_b"))
    assert (idx / POINTER_NAME).read_bytes() == before, "旧指针必须原样保留"
    assert not (idx / ".ACTIVE_INDEX.tmp").exists(), "失败后应清理临时指针"
    assert resolve_active_lance_dir(idx) == old / "lance_db"


def test_fallback_skips_unvalidated_and_failed_builds(tmp_path):
    idx = _index(tmp_path)
    good = _fake_build(idx, "build_good")
    bad = idx / "builds" / "build_bad"  # 无 manifest → 未 validated
    (bad / "lance_db").mkdir(parents=True)
    failed = idx / "builds" / "build_failed"
    (failed / "lance_db").mkdir(parents=True)
    (failed / "manifest.json").write_text("{}", encoding="utf-8")
    (failed / ".failed").write_text("boom", encoding="utf-8")
    (idx / POINTER_NAME).write_bytes(b"torn")
    assert resolve_active_lance_dir(idx) == good / "lance_db"

    # 全部候选无效 → 才回退 legacy 顶层布局
    idx2 = _index(tmp_path / "idx2")
    legacy = idx2 / "lance_db"
    legacy.mkdir()
    (idx2 / POINTER_NAME).write_bytes(b"torn")
    assert resolve_active_lance_dir(idx2) == legacy
