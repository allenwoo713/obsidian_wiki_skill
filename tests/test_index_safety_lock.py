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
    RebuildRequiredError,
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


def _fake_build(idx: Path, name: str, body: str = "ok", generation: int = 1) -> Path:
    """构造一份"已验证"构建产物：manifest.json（含 generation）+ lance_db 目录。"""
    build = idx / "builds" / name
    (build / "lance_db").mkdir(parents=True, exist_ok=True)
    (build / "manifest.json").write_text(
        json.dumps({"layout": "sparse_chunks+dense_chunks", "body": body, "generation": generation}),
        encoding="utf-8",
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
# 2) ACTIVE_INDEX 严格指针契约（PR2：generation/schema/path/durability）
# ---------------------------------------------------------------------------
def test_publish_includes_strict_schema(tmp_path):
    """publish 写入 schema_version=4 + generation + build_id + manifest_sha256 + 相对 target。"""
    idx = _index(tmp_path)
    build = _fake_build(idx, "build_a", generation=1)
    publish_pointer(idx, build, generation=1, build_id="build_a")
    data = _pointer_data(idx)
    assert data["schema_version"] == 4
    assert data["generation"] == 1
    assert data["build_id"] == "build_a"
    assert Path(data["active_lance"]) == Path("builds/build_a/lance_db")
    assert data["manifest_sha256"]
    assert resolve_active_lance_dir(idx) == build / "lance_db"


def test_checksumless_pointer_rejected_via_recovery(tmp_path):
    """旧 schema (<4) 指针不直接接受，走 recovery 回退到已验证 build。"""
    idx = _index(tmp_path)
    build = _fake_build(idx, "build_a", generation=1)
    (idx / POINTER_NAME).write_text(
        json.dumps({"active_lance": "builds/build_a/lance_db", "schema_version": 2}),
        encoding="utf-8",
    )
    # 旧 schema → recovery → 找到 build_a（generation=1）
    assert resolve_active_lance_dir(idx) == build / "lance_db"


def test_torn_pointer_falls_back_by_generation(tmp_path):
    """截断指针 → recovery 按 generation 回退到最高代已验证 build。"""
    idx = _index(tmp_path)
    _fake_build(idx, "build_old", generation=1)
    newest = _fake_build(idx, "build_new", generation=2)
    (idx / POINTER_NAME).write_bytes(b"{torn-json")
    assert resolve_active_lance_dir(idx) == newest / "lance_db"


def test_checksum_mismatch_falls_back_to_previous_gen(tmp_path):
    """checksum 不匹配 → recovery 回退到上一代已验证 build。"""
    idx = _index(tmp_path)
    old = _fake_build(idx, "build_old", generation=1)
    new = _fake_build(idx, "build_new", generation=2)
    publish_pointer(idx, old, generation=1, build_id="build_old")
    publish_pointer(idx, new, generation=2, build_id="build_new")
    (new / "manifest.json").write_text(json.dumps({"body": "tampered"}), encoding="utf-8")
    # 指针指向 new 但 checksum 失配 → recovery 按 generation 扫描
    # new 的 manifest 被篡改（无 generation → gen=0），old gen=1 → 回退 old
    assert resolve_active_lance_dir(idx) == old / "lance_db"


def test_absolute_path_target_rejected(tmp_path):
    """active_lance 为绝对路径 → 拒绝，走 recovery。"""
    idx = _index(tmp_path)
    build = _fake_build(idx, "build_a", generation=1)
    (idx / POINTER_NAME).write_text(
        json.dumps({"schema_version": 4, "generation": 1, "build_id": "x",
                    "active_lance": str(idx / "builds" / "build_a" / "lance_db"),
                    "manifest_sha256": "x", "published_at": "x"}),
        encoding="utf-8",
    )
    # 绝对路径 → 拒绝 → recovery → build_a
    assert resolve_active_lance_dir(idx) == build / "lance_db"


def test_traversal_target_rejected(tmp_path):
    """active_lance 含 .. → 拒绝，走 recovery。"""
    idx = _index(tmp_path)
    build = _fake_build(idx, "build_a", generation=1)
    (idx / POINTER_NAME).write_text(
        json.dumps({"schema_version": 4, "generation": 1, "build_id": "x",
                    "active_lance": "builds/../../etc/lance_db",
                    "manifest_sha256": "x", "published_at": "x"}),
        encoding="utf-8",
    )
    assert resolve_active_lance_dir(idx) == build / "lance_db"


def test_null_and_list_pointer_rejected(tmp_path):
    """null / list JSON payload → recovery，不 AttributeError。"""
    idx = _index(tmp_path)
    build = _fake_build(idx, "build_a", generation=1)
    for bad in (b"null", b"[]", b"42"):
        (idx / POINTER_NAME).write_bytes(bad)
        assert resolve_active_lance_dir(idx) == build / "lance_db"


def test_no_valid_builds_raises_rebuild_required(tmp_path):
    """无可用 build 且无 legacy → RebuildRequiredError。"""
    idx = _index(tmp_path)
    (idx / POINTER_NAME).write_bytes(b"torn")
    with pytest.raises(RebuildRequiredError):
        resolve_active_lance_dir(idx)


def test_publish_failure_keeps_old_pointer(tmp_path, monkeypatch):
    """publish 失败时旧指针逐字节保留。"""
    idx = _index(tmp_path)
    old = _fake_build(idx, "build_a", generation=1)
    publish_pointer(idx, old, generation=1, build_id="build_a")
    before = (idx / POINTER_NAME).read_bytes()

    def _locked(*_args, **_kwargs):
        raise PermissionError("simulated pointer held by Obsidian")

    monkeypatch.setattr(os, "replace", _locked)
    with pytest.raises(RuntimeError, match="ACTIVE_INDEX 发布失败"):
        publish_pointer(idx, _fake_build(idx, "build_b", generation=2), generation=2, build_id="build_b")
    assert (idx / POINTER_NAME).read_bytes() == before, "旧指针必须原样保留"
    assert not (idx / ".ACTIVE_INDEX.tmp").exists(), "失败后应清理临时指针"
    assert resolve_active_lance_dir(idx) == old / "lance_db"


def test_generation_based_fallback_not_mtime(tmp_path):
    """recovery 按 generation（非 st_mtime）选择最高代 build。"""
    idx = _index(tmp_path)
    old_gen2 = _fake_build(idx, "build_high_gen", generation=2)
    time.sleep(0.05)
    new_gen1 = _fake_build(idx, "build_low_gen", generation=1)
    # new_gen1 的 mtime 更新但 generation 更低 → 应选 old_gen2
    (idx / POINTER_NAME).write_bytes(b"torn")
    assert resolve_active_lance_dir(idx) == old_gen2 / "lance_db"
