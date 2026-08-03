"""#21 Index Safety：单写者构建锁 + ACTIVE_INDEX 指针契约（模型无关）。

覆盖验收点：
- 并发构建只有一个写者，另一方得到明确 BuildLockHeldError（跨进程子进程实测）；
- 锁元数据含 pid/hostname/started_at/build_id；stale（pid 已死）自动回收；
  存活进程的锁不被抢占；同进程双层获取可重入；
- 指针发布仅原子替换，失败保留旧指针且清理 tmp；
- 指针含 manifest checksum，读取端校验后才切换；损坏/失配/指向缺失时
  回退最近已验证 build，绝不盲目用 legacy 顶层目录。

纯 stdlib + 文件系统，可在 ci.yml architecture job（Windows + Linux）运行。
"""
import json
import os
import socket
import subprocess
import sys
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
# 1) 单写者构建锁
# ---------------------------------------------------------------------------
def test_lock_acquire_release_writes_metadata(tmp_path):
    idx = _index(tmp_path)
    lock = BuildLock(idx, build_id="build_x")
    lock.acquire()
    data = json.loads((idx / LOCK_NAME).read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    assert data["hostname"] == socket.gethostname()
    assert data["started_at"]
    assert data["build_id"] == "build_x"
    lock.release()
    assert not (idx / LOCK_NAME).exists()


def test_lock_reentrant_within_process(tmp_path):
    idx = _index(tmp_path)
    outer = BuildLock(idx)
    outer.acquire()
    inner = BuildLock(idx)  # facade + service 双层获取
    inner.acquire()
    assert (idx / LOCK_NAME).exists()
    outer.release()
    assert (idx / LOCK_NAME).exists(), "内层仍持有，锁不应被外层释放"
    inner.release()
    assert not (idx / LOCK_NAME).exists()


def test_lock_held_by_live_pid_is_rejected(tmp_path):
    idx = _index(tmp_path)
    (idx / LOCK_NAME).write_text(
        json.dumps({"pid": os.getpid(), "hostname": "h", "started_at": "x", "build_id": ""}),
        encoding="utf-8",
    )
    with pytest.raises(BuildLockHeldError):
        BuildLock(idx).acquire()


def test_stale_lock_from_dead_pid_is_reclaimed(tmp_path):
    idx = _index(tmp_path)
    (idx / LOCK_NAME).write_text(
        json.dumps({"pid": 2 ** 31 - 1, "hostname": "h", "started_at": "x", "build_id": ""}),
        encoding="utf-8",
    )
    lock = BuildLock(idx)
    lock.acquire()
    data = json.loads((idx / LOCK_NAME).read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    lock.release()


def test_concurrent_process_excluded_by_lock(tmp_path):
    """跨进程实测：两个构建进程同时启动，后到者得到明确 BuildLockHeldError。"""
    idx = _index(tmp_path)
    ready = tmp_path / "ready"
    scripts_dir = REPO_ROOT / "scripts"
    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {json.dumps(str(scripts_dir))})\n"
        "from pathlib import Path\n"
        "from obsidian_wiki.application.build_lock import BuildLock\n"
        f"lock = BuildLock(Path({json.dumps(str(idx))}))\n"
        "lock.acquire()\n"
        f"Path({json.dumps(str(ready))}).write_text('ready', encoding='utf-8')\n"
        "time.sleep(30)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=REPO_ROOT)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if ready.exists():
                break
            time.sleep(0.1)
        else:
            raise AssertionError("子进程未能在 15s 内取得锁")
        with pytest.raises(BuildLockHeldError):
            BuildLock(idx).acquire()
    finally:
        proc.kill()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# 2) ACTIVE_INDEX 指针契约
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
