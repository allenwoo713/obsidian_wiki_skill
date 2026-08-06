"""#21/#34/#35/#36/#37 Index Safety 回归（模型无关，Windows + Linux CI 执行）。

- #34：BuildContext 身份统一（完整 UUID 贯穿 lock/build_dir/manifest/pointer）；
  BUILD.lock 稳定 pathname（release 不 rename/unlink）；三进程交错、终止重取、
  1000 并发 ID 唯一。
- #35：严格 schema v4 校验（sv==4 / generation 正整数 / build_id 格式 / published_at
  / manifest_sha256 / 精确三段 target）；generation 生命周期 validated→published→
  superseded 落盘；recovery 只从 published 回退；staging/篡改/legacy 语义。
- #36：耐久发布（pointer 写入失败传播，旧指针保留）。
- #37：确定性文件 barrier（无时序 sleep 作为正确性前提）；无重复测试定义。

纯 stdlib + 文件系统（端到端 service 测试用真实 LanceDB，CI architecture job 已装）。
"""
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from obsidian_wiki.application.active_index_pointer import (  # noqa: E402
    POINTER_NAME,
    RebuildRequiredError,
    publish_pointer,
    read_generation_record,
    resolve_active_lance_dir,
)
from obsidian_wiki.application.build_lock import (  # noqa: E402
    BuildLock,
    BuildLockHeldError,
    new_build_context,
)
from obsidian_wiki.domain.index_models import BuildContext  # noqa: E402
from obsidian_wiki.domain.index_publication_models import (  # noqa: E402
    GenerationRecord,
    GenerationState,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_NAME = "BUILD.lock"
_BUILD_ID_RE = re.compile(r"^build_\d{8}T\d{12}_[0-9a-f]{32}$")


def _index(tmp_path: Path) -> Path:
    idx = tmp_path / ".index"
    idx.mkdir(parents=True)
    return idx


def _valid_build_id(tag: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    suffix = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:32] if tag else uuid.uuid4().hex
    return f"build_{ts}_{suffix}"


def _ctx(build_id: str | None = None) -> BuildContext:
    c = new_build_context()
    return BuildContext(build_id=build_id or c.build_id, started_at=c.started_at, owner_nonce=c.owner_nonce)


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_build(idx: Path, build_id: str, body: str = "ok", generation: int = 1,
                state: str = "published") -> Path:
    """构造 build 产物：manifest.json（含 build_id/generation）+ lance_db + .generation.json。

    #35：default state=published 供 recovery 类测试；发布类测试须传 state='validated'，
    因为 publish_pointer 只接受存在完全匹配 VALIDATED record 的 build。
    """
    build = idx / "builds" / build_id
    (build / "lance_db").mkdir(parents=True, exist_ok=True)
    (build / "manifest.json").write_text(
        json.dumps({"layout": "sparse_chunks+dense_chunks", "body": body,
                    "generation": generation, "build_id": build_id}),
        encoding="utf-8",
    )
    if state:
        record = GenerationRecord(
            generation=generation, build_id=build_id,
            state=GenerationState(state), manifest_sha256=_sha256_of(build / "manifest.json"),
            validated_at="2026-08-06T00:00:00+00:00",
            published_at="2026-08-06T00:00:00+00:00" if state == "published" else None,
            superseded_at="2026-08-06T00:00:00+00:00" if state == "superseded" else None,
        )
        (build / ".generation.json").write_text(
            json.dumps(record.to_json(), sort_keys=True), encoding="utf-8")
    return build


def _pointer_data(idx: Path) -> dict:
    return json.loads((idx / POINTER_NAME).read_text(encoding="utf-8"))


def _write_pointer(idx: Path, *, sha: str = "",
                   published_at: str = "2026-08-06T00:00:00+00:00", **fields) -> None:
    bid = fields["build_id"]
    payload = {
        "schema_version": 4,
        "generation": 1,
        "build_id": bid,
        "active_lance": f"builds/{bid}/lance_db",
        "manifest_sha256": sha or _sha256_of(idx / "builds" / bid / "manifest.json"),
        "published_at": published_at,
    }
    payload.update(fields)
    (idx / POINTER_NAME).write_text(json.dumps(payload), encoding="utf-8")


def _wait_file(path: Path, timeout: float = 25.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"timeout waiting {path}")


def _wait_glob_count(directory: Path, pattern: str, count: int, timeout: float = 25.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(list(directory.glob(pattern))) >= count:
            return
        time.sleep(0.02)
    raise AssertionError(
        f"timeout waiting for {count} files matching {directory / pattern}; "
        f"found {sorted(path.name for path in directory.glob(pattern))}"
    )


# ---------------------------------------------------------------------------
# 1) #34 owner-scoped 单写者构建锁（BUILD.lock 稳定 pathname）
# ---------------------------------------------------------------------------
def test_lock_acquire_release_keeps_stable_lockfile(tmp_path):
    """acquire 写完整 metadata；release 只释放 descriptor，BUILD.lock 保持存在（稳定 inode）。"""
    idx = _index(tmp_path)
    lock = BuildLock(idx, ctx=_ctx("build_x"))
    lock.acquire()
    data = json.loads((idx / LOCK_NAME).read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    assert data["hostname"] == socket.gethostname()
    assert data["started_at"]
    assert data["build_id"] == "build_x"
    assert data["owner_nonce"]
    lock.release()
    assert (idx / LOCK_NAME).exists(), "#34：BUILD.lock 是稳定 pathname，release 不删除"


def test_lock_reentrant_same_thread(tmp_path):
    """同线程嵌套 acquire（facade + service 双层）可重入；内层 release 不释放外层。"""
    idx = _index(tmp_path)
    outer = BuildLock(idx, ctx=_ctx("outer"))
    outer.acquire()
    inner = BuildLock(idx, ctx=_ctx("inner"))
    inner.acquire()
    assert (idx / LOCK_NAME).exists()
    inner.release()
    assert (idx / LOCK_NAME).exists(), "内层 release 后外层仍持有"
    outer.release()
    assert (idx / LOCK_NAME).exists(), "#34：release 后 canonical path 仍在（稳定 pathname）"


def test_lock_blocks_different_thread(tmp_path):
    """同进程不同线程（非嵌套）不得绕过文件锁——review 核心要求。"""
    idx = _index(tmp_path)
    outer = BuildLock(idx, ctx=_ctx())
    outer.acquire()
    errors: list[str] = []

    def _try_inner():
        try:
            BuildLock(idx, ctx=_ctx()).acquire(wait=False)
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
        lock = BuildLock(idx, ctx=_ctx())
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
    release = tmp_path / "release"
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    scripts_dir = REPO_ROOT / "scripts"
    code = (
        "import sys, time, os, threading, socket, uuid\n"
        f"sys.path.insert(0, {json.dumps(str(scripts_dir))})\n"
        "from pathlib import Path\n"
        "from obsidian_wiki.application.build_lock import BuildLock, BuildLockHeldError\n"
        "from obsidian_wiki.domain.index_models import BuildContext\n"
        f"idx = Path({json.dumps(str(idx))})\n"
        f"go = Path({json.dumps(str(go))})\n"
        f"release = Path({json.dumps(str(release))})\n"
        f"ready = Path({json.dumps(str(ready_dir))})\n"
        f"results = Path({json.dumps(str(results_dir))})\n"
        "pid = os.getpid()\n"
        "(ready / f'ready_{pid}').write_text('ready', encoding='utf-8')\n"
        "for _ in range(200):\n"
        "    if go.exists(): break\n"
        "    time.sleep(0.05)\n"
        "ctx = BuildContext(build_id='proc', started_at='t', owner_nonce=f'{socket.gethostname()}:{pid}:{threading.get_ident()}:{uuid.uuid4().hex}')\n"
        "try:\n"
        "    lock = BuildLock(idx, ctx=ctx)\n"
        "    lock.acquire(wait=False)\n"
        "    (results / f'acquired_{pid}').write_text('ok', encoding='utf-8')\n"
        "    for _ in range(400):\n"
        "        if release.exists(): break\n"
        "        time.sleep(0.05)\n"
        "    lock.release()\n"
        "except BuildLockHeldError:\n"
        "    (results / f'blocked_{pid}').write_text('blocked', encoding='utf-8')\n"
    )
    p1 = subprocess.Popen([sys.executable, "-c", code], cwd=REPO_ROOT)
    p2 = subprocess.Popen([sys.executable, "-c", code], cwd=REPO_ROOT)
    try:
        _wait_glob_count(ready_dir, "ready_*", 2)
        go.write_text("go", encoding="utf-8")
        _wait_glob_count(results_dir, "*", 2)
        release.write_text("release", encoding="utf-8")
        p1.wait(timeout=20)
        p2.wait(timeout=20)
    finally:
        for p in (p1, p2):
            if p.poll() is None:
                p.kill()
                p.wait(timeout=5)
    assert p1.returncode == 0, f"p1 exit code={p1.returncode}"
    assert p2.returncode == 0, f"p2 exit code={p2.returncode}"
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
        "import sys, time, threading, socket, uuid, os\n"
        f"sys.path.insert(0, {json.dumps(str(scripts_dir))})\n"
        "from pathlib import Path\n"
        "from obsidian_wiki.application.build_lock import BuildLock\n"
        "from obsidian_wiki.domain.index_models import BuildContext\n"
        f"lock = BuildLock(Path({json.dumps(str(idx))}), ctx=BuildContext(build_id='holder', started_at='t', owner_nonce=f'{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}'))\n"
        "lock.acquire()\n"
        f"Path({json.dumps(str(ready))}).write_text('ready', encoding='utf-8')\n"
        "time.sleep(15)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=REPO_ROOT)
    try:
        _wait_file(ready)
        (idx / LOCK_NAME).write_bytes(b"corrupt-metadata")
        with pytest.raises(BuildLockHeldError):
            BuildLock(idx, ctx=_ctx()).acquire(wait=False)
    finally:
        proc.kill()
        proc.wait(timeout=10)
    assert proc.returncode is not None and proc.returncode != 0


def test_foreign_host_metadata_reclaimable_when_os_unlocked(tmp_path):
    """foreign-host metadata 但 OS lock 未持有 → 可获取（不因 PID 不存在而抢占）。"""
    idx = _index(tmp_path)
    (idx / LOCK_NAME).write_text(
        json.dumps({"pid": 2 ** 31 - 1, "hostname": "other-host",
                    "started_at": "x", "build_id": "foreign"}),
        encoding="utf-8",
    )
    lock = BuildLock(idx, ctx=_ctx("mine"))
    lock.acquire()  # 不抛异常
    data = json.loads((idx / LOCK_NAME).read_text(encoding="utf-8"))
    assert data["build_id"] == "mine"
    assert data["hostname"] == socket.gethostname()
    lock.release()


def test_three_process_release_race_single_writer(tmp_path):
    """#34：三进程交错——A 释放后 B 持有期间 C 必须无法取得锁（稳定 pathname）。

    使用双端 barrier（各角色先写 ready，父进程等所有 ready 后才发 continue；
    无时序 sleep 作为正确性前提）。逐一断言子进程 numeric exit code；
    B 释放后 D 必须能取得同一 stable lockfile。
    """
    idx = _index(tmp_path)
    scripts_dir = REPO_ROOT / "scripts"
    stage = tmp_path / "stage"
    stage.mkdir()
    child = (
        "import sys, time, threading, socket, uuid, os\n"
        f"sys.path.insert(0, {json.dumps(str(scripts_dir))})\n"
        "from pathlib import Path\n"
        "from obsidian_wiki.application.build_lock import BuildLock, BuildLockHeldError\n"
        "from obsidian_wiki.domain.index_models import BuildContext\n"
        f"idx = Path({json.dumps(str(idx))})\n"
        f"stage = Path({json.dumps(str(stage))})\n"
        "def mark(name): (stage / name).write_text('1', encoding='utf-8')\n"
        "def wait(name, timeout=30):\n"
        "    deadline = time.monotonic() + timeout\n"
        "    while time.monotonic() < deadline:\n"
        "        if (stage / name).exists(): return\n"
        "        time.sleep(0.02)\n"
        "    raise RuntimeError('timeout waiting ' + name)\n"
        "def ctx(tag):\n"
        "    return BuildContext(build_id=tag, started_at='t', owner_nonce=f'{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}')\n"
        "role = sys.argv[1]\n"
        "mark(role + '_ready')\n"
        "wait('continue')\n"
        "if role == 'A':\n"
        "    lock = BuildLock(idx, ctx=ctx('build_A'))\n"
        "    lock.acquire()\n"
        "    mark('A_held')\n"
        "    wait('a_go')\n"
        "    lock.release()\n"
        "    mark('A_released')\n"
        "elif role == 'B':\n"
        "    wait('a_go')\n"
        "    lock = BuildLock(idx, ctx=ctx('build_B'))\n"
        "    lock.acquire(wait=True, timeout=30)\n"
        "    mark('B_held')\n"
        "    wait('b_go')\n"
        "    lock.release()\n"
        "    mark('B_released')\n"
        "elif role == 'C':\n"
        "    wait('B_held')       # B 明确持有后才允许 C 尝试（双端 barrier）\n"
        "    try:\n"
        "        lock = BuildLock(idx, ctx=ctx('build_C'))\n"
        "        lock.acquire(wait=False)\n"
        "        mark('C_acquired')\n"
        "    except BuildLockHeldError:\n"
        "        mark('C_blocked')\n"
        "else:\n"
        "    wait('B_released')\n"
        "    lock = BuildLock(idx, ctx=ctx('build_D'))\n"
        "    lock.acquire()\n"
        "    mark('D_acquired')\n"
        "    lock.release()\n"
        "    mark('D_released')\n"
    )
    procs = {
        role: subprocess.Popen([sys.executable, "-c", child, role], cwd=REPO_ROOT)
        for role in ("A", "B", "C", "D")
    }
    try:
        # 双端 barrier：等全部角色 ready 才 continue（无 sleep 代表 ready）
        for role in ("A", "B", "C", "D"):
            _wait_file(stage / f"{role}_ready")
        (stage / "continue").write_text("1", encoding="utf-8")
        _wait_file(stage / "A_held")
        (stage / "a_go").write_text("1", encoding="utf-8")   # B 开始 acquire（等待 A 释放）
        _wait_file(stage / "B_held")                          # B 已确认持有
        # C 等 B_held 自动尝试；必须等 C 尝试完成（blocked/acquired）后才放行 B 释放，
        # 保证 C 的 acquire 发生在 B 持有期间（非并发竞速）。
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if (stage / "C_blocked").exists() or (stage / "C_acquired").exists():
                break
            time.sleep(0.02)
        (stage / "b_go").write_text("1", encoding="utf-8")   # 现在才让 B 释放
        for role in ("A", "B", "C", "D"):
            procs[role].wait(timeout=40)
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
                p.wait(timeout=5)
    # 逐一断言 numeric exit code（全部 0 = 无超时/未捕获异常）
    for role, p in procs.items():
        assert p.returncode == 0, f"{role} exit code={p.returncode}"
    assert (stage / "A_released").exists()
    assert (stage / "B_held").exists()
    assert (stage / "B_released").exists()
    assert (stage / "C_blocked").exists(), "B 持有期间 C 必须无法取得锁"
    assert not (stage / "C_acquired").exists()
    assert (stage / "D_acquired").exists(), "B 释放后 D 必须可取得同一 stable lockfile"
    assert (stage / "D_released").exists()


def test_killed_holder_allows_reacquire_same_pathname(tmp_path):
    """#34：强制终止持锁进程后，新进程可取得同一稳定 pathname 的锁。"""
    idx = _index(tmp_path)
    scripts_dir = REPO_ROOT / "scripts"
    ready = tmp_path / "ready"
    code = (
        "import sys, time, threading, socket, uuid, os\n"
        f"sys.path.insert(0, {json.dumps(str(scripts_dir))})\n"
        "from pathlib import Path\n"
        "from obsidian_wiki.application.build_lock import BuildLock\n"
        "from obsidian_wiki.domain.index_models import BuildContext\n"
        f"idx = Path({json.dumps(str(idx))})\n"
        f"ready = Path({json.dumps(str(ready))})\n"
        "lock = BuildLock(idx, ctx=BuildContext(build_id='build_killed', started_at='t', owner_nonce=f'{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}'))\n"
        "lock.acquire()\n"
        "ready.write_text('ready', encoding='utf-8')\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=REPO_ROOT)
    try:
        _wait_file(ready)
        proc.kill()
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert proc.returncode is not None and proc.returncode != 0
    # OS lock 随进程终止自动释放；同一稳定 pathname 可重新获取
    lock = BuildLock(idx, ctx=_ctx("after-kill"))
    lock.acquire()
    assert (idx / LOCK_NAME).exists()
    lock.release()


def test_build_context_id_concurrent_uniqueness():
    """#34：并发生成 1000 个 build_id 全部唯一，且 UUID 后缀为完整 32 hex。"""
    ids: set[str] = set()
    barrier = threading.Barrier(10)

    def _gen():
        barrier.wait()
        for _ in range(100):
            ids.add(new_build_context().build_id)

    threads = [threading.Thread(target=_gen) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert len(ids) == 1000, f"1000 个并发 build_id 应全部唯一，got {len(ids)}"
    for build_id in ids:
        assert _BUILD_ID_RE.fullmatch(build_id), f"build_id 必须是 UTC 微秒 + 完整 UUID: {build_id}"


def test_build_ids_unique_when_clock_frozen(tmp_path, monkeypatch):
    """#34：UTC 时钟固定到同一微秒时，并发生成 1000 个 build_id 仍全部唯一（UUID 保证）。"""
    import obsidian_wiki.application.build_lock as bl

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=timezone.utc):
            return datetime(2026, 8, 6, 0, 0, 0, 123456, tzinfo=timezone.utc)

    monkeypatch.setattr(bl, "datetime", _FixedDatetime)
    ids: set[str] = set()
    barrier = threading.Barrier(10)

    def _gen():
        barrier.wait()
        for _ in range(100):
            ids.add(new_build_context().build_id)

    threads = [threading.Thread(target=_gen) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert len(ids) == 1000, f"时钟冻结时 1000 个 build_id 仍应全部唯一，got {len(ids)}"
    for build_id in ids:
        assert _BUILD_ID_RE.fullmatch(build_id), f"完整 UUID 后缀: {build_id}"


def test_build_context_flows_through_service_build(tmp_path):
    """#34 端到端：单次 build 的 build_id 贯穿 lock metadata / build 目录 / manifest /
    pointer / 生命周期记录与返回 artifact，且只生成一次。"""
    lancedb = pytest.importorskip("lancedb")  # CI architecture job 已安装
    from obsidian_wiki.application.index_build_service import IndexBuildService
    from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest
    from obsidian_wiki.infrastructure.filesystem_post_commit_journal import FilesystemPostCommitJournal
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    wiki = tmp_path / "Wiki"
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "concepts" / "a.md").write_text(
        "---\ntype: concept\ntitle: Acme\ntags: []\n---\n\nradar calibration stable token",
        encoding="utf-8",
    )
    index_dir = tmp_path / ".index"
    ctx = _ctx()
    embed = lambda texts: [[1.0] * 16 for _ in texts]  # noqa: E731
    artifact = IndexBuildService(
        LanceDbIndexRepository(index_dir),
        reopen_storage=LanceDbIndexRepository,
        manifest_store=FilesystemIndexManifest(),
        post_commit_journal=FilesystemPostCommitJournal(index_dir),
    ).build(wiki, index_dir, embed=embed, ctx=ctx)

    lock_data = json.loads((index_dir / LOCK_NAME).read_text(encoding="utf-8"))
    assert lock_data["build_id"] == ctx.build_id, "lock metadata build_id 必须与 ctx 一致"
    assert artifact.build_id == ctx.build_id, "artifact.build_id 必须与 ctx 一致"
    assert artifact.generation >= 1
    assert artifact.lance_dir.parent.name == ctx.build_id, "build 目录名必须与 ctx.build_id 一致"
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["build_id"] == ctx.build_id, "manifest build_id 必须与 ctx 一致"
    assert manifest["generation"] == artifact.generation
    ptr = _pointer_data(index_dir)
    assert ptr["build_id"] == ctx.build_id
    assert ptr["active_lance"] == f"builds/{ctx.build_id}/lance_db"
    record = read_generation_record(artifact.lance_dir.parent)
    assert record is not None and record.build_id == ctx.build_id
    assert record.state == GenerationState.PUBLISHED
    assert record.generation == manifest["generation"] == ptr["generation"]
    assert resolve_active_lance_dir(index_dir) == artifact.lance_dir


# ---------------------------------------------------------------------------
# 2) #35 ACTIVE_INDEX 严格 schema v4 + generation 生命周期
# ---------------------------------------------------------------------------
def test_publish_strict_schema_and_resolves(tmp_path):
    """publish 写入严格 schema v4，且可被 resolve 校验通过。"""
    idx = _index(tmp_path)
    build_id = _valid_build_id("a")
    build = _fake_build(idx, build_id, generation=1, state="validated")
    publish_pointer(idx, build, generation=1, build_id=build_id)
    data = _pointer_data(idx)
    assert data["schema_version"] == 4
    assert data["generation"] == 1
    assert data["build_id"] == build_id
    assert data["active_lance"] == f"builds/{build_id}/lance_db"
    assert re.fullmatch(r"[0-9a-f]{64}", data["manifest_sha256"])
    assert resolve_active_lance_dir(idx) == build / "lance_db"
    record = read_generation_record(build)
    assert record is not None and record.state == GenerationState.PUBLISHED


def test_publish_requires_validated_record(tmp_path):
    """#35：无完全匹配 VALIDATED record 的 build（含任意 manifest）不可直接 PUBLISHED。"""
    idx = _index(tmp_path)
    build_id = _valid_build_id("a")
    _fake_build(idx, build_id, generation=1, state="published")  # 直接标 published 绕过 validated
    with pytest.raises(RuntimeError, match="无 VALIDATED 生命周期记录"):
        publish_pointer(idx, idx / "builds" / build_id, generation=1, build_id=build_id)
    assert not (idx / POINTER_NAME).exists()


@pytest.mark.parametrize("overrides", [
    {"schema_version": 5},
    {"schema_version": True},
    {"schema_version": 3},
    {"generation": True},
    {"generation": "2"},
    {"generation": 0},
    {"generation": -1},
    {"build_id": "build_short"},
    {"build_id": "build_20260806T000000000000_abcd"},
    {"manifest_sha256": "not-hex"},
    {"manifest_sha256": "abc"},
    {"published_at": "not-a-date"},
    {"published_at": "2026-08-06T00:00:00+08:00"},  # 非 UTC（offset +08:00）拒绝
    {"published_at": "2026-08-06T00:00:00"},        # naive 时间拒绝
    {"active_lance": "builds/other/lance_db"},   # 与 build_id 不一致
    {"extra_field": "x"},                        # 额外字段拒绝（字段集合精确相等）
])
def test_pointer_schema_variants_rejected(tmp_path, overrides):
    """#35：任一 schema 字段非法 → 拒绝并走 recovery 回退到已验证 build。"""
    idx = _index(tmp_path)
    build_id = _valid_build_id("a")
    build = _fake_build(idx, build_id, generation=1)
    _write_pointer(idx, **{
        "build_id": build_id, "generation": 1,
        "sha": _sha256_of(build / "manifest.json"), **overrides,
    })
    assert resolve_active_lance_dir(idx) == build / "lance_db"


@pytest.mark.parametrize("rel", [
    "",
    "builds/../etc/lance_db",
    "builds/a/./lance_db",
    "builds/a/lance_db/extra",
    "builds//a/lance_db",
    "builds/a/lance_db/..",
    "/abs/builds/a/lance_db",
])
def test_pointer_target_variants_rejected(tmp_path, rel):
    """#35：active_lance 非精确三段（.. / 重复分隔符 / 额外组件 / 绝对路径）→ recovery。"""
    idx = _index(tmp_path)
    build_id = _valid_build_id("a")
    build = _fake_build(idx, build_id, generation=1)
    _write_pointer(idx, build_id=build_id, generation=1, active_lance=rel)
    assert resolve_active_lance_dir(idx) == build / "lance_db"


def test_pointer_absolute_target_rejected(tmp_path):
    idx = _index(tmp_path)
    build_id = _valid_build_id("a")
    build = _fake_build(idx, build_id, generation=1)
    _write_pointer(idx, build_id=build_id, generation=1, active_lance=str(build / "lance_db"))
    assert resolve_active_lance_dir(idx) == build / "lance_db"


def test_pointer_symlink_build_dir_rejected(tmp_path):
    """#35：builds/<build_id> 是 symlink → 拒绝（recovery 无其它候选 → rebuild-required）。"""
    idx = _index(tmp_path)
    build_id = _valid_build_id("a")
    real = tmp_path / "real"
    real.mkdir()
    (real / "lance_db").mkdir()
    (real / "manifest.json").write_text("{}", encoding="utf-8")
    try:
        (idx / "builds" / build_id).symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("平台不支持 symlink")
    _write_pointer(idx, build_id=build_id, generation=1, sha=_sha256_of(real / "manifest.json"))
    with pytest.raises(RebuildRequiredError):
        resolve_active_lance_dir(idx)


def test_checksumless_pointer_rejected_via_recovery(tmp_path):
    """旧 schema (<4) 指针不直接接受，走 recovery 回退到已验证 build。"""
    idx = _index(tmp_path)
    build_id = _valid_build_id("a")
    build = _fake_build(idx, build_id, generation=1)
    (idx / POINTER_NAME).write_text(
        json.dumps({"active_lance": f"builds/{build_id}/lance_db", "schema_version": 2}),
        encoding="utf-8",
    )
    assert resolve_active_lance_dir(idx) == build / "lance_db"


def test_torn_pointer_falls_back_by_generation(tmp_path):
    """截断指针 → recovery 按 generation 回退到最高代已验证 published build。"""
    idx = _index(tmp_path)
    _fake_build(idx, _valid_build_id("old"), generation=1)
    newest = _fake_build(idx, _valid_build_id("new"), generation=2)
    (idx / POINTER_NAME).write_bytes(b"{torn-json")
    assert resolve_active_lance_dir(idx) == newest / "lance_db"


def test_manifest_tamper_skips_current_falls_back(tmp_path):
    """#35：当前 generation 的 manifest 被篡改 → 跳过，回退上一份已验证 published。"""
    idx = _index(tmp_path)
    old_id = _valid_build_id("old")
    new_id = _valid_build_id("new")
    old = _fake_build(idx, old_id, generation=1, state="validated")
    new = _fake_build(idx, new_id, generation=2, state="validated")
    publish_pointer(idx, old, generation=1, build_id=old_id)
    publish_pointer(idx, new, generation=2, build_id=new_id)
    (new / "manifest.json").write_text(json.dumps({"body": "tampered"}), encoding="utf-8")
    # 指针指向 new 但 checksum 与生命周期记录均失配 → recovery 跳过 new，回退 old
    assert resolve_active_lance_dir(idx) == old / "lance_db"


def test_staging_validated_not_selected_by_recovery(tmp_path):
    """#35：manifest 已写 + validated 记录但 pointer 未发布（模拟中断）→ 绝不被选中。"""
    idx = _index(tmp_path)
    good_id = _valid_build_id("good")
    good = _fake_build(idx, good_id, generation=1, state="validated")
    publish_pointer(idx, good, generation=1, build_id=good_id)
    staging = _fake_build(idx, _valid_build_id("staging"), generation=2, state="validated")
    (idx / POINTER_NAME).write_bytes(b"torn")
    assert resolve_active_lance_dir(idx) == good / "lance_db"
    record = read_generation_record(staging)
    assert record is not None and record.state == GenerationState.VALIDATED


def test_lifecycle_supersedes_previous_published(tmp_path):
    """#35：新 generation published 后，旧 generation 标 superseded（时序持久化）。"""
    idx = _index(tmp_path)
    id1 = _valid_build_id("g1")
    id2 = _valid_build_id("g2")
    build1 = _fake_build(idx, id1, generation=1, state="validated")
    publish_pointer(idx, build1, generation=1, build_id=id1)
    assert read_generation_record(build1).state == GenerationState.PUBLISHED
    build2 = _fake_build(idx, id2, generation=2, state="validated")
    publish_pointer(idx, build2, generation=2, build_id=id2)
    rec1 = read_generation_record(build1)
    rec2 = read_generation_record(build2)
    assert rec1.state == GenerationState.SUPERSEDED, "旧代发布后被标 superseded"
    assert rec1.superseded_at
    assert rec2.state == GenerationState.PUBLISHED


def test_generation_based_fallback_not_mtime(tmp_path):
    """recovery 按 generation（非 st_mtime）选择最高代 build。"""
    idx = _index(tmp_path)
    high = _fake_build(idx, _valid_build_id("high"), generation=2)
    time.sleep(0.05)
    _fake_build(idx, _valid_build_id("low"), generation=1)
    (idx / POINTER_NAME).write_bytes(b"torn")
    assert resolve_active_lance_dir(idx) == high / "lance_db"


def test_legacy_rejected_requires_rebuild(tmp_path):
    """#35：删除普通 legacy fallback——无法验证的 legacy 一律 RebuildRequiredError。"""
    idx = _index(tmp_path)
    legacy = idx / "lance_db"
    legacy.mkdir(parents=True)
    (idx / "manifest.json").write_text("{}", encoding="utf-8")
    (idx / POINTER_NAME).write_bytes(b"torn")
    with pytest.raises(RebuildRequiredError):
        resolve_active_lance_dir(idx)


def test_legacy_unverifiable_raises_rebuild_required(tmp_path):
    """#35：无 legacy 且无 published build → RebuildRequiredError（不静默回退）。"""
    idx = _index(tmp_path)
    (idx / POINTER_NAME).write_bytes(b"torn")
    with pytest.raises(RebuildRequiredError):
        resolve_active_lance_dir(idx)


def test_no_valid_builds_raises_rebuild_required(tmp_path):
    idx = _index(tmp_path)
    (idx / POINTER_NAME).write_bytes(b"torn")
    _fake_build(idx, _valid_build_id("solo"), generation=1, state="validated")
    with pytest.raises(RebuildRequiredError):
        resolve_active_lance_dir(idx)


def test_publish_failure_keeps_old_pointer(tmp_path, monkeypatch):
    """#36：pointer 写入失败 → RuntimeError + 旧指针逐字节保留；staging 不被提升。"""
    import obsidian_wiki.application.active_index_pointer as aip

    idx = _index(tmp_path)
    old_id = _valid_build_id("old")
    old = _fake_build(idx, old_id, generation=1, state="validated")
    publish_pointer(idx, old, generation=1, build_id=old_id)
    before = (idx / POINTER_NAME).read_bytes()
    new_id = _valid_build_id("new")
    _fake_build(idx, new_id, generation=2, state="validated")

    def _boom(*_a, **_k):
        raise OSError("simulated durable write failure")

    monkeypatch.setattr(aip, "atomic_write_bytes", _boom)
    with pytest.raises(RuntimeError, match="ACTIVE_INDEX 发布失败"):
        publish_pointer(idx, idx / "builds" / new_id, generation=2, build_id=new_id)
    assert (idx / POINTER_NAME).read_bytes() == before, "旧指针必须原样保留"
    assert resolve_active_lance_dir(idx) == old / "lance_db"
