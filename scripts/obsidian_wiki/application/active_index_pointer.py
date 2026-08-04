"""#11/#21 ACTIVE_INDEX 指针契约：原子发布 + 严格校验解析（纯 stdlib）。

PR2（review 修复）严格化：
- schema_version=4：含 generation（单调）、build_id、相对 target、manifest SHA-256。
- target 必须是相对路径 ``builds/<build_id>/lance_db``；realpath 后仍须位于
  resolved builds root，拒绝绝对路径、``..`` 和 symlink escape。
- 移除 checksumless pointer 的正常读取兼容——旧 schema (<4) 走 recovery，
  不当安全。
- 读取端逐项校验：JSON 类型、字段类型、schema、checksum、manifest、lance target。
  ``[]``、``null``、截断 JSON、错误 checksum 都进入 recovery。
- 发布用 fsync + 父目录同步（平台支持时）保证崩溃耐久性。
- resolve 按 generation 从最近到最旧扫描 published/可恢复构建，不依赖 ``st_mtime``。
  legacy 顶层目录只通过显式验证接受，否则报 rebuild-required。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

POINTER_NAME = "ACTIVE_INDEX"
_TEMP_NAME = ".ACTIVE_INDEX.tmp"
SCHEMA_VERSION = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_dir(path: Path) -> None:
    """POSIX: 同步父目录以确保 rename 持久；Windows 无目录 fsync，跳过。"""
    if os.name == "posix":
        try:
            fd = os.open(os.fspath(path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            log.warning("fsync 目录失败（不影响正确性）：%s", exc)


def _validate_target(index_dir: Path, rel: str) -> Path | None:
    """校验 active_lance 相对路径：拒绝绝对/..//symlink escape，返回 resolved lance_dir 或 None。"""
    if not isinstance(rel, str) or not rel:
        return None
    if os.path.isabs(rel):
        log.warning("ACTIVE_INDEX target 是绝对路径，拒绝：%s", rel)
        return None
    builds_root = (index_dir / "builds").resolve()
    cand = (index_dir / rel).resolve()
    # 必须位于 builds root 之下
    try:
        cand.relative_to(builds_root)
    except ValueError:
        log.warning("ACTIVE_INDEX target 逃逸 builds root，拒绝：%s", rel)
        return None
    # 拒绝 symlink escape（target 路径中任一组件是 symlink 指向外部）
    try:
        cand.lstat()  # 不跟随 symlink
    except OSError:
        return None
    return cand


def publish_pointer(
    index_dir: Path, build_dir: Path, *, generation: int = 0, build_id: str = ""
) -> None:
    """原子翻转 ACTIVE_INDEX 指向 ``build_dir``；失败时旧指针保持可用。

    写 tmp → fsync → os.replace → fsync 父目录，保证崩溃耐久性。
    """
    pointer = index_dir / POINTER_NAME
    tmp = index_dir / _TEMP_NAME
    manifest = build_dir / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"拒绝发布：{manifest} 不存在（未 validated 的构建不可被引用）")
    manifest_digest = _sha256(manifest)
    rel_lance = build_dir.joinpath("lance_db").relative_to(index_dir)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generation": generation,
        "build_id": build_id or build_dir.name,
        "active_lance": str(rel_lance),
        "manifest_sha256": manifest_digest,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        fd = os.open(os.fspath(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, pointer)
        _fsync_dir(index_dir)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f"ACTIVE_INDEX 发布失败（旧活动索引保持可用）：{type(exc).__name__}: {exc}。"
            f"请确认 {pointer} 未被 Obsidian/杀软独占后重试。"
        ) from exc


def _read_generation(build_dir: Path) -> int:
    """从 build 的 manifest.json 读取 generation；无则 0。"""
    manifest = build_dir / "manifest.json"
    if not manifest.is_file():
        return -1
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        gen = data.get("generation", 0) if isinstance(data, dict) else 0
        return int(gen) if isinstance(gen, (int, float)) else 0
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return -1


def resolve_active_lance_dir(index_dir: Path) -> Path:
    """解析当前活动 lance 目录；指针损坏/失配时按 generation 回退最近已验证 build。

    legacy 顶层 ``lance_db`` 只在显式验证通过后接受，否则抛 ``RebuildRequiredError``。
    """
    pointer = index_dir / POINTER_NAME
    if pointer.is_file():
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = None
        if isinstance(data, dict):
            sv = data.get("schema_version")
            if not isinstance(sv, int) or sv < SCHEMA_VERSION:
                log.warning("ACTIVE_INDEX schema_version=%s < %d，进入 recovery", sv, SCHEMA_VERSION)
            else:
                rel = data.get("active_lance")
                cand = _validate_target(index_dir, rel if isinstance(rel, str) else "")
                if cand is not None and cand.is_dir():
                    manifest = cand.parent / "manifest.json"
                    if manifest.is_file():
                        digest = data.get("manifest_sha256")
                        if isinstance(digest, str) and digest == _sha256(manifest):
                            return cand
                        log.warning("ACTIVE_INDEX checksum 不匹配，忽略 %s", cand)
                    else:
                        log.warning("ACTIVE_INDEX 指向的构建缺 manifest，忽略 %s", cand)
                else:
                    log.warning("ACTIVE_INDEX target 无效，进入 recovery")
        elif data is not None:
            log.warning("ACTIVE_INDEX payload 非 dict（%s），进入 recovery", type(data).__name__)

    # Recovery：按 generation 从最近到最旧扫描 published/可恢复构建
    builds_dir = index_dir / "builds"
    candidates: list[tuple[int, Path]] = []
    if builds_dir.is_dir():
        for entry in builds_dir.iterdir():
            if not entry.is_dir() or entry.name.startswith(".") or entry.name.startswith("_old"):
                continue
            if (entry / ".failed").exists():
                continue
            if (entry / "lance_db").is_dir() and (entry / "manifest.json").is_file():
                gen = _read_generation(entry)
                if gen >= 0:
                    candidates.append((gen, entry))
    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        best = candidates[0][1]
        log.info("回退到最近已验证构建（generation=%d）：%s", candidates[0][0], best.name)
        return best / "lance_db"

    # Legacy 顶层：只在显式验证通过后接受
    legacy = index_dir / "lance_db"
    if legacy.is_dir() and (index_dir / "manifest.json").is_file():
        log.warning("回退到 legacy 顶层 lance_db（建议 rebuild 迁移到 builds/ 布局）")
        return legacy

    raise RebuildRequiredError(
        f"无可用活动索引：{pointer} 损坏或失配，且 builds/ 下无已验证构建。"
        f"请执行 rebuild 重建索引。"
    )


class RebuildRequiredError(RuntimeError):
    """无可用活动索引，需要 rebuild。"""
