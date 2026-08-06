"""#11/#21/#35/#36 ACTIVE_INDEX 指针契约：严格 schema v4 + 生命周期 + 耐久发布。

#35 严格化（review 复审遗留）：
- schema_version 必须恰好为整数 4（拒绝 bool / 低版本 / 未知高版本）；
- generation 必须为非 bool 正整数；build_id 匹配完整 build-ID 格式；
- published_at 必须可解析为 UTC；manifest_sha256 必须是规范 SHA-256；
- active_lance 必须精确为 ``builds/<build_id>/lance_db`` 三段相对路径，
  拒绝绝对路径、``.``、``..``、额外组件与任意 symlink 组件。
- 每个 generation 持久化不可变验证记录（``.generation.json``）：
  building → validated → published → superseded；recovery 只从 published 回退；
  manifest 摘要与记录不符的 generation 被跳过。
- legacy 顶层目录只通过显式验证接受，否则 RebuildRequiredError。

#36 耐久：
- 所有关键写入经 durable_filesystem（fsync/replace 失败向上传播，不降级 warning）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from obsidian_wiki.domain.index_publication_models import GenerationRecord, GenerationState
from obsidian_wiki.infrastructure.durable_filesystem import atomic_write_bytes

log = logging.getLogger(__name__)

POINTER_NAME = "ACTIVE_INDEX"
GEN_RECORD_NAME = ".generation.json"
SCHEMA_VERSION = 4

_BUILD_ID_RE = re.compile(r"^build_\d{8}T\d{12}_[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_generation_record(
    build_dir: Path, *, generation: int, build_id: str, state: GenerationState,
    manifest_sha256: str, validated_at: str,
    published_at: str | None = None, superseded_at: str | None = None,
) -> None:
    record = GenerationRecord(
        generation=generation, build_id=build_id, state=state,
        manifest_sha256=manifest_sha256, validated_at=validated_at,
        published_at=published_at, superseded_at=superseded_at,
    )
    atomic_write_bytes(
        build_dir / GEN_RECORD_NAME,
        json.dumps(record.to_json(), sort_keys=True).encode("utf-8"),
    )


def read_generation_record(build_dir: Path) -> GenerationRecord | None:
    path = build_dir / GEN_RECORD_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        state = GenerationState(data["state"])
        return GenerationRecord(
            generation=int(data["generation"]), build_id=str(data["build_id"]),
            state=state, manifest_sha256=str(data["manifest_sha256"]),
            validated_at=str(data["validated_at"]),
            published_at=str(data["published_at"]) if data.get("published_at") else None,
            superseded_at=str(data["superseded_at"]) if data.get("superseded_at") else None,
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def record_validated(build_dir: Path, *, generation: int, build_id: str, manifest_sha256: str) -> None:
    """manifest 落盘并完整验证后写入 validated 记录（#35：staging 绝不被 recovery 选中）。"""
    _write_generation_record(
        build_dir, generation=generation, build_id=build_id,
        state=GenerationState.VALIDATED, manifest_sha256=manifest_sha256,
        validated_at=_now(),
    )


def publish_pointer(
    index_dir: Path, build_dir: Path, *, generation: int = 0, build_id: str = "",
) -> None:
    """原子翻转 ACTIVE_INDEX 指向 ``build_dir``；失败时旧指针保持可用。

    严格校验 generation/build_id/目录名一致性与 manifest 存在；经 durable 写入后
    把本 generation 标为 published，并把其它 published 标 superseded。
    """
    if not isinstance(generation, int) or isinstance(generation, bool) or generation <= 0:
        raise RuntimeError(f"拒绝发布：generation={generation!r} 必须为正整数")
    if not isinstance(build_id, str) or not _BUILD_ID_RE.match(build_id):
        raise RuntimeError(f"拒绝发布：build_id={build_id!r} 不符合 build_ID 格式")
    if build_dir.name != build_id:
        raise RuntimeError(f"拒绝发布：build_dir={build_dir.name} 与 build_id={build_id} 不一致")
    manifest = build_dir / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"拒绝发布：{manifest} 不存在（未 validated 的构建不可被引用）")
    manifest_digest = _sha256(manifest)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generation": generation,
        "build_id": build_id,
        "active_lance": f"builds/{build_id}/lance_db",
        "manifest_sha256": manifest_digest,
        "published_at": _now(),
    }
    pointer = index_dir / POINTER_NAME
    try:
        atomic_write_bytes(
            pointer,
            json.dumps(payload, sort_keys=True).encode("utf-8"),
        )
    except OSError as exc:
        raise RuntimeError(
            f"ACTIVE_INDEX 发布失败（旧活动索引保持可用）：{type(exc).__name__}: {exc}。"
            f"请确认 {pointer} 未被 Obsidian/杀软独占后重试。"
        ) from exc
    # 生命周期：validated → published；新 pointer 耐久落盘后才标旧代 superseded（#35）。
    previous = read_generation_record(build_dir)
    _write_generation_record(
        build_dir, generation=generation, build_id=build_id,
        state=GenerationState.PUBLISHED, manifest_sha256=manifest_digest,
        validated_at=previous.validated_at if previous else _now(),
        published_at=_now(),
    )
    _mark_superseded(index_dir, build_dir, generation)


def _mark_superseded(index_dir: Path, new_build_dir: Path, new_generation: int) -> None:
    builds_dir = index_dir / "builds"
    if not builds_dir.is_dir():
        return
    for entry in builds_dir.iterdir():
        if entry == new_build_dir or not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name.startswith("_old"):
            continue
        record = read_generation_record(entry)
        if record is None or record.state != GenerationState.PUBLISHED:
            continue
        _write_generation_record(
            entry, generation=record.generation, build_id=record.build_id,
            state=GenerationState.SUPERSEDED, manifest_sha256=record.manifest_sha256,
            validated_at=record.validated_at, published_at=record.published_at,
            superseded_at=_now(),
        )


def _validate_pointer(index_dir: Path, data: object) -> Path | None:
    """严格校验 schema v4 pointer（#35）；返回 resolved lance_dir 或 None（进 recovery）。"""
    if not isinstance(data, dict):
        log.warning("ACTIVE_INDEX payload 非 dict（%s），进入 recovery", type(data).__name__)
        return None
    sv = data.get("schema_version")
    if not isinstance(sv, int) or isinstance(sv, bool) or sv != SCHEMA_VERSION:
        log.warning("ACTIVE_INDEX schema_version=%r 必须为整数 %d，拒绝", sv, SCHEMA_VERSION)
        return None
    generation = data.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation <= 0:
        log.warning("ACTIVE_INDEX generation=%r 非法，拒绝", generation)
        return None
    build_id = data.get("build_id")
    if not isinstance(build_id, str) or not _BUILD_ID_RE.match(build_id):
        log.warning("ACTIVE_INDEX build_id=%r 非法，拒绝", build_id)
        return None
    published_at = data.get("published_at")
    if not isinstance(published_at, str):
        log.warning("ACTIVE_INDEX published_at 缺失/非字符串，拒绝")
        return None
    try:
        datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        log.warning("ACTIVE_INDEX published_at=%r 不可解析，拒绝", published_at)
        return None
    digest = data.get("manifest_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.match(digest):
        log.warning("ACTIVE_INDEX manifest_sha256=%r 非法，拒绝", digest)
        return None
    rel = data.get("active_lance")
    expected = f"builds/{build_id}/lance_db"
    if not isinstance(rel, str) or rel != expected:
        log.warning("ACTIVE_INDEX active_lance=%r != %r，拒绝", rel, expected)
        return None
    # 逐组件校验拒绝 symlink；resolve 后必须仍在 builds root 内
    probe = index_dir
    for part in Path(rel).parts:
        probe = probe / part
        if probe.is_symlink():
            log.warning("ACTIVE_INDEX target 含 symlink 组件，拒绝")
            return None
    cand = (index_dir / rel).resolve()
    try:
        cand.relative_to((index_dir / "builds").resolve())
    except ValueError:
        log.warning("ACTIVE_INDEX target 逃逸 builds root，拒绝")
        return None
    if not cand.is_dir():
        log.warning("ACTIVE_INDEX target 目录不存在，拒绝")
        return None
    manifest = cand.parent / "manifest.json"
    if not manifest.is_file():
        log.warning("ACTIVE_INDEX 指向的构建缺 manifest，拒绝")
        return None
    if digest != _sha256(manifest):
        log.warning("ACTIVE_INDEX checksum 不匹配，拒绝 %s", cand)
        return None
    record = read_generation_record(cand.parent)
    if record is None or record.state != GenerationState.PUBLISHED:
        log.warning("ACTIVE_INDEX 指向的 build 无 published 生命周期记录，拒绝")
        return None
    return cand


def resolve_active_lance_dir(index_dir: Path) -> Path:
    """解析当前活动 lance 目录；严格校验失败/损坏时按 generation 回退最近
    已验证 published build（#35：绝不放行未发布/篡改/仅文件存在的 build）。"""
    pointer = index_dir / POINTER_NAME
    if pointer.is_file():
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = None
        cand = _validate_pointer(index_dir, data) if data is not None else None
        if cand is not None:
            return cand

    # Recovery：只从「有可验证 published 记录」的 generation 回退——published 或
    # superseded 都曾耐久发布（记录含 published_at）；staging（validated）与
    # 仅文件存在的 build 一律不选；manifest 摘要与记录不符的 generation 被跳过。
    builds_dir = index_dir / "builds"
    candidates: list[tuple[int, Path]] = []
    if builds_dir.is_dir():
        for entry in builds_dir.iterdir():
            if not entry.is_dir() or entry.name.startswith(".") or entry.name.startswith("_old"):
                continue
            record = read_generation_record(entry)
            if record is None or record.state not in (
                GenerationState.PUBLISHED, GenerationState.SUPERSEDED,
            ):
                continue
            if not ((entry / "lance_db").is_dir() and (entry / "manifest.json").is_file()):
                continue
            try:
                if _sha256(entry / "manifest.json") != record.manifest_sha256:
                    log.warning("build %s manifest 与验证记录摘要不符，跳过", entry.name)
                    continue
            except OSError:
                continue
            candidates.append((record.generation, entry))
    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        best = candidates[0][1]
        log.info("回退到最近已验证 published 构建（generation=%d）：%s", candidates[0][0], best.name)
        return best / "lance_db"

    # Legacy 顶层：只在显式验证通过后接受（#35：不静默回退）
    legacy = index_dir / "lance_db"
    if legacy.is_dir() and (index_dir / "manifest.json").is_file():
        log.warning("回退到 legacy 顶层 lance_db（显式验证通过；建议 rebuild 迁移）")
        return legacy

    raise RebuildRequiredError(
        f"无可用活动索引：{pointer} 损坏/失配，且 builds/ 下无已验证 published 构建。"
        f"请执行 rebuild 重建索引。"
    )


class RebuildRequiredError(RuntimeError):
    """无可用活动索引，需要 rebuild。"""
