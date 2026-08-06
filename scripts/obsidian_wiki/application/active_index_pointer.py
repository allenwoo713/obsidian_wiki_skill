"""#11/#21/#35/#36/#37 ACTIVE_INDEX 指针契约：严格 schema v4 + 生命周期 + 耐久发布。

#35（review 复审）：
- pointer/record 经 ``ActiveIndexPointerV4.from_json`` / ``GenerationRecord.from_json``
  严格构造（字段集合精确相等、正整数 generation、UTC、fullmatch digest/build_id）。
- 集中生命周期转换 missing→building→validated→published→superseded，其它拒绝；
  build 目录创建后立即耐久写 BUILDING。
- ``publish_pointer`` 只接受存在完全匹配 VALIDATED record 的 build（任意含 manifest
  的目录不可直接跳 PUBLISHED）。
- direct resolve / recovery 身份绑定：pointer.build_id == build_dir.name ==
  record.build_id == manifest.build_id，generation/digest 全链一致。
- 删除普通 legacy fallback：无法验证的 legacy 一律 RebuildRequiredError。

#36 耐久：所有关键写入经 durable_filesystem（fsync/replace 失败向上传播）。
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from obsidian_wiki.application.durable_filesystem import (
    CommitUncertainError,
    atomic_write_bytes,
)
from obsidian_wiki.domain.index_publication_models import (
    ALLOWED_TRANSITIONS,
    ActiveIndexPointerV4,
    GenerationRecord,
    GenerationState,
)

log = logging.getLogger(__name__)

POINTER_NAME = "ACTIVE_INDEX"
GEN_RECORD_NAME = ".generation.json"
SCHEMA_VERSION = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_record(build_dir: Path, record: GenerationRecord) -> None:
    atomic_write_bytes(
        build_dir / GEN_RECORD_NAME,
        json.dumps(record.to_json(), sort_keys=True).encode("utf-8"),
    )


def read_generation_record(build_dir: Path) -> GenerationRecord | None:
    path = build_dir / GEN_RECORD_NAME
    if not path.is_file():
        return None
    try:
        return GenerationRecord.from_json(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _transition(build_dir: Path, target: GenerationState, *,
                build_id: str, generation: int, manifest_sha256: str) -> None:
    """集中生命周期转换（#35）：仅 missing→building→validated→published→superseded。"""
    current = read_generation_record(build_dir)
    current_state = current.state if current is not None else None
    if target not in ALLOWED_TRANSITIONS.get(current_state, frozenset()):
        raise RuntimeError(
            f"非法生命周期转换: {current_state} -> {target}（{build_dir.name}）")
    now = _now()
    record = GenerationRecord(
        generation=generation, build_id=build_id, state=target,
        manifest_sha256=manifest_sha256,
        validated_at=current.validated_at if current is not None else now,
        published_at=(now if target is GenerationState.PUBLISHED
                      else (current.published_at if current is not None else None)),
        superseded_at=(now if target is GenerationState.SUPERSEDED
                       else (current.superseded_at if current is not None else None)),
    )
    _write_record(build_dir, record)


def record_building(build_dir: Path, *, build_id: str, generation: int) -> None:
    """build 目录创建后立即耐久写 BUILDING（#35：missing → building）。"""
    _transition(build_dir, GenerationState.BUILDING, build_id=build_id,
                generation=generation, manifest_sha256="")


def record_validated(build_dir: Path, *, build_id: str, generation: int,
                     manifest_sha256: str) -> None:
    """manifest 落盘并完整验证后 building → validated（#35：staging 绝不被 recovery 选中）。"""
    _transition(build_dir, GenerationState.VALIDATED, build_id=build_id,
                generation=generation, manifest_sha256=manifest_sha256)


def publish_pointer(
    index_dir: Path, build_dir: Path, *, generation: int = 0, build_id: str = "",
) -> None:
    """原子翻转 ACTIVE_INDEX 指向 ``build_dir``；失败时旧指针保持可用。

    前置条件（#35）：存在完全匹配的 VALIDATED record（build_id/generation/digest 一致），
    任意仅含 manifest 的目录不可直接跳到 PUBLISHED。
    """
    if not isinstance(generation, int) or isinstance(generation, bool) or generation <= 0:
        raise RuntimeError(f"拒绝发布：generation={generation!r} 必须为正整数")
    if not isinstance(build_id, str) or build_dir.name != build_id:
        raise RuntimeError(f"拒绝发布：build_dir={build_dir.name} 与 build_id={build_id} 不一致")
    manifest = build_dir / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"拒绝发布：{manifest} 不存在")
    manifest_digest = _sha256(manifest)
    record = read_generation_record(build_dir)
    if record is None or record.state is not GenerationState.VALIDATED:
        raise RuntimeError(
            f"拒绝发布：{build_dir.name} 无 VALIDATED 生命周期记录"
            f"（state={record.state if record is not None else None}）")
    if (record.build_id != build_id or record.generation != generation
            or record.manifest_sha256 != manifest_digest):
        raise RuntimeError("拒绝发布：VALIDATED 记录身份与 build 不一致")
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
    except CommitUncertainError:
        # #36：replace 已成功（commit point 已过）。读回 pointer 确认内容与本 build
        # 一致（record 尚未 transition 到 PUBLISHED，故不要求 record 状态）；一致则
        # 继续 lifecycle 视为已发布 + reconciliation warning。
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if not isinstance(data, dict) or data.get("build_id") != build_id \
                or data.get("generation") != generation \
                or data.get("manifest_sha256") != manifest_digest:
            raise RuntimeError(
                "ACTIVE_INDEX 发布状态不确定（replace 后耐久确认失败且新指针不可验证）；"
                "请重新解析确认状态后重试。")
        log.warning("ACTIVE_INDEX 已替换但目录同步确认失败（commit uncertain，已 reconciliation）")
    except OSError as exc:
        raise RuntimeError(
            f"ACTIVE_INDEX 发布失败（旧活动索引保持可用）：{type(exc).__name__}: {exc}。"
            f"请确认 {pointer} 未被 Obsidian/杀软独占后重试。"
        ) from exc
    # 生命周期：validated → published；新 pointer 耐久落盘后才标旧代 superseded（#35）。
    # #37：commit 点之后的 reconciliation 失败不得伪装为 pre-commit 失败（不写 .failed）。
    try:
        _transition(build_dir, GenerationState.PUBLISHED, build_id=build_id,
                    generation=generation, manifest_sha256=manifest_digest)
        _mark_superseded(index_dir, build_dir, generation)
    except (OSError, RuntimeError) as exc:
        log.warning("发布后生命周期 reconciliation 失败（索引已发布）：%s", exc)


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
        if record is None or record.state is not GenerationState.PUBLISHED:
            continue
        try:
            _transition(entry, GenerationState.SUPERSEDED,
                        build_id=record.build_id, generation=record.generation,
                        manifest_sha256=record.manifest_sha256)
        except RuntimeError as exc:
            log.warning("标记 superseded 失败（不影响新发布）：%s", exc)


def _validate_pointer(index_dir: Path, data: object) -> Path | None:
    """严格校验 schema v4 pointer 并绑定身份（#35）；返回 lance_dir 或 None（进 recovery）。"""
    try:
        ptr = ActiveIndexPointerV4.from_json(data)
    except ValueError:
        return None
    # 拒绝任意 symlink 组件（builds / <build_id> / lance_db）
    probe = index_dir
    for part in Path(ptr.active_lance).parts:
        probe = probe / part
        if probe.is_symlink():
            return None
    build_dir = index_dir / "builds" / ptr.build_id
    if not build_dir.is_dir():
        return None
    manifest = build_dir / "manifest.json"
    if not manifest.is_file() or ptr.manifest_sha256 != _sha256(manifest):
        return None
    # manifest 身份一致（build_id / generation）
    try:
        m = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(m, dict) or m.get("build_id") != ptr.build_id or m.get("generation") != ptr.generation:
        return None
    # record：仅 PUBLISHED，且 build_id/generation/digest 全链一致
    record = read_generation_record(build_dir)
    if record is None or record.state is not GenerationState.PUBLISHED:
        return None
    if (record.build_id != ptr.build_id or record.generation != ptr.generation
            or record.manifest_sha256 != ptr.manifest_sha256):
        return None
    return build_dir / "lance_db"


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

    # Recovery：只从有 published/superseded 生命周期记录且完整身份一致的 generation 回退
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
            if entry.name != record.build_id:
                continue  # 目录名必须等于 record.build_id（#35）
            if not ((entry / "lance_db").is_dir() and (entry / "manifest.json").is_file()):
                continue
            try:
                manifest_digest = _sha256(entry / "manifest.json")
            except OSError:
                continue
            if manifest_digest != record.manifest_sha256:
                continue
            try:
                m = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(m, dict) or m.get("build_id") != record.build_id \
                    or m.get("generation") != record.generation:
                continue
            candidates.append((record.generation, entry))
    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        best = candidates[0][1]
        log.info("回退到最近已验证 published 构建（generation=%d）：%s", candidates[0][0], best.name)
        return best / "lance_db"

    # #35：删除普通 legacy fallback——无法验证的 legacy 一律要求 rebuild。
    raise RebuildRequiredError(
        f"无可用活动索引：{pointer} 损坏/失配，且 builds/ 下无已验证 published 构建。"
        f"请执行 rebuild 重建索引。"
    )


class RebuildRequiredError(RuntimeError):
    """无可用活动索引，需要 rebuild。"""
