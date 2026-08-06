"""#35 严格 generation 发布模型：生命周期状态与不可变验证记录（纯 stdlib）。

生命周期：missing → building → validated → published → superseded（集中 transition 表）。
``ActiveIndexPointerV4`` / ``GenerationRecord`` 提供严格 ``from_json`` 构造器：
字段集合精确相等（拒绝额外/缺失）、generation 必须 ``type is int`` 正整数（拒绝 bool/float/str）、
build_id/digest fullmatch、时间戳必须 timezone-aware 且 offset 0（UTC）。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

_BUILD_ID_RE = re.compile(r"^build_\d{8}T\d{12}_[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
POINTER_FIELDS = frozenset({
    "schema_version", "generation", "build_id",
    "active_lance", "manifest_sha256", "published_at",
})
GENERATION_RECORD_FIELDS = frozenset({
    "generation", "build_id", "state", "manifest_sha256",
    "validated_at", "published_at", "superseded_at",
})


class GenerationState(str, Enum):
    BUILDING = "building"
    VALIDATED = "validated"
    PUBLISHED = "published"
    SUPERSEDED = "superseded"


def require_positive_int(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"必须为正整数，got {value!r}")
    return value


def require_utc(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须为字符串，got {value!r}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{field} 不可解析: {value!r}")
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field} 必须为 UTC（offset 0），got {value!r}")
    return value


def require_build_id(value: object) -> str:
    if not isinstance(value, str) or not _BUILD_ID_RE.fullmatch(value):
        raise ValueError(f"build_id 非法: {value!r}")
    return value


def require_sha256(value: object) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"manifest_sha256 非法: {value!r}")
    return value


class ActiveIndexPointerV4:
    """严格 schema v4 pointer：字段集合精确相等 + 严格类型/UTC/digest 校验（#35）。"""

    __slots__ = ("schema_version", "generation", "build_id", "active_lance",
                 "manifest_sha256", "published_at")

    def __init__(self, *, generation: int, build_id: str, active_lance: str,
                 manifest_sha256: str, published_at: str) -> None:
        self.schema_version = 4
        self.generation = require_positive_int(generation)
        self.build_id = require_build_id(build_id)
        if active_lance != f"builds/{self.build_id}/lance_db":
            raise ValueError(f"active_lance 必须精确为 builds/<build_id>/lance_db: {active_lance!r}")
        self.active_lance = active_lance
        self.manifest_sha256 = require_sha256(manifest_sha256)
        self.published_at = require_utc(published_at, "published_at")

    @classmethod
    def from_json(cls, data: object) -> "ActiveIndexPointerV4":
        if not isinstance(data, dict):
            raise ValueError("pointer payload 必须为 dict")
        if set(data) != POINTER_FIELDS:
            raise ValueError(
                f"pointer 字段集合必须精确为 {sorted(POINTER_FIELDS)}，got {sorted(data)}")
        sv = data.get("schema_version")
        if type(sv) is not int or sv != 4:
            raise ValueError("schema_version 必须为整数 4")
        return cls(
            generation=data["generation"], build_id=data["build_id"],
            active_lance=data["active_lance"],
            manifest_sha256=data["manifest_sha256"], published_at=data["published_at"],
        )


@dataclass(frozen=True)
class GenerationRecord:
    """不可变 generation 验证/发布记录（落盘于 build 目录 ``.generation.json``）。"""

    generation: int
    build_id: str
    state: GenerationState
    manifest_sha256: str
    validated_at: str
    published_at: str | None = None
    superseded_at: str | None = None

    def to_json(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=str, ensure_ascii=False))

    @classmethod
    def from_json(cls, data: object) -> "GenerationRecord":
        if not isinstance(data, dict):
            raise ValueError("generation record 必须为 dict")
        if set(data) != GENERATION_RECORD_FIELDS:
            raise ValueError(
                f"generation record 字段集合必须精确为 {sorted(GENERATION_RECORD_FIELDS)}，"
                f"got {sorted(data)}")
        try:
            state = GenerationState(data["state"])
        except ValueError:
            raise ValueError(f"state 非法: {data.get('state')!r}")
        digest = data["manifest_sha256"]
        if state is not GenerationState.BUILDING:
            digest = require_sha256(digest)
        elif digest != "":
            raise ValueError(f"BUILDING 记录的 manifest_sha256 必须为空: {digest!r}")
        return cls(
            generation=require_positive_int(data["generation"]),
            build_id=require_build_id(data["build_id"]),
            state=state,
            manifest_sha256=digest,
            validated_at=require_utc(data["validated_at"], "validated_at"),
            published_at=require_utc(data["published_at"], "published_at")
            if data.get("published_at") is not None else None,
            superseded_at=require_utc(data["superseded_at"], "superseded_at")
            if data.get("superseded_at") is not None else None,
        )


# 集中生命周期转换表：#35 不允许跳跃/逆向/重复转换。
ALLOWED_TRANSITIONS: dict[GenerationState | None, frozenset[GenerationState]] = {
    None: frozenset({GenerationState.BUILDING}),
    GenerationState.BUILDING: frozenset({GenerationState.VALIDATED}),
    GenerationState.VALIDATED: frozenset({GenerationState.PUBLISHED}),
    GenerationState.PUBLISHED: frozenset({GenerationState.SUPERSEDED}),
}
