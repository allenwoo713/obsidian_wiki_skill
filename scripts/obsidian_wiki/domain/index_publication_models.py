"""#35 严格 generation 发布模型：生命周期状态与不可变验证记录（纯 stdlib）。

生命周期：building → validated → published → superseded。
每份 build 目录内的 ``.generation.json`` 持久化唯一不可变验证记录；
recovery 只从有 ``published`` 记录的 generation 回退。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class GenerationState(str, Enum):
    BUILDING = "building"
    VALIDATED = "validated"
    PUBLISHED = "published"
    SUPERSEDED = "superseded"


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
