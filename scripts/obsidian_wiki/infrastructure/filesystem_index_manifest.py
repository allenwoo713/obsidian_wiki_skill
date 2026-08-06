"""Filesystem persistence for reproducible #17 staged-build evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from obsidian_wiki.infrastructure.durable_filesystem import atomic_write_bytes


class FilesystemIndexManifest:
    """Write only complete JSON records; build orchestration owns publication.

    #36：manifest 经 durable 原子写（同目录 tmp → fsync → replace → 父目录同步），
    禁止对已发布或 staged manifest 使用原位 ``write_text`` 覆盖。
    """

    def write(self, path: Path, manifest: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
        )
