"""Stable persisted-manifest boundary for index builds."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from obsidian_wiki.domain.index_models import IndexManifest


class IndexManifestStore(Protocol):
    def write(self, path: Path, manifest: IndexManifest) -> None: ...

    def read(self, path: Path) -> IndexManifest: ...
