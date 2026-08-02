"""Filesystem persistence for reproducible #17 staged-build evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


class FilesystemIndexManifest:
    """Write only complete JSON records; build orchestration owns publication."""

    def write(self, path: Path, manifest: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
