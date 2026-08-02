"""Build publication boundary; concrete filesystem mechanics remain outside core."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class IndexPublisher(Protocol):
    def publish(self, staged_build: Path, active_index_file: Path) -> None: ...
