"""SDK-neutral dense embedding boundary."""
from __future__ import annotations

from typing import Protocol, Sequence, Tuple


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Tuple[float, ...]]: ...
