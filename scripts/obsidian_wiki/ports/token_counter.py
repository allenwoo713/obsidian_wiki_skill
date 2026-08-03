"""Strict report token-budget boundary."""
from __future__ import annotations

from typing import Protocol


class TokenCounter(Protocol):
    identity: str

    def count(self, text: str) -> int: ...
