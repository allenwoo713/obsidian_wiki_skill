"""Strict report token-budget boundary."""
from __future__ import annotations

from typing import Protocol


class TokenCounterUnavailable(RuntimeError):
    """The configured report token counter cannot safely count text."""


class TokenCounter(Protocol):
    identity: str

    def count(self, text: str) -> int: ...
