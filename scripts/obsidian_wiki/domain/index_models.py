"""SDK-free values for the D-01 LanceDB storage contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


class RebuildRequiredError(RuntimeError):
    """Raised when a persisted index belongs to the retired ``chunks`` layout."""


@dataclass(frozen=True)
class FtsIndexConfig:
    """The immutable D-04 tokenizer contract, independent of LanceDB's API."""

    base_tokenizer: str = "whitespace"
    lower_case: bool = False
    stem: bool = False
    remove_stop_words: bool = False
    ascii_folding: bool = False
    max_token_length: int = 256


@dataclass(frozen=True)
class SparseChunk:
    chunk_id: str
    page_id: str
    path: str
    title: str
    text: str
    fts_text: str


@dataclass(frozen=True)
class DenseChunk:
    chunk_id: str
    page_id: str
    path: str
    title: str
    text: str
    vector: Tuple[float, ...]


@dataclass(frozen=True)
class StorageArtifact:
    lance_dir: Path
    manifest_path: Path
    sparse_count: int
    dense_count: int
