"""Shared SDK-free publication collaborator for snapshot and staged builds.

The application services supply their existing, tested pure publication operations as
callables.  Keeping those seams in one explicit object makes the composition root
responsible for wiring while allowing the two build routes to share observable rules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from obsidian_wiki.domain.index_models import DenseChunk, SparseChunk
from obsidian_wiki.ports.chunk_repository import ChunkRepository


class IndexPublicationService:
    """One request-graph collaborator for generation, validation and manifests."""

    def __init__(self) -> None:
        self._next_generation: Callable[[Path], int] | None = None
        self._exact_term: Callable[[Sequence[SparseChunk]], str] | None = None
        self._disk_bytes: Callable[[Path], int] | None = None
        self._candidate_validation: Callable[..., Any] | None = None
        self._manifest: Callable[..., dict] | None = None
        self.ann_policy: Any = None

    def bind(
        self,
        *,
        next_generation: Callable[[Path], int],
        exact_term: Callable[[Sequence[SparseChunk]], str],
        disk_bytes: Callable[[Path], int],
        candidate_validation: Callable[..., Any],
        manifest: Callable[..., dict],
        ann_policy: Any,
    ) -> None:
        self._next_generation = next_generation
        self._exact_term = exact_term
        self._disk_bytes = disk_bytes
        self._candidate_validation = candidate_validation
        self._manifest = manifest
        self.ann_policy = ann_policy

    @staticmethod
    def _required(callback: Callable[..., Any] | None, name: str) -> Callable[..., Any]:
        if callback is None:
            raise RuntimeError(f"index_publication_service_unbound:{name}")
        return callback

    def allocate_generation(self, index_dir: Path) -> int:
        return self._required(self._next_generation, "generation")(index_dir)

    def canonical_exact_term(self, chunks: Sequence[SparseChunk]) -> str:
        return self._required(self._exact_term, "exact_term")(chunks)

    def staged_disk_bytes(self, build_dir: Path) -> int:
        return self._required(self._disk_bytes, "disk_bytes")(build_dir)

    def validate_candidate(
        self, repository: ChunkRepository, dense_chunks: Sequence[DenseChunk], **kwargs: Any,
    ) -> Any:
        return self._required(self._candidate_validation, "candidate_validation")(
            repository, dense_chunks, **kwargs
        )

    def construct_manifest(self, **kwargs: Any) -> dict:
        return self._required(self._manifest, "manifest")(**kwargs)
