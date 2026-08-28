"""Local SentenceTransformer adapter behind the SDK-free embedding port."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple


class SentenceTransformerEmbedder:
    """Lazily encode dense text using only a verified local model directory."""

    def __init__(self, model_path: Path):
        self._model_path = Path(model_path)
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            if not (self._model_path / "model.safetensors").is_file():
                raise RuntimeError(
                    "Local embedding model assets are unavailable at "
                    f"{self._model_path}; download the configured model before building an index."
                )
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(str(self._model_path), local_files_only=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to load local embedding model at {self._model_path}: {exc}"
                ) from exc

    def embed(self, texts: Sequence[str], *, show_progress_bar: bool = False) -> Sequence[Tuple[float, ...]]:
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Dense embedding inputs must be text strings")
        self._ensure_model()
        vectors = self._model.encode(
            list(texts), show_progress_bar=show_progress_bar, normalize_embeddings=False
        )
        return [tuple(float(value) for value in vector) for vector in vectors]

    @property
    def tokenizer(self):
        """Expose the underlying HF tokenizer for token-aware chunking (issue #39)."""
        self._ensure_model()
        return self._model.tokenizer
