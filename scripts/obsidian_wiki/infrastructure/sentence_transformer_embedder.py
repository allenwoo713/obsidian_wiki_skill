"""Local SentenceTransformer adapter behind the SDK-free embedding port."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple


class SentenceTransformerEmbedder:
    """Lazily encode dense text using only a verified local model directory."""

    def __init__(self, model_path: Path):
        self._model_path = Path(model_path)
        self._model = None

    def embed(self, texts: Sequence[str]) -> Sequence[Tuple[float, ...]]:
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Dense embedding inputs must be text strings")
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
        vectors = self._model.encode(
            list(texts), show_progress_bar=False, normalize_embeddings=False
        )
        return [tuple(float(value) for value in vector) for vector in vectors]
