"""Independent, strict local tokenizer for community-report budgets."""
from __future__ import annotations

import hashlib
from pathlib import Path


class TokenCounterUnavailable(RuntimeError):
    """The configured local tokenizer cannot safely count report text."""


class LocalReportTokenCounter:
    def __init__(self, model_dir: Path):
        tokenizer_path = Path(model_dir) / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise TokenCounterUnavailable("token_counter_unavailable: local tokenizer artifact is missing")
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
        except Exception as exc:
            raise TokenCounterUnavailable("token_counter_unavailable: local tokenizer could not be loaded") from exc
        digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        self.identity = f"hf-autotokenizer:{Path(model_dir).name}:{digest}:special=true:truncation=false"

    def count(self, text: str) -> int:
        try:
            encoded = self._tokenizer(
                text, add_special_tokens=True, truncation=False, return_attention_mask=False,
                return_token_type_ids=False, verbose=False,
            )
            return len(encoded["input_ids"])
        except Exception as exc:
            raise TokenCounterUnavailable("token_counter_unavailable: report token counting failed") from exc
