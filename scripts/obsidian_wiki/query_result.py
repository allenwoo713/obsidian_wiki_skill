"""Parse ``query.py --json`` output into one stable, flat hit list.

``query.py`` emits two top-level groups — ``text`` (paragraph hits) and
``images`` (image hits) — each item sharing the same shape
(``score``/``title``/``path``/``evidence``/``sources``/``citation``/...).
Scripting agents kept mis-reading the schema (looking for ``results``/``hits``,
or only one of the two groups). This module is the single contract: call
``load_hits`` and iterate; never touch the raw top-level keys.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_hits(source: str | Path | dict) -> list[dict[str, Any]]:
    """Return every query hit (text + images) as a flat list.

    Each item keeps all original fields and gains a ``kind`` discriminator
    (``"text"`` / ``"image"``) so callers can branch without re-reading the
    top-level grouping.

    ``source`` may be a JSON file path or an already-parsed ``dict``.
    """
    data = source if isinstance(source, dict) else json.loads(
        Path(source).read_text(encoding="utf-8"))
    hits: list[dict[str, Any]] = []
    for kind, key in (("text", "text"), ("image", "images")):
        for item in data.get(key, []) or []:
            merged = dict(item)
            merged["kind"] = kind
            hits.append(merged)
    return hits


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "-"
    payload = json.load(sys.stdin if src == "-" else open(src, encoding="utf-8"))
    for i, h in enumerate(load_hits(payload if src == "-" else src)):
        print(f"[{i}] {h.get('kind')} score={h.get('score')} "
              f"{h.get('title')} -> {h.get('path')}")
