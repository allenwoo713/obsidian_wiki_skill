"""Lexical tokenizer for FTS/BM25 (GitHub issue #2).

Shared by indexing **and** querying — both must call the same
implementation so indexed and queried terms never diverge.

Produces two term streams:

* ``fts_terms`` — Jieba Chinese words + CJK char bigrams (OOV / bad-segment
  fallback) + English / identifier tokens (CamelCase / snake_case /
  kebab-case split). No stop-word removal, no English stemming.
* ``exact_terms`` — model numbers, error codes, file paths, IPs, ports,
  URLs, numbers+units, CLI flags. Matched exactly.

Deterministic by design: application-layer segmentation + LanceDB
*whitespace* FTS. This avoids any runtime dependency on
``LANCE_LANGUAGE_MODEL_HOME`` and keeps indexing/query reproducible
across machines.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

try:
    import jieba
    jieba.setLogLevel(60)  # silence debug output
    _HAS_JIEBA = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_JIEBA = False

# Ordered most-specific first so "ARS540" wins over a bare number match.
_EXACT_SPECS: List[Tuple[str, str]] = [
    (r"https?://[^\s)\]]+", "url"),
    (r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b", "ip"),
    (r"[A-Za-z]:[\\/][^\s]+", "path"),
    (r"(?<![A-Za-z0-9/])(?:/[\w.\-]+){2,}", "path"),
    (r"--?[A-Za-z][\w\-]*", "flag"),
    (r"\b\d+(?:\.\d+)?\s?(?:mm|cm|m|km|kg|g|mg|s|ms|µs|Hz|kHz|MHz|GHz|Mbps|Gbps|Mbps|V|mV|A|mA|W|kW|°C|%|fps|rpm|px)\b",
     "numunit"),
    (r"\b(?:0x)?[A-Z]{1,4}-?\d{2,}(?:\.\d+)*\b", "code"),
    (r"(?:0[xX])[0-9A-Fa-f]+\b", "hex"),
    (r"\b\d{2,}(?:\.\d+)?\b", "number"),
]

_COMPILED = [(re.compile(p), tag) for p, tag in _EXACT_SPECS]
_CJK_RUN_RE = re.compile(r"[一-鿿]+")
_IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[_\-][A-Za-z0-9]+)*")
_SUBWORD_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")
_OOV_BIGRAM_RE = re.compile(r"[一-鿿]")


def load_lexicon(project_root: Optional[Path]) -> Set[str]:
    """Load ``<project_root>/lexicon.txt`` (one term per line).

    Terms are added to Jieba's user dict (if available) and returned as a
    set for direct inclusion in ``fts_terms``.
    """
    terms: Set[str] = set()
    if not project_root:
        return terms
    lex = Path(project_root) / "lexicon.txt"
    if not lex.exists():
        return terms
    for line in lex.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if not w or w.startswith("#"):
            continue
        terms.add(w)
        if _HAS_JIEBA:
            jieba.add_word(w)
    return terms


def extract_exact_terms(text: str) -> List[str]:
    """Extract exact-match terms (model numbers, codes, paths, ...).

    Order preserved, de-duplicated, original case kept.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for regex, _tag in _COMPILED:
        for m in regex.finditer(text):
            t = m.group(0)
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _cjk_bigrams(text: str) -> List[str]:
    """Adjacent char bigrams over each contiguous CJK run (OOV coverage)."""
    out: List[str] = []
    for run in _CJK_RUN_RE.findall(text):
        for i in range(len(run) - 1):
            out.append(run[i:i + 2])
    return out


def fts_terms(text: str, lexicon: Optional[Set[str]] = None) -> List[str]:
    """Produce the space-separated term stream for FTS indexing/query.

    Merges Jieba words (if available) + CJK bigrams + English/identifier
    tokens (with CamelCase/snake/kebab sub-tokens). De-duplicated.
    """
    terms: Set[str] = set()
    if lexicon:
        terms.update(lexicon)
    # English / identifiers + sub-tokens
    for m in _IDENT_RE.finditer(text):
        tok = m.group(0)
        terms.add(tok)
        for sub in _SUBWORD_RE.findall(tok):
            terms.add(sub.lower())
    # CJK bigrams (covers OOV and mis-segmented words)
    terms.update(_cjk_bigrams(text))
    # Jieba Chinese segmentation
    if _HAS_JIEBA:
        for w in jieba.cut(text):
            w = w.strip()
            if w:
                terms.add(w)
    return sorted(terms)


def tokenize_doc(text: str, lexicon: Optional[Set[str]] = None) -> List[str]:
    """Indexing entry point → ``fts_terms`` only."""
    return fts_terms(text, lexicon=lexicon)


def tokenize_query(text: str, lexicon: Optional[Set[str]] = None
                    ) -> Tuple[List[str], List[str]]:
    """Query entry point → ``(fts_terms, exact_terms)``."""
    return fts_terms(text, lexicon=lexicon), extract_exact_terms(text)
