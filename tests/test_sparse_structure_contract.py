"""Issue #47 A — block-first sparse structure contract (pure, no model/LanceDB).

Guards the structure-aware provenance that replaced the over-fine
sentence/line/row plan (commit 062f3e6):

- every block becomes one exact-span sparse chunk (paragraph/list/code/quote/small
  table as a whole; large table split into header-aware row windows),
- ``structure_kind`` + ``table_header_*`` provenance are populated honestly,
- force-split long blocks tile the original text exactly (no fabricated continuous
  slice), and large-table windows cover each data row exactly once.
"""
from __future__ import annotations

from pathlib import Path

from chunking import ChunkRecord, chunk_page


def _chunk_page(content: str) -> list[ChunkRecord]:
    return chunk_page(
        "p1", Path("p1.md"), "Spec", "concept", content,
    )


def test_sparse_chunks_carry_structure_kind_and_real_spans():
    content = (
        "# Radar Spec\n\n"
        "The detection range reaches 250 meters under nominal conditions.\n\n"
        "## Notes\n\n"
        "- first bullet about calibration\n"
        "- second bullet about temperature\n"
    )
    records = _chunk_page(content)
    sparse = [r for r in records if r.chunk_kind == "sparse"]
    dense = [r for r in records if r.chunk_kind == "dense"]
    # Both retrieval kinds must be emitted (issue #47 C invariant).
    assert sparse and dense
    # Every sparse chunk reports a concrete structure kind (never blank/unknown).
    assert all(r.structure_kind for r in sparse)
    para = [r for r in sparse if r.structure_kind == "paragraph"]
    bullets = [r for r in sparse if r.structure_kind == "list"]
    assert para and bullets
    # Real span: content[start:end] maps back to the original paragraph verbatim.
    para_text = "The detection range reaches 250 meters under nominal conditions."
    assert content[para[0].start_char:para[0].end_char].strip() == para_text
    # Each bullet is its own exact-span block, not a merged blob.
    for b in bullets:
        assert content[b.start_char:b.end_char].strip().startswith("- ")


def test_force_split_tiles_original_text_exactly():
    """A single over-long paragraph must tile its source exactly when force-split
    (issue #47: keep real spans, never forge one continuous slice)."""
    para = "word " * 400  # ~2000 chars, exceeds SPARSE_HARD_MAX_CHARS
    content = "# Long\n\n" + para + "\n"
    records = _chunk_page(content)
    chunks = [r for r in records if r.chunk_kind == "sparse" and r.structure_kind == "paragraph"]
    assert len(chunks) >= 2  # it was actually split
    spans = sorted((c.start_char, c.end_char) for c in chunks)
    # Contiguous, non-overlapping, no gaps.
    for prev, cur in zip(spans, spans[1:]):
        assert cur[0] == prev[1]
    joined = "".join(content[s:e] for s, e in spans)
    assert joined == para


def test_small_table_is_single_exact_span_chunk():
    content = (
        "# Codes\n\n"
        "| Code | Meaning |\n|---|---|\n| 0x01 | timeout |\n| 0x02 | overflow |\n"
    )
    records = _chunk_page(content)
    tables = [r for r in records if r.structure_kind == "table"]
    assert len(tables) == 1  # fits => one chunk, not row-split
    t = tables[0]
    assert t.table_header_text == ""  # not split => no separate header provenance
    assert t.table_header_start_char == -1 and t.table_header_end_char == -1
    # The whole table is recoverable from its real span.
    assert content[t.start_char:t.end_char].strip().startswith("| Code | Meaning |")


def test_large_table_windows_cover_each_row_exactly_once():
    """Large table => header-aware row windows; every data row appears once,
    header is carried as provenance (not duplicated into chunk text)."""
    rows = [f"| {i:04d} | {'X' * 60} |" for i in range(20)]
    table = "| ID | Payload |\n|---|---|\n" + "\n".join(rows) + "\n"
    content = "# Big Table\n\n" + table
    records = _chunk_page(content)
    tables = [r for r in records if r.structure_kind == "table"]
    assert len(tables) >= 2  # it actually windowed
    header = "| ID | Payload |\n|---|---|"
    for t in tables:
        # Header provenance is the real header text, with real absolute offsets.
        assert t.table_header_text == header
        assert t.table_header_start_char >= 0
        assert t.table_header_start_char < t.table_header_end_char
        # The window body itself must NOT contain the header (header is separate).
        assert header not in t.text
    # Coverage-once: each data row appears exactly once across all windows.
    # (The section-prefix line "Big Table" is prepended to every window's text
    # as heading context, so count only table-row lines.)
    seen: dict[str, int] = {}
    for t in tables:
        for line in t.text.splitlines():
            line = line.strip()
            if line.startswith("|"):
                seen[line] = seen.get(line, 0) + 1
    for row in rows:
        assert seen.get(row.strip(), 0) == 1, f"row not covered exactly once: {row}"
    assert set(seen.keys()) == set(r.strip() for r in rows)
