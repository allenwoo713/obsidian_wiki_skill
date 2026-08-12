"""Issue #47 E — page-level RRF contributes once per page per channel.

Before #47, a single page with N FTS fragments received N RRF contributions,
inflating its score and biasing retrieval toward fragmented pages. The fix
scores each page once per channel while still retaining every fragment as
evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from models import ChunkHit
from fusion import page_level_rrf


def _hit(page_id, channel, score, text="x", chunk_kind="dense"):
    return ChunkHit(
        chunk_id=f"v1:{page_id}:{channel}:{score}",
        page_id=page_id, path=f"/wiki/{page_id}.md", title=page_id,
        page_type="concept", section_path=[], heading="",
        chunk_kind=chunk_kind, text=text, channel=channel, score=score,
    )


def test_one_rrf_contribution_per_page_per_channel():
    # Page P has 3 FTS fragments (ranks 1..3) but must score only ONCE on fts.
    fts = [_hit("P", "fts", 5.0 - i, text=f"f{i}") for i in range(3)]
    out = page_level_rrf(fts, [], k=5)
    assert len(out) == 1
    # Exactly 1/(60+1) — not 1/61 + 1/62 + 1/63.
    assert out[0].rrf_score == pytest.approx(1.0 / 61.0)


def test_all_fragments_retained_as_evidence():
    fts = [_hit("P", "fts", 5.0 - i, text=f"f{i}") for i in range(3)]
    out = page_level_rrf(fts, [], k=5)
    # Evidence is preserved per fragment even though the RRF score is counted once.
    assert len(out[0].sparse_evidence) == 3


def test_two_channels_give_two_contributions():
    fts = [_hit("P", "fts", 5.0, text="a")]
    vec = [_hit("P", "vector", 9.0, text="b")]
    out = page_level_rrf(fts, vec, k=5)
    assert len(out) == 1
    # fts once + vector once = 2/61.
    assert out[0].rrf_score == pytest.approx(2.0 / 61.0)
    assert out[0].sparse_evidence and out[0].dense_evidence


def test_dual_channel_outranks_single_channel():
    dual = page_level_rrf(
        [_hit("P", "fts", 5.0, text="a")],
        [_hit("P", "vector", 9.0, text="b")],
        k=5,
    )[0]
    single = page_level_rrf(
        [_hit("Q", "fts", 5.0, text="a")],
        [],
        k=5,
    )[0]
    # A page present in both channels scores exactly double a single-channel page.
    assert dual.rrf_score == pytest.approx(2.0 * single.rrf_score)
