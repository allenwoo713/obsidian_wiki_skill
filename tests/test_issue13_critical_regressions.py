"""Regression coverage for the four remaining Issue #13 Critical findings."""
from pathlib import Path
from unittest import mock

import pytest

import build_index
import chunking
from build_index import ChunkBuildError, WikiIndex
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository


def _wiki(tmp_path: Path, pages: int = 2) -> Path:
    wiki = tmp_path / "Wiki"
    wiki.mkdir()
    for i in range(pages):
        (wiki / f"p{i}.md").write_text(
            f"---\ntitle: P{i}\n---\n\n# P{i}\nold indexed body {i}.\n",
            encoding="utf-8",
        )
    return wiki


def _active_rows(index_dir: Path):
    loaded = WikiIndex(index_dir)
    loaded.load()
    return loaded._get_repository().context_rows("1 = 1")


def test_rejects_dense_only_page_and_preserves_active_index(tmp_path):
    wiki = _wiki(tmp_path)
    index_dir = tmp_path / ".index"
    wi = WikiIndex(index_dir)
    wi.build(wiki)
    old_pointer = (index_dir / "ACTIVE_INDEX").read_bytes()
    old_rows = _active_rows(index_dir)
    real = chunking.chunk_page

    def _dense_only(*args, **kwargs):
        return [r for r in real(*args, **kwargs) if r.chunk_kind == "dense"]

    with mock.patch("build_index.chunk_page", _dense_only):
        with pytest.raises(ChunkBuildError, match="retrieval kinds"):
            wi.build(wiki)

    assert (index_dir / "ACTIVE_INDEX").read_bytes() == old_pointer
    assert _active_rows(index_dir) == old_rows


def test_rejects_silent_persisted_row_loss_before_publish(tmp_path, monkeypatch):
    wiki = _wiki(tmp_path)
    index_dir = tmp_path / ".index"
    wi = WikiIndex(index_dir)
    wi.build(wiki)
    old_pointer = (index_dir / "ACTIVE_INDEX").read_bytes()

    original_persist = LanceDbIndexRepository.persist

    def _drop_sparse_row(self, lance_dir, sparse_chunks, dense_chunks, fts_config):
        # Exercise the physical D-01 sparse write, not the retired private
        # mixed-table accessor.  Reopened validation must reject row loss.
        return original_persist(self, lance_dir, sparse_chunks[:-1], dense_chunks, fts_config)

    monkeypatch.setattr(LanceDbIndexRepository, "persist", _drop_sparse_row)
    with pytest.raises(RuntimeError, match="持久化完整性"):
        wi.build(wiki)

    assert (index_dir / "ACTIVE_INDEX").read_bytes() == old_pointer


def test_same_clock_failed_build_never_mutates_active_directory(tmp_path):
    wiki = _wiki(tmp_path)
    index_dir = tmp_path / ".index"
    wi = WikiIndex(index_dir)
    with mock.patch("build_index.time.time_ns", return_value=42):
        wi.build(wiki)
        old_rows = _active_rows(index_dir)
        old_pointer = (index_dir / "ACTIVE_INDEX").read_bytes()

        p0 = str((wiki / "p0.md").resolve())
        real = chunking.chunk_page

        def _fail_second(page_id, *args, **kwargs):
            if page_id == p0:
                raise RuntimeError("injected failure")
            return real(page_id, *args, **kwargs)

        with mock.patch("build_index.chunk_page", _fail_second):
            with pytest.raises(ChunkBuildError):
                wi.build(wiki)

    assert (index_dir / "ACTIVE_INDEX").read_bytes() == old_pointer
    assert _active_rows(index_dir) == old_rows
    builds = list((index_dir / "builds").iterdir())
    assert len(builds) == 2
    assert len({p.name for p in builds}) == 2
    assert any((p / ".failed").exists() for p in builds)


def _sparse(records):
    return [record for record in records if record.chunk_kind == "sparse"]


def _body(record):
    return record.text.split("\n", 1)[1] if "\n" in record.text else record.text


def test_sparse_spans_are_exact_for_repeated_code_and_no_trailing_pipe_table():
    repeated = "x" * 1200
    table_cell = "y" * 1200
    content = (
        "# A\n```\n" + repeated + "\n" + repeated + "\n```\n"
        "| head | details\n|---|---\n| left | " + table_cell + " | tail"
    )
    records = chunking.chunk_page("page", Path("Wiki/a.md"), "A", "concept", content,
                                  tokenizer=len)
    sparse = _sparse(records)
    assert sparse
    starts = [record.start_char for record in sparse]
    assert starts == sorted(starts)
    for record in sparse:
        assert 0 <= record.start_char <= record.end_char <= len(content)
        assert _body(record) == content[record.start_char:record.end_char]


def test_dense_hard_limit_covers_long_url_and_overlong_prefix():
    url = "https://example.test/" + ("unbroken_identifier_" * 100)
    heading = "section-path-" * 30
    content = f"# {heading}\n{url}"
    records = chunking.chunk_page("page", Path("Wiki/a.md"), "A", "concept", content,
                                  tokenizer=len)
    dense = [record for record in records if record.chunk_kind == "dense"]
    assert dense
    assert all(record.token_count <= chunking.DENSE_HARD_MAX_TOKENS for record in dense)
    assert all(len(record.text) <= chunking.DENSE_HARD_MAX_TOKENS for record in dense)
