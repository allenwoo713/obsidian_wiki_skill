"""Issue #13 integration tests: production build injects the REAL embedding
tokenizer and persists stable, content-derived chunk IDs.

These build a real LanceDB index (loading the embedding model), so they are
heavier than the pure-chunking unit tests in ``test_chunking.py``.

Uses ``tempfile.mkdtemp`` (not pytest ``tmp_path``) so the Windows sandbox
safe-delete guard on ``.pytest_tmp`` teardown does not mask results.
"""
import json
import re
import tempfile
from pathlib import Path

import chunking
from build_index import WikiIndex


def _norm_body(row):
    sp = json.loads(row["section_path"]) if row.get("section_path") else []
    prefix = (" / ".join(sp) + "\n") if sp else ""
    return row["text"][len(prefix):] if row["text"].startswith(prefix) else row["text"]


def _build_tiny():
    tmp = Path(tempfile.mkdtemp())
    wiki = tmp / "Wiki"
    wiki.mkdir()

    def _w(name, body):
        (wiki / name).write_text(
            f"---\ntitle: T\n---\n\n{body}", encoding="utf-8")

    _w("a.md", "# A\n" + "数据库异常增长导致WAL文件膨胀需紧急处理。" * 40
       + "\n## B\n第二段配置内容在此。\n## C\n[[SomeLink 说明]] 配置参考。\n")
    _w("b.md", "# B\n普通段落文字。\n| 码 | 说明 |\n|---|---|\n"
       "| 0x01 | 错误一 |\n| 0x02 | 错误二 |\n")
    idx = tmp / ".index"
    wi = WikiIndex(idx)
    wi.build(wiki)
    wi.load()
    return wi, wiki


def _persisted_rows(wi):
    """Read canonical ChunkRecord metadata through the split-table port."""
    return wi._get_repository().context_rows("chunk_id IS NOT NULL")


def test_build_uses_real_tokenizer():
    wi, _ = _build_tiny()
    rows = _persisted_rows(wi)
    dense = [r for r in rows if r["chunk_kind"] == "dense"]

    # 1) every dense row respects the hard token budget
    for r in dense:
        assert r["token_count"] <= chunking.DENSE_HARD_MAX_TOKENS, \
            (r["chunk_id"], r["token_count"])

    # 2) chunk_id written verbatim from ChunkRecord.chunk_id (page_id::{16hex})
    for r in rows:
        assert re.fullmatch(r".*::[0-9a-f]{16}", r["chunk_id"]), r["chunk_id"]

    # 3) real tokenizer used — token_count must NOT equal the char//4 fallback
    #    for at least some dense row (Mixed-Chinese embedding differs from 4c/tok)
    mismatched = sum(1 for r in dense if r["token_count"] != len(r["text"]) // 4)
    assert mismatched > 0, "build appears to have used char//4 fallback"


def test_build_stable_ids_under_insertion():
    wi, wiki = _build_tiny()
    before = {(r["chunk_kind"], chunking._norm(_norm_body(r))): r["chunk_id"]
              for r in _persisted_rows(wi)}

    # insert an unrelated section at the very top of page a.md
    base = (wiki / "a.md").read_text(encoding="utf-8")
    (wiki / "a.md").write_text(
        "---\ntitle: T\n---\n\n## UNRELATED\nirrelevant stuff here.\n\n"
        + base.split("---\n", 2)[-1], encoding="utf-8")

    wi.build(wiki)  # incremental rebuild
    after = {(r["chunk_kind"], chunking._norm(_norm_body(r))): r["chunk_id"]
             for r in _persisted_rows(wi)}

    for key, cid in before.items():
        assert after.get(key) == cid, \
            f"unchanged chunk {key} changed ID after top insertion"
