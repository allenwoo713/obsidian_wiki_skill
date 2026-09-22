"""Issue #65 integration gates. Requires the repository's locked dependencies.

验收边界（对应 #65 comment 3）：三 mode 发布 provenance、构建期编辑不换 ACTIVE_INDEX、
load 固定 generation、strict 先于 planner、缺 provenance=unknown、锁内 capture、
长生命周期对象重查、查询中编辑在返回前被拒。
"""
from __future__ import annotations
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from build_index import WikiIndex, build_storage_contract, parse_wiki_page
from obsidian_wiki.application.wiki_freshness import (
    FreshnessError, SnapshotError, capture_wiki, inspect_snapshot,
)


def write_page(wiki: Path, name="a.md", marker="before65") -> Path:
    wiki.mkdir(parents=True, exist_ok=True)
    page = wiki / name
    body = "".join(
        f"\n## Section {i}\n{marker} Calibration procedure detail number {i}. " * 12
        for i in range(12)
    )
    page.write_text(
        "---\ntitle: Freshness test\ntype: concept\nsources: []\n---\n" + body,
        encoding="utf-8",
    )
    return page


def embed384(texts):
    vectors = []
    for row, _ in enumerate(texts):
        rng = random.Random(77 + row)
        raw = [rng.gauss(0.0, 1.0) for _ in range(384)]
        norm = math.sqrt(sum(x*x for x in raw))
        vectors.append([x / norm for x in raw])
    return vectors


def build(root: Path, *, mode="snapshot", embed=embed384):
    if mode == "auto":
        # auto 需要 project 内 policy load；disabled policy 使 auto 安全选择
        # snapshot（select_auto_build_mode → policy_disabled），走真实 loader 路径。
        policy_dir = root / ".index"
        policy_dir.mkdir(parents=True, exist_ok=True)
        (policy_dir / "build-mode-policy.json").write_text(
            json.dumps({"schema_version": 1, "enabled": False}), encoding="utf-8")
        from obsidian_wiki.application.incremental_policy import load_build_mode_policy
        policy = load_build_mode_policy(root)
    else:
        policy = None
    return build_storage_contract(
        root / "Wiki", root / ".index", embed=embed,
        tokenizer=lambda text: max(1, len(text) // 10),
        lexicon={}, build_mode=mode, build_mode_policy=policy,
    )


def test_parser_does_not_hash_a_second_disk_version(tmp_path, monkeypatch):
    page = write_page(tmp_path / "Wiki")
    original = page.read_bytes()
    changed = original.replace(b"before65", b"after_65")
    real_read_text = Path.read_text

    def edit_after_text_read(path, *args, **kwargs):
        result = real_read_text(path, *args, **kwargs)
        if path == page:
            path.write_bytes(changed)
        return result

    # Old parse_wiki_page reads body first, then hashes changed bytes: test fails there.
    monkeypatch.setattr(Path, "read_text", edit_after_text_read)
    parsed = parse_wiki_page(page, tmp_path)
    assert parsed is not None
    assert "before65" in parsed.content
    assert parsed.sha256 == hashlib.sha256(original).hexdigest()


@pytest.mark.parametrize("mode", ["snapshot", "incremental", "auto"])
def test_all_public_build_modes_publish_matching_provenance(tmp_path, mode):
    page = write_page(tmp_path / "Wiki")
    build(tmp_path)
    page.write_text(page.read_text().replace("before65", "changed65"))
    outcome = build(tmp_path, mode=mode)
    manifest = json.loads(outcome.artifact.manifest_path.read_bytes())
    assert manifest["build_id"] == outcome.build_id
    assert manifest["generation"] == outcome.generation
    assert inspect_snapshot(tmp_path / "Wiki", manifest["wiki_snapshot"])["status"] == "fresh"


@pytest.mark.parametrize("mode", ["snapshot", "incremental"])
def test_edit_after_planning_never_replaces_active_pointer(tmp_path, mode):
    page = write_page(tmp_path / "Wiki")
    build(tmp_path)
    pointer = tmp_path / ".index" / "ACTIVE_INDEX"
    old_pointer = pointer.read_bytes()

    def editing_embed(texts):
        page.write_text(page.read_text() + "\nchanged during build\n")
        return embed384(texts)

    with pytest.raises(SnapshotError):
        build(tmp_path, mode=mode, embed=editing_embed)
    assert pointer.read_bytes() == old_pointer


def test_load_resolves_active_only_once(tmp_path, monkeypatch):
    import build_index as bi
    import obsidian_wiki.application.active_index_pointer as pointers
    write_page(tmp_path / "Wiki")
    outcome = build(tmp_path)
    expected = outcome.artifact.lance_dir
    calls = []

    def resolver(index_dir):
        calls.append(index_dir)
        if len(calls) > 1:
            raise AssertionError("ACTIVE_INDEX was re-resolved inside one load")
        return expected

    # Keep real repository/schema validation; only replace the pointer read.
    monkeypatch.setattr(pointers, "resolve_active_lance_dir", resolver)
    wi = bi.WikiIndex(tmp_path / ".index")
    wi.load()
    loaded = wi.get_loaded_manifest()
    assert loaded["build_id"] == outcome.build_id
    assert len(calls) == 1
    # Actual retrieval must use the pinned repository, not resolve anew.
    wi.search_fts_terms(["before65"], [], k=5)
    assert len(calls) == 1


def test_direct_hybrid_api_strict_gate_precedes_planner(tmp_path):
    import query
    page = write_page(tmp_path / "Wiki")
    saved = capture_wiki(tmp_path / "Wiki").to_json()
    page.write_text(page.read_text() + "\nnew bytes\n")
    wi = SimpleNamespace(
        index_dir=tmp_path / ".index",
        get_loaded_manifest=lambda: {"build_id": "test", "generation": 1,
                                     "wiki_snapshot": saved},
    )

    class NeverPlanner:
        def plan(self, *_args):
            pytest.fail("stale strict request reached the planner")

    with pytest.raises(FreshnessError) as exc:
        query.hybrid_search(
            wi, "query", NeverPlanner(), wiki_dir=tmp_path / "Wiki",
            freshness_policy="strict", enable_graph=False,
        )
    assert exc.value.exit_code == 1


def test_missing_provenance_is_unknown_not_fresh(tmp_path):
    import query
    write_page(tmp_path / "Wiki")
    wi = SimpleNamespace(index_dir=tmp_path / ".index", get_loaded_manifest=lambda: {})
    with pytest.raises(FreshnessError) as exc:
        query.hybrid_search(
            wi, "query", object(), wiki_dir=tmp_path / "Wiki",
            freshness_policy="strict", enable_graph=False,
        )
    assert exc.value.exit_code == 2


def test_build_capture_is_after_lock_acquire(tmp_path, monkeypatch):
    import build_index as bi
    import obsidian_wiki.application.index_build_service as service
    write_page(tmp_path / "Wiki")
    events = []
    real_acquire = service.BuildLock.acquire
    real_capture = bi.capture_wiki

    def acquire(lock, *args, **kwargs):
        result = real_acquire(lock, *args, **kwargs)
        events.append("lock")
        return result

    def capture(*args, **kwargs):
        events.append("capture")
        return real_capture(*args, **kwargs)

    monkeypatch.setattr(service.BuildLock, "acquire", acquire)
    monkeypatch.setattr(bi, "capture_wiki", capture)
    build(tmp_path)
    assert events.index("lock") < events.index("capture")


def test_long_lived_loaded_index_rechecks_current_wiki(tmp_path):
    import query
    page = write_page(tmp_path / "Wiki")
    build(tmp_path)
    wi = WikiIndex(tmp_path / ".index")
    wi.load()
    loaded_id = wi.get_loaded_manifest()["build_id"]
    # First request: real FTS/context, no model download needed.
    wi.search_vector = lambda *_args, **_kwargs: []
    wi.count_tokens = lambda text: max(1, len(text) // 4)
    planner = query.DefaultQueryPlanner(project_root=tmp_path, config={"rewrite": "off"})
    result = query.hybrid_search(
        wi, "before65", planner, wiki_dir=tmp_path / "Wiki",
        freshness_policy="strict", enable_graph=False, intent_override="lookup",
    )
    assert result.index_freshness["status"] == "fresh"
    page.write_text(page.read_text() + "\nmanual edit after first query\n")
    with pytest.raises(FreshnessError) as caught:
        query.hybrid_search(
            wi, "before65", planner, wiki_dir=tmp_path / "Wiki",
            freshness_policy="strict", enable_graph=False,
        )
    assert caught.value.exit_code == 1
    assert wi.get_loaded_manifest()["build_id"] == loaded_id


def test_edit_during_retrieval_is_rejected_before_return(tmp_path, monkeypatch):
    import query
    page = write_page(tmp_path / "Wiki")
    build(tmp_path)
    wi = WikiIndex(tmp_path / ".index")
    wi.load()
    wi.search_vector = lambda *_args, **_kwargs: []
    wi.count_tokens = lambda text: max(1, len(text) // 4)
    planner = query.DefaultQueryPlanner(project_root=tmp_path, config={"rewrite": "off"})
    real_retrieve = query._retrieve_for_plan

    def edit_during_retrieve(*args, **kwargs):
        page.write_text(page.read_text() + "\nquery-time edit\n")
        return real_retrieve(*args, **kwargs)

    monkeypatch.setattr(query, "_retrieve_for_plan", edit_during_retrieve)
    with pytest.raises(FreshnessError) as caught:
        query.hybrid_search(
            wi, "before65", planner, wiki_dir=tmp_path / "Wiki",
            freshness_policy="strict", enable_graph=False, intent_override="lookup",
        )
    assert caught.value.exit_code == 1
