import hashlib
import json
from pathlib import Path

import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class _Counter:
    identity = "test-token-counter/v1"

    def count(self, text):
        return len(text.split())


def _write_graph_fixture(root: Path) -> None:
    wiki = root / "Wiki"
    wiki.mkdir()
    pages = []
    for name in ("a.md", "b.md"):
        path = wiki / name
        path.write_text(f"---\ntitle: {name}\n---\n\n{name}", encoding="utf-8")
        pages.append(str(path.resolve()))
    (root / ".index").mkdir()
    (root / ".index" / "graph.json").write_text(json.dumps({
        "nodes": [{"id": page} for page in pages],
        "edges": [{"source": pages[0], "target": pages[1], "weight": 1.0, "signals": ["direct_link"]}],
        "communities": [pages],
    }), encoding="utf-8")


def _service(root: Path):
    from obsidian_wiki.application.community_report_service import CommunityReportService
    from obsidian_wiki.infrastructure.filesystem_community_reports import FilesystemCommunityReportStore
    from obsidian_wiki.infrastructure.filesystem_graph_snapshot import FilesystemGraphSnapshot

    return CommunityReportService(
        FilesystemCommunityReportStore(root / ".index"), FilesystemGraphSnapshot(root), _Counter()
    )


def test_staged_publication_activates_reopened_v2_set(tmp_path):
    """Only a complete reopened schema-v2 set becomes the active pointer target."""
    _write_graph_fixture(tmp_path)
    manifest = _service(tmp_path).build()
    pointer = json.loads((tmp_path / ".index" / "ACTIVE_COMMUNITY_REPORTS").read_text(encoding="utf-8"))
    staged = tmp_path / ".index" / pointer["active_build"]

    assert staged.parent == tmp_path / ".index" / "community_report_builds"
    assert (staged / "reports.jsonl").is_file()
    assert (staged / "manifest.json").is_file()
    assert manifest.report_schema_version == 2
    outcome = _service(tmp_path).retrieve()
    assert outcome.status.value == "community_reports_fresh"
    assert outcome.reports[0].content_hash == hashlib.sha256(outcome.reports[0].text.encode()).hexdigest()


def test_reader_rejects_torn_or_escaped_active_pointer(tmp_path):
    """Unsafe pointers are rejected before opening any report text and remain byte-stable."""
    _write_graph_fixture(tmp_path)
    service = _service(tmp_path)
    service.build()
    pointer_path = tmp_path / ".index" / "ACTIVE_COMMUNITY_REPORTS"
    prior = pointer_path.read_bytes()
    for payload in ({"active_build": "../escape"}, {"active_build": "/tmp/escape"}, {"wrong": "shape"}):
        pointer_path.write_text(json.dumps(payload), encoding="utf-8")
        outcome = service.retrieve()
        assert outcome.status.value == "community_reports_missing"
        assert outcome.reports == ()
    pointer_path.write_bytes(prior)
    assert service.retrieve().status.value == "community_reports_fresh"


def test_invalid_staged_set_preserves_previous_active_pointer(tmp_path):
    from dataclasses import replace

    from obsidian_wiki.application.community_report_service import CommunityReportService
    from obsidian_wiki.infrastructure.filesystem_community_reports import FilesystemCommunityReportStore
    from obsidian_wiki.infrastructure.filesystem_graph_snapshot import FilesystemGraphSnapshot

    _write_graph_fixture(tmp_path)
    service = _service(tmp_path)
    service.build()
    pointer_path = tmp_path / ".index" / "ACTIVE_COMMUNITY_REPORTS"
    prior = pointer_path.read_bytes()
    class _BadStageStore(FilesystemCommunityReportStore):
        def read_staged(self, build_id):
            reports, manifest = super().read_staged(build_id)
            return reports, replace(manifest, report_schema_version=1)

    bad_store = _BadStageStore(tmp_path / ".index")
    try:
        CommunityReportService(bad_store, FilesystemGraphSnapshot(tmp_path), _Counter()).build()
    except RuntimeError:
        pass
    else:
        raise AssertionError("an unsupported staged set must not activate")
    assert pointer_path.read_bytes() == prior
    assert service.retrieve().status.value == "community_reports_fresh"
    assert list((tmp_path / ".index" / "community_report_builds").glob("*/.failed"))


def test_reader_rejects_malformed_or_stale_records_without_text(tmp_path):
    _write_graph_fixture(tmp_path)
    service = _service(tmp_path)
    service.build()
    pointer = json.loads((tmp_path / ".index" / "ACTIVE_COMMUNITY_REPORTS").read_text())
    staged = tmp_path / ".index" / pointer["active_build"]
    reports_path = staged / "reports.jsonl"
    original = reports_path.read_text()
    reports_path.write_text("{not json}\n", encoding="utf-8")
    assert service.retrieve().reports == ()
    assert service.retrieve().status.value == "community_reports_missing"
    reports_path.write_text(original, encoding="utf-8")
    manifest_path = staged / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stale_reason"] = "graph_published"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    outcome = service.retrieve()
    assert outcome.status.value == "community_reports_stale"
    assert outcome.reports == ()


def test_successful_producers_invalidate_only_after_publication(tmp_path, monkeypatch):
    """Graph/index publication marks the active set stale; producer failures do not."""
    _write_graph_fixture(tmp_path)
    service = _service(tmp_path)
    service.build()
    pointer = json.loads((tmp_path / ".index" / "ACTIVE_COMMUNITY_REPORTS").read_text())
    manifest_path = tmp_path / ".index" / pointer["active_build"] / "manifest.json"

    import build_graph

    monkeypatch.setattr(build_graph, "render_html", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["build_graph.py", str(tmp_path)])
    build_graph.main()
    graph_stale = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert graph_stale["stale_producer"] == "build_graph"
    assert graph_stale["stale_reason"] == "graph_published"
    assert graph_stale["stale_at"]
    assert (tmp_path / ".index" / "graph.json").is_file()

    graph_state = manifest_path.read_bytes()
    monkeypatch.setattr(build_graph, "build_graph", lambda _wiki: (_ for _ in ()).throw(RuntimeError("graph failed")))
    with pytest.raises(RuntimeError, match="graph failed"):
        build_graph.main()
    assert manifest_path.read_bytes() == graph_state

    import build_index
    from obsidian_wiki.application import index_build_service

    class _SuccessfulIndexBuild:
        def __init__(self, *_args, **_kwargs):
            pass

        def build(self, *_args, **_kwargs):
            # build_storage_contract 读 manifest_path 取 generation；mock 场景无真实
            # manifest（FileNotFoundError → generation 0），post-commit 仍执行。
            return type("_Artifact", (), {"manifest_path": Path("nonexistent")})()

    monkeypatch.setattr(index_build_service, "IndexBuildService", _SuccessfulIndexBuild)
    build_index.build_storage_contract(tmp_path / "Wiki", tmp_path / ".index", embed=lambda _texts: [])
    first_index_stale = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert first_index_stale["stale_producer"] == "build_index"
    assert first_index_stale["stale_reason"] == "index_published"
    first_index_bytes = manifest_path.read_bytes()

    build_index.build_storage_contract(tmp_path / "Wiki", tmp_path / ".index", embed=lambda _texts: [])
    assert manifest_path.read_bytes() != first_index_bytes

    class _FailedIndexBuild(_SuccessfulIndexBuild):
        def build(self, *_args, **_kwargs):
            raise RuntimeError("index failed")

    monkeypatch.setattr(index_build_service, "IndexBuildService", _FailedIndexBuild)
    failed_index_state = manifest_path.read_bytes()
    with pytest.raises(RuntimeError, match="index failed"):
        build_index.build_storage_contract(tmp_path / "Wiki", tmp_path / ".index", embed=lambda _texts: [])
    assert manifest_path.read_bytes() == failed_index_state
