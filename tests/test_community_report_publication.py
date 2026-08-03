import hashlib
import json
from pathlib import Path

import sys

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
