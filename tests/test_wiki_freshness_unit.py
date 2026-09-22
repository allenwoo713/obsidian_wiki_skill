from __future__ import annotations
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from obsidian_wiki.application import wiki_freshness as f


class WikiFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "vault"
        self.wiki = self.root / "Wiki"
        self.wiki.mkdir(parents=True)
        self.page = self.wiki / "a.md"
        self.page.write_bytes(b"---\ntitle: A\ntype: concept\n---\nold\n")
        self.saved = f.capture_wiki(self.wiki).to_json()

    def check(self):
        return f.inspect_snapshot(self.wiki, self.saved)

    def test_unchanged_and_touch_are_fresh(self):
        self.assertEqual(self.check()["status"], "fresh")
        os.utime(self.page, None)
        self.assertEqual(self.check()["status"], "fresh")

    def test_same_size_same_mtime_edit_is_stale(self):
        st = self.page.stat()
        self.page.write_bytes(self.page.read_bytes().replace(b"old", b"new"))
        os.utime(self.page, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertEqual(self.check()["modified"], ["a.md"])

    def test_new_nested_page(self):
        (self.wiki / "comparisons").mkdir()
        (self.wiki / "comparisons" / "B.md").write_text("new")
        self.assertEqual(self.check()["added"], ["comparisons/B.md"])

    def test_deleted_page(self):
        self.page.unlink()
        self.assertEqual(self.check()["deleted"], ["a.md"])

    def test_rename_is_add_and_delete(self):
        self.page.rename(self.wiki / "renamed.md")
        report = self.check()
        self.assertEqual(report["added"], ["renamed.md"])
        self.assertEqual(report["deleted"], ["a.md"])

    def test_dotgraph_is_excluded(self):
        (self.wiki / ".graph").mkdir()
        (self.wiki / ".graph" / "output.md").write_text("generated")
        self.assertEqual(self.check()["status"], "fresh")

    def test_non_frontmatter_file_is_watched(self):
        plain = self.wiki / "plain.md"
        plain.write_text("plain")
        saved = f.capture_wiki(self.wiki).to_json()
        plain.write_text("---\ntitle: New\n---\nplain\n")
        self.assertEqual(f.inspect_snapshot(self.wiki, saved)["status"], "stale")

    def test_crlf_raw_hash_and_parser_newlines(self):
        payload = b"---\r\ntitle: A\r\n---\r\nbody\r\n"
        self.page.write_bytes(payload)
        snap = f.capture_wiki(self.wiki, retain_bytes=True)
        self.assertEqual(snap.hashes["a.md"], hashlib.sha256(payload).hexdigest())
        self.assertNotIn("\r", f.decode_markdown(snap.raw["a.md"]))

    def test_retained_bytes_are_the_input_not_a_later_read(self):
        snap = f.capture_wiki(self.wiki, retain_bytes=True)
        self.page.write_text("replacement")
        self.assertIn(b"old", snap.raw["a.md"])
        self.assertEqual(snap.hashes["a.md"], hashlib.sha256(snap.raw["a.md"]).hexdigest())
        self.assertEqual(f.inspect_snapshot(self.wiki, snap.to_json())["status"], "stale")

    def test_missing_or_invalid_legacy_snapshot_is_unknown(self):
        for value in (None, {}, [], {"schema_version": 1}):
            with self.subTest(value=value):
                self.assertEqual(f.inspect_snapshot(self.wiki, value)["status"], "unknown")

    def test_digest_tampering_and_schema_rejected(self):
        for key, value in (("tree_sha256", "0" * 64), ("schema_version", True),
                           ("schema_version", 999), ("scope", "other")):
            saved = dict(self.saved)
            saved[key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(f.inspect_snapshot(self.wiki, saved)["status"], "unknown")

    def test_path_traversal_rejected_even_with_valid_tree_digest(self):
        saved = dict(self.saved)
        saved["files"] = {"../escape.md": "a" * 64}
        saved["tree_sha256"] = f.tree_digest(saved["files"])
        self.assertEqual(f.inspect_snapshot(self.wiki, saved)["status"], "unknown")

    def test_unreadable_file_is_unknown_not_empty_or_fresh(self):
        with patch.object(f, "_read_file", side_effect=PermissionError("denied")):
            self.assertEqual(self.check()["status"], "unknown")

    def test_directory_error_is_not_swallowed(self):
        def walk(*args, **kwargs):
            kwargs["onerror"](PermissionError("directory denied"))
            yield
        with patch.object(f.os, "walk", side_effect=walk):
            self.assertEqual(self.check()["status"], "unknown")

    def test_membership_change_during_scan_is_unknown(self):
        paths = f._paths(self.wiki)
        with patch.object(f, "_paths", side_effect=[paths, ()]):
            self.assertEqual(self.check()["status"], "unknown")

    def test_mid_read_change_is_unknown(self):
        with patch.object(f, "_read_file", side_effect=f.SnapshotError("changed")):
            self.assertEqual(self.check()["status"], "unknown")

    def test_relocated_vault_is_stale(self):
        other = self.root.parent / "copied" / "Wiki"
        shutil.copytree(self.wiki, other)
        report = f.inspect_snapshot(other, self.saved)
        self.assertEqual(report["status"], "stale")
        self.assertIn("wiki_root_changed", report["reasons"])

    def test_publish_guard_rejects_changed_input(self):
        self.page.write_text("changed")
        with self.assertRaises(f.SnapshotError):
            f.require_current(self.wiki, self.saved)

    def test_manifest_reuses_page_hash_and_rejects_mismatch(self):
        manifest = {"pages": [{"path": str(self.page),
                               "sha256": self.saved["files"]["a.md"]}]}
        f.attach_snapshot(manifest, self.wiki, self.saved)
        self.assertEqual(manifest["wiki_snapshot"], self.saved)
        manifest["pages"][0]["sha256"] = "0" * 64
        with self.assertRaises(f.SnapshotError):
            f.attach_snapshot(manifest, self.wiki, self.saved)

    def test_synthetic_plan_does_not_claim_provenance(self):
        manifest = {}
        f.attach_snapshot(manifest, self.wiki, None)
        self.assertIsNone(manifest["wiki_snapshot"])

    def test_exit_codes_and_unknown_priority(self):
        for state, expected in (("fresh", 0), ("stale", 1), ("unknown", 2)):
            self.assertEqual(f.diagnostic_exit_code({"index": {"status": state}}), expected)
        self.assertEqual(f.diagnostic_exit_code({
            "index": {"status": "stale"}, "graph": {"status": "unknown"}}), 2)

    def test_graph_and_index_are_independent(self):
        graph = self.saved
        self.page.write_text("new revision")
        rebuilt_index = f.capture_wiki(self.wiki).to_json()
        now = f.capture_wiki(self.wiki)
        self.assertEqual(f.compare_snapshot(rebuilt_index, now)["status"], "fresh")
        self.assertEqual(f.compare_snapshot(graph, now)["status"], "stale")

    def test_case_and_unicode_keys_are_preserved(self):
        name = "资料测试.md"
        (self.wiki / name).write_text("content", encoding="utf-8")
        snap = f.capture_wiki(self.wiki)
        self.assertIn(name, snap.hashes)
        self.assertEqual(f.tree_digest({"Z.md": "a", "a.md": "b"}),
                         f.tree_digest({"a.md": "b", "Z.md": "a"}))

    def test_symlink_is_explicitly_rejected(self):
        link = self.wiki / "alias.md"
        try:
            link.symlink_to(self.page)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable for this test user")
        self.assertEqual(self.check()["status"], "unknown")

    def test_full_page_guard_hashes_rendered_bytes(self):
        from types import SimpleNamespace
        manifest = {"pages": [{"page_id": "a", "path": str(self.page),
                               "sha256": self.saved["files"]["a.md"]}]}
        repo = f.GuardedContextRepository(
            SimpleNamespace(read_page=lambda _: "wrong delegate"), manifest, self.wiki)
        self.page.write_bytes(self.page.read_bytes().replace(b"old", b"new"))
        self.assertEqual(repo.read_page("a"), "new")
        self.assertEqual(repo.reports["a"]["status"], "stale")
        self.page.write_text("a later edit")
        self.assertEqual(repo.read_page("a"), "new")

    def test_policy_and_graph_absence(self):
        reports = {"index": {"status": "stale"}}
        f.enforce_freshness(reports, "warn")
        f.enforce_freshness(reports, "allow")
        with self.assertRaises(f.FreshnessError) as caught:
            f.enforce_freshness(reports, "strict")
        self.assertEqual(caught.exception.exit_code, 1)
        report = f.collect_reports(self.wiki, {"wiki_snapshot": self.saved})
        self.assertEqual(report["graph"]["status"], "not_built")
        self.assertEqual(f.diagnostic_exit_code(report), 0)
        self.assertEqual(f.collect_reports(
            self.wiki, {"wiki_snapshot": self.saved}, graph_error="broken"
        )["graph"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
