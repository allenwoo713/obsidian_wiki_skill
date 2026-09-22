"""Real CLI acceptance for #65. Run after applying the proposal; never use a real vault.

只操作仓库 `.review-tmp` 下自动生成的临时 vault，不接受用户真实 vault 作为测试目标。
走真实 build_index.py / build_graph.py / query.py / update_wiki.py，覆盖：默认警告、
严格阻断、显式允许、输出文件契约、双向独立重建、增删改名、Raw 更新提示，以及
修复后新章节的正文与 evidence.section_path。
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per subprocess timeout; failure is not skipped")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    scratch = repo / ".review-tmp"
    scratch.mkdir(exist_ok=True)
    results = []

    with tempfile.TemporaryDirectory(prefix="issue65-e2e-", dir=scratch) as directory:
        root = Path(directory)
        wiki = root / "Wiki"
        wiki.mkdir()
        (root / "Raw" / "sources").mkdir(parents=True)
        for n in range(24):
            body = "\n".join(
                f"## Topic {section}\nDocument {n} engineering calibration protocol "
                f"range {n + section} procedure. " * 8
                for section in range(8)
            )
            (wiki / f"page-{n:02d}.md").write_text(
                f"---\ntitle: Document {n}\ntype: concept\nsources: []\n---\n{body}\n",
                encoding="utf-8",
            )
        target = wiki / "page-00.md"

        def run(script, extra=(), expected=0):
            command = [args.python, str(repo / "scripts" / script), str(root), *map(str, extra)]
            p = subprocess.run(command, cwd=repo, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=args.timeout)
            if p.returncode != expected:
                raise AssertionError(json.dumps({
                    "command": command, "expected": expected, "actual": p.returncode,
                    "stdout": p.stdout[-6000:], "stderr": p.stderr[-6000:],
                }, ensure_ascii=False, indent=2))
            results.append({"script": script, "arguments": list(map(str, extra)),
                            "exit_code": p.returncode})
            return p

        def check(expected):
            return json.loads(run("check_index_staleness.py", ["--json"], expected).stdout)

        def query(flags=(), expected=0):
            output = root / "query.json"
            output.unlink(missing_ok=True)
            run("query.py", [
                "needleissue65updated", "--rewrite", "off", "--intent", "lookup",
                "--json", "--out", output, *flags,
            ], expected)
            payload = json.loads(output.read_text(encoding="utf-8"))
            assert isinstance(payload["text"], list)
            assert isinstance(payload["images"], list)
            return payload

        run("build_index.py", ["--build-mode", "snapshot"])
        run("build_graph.py")
        assert check(0)["components"]["index"]["status"] == "fresh"
        assert query(["--strict-freshness"])["index_freshness"]["status"] == "fresh"

        target.write_text(target.read_text(encoding="utf-8") +
                          "\n## Issue65NewSection\nneedleissue65updated new content.\n",
                          encoding="utf-8")
        state = check(1)["components"]
        assert state["index"]["status"] == state["graph"]["status"] == "stale"
        assert query()["index_freshness"]["status"] == "stale"
        blocked = query(["--strict-freshness"], expected=1)
        assert blocked["text"] == blocked["images"] == []
        allowed = query(["--allow-stale"])
        assert allowed["index_freshness"]["acknowledged"]
        assert allowed["index_freshness"]["status"] == "stale"

        # Rebuilding only the text index must not certify the old graph.
        run("build_index.py", ["--build-mode", "snapshot"])
        state = check(1)["components"]
        assert state["index"]["status"] == "fresh" and state["graph"]["status"] == "stale"
        query(["--strict-freshness"], expected=1)
        run("build_graph.py")
        check(0)
        recalled = query(["--strict-freshness"])
        assert any("needleissue65updated" in item["text"] for item in recalled["text"])
        assert any("Issue65NewSection" in str(hit["section_path"])
                   for item in recalled["text"] for hit in item["evidence"])

        # The inverse direction is equally important.
        target.write_text(target.read_text(encoding="utf-8") + "\nsecond revision\n",
                          encoding="utf-8")
        run("build_graph.py")
        state = check(1)["components"]
        assert state["index"]["status"] == "stale" and state["graph"]["status"] == "fresh"
        run("build_index.py", ["--build-mode", "incremental"])
        check(0)

        extra = wiki / "extra.md"
        extra.write_text("---\ntitle: extra\n---\nnew\n", encoding="utf-8")
        assert "extra.md" in check(1)["components"]["index"]["added"]
        extra.unlink()
        check(0)
        renamed = wiki / "renamed.md"
        target.rename(renamed)
        state = check(1)["components"]["index"]
        assert "page-00.md" in state["deleted"] and "renamed.md" in state["added"]
        renamed.rename(target)
        check(0)

        # Actual current CLI is default-write / --dry-run, not --apply.
        source = root / "Raw" / "sources" / "issue65.txt"
        source.write_text("A new source requiring a Wiki update.", encoding="utf-8")
        pointer = (root / ".index" / "ACTIVE_INDEX").read_bytes()
        run("update_wiki.py", ["--dry-run"])
        assert (root / ".index" / "ACTIVE_INDEX").read_bytes() == pointer
        check(0)
        update = run("update_wiki.py")
        assert "index stale" in (update.stdout + update.stderr).lower()
        assert check(1)["components"]["index"]["status"] == "stale"

        # Isolate graph UNKNOWN from an already-stale text index.
        run("build_index.py", ["--build-mode", "snapshot"])
        run("build_graph.py")
        check(0)
        # Corruption is UNKNOWN=2, not a clean or merely stale result.
        (root / ".index" / "graph.json").write_bytes(b"{broken")
        assert check(2)["components"]["graph"]["status"] == "unknown"
        query(["--strict-freshness"], expected=2)

    result = {"status": "pass", "commands": results}
    output = scratch / "issue65-e2e-results.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {output}")


if __name__ == "__main__":
    main()
