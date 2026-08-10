"""Static safety assertions for the fail-closed PR baseline workflow guard."""
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT / "eval"))

from models import ContextBundle, ContextItem  # noqa: E402
import run_eval  # noqa: E402
from run_eval import _citation_violations  # noqa: E402


def test_baseline_pr_guard_fetches_history_and_fails_closed():
    workflow = (SKILL_ROOT / ".github" / "workflows" / "eval.yml").read_text(
        encoding="utf-8")
    assert "fetch-depth: 0" in workflow
    assert "git cat-file -e \"$base^{commit}\"" in workflow
    assert "git cat-file -e \"$head^{commit}\"" in workflow
    assert 'changed_files="$(git diff --name-only "$base" "$head")"' in workflow
    assert "git diff --name-only" not in workflow.split('changed_files="$(git diff --name-only "$base" "$head")"', 1)[1]


def _bundle(path, *, reason="rrf", context_text=None):
    item = ContextItem(page_id="page-a", path=path, title="Page A",
                       inclusion_reason=reason, scope="chunk", text="body")
    bundle = ContextBundle(query="q", mode="snippet", items=[item])
    bundle.context_text = f"[来源: {path}]" if context_text is None else context_text
    return bundle


@pytest.mark.parametrize("path", [
    "/tmp/project/Wiki/page.md",
    r"D:\project\Wiki\page.md",
    r"Wiki\page.md",
    "page.md",
    "Wiki/../Raw/secret.md",
    "Wiki/./page.md",
    "Wiki//page.md",
])
def test_eval_rejects_noncanonical_citation_paths(path):
    """issue #43: the publication gate must catch every non-canonical shape."""
    assert _citation_violations(_bundle(path))


def test_eval_accepts_canonical_citation_path():
    assert _citation_violations(_bundle("Wiki/page.md")) == []


def test_eval_rejects_citation_token_missing_from_context_text():
    bundle = _bundle("Wiki/page.md", context_text="### Page A\nbody")
    violations = _citation_violations(bundle)
    assert violations
    assert violations[0]["reasons"] == ["citation_token_missing_from_context_text"]


def test_eval_exempts_community_report_rows():
    bundle = _bundle("community-3", reason="global_community_report",
                     context_text="### Report\nbody")
    assert _citation_violations(bundle) == []


@pytest.mark.parametrize("init_baseline", [False, True])
def test_eval_citation_gate_precedes_missing_or_initial_baseline(monkeypatch, tmp_path, init_baseline):
    """Unsafe evidence cannot become a new baseline or pass without one."""
    wiki = tmp_path / "Wiki"
    wiki.mkdir()
    queries = tmp_path / "queries.jsonl"
    queries.write_text("", encoding="utf-8")
    baseline = tmp_path / "baselines.json"
    output_dir = tmp_path / "eval-output"
    output_dir.mkdir()
    metrics = {
        "quality": {
            "citation_path_contract_violation_count": 1,
            "citation_paths": [{"path": "Wiki/../Raw/secret.md"}],
        },
    }
    monkeypatch.setattr(run_eval, "HERE", output_dir)
    monkeypatch.setattr(run_eval, "run_evaluation", lambda *args, **kwargs: (metrics, []))
    argv = [
        "run_eval.py", "--wiki", str(wiki), "--queries", str(queries),
        "--baselines", str(baseline), "--work-dir", str(tmp_path / "work"),
    ]
    if init_baseline:
        argv.append("--init-baseline")
    monkeypatch.setattr(sys, "argv", argv)

    assert run_eval.main() == 1
    assert not baseline.exists()
