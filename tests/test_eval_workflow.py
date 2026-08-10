"""Static safety assertions for the fail-closed PR baseline workflow guard."""
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT / "eval"))

from models import ContextBundle, ContextItem  # noqa: E402
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
