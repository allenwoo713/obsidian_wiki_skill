"""Static safety assertions for the fail-closed PR baseline workflow guard."""
from pathlib import Path


def test_baseline_pr_guard_fetches_history_and_fails_closed():
    workflow = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "eval.yml").read_text(
        encoding="utf-8")
    assert "fetch-depth: 0" in workflow
    assert "git cat-file -e \"$base^{commit}\"" in workflow
    assert "git cat-file -e \"$head^{commit}\"" in workflow
    assert 'changed_files="$(git diff --name-only "$base" "$head")"' in workflow
    assert "git diff --name-only" not in workflow.split('changed_files="$(git diff --name-only "$base" "$head")"', 1)[1]
