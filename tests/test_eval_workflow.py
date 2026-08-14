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


def test_eval_builds_ann_candidate_through_auto_policy(monkeypatch, tmp_path):
    """Dedicated Eval must opt into internal policy, not misuse a public force flag."""
    requested_modes = []

    class Planner:
        def plan(self, _query):
            return None

    class FakeIndex:
        def build(self, *_args, **_kwargs):
            return None

    def fake_build(root, _wiki, mode, full_rebuild):
        requested_modes.append((root.name, mode, full_rebuild))
        staged_wiki = root / "Wiki"
        staged_wiki.mkdir(parents=True, exist_ok=True)
        (staged_wiki / "page.md").write_text("fixture", encoding="utf-8")
        return FakeIndex(), staged_wiki, 0.01

    monkeypatch.setattr(run_eval, "_build", fake_build)
    monkeypatch.setattr(run_eval, "_active_benchmark_contract", lambda _wi: {})
    monkeypatch.setattr(run_eval, "DefaultQueryPlanner", lambda project_root: Planner())
    monkeypatch.setattr(run_eval.tracemalloc, "start", lambda: None)
    monkeypatch.setattr(run_eval.tracemalloc, "stop", lambda: None)
    monkeypatch.setattr(run_eval.tracemalloc, "get_traced_memory", lambda: (0, 0))
    monkeypatch.setattr(run_eval.statistics, "mean", lambda _values: 0.0)

    metrics, detail = run_eval.run_evaluation(
        tmp_path / "fixture-wiki", [], tmp_path / "work", 4096,
        build_ann=True, regression_pp=2.0,
    )

    assert requested_modes[:2] == [
        ("main", "exact", True),
        ("ann", "auto", True),
    ]
    assert metrics["index_benchmark"] == {"main": {}, "ann": {}}
    assert detail == []


def test_small_fixture_metric_and_decision_records_are_separate() -> None:
    """A 157-row functional observation cannot masquerade as scale evidence."""
    assert run_eval.FUNCTIONAL_FINAL_RETRIEVAL_METRIC == "functional_final_retrieval_ann_overlap_at_10"
    with pytest.raises(ValueError, match="candidate records"):
        run_eval.validate_candidate_decision_records(
            {"schema_version": 1, "records": []},
            {"evidence_schema_version": run_eval.EVIDENCE_SCHEMA_VERSION},
        )

    workflow = (SKILL_ROOT / ".github" / "workflows" / "eval.yml").read_text(encoding="utf-8")
    assert "model-backed-ann-decision" in workflow
    assert "--decision-evidence" in workflow
    assert "--init-baseline" not in workflow.split("model-backed-ann-decision:", 1)[1]


def test_scale_workflow_is_locked_and_reconciliation_is_an_always_run_gate() -> None:
    workflow = (SKILL_ROOT / ".github" / "workflows" / "eval.yml").read_text(encoding="utf-8")
    scale = workflow.split("issue41-scale-benchmark:", 1)[1].split("model-backed-ann-decision:", 1)[0]
    calibration = workflow.split("issue41-manual-calibration:", 1)[1].split("issue41-scale-benchmark:", 1)[0]
    assert "runs-on: ubuntu-latest" in scale
    assert "self-hosted" not in scale
    assert "ANN_SCALE_RUNNER_LABEL" not in scale
    assert "python -m pip install -r requirements.txt pytest" in scale
    assert '"numpy>=1.26"' not in scale
    assert "LanceDB 0.34.0 / NumPy 2.2.6 / PyArrow 25.0.0" in scale
    assert "--rows 77348" in scale
    assert "--dimensions 384" in scale
    assert "--max-probes 256" in scale
    assert "--ef-grid 30,50,75,100,150,200" in scale
    assert "--max-seconds 60" in scale
    assert "--calibrate" not in scale
    assert "--approved-static-cap" in scale
    assert "approved_ann_calibration.json" in scale
    assert "--calibrate" in calibration
    assert "--calibration-output .review-tmp/issue41-scale/ann-calibration.json" in calibration
    assert "--calibration-batch-output .review-tmp/issue41-scale/ann-calibration-batch-spike.json" in calibration
    assert "ann-calibration.json" in calibration
    assert "issue41-ann-calibration" in calibration
    assert "issue41-ann-calibration-batch-spike" in calibration
    assert "if: ${{ always() }}" in scale
    assert "--error-output .review-tmp/issue41-scale/index-benchmark-error.json" in scale
    assert "if: success()" in scale
    assert "index-benchmark-error.json" in scale
    assert "issue41-index-benchmark-error" in scale
    assert "Upload rejected issue #41 comparator error evidence" in scale

    reconciliation = workflow.split("reconcile-ann-decision:", 1)[1]
    assert "if: ${{ always() }}" in reconciliation
    assert "test-and-eval" in reconciliation
    assert "issue41-scale-benchmark" in reconciliation
    assert "model-backed-ann-decision" in reconciliation
    assert "eval/reconcile_ann_gate.py" in reconciliation
    assert "Architecture (ubuntu-latest, Python 3.10)" in reconciliation
    assert "Architecture (windows-latest, Python 3.13)" in reconciliation


def test_reconciliation_cli_is_fail_closed_for_missing_jobs_and_artifacts() -> None:
    reconciliation = SKILL_ROOT / "eval" / "reconcile_ann_gate.py"
    source = reconciliation.read_text(encoding="utf-8")
    assert "validate_evidence" in source
    assert "validate_candidate_decision_records" in source
    assert "all_required_jobs_numeric_success" in source
    assert "!= \"success\"" in source


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
    monkeypatch.setattr(
        run_eval,
        "run_graph_contract_evaluation",
        lambda *args, **kwargs: pytest.fail(
            "graph evaluation must not run after a primary citation failure"),
    )
    argv = [
        "run_eval.py", "--wiki", str(wiki), "--queries", str(queries),
        "--baselines", str(baseline), "--work-dir", str(tmp_path / "work"),
    ]
    if init_baseline:
        argv.append("--init-baseline")
    monkeypatch.setattr(sys, "argv", argv)

    assert run_eval.main() == 1
    assert not baseline.exists()


def test_eval_valid_primary_output_runs_graph_contract(monkeypatch, tmp_path):
    """Valid primary evidence must continue through the production graph gate."""
    wiki = tmp_path / "Wiki"
    wiki.mkdir()
    queries = tmp_path / "queries.jsonl"
    queries.write_text("", encoding="utf-8")
    baseline = tmp_path / "missing-baseline.json"
    output_dir = tmp_path / "eval-output"
    output_dir.mkdir()
    graph_calls = []
    metrics = {
        "quality": {
            "citation_path_contract_violation_count": 0,
            "citation_paths": [],
        },
    }

    def graph_evaluation(*args, **kwargs):
        graph_calls.append((args, kwargs))
        return ({"failures": []}, [])

    monkeypatch.setattr(run_eval, "HERE", output_dir)
    monkeypatch.setattr(run_eval, "run_evaluation", lambda *args, **kwargs: (metrics, []))
    monkeypatch.setattr(run_eval, "run_graph_contract_evaluation", graph_evaluation)
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--wiki", str(wiki), "--queries", str(queries),
        "--baselines", str(baseline), "--work-dir", str(tmp_path / "work"),
    ])

    assert run_eval.main() == 0
    assert len(graph_calls) == 1
    assert metrics["graph_contract"] == {"failures": []}
