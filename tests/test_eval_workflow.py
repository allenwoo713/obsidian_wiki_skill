"""Static safety assertions for the fail-closed PR baseline workflow guard."""
import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
APPROVED_CALIBRATION = SKILL_ROOT / "eval" / "approved_ann_calibration.json"
sys.path.insert(0, str(SKILL_ROOT / "eval"))

from models import ChunkHit, ContextBundle, ContextItem  # noqa: E402
import run_eval  # noqa: E402
from run_eval import _citation_violations  # noqa: E402


def _candidate_comparator() -> dict:
    return {
        "evidence_schema_version": run_eval.EVIDENCE_SCHEMA_VERSION,
        "corpus": {"sha256": "c" * 64},
        "queries": {"sha256": "q" * 64},
    }


def _candidate_record(candidate: str, query_ef: int, ordinal: int) -> dict:
    record = {
        "candidate": candidate,
        "query_ef": query_ef,
        "head_sha": "head",
        "pr_head_sha": "pr-head",
        "actions_merge_checkout_sha": "merge-head",
        "environment": {
            "python": "3.13", "platform": "test", "lancedb": "0.34.0",
            "numpy": "2.2.6", "pyarrow": "25.0.0",
        },
        "corpus_sha256": "c" * 64,
        "queries_sha256": "q" * 64,
        "candidate_run_id": "pending",
        "candidate_index": {
            "build_id": f"build-{ordinal}",
            "manifest_sha256": f"{ordinal:064x}",
            "root_identity": f"candidate-{ordinal}",
        },
        "applied_policy": {"candidate": candidate, "query_ef": query_ef},
        "hybrid_invocation": {
            "entrypoint": "query.hybrid_search",
            "trace_id": "pending",
            "digest": "pending",
            "query_count": 1,
            "traces": [{
                "ordinal": 0,
                "query_sha256": f"{ordinal + 10:064x}",
                "result_sha256": f"{ordinal + 20:064x}",
                "latency_s": 0.01,
            }],
        },
        "result_sha256": "pending",
        "input_binding": {
            "fixture_sha256": "f" * 64,
            "evaluation_queries_sha256": "e" * 64,
            "comparator_corpus_sha256": "c" * 64,
            "comparator_queries_sha256": "q" * 64,
        },
        "final_retrieval": {
            "retrieval": {
                run_eval.FUNCTIONAL_FINAL_RETRIEVAL_METRIC: 1.0,
                "result_payload": [{"binding": ordinal, "pages": ["page-a"]}],
            },
            "page": {"page_recall_at_5": 1.0},
            "evidence": {"evidence_recall_at_10": 1.0},
            "mrr": {"mrr_at_10": 1.0},
            "latency": {"samples_s": [0.01], "p50_s": 0.01, "p95_s": 0.01},
            "context": {"overflow_count": 0, "budget_violation_count": 0},
            "citation": {"violation_count": 0},
            "graph": {"validated_count": 0, "unsupported_count": 0},
            "non_regression": {"baseline_refresh": False, "failures": []},
        },
    }
    traces = record["hybrid_invocation"]["traces"]
    run_id = run_eval._stable_json_digest({
        "candidate": candidate,
        "query_ef": query_ef,
        "build_id": record["candidate_index"]["build_id"],
        "traces": traces,
    })
    record["candidate_run_id"] = run_id
    record["hybrid_invocation"]["trace_id"] = run_eval._stable_json_digest({
        "run_id": run_id, "entrypoint": "query.hybrid_search",
    })
    record["hybrid_invocation"]["digest"] = run_eval._stable_json_digest({
        "run_id": run_id, "traces": traces,
    })
    record["result_sha256"] = run_eval._stable_json_digest({
        "run_id": run_id, "final_retrieval": record["final_retrieval"],
    })
    return record


def _candidate_packet() -> dict:
    records = [
        _candidate_record(candidate, query_ef, ordinal)
        for ordinal, (candidate, query_ef) in enumerate(
            (candidate, query_ef)
            for candidate in run_eval.CANDIDATES
            for query_ef in run_eval.DECISION_EF_GRID
        )
    ]
    return {
        "schema_version": run_eval.FINAL_RETRIEVAL_DECISION_SCHEMA_VERSION,
        "head_sha": "head",
        "pr_head_sha": "pr-head",
        "actions_merge_checkout_sha": "merge-head",
        "environment": {
            "python": "3.13", "platform": "test", "lancedb": "0.34.0",
            "numpy": "2.2.6", "pyarrow": "25.0.0",
        },
        "comparator_schema_version": run_eval.EVIDENCE_SCHEMA_VERSION,
        "records": records,
    }


def test_committed_ann_calibration_reference_is_a_complete_approved_static_binding():
    approval = json.loads(APPROVED_CALIBRATION.read_text(encoding="utf-8"))

    assert approval["schema_version"] == 1
    assert approval["static_cap_seconds"] == 180
    assert approval["omp_threads"] == 2
    assert approval["calibration_rule_version"] == "omp-median-mad-v1"
    assert approval["calibration_sha256"] == "ddf6741749623400db007b2b0f1480e6378df332b52138bb00a3258b10465034"
    assert approval["calibration_source_head_sha"] == "e872969f8a138e539ab88374703247ec01bc48f6"


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
            {
                "schema_version": run_eval.FINAL_RETRIEVAL_DECISION_SCHEMA_VERSION,
                "comparator_schema_version": run_eval.EVIDENCE_SCHEMA_VERSION,
                "records": [],
            },
            {"evidence_schema_version": run_eval.EVIDENCE_SCHEMA_VERSION},
        )

    workflow = (SKILL_ROOT / ".github" / "workflows" / "eval.yml").read_text(encoding="utf-8")
    assert "model-backed-ann-decision" in workflow
    assert "--decision-evidence" in workflow
    assert "--init-baseline" not in workflow.split("model-backed-ann-decision:", 1)[1]
    model_job = workflow.split("model-backed-ann-decision:", 1)[1].split("reconcile-ann-decision:", 1)[0]
    assert "candidate-hybrid-ann-decision.json" in model_job
    assert "--validate-candidate-hybrid-evidence" in model_job
    assert "GITHUB_PR_HEAD_SHA: ${{ github.event.pull_request.head.sha }}" in model_job
    assert "if: success()" in model_job
    assert "candidate-hybrid-ann-decision-error" in model_job


@pytest.mark.parametrize(
    ("shared_field", "message"),
    [
        ("candidate_run_id", "candidate run identity"),
        ("candidate_index", "candidate index identity"),
        ("hybrid_invocation", "hybrid invocation identity"),
        ("result_sha256", "candidate result digest"),
        ("final_retrieval", "candidate result payload"),
    ],
)
def test_candidate_hybrid_packet_rejects_shared_candidate_provenance(
    shared_field: str, message: str,
) -> None:
    packet = _candidate_packet()
    first, second = packet["records"][:2]
    second[shared_field] = first[shared_field]

    with pytest.raises(ValueError, match=message):
        run_eval.validate_candidate_decision_records(packet, _candidate_comparator())


@pytest.mark.parametrize(
    "missing_field",
    ["retrieval", "page", "evidence", "mrr", "latency", "context", "citation", "graph", "non_regression"],
)
def test_candidate_hybrid_packet_requires_complete_candidate_observations(
    missing_field: str,
) -> None:
    packet = _candidate_packet()
    del packet["records"][0]["final_retrieval"][missing_field]

    with pytest.raises(ValueError, match="candidate-specific final retrieval"):
        run_eval.validate_candidate_decision_records(packet, _candidate_comparator())


def test_candidate_hybrid_packet_rejects_nonlocked_numpy_identity() -> None:
    packet = _candidate_packet()
    packet["environment"] = {**packet["environment"], "numpy": "2.5.1"}
    for record in packet["records"]:
        record["environment"] = packet["environment"]

    with pytest.raises(ValueError, match="locked candidate environment"):
        run_eval.validate_candidate_decision_records(packet, _candidate_comparator())


def test_candidate_hybrid_driver_builds_every_binding_and_calls_production_entrypoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    builds = []
    invocations = []

    class FakeIndex:
        def __init__(self, index_dir):
            self.index_dir = index_dir
            self.pages = []
            self._lexicon = set()

        @staticmethod
        def _hit(channel):
            return ChunkHit(
                chunk_id=f"page-a:{channel}", page_id="page-a",
                path="Wiki/page-a.md", title="Page A", page_type="concept",
                section_path=[], heading="", chunk_kind="dense",
                text="needle fact", channel=channel, score=1.0,
            )

        def search_fts_terms(self, *_args, **_kwargs):
            return [self._hit("fts")]

        def search_vector(self, *_args, **_kwargs):
            return [self._hit("vector")]

        def search_page(self, *_args, **_kwargs):
            return []

        def get_chunk(self, chunk_id):
            return self._hit("vector" if chunk_id.endswith("vector") else "fts")

        def get_page_sources(self, _page_id):
            return []

        @staticmethod
        def count_tokens(text):
            return max(1, len(text) // 4)

    def fake_build(root, _wiki, mode, full_rebuild, *, candidate_query_policy=None):
        builds.append((root.name, mode, full_rebuild, candidate_query_policy))
        staged_wiki = root / "Wiki"
        staged_wiki.mkdir(parents=True)
        (staged_wiki / "page-a.md").write_text("needle fact", encoding="utf-8")
        return FakeIndex(root / ".index"), staged_wiki, 0.01

    real_hybrid_search = run_eval.hybrid_search
    def fake_hybrid(index, query, planner, **kwargs):
        invocations.append((index, query, planner, kwargs))
        return real_hybrid_search(index, query, planner, **kwargs)

    monkeypatch.setattr(run_eval, "_build", fake_build)
    monkeypatch.setattr(run_eval, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(run_eval, "_candidate_index_identity", lambda **kwargs: {
        "build_id": kwargs["policy"].candidate + str(kwargs["policy"].query_ef),
        "manifest_sha256": hashlib.sha256(str(kwargs["root"]).encode()).hexdigest(),
        "root_identity": kwargs["root"].name,
    })
    monkeypatch.setattr(run_eval, "validate_evidence", lambda _evidence: _evidence)
    monkeypatch.setattr(run_eval, "_decision_environment", lambda: {
        "python": "3.13", "platform": "test", "lancedb": "0.34.0",
        "numpy": "2.2.6", "pyarrow": "25.0.0",
    })

    packet = run_eval.run_candidate_hybrid_evaluation(
        tmp_path / "wiki", [{"query": "needle", "relevant_pages": ["page-a"], "required_facts": ["needle fact"]}],
        tmp_path / "candidates", 4096, _candidate_comparator(), baseline_quality={},
    )

    assert len(builds) == len(run_eval.CANDIDATES) * len(run_eval.DECISION_EF_GRID) == 12
    assert len(invocations) == 12
    assert all(call[1] == "needle" for call in invocations)
    assert len(packet["records"]) == 12


def test_hybrid_search_interface_remains_candidate_unaware() -> None:
    parameters = inspect.signature(run_eval.hybrid_search).parameters
    assert "candidate" not in parameters
    assert "query_ef" not in parameters


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
    assert "OMP_NUM_THREADS: ${{ env.ANN_APPROVED_OMP_THREADS }}" in scale
    assert "OPENBLAS_NUM_THREADS: ${{ env.ANN_APPROVED_OMP_THREADS }}" in scale
    assert "MKL_NUM_THREADS: ${{ env.ANN_APPROVED_OMP_THREADS }}" in scale
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
    assert '--expected-head "${{ github.sha }}"' in reconciliation
    assert '--expected-pr-head "${{ github.event.pull_request.head.sha }}"' in reconciliation
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
