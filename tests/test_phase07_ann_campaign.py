from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from eval.phase07_ann_campaign import (
    CampaignConfig,
    Phase07AnnCampaignRunner,
    canonical_digest,
    confirmation_packet_from_result,
    execute,
    select_stage1_nominees,
    validate_request,
)
from eval import phase07_operator_gate as operator
from eval.run_eval import _representative_indexed_query_separation, run_phase07_representative_campaign
from eval.ann_corpus_manifest import canonical_content_tree_sha256
from eval.ann_corpus_manifest import PHASE07_CURRENT_BASELINE, phase07_current_baseline_sha256, validate_indexed_query_digest_separation


ROOT = Path(__file__).resolve().parent.parent
HEAD = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
MODEL_MANIFEST_SHA256 = hashlib.sha256((ROOT / "eval" / "model-manifest.json").read_bytes()).hexdigest()
CORPUS_MANIFEST_SHA256 = hashlib.sha256((ROOT / "eval" / "personal-wiki-corpus-manifest.json").read_bytes()).hexdigest()
REQUIREMENTS_SHA256 = hashlib.sha256((ROOT / "requirements.txt").read_bytes()).hexdigest()
STAGE1_LEDGER = ROOT / "eval" / "phase07-stage1-authority.json"
DENSE_SOURCE_HEAD = "2f15d6a4fef54dda9b0f4a258e78898e2ef6ea57"


def _digest(kind: str) -> str:
    """Use canonical project fixture digests, never shape-only fake hexadecimal."""
    return {
        "model": MODEL_MANIFEST_SHA256,
        "corpus": CORPUS_MANIFEST_SHA256,
        "requirements": REQUIREMENTS_SHA256,
        "evidence": hashlib.sha256((ROOT / "eval" / "ann-policy.json").read_bytes()).hexdigest(),
    }[kind]


def _locked_confirmation_environment() -> dict:
    return {
        "head_sha": HEAD,
        "runtime": {
            "python": "3.13", "lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0",
            "omp_num_threads": 2, "openblas_num_threads": 2, "mkl_num_threads": 2,
        },
        "source_digests": {
            "requirements_sha256": REQUIREMENTS_SHA256,
            "model_manifest_sha256": MODEL_MANIFEST_SHA256,
            "corpus_manifest_sha256": CORPUS_MANIFEST_SHA256,
        },
        "host": {"os": "Linux", "architecture": "X64", "image": "fixture-image", "hostname": "fixture-host", "cpu_count": 2, "cpu_model": "fixture-cpu"},
    }


def _request(stage: str = "screening", *, mode: str = "stage2_sq") -> dict:
    request = {
        "schema_version": 1, "stage": stage, "request_id": "request-1", "environment": {},
        "model_manifest_sha256": _digest("model"), "corpus_manifest_sha256": _digest("corpus"),
    }
    if stage == "screening":
        request["environment"] = {
            "branch": "feature/issue-50-dense-ann-recall",
            "workflow_path": ".github/workflows/eval.yml",
            "head_sha": HEAD,
            "run_id": 1,
            "run_attempt": 1,
            "job_key": "phase07-screening",
            "job_allocation_nonce": "1-1-phase07-screening",
            "runtime": {"python": "3.13", "lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0", "omp_num_threads": 2},
        }
        request["lock_identity"] = _digest("requirements")
    if stage == "confirmation":
        plan = operator.build_confirmation_plan(STAGE1_LEDGER, post_task0_head=HEAD)
        request.update(
            environment=_locked_confirmation_environment(),
            workflow_inputs=plan["workflow_inputs"][0],
            run_identity={"run_id": 1, "run_attempt": 1, "job_id": 2, "job_allocation_nonce": "a" * 32},
        )
    if stage == "continuation":
        configs = {
            "stage2_sq": {"approved_d04_sha256": _digest("evidence"), "m": 16},
            "flat_diagnostic": {"no_confirmed_sq_sha256": _digest("evidence"), "m": 16, "query_ef": 300},
            "refinement": {"ceiling_sha256": _digest("evidence"), "m": 16, "ef_construction": 300, "query_ef": 300},
            "representative_ann": _representative_config(),
            "hybrid_non_regression": _representative_config(),
        }
        request.update(mode=mode, prior_evidence_sha256=_digest("evidence"), config=configs[mode])
    return request


def _representative_config():
    baseline = dict(PHASE07_CURRENT_BASELINE)
    finalist = {"candidate": "ivf-hnsw-sq", "m": 16, "ef_construction": 300, "query_ef": 200, "refine_factor": None}
    return {"size": 1000, "baseline": baseline, "finalist": finalist,
            "baseline_sha256": phase07_current_baseline_sha256(), "finalist_sha256": canonical_digest(finalist)}


def _tiny_runner(tmp_path: Path) -> Phase07AnnCampaignRunner:
    # Explicit trusted Python seam: request JSON has no rows/probes/dimensions fields.
    return Phase07AnnCampaignRunner(CampaignConfig(rows=64, dimensions=8, probes=4, per_build_max_seconds=30, work_dir=tmp_path))


def _nomination_build(m: int, *, r10: float, r20: float, p95: float, index_bytes: int) -> dict:
    return {
        "build": {"m": m, "index_bytes": index_bytes},
        "queries": [
            {"query_ef": 300, "recall_at_10": r10, "recall_at_20": r20, "latency_p95_ms": p95},
        ],
    }


def _nomination_stat(m: int, metric: str, effect: float, *, significant: bool = True) -> dict:
    return {
        "comparison": {"m": m, "metric": metric, "baseline_ef": 200, "candidate_ef": 300},
        "mean_effect": effect,
        "basic_ci_95": [0.01, 0.02] if significant else [-0.02, -0.01],
        "holm_adjusted_p": 0.01 if significant else 0.5,
    }


def _embed384():
    def encode(texts):
        vectors = []
        for text in texts:
            rng = random.Random(f"phase07::{text}")
            raw = [rng.uniform(-1, 1) for _ in range(384)]
            norm = math.sqrt(sum(value * value for value in raw))
            vectors.append([value / norm for value in raw])
        return vectors
    return encode


def _write_sealed_dense_ledger(tmp_path: Path) -> Path:
    """Materialize the real 07-05 ledger shape without a developer-machine path.

    This is deliberately an evidence fixture rather than a tiny synthetic
    request: it has all three physical D-25 ordinal identities, disjoint
    baseline/m20/m32 build identities, exact hosted provenance fields, and a
    canonical self-digest made by the production JSON helper.
    """
    ordinals = (
        (1, 32801985769, 97664517767, 9546915747),
        (2, 32802007002, 97664580321, 9546916208),
        (3, 32802027355, 97664640212, 9546924769),
    )

    def digest(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    records = []
    for ordinal, run_id, job_id, artifact_id in ordinals:
        builds = [
            {"build_id": digest(f"ordinal-{ordinal}-m-{m}"), "m": m,
             "ef_construction": 300, "query_ef": [ef]}
            for m, ef in ((16, 100), (20, 300), (32, 300))
        ]
        baseline_build_id = builds[0]["build_id"]
        candidate_build_ids = {str(row["m"]): row["build_id"] for row in builds[1:]}
        record = {
            "schema_version": 1,
            "campaign_stage": "confirmation",
            "slot": {"ordinal": ordinal},
            "run_id": run_id,
            "run_attempt": 1,
            "job_id": job_id,
            "job_key": "phase07-confirmation",
            "job_allocation_nonce": digest(f"allocation-{ordinal}")[:32],
            "status": "numeric-success",
            "failure_class": None,
            "replacement_for_run_id": None,
            "retention_days": 90,
            "workflow_inputs_sha256": digest(f"workflow-input-{ordinal}"),
            "raw_tree_sha256": digest(f"tree-{ordinal}"),
            "builds": builds,
            "d25": {"baseline_build_id": baseline_build_id, "candidate_build_ids": candidate_build_ids},
            "measurements": {"authorization": "none", "build_count": 3, "builds": builds},
            "locked_execution": {"head_sha": DENSE_SOURCE_HEAD},
            "validated_measurements": {
                "authorization": "none", "build_count": 3, "builds": builds,
                "baseline_build_id": baseline_build_id, "candidate_build_ids": candidate_build_ids,
                "locked_execution": {"head_sha": DENSE_SOURCE_HEAD},
                "run_identity": {"run_id": run_id, "run_attempt": 1, "job_id": job_id,
                                 "job_allocation_nonce": digest(f"allocation-{ordinal}")[:32]},
                "slot": {"ordinal": ordinal}, "workflow_inputs_sha256": digest(f"workflow-input-{ordinal}"),
                "paired_statistics": {"family_name": "d25_candidate_vs_production_baseline", "family_size": 4},
            },
            "validated_provenance": {
                "run_id": run_id, "run_attempt": 1, "job_id": job_id, "artifact_id": artifact_id,
                "job_key": "phase07-confirmation", "head_sha": DENSE_SOURCE_HEAD,
                "status": "completed", "conclusion": "success", "retention_days": 90,
                "artifact_name": f"phase07-confirmation-{run_id}-1",
                "api_archive_sha256": digest(f"archive-{ordinal}"),
            },
        }
        record["record_self_sha256"] = operator.canonical_digest(record)
        records.append(record)
    ledger = {
        "schema_version": 1,
        "campaign_stage": "confirmation",
        "confirmation_plan_sha256": digest("confirmation-plan"),
        "eligible_evidence_runs": records,
        "all_physical_workflow_runs": records,
        "paired_ordinal_families": [{"ordinal": ordinal} for ordinal, *_ in ordinals],
    }
    ledger["record_self_sha256"] = operator.canonical_digest(ledger)
    target = tmp_path / "07-05-dense-ledger.json"
    target.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")
    return target


def test_request_schema_seals_success_and_rejection_artifacts(tmp_path: Path) -> None:
    runner = _tiny_runner(tmp_path)
    result = execute(_request(), tmp_path / "success", runner=runner.run)
    assert result["authorization"] == "none"
    sealed = json.loads((tmp_path / "success" / "screening-result.json").read_text())
    assert sealed["record_self_sha256"] and sealed["result"]["build_count"] == 3
    bad = _request(); bad["rows"] = 64
    with pytest.raises(ValueError):
        execute(bad, tmp_path / "rejected", runner=runner.run)
    assert not (tmp_path / "rejected").exists(), "invalid input performs no artifact write"
    timeout = Phase07AnnCampaignRunner(CampaignConfig(rows=64, dimensions=8, probes=4,
                                                       per_build_max_seconds=0.000001,
                                                       work_dir=tmp_path))
    with pytest.raises(RuntimeError, match="watchdog"):
        execute(_request(), tmp_path / "watchdog", runner=timeout.run)
    rejected = json.loads((tmp_path / "watchdog" / "screening-rejection.json").read_text())
    assert rejected["status"] == "reject-evidence" and rejected["record_self_sha256"]


def test_module_campaign_cli_reaches_request_validation_without_running_stress_matrix(
    tmp_path: Path,
) -> None:
    """The hosted workflow invokes the production package entry point."""
    request = tmp_path / "invalid-request.json"
    request.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.phase07_ann_campaign",
            "--request-file",
            str(request),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "[FAIL] Phase 7 campaign: typed Phase 7 request" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "mode",
    ["stage2_sq", "flat_diagnostic", "refinement", "representative_ann", "hybrid_non_regression"],
)
def test_d25_rejects_every_retired_continuation_before_artifacts(
    tmp_path: Path, mode: str,
) -> None:
    request = _request("continuation", mode=mode)
    output = tmp_path / mode
    with pytest.raises(ValueError, match="D-25|typed Phase 7 request"):
        validate_request(request)
    assert not output.exists(), "retired work must fail before build or artifact creation"


def test_representative_query_overlap_guard_remains_available_for_later_hybrid_gates() -> None:
    overlapping_digest = hashlib.sha256(b"overlapping-indexed-query-fixture").hexdigest()
    with pytest.raises(ValueError):
        validate_indexed_query_digest_separation(
            indexed_row_digests=[overlapping_digest],
            query_row_digests=[overlapping_digest],
        )


def test_tiny_screening_uses_real_lancedb_reopen_and_ann_grid(tmp_path: Path) -> None:
    record = execute(_request(), tmp_path / "campaign", runner=_tiny_runner(tmp_path).run)
    result = record["result"]
    assert result["exact_truth_computed_once"] is True and result["build_count"] == 3
    assert result["d04_statistics"]["family_size"] == 6
    assert len(result["nominated_m"]) <= 2
    assert [build["build"]["m"] for build in result["builds"]] == [16, 20, 32]
    for build in result["builds"]:
        assert build["build"]["normal_ann_request_count"] == 16
        assert [group["query_ef"] for group in build["queries"]] == [100, 150, 200, 300]
        assert build["build"]["dense_table_open_count"] >= 1
        assert build["build"]["watchdog"]["owner"] == "parent"


@pytest.mark.parametrize(
    ("builds", "statistics", "expected"),
    [
        (
            [
                _nomination_build(16, r10=0.8, r20=0.8, p95=4, index_bytes=100),
                _nomination_build(20, r10=0.7, r20=0.7, p95=5, index_bytes=200),
                _nomination_build(32, r10=0.9, r20=0.9, p95=6, index_bytes=300),
            ],
            {"comparisons": [
                _nomination_stat(16, "recall_at_10", 0.1), _nomination_stat(16, "recall_at_20", -0.1, significant=False),
                _nomination_stat(20, "recall_at_10", 0.1), _nomination_stat(20, "recall_at_20", 0.0, significant=False),
                _nomination_stat(32, "recall_at_10", 0.0, significant=False), _nomination_stat(32, "recall_at_20", 0.1),
            ]},
            [32, 20],
        ),
        (
            [
                _nomination_build(16, r10=0.8, r20=0.8, p95=9, index_bytes=100),
                _nomination_build(20, r10=0.8, r20=0.8, p95=8, index_bytes=200),
                _nomination_build(32, r10=0.8, r20=0.8, p95=8, index_bytes=150),
            ],
            {"comparisons": [
                *[_nomination_stat(m, metric, 0.1) for m in (16, 20, 32) for metric in ("recall_at_10", "recall_at_20")],
            ]},
            [32, 20],
        ),
        (
            [
                _nomination_build(16, r10=0.8, r20=0.8, p95=8, index_bytes=100),
                _nomination_build(20, r10=0.8, r20=0.8, p95=8, index_bytes=100),
                _nomination_build(32, r10=0.8, r20=0.8, p95=8, index_bytes=150),
            ],
            {"comparisons": [
                *[_nomination_stat(m, metric, 0.1) for m in (16, 20, 32) for metric in ("recall_at_10", "recall_at_20")],
            ]},
            [16, 20],
        ),
    ],
    ids=["filter-negative-other-metric-and-rank-joint-recall", "tie-p95-then-index", "tie-index-then-lower-m"],
)
def test_stage1_nominee_selection_is_qualified_and_deterministically_ranked(
    builds: list[dict], statistics: dict, expected: list[int],
) -> None:
    assert select_stage1_nominees(builds, statistics) == expected


def test_confirmation_is_one_paired_d25_ordinal_and_only_queries_fixed_efs(tmp_path: Path) -> None:
    runner = _tiny_runner(tmp_path)
    confirmation = execute(_request("confirmation"), tmp_path / "confirm", runner=runner.run)["result"]
    assert confirmation["build_count"] == 3
    assert {build["build"]["m"] for build in confirmation["builds"]} == {16, 20, 32}
    assert {
        build["build"]["m"]: [group["query_ef"] for group in build["queries"]]
        for build in confirmation["builds"]
    } == {16: [100], 20: [300], 32: [300]}
    assert confirmation["paired_statistics"]["family_name"] == "d25_candidate_vs_production_baseline"
    assert confirmation["paired_statistics"]["family_size"] == 4
    assert [
        (row["comparison"]["candidate_m"], row["comparison"]["metric"])
        for row in confirmation["paired_statistics"]["comparisons"]
    ] == [
        (20, "recall_at_10"), (20, "recall_at_20"),
        (32, "recall_at_10"), (32, "recall_at_20"),
    ]
    packet = confirmation_packet_from_result(result=confirmation, workflow_inputs=_request("confirmation")["workflow_inputs"],
                                             run_id=1, run_attempt=1, job_id=2, job_key="phase07-confirmation",
                                             job_allocation_nonce="a" * 32, raw_tree_sha256=_digest("evidence"))
    assert packet["d25"]["family_size"] == 4
    assert packet["slot"] == {"ordinal": 1}


def test_hybrid_plan_is_derived_from_the_sealed_dense_ledger_and_rejects_tampering(tmp_path: Path) -> None:
    """D-25 hybrid authority is a fresh, exact two-member derivation only."""
    dense_ledger = _write_sealed_dense_ledger(tmp_path)
    dense_ledger_sha256 = operator.canonical_digest(json.loads(dense_ledger.read_text(encoding="utf-8")))
    plan = operator.build_hybrid_plan(dense_ledger, post_implementation_head=HEAD)

    request = plan["hybrid_request"]
    assert request["dense_ledger_sha256"] == dense_ledger_sha256
    assert request["dense_source_head"] == DENSE_SOURCE_HEAD
    assert request["hybrid_implementation_head"] == HEAD
    assert request["authorization"] == "none"
    inputs = plan["workflow_inputs"]
    assert [row["candidate"] for row in inputs] == [
        {"index_type": "hnsw_sq", "m": 20, "ef_construction": 300, "query_ef": 300},
        {"index_type": "hnsw_sq", "m": 32, "ef_construction": 300, "query_ef": 300},
    ]
    assert all(row["baseline"] == {"index_type": "hnsw_sq", "m": 16, "ef_construction": 300, "query_ef": 100}
               and row["scale"] == 30000 and row["query_count"] == 105
               and row["replacement_for_run_id"] is None for row in inputs)

    for field, value in (
        ("dense_ledger_sha256", "0" * 64),
        ("dense_source_head", "0" * 40),
        ("authorization", "approve-sq"),
        ("retention_days", 1),
        ("replacement_for_run_id", 1),
    ):
        tampered = deepcopy(plan)
        tampered["hybrid_request"][field] = value
        with pytest.raises(ValueError):
            operator.validate_hybrid_plan(tampered)
    for field, value in (("run_id", 1), ("job_id", 1), ("artifact_id", 1), ("build_ids", [])):
        tampered = deepcopy(plan)
        tampered["hybrid_request"]["dense_ordinal_identities"][0][field] = value
        with pytest.raises(ValueError):
            operator.validate_hybrid_plan(tampered)


def test_hybrid_m20_bundle_exercises_public_hybrid_path_without_retired_campaign_modes(tmp_path: Path) -> None:
    """The injected encoder is the sole test seam; retrieval remains production code."""
    plan = operator.build_hybrid_plan(_write_sealed_dense_ledger(tmp_path), post_implementation_head=HEAD)
    m20 = plan["workflow_inputs"][0]
    from eval.run_eval import run_phase07_hybrid_campaign

    result = run_phase07_hybrid_campaign(
        bundle=m20, work_dir=tmp_path, embed=_embed384(), query_limit=2,
    )

    assert result["hybrid_invocation"] == {
        "entrypoint": "query.hybrid_search",
        "candidate_aware_public_arguments": False,
        "original_baseline_calls": 2,
        "original_candidate_calls": 2,
        "expanded_baseline_calls": 2,
        "expanded_candidate_calls": 2,
    }
    assert len(result["original_absolute_observations"]) == 2
    assert len(result["expanded_paired_observations"]) == 2
    assert set(result["expanded_paired_observations"][0]) >= {"ordinal", "query_sha256", "baseline", "candidate"}
    assert "representative_ann" not in json.dumps(result, sort_keys=True)
    assert all(token not in json.dumps(result, sort_keys=True) for token in ("continuation", "flat", "refinement", "ef_construction=500"))


def test_public_hybrid_facade_uses_real_build_service_and_lancedb(tmp_path: Path) -> None:
    # This is deliberately not a mock: injected encoding is the only test seam.
    config = _representative_config()
    result = run_phase07_representative_campaign(mode="representative_ann", size=1000,
                                                  baseline=config["baseline"], finalist=config["finalist"],
                                                  work_dir=tmp_path, authorization="none", embed=_embed384(), query_limit=2)
    assert result["authorization"] == "none"
    assert result["hybrid_invocation"]["entrypoint"] == "query.hybrid_search"
    assert result["hybrid_invocation"] == {"entrypoint": "query.hybrid_search", "original_baseline_calls": 2, "baseline_calls": 2, "finalist_calls": 2}
    assert result["original_fixture"]["corpus_sha256"] != result["expanded"]["corpus_sha256"]
    assert result["original_fixture"]["absolute_baseline"] is not result["expanded"]["paired_observations"]
    row = result["personal_wiki_ann_exact"]["rows"][0]
    assert {"baseline_recall_at_10", "baseline_recall_at_20", "finalist_recall_at_10", "finalist_recall_at_20"} <= set(row)
    assert result["personal_wiki_ann_exact"]["indexed_query_overlap_count"] == 0
    from build_index import WikiIndex
    wi = WikiIndex(tmp_path / "representative-representative_ann-1000" / "expanded-baseline" / ".index")
    wi.load()
    indexed_text = wi._get_repository()._dense_table().to_arrow().to_pylist()[0]["text"]
    with pytest.raises(ValueError, match="query/corpus overlap"):
        _representative_indexed_query_separation(wi._get_repository(), [indexed_text])


def test_corpus_content_identity_is_root_independent_and_content_sensitive(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    shutil.copytree(Path(__file__).parent / "fixtures" / "wiki", first)
    shutil.copytree(first, second)
    assert canonical_content_tree_sha256(first) == canonical_content_tree_sha256(second)
    (second / "distractor.md").write_text("public distractor changes corpus content", encoding="utf-8")
    assert canonical_content_tree_sha256(first) != canonical_content_tree_sha256(second)
