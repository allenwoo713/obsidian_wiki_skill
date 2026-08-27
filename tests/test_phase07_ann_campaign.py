from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

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
from eval.run_eval import (_representative_indexed_query_separation, run_phase07_representative_campaign,
                           expected_phase07_expanded_corpus_identity, aggregate_hybrid_serialized_metrics)
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


def test_frozen_role_provenance_requires_three_distinct_attempt_one_roles() -> None:
    """Prepare evidence is non-authorizing until all independent role records bind it."""
    from eval.phase07_frozen_base import validate_frozen_role_provenance  # noqa: PLC0415

    base = {
        "prepare_run_id": 101, "prepare_run_attempt": 1, "prepare_job_id": 201,
        "prepare_artifact_id": 301, "prepare_archive_sha256": hashlib.sha256(b"archive").hexdigest(),
        "prepare_descriptor_sha256": hashlib.sha256(b"descriptor").hexdigest(),
        "prepare_tree_sha256": hashlib.sha256(b"tree").hexdigest(), "retention_days": 90,
        "head_sha": HEAD, "runtime": _locked_confirmation_environment()["runtime"],
        "corpus_sha256": CORPUS_MANIFEST_SHA256, "model_manifest_sha256": MODEL_MANIFEST_SHA256,
    }
    roles = [
        {"role": "baseline", "m": 16, "run_id": 1, "run_attempt": 1, "job_id": 11, "artifact_id": 21, **base},
        {"role": "m20", "m": 20, "run_id": 2, "run_attempt": 1, "job_id": 12, "artifact_id": 22, **base},
        {"role": "m32", "m": 32, "run_id": 3, "run_attempt": 1, "job_id": 13, "artifact_id": 23, **base},
    ]
    assert validate_frozen_role_provenance(roles, expected_head=HEAD) == roles
    duplicate = [dict(record) for record in roles]
    duplicate[2]["run_id"] = 2
    with pytest.raises(ValueError, match="distinct"):
        validate_frozen_role_provenance(duplicate, expected_head=HEAD)
    mixed = [dict(record) for record in roles]
    mixed[1]["prepare_descriptor_sha256"] = hashlib.sha256(b"replacement").hexdigest()
    with pytest.raises(ValueError, match="mixed"):
        validate_frozen_role_provenance(mixed, expected_head=HEAD)


def test_frozen_prepare_has_no_future_id_gate_and_failed_prepare_emits_zero_roles() -> None:
    preflight = {
        "schema_version": 1, "head_sha": HEAD, "target_size": 30000, "wall_time_seconds": 7199,
        "cap_minutes": 120, "uncompressed_bytes": 12, "archive_bytes": 8, "file_count": 4,
        "largest_file": 6, "descriptor_sha256": hashlib.sha256(b"descriptor").hexdigest(),
        "tree_sha256": hashlib.sha256(b"tree").hexdigest(), "repository_capability": True,
        "human_authorized": True,
    }
    bundle = operator.build_frozen_prepare_bundle(preflight, expected_head=HEAD)
    assert bundle["campaign_stage"] == "hybrid-prepare"
    assert "artifact_id" not in bundle["local_preflight"]
    assert operator.build_frozen_role_bundles({"status": "cancelled"}, expected_head=HEAD) == []
    roles = operator.build_frozen_role_bundles(
        _frozen_prepare_identity(), expected_head=HEAD,
        locked_execution=_locked_confirmation_environment(),
    )
    assert [(row["role"], row["config"]["m"]) for row in roles] == [("baseline", 16), ("m20", 20), ("m32", 32)]


def _frozen_prepare_identity() -> dict:
    """One successful API-bound base identity shared by all frozen roles."""
    runtime = _locked_confirmation_environment()["runtime"]
    return {
        "repository": "allenwoo713/obsidian_wiki_skill", "head_sha": HEAD,
        "run_id": 101, "run_attempt": 1, "job_id": 201, "artifact_id": 301,
        "artifact_name": "phase07-frozen-base-101-1",
        "archive_sha256": hashlib.sha256(b"frozen archive").hexdigest(),
        "archive_size_bytes": 1234,
        "descriptor_self_sha256": hashlib.sha256(b"frozen descriptor").hexdigest(),
        "base_tree_sha256": hashlib.sha256(b"frozen tree").hexdigest(),
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "corpus_manifest_sha256": CORPUS_MANIFEST_SHA256,
        "generator_recipe_sha256": hashlib.sha256(b"frozen generator recipe").hexdigest(),
        "runtime": runtime,
        "retention_days": 90, "replacement_for_run_id": None,
        "status": "success",
        "artifact_created_at": "2026-08-01T00:00:00Z",
        "artifact_expires_at": "2026-10-30T00:00:00Z",
    }


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.pop("descriptor_self_sha256"),
        lambda value: value.__setitem__("run_attempt", 2),
        lambda value: value.__setitem__("head_sha", "0" * 64),
        lambda value: value.__setitem__("replacement_for_run_id", 1),
        lambda value: value.__setitem__("status", "cancelled"),
        lambda value: value.__setitem__("artifact_expires_at", "2000-04-01T00:00:00Z"),
        lambda value: value["runtime"].__setitem__("numpy", "0.0.0"),
        lambda value: value.__setitem__("model_manifest_sha256", "0" * 64),
    ),
)
def test_frozen_prepare_identity_fails_closed_before_role_authorization(mutate) -> None:
    """No missing, stale, retry, replacement, failed, or mixed base may enter a role."""
    identity = _frozen_prepare_identity()
    assert operator.validate_frozen_prepare_identity(
        identity, expected_head=HEAD, locked_execution=_locked_confirmation_environment(),
    ) == identity
    invalid = deepcopy(identity)
    mutate(invalid)
    with pytest.raises(ValueError):
        operator.validate_frozen_prepare_identity(
            invalid, expected_head=HEAD, locked_execution=_locked_confirmation_environment(),
        )


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


class _DeterministicFrozenTokenizer:
    """A tokenizer-shaped test double whose count is never char//4."""

    def __call__(self, text: str, **_kwargs) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split()) + 2))}


class _DeterministicFrozenEmbedder:
    """Verified-loader compatible encoder used only behind private test seams."""

    def __init__(self) -> None:
        self.tokenizer = _DeterministicFrozenTokenizer()
        self.encode_calls: list[int] = []

    def embed(self, texts) -> list[list[float]]:
        self.encode_calls.append(len(texts))
        return _embed384()(texts)

    def encode(self, texts, **_kwargs) -> list[list[float]]:
        return self.embed(texts)

    def get_embedding_dimension(self) -> int:
        return 384


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

    build_ids = {
        1: ("078cc8451c21e17dfac726d6a26aa7519375756d1ec71c38ce7ecf8bc5f256dc", "d550c98e3aff255c53d459b0ffe19b00d19d6afa702023e37558777fe9223e73", "bb0b4a1fd23cbdfab4845a4975a0119f4d9963fc65bf1d6afd825d4ba6d2b42a"),
        2: ("45d772249ce3790b85955ca68cbea16d5a003e8db34177609bb18f3a9536fd02", "42aaee989e396e3b9030bdd099b183a1903b80f8fcd1d0d07c9694828e2f8744", "2674f4dd7caba0744a0cf499fece5d8bf0b2c7841b22cc8b62113e3a976a87c4"),
        3: ("825c1a3f523e62affe8385443f628aa3d0638db7468f8c6871e76a9038ef44c0", "b6f66e475137e3e58c7554d503b0714586df6a0558b5ab9c22b2118cf28ea4e4", "95194274760a8b9f083c629250e4437d4199efe633241a0ae02bb161300b848f"),
    }
    records = []
    for ordinal, run_id, job_id, artifact_id in ordinals:
        builds = [
            {"build_id": build_id, "m": m,
             "ef_construction": 300, "query_ef": [ef]}
            for build_id, (m, ef) in zip(build_ids[ordinal], ((16, 100), (20, 300), (32, 300)), strict=True)
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
                "artifact_expires_at": "2026-11-23T02:35:30Z",
                "artifact_name": f"phase07-confirmation-{run_id}-1",
                "api_archive_sha256": digest(f"archive-{ordinal}"),
            },
        }
        record["record_self_sha256"] = operator.canonical_digest(record)
        records.append(record)
    physical_records = []
    for record in records:
        physical = deepcopy(record)
        physical["eligible"] = True
        physical["record_self_sha256"] = operator.canonical_digest(physical)
        physical_records.append(physical)
    ledger = {
        "schema_version": 1,
        "campaign_stage": "confirmation",
        "confirmation_plan_sha256": digest("confirmation-plan"),
        "eligible_evidence_runs": records,
        "all_physical_workflow_runs": physical_records,
        "paired_ordinal_families": [{"ordinal": ordinal} for ordinal, *_ in ordinals],
    }
    ledger["record_self_sha256"] = operator.canonical_digest(ledger)
    target = tmp_path / "07-05-dense-ledger.json"
    target.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")
    return target


def _build_test_hybrid_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, str]:
    dense_ledger = _write_sealed_dense_ledger(tmp_path)
    digest = operator.canonical_digest(json.loads(dense_ledger.read_text(encoding="utf-8")))
    # Production remains pinned to the hosted ledger digest.  The self-contained
    # fixture supplies the same schema/identity contract while avoiding an
    # ignored developer-machine artifact in CI.
    monkeypatch.setattr(operator, "DENSE_LEDGER_DIGEST", digest)
    return operator.build_hybrid_plan(dense_ledger, post_implementation_head=HEAD), digest


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


def test_hybrid_plan_is_derived_from_the_sealed_dense_ledger_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-25 hybrid authority is a fresh, exact two-member derivation only."""
    plan, dense_ledger_sha256 = _build_test_hybrid_plan(tmp_path, monkeypatch)

    request = plan["hybrid_request"]
    assert request["dense_ledger_sha256"] == dense_ledger_sha256
    assert request["dense_source_head"] == DENSE_SOURCE_HEAD
    assert request["hybrid_implementation_head"] == HEAD
    assert request["authorization"] == "none"
    inputs = plan["workflow_inputs"]
    assert [(row["role"], row["config"]) for row in inputs] == [
        ("baseline", {"index_type": "hnsw_sq", "m": 16, "ef_construction": 300, "query_ef": 100}),
        ("candidate", {"index_type": "hnsw_sq", "m": 20, "ef_construction": 300, "query_ef": 300}),
        ("candidate", {"index_type": "hnsw_sq", "m": 32, "ef_construction": 300, "query_ef": 300}),
    ]
    assert all(row["scale"] == 30000 and row["query_count"] == 105
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


def test_hybrid_m20_dispatch_mints_an_internal_execution_capability_without_retired_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public operator boundary, not a bare member, grants campaign execution."""
    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    dispatched: list[dict] = []
    result = operator.execute_hybrid_dispatch(
        bundle=_hybrid_dispatch_bundle(plan), locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
        work_dir=tmp_path / "minted", runner=lambda **kwargs: dispatched.append(kwargs) or {"authorization": "none"},
    )
    assert result["authorization"] == "none" and len(dispatched) == 1
    assert "bundle" not in dispatched[0]
    assert not isinstance(dispatched[0]["capability"], dict)


def test_bare_sealed_workflow_member_cannot_reach_any_campaign_build_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the locked dispatch boundary may mint the internal campaign capability."""
    import eval.run_eval as run_eval
    import build_index

    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(run_eval, "_materialize_phase07_expanded_corpus",
                        lambda **_kwargs: calls.append("materialize") or (_ for _ in ()).throw(ValueError("build reached")))
    monkeypatch.setattr(run_eval.WikiIndex, "build", lambda *_args, **_kwargs: calls.append("WikiIndex.build"))
    monkeypatch.setattr(build_index, "build_storage_contract", lambda *_args, **_kwargs: calls.append("build_storage_contract"))
    monkeypatch.setattr(run_eval, "hybrid_search", lambda *_args, **_kwargs: calls.append("hybrid_search"))
    # A self-sealed member is intentionally not a callable execution token:
    # there is no public dict/member campaign entry point to invoke.
    assert not hasattr(run_eval, "run_phase07_hybrid_campaign")
    with pytest.raises(ValueError):
        run_eval._run_phase07_hybrid_campaign_with_capability(
            capability=plan["workflow_inputs"][0], work_dir=tmp_path / "bare-member",
            embed=_embed384(), query_limit=2,
        )
    assert calls == []


def _hybrid_dispatch_bundle(plan: dict, *, role: str = "m20", m: int = 20) -> dict:
    """Return a real B-generated frozen dispatch, never the retired generic shape."""
    role = f"m{m}" if role == "candidate" else role
    bundles = operator.build_frozen_role_dispatch_bundles(
        hybrid_request=plan["hybrid_request"], frozen_prepare=_frozen_prepare_identity(),
        expected_head=HEAD,
    )
    return next(bundle for bundle in bundles
                if bundle["workflow_input"]["role"] == role
                and bundle["workflow_input"]["config"]["m"] == (16 if role == "baseline" else m))


def test_frozen_runner_uses_the_prepared_source_and_keeps_the_full_public_query_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production dispatcher exports one real frozen 30k/105 role packet."""
    import inspect
    import build_index
    import eval.run_eval as run_eval
    from eval import phase07_ann_campaign as campaign  # noqa: PLC0415
    from eval import phase07_frozen_base as frozen_base  # noqa: PLC0415
    from eval.phase07_frozen_base import prepare_frozen_base, validate_frozen_base  # noqa: PLC0415
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository  # noqa: PLC0415

    frozen = tmp_path / "frozen-source"
    frozen_wiki = frozen / "Wiki"
    shutil.copytree(ROOT / "tests" / "fixtures" / "wiki", frozen_wiki)
    source_identity = {
        "expanded_content_tree_sha256": canonical_content_tree_sha256(frozen_wiki),
        "expanded_member_count": sum(1 for path in frozen_wiki.rglob("*.md") if path.is_file()),
    }
    # These private dependencies are the only test authority for a tiny
    # source.  Neither the dispatch bundle nor the CLI can set either seam.
    monkeypatch.setattr(
        frozen_base, "_expected_corpus_identity",
        lambda value, *, allow_explicit=False: dict(source_identity),
    )
    monkeypatch.setattr(
        run_eval, "expected_phase07_expanded_corpus_identity",
        lambda **_kwargs: dict(source_identity),
    )
    frozen_embedder = _DeterministicFrozenEmbedder()
    descriptor = prepare_frozen_base(
        wiki_dir=frozen_wiki, frozen_dir=frozen, embed=frozen_embedder.embed,
        tokenizer=frozen_embedder.tokenizer,
    )
    source_before = validate_frozen_base(
        frozen, expected_wiki_root=frozen_wiki, tokenizer=frozen_embedder.tokenizer,
    )
    frozen_embedder.encode_calls.clear()
    frozen_prepare = _frozen_prepare_identity()
    frozen_prepare.update(
        descriptor_self_sha256=descriptor["record_self_sha256"],
        base_tree_sha256=source_before,
        model_manifest_sha256=descriptor["model_manifest_sha256"],
        corpus_manifest_sha256=descriptor["corpus_manifest_sha256"],
        generator_recipe_sha256=descriptor["generator_recipe_sha256"],
        runtime=descriptor["runtime"],
    )
    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    # Rebuild the dispatch from the actual frozen descriptor rather than its
    # shape-only fixture identity.
    bundle = next(
        item for item in operator.build_frozen_role_dispatch_bundles(
            hybrid_request=plan["hybrid_request"], frozen_prepare=frozen_prepare, expected_head=HEAD,
        )
        if item["workflow_input"]["role"] == "m20"
    )

    root = tmp_path / "campaign" / "hybrid-m20-m20"
    materialize_calls: list[object] = []
    original_builds: list[Path] = []
    scan_calls: list[Path] = []
    chunk_calls: list[Path] = []
    persist_calls: list[Path] = []
    graph_calls: list[tuple[Path, Path]] = []
    loads: list[Path] = []
    public_queries: list[dict] = []

    monkeypatch.setattr(
        run_eval, "_materialize_phase07_expanded_corpus",
        lambda **_kwargs: materialize_calls.append(_kwargs) or pytest.fail("frozen role materialized expanded corpus"),
    )
    real_index_build = run_eval.WikiIndex.build

    def record_public_build(self, wiki_dir, **kwargs):
        assert Path(wiki_dir) != frozen_wiki
        original_builds.append(Path(wiki_dir))
        assert kwargs["full_rebuild"] is True
        return real_index_build(self, wiki_dir, **kwargs)

    monkeypatch.setattr(run_eval.WikiIndex, "build", record_public_build)
    real_scan_wiki = build_index.scan_wiki

    def record_scan_chunk(wiki_dir, project_root):
        assert Path(wiki_dir) != frozen_wiki
        scan_calls.append(Path(wiki_dir))
        return real_scan_wiki(wiki_dir, project_root)

    monkeypatch.setattr(build_index, "scan_wiki", record_scan_chunk)
    real_chunk_page = build_index.chunk_page

    def record_chunk_page(*args, **kwargs):
        page = kwargs.get("path")
        assert page is not None and Path(page).is_relative_to(root / "original" / "Wiki")
        chunk_calls.append(Path(page))
        return real_chunk_page(*args, **kwargs)

    monkeypatch.setattr(build_index, "chunk_page", record_chunk_page)
    real_persist = LanceDbIndexRepository.persist

    def record_persist(self, lance_dir, *args, **kwargs):
        assert not str(Path(lance_dir)).startswith(str(root / "expanded-private-clone"))
        persist_calls.append(Path(lance_dir))
        return real_persist(self, lance_dir, *args, **kwargs)

    monkeypatch.setattr(LanceDbIndexRepository, "persist", record_persist)
    real_graph_writer = run_eval.write_graph_artifact

    def record_graph_writer(wiki, index_dir):
        assert Path(wiki) != frozen_wiki
        graph_calls.append((Path(wiki), Path(index_dir)))
        return real_graph_writer(wiki, index_dir)

    monkeypatch.setattr(run_eval, "write_graph_artifact", record_graph_writer)
    real_load = run_eval.WikiIndex.load

    def record_load(self, *args, **kwargs):
        loads.append(Path(self.index_dir))
        return real_load(self, *args, **kwargs)

    monkeypatch.setattr(run_eval.WikiIndex, "load", record_load)
    real_hybrid_search = run_eval.hybrid_search

    def record_public_hybrid_search(*args, **kwargs):
        assert not {"candidate", "candidate_config", "candidate_aware"} & set(kwargs)
        public_queries.append(dict(kwargs))
        return real_hybrid_search(*args, **kwargs)

    monkeypatch.setattr(run_eval, "hybrid_search", record_public_hybrid_search)
    monkeypatch.setattr(run_eval, "_load_phase07_frozen_runner_embedder", lambda: frozen_embedder)
    real_get_embedder = run_eval.WikiIndex._get_embedder

    def inject_verified_embedder(self):
        if self.index_dir == root / "original" / ".index":
            return frozen_embedder
        return real_get_embedder(self)

    monkeypatch.setattr(run_eval.WikiIndex, "_get_embedder", inject_verified_embedder)

    # The public dispatcher and CLI expose no identity, tokenizer, or encoder
    # seam.  runner=None therefore crosses the true production export path.
    assert "_expected_frozen_corpus_identity" not in inspect.signature(operator.execute_hybrid_dispatch).parameters
    assert "tokenizer" not in inspect.signature(operator.execute_hybrid_dispatch).parameters
    assert "embed" not in inspect.signature(operator.execute_hybrid_dispatch).parameters
    cli_source = inspect.getsource(operator.main)
    assert 'add_argument("--tokenizer"' not in cli_source
    assert 'add_argument("--embed"' not in cli_source
    assert "_expected_frozen_corpus_identity" not in cli_source
    result = operator.execute_hybrid_dispatch(
        bundle=bundle, locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8,
                    "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
        work_dir=tmp_path, runner=None, frozen_dir=frozen,
    )

    assert materialize_calls == []
    assert len(original_builds) == len(scan_calls) == len(persist_calls) == len(graph_calls) == 1
    assert chunk_calls
    assert frozen_embedder.encode_calls and all(count > 0 for count in frozen_embedder.encode_calls)
    assert graph_calls[0][0] == root / "original" / "Wiki"
    assert not (root / "expanded" / "Wiki").exists()
    assert (root / "expanded-private-clone").is_dir()
    assert result["frozen_prepare"] == frozen_prepare
    assert result["source_before_sha256"] == result["source_after_sha256"] == source_before
    assert validate_frozen_base(
        frozen, expected_wiki_root=frozen_wiki, tokenizer=frozen_embedder.tokenizer,
    ) == source_before
    assert result["query_count"] == 105
    assert result["hybrid_invocation"] == {
        "entrypoint": "query.hybrid_search", "candidate_aware_public_arguments": False,
        "original_calls": 105, "expanded_calls": 105,
    }
    assert len(public_queries) == 210
    assert {Path(row["wiki_dir"]) for row in public_queries} == {root / "original" / "Wiki", frozen_wiki}
    assert all(row["intent_override"] == "auto" and row["allow_local_fallback"] is True for row in public_queries)
    assert loads == [root / "original" / ".index", root / "expanded" / ".index"]
    raw = tmp_path / "raw"
    assert {path.name for path in raw.iterdir()} == {
        "hybrid-request.json", "dispatch-bundle.json", "locked-execution.json", "allocation.json",
        "hybrid-result.json", "hybrid-ledger.json", "hybrid-packet.json",
    }
    assert campaign.validate_hybrid_artifact_tree(raw)["result"]["frozen_prepare"] == frozen_prepare
    for observations in (result["original_observations"], result["expanded_observations"]):
        assert len(observations) == 105
        payloads = [row["observation"]["result"] for row in observations]
        assert all(payload["context_text"] and payload["context_sha256"] for payload in payloads)
        assert any(payload["items"] for payload in payloads)
        for payload in payloads:
            for item in payload["items"]:
                assert item["page_id"] and item["path"] and item["evidence"]
                assert all({"source", "target", "signals"} == set(path) for path in item["graph_paths"])


def test_hybrid_dispatch_module_cli_consumes_its_opaque_capability_before_filesystem_failure(
    tmp_path: Path,
) -> None:
    """`python -m` must share the capability identity imported by the production runner."""
    request = {
        "schema_version": 1, "campaign_stage": "hybrid",
        "dense_ledger_sha256": operator.DENSE_LEDGER_DIGEST,
        "dense_source_head": operator.DENSE_SOURCE_HEAD,
        "hybrid_implementation_head": HEAD,
        "dense_ordinal_identities": [dict(row) for row in operator.HYBRID_DENSE_ORDINAL_IDENTITIES],
        "baseline": dict(operator.HYBRID_BASELINE),
        "candidates": [dict(row) for row in operator.HYBRID_CANDIDATES],
        "scale": 30000, "query_count": 105, "authorization": "none", "retention_days": 90,
        "replacement_for_run_id": None, "record_self_sha256": "",
    }
    request["record_self_sha256"] = operator.canonical_digest(request)
    candidate = dict(operator.HYBRID_CANDIDATES[0])
    member = {
        "schema_version": 1, "campaign_stage": "hybrid",
        "hybrid_request_sha256": request["record_self_sha256"],
        "dense_source_head": operator.DENSE_SOURCE_HEAD, "hybrid_implementation_head": HEAD,
        "role": "candidate", "config": candidate,
        "scale": 30000, "query_count": 105, "authorization": "none", "retention_days": 90,
        "replacement_for_run_id": None, "dispatch_identity": "phase07-hybrid/m20",
        "record_self_sha256": "",
    }
    member["record_self_sha256"] = operator.canonical_digest(member)
    bundle = {
        "schema_version": 1, "hybrid_request": request, "workflow_input": member,
        "replacement_for_run_id": None, "record_self_sha256": "",
    }
    bundle["record_self_sha256"] = operator.canonical_digest(bundle)
    execution = _locked_confirmation_environment()
    allocation = {
        "run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid",
        "job_allocation_nonce": "a" * 32,
    }
    bundle_file, execution_file, allocation_file = (tmp_path / "dispatch.json", tmp_path / "execution.json",
                                                      tmp_path / "allocation.json")
    for path, value in ((bundle_file, bundle), (execution_file, execution), (allocation_file, allocation)):
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    blocked_output = tmp_path / "blocked-output"
    blocked_output.write_text("not a directory", encoding="utf-8")
    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "eval.phase07_operator_gate", "hybrid-dispatch",
         "--dispatch-bundle", str(bundle_file), "--locked-execution", str(execution_file),
         "--allocation", str(allocation_file), "--output-dir", str(blocked_output)],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 1
    assert "unminted hybrid execution capability" not in result.stderr
    assert str(blocked_output) in result.stderr


def _reseal_hybrid_dispatch_bundle(bundle: dict) -> dict:
    value = deepcopy(bundle)
    value["hybrid_request"]["record_self_sha256"] = operator.canonical_digest(value["hybrid_request"])
    value["workflow_input"]["hybrid_request_sha256"] = value["hybrid_request"]["record_self_sha256"]
    value["workflow_input"]["record_self_sha256"] = operator.canonical_digest(value["workflow_input"])
    value["record_self_sha256"] = operator.canonical_digest(value)
    return value


def test_hybrid_dispatch_bundle_is_exact_current_head_authority_before_any_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-sealed member alone must never be hybrid dispatch authority."""
    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    bundle = _hybrid_dispatch_bundle(plan)
    assert operator.validate_hybrid_dispatch_bundle(bundle, expected_head=HEAD) == bundle["workflow_input"]
    for invalid in (
        bundle["workflow_input"],
        {**bundle, "unexpected": True},
        {**bundle, "replacement_for_run_id": 1},
        {**bundle, "hybrid_request": {**bundle["hybrid_request"], "record_self_sha256": "0" * 64}},
    ):
        with pytest.raises(ValueError):
            operator.validate_hybrid_dispatch_bundle(invalid, expected_head=HEAD)
    stale = deepcopy(bundle)
    stale["workflow_input"]["hybrid_implementation_head"] = "0" * 40
    stale["workflow_input"]["record_self_sha256"] = operator.canonical_digest(stale["workflow_input"])
    with pytest.raises(ValueError):
        operator.validate_hybrid_dispatch_bundle(stale, expected_head=HEAD)
    forged = deepcopy(bundle)
    forged["workflow_input"]["config"] = {
        "index_type": "hnsw_sq", "m": 16, "ef_construction": 300, "query_ef": 100,
    }
    forged["workflow_input"]["record_self_sha256"] = operator.canonical_digest(forged["workflow_input"])
    with pytest.raises(ValueError):
        operator.validate_hybrid_dispatch_bundle(forged, expected_head=HEAD)
    forged_request = deepcopy(bundle)
    forged_request["hybrid_request"]["dense_ordinal_identities"][0]["build_ids"] = ["0" * 64] * 3
    forged_request = _reseal_hybrid_dispatch_bundle(forged_request)
    with pytest.raises(ValueError):
        operator.validate_hybrid_dispatch_bundle(forged_request, expected_head=HEAD)
    generated = tmp_path / "generated"; generated.mkdir()
    (generated / "hybrid-m20.json").write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError):
        operator.validate_hybrid_workflow_inputs_dir(generated, expected_head=HEAD)
    (generated / "hybrid-m32.json").write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError):
        operator.validate_hybrid_workflow_inputs_dir(generated, expected_head=HEAD)
    (generated / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        operator.validate_hybrid_workflow_inputs_dir(generated, expected_head=HEAD)


def test_hybrid_preflight_dispatch_and_export_require_a_complete_raw_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production entry point must reach real hybrid work only after typed allocation."""
    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    bundle = _hybrid_dispatch_bundle(plan)
    preflight = operator.build_hybrid_preflight(bundle, expected_head=HEAD)
    branch = "feature/issue-50-dense-ann-recall"
    merge_ref = f"refs/heads/{branch}"
    preflight.update(branch=branch, worktree_root=str(ROOT), head_sha=HEAD)
    assert preflight["allowed_dirty_paths"] == []
    replies = {
        ("rev-parse", "--show-toplevel"): str(ROOT),
        ("branch", "--show-current"): branch,
        ("rev-parse", "HEAD"): HEAD,
        ("rev-parse", "--verify", "@{upstream}"): HEAD,
        ("config", "--get", f"branch.{branch}.remote"): "origin",
        ("config", "--get", f"branch.{branch}.merge"): merge_ref,
        ("ls-remote", "--exit-code", "--refs", "origin", merge_ref): f"{HEAD}\t{merge_ref}",
        ("status", "--porcelain=v1"): "",
    }
    monkeypatch.setattr(operator, "_git", lambda *args: replies[args])
    assert operator.validate_feature_worktree_preflight(preflight)["head_sha"] == HEAD
    calls: list[dict] = []
    result = operator.execute_hybrid_dispatch(
        bundle=bundle, locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
        work_dir=tmp_path / "work", runner=lambda **kwargs: calls.append(kwargs) or {"authorization": "none"},
    )
    assert result["authorization"] == "none" and len(calls) == 1
    for invalid in (bundle["workflow_input"], {**bundle, "extra": True}):
        calls.clear()
        with pytest.raises(ValueError):
            operator.execute_hybrid_dispatch(
                bundle=invalid, locked_execution=_locked_confirmation_environment(),
                allocation={"run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
                work_dir=tmp_path / "rejected", runner=lambda **kwargs: calls.append(kwargs),
            )
        assert calls == []
    raw_tree = tmp_path / "raw"; raw_tree.mkdir()
    for name in ("hybrid-request.json", "hybrid-ledger.json", "hybrid-result.json", "dispatch-bundle.json", "allocation.json"):
        (raw_tree / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        operator.validate_hybrid_artifact_tree(raw_tree)
    (raw_tree / "unexpected.txt").write_text("residual", encoding="utf-8")
    with pytest.raises(ValueError):
        operator.validate_hybrid_artifact_tree(raw_tree)


def test_feature_worktree_preflight_requires_exact_resolved_upstream_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requested upstream binding rejects divergence and absent tracking refs."""
    preflight = {
        "repository": "allenwoo713/obsidian_wiki_skill",
        "branch": "feature/issue-50-dense-ann-recall",
        "worktree_root": str(ROOT), "head_sha": HEAD, "allowed_dirty_paths": [],
        "workflow_name": "eval", "campaign_stage": "hybrid",
        "require_upstream_head": True,
    }
    branch = preflight["branch"]
    merge_ref = f"refs/heads/{branch}"
    replies = {
        ("rev-parse", "--show-toplevel"): str(ROOT),
        ("branch", "--show-current"): branch,
        ("rev-parse", "HEAD"): HEAD,
        ("rev-parse", "--verify", "@{upstream}"): HEAD,
        ("config", "--get", f"branch.{branch}.remote"): "origin",
        ("config", "--get", f"branch.{branch}.merge"): merge_ref,
        ("ls-remote", "--exit-code", "--refs", "origin", merge_ref): f"{HEAD}\t{merge_ref}",
        ("status", "--porcelain=v1"): "",
    }

    def diverged_git(*args: str) -> str:
        if args == ("rev-parse", "--verify", "@{upstream}"):
            return "0" * 40
        return replies[args]

    monkeypatch.setattr(operator, "_git", diverged_git)
    with pytest.raises(ValueError, match="upstream"):
        operator.validate_feature_worktree_preflight(preflight)

    request_file, ledger_file = tmp_path / "preflight-request.json", tmp_path / "preflight-ledger.json"
    request_file.write_text(json.dumps({**preflight, "ledger_path": str(ledger_file.resolve())}), encoding="utf-8")
    assert operator.main(["preflight", "--request-file", str(request_file), "--ledger-file", str(ledger_file)]) == 1
    assert not ledger_file.exists()

    def unresolved_git(*args: str) -> str:
        if args == ("rev-parse", "--verify", "@{upstream}"):
            raise subprocess.CalledProcessError(128, ["git", *args])
        return replies[args]

    monkeypatch.setattr(operator, "_git", unresolved_git)
    with pytest.raises(ValueError, match="upstream"):
        operator.validate_feature_worktree_preflight(preflight)


def test_feature_worktree_preflight_rejects_detached_current_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A detached checkout never obtains feature-worktree authority."""
    branch = "feature/issue-50-dense-ann-recall"
    merge_ref = f"refs/heads/{branch}"
    request = {
        "repository": "allenwoo713/obsidian_wiki_skill", "branch": branch,
        "worktree_root": str(ROOT), "head_sha": HEAD, "allowed_dirty_paths": [],
        "workflow_name": "eval", "campaign_stage": "hybrid", "require_upstream_head": True,
    }
    replies = {
        ("rev-parse", "--show-toplevel"): str(ROOT),
        ("branch", "--show-current"): "",
        ("rev-parse", "HEAD"): HEAD,
        ("rev-parse", "--verify", "@{upstream}"): HEAD,
        ("config", "--get", f"branch.{branch}.remote"): "origin",
        ("config", "--get", f"branch.{branch}.merge"): merge_ref,
        ("ls-remote", "--exit-code", "--refs", "origin", merge_ref): f"{HEAD}\t{merge_ref}",
        ("status", "--porcelain=v1"): "",
    }
    monkeypatch.setattr(operator, "_git", lambda *args: replies[args])
    with pytest.raises(ValueError, match="feature branch"):
        operator.validate_feature_worktree_preflight(request)


@pytest.mark.parametrize(
    ("mutation", "remote_reply"),
    (
        ("missing-remote", None),
        ("missing-merge", None),
        ("remote-command-failure", None),
        ("remote-empty", ""),
        ("remote-multiple-lines", f"{HEAD}\trefs/heads/feature/issue-50-dense-ann-recall\n{HEAD}\trefs/heads/other"),
        ("remote-non-sha", "not-a-sha\trefs/heads/feature/issue-50-dense-ann-recall"),
        ("remote-different", f"{'0' * 40}\trefs/heads/feature/issue-50-dense-ann-recall"),
    ),
)
def test_feature_worktree_preflight_requires_live_configured_remote_head(
    monkeypatch: pytest.MonkeyPatch, mutation: str, remote_reply: str | None,
) -> None:
    """A local tracking ref is insufficient without one matching live configured ref."""
    branch = "feature/issue-50-dense-ann-recall"
    merge_ref = f"refs/heads/{branch}"
    preflight = {
        "repository": "allenwoo713/obsidian_wiki_skill", "branch": branch,
        "worktree_root": str(ROOT), "head_sha": HEAD, "allowed_dirty_paths": [],
        "workflow_name": "eval", "campaign_stage": "hybrid", "require_upstream_head": True,
    }

    def fake_git(*args: str) -> str:
        replies = {
            ("rev-parse", "--show-toplevel"): str(ROOT),
            ("branch", "--show-current"): branch,
            ("rev-parse", "HEAD"): HEAD,
            ("rev-parse", "--verify", "@{upstream}"): HEAD,
            ("status", "--porcelain=v1"): "",
            ("config", "--get", f"branch.{branch}.remote"): "origin",
            ("config", "--get", f"branch.{branch}.merge"): merge_ref,
            ("ls-remote", "--exit-code", "--refs", "origin", merge_ref): f"{HEAD}\t{merge_ref}",
        }
        if mutation == "missing-remote" and args == ("config", "--get", f"branch.{branch}.remote"):
            return ""
        if mutation == "missing-merge" and args == ("config", "--get", f"branch.{branch}.merge"):
            return ""
        if mutation == "remote-command-failure" and args[:1] == ("ls-remote",):
            raise subprocess.CalledProcessError(2, ["git", *args])
        if args[:1] == ("ls-remote",):
            return remote_reply if remote_reply is not None else replies[args]
        return replies[args]

    monkeypatch.setattr(operator, "_git", fake_git)
    with pytest.raises(ValueError, match="upstream"):
        operator.validate_feature_worktree_preflight(preflight)


def test_feature_worktree_preflight_accepts_only_matching_live_configured_remote_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accepted path binds local HEAD, local upstream, and remote ref to one SHA."""
    branch = "feature/issue-50-dense-ann-recall"
    merge_ref = f"refs/heads/{branch}"
    replies = {
        ("rev-parse", "--show-toplevel"): str(ROOT),
        ("branch", "--show-current"): branch,
        ("rev-parse", "HEAD"): HEAD,
        ("rev-parse", "--verify", "@{upstream}"): HEAD,
        ("status", "--porcelain=v1"): "",
        ("config", "--get", f"branch.{branch}.remote"): "origin",
        ("config", "--get", f"branch.{branch}.merge"): merge_ref,
        ("ls-remote", "--exit-code", "--refs", "origin", merge_ref): f"{HEAD}\t{merge_ref}",
    }
    monkeypatch.setattr(operator, "_git", lambda *args: replies[args])
    request = {
        "repository": "allenwoo713/obsidian_wiki_skill", "branch": branch,
        "worktree_root": str(ROOT), "head_sha": HEAD, "allowed_dirty_paths": [],
        "workflow_name": "eval", "campaign_stage": "hybrid", "require_upstream_head": True,
    }
    assert operator.validate_feature_worktree_preflight(request)["head_sha"] == HEAD


def test_hybrid_cli_routes_sealed_dispatch_to_numeric_raw_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The module CLI is a production route, not a dormant helper collection."""
    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    bundle_file = tmp_path / "dispatch.json"; bundle_file.write_text(json.dumps(_hybrid_dispatch_bundle(plan)), encoding="utf-8")
    execution_file = tmp_path / "execution.json"; execution_file.write_text(json.dumps(_locked_confirmation_environment()), encoding="utf-8")
    allocation_file = tmp_path / "allocation.json"; allocation_file.write_text(json.dumps({"run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32}), encoding="utf-8")
    dispatched: list[dict] = []
    monkeypatch.setattr(operator, "execute_hybrid_dispatch", lambda **kwargs: dispatched.append(kwargs) or {"authorization": "none"}, raising=False)
    assert operator.main([
        "hybrid-dispatch", "--dispatch-bundle", str(bundle_file), "--locked-execution", str(execution_file),
        "--allocation", str(allocation_file), "--output-dir", str(tmp_path / "raw"),
    ]) == 0
    assert len(dispatched) == 1


def test_retired_direct_hybrid_packet_api_and_cli_options_are_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raw artifact directory is the only packet export authority."""
    from eval import phase07_ann_campaign as campaign

    assert not hasattr(campaign, "hybrid_packet_from_result")
    assert not hasattr(campaign, "validate_hybrid_packet")
    monkeypatch.setattr(sys, "argv", ["phase07_ann_campaign.py", "export-hybrid-packet",
                                       "--result-file", str(tmp_path / "result.json"),
                                       "--workflow-input", str(tmp_path / "member.json"),
                                       "--output", str(tmp_path / "packet.json")])
    with pytest.raises(SystemExit) as exited:
        campaign.main()
    assert exited.value.code != 0


def _hybrid_gate_metrics(**overrides: object) -> dict:
    metrics = {
        "functional_final_retrieval_ann_overlap_at_10": 1.0,
        "page_recall_at_5": 1.0, "evidence_recall_at_10": 1.0, "exact_lookup_hit_at_3": 1.0, "mrr_at_10": 1.0,
        "citation_violation_count": 0, "context_overflow_count": 0,
        "budget_violation_count": 0, "graph_unsupported_count": 0,
    }
    metrics.update(overrides)
    return metrics


def test_hybrid_gates_are_recomputable_and_numeric_regressions_are_rejected_not_authorized() -> None:
    """Absolute and paired evidence are separate, with a per-candidate reject verdict."""
    baseline = _hybrid_gate_metrics()
    original = {"baseline": baseline, "candidate": {**baseline, "page_recall_at_5": 0.97}}
    paired = {"baseline": baseline, "candidate": baseline}
    verdict = operator.recompute_hybrid_gate_verdicts(
        original_absolute=original, expanded_paired=paired,
        committed_baseline=_committed_hybrid_floors(), baselines_sha256=_committed_baselines_sha256(),
    )
    assert verdict["candidate_verdict"] == "rejected-candidate"
    assert verdict["authorization"] == "none"
    assert set(verdict) >= {"original_absolute_gate", "paired_30k_non_regression_gate", "write_graph_artifact"}


def test_hybrid_gate_rejection_table_covers_absolute_paired_and_zero_tolerance_contracts() -> None:
    """Every quality axis is a candidate verdict, never an authorization switch."""
    base = {
        **_hybrid_gate_metrics(),
    }
    for stratum, regression in (
        ("original_absolute", {"page_recall_at_5": 0.97}),
        ("paired_30k", {"evidence_recall_at_10": 0.97}),
        ("original_absolute", {"mrr_at_10": 0.97}),
        ("paired_30k", {"functional_final_retrieval_ann_overlap_at_10": 0.97}),
        ("original_absolute", {"citation_violation_count": 1}),
        ("paired_30k", {"context_overflow_count": 1}),
        ("original_absolute", {"budget_violation_count": 1}),
        ("paired_30k", {"graph_unsupported_count": 1}),
    ):
        verdict = operator.recompute_hybrid_gate_verdicts(
            original_absolute={"baseline": base, "candidate": {**base, **regression} if stratum == "original_absolute" else base},
            expanded_paired={"baseline": base, "candidate": {**base, **regression} if stratum == "paired_30k" else base},
            committed_baseline=_committed_hybrid_floors(), baselines_sha256=_committed_baselines_sha256(),
        )
        assert verdict["candidate_verdict"] == "rejected-candidate"
        assert verdict["authorization"] == "none"


def test_minted_hybrid_execution_capability_is_bound_to_one_two_index_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    captured: list[object] = []
    result = operator.execute_hybrid_dispatch(
        bundle=_hybrid_dispatch_bundle(plan), locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
        work_dir=tmp_path / "minted-graph",
        runner=lambda **kwargs: captured.append(kwargs["capability"]) or {"authorization": "none", "required_index_count": 2,
                                                                         "role": "candidate"},
    )
    assert result["required_index_count"] == 2 and result["role"] == "candidate"
    assert len(captured) == 1 and not isinstance(captured[0], dict)


def _faithful_hybrid_result(*, query: str, page_id: str, context: str, bad_citation: bool = False,
                            overflow: bool = False, budget_violation: bool = False,
                            unsupported_graph: bool = False) -> SimpleNamespace:
    """Small object graph with the fields produced by public ``hybrid_search``."""
    path = "Wiki/invalid.md" if bad_citation else "Wiki/valid.md"
    item = SimpleNamespace(
        page_id=page_id, path=path, scope="section", evidence=[] if unsupported_graph else [SimpleNamespace(chunk_id="e1")],
        graph_paths=(), inclusion_reason="graph_expansion" if unsupported_graph else "dense_retrieval",
    )
    bundle = SimpleNamespace(
        items=[item], context_text=context if not bad_citation else context,
        token_count=11 if overflow else 5, effective_budget_tokens=10,
        budget_contract_violations=lambda: ["over"] if budget_violation else [],
    )
    return SimpleNamespace(
        query=query, candidates=[SimpleNamespace(page_id=page_id)], bundle=bundle,
        plan=SimpleNamespace(to_json=lambda: {"intent": "lookup"}),
    )


def test_hybrid_aggregates_real_result_metrics_and_rejects_fact_citation_budget_graph_failures() -> None:
    """Hybrid gates must consume HybridResult facts, not page aliases or hard-coded zero counters."""
    import eval.run_eval as run_eval

    specification = [{"query": "needle", "relevant_pages": ["page-1"], "required_facts": ["must appear"]}]
    baseline = [_faithful_hybrid_result(query="needle", page_id="page-1", context="must appear [来源: Wiki/valid.md]")]
    candidate = [_faithful_hybrid_result(
        query="needle", page_id="page-1", context="page hit only", bad_citation=True,
        overflow=True, budget_violation=True, unsupported_graph=True,
    )]
    aggregate = run_eval.aggregate_hybrid_result_metrics(
        specifications=specification, baseline_results=baseline, candidate_results=candidate,
    )
    assert aggregate["candidate"]["page_recall_at_5"] == 1.0
    assert aggregate["candidate"]["evidence_recall_at_10"] == 0.0
    assert aggregate["candidate"]["exact_lookup_hit_at_3"] == 1.0
    assert aggregate["candidate"]["citation_violation_count"] == 1
    assert aggregate["candidate"]["context_overflow_count"] == 1
    assert aggregate["candidate"]["budget_violation_count"] == 1
    assert aggregate["candidate"]["graph_unsupported_count"] == 1
    verdict = operator.recompute_hybrid_gate_verdicts(
        original_absolute=aggregate, expanded_paired=aggregate,
        committed_baseline=_committed_hybrid_floors(), baselines_sha256=_committed_baselines_sha256(),
    )
    assert verdict["candidate_verdict"] == "rejected-candidate"


def test_live_and_serialized_hybrid_metrics_agree_for_case_variant_page_identities() -> None:
    """Serialization must use the same page-identity semantics as live HybridResult evaluation."""
    import eval.run_eval as run_eval

    specification = [{"query": "needle", "relevant_pages": ["foo.md"], "required_facts": ["fact"]}]
    live = _faithful_hybrid_result(query="needle", page_id="Wiki/Foo.MD", context="fact [来源: Wiki/valid.md]")
    live_metrics = run_eval.aggregate_hybrid_result_metrics(
        specifications=specification, baseline_results=[live], candidate_results=[live],
    )
    payload = {
        "query": "needle", "plan": {"intent": "lookup"}, "pages": ["Wiki/Foo.MD"],
        "items": [{"page_id": "Wiki/Foo.MD", "path": "Wiki/valid.md", "scope": "section",
                   "inclusion_reason": "dense_retrieval", "evidence": ["e1"], "graph_paths": []}],
        "context_text": "fact [来源: Wiki/valid.md]", "context_sha256": hashlib.sha256("fact [来源: Wiki/valid.md]".encode()).hexdigest(),
        "token_count": 5,
        "budget": {"requested_base_budget_tokens": 10, "budget_multiplier": 1.0,
                   "effective_budget_tokens": 10, "hard_max_tokens": 20,
                   "budget_policy": "context_mode_multiplier_v1", "max_context_tokens": 10},
    }
    serialized = run_eval.aggregate_hybrid_serialized_metrics(
        specifications=specification,
        observations=[{"baseline": {"result": payload, "duration_ms": 1.0},
                       "candidate": {"result": payload, "duration_ms": 1.0}}],
    )
    expected_metric_keys = {
        "functional_final_retrieval_ann_overlap_at_10", "page_recall_at_5", "evidence_recall_at_10",
        "exact_lookup_hit_at_3", "mrr_at_10", "citation_violation_count", "context_overflow_count",
        "budget_violation_count", "graph_unsupported_count",
    }
    assert set(live_metrics["baseline"]) == expected_metric_keys
    assert set(serialized["baseline"]) == expected_metric_keys
    assert serialized == live_metrics


def _committed_baselines_sha256() -> str:
    return hashlib.sha256((ROOT / "eval" / "baselines.json").read_bytes()).hexdigest()


def _committed_hybrid_floors() -> dict[str, float | int]:
    quality = json.loads((ROOT / "eval" / "baselines.json").read_text(encoding="utf-8"))["quality"]
    return {
        # ``functional_final_retrieval_ann_overlap_at_10`` is a distinct
        # hybrid contract, so it must never be silently substituted with ANN
        # recall.  The committed original-absolute floor is exact lookup.
        "page_recall_at_5": quality["page_recall_at_5"],
        "evidence_recall_at_10": quality["evidence_recall_at_10"], "mrr_at_10": quality["mrr_at_10"],
        "exact_lookup_hit_at_3": quality["exact_lookup_hit_at_3"],
        "citation_violation_count": 0, "context_overflow_count": 0,
        "budget_violation_count": 0, "graph_unsupported_count": 0,
    }


def test_hybrid_original_absolute_fails_when_observed_baseline_is_below_committed_floor_or_tampered() -> None:
    """The original stratum is also an absolute baseline sanity check; paired is relative only."""
    floors = _committed_hybrid_floors()
    full = {**_hybrid_gate_metrics(), "page_recall_at_5": floors["page_recall_at_5"],
            "evidence_recall_at_10": floors["evidence_recall_at_10"], "mrr_at_10": floors["mrr_at_10"],
            "exact_lookup_hit_at_3": floors["exact_lookup_hit_at_3"]}
    weak = {**full, "page_recall_at_5": floors["page_recall_at_5"] - 0.03,
            "exact_lookup_hit_at_3": floors["exact_lookup_hit_at_3"] - 0.03}
    verdict = operator.recompute_hybrid_gate_verdicts(
        original_absolute={"baseline": weak, "candidate": weak},
        expanded_paired={"baseline": full, "candidate": full},
        committed_baseline=floors, baselines_sha256=_committed_baselines_sha256(),
    )
    assert verdict["candidate_verdict"] == "rejected-candidate"
    with pytest.raises(ValueError):
        operator.recompute_hybrid_gate_verdicts(
            original_absolute={"baseline": full, "candidate": full},
            expanded_paired={"baseline": full, "candidate": full},
            committed_baseline={**floors, "mrr_at_10": 0.0}, baselines_sha256="0" * 64,
        )


def test_self_sealed_hybrid_member_never_reaches_build_spy_without_canonical_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid self-digest is evidence only; dispatch membership is the build authority."""
    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    member = deepcopy(plan["workflow_inputs"][0])
    member["record_self_sha256"] = operator.canonical_digest(member)
    calls: list[dict] = []
    with pytest.raises(ValueError):
        operator.execute_hybrid_dispatch(
            bundle=member, locked_execution=_locked_confirmation_environment(),
            allocation={"run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
            work_dir=tmp_path / "rejected", runner=lambda **kwargs: calls.append(kwargs),
        )
    assert calls == []


def test_hybrid_campaign_binds_actual_expanded_tree_not_fixture_or_label_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """One role builds two indexes and invokes public hybrid search twice per query."""
    import eval.run_eval as run_eval
    import build_index

    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(run_eval, "write_graph_artifact", lambda *_args: None)
    storage_builds: list[Path] = []
    real_build_storage_contract = build_index.build_storage_contract

    def track_build_storage_contract(*args, **kwargs):
        storage_builds.append(Path(args[1]))
        return real_build_storage_contract(*args, **kwargs)

    monkeypatch.setattr(build_index, "build_storage_contract", track_build_storage_contract)
    hybrid_calls: list[Path] = []
    real_hybrid_search = run_eval.hybrid_search

    def track_hybrid_search(*args, **kwargs):
        hybrid_calls.append(Path(kwargs["wiki_dir"]))
        return real_hybrid_search(*args, **kwargs)

    monkeypatch.setattr(run_eval, "hybrid_search", track_hybrid_search)
    # A direct dict/member runner is deliberately absent.  The real, private
    # runner receives its opaque capability only from the fully validated
    # dispatch boundary; the injected encoder/limit are the finite test seam.
    assert not hasattr(run_eval, "run_phase07_hybrid_campaign")
    observed_capabilities: list[object] = []
    def production_test_runner(*, capability: object, work_dir: Path) -> dict:
        observed_capabilities.append(capability)
        return run_eval._run_phase07_hybrid_campaign_with_capability(
            capability=capability, work_dir=work_dir, embed=_embed384(), query_limit=2,
        )
    result = operator.execute_hybrid_dispatch(
        bundle=_hybrid_dispatch_bundle(plan), locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8,
                    "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
        work_dir=tmp_path, runner=production_test_runner,
    )
    assert len(observed_capabilities) == 1 and not isinstance(observed_capabilities[0], dict)
    expanded = tmp_path / "hybrid-m20-m20" / "expanded" / "Wiki"
    actual_digest = canonical_content_tree_sha256(expanded)
    actual_count = len([path for path in expanded.rglob("*") if path.is_file()])
    expected = run_eval.expected_phase07_expanded_corpus_identity(
        fixture_root=ROOT / "tests" / "fixtures" / "wiki", target_size=actual_count, test_only=True,
    )
    assert expected == {"expanded_content_tree_sha256": actual_digest, "expanded_member_count": actual_count}
    assert result["expanded_content_tree_sha256"] == actual_digest
    assert result["expanded_member_count"] == actual_count
    assert result["role"] == "m20"
    assert result["config"] == {"index_type": "hnsw_sq", "m": 20, "ef_construction": 300, "query_ef": 300}
    assert "aggregate_metrics" not in result and "candidate_verdict" not in result
    assert len(storage_builds) == 2
    assert len(hybrid_calls) == 4
    assert sum(path == expanded for path in hybrid_calls) == 2
    original = tmp_path / "hybrid-m20-m20" / "original" / "Wiki"
    assert sum(path == original for path in hybrid_calls) == 2
    assert result["hybrid_invocation"] == {
        "entrypoint": "query.hybrid_search", "candidate_aware_public_arguments": False,
        "original_calls": 2, "expanded_calls": 2,
    }
    assert result["campaign_progress"] == {
        "role": "m20", "original_completed": 2, "expanded_completed": 2,
    }
    for observations in (result["original_observations"], result["expanded_observations"]):
        assert len(observations) == 2
        assert all(set(row) == {"ordinal", "query_sha256", "observation"} for row in observations)
    progress = capsys.readouterr().out
    assert "[phase07-hybrid] role=m20 build=original complete" in progress
    assert "[phase07-hybrid] role=m20 build=expanded complete" in progress
    assert "[phase07-hybrid] role=m20 original_queries=2/2 complete" in progress
    assert "[phase07-hybrid] role=m20 expanded_queries=2/2 complete" in progress
    assert "authorization" not in progress


def test_hybrid_campaign_emits_safe_elapsed_stage_markers_through_real_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One role reports only completed production-build stages, without altering its result."""
    import eval.run_eval as run_eval

    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(run_eval, "write_graph_artifact", lambda *_args: None)
    markers: list[dict] = []

    def runner(*, capability: object, work_dir: Path) -> dict:
        return run_eval._run_phase07_hybrid_campaign_with_capability(
            capability=capability,
            work_dir=work_dir,
            embed=_embed384(),
            query_limit=2,
            progress_sink=markers.append,
        )

    result = operator.execute_hybrid_dispatch(
        bundle=_hybrid_dispatch_bundle(plan),
        locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8,
                    "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
        work_dir=tmp_path,
        runner=runner,
    )

    assert result["hybrid_invocation"] == {
        "entrypoint": "query.hybrid_search",
        "candidate_aware_public_arguments": False,
        "original_calls": 2,
        "expanded_calls": 2,
    }
    assert markers
    expected_stages = [
        "scan_chunk",
        "dense_embedding",
        "lance_fts_persist",
        "hnsw_create_index",
        "validation_seal_publication",
        "graph_artifact",
        "load",
    ]
    for corpus in ("original", "expanded"):
        corpus_markers = [marker for marker in markers if marker["corpus"] == corpus]
        assert [marker["stage"] for marker in corpus_markers] == expected_stages
        assert all(set(marker) == {"role", "corpus", "stage", "state", "elapsed_ms"}
                   for marker in corpus_markers)
        assert all(marker["role"] == "m20" and marker["state"] == "complete"
                   and isinstance(marker["elapsed_ms"], (int, float))
                   and marker["elapsed_ms"] >= 0 for marker in corpus_markers)
        assert [marker["elapsed_ms"] for marker in corpus_markers] == sorted(
            marker["elapsed_ms"] for marker in corpus_markers
        )
    forbidden = {"path", "text", "query", "model", "token", "secret", "verdict", "authorization"}
    assert not any(forbidden & set(marker) for marker in markers)


def test_hybrid_stage_sink_is_optional_and_propagates_instrumentation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent instrumentation remains inert; an explicit sink failure cannot authorize a run."""
    import eval.run_eval as run_eval

    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(run_eval, "write_graph_artifact", lambda *_args: None)
    bundle = _hybrid_dispatch_bundle(plan)
    allocation = {"run_id": 7, "run_attempt": 1, "job_id": 8,
                  "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32}

    def run_with_sink(sink=None) -> dict:
        def runner(*, capability: object, work_dir: Path) -> dict:
            kwargs = {"capability": capability, "work_dir": work_dir,
                      "embed": _embed384(), "query_limit": 1}
            if sink is not None:
                kwargs["progress_sink"] = sink
            return run_eval._run_phase07_hybrid_campaign_with_capability(**kwargs)

        return operator.execute_hybrid_dispatch(
            bundle=bundle, locked_execution=_locked_confirmation_environment(),
            allocation=allocation, work_dir=tmp_path, runner=runner,
        )

    printed: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        run_eval,
        "print",
        lambda *args, **kwargs: printed.append((args, kwargs)),
        raising=False,
    )
    without_sink = run_with_sink()
    assert without_sink["query_count"] == 1
    assert without_sink["campaign_progress"] == {
        "role": "m20", "original_completed": 1, "expanded_completed": 1,
    }
    marker_prints = [
        (args, kwargs) for args, kwargs in printed
        if args and str(args[0]).startswith("[phase07-hybrid-progress] ")
    ]
    assert len(marker_prints) == 14
    assert all(kwargs.get("flush") is True for _args, kwargs in marker_prints)
    with pytest.raises(RuntimeError, match="timing sink failed"):
        run_with_sink(lambda _marker: (_ for _ in ()).throw(RuntimeError("timing sink failed")))


def test_export_hybrid_packet_rejects_legacy_direct_result_and_workflow_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a full raw artifact directory can be exported; direct JSON cannot bypass reconstruction."""
    from eval import phase07_ann_campaign as campaign

    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    tree = _make_valid_test_hybrid_raw_tree(
        tmp_path / "artifact", dispatch_bundle=_hybrid_dispatch_bundle(plan),
        locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
    )
    result = json.loads((tree / "hybrid-result.json").read_text(encoding="utf-8"))
    legacy = {key: result[key] for key in {
        "schema_version", "campaign_stage", "bundle_sha256", "role", "config", "planned_scale", "executed_scale",
        "query_count", "authorization", "original_observations", "expanded_observations", "hybrid_invocation",
    }}
    result_file, member_file = tmp_path / "legacy-result.json", tmp_path / "member.json"
    result_file.write_text(json.dumps(legacy), encoding="utf-8")
    member_file.write_text(json.dumps(plan["workflow_inputs"][0]), encoding="utf-8")
    with pytest.raises(TypeError):
        campaign.export_hybrid_packet(result_file=result_file, workflow_input_file=member_file, output=tmp_path / "packet.json")


def test_hybrid_payload_allows_empty_and_community_rows_but_requires_full_production_schema() -> None:
    """Empty final retrieval and community reports are normal, unlike a malformed payload object."""
    from eval import phase07_ann_campaign as campaign

    query = "empty is a valid answer"
    budget = {"requested_base_budget_tokens": 10, "budget_multiplier": 1.0,
              "effective_budget_tokens": 10, "hard_max_tokens": 20,
              "budget_policy": "context_mode_multiplier_v1", "max_context_tokens": 10}
    empty = {"query": query, "plan": {}, "pages": [], "items": [], "context_text": "",
             "context_sha256": hashlib.sha256(b"").hexdigest(), "token_count": 0, "budget": budget}
    community = {"query": query, "plan": {}, "pages": [], "items": [{
        "page_id": "community", "path": "community/report", "scope": "report", "inclusion_reason": "global_community_report", "evidence": [], "graph_paths": [],
    }], "context_text": "report", "context_sha256": hashlib.sha256(b"report").hexdigest(), "token_count": 1, "budget": budget}
    normal = {"query": query, "plan": {}, "pages": ["page"], "items": [{
        "page_id": "page", "path": "Wiki/page.md", "scope": "section", "inclusion_reason": "dense_retrieval", "evidence": ["chunk"], "graph_paths": [],
    }], "context_text": "ctx", "context_sha256": hashlib.sha256(b"ctx").hexdigest(), "token_count": 1, "budget": budget}
    assert campaign._validate_hybrid_result_payload(empty, query=query) is None
    assert campaign._validate_hybrid_result_payload(community, query=query) is None
    assert campaign._validate_hybrid_result_payload(normal, query=query) is None
    with pytest.raises(ValueError):
        campaign._validate_hybrid_result_payload({"query": query}, query=query)
    for incomplete in (
        {key: value for key, value in normal.items() if key != "context_text"},
        {key: value for key, value in normal.items() if key != "budget"},
        {**normal, "items": [{key: value for key, value in normal["items"][0].items() if key != "inclusion_reason"}]},
    ):
        with pytest.raises(ValueError):
            campaign._validate_hybrid_result_payload(incomplete, query=query)


def test_hybrid_artifact_payload_is_exact_production_serializer_shape_without_duplicate_quality_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw trees carry one canonical result payload, not a second mutable evidence copy."""
    from eval import phase07_ann_campaign as campaign

    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    tree = _make_valid_test_hybrid_raw_tree(
        tmp_path / "artifact", dispatch_bundle=_hybrid_dispatch_bundle(plan),
        locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
    )
    result_path = tree / "hybrid-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for rows in (result["original_observations"], result["expanded_observations"]):
        for row in rows:
            observation = row["observation"]
            payload = observation["result"]
            assert set(payload) == {"query", "plan", "pages", "items", "context_sha256", "context_text", "token_count", "budget"}
            assert set(observation) == {"result", "duration_ms"}
            assert all("inclusion_reason" in item for item in payload["items"])
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _reseal_test_hybrid_raw_tree(tree)
    assert campaign.validate_hybrid_artifact_tree(tree)["config"]["m"] == 20


def test_hybrid_payload_validator_requires_exact_single_raw_schema_and_rejects_resealed_field_mutations() -> None:
    """Context, budget, inclusion reason and graph paths are raw evidence, not optional decorations."""
    from eval import phase07_ann_campaign as campaign

    query = "strict raw schema"
    payload = {
        "query": query, "plan": {"intent": "lookup"}, "pages": ["Wiki/page.md"],
        "items": [{"page_id": "page", "path": "Wiki/page.md", "scope": "section",
                   "inclusion_reason": "dense_retrieval", "evidence": ["chunk"], "graph_paths": []}],
        "context_text": "fact [来源: Wiki/page.md]",
        "context_sha256": hashlib.sha256("fact [来源: Wiki/page.md]".encode()).hexdigest(), "token_count": 1,
        "budget": {"requested_base_budget_tokens": 10, "budget_multiplier": 1.0,
                   "effective_budget_tokens": 10, "hard_max_tokens": 20,
                   "budget_policy": "context_mode_multiplier_v1", "max_context_tokens": 10},
    }
    assert campaign._validate_hybrid_result_payload(payload, query=query) is None
    mutations = [
        {key: value for key, value in payload.items() if key != "context_text"},
        {**payload, "extra": "resealed"},
        {**payload, "graph_validated_count": 0},
        {**payload, "budget": {key: value for key, value in payload["budget"].items() if key != "hard_max_tokens"}},
        {**payload, "items": [{**payload["items"][0], "inclusion_reason": ""}]},
        {**payload, "items": [{**payload["items"][0], "graph_paths": [{"source": "a"}]}]},
    ]
    for mutated in mutations:
        with pytest.raises(ValueError):
            campaign._validate_hybrid_result_payload(mutated, query=query)


@pytest.mark.parametrize("kind", ("empty", "community", "normal"))
def test_full_hybrid_artifact_accepts_legal_empty_community_and_normal_payloads_from_single_raw_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    """Legal final retrieval variants stay valid after the whole artifact is resealed."""
    from eval import phase07_ann_campaign as campaign
    import eval.run_eval as run_eval

    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    tree = _make_valid_test_hybrid_raw_tree(
        tmp_path / kind, dispatch_bundle=_hybrid_dispatch_bundle(plan),
        locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
    )
    result_path = tree / "hybrid-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    target = result["original_observations"][0]["observation"]["result"]
    if kind == "empty":
        target.update({"pages": [], "items": [], "context_text": "", "context_sha256": hashlib.sha256(b"").hexdigest(), "token_count": 0})
    elif kind == "community":
        text = "community report"
        target.update({"pages": [], "items": [{"page_id": "community", "path": "community/report", "scope": "report", "inclusion_reason": "global_community_report", "evidence": [], "graph_paths": []}], "context_text": text, "context_sha256": hashlib.sha256(text.encode()).hexdigest()})
    # Normal is the production-shaped fixture payload unchanged.  Role
    # artifacts intentionally carry no aggregate or verdict authority.
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _reseal_test_hybrid_raw_tree(tree)
    assert campaign.validate_hybrid_artifact_tree(tree)["role"] == "m20"


@pytest.mark.parametrize("mutation", ("budget", "inclusion_reason", "graph_paths"))
def test_resealed_single_payload_evidence_mutations_fail_the_full_hybrid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    """Self-digests do not make a changed raw context/budget/item semantically valid."""
    from eval import phase07_ann_campaign as campaign

    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    tree = _make_valid_test_hybrid_raw_tree(
        tmp_path / mutation, dispatch_bundle=_hybrid_dispatch_bundle(plan),
        locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
    )
    result_path = tree / "hybrid-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    payload = result["original_observations"][0]["observation"]["result"]
    if mutation == "budget":
        payload["budget"]["hard_max_tokens"] = -1
    elif mutation == "inclusion_reason":
        payload["items"][0]["inclusion_reason"] = ""
    else:
        payload["items"][0]["graph_paths"] = [{"source": "only-source"}]
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _reseal_test_hybrid_raw_tree(tree)
    with pytest.raises(ValueError):
        campaign.validate_hybrid_artifact_tree(tree)


def test_hybrid_role_artifact_rejects_smuggled_aggregate_evidence_after_full_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pairwise gates are reconciliation-only and cannot appear in one role artifact."""
    from eval import phase07_ann_campaign as campaign

    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    tree = _make_valid_test_hybrid_raw_tree(
        tmp_path / "artifact", dispatch_bundle=_hybrid_dispatch_bundle(plan),
        locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
    )
    assert campaign.validate_hybrid_artifact_tree(tree)["role"] == "m20"
    result_path = tree / "hybrid-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["aggregate_metrics"] = {"forged": "quality authority"}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _reseal_test_hybrid_raw_tree(tree)
    with pytest.raises(ValueError):
        campaign.validate_hybrid_artifact_tree(tree)


@pytest.mark.parametrize("location", ("dispatch", "result", "ledger", "packet", "source-after"))
def test_frozen_prepare_is_one_exact_object_across_production_artifact_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str,
) -> None:
    """A resealed role tree cannot substitute or mutate its collector-bound base."""
    from eval import phase07_ann_campaign as campaign

    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    tree = _make_valid_test_hybrid_raw_tree(
        tmp_path / location, dispatch_bundle=_hybrid_dispatch_bundle(plan),
        locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8,
                    "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
    )
    assert campaign.validate_hybrid_artifact_tree(tree)["frozen_prepare"] == _frozen_prepare_identity()
    target = tree / ({
        "dispatch": "dispatch-bundle.json", "result": "hybrid-result.json",
        "ledger": "hybrid-ledger.json", "packet": "hybrid-packet.json",
        "source-after": "hybrid-result.json",
    }[location])
    document = json.loads(target.read_text(encoding="utf-8"))
    if location == "source-after":
        document["source_after_sha256"] = "0" * 64
    else:
        document["frozen_prepare"]["base_tree_sha256"] = "0" * 64
    _seal_test_json(target, document)
    if location in {"dispatch", "result", "ledger"}:
        _reseal_test_hybrid_raw_tree(tree)
    with pytest.raises(ValueError):
        campaign.validate_hybrid_artifact_tree(tree)


def _small_expanded_corpus(root: Path) -> Path:
    """Deterministic test-only analogue of the public distractor expansion."""
    shutil.copytree(ROOT / "tests" / "fixtures" / "wiki", root)
    pages = sorted(root.rglob("*.md"))
    for ordinal in range(2):
        (root / "phase07_distractors").mkdir(exist_ok=True)
        (root / "phase07_distractors" / f"hybrid-{ordinal:05d}.md").write_text(
            pages[ordinal % len(pages)].read_text(encoding="utf-8") + f"\n\nphase07 hybrid distractor {ordinal}\n",
            encoding="utf-8",
        )
    return root


def test_raw_hybrid_artifact_binds_expanded_content_identity_and_member_count_after_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A label digest or wrong count cannot stand in for the fixed recipe-derived corpus identity."""
    from eval import phase07_ann_campaign as campaign
    import eval.run_eval as run_eval

    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    tree = _make_valid_test_hybrid_raw_tree(
        tmp_path / "artifact", dispatch_bundle=_hybrid_dispatch_bundle(plan),
        locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
    )
    expanded = _small_expanded_corpus(tmp_path / "expanded")
    small_expected = run_eval.expected_phase07_expanded_corpus_identity(
        fixture_root=ROOT / "tests" / "fixtures" / "wiki",
        target_size=len([path for path in expanded.rglob("*") if path.is_file()]), test_only=True,
    )
    assert small_expected == {
        "expanded_content_tree_sha256": canonical_content_tree_sha256(expanded),
        "expanded_member_count": len([path for path in expanded.rglob("*") if path.is_file()]),
    }
    expected = run_eval.expected_phase07_expanded_corpus_identity(
        fixture_root=ROOT / "tests" / "fixtures" / "wiki", target_size=30000,
    )
    result_path = tree / "hybrid-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(expected)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _reseal_test_hybrid_raw_tree(tree)
    assert campaign.validate_hybrid_artifact_tree(tree)["config"]["m"] == 20
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["expanded_content_tree_sha256"] = canonical_digest({"label": "30k-expanded"})
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _reseal_test_hybrid_raw_tree(tree)
    with pytest.raises(ValueError):
        campaign.validate_hybrid_artifact_tree(tree)
    result.update(expected)
    result["expanded_member_count"] -= 1
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _reseal_test_hybrid_raw_tree(tree)
    with pytest.raises(ValueError):
        campaign.validate_hybrid_artifact_tree(tree)


_HYBRID_RAW_FILES = (
    "hybrid-request.json", "hybrid-ledger.json", "hybrid-result.json",
    "dispatch-bundle.json", "locked-execution.json", "allocation.json",
)
_HYBRID_RAW_LEAF_FILES = tuple(name for name in _HYBRID_RAW_FILES if name != "hybrid-ledger.json")


def _seal_test_json(path: Path, value: dict) -> None:
    value["record_self_sha256"] = canonical_digest(value)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _hybrid_raw_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for name in _HYBRID_RAW_FILES:
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update((root / name).read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def _hybrid_raw_file_sha256s(root: Path) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _HYBRID_RAW_FILES}


def _hybrid_raw_leaf_sha256s(root: Path) -> dict[str, str]:
    """The ledger cannot self-reference its own serialized digest."""
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _HYBRID_RAW_LEAF_FILES}


def _reseal_test_hybrid_raw_tree(root: Path) -> None:
    result_path = root / "hybrid-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _seal_test_json(result_path, result)
    ledger_path = root / "hybrid-ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    ledger["raw_leaf_sha256s"] = _hybrid_raw_leaf_sha256s(root)
    _seal_test_json(ledger_path, ledger)
    packet_path = root / "hybrid-packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["raw_tree_sha256"] = _hybrid_raw_tree_sha256(root)
    packet["raw_file_sha256s"] = _hybrid_raw_file_sha256s(root)
    packet["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    packet["ledger_sha256"] = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    _seal_test_json(packet_path, packet)


def _strict_test_hybrid_payload(*, query: str, ordinal: int, variant: str) -> dict:
    """Mirror ``run_phase07_hybrid_campaign``'s public result payload shape."""
    page_id = f"fixture-page-{ordinal:03d}"
    return {
        "query": query,
        "plan": {"intent": "lookup", "retrieval_mode": "hybrid", "fixture_variant": variant},
        "pages": [page_id],
        "items": [{
            "page_id": page_id, "path": f"Wiki/fixture-{ordinal:03d}.md", "scope": "section",
            "inclusion_reason": "dense_retrieval",
            "evidence": [f"fixture-{ordinal:03d}#evidence"],
            "graph_paths": [{"source": page_id, "target": "fixture-root", "signals": ["fixture"]}],
        }],
        "context_text": f"{variant}-context-{ordinal}",
        "context_sha256": hashlib.sha256(f"{variant}-context-{ordinal}".encode("utf-8")).hexdigest(),
        "token_count": 128,
        "budget": {
            "requested_base_budget_tokens": 128, "budget_multiplier": 1.0,
            "effective_budget_tokens": 128, "hard_max_tokens": 256,
            "budget_policy": "context_mode_multiplier_v1", "max_context_tokens": 128,
        },
    }


def _strict_test_hybrid_metrics() -> dict:
    """Complete numeric hybrid gate schema, including zero-tolerance counters."""
    return {
        "functional_final_retrieval_ann_overlap_at_10": 1.0,
        "page_recall_at_5": 1.0,
        "evidence_recall_at_10": 1.0,
        "exact_lookup_hit_at_3": 1.0,
        "mrr_at_10": 1.0,
        "citation_violation_count": 0,
        "context_overflow_count": 0,
        "budget_violation_count": 0,
        "graph_unsupported_count": 0,
    }


def _make_valid_test_hybrid_raw_tree(root: Path, *, dispatch_bundle: dict,
                                     locked_execution: dict, allocation: dict) -> Path:
    """Test-owned complete raw evidence tree; production only validates/exports it."""
    root.mkdir(parents=True)
    workflow_input = dispatch_bundle["workflow_input"]
    request = dispatch_bundle["hybrid_request"]
    query_file = ROOT / "eval" / "queries.jsonl"
    queries = [json.loads(line) for line in query_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(queries) == 105
    query_file_sha256 = hashlib.sha256(query_file.read_bytes()).hexdigest()
    baselines_sha256 = hashlib.sha256((ROOT / "eval" / "baselines.json").read_bytes()).hexdigest()
    fixture_tree_sha256 = canonical_content_tree_sha256(ROOT / "tests" / "fixtures" / "wiki")
    generator_recipe = {
        "version": "public-distractor-v1", "seed": "phase07-public-corpus", "target_size": 30000,
        "source_selection": "sorted-round-robin-markdown",
        "content_suffix": "phase07 hybrid distractor {ordinal}", "query_injection": False,
    }
    generator_recipe["record_self_sha256"] = canonical_digest(generator_recipe)
    corpus_identity = {
        "schema_version": 1, "target_size": 30000, "fixture_tree_sha256": fixture_tree_sha256,
        "generator_recipe_sha256": generator_recipe["record_self_sha256"],
        "corpus_manifest_sha256": CORPUS_MANIFEST_SHA256,
    }
    corpus_sha256 = expected_phase07_expanded_corpus_identity(
        fixture_root=ROOT / "tests" / "fixtures" / "wiki", target_size=30000)["expanded_content_tree_sha256"]
    rows = []
    for ordinal, specification in enumerate(queries):
        payload = _strict_test_hybrid_payload(
            query=specification["query"], ordinal=ordinal, variant=workflow_input["role"],
        )
        if specification.get("relevant_pages"):
            payload["pages"] = list(specification["relevant_pages"])[:10]
        context = " ".join(specification.get("required_facts") or ()) + " " + " ".join(
            f"[来源: {item['path']}]" for item in payload["items"])
        payload["context_text"] = context
        payload["context_sha256"] = hashlib.sha256(context.encode("utf-8")).hexdigest()
        rows.append({
            "ordinal": ordinal,
            "query_sha256": hashlib.sha256(specification["query"].encode("utf-8")).hexdigest(),
            "observation": {"result": payload, "duration_ms": 1.0},
        })
    execution_document = deepcopy(locked_execution)
    allocation_document = {
        "schema_version": 1, "campaign_stage": "hybrid", "allocation": deepcopy(allocation),
    }
    _seal_test_json(root / "locked-execution.json", execution_document)
    _seal_test_json(root / "allocation.json", allocation_document)
    execution_sha256 = hashlib.sha256((root / "locked-execution.json").read_bytes()).hexdigest()
    allocation_sha256 = hashlib.sha256((root / "allocation.json").read_bytes()).hexdigest()
    result = {
        "schema_version": 1, "campaign_stage": "hybrid",
        "bundle_sha256": workflow_input["record_self_sha256"],
        "hybrid_request_sha256": request["record_self_sha256"],
        "role": workflow_input["role"], "config": workflow_input["config"],
        "planned_scale": 30000, "executed_scale": 30000, "query_count": 105,
        **expected_phase07_expanded_corpus_identity(
            fixture_root=ROOT / "tests" / "fixtures" / "wiki", target_size=30000),
        "authorization": "none",
        "queries_sha256": query_file_sha256, "baselines_sha256": baselines_sha256,
        "fixture_tree_sha256": fixture_tree_sha256, "corpus_sha256": corpus_sha256,
        "corpus_manifest_sha256": CORPUS_MANIFEST_SHA256, "generator_recipe": generator_recipe,
        "generator_sha256": generator_recipe["record_self_sha256"],
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "source_digests": {**locked_execution["source_digests"], "queries_sha256": query_file_sha256, "baselines_sha256": baselines_sha256},
        "runtime": locked_execution["runtime"], "head_sha": HEAD,
        "locked_execution": execution_document, "locked_execution_sha256": execution_sha256,
        "allocation": allocation_document, "allocation_sha256": allocation_sha256,
        "original_observations": deepcopy(rows), "expanded_observations": deepcopy(rows),
        "hybrid_invocation": {
            "entrypoint": "query.hybrid_search", "candidate_aware_public_arguments": False,
            "original_calls": 105, "expanded_calls": 105,
        },
        "campaign_progress": {
            "role": workflow_input["role"], "original_completed": 105, "expanded_completed": 105,
        },
    }
    frozen_prepare = dispatch_bundle.get("frozen_prepare")
    if frozen_prepare is not None:
        result.update(
            frozen_prepare=deepcopy(frozen_prepare),
            source_before_sha256=frozen_prepare["base_tree_sha256"],
            source_after_sha256=frozen_prepare["base_tree_sha256"],
        )
    (root / "hybrid-request.json").write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    (root / "dispatch-bundle.json").write_text(json.dumps(dispatch_bundle, sort_keys=True), encoding="utf-8")
    _seal_test_json(root / "hybrid-result.json", result)
    ledger = {
        "schema_version": 1, "campaign_stage": "hybrid", "bundle_sha256": workflow_input["record_self_sha256"],
        "hybrid_request_sha256": request["record_self_sha256"], "role": workflow_input["role"], "config": workflow_input["config"],
        "authorization": "none", "result_sha256": hashlib.sha256((root / "hybrid-result.json").read_bytes()).hexdigest(),
        "locked_execution_sha256": execution_sha256, "allocation_sha256": allocation_sha256,
        "raw_leaf_sha256s": _hybrid_raw_leaf_sha256s(root),
    }
    if frozen_prepare is not None:
        ledger.update(frozen_prepare=deepcopy(frozen_prepare),
                      source_before_sha256=frozen_prepare["base_tree_sha256"],
                      source_after_sha256=frozen_prepare["base_tree_sha256"])
    _seal_test_json(root / "hybrid-ledger.json", ledger)
    packet = {
        "schema_version": 1, "campaign_stage": "hybrid", "packet_kind": "phase07-hybrid-packet/v1",
        "bundle_sha256": workflow_input["record_self_sha256"], "hybrid_request_sha256": request["record_self_sha256"],
        "role": workflow_input["role"], "config": workflow_input["config"], "authorization": "none",
        "result_sha256": hashlib.sha256((root / "hybrid-result.json").read_bytes()).hexdigest(),
        "ledger_sha256": hashlib.sha256((root / "hybrid-ledger.json").read_bytes()).hexdigest(),
        "locked_execution_sha256": execution_sha256, "allocation_sha256": allocation_sha256,
        "raw_file_sha256s": _hybrid_raw_file_sha256s(root), "raw_tree_sha256": _hybrid_raw_tree_sha256(root),
    }
    if frozen_prepare is not None:
        packet.update(frozen_prepare=deepcopy(frozen_prepare),
                      source_before_sha256=frozen_prepare["base_tree_sha256"],
                      source_after_sha256=frozen_prepare["base_tree_sha256"])
    _seal_test_json(root / "hybrid-packet.json", packet)
    return root


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("executed_scale", 1),
        ("query_sha256", "0" * 64),
        ("nested_result_payload", {}),
        ("corpus_sha256", "0" * 64),
        ("generator_sha256", "0" * 64),
        ("model_manifest_sha256", "0" * 64),
        ("runtime", {"python": "3.10"}),
        ("head_sha", "0" * 40),
        ("allocation", {"run_id": 999}),
        ("raw_tree_sha256", "0" * 64),
    ],
)
def test_hybrid_exporter_reconstructs_only_complete_30k_raw_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object,
) -> None:
    """Re-signing a packet cannot replace reconstruction from raw campaign files."""
    from eval import phase07_ann_campaign as campaign

    plan, _ = _build_test_hybrid_plan(tmp_path, monkeypatch)
    tree = _make_valid_test_hybrid_raw_tree(
        root=tmp_path / "artifact", dispatch_bundle=_hybrid_dispatch_bundle(plan),
        locked_execution=_locked_confirmation_environment(),
        allocation={"run_id": 7, "run_attempt": 1, "job_id": 8, "job_key": "phase07-hybrid", "job_allocation_nonce": "a" * 32},
    )
    assert campaign.validate_hybrid_artifact_tree(tree)["config"]["m"] == 20
    result_path = tree / "hybrid-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if field == "nested_result_payload":
        result["original_observations"][0]["observation"]["result"]["items"] = value
    elif field == "query_sha256":
        result["original_observations"][0]["query_sha256"] = value
    elif field == "raw_tree_sha256":
        packet_path = tree / "hybrid-packet.json"
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        packet["raw_tree_sha256"] = value
        _seal_test_json(packet_path, packet)
        with pytest.raises(ValueError):
            campaign.validate_hybrid_artifact_tree(tree)
        return
    else:
        result[field] = value
    result_path.write_text(json.dumps(result), encoding="utf-8")
    # This emulates an attacker who re-seals raw result, ledger, raw-tree, and
    # packet digest chain.  Only semantic identity reconstruction may reject it.
    _reseal_test_hybrid_raw_tree(tree)
    with pytest.raises(ValueError):
        campaign.validate_hybrid_artifact_tree(tree)


def test_dense_expiry_and_structured_retired_authority_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expiry is evidence identity; plain text such as platform is not an authority mode."""
    dense = _write_sealed_dense_ledger(tmp_path)
    ledger = json.loads(dense.read_text(encoding="utf-8"))
    ledger["eligible_evidence_runs"][0]["validated_provenance"]["artifact_expires_at"] = "2000-01-01T00:00:00Z"
    ledger["record_self_sha256"] = operator.canonical_digest(ledger)
    dense.write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(operator, "DENSE_LEDGER_DIGEST", ledger["record_self_sha256"])
    with pytest.raises(ValueError, match="expiry"):
        operator.build_hybrid_plan(dense, post_implementation_head=HEAD)
    assert operator.reject_retired_hybrid_authority({"metadata": {"platform": "github"}}) is None
    with pytest.raises(ValueError):
        operator.reject_retired_hybrid_authority({"mode": "flat"})


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


def test_expanded_corpus_identity_normalizes_source_line_endings(tmp_path: Path) -> None:
    """Expected and materialized 30k-style identities are platform independent."""
    import eval.run_eval as run_eval

    lf_root, crlf_root = tmp_path / "lf-source", tmp_path / "crlf-source"
    lf_root.mkdir(); crlf_root.mkdir()
    (lf_root / "source.md").write_bytes(b"# Source\nbody\n")
    (crlf_root / "source.md").write_bytes(b"# Source\r\nbody\r\n")

    expected_lf = run_eval.expected_phase07_expanded_corpus_identity(
        fixture_root=lf_root, target_size=3, test_only=True,
    )
    expected_crlf = run_eval.expected_phase07_expanded_corpus_identity(
        fixture_root=crlf_root, target_size=3, test_only=True,
    )
    assert expected_lf == expected_crlf

    materialized = run_eval._materialize_phase07_expanded_corpus(
        fixture_root=crlf_root, output_root=tmp_path / "expanded", target_size=3, test_only=True,
    )
    assert materialized == expected_crlf
