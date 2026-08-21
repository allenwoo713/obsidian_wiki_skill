from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from eval.phase07_ann_campaign import (
    CampaignConfig,
    Phase07AnnCampaignRunner,
    canonical_digest,
    execute,
    select_stage1_nominees,
    validate_request,
)
from eval.phase07_operator_gate import canonical_digest as operator_digest
from eval.run_eval import _representative_indexed_query_separation, run_phase07_representative_campaign
from eval.ann_corpus_manifest import canonical_content_tree_sha256
from eval.ann_corpus_manifest import PHASE07_CURRENT_BASELINE, phase07_current_baseline_sha256, validate_indexed_query_digest_separation


def _digest(letter: str) -> str:
    return letter * 64


def _request(stage: str = "screening", *, mode: str = "stage2_sq") -> dict:
    request = {
        "schema_version": 1, "stage": stage, "request_id": "request-1", "environment": {},
        "model_manifest_sha256": _digest("a"), "corpus_manifest_sha256": _digest("b"),
    }
    if stage == "screening":
        request["environment"] = {
            "branch": "feature/issue-50-dense-ann-recall",
            "workflow_path": ".github/workflows/eval.yml",
            "head_sha": "c" * 40,
            "run_id": 1,
            "run_attempt": 1,
            "job_key": "phase07-screening",
            "job_allocation_nonce": "1-1-phase07-screening",
            "runtime": {"python": "3.13", "lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0", "omp_num_threads": 2},
        }
        request["lock_identity"] = _digest("d")
    if stage == "confirmation":
        request["environment"] = {"head_sha": "c" * 40}
        slot = {"m": 32, "ordinal": 1}
        binding = {"kind": "phase07-confirmation/v1", "prior_screening_sha256": "6c424135ec4db8983136826575d9436b6ba88da5029384ada47fadb2d1918e33", "nominated_m": 32, "run_ordinal": 1}
        workflow_inputs = {"schema_version": 1, "campaign_stage": "confirmation", "confirmation_request_sha256": _digest("c"), "slot": slot, "post_task0_head": "c" * 40, "continuation_binding": binding, "continuation_binding_sha256": operator_digest(binding), "dispatch_identity": "phase07-confirmation/32/1"}
        workflow_inputs["record_self_sha256"] = operator_digest(workflow_inputs)
        request.update(workflow_inputs=workflow_inputs,
                       run_identity={"run_id": "1", "run_attempt": 1, "job_id": "2", "job_allocation_nonce": "nonce-00000000001"})
    if stage == "continuation":
        configs = {
            "stage2_sq": {"approved_d04_sha256": _digest("d"), "m": 16},
            "flat_diagnostic": {"no_confirmed_sq_sha256": _digest("d"), "m": 16, "query_ef": 300},
            "refinement": {"ceiling_sha256": _digest("d"), "m": 16, "ef_construction": 300, "query_ef": 300},
            "representative_ann": _representative_config(),
            "hybrid_non_regression": _representative_config(),
        }
        request.update(mode=mode, prior_evidence_sha256=_digest("c"), config=configs[mode])
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


def test_direct_campaign_script_cli_reaches_request_validation_without_running_stress_matrix(
    tmp_path: Path,
) -> None:
    """The hosted workflow invokes the script path, not ``python -m``."""
    request = tmp_path / "invalid-request.json"
    request.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "eval/phase07_ann_campaign.py",
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


def test_representative_request_rejects_noncurrent_baseline_before_artifacts(tmp_path: Path) -> None:
    for key, value in (("candidate", "ivf-hnsw-flat"), ("m", 20), ("ef_construction", 500), ("query_ef", 300), ("refine_factor", 2)):
        request = _request("continuation", mode="representative_ann")
        request["config"]["baseline"][key] = value
        request["config"]["baseline_sha256"] = canonical_digest(request["config"]["baseline"])
        with pytest.raises(ValueError):
            execute(request, tmp_path / key)
        assert not (tmp_path / key).exists()
    stale = _request("continuation", mode="representative_ann")
    stale["config"]["baseline_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_request(stale)
    with pytest.raises(ValueError):
        validate_indexed_query_digest_separation(indexed_row_digests=["z" * 64], query_row_digests=["a" * 64])


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


def test_confirmation_and_continuations_are_bounded_and_prerequisite_gated(tmp_path: Path) -> None:
    runner = _tiny_runner(tmp_path)
    confirmation = execute(_request("confirmation"), tmp_path / "confirm", runner=runner.run)["result"]
    assert confirmation["build_count"] == 3
    assert {build["build"]["m"] for build in confirmation["builds"]} == {16, 20, 32}
    assert [group["query_ef"] for group in confirmation["builds"][0]["queries"]] == [100, 200, 300]
    assert confirmation["d04_statistics"]["family_size"] == 6
    assert confirmation["d20_member_statistics"]["family_size"] == 2
    stage2 = execute(_request("continuation", mode="stage2_sq"), tmp_path / "stage2", runner=runner.run)["result"]
    assert [group["query_ef"] for group in stage2["stage2"]["queries"]] == [300, 500]
    refined = execute(_request("continuation", mode="refinement"), tmp_path / "refine", runner=runner.run)["result"]
    assert refined["build_count"] == 1 and [row["refine_factor"] for row in refined["refinement"]] == [2, 5, 10]
    missing = _request("continuation", mode="flat_diagnostic"); missing["config"].pop("no_confirmed_sq_sha256")
    with pytest.raises(ValueError, match="FLAT requires"):
        validate_request(missing)


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
