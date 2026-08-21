from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from eval.phase07_ann_campaign import CampaignConfig, Phase07AnnCampaignRunner, execute, validate_request
from eval.run_eval import run_phase07_representative_campaign


def _digest(letter: str) -> str:
    return letter * 64


def _request(stage: str = "screening", *, mode: str = "stage2_sq") -> dict:
    request = {
        "schema_version": 1, "stage": stage, "request_id": "request-1", "environment": {},
        "model_manifest_sha256": _digest("a"), "corpus_manifest_sha256": _digest("b"),
    }
    if stage == "confirmation":
        request.update(prior_screening_sha256=_digest("c"), nominated_m=[16], run_ordinal=1,
                       run_identity={"run_id": "1", "run_attempt": 1, "job_id": "2", "job_allocation_nonce": "nonce-00000000001"})
    if stage == "continuation":
        configs = {
            "stage2_sq": {"approved_d04_sha256": _digest("d"), "m": 16},
            "flat_diagnostic": {"no_confirmed_sq_sha256": _digest("d"), "m": 16, "query_ef": 300},
            "refinement": {"ceiling_sha256": _digest("d"), "m": 16, "ef_construction": 300, "query_ef": 300},
            "representative_ann": {"size": 1000, "baseline_sha256": _digest("d"), "finalist_sha256": _digest("e")},
            "hybrid_non_regression": {"size": 1000, "baseline_sha256": _digest("d"), "finalist_sha256": _digest("e")},
        }
        request.update(mode=mode, prior_evidence_sha256=_digest("c"), config=configs[mode])
    return request


def _tiny_runner(tmp_path: Path) -> Phase07AnnCampaignRunner:
    # Explicit trusted Python seam: request JSON has no rows/probes/dimensions fields.
    return Phase07AnnCampaignRunner(CampaignConfig(rows=64, dimensions=8, probes=4, per_build_max_seconds=30, work_dir=tmp_path))


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


def test_tiny_screening_uses_real_lancedb_reopen_and_ann_grid(tmp_path: Path) -> None:
    record = execute(_request(), tmp_path / "campaign", runner=_tiny_runner(tmp_path).run)
    result = record["result"]
    assert result["exact_truth_computed_once"] is True and result["build_count"] == 3
    assert [build["build"]["m"] for build in result["builds"]] == [16, 20, 32]
    for build in result["builds"]:
        assert build["build"]["normal_ann_request_count"] == 16
        assert [group["query_ef"] for group in build["queries"]] == [100, 150, 200, 300]
        assert build["build"]["dense_table_open_count"] >= 1


def test_confirmation_and_continuations_are_bounded_and_prerequisite_gated(tmp_path: Path) -> None:
    runner = _tiny_runner(tmp_path)
    confirmation = execute(_request("confirmation"), tmp_path / "confirm", runner=runner.run)["result"]
    assert confirmation["build_count"] == 1
    assert [group["query_ef"] for group in confirmation["replicate"]["queries"]] == [200, 300]
    stage2 = execute(_request("continuation", mode="stage2_sq"), tmp_path / "stage2", runner=runner.run)["result"]
    assert [group["query_ef"] for group in stage2["stage2"]["queries"]] == [300, 500]
    refined = execute(_request("continuation", mode="refinement"), tmp_path / "refine", runner=runner.run)["result"]
    assert refined["build_count"] == 1 and [row["refine_factor"] for row in refined["refinement"]] == [2, 5, 10]
    missing = _request("continuation", mode="flat_diagnostic"); missing["config"].pop("no_confirmed_sq_sha256")
    with pytest.raises(ValueError, match="FLAT requires"):
        validate_request(missing)


def test_public_hybrid_facade_uses_real_build_service_and_lancedb(tmp_path: Path) -> None:
    # This is deliberately not a mock: injected encoding is the only test seam.
    result = run_phase07_representative_campaign(mode="representative_ann", size=1000,
                                                  work_dir=tmp_path, authorization="none", embed=_embed384())
    assert result["authorization"] == "none"
    assert result["hybrid_invocation"]["entrypoint"] == "query.hybrid_search"
    assert isinstance(result["result"], dict)
