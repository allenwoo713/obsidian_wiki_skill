"""End-to-end fail-closed contracts for the Phase 07 Stage 1 handoff."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.phase07_ann_campaign import (
    CampaignConfig,
    create_stage1_screening_artifact,
    create_stage1_screening_request,
)
from eval.reconcile_ann_gate import reconcile_stage1_screening


def _hosted_identity() -> dict[str, object]:
    return {
        "repository": "allenwoo713/obsidian_wiki_skill",
        "head_sha": "a" * 40,
        "run_id": 991,
        "run_attempt": 1,
        "job_id": 881,
        "job_key": "phase07-stage1-screening",
        "job_allocation_nonce": "991-1-phase07-stage1-screening",
        "runtime": {
            "python": "3.13",
            "lancedb": "0.34.0",
            "numpy": "2.2.6",
            "pyarrow": "25.0.0",
            "omp_num_threads": 2,
        },
        "model_manifest_sha256": "b" * 64,
        "corpus_manifest_sha256": "c" * 64,
        "authorization": "none",
    }


def test_stage1_tiny_three_build_artifact_reconciles_without_rep_manifest_truth(
    tmp_path: Path,
) -> None:
    """Three fresh SQ builds carry stress-derived truth, not representative placeholders."""
    artifact_dir = tmp_path / "artifact"
    result = create_stage1_screening_artifact(
        artifact_dir,
        config=CampaignConfig(rows=48, dimensions=384, probes=8, work_dir=tmp_path / "builds"),
    )
    request = create_stage1_screening_request(
        hosted_identity=_hosted_identity(), artifact_dir=artifact_dir, result=result,
    )
    request_path = tmp_path / "stage1-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output = tmp_path / "stage1-ledger.json"

    ledger = reconcile_stage1_screening(
        stage1_request=request_path, artifact_dir=artifact_dir, output=output, mode="screening",
    )

    assert ledger["status"] == "success"
    assert ledger["authorization"] == "none"
    assert ledger["stress_identity"]["corpus_sha256"] != request["corpus_manifest_sha256"]
    assert {build["m"] for build in ledger["builds"]} == {16, 20, 32}
    assert len({build["build_id"] for build in ledger["builds"]}) == 3
    assert all(len(build["query_samples"]) == 4 * 8 for build in ledger["builds"])
    assert output.exists()


@pytest.mark.parametrize("tamper", ["one-build", "symlink", "secret", "stale-head", "extra-file"])
def test_stage1_reconciler_rejects_tampered_or_incomplete_evidence(
    tmp_path: Path, tamper: str,
) -> None:
    artifact_dir = tmp_path / "artifact"
    result = create_stage1_screening_artifact(
        artifact_dir,
        config=CampaignConfig(rows=48, dimensions=384, probes=8, work_dir=tmp_path / "builds"),
    )
    request = create_stage1_screening_request(
        hosted_identity=_hosted_identity(), artifact_dir=artifact_dir, result=result,
    )
    if tamper == "one-build":
        payload = json.loads((artifact_dir / "stage1-screening-result.json").read_text(encoding="utf-8"))
        payload["builds"] = payload["builds"][:1]
        (artifact_dir / "stage1-screening-result.json").write_text(json.dumps(payload), encoding="utf-8")
    elif tamper == "symlink":
        (artifact_dir / "substitute.json").symlink_to(artifact_dir / "stage1-screening-result.json")
    elif tamper == "secret":
        request["token"] = "must-reject"
    elif tamper == "stale-head":
        request["head_sha"] = "d" * 40
    else:
        (artifact_dir / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    request_path = tmp_path / "stage1-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError):
        reconcile_stage1_screening(
            stage1_request=request_path, artifact_dir=artifact_dir,
            output=tmp_path / "stage1-ledger.json", mode="screening",
        )
