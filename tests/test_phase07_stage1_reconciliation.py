"""End-to-end fail-closed contracts for the Phase 07 Stage 1 handoff."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

from eval.phase07_ann_campaign import (
    CampaignConfig,
    Phase07AnnCampaignRunner,
    execute,
)
from eval.reconcile_ann_gate import reconcile_stage1_screening


def _hosted_identity() -> dict[str, object]:
    return {
        "repository": "allenwoo713/obsidian_wiki_skill",
        "head_sha": "a" * 40,
        "run_id": 991,
        "run_attempt": 1,
        "job_id": 881,
        "job_key": "phase07-screening",
        "job_allocation_nonce": "991-1-phase07-screening",
        "runtime": {
            "python": "3.13",
            "lancedb": "0.34.0",
            "numpy": "2.2.6",
            "pyarrow": "25.0.0",
            "omp_num_threads": 2,
        },
        "model_manifest_sha256": "b" * 64,
        "corpus_manifest_sha256": "c" * 64,
        "workflow_name": "eval.yml",
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "runner": {
            "name": "GitHub Actions 42",
            "group": "GitHub Actions",
            "labels": ["ubuntu-latest", "X64"],
            "os": "Linux",
            "image": "ubuntu-24.04",
            "architecture": "X64",
        },
        "lock_identity": "e" * 64,
        "retry_lineage": {
            "failure_class": None,
            "original_run_id": None,
            "replacement_run_id": None,
        },
    }


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _seal(payload: dict[str, object]) -> None:
    payload["record_self_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in payload.items() if key != "record_self_sha256"},
                   sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_raw_archive(bundle: Path) -> None:
    extracted = bundle / "extracted"
    with zipfile.ZipFile(bundle / "archive.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(extracted.iterdir()):
            info = zipfile.ZipInfo(path.name, date_time=(2026, 8, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())


def _campaign_request() -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "screening",
        "request_id": "tiny-stage1",
        "environment": {
            "branch": "feature/issue-50-dense-ann-recall",
            "workflow_path": ".github/workflows/eval.yml",
            "head_sha": "a" * 40,
            "run_id": 991,
            "run_attempt": 1,
            "job_key": "phase07-screening",
            "job_allocation_nonce": "991-1-phase07-screening",
            "runtime": {
                "python": "3.13", "lancedb": "0.34.0", "numpy": "2.2.6",
                "pyarrow": "25.0.0", "omp_num_threads": 2,
            },
        },
        "model_manifest_sha256": "b" * 64,
        # This tracks a repository input only; it is never Stage 1 stress truth.
        "corpus_manifest_sha256": "c" * 64,
        "lock_identity": "e" * 64,
    }


@pytest.fixture(scope="session")
def tiny_stage1_artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    root = tmp_path_factory.mktemp("stage1-real-lancedb")
    artifact_dir = root / "artifact" / "extracted"
    runner = Phase07AnnCampaignRunner(
        CampaignConfig(rows=48, dimensions=384, probes=8, work_dir=root / "builds"),
    )
    result = execute(_campaign_request(), artifact_dir, runner=runner.run)
    bundle = artifact_dir.parent
    # The request below is constructed directly in the test, not by campaign code.
    _write_raw_archive(bundle)
    return bundle, result


def _post_download_request(artifact_dir: Path, result: dict[str, object]) -> dict[str, object]:
    identity = _hosted_identity()
    extracted = artifact_dir / "extracted"
    raw_archive = artifact_dir / "archive.zip"
    archive_sha256 = hashlib.sha256(raw_archive.read_bytes()).hexdigest()
    request = {
        "schema_version": 1,
        "mode": "screening",
        "authorization": "none",
        **identity,
        "branch": "feature/issue-50-dense-ann-recall",
        "workflow_path": ".github/workflows/eval.yml",
        "artifact": {
            "artifact_id": 771,
            "name": "phase07-screening-991-1",
            "job_id": 881,
            "job_key": "phase07-screening",
            "retention_days_requested": 90,
            "retention_days_accepted": 90,
            "created_at": "2026-08-21T00:00:00Z",
            "expires_at": "2026-11-19T00:00:00Z",
            "expired": False,
            "api_archive_sha256": archive_sha256,
            "local_archive_path": "archive.zip",
            "local_archive_sha256": archive_sha256,
            "content_tree_sha256": _tree_digest(extracted),
        },
        "campaign_result_sha256": hashlib.sha256(
            (extracted / "screening-result.json").read_bytes()
        ).hexdigest(),
        "campaign_request_sha256": hashlib.sha256(
            (extracted / "screening-request.json").read_bytes()
        ).hexdigest(),
        "campaign_ledger_sha256": hashlib.sha256(
            (extracted / "screening-ledger.json").read_bytes()
        ).hexdigest(),
        "record_self_sha256": "",
    }
    _seal(request)
    return request


def _reseal_bundle(bundle: Path, result: dict[str, object]) -> dict[str, object]:
    extracted = bundle / "extracted"
    for name in ("screening-request.json", "screening-ledger.json", "screening-result.json"):
        path = extracted / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        _seal(payload)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    _write_raw_archive(bundle)
    return _post_download_request(bundle, result)


def test_stage1_tiny_three_build_artifact_reconciles_without_rep_manifest_truth(
    tmp_path: Path, tiny_stage1_artifact: tuple[Path, dict[str, object]],
) -> None:
    """Three fresh SQ builds carry stress-derived truth, not representative placeholders."""
    source, result = tiny_stage1_artifact
    artifact_dir = tmp_path / "artifact"
    shutil.copytree(source, artifact_dir)
    request = _post_download_request(artifact_dir, result)
    request_path = tmp_path / "stage1-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output = tmp_path / "stage1-ledger.json"

    ledger = reconcile_stage1_screening(
        stage1_request=request_path, artifact_dir=artifact_dir, output=output, mode="screening",
        expected_shape=(48, 384, 8),
    )

    assert ledger["status"] == "success"
    assert ledger["authorization"] == "none"
    assert ledger["stress_identity"]["corpus_sha256"] != request["corpus_manifest_sha256"]
    assert {build["build"]["m"] for build in ledger["builds"]} == {16, 20, 32}
    assert len({build["build_id"] for build in ledger["builds"]}) == 3
    assert all([group["query_ef"] for group in build["queries"]] == [100, 150, 200, 300] for build in ledger["builds"])
    assert all(sum(len(group["queries"]) for group in build["queries"]) == 4 * 8 for build in ledger["builds"])
    assert output.exists()


def test_stage1_reconciler_allows_large_index_bytes_when_build_stays_bounded(
    tmp_path: Path, tiny_stage1_artifact: tuple[Path, dict[str, object]],
) -> None:
    source, result = tiny_stage1_artifact
    artifact_dir = tmp_path / "artifact"
    shutil.copytree(source, artifact_dir)
    extracted = artifact_dir / "extracted"
    payload = json.loads((extracted / "screening-result.json").read_text(encoding="utf-8"))
    for build in payload["result"]["builds"]:
        build["build"]["index_bytes"] = 180_001
    (extracted / "screening-result.json").write_text(json.dumps(payload), encoding="utf-8")
    request = _reseal_bundle(artifact_dir, result)
    request_path = tmp_path / "stage1-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    ledger = reconcile_stage1_screening(
        stage1_request=request_path, artifact_dir=artifact_dir, output=tmp_path / "ledger.json",
        mode="screening", expected_shape=(48, 384, 8),
    )

    assert all(build["build"]["index_bytes"] > 180_000 for build in ledger["builds"])


@pytest.mark.parametrize(
    "tamper",
    ["one-build", "symlink", "secret", "stale-head", "extra-file", "archive", "content-tree",
     "job", "artifact", "rejection", "recall", "statistics", "self-digest", "corpus-digest",
     "corpus-seed", "algorithm"],
)
def test_stage1_reconciler_rejects_tampered_or_incomplete_evidence(
    tmp_path: Path, tamper: str, tiny_stage1_artifact: tuple[Path, dict[str, object]],
) -> None:
    source, result = tiny_stage1_artifact
    artifact_dir = tmp_path / "artifact"
    shutil.copytree(source, artifact_dir)
    request = _post_download_request(artifact_dir, result)
    extracted = artifact_dir / "extracted"
    if tamper == "one-build":
        payload = json.loads((extracted / "screening-result.json").read_text(encoding="utf-8"))
        payload["result"]["builds"] = payload["result"]["builds"][:1]
        (extracted / "screening-result.json").write_text(json.dumps(payload), encoding="utf-8")
        request = _reseal_bundle(artifact_dir, result)
    elif tamper == "symlink":
        (extracted / "substitute.json").symlink_to(extracted / "screening-result.json")
    elif tamper == "secret":
        request["token"] = "must-reject"
    elif tamper == "stale-head":
        request["head_sha"] = "d" * 40
        _seal(request)
    elif tamper == "archive":
        request["artifact"]["local_archive_sha256"] = "f" * 64
        _seal(request)
    elif tamper == "content-tree":
        request["artifact"]["content_tree_sha256"] = "f" * 64
        _seal(request)
    elif tamper == "job":
        request["job_id"] = 999
        _seal(request)
    elif tamper == "artifact":
        request["artifact"]["retention_days_accepted"] = 7
        _seal(request)
    elif tamper == "rejection":
        (extracted / "screening-rejection.json").write_text('{"status":"reject-evidence"}', encoding="utf-8")
    elif tamper in {"recall", "statistics", "self-digest", "corpus-digest", "corpus-seed", "algorithm"}:
        payload = json.loads((extracted / "screening-result.json").read_text(encoding="utf-8"))
        if tamper == "recall":
            payload["result"]["builds"][0]["queries"][0]["queries"][0]["recall_at_10"] = 0.123
        elif tamper == "statistics":
            payload["result"]["d04_statistics"]["comparisons"][0]["holm_adjusted_p"] = 0.0
        elif tamper == "corpus-digest":
            payload["result"]["stress_identity"]["corpus_sha256"] = "f" * 64
        elif tamper == "corpus-seed":
            payload["result"]["stress_identity"]["corpus_seed"] = 0
        elif tamper == "algorithm":
            payload["result"]["stress_identity"]["algorithm"]["vectors"] = "untrusted"
        else:
            payload["record_self_sha256"] = "0" * 64
        (extracted / "screening-result.json").write_text(json.dumps(payload), encoding="utf-8")
        if tamper != "self-digest":
            request = _reseal_bundle(artifact_dir, result)
    else:
        (artifact_dir / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    request_path = tmp_path / "stage1-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError):
        reconcile_stage1_screening(
            stage1_request=request_path, artifact_dir=artifact_dir,
            output=tmp_path / "stage1-ledger.json", mode="screening", expected_shape=(48, 384, 8),
        )
