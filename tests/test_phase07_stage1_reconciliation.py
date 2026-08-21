"""End-to-end fail-closed contracts for the Phase 07 Stage 1 handoff."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
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
    select_stage1_nominees,
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
            "labels": ["ubuntu-latest"],
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
        "run_created_at": "2026-08-21T00:00:00Z",
        "api_provenance": {
            "workflow_run": {
                "run_id": 991,
                "run_attempt": 1,
                "head_branch": "feature/issue-50-dense-ann-recall",
                "head_sha": "a" * 40,
                "event": "workflow_dispatch",
                "status": "completed",
                "conclusion": "success",
                "created_at": "2026-08-21T00:00:00Z",
            },
            "job": {
                "job_id": 881,
                "run_id": 991,
                "name": "Phase 07 bounded SQ screening campaign",
                "status": "completed",
                "conclusion": "success",
                "runner_name": "GitHub Actions 42",
                "runner_group_name": "GitHub Actions",
                "labels": ["ubuntu-latest"],
            },
            "artifact": {
                "artifact_id": 771,
                "job_id": 881,
                "job_key": "phase07-screening",
                "run_id": 991,
                "name": "phase07-screening-991-1",
                "created_at": "2026-08-21T00:00:00Z",
                "expires_at": "2026-11-19T00:00:00Z",
                "expired": False,
            },
        },
        "workflow_retention_days": 90,
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


def test_stage1_reconciler_direct_script_cli_bootstraps_package_imports() -> None:
    """The production ``python eval/reconcile_ann_gate.py`` entry point imports cleanly."""
    completed = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "reconcile_ann_gate.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--stage1-request" in completed.stdout


def test_stage1_reconciler_module_cli_uses_package_qualified_imports() -> None:
    """``python -m`` starts from the repository package, without test path injection."""
    completed = subprocess.run(
        [sys.executable, "-m", "eval.reconcile_ann_gate", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--stage1-request" in completed.stdout


@pytest.mark.parametrize(
    "mutate",
    ["missing-anchor", "too-short", "too-long", "api-anchor-mismatch", "api-artifact-mismatch", "workflow-retention"],
)
def test_stage1_retention_uses_trusted_workflow_run_creation_anchor(
    tmp_path: Path, tiny_stage1_artifact: tuple[Path, dict[str, object]], mutate: str,
) -> None:
    """GitHub expiry is 90 days from run creation, never artifact upload completion."""
    source, result = tiny_stage1_artifact
    artifact_dir = tmp_path / "artifact"
    shutil.copytree(source, artifact_dir)
    request = _post_download_request(artifact_dir, result)
    artifact = request["artifact"]
    api = request["api_provenance"]
    assert isinstance(artifact, dict) and isinstance(api, dict)
    if mutate == "missing-anchor":
        request.pop("run_created_at")
    elif mutate == "too-short":
        artifact["expires_at"] = "2026-11-18T23:59:29Z"
        api["artifact"]["expires_at"] = artifact["expires_at"]
    elif mutate == "too-long":
        artifact["expires_at"] = "2026-11-19T00:00:31Z"
        api["artifact"]["expires_at"] = artifact["expires_at"]
    elif mutate == "api-anchor-mismatch":
        api["workflow_run"]["created_at"] = "2026-08-21T00:05:00Z"
    elif mutate == "api-artifact-mismatch":
        api["artifact"]["job_id"] = 999
    else:
        request["workflow_retention_days"] = 89
    _seal(request)
    request_path = tmp_path / "stage1-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError):
        reconcile_stage1_screening(
            stage1_request=request_path, artifact_dir=artifact_dir,
            output=tmp_path / "stage1-ledger.json", mode="screening", expected_shape=(48, 384, 8),
        )


@pytest.mark.parametrize(
    "mutate",
    ["os", "architecture", "labels", "group", "name", "api-name", "api-group", "api-labels"],
)
def test_stage1_hosted_runner_identity_and_api_job_provenance_are_exact(
    tmp_path: Path, tiny_stage1_artifact: tuple[Path, dict[str, object]], mutate: str,
) -> None:
    """Stage 1 accepts only the API-bound ubuntu-latest hosted Linux/X64 allocation."""
    source, result = tiny_stage1_artifact
    artifact_dir = tmp_path / "artifact"
    shutil.copytree(source, artifact_dir)
    request = _post_download_request(artifact_dir, result)
    runner = request["runner"]
    api = request["api_provenance"]
    assert isinstance(runner, dict) and isinstance(api, dict)
    job = api["job"]
    assert isinstance(job, dict)
    if mutate == "os":
        runner["os"] = "Windows"
    elif mutate == "architecture":
        runner["architecture"] = "ARM64"
    elif mutate == "labels":
        runner["labels"] = ["self-hosted"]
    elif mutate == "group":
        runner["group"] = "Default"
    elif mutate == "name":
        runner["name"] = "self-hosted"
    elif mutate == "api-name":
        job["runner_name"] = "GitHub Actions 99"
    elif mutate == "api-group":
        job["runner_group_name"] = "Default"
    else:
        job["labels"] = ["self-hosted"]
    _seal(request)
    request_path = tmp_path / "stage1-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError):
        reconcile_stage1_screening(
            stage1_request=request_path, artifact_dir=artifact_dir,
            output=tmp_path / "stage1-ledger.json", mode="screening", expected_shape=(48, 384, 8),
        )


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


def test_stage1_ledger_distinguishes_artifact_and_reconciled_nominees(
    tmp_path: Path, tiny_stage1_artifact: tuple[Path, dict[str, object]],
) -> None:
    """A corrected ranking must not rewrite the immutable artifact's reported list."""
    source, result = tiny_stage1_artifact
    artifact_dir = tmp_path / "artifact"
    shutil.copytree(source, artifact_dir)
    extracted = artifact_dir / "extracted"
    result_record = json.loads((extracted / "screening-result.json").read_text(encoding="utf-8"))
    authoritative = select_stage1_nominees(
        result_record["result"]["builds"], result_record["result"]["d04_statistics"],
    )
    reported = [m for m in (16, 20, 32) if m not in authoritative][:1]
    assert reported and reported != authoritative
    result_record["result"]["nominated_m"] = reported
    _seal(result_record)
    (extracted / "screening-result.json").write_text(json.dumps(result_record), encoding="utf-8")
    request = _reseal_bundle(artifact_dir, result)
    request_path = tmp_path / "stage1-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    ledger = reconcile_stage1_screening(
        stage1_request=request_path, artifact_dir=artifact_dir,
        output=tmp_path / "stage1-ledger.json", mode="screening", expected_shape=(48, 384, 8),
    )

    assert ledger["artifact_reported_nominated_m"] == reported
    assert ledger["nominated_m"] == authoritative
    assert ledger["artifact_reported_nominated_m"] != ledger["nominated_m"]
    for field in ("artifact_reported_nominated_m", "nominated_m"):
        assert len(ledger[field]) <= 2 and len(set(ledger[field])) == len(ledger[field])
        assert set(ledger[field]) <= {16, 20, 32}


def test_stage1_reconciler_accepts_runner_bound_corpus_digest_not_locally_regenerated(
    tmp_path: Path, tiny_stage1_artifact: tuple[Path, dict[str, object]],
) -> None:
    """Hosted numerical identities remain bound to the sealed runner artifact, not macOS BLAS."""
    source, result = tiny_stage1_artifact
    artifact_dir = tmp_path / "artifact"
    shutil.copytree(source, artifact_dir)
    request = _post_download_request(artifact_dir, result)
    extracted = artifact_dir / "extracted"
    payload = json.loads((extracted / "screening-result.json").read_text(encoding="utf-8"))
    # A different, sealed hosted floating-point reduction is valid only when all
    # of the outer API/archive/content and inner exact-query evidence agree.
    payload["result"]["stress_identity"]["corpus_sha256"] = "d" * 64
    _seal(payload)
    (extracted / "screening-result.json").write_text(json.dumps(payload), encoding="utf-8")
    request = _reseal_bundle(artifact_dir, result)
    request_path = tmp_path / "stage1-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    ledger = reconcile_stage1_screening(
        stage1_request=request_path, artifact_dir=artifact_dir,
        output=tmp_path / "stage1-ledger.json", mode="screening", expected_shape=(48, 384, 8),
    )

    assert ledger["stress_identity"]["corpus_sha256"] == "d" * 64
    assert len({build["build_id"] for build in ledger["builds"]}) == 3


def test_stage1_reconciler_rejects_unsealed_outer_corpus_digest_substitution(
    tmp_path: Path, tiny_stage1_artifact: tuple[Path, dict[str, object]],
) -> None:
    """Runner-bound portability never permits changing a downloaded artifact in place."""
    source, result = tiny_stage1_artifact
    artifact_dir = tmp_path / "artifact"
    shutil.copytree(source, artifact_dir)
    request = _post_download_request(artifact_dir, result)
    extracted = artifact_dir / "extracted"
    payload = json.loads((extracted / "screening-result.json").read_text(encoding="utf-8"))
    payload["result"]["stress_identity"]["corpus_sha256"] = "d" * 64
    _seal(payload)
    (extracted / "screening-result.json").write_text(json.dumps(payload), encoding="utf-8")
    request_path = tmp_path / "stage1-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError):
        reconcile_stage1_screening(
            stage1_request=request_path, artifact_dir=artifact_dir,
            output=tmp_path / "stage1-ledger.json", mode="screening", expected_shape=(48, 384, 8),
        )


def test_stage1_reconciler_rejects_resealed_ledger_with_substituted_request_binding(
    tmp_path: Path, tiny_stage1_artifact: tuple[Path, dict[str, object]],
) -> None:
    """The inner ledger must bind exactly to the sealed screening request record."""
    source, result = tiny_stage1_artifact
    artifact_dir = tmp_path / "artifact"
    shutil.copytree(source, artifact_dir)
    extracted = artifact_dir / "extracted"
    ledger = json.loads((extracted / "screening-ledger.json").read_text(encoding="utf-8"))
    ledger["request_sha256"] = "f" * 64
    _seal(ledger)
    (extracted / "screening-ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    request = _reseal_bundle(artifact_dir, result)
    request_path = tmp_path / "stage1-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError):
        reconcile_stage1_screening(
            stage1_request=request_path, artifact_dir=artifact_dir,
            output=tmp_path / "stage1-ledger.json", mode="screening", expected_shape=(48, 384, 8),
        )


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
     "job", "artifact", "rejection", "recall", "statistics", "self-digest", "cross-build-truth",
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
    elif tamper in {"recall", "statistics", "self-digest", "cross-build-truth", "corpus-seed", "algorithm"}:
        payload = json.loads((extracted / "screening-result.json").read_text(encoding="utf-8"))
        if tamper == "recall":
            payload["result"]["builds"][0]["queries"][0]["queries"][0]["recall_at_10"] = 0.123
        elif tamper == "statistics":
            payload["result"]["d04_statistics"]["comparisons"][0]["holm_adjusted_p"] = 0.0
        elif tamper == "cross-build-truth":
            sample = payload["result"]["builds"][1]["queries"][0]["queries"][0]
            sample["exact_top_10"][0] = "substituted-exact-id"
            sample["exact_top_20"][0] = "substituted-exact-id"
            sample["recall_at_10"] = len(set(sample["exact_top_10"]) & set(sample["candidate_top_10"][:10])) / 10
            sample["recall_at_20"] = len(set(sample["exact_top_20"]) & set(sample["candidate_top_20"][:20])) / 20
            group = payload["result"]["builds"][1]["queries"][0]
            group["recall_at_10"] = sum(item["recall_at_10"] for item in group["queries"]) / len(group["queries"])
            group["recall_at_20"] = sum(item["recall_at_20"] for item in group["queries"]) / len(group["queries"])
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
