from __future__ import annotations

import json
import subprocess
import sys
import hashlib
import zipfile
from pathlib import Path

import pytest

from eval import phase07_operator_gate as operator
from eval import reconcile_ann_gate as reconcile
from eval.ann_frontier_statistics import holm_adjust, paired_basic_effect, paired_permutation_p


LEDGER = Path("/Users/ww/Workspace/General/obsidian_wiki_skill/.planning/phases/07-issue-50-improve-dense-ann-recall/operator/07-04-repair2-stage1-ledger.json")
HEAD = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _plan() -> dict:
    return operator.build_confirmation_plan(LEDGER, post_task0_head=HEAD)


def _packet(slot: dict, *, run_id: int, failure_class: str | None = None, replacement_for: int | None = None) -> dict:
    builds = [
        {"build_id": f"{run_id:02x}{m:02x}".ljust(64, "a"), "m": m, "ef_construction": 300,
         "query_ef": [100, 200, 300] if m == 16 else [200, 300]}
        for m in (16, 20, 32)
    ]
    d04_comparisons = []
    for m in (16, 20, 32):
        for metric in ("recall_at_10", "recall_at_20"):
            comparison = {"m": m, "metric": metric, "baseline_ef": 200, "candidate_ef": 300}
            rows = [[0.1, 0.2]]
            d04_comparisons.append({"comparison": comparison, **paired_basic_effect(rows, comparison=comparison),
                                    "raw_permutation_p": paired_permutation_p(rows, comparison=comparison), "paired_rows": rows})
    for comparison, adjusted in zip(d04_comparisons, holm_adjust([row["raw_permutation_p"] for row in d04_comparisons]), strict=True): comparison["holm_adjusted_p"] = adjusted
    d20_comparisons = []
    for metric in ("recall_at_10", "recall_at_20"):
        comparison = {"metric": metric, "baseline_m": 16, "candidate_m": slot["slot"]["m"], "baseline_ef": 100, "candidate_ef": 300, "baseline_build_id": builds[0]["build_id"], "candidate_build_id": next(build["build_id"] for build in builds if build["m"] == slot["slot"]["m"])}
        rows = [[0.1, 0.2]]
        d20_comparisons.append({"comparison": comparison, **paired_basic_effect(rows, comparison=comparison),
                                "raw_permutation_p": paired_permutation_p(rows, comparison=comparison), "paired_rows": rows})
    packet = {
        "schema_version": 1, "campaign_stage": "confirmation", "workflow_inputs_sha256": slot["record_self_sha256"],
        "slot": slot["slot"], "run_id": run_id, "run_attempt": 1, "job_id": run_id + 100,
        "job_key": "phase07-confirmation", "job_allocation_nonce": (f"nonce-{run_id:032x}"),
        "status": "numeric-success" if failure_class is None else "rejected",
        "failure_class": failure_class, "replacement_for_run_id": replacement_for,
        "builds": builds,
        "d04": {"family_name": "d04_ef_300_vs_200", "family_size": 6,
                "raw_p_values": [row["raw_permutation_p"] for row in d04_comparisons], "holm_adjusted_p_values": [row["holm_adjusted_p"] for row in d04_comparisons],
                "basic_ci_95": [row["basic_ci_95"] for row in d04_comparisons], "comparisons": d04_comparisons},
        "d20": {"family_name": "d20_current_baseline_member", "family_size": 2,
                "baseline_build_id": builds[0]["build_id"], "raw_p_values": [row["raw_permutation_p"] for row in d20_comparisons],
                "basic_ci_95": [row["basic_ci_95"] for row in d20_comparisons], "comparisons": d20_comparisons},
        "raw_tree_sha256": "a" * 64, "retention_days": 90,
    }
    packet["record_self_sha256"] = reconcile.canonical_digest(packet)
    return packet


def test_confirmation_plan_is_exactly_six_immutable_slots_with_generated_only_inputs() -> None:
    plan = _plan()
    assert [record["slot"] for record in plan["workflow_inputs"]] == [
        {"m": 32, "ordinal": 1}, {"m": 32, "ordinal": 2}, {"m": 32, "ordinal": 3},
        {"m": 20, "ordinal": 1}, {"m": 20, "ordinal": 2}, {"m": 20, "ordinal": 3},
    ]
    assert plan["artifact_reported_nominated_m"] == [16, 20]
    assert plan["authoritative_nominated_m"] == [32, 20]
    assert all(record["campaign_stage"] == "confirmation" for record in plan["workflow_inputs"])
    assert all(set(record) == operator.CONFIRMATION_WORKFLOW_INPUT_FIELDS for record in plan["workflow_inputs"])
    operator.validate_confirmation_plan(plan)


@pytest.mark.parametrize("mutation", [
    lambda plan: plan["workflow_inputs"].pop(),
    lambda plan: plan["workflow_inputs"].append(dict(plan["workflow_inputs"][0])),
    lambda plan: plan["workflow_inputs"][0].update(slot={"m": 16, "ordinal": 1}),
    lambda plan: plan["workflow_inputs"][0].update(record_self_sha256="0" * 64),
    lambda plan: plan.update(authoritative_nominated_m=[16, 20]),
])
def test_confirmation_plan_rejects_missing_extra_hand_authored_replayed_or_stale_inputs(mutation) -> None:
    plan = _plan()
    mutation(plan)
    with pytest.raises(ValueError):
        operator.validate_confirmation_plan(plan)


def test_confirmation_reconciliation_requires_six_eligible_and_preserves_typed_infra_origin() -> None:
    plan = _plan()
    packets = [_packet(slot, run_id=index + 1) for index, slot in enumerate(plan["workflow_inputs"])]
    ledger = reconcile.reconcile_confirmation(plan, packets)
    assert len(ledger["eligible_evidence_runs"]) == 6
    assert len(ledger["all_physical_workflow_runs"]) == 6
    assert [family["family_size"] for family in ledger["d20_ordinal_families"]] == [4, 4, 4]
    origin = _packet(plan["workflow_inputs"][0], run_id=1, failure_class="github_infrastructure")
    replacement = _packet(plan["workflow_inputs"][0], run_id=7, replacement_for=1)
    retried = [origin, replacement, *packets[1:]]
    ledger = reconcile.reconcile_confirmation(plan, retried)
    assert len(ledger["eligible_evidence_runs"]) == 6
    assert len(ledger["all_physical_workflow_runs"]) == 7
    assert ledger["all_physical_workflow_runs"][0]["eligible"] is False
    bad = [_packet(plan["workflow_inputs"][0], run_id=1, failure_class="numeric"), *packets[1:]]
    with pytest.raises(ValueError, match="non-infrastructure"): reconcile.reconcile_confirmation(plan, bad)


def test_packet_proves_three_fresh_builds_and_distinct_d04_d20_families() -> None:
    plan = _plan(); packet = _packet(plan["workflow_inputs"][0], run_id=1)
    reconcile.validate_confirmation_packet(packet, plan["workflow_inputs"][0])
    packet["builds"].append(dict(packet["builds"][0]))
    packet["record_self_sha256"] = reconcile.canonical_digest(packet)
    with pytest.raises(ValueError, match="three"): reconcile.validate_confirmation_packet(packet, plan["workflow_inputs"][0])


def test_confirmation_packet_rejects_missing_pairs_nonfinite_and_wrong_m16_baseline() -> None:
    plan = _plan(); packet = _packet(plan["workflow_inputs"][0], run_id=1)
    packet["d20"]["comparisons"][0]["paired_rows"] = []
    packet["record_self_sha256"] = reconcile.canonical_digest(packet)
    with pytest.raises(ValueError, match="paired"): reconcile.validate_confirmation_packet(packet, plan["workflow_inputs"][0])
    packet = _packet(plan["workflow_inputs"][0], run_id=1)
    packet["d20"]["raw_p_values"][0] = float("nan")
    packet["record_self_sha256"] = reconcile.canonical_digest(packet)
    with pytest.raises(ValueError, match="non-finite"): reconcile.validate_confirmation_packet(packet, plan["workflow_inputs"][0])
    packet = _packet(plan["workflow_inputs"][0], run_id=1)
    packet["d20"]["baseline_build_id"] = "0" * 64
    packet["record_self_sha256"] = reconcile.canonical_digest(packet)
    with pytest.raises(ValueError, match="m=16"): reconcile.validate_confirmation_packet(packet, plan["workflow_inputs"][0])


def test_confirmation_packet_requires_all_six_d04_members_and_declared_values() -> None:
    plan = _plan(); packet = _packet(plan["workflow_inputs"][0], run_id=1)
    packet["d04"]["comparisons"] = packet["d04"]["comparisons"][:-1]
    packet["record_self_sha256"] = reconcile.canonical_digest(packet)
    with pytest.raises(ValueError, match="members|cardinality"): reconcile.validate_confirmation_packet(packet, plan["workflow_inputs"][0])
    packet = _packet(plan["workflow_inputs"][0], run_id=1)
    packet["d04"]["raw_p_values"][0] = 0.02
    packet["record_self_sha256"] = reconcile.canonical_digest(packet)
    with pytest.raises(ValueError, match="declared"): reconcile.validate_confirmation_packet(packet, plan["workflow_inputs"][0])


def test_numeric_packet_rejects_omitted_or_duplicate_canonical_members() -> None:
    plan = _plan(); packet = _packet(plan["workflow_inputs"][0], run_id=1)
    packet["d04"].pop("comparisons")
    packet["record_self_sha256"] = reconcile.canonical_digest(packet)
    with pytest.raises(ValueError, match="members"): reconcile.validate_confirmation_packet(packet, plan["workflow_inputs"][0])
    packet = _packet(plan["workflow_inputs"][0], run_id=1)
    packet["d04"]["comparisons"][1]["comparison"] = dict(packet["d04"]["comparisons"][0]["comparison"])
    packet["record_self_sha256"] = reconcile.canonical_digest(packet)
    with pytest.raises(ValueError, match="canonical"): reconcile.validate_confirmation_packet(packet, plan["workflow_inputs"][0])


class _FakeActions:
    def __init__(self, jobs: list[dict]) -> None: self.jobs = jobs; self.urls: list[str] = []
    def get_json(self, url: str, token: str) -> dict:
        self.urls.append(url); return {"jobs": self.jobs}


def test_confirmation_allocation_seals_inner_workflow_input_digest_from_outer_dispatch_bundle(tmp_path: Path) -> None:
    """The hosted producer accepts the outer bundle but records only its sealed member."""
    plan = _plan()
    workflow_input = plan["workflow_inputs"][0]
    bundle = {"confirmation_request": plan["confirmation_request"], "workflow_input": workflow_input}
    output = tmp_path / "allocation.json"
    client = _FakeActions([{
        "id": 101, "name": "Phase 07 independent confirmation campaign",
        "run_id": 1, "run_attempt": 1,
    }])

    assert operator.seal_confirmation_allocation(
        workflow_inputs=bundle, output=output, repository="owner/repo", run_id=1,
        run_attempt=1, job_key="phase07-confirmation", head_sha=HEAD,
        token="test-token", client=client,
    ) == 0

    allocation = json.loads(output.read_text(encoding="utf-8"))
    assert allocation["workflow_inputs_sha256"] == workflow_input["record_self_sha256"]
    assert allocation["record_self_sha256"] == operator.canonical_digest(allocation)


def _confirmation_provenance(tmp_path: Path, *, expires_at: str = "2026-11-18T00:00:00Z",
                             duplicate: str | None = None, directory: bool = False) -> dict:
    """Create a byte-identical archive/extraction pair for provenance boundary tests."""
    archive, extracted = tmp_path / "confirmation.zip", tmp_path / "extracted"
    extracted.mkdir()
    contents = {name: f"{name}-content".encode("utf-8") for name in reconcile._CONFIRMATION_ARTIFACT_FILES}
    for name, content in contents.items():
        (extracted / name).write_bytes(content)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as compressed:
        for name, content in contents.items():
            compressed.writestr(name, content)
        if duplicate is not None:
            compressed.writestr(duplicate, contents[duplicate])
        if directory:
            compressed.writestr("unexpected-directory/", b"")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return {
        "run_id": 1, "run_attempt": 1, "job_id": 2, "artifact_id": 3,
        "artifact_name": "phase07-confirmation-1", "status": "completed", "conclusion": "success",
        "runner": {"name": "GitHub Actions test", "group": "GitHub Actions", "labels": ["ubuntu-latest"],
                   "os": "Linux", "image": "ubuntu", "architecture": "X64"},
        "run_created_at": "2026-08-20T00:00:00Z", "artifact_expires_at": expires_at,
        "api_archive_sha256": digest, "local_archive_sha256": digest,
        "archive": str(archive), "extracted_dir": str(extracted),
    }


@pytest.mark.parametrize("expires_at", [
    "2026-11-17T00:00:00Z",  # 89 days
    "2026-11-19T00:00:00Z",  # 91 days
    "2027-08-20T00:00:00Z",  # 365 days
])
def test_confirmation_provenance_rejects_nonexact_retention_windows(tmp_path: Path, expires_at: str) -> None:
    with pytest.raises(ValueError, match="retention"):
        reconcile._validate_confirmation_provenance(_confirmation_provenance(tmp_path, expires_at=expires_at))


@pytest.mark.parametrize("expires_at", [
    "2026-11-17T23:59:30Z",  # Stage 1 lower API tolerance
    "2026-11-18T00:00:00Z",  # exactly 90 days
    "2026-11-18T00:00:30Z",  # Stage 1 upper API tolerance
])
def test_confirmation_provenance_accepts_exact_retention_with_stage1_tolerance(tmp_path: Path, expires_at: str) -> None:
    provenance = _confirmation_provenance(tmp_path, expires_at=expires_at)
    assert reconcile._validate_confirmation_provenance(provenance) == provenance


@pytest.mark.parametrize("duplicate", ["confirmation-request.json", "confirmation-packet.json"])
def test_confirmation_provenance_rejects_duplicate_canonical_zip_members(tmp_path: Path, duplicate: str) -> None:
    with pytest.raises(ValueError, match="archive"):
        reconcile._validate_confirmation_provenance(_confirmation_provenance(tmp_path, duplicate=duplicate))


def test_confirmation_provenance_rejects_directory_zip_entry(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="archive"):
        reconcile._validate_confirmation_provenance(_confirmation_provenance(tmp_path, directory=True))


def test_attempt_scoped_job_allocation_is_unique_and_never_serializes_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeActions([{"id": 41, "name": "Phase 07 independent confirmation campaign", "run_id": 9, "run_attempt": 2}])
    allocation = operator.allocate_confirmation_job(client, repository="owner/repo", run_id=9, run_attempt=2,
                                                     job_key="phase07-confirmation", token="ghp_private_token")
    assert allocation["job_id"] == 41 and len(allocation["job_allocation_nonce"]) == 32
    assert client.urls == ["/repos/owner/repo/actions/runs/9/attempts/2/jobs"]
    assert "ghp_" not in json.dumps(allocation)
    for jobs in ([], client.jobs * 2, [{"id": 41, "name": "Phase 07 independent confirmation campaign", "run_id": 9, "run_attempt": 1}]):
        with pytest.raises(ValueError):
            operator.allocate_confirmation_job(_FakeActions(jobs), repository="owner/repo", run_id=9,
                                                            run_attempt=2, job_key="phase07-confirmation", token="x")
    monkeypatch.setattr(operator.secrets, "token_hex", lambda _: (_ for _ in ()).throw(RuntimeError("entropy")))
    with pytest.raises(ValueError, match="nonce"):
        operator.allocate_confirmation_job(client, repository="owner/repo", run_id=9, run_attempt=2,
                                           job_key="phase07-confirmation", token="x")


def test_confirmation_plan_cli_generates_request_inputs_and_preflight_bundle(tmp_path: Path) -> None:
    request, inputs, preflight = tmp_path / "request.json", tmp_path / "inputs", tmp_path / "preflight.json"
    result = subprocess.run([
        sys.executable, "eval/phase07_operator_gate.py", "confirmation-plan", "--stage1-ledger", str(LEDGER),
        "--request-file", str(request), "--workflow-inputs-dir", str(inputs), "--preflight-request", str(preflight),
    ], cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    generated = json.loads(request.read_text())
    bundles = [json.loads(path.read_text()) for path in inputs.glob("*.json")]
    records = sorted((bundle["workflow_input"] for bundle in bundles), key=lambda row: (-row["slot"]["m"], row["slot"]["ordinal"]))
    assert len(records) == 6 and generated["post_task0_head"] == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert preflight.exists()
    plan = {"schema_version": 1, "confirmation_request": generated, "workflow_inputs": records,
            "artifact_reported_nominated_m": [16, 20], "authoritative_nominated_m": [32, 20]}
    plan["record_self_sha256"] = operator.canonical_digest(plan)
    operator.validate_confirmation_plan(plan)
    assert all(operator.validate_confirmation_dispatch_bundle(bundle, expected_head=generated["post_task0_head"]) for bundle in bundles)


def test_actions_allocator_uses_workflow_display_name_and_nonce_is_unique_across_six() -> None:
    jobs = [{"id": 41, "name": "Phase 07 independent confirmation campaign", "run_id": 9, "run_attempt": 2}]
    allocations = [operator.allocate_confirmation_job(_FakeActions(jobs), repository="owner/repo", run_id=9,
                                                    run_attempt=2, job_key="phase07-confirmation", token="x")
                   for _ in range(6)]
    assert {row["job_allocation_nonce"] for row in allocations}.__len__() == 6


def test_dispatch_bundle_rejects_tampered_stale_and_cross_request_inputs() -> None:
    plan = _plan()
    bundle = {"confirmation_request": plan["confirmation_request"], "workflow_input": dict(plan["workflow_inputs"][0])}
    operator.validate_confirmation_dispatch_bundle(bundle, expected_head=HEAD)
    bundle["workflow_input"]["slot"] = {"m": 20, "ordinal": 1}
    bundle["workflow_input"]["record_self_sha256"] = operator.canonical_digest(bundle["workflow_input"])
    with pytest.raises(ValueError): operator.validate_confirmation_dispatch_bundle(bundle, expected_head=HEAD)
    pristine = {"confirmation_request": plan["confirmation_request"], "workflow_input": plan["workflow_inputs"][0]}
    with pytest.raises(ValueError, match="feature head|mismatch"): operator.validate_confirmation_dispatch_bundle(pristine, expected_head="e" * 40)
    other = operator.build_confirmation_plan(LEDGER, post_task0_head="d" * 40); pristine["confirmation_request"] = other["confirmation_request"]
    with pytest.raises(ValueError): operator.validate_confirmation_dispatch_bundle(pristine, expected_head=HEAD)


def test_dispatch_bundle_rejects_fully_resigned_noncanonical_authority() -> None:
    forged = operator.build_confirmation_plan(LEDGER, post_task0_head="e" * 40)
    bundle = {"confirmation_request": forged["confirmation_request"], "workflow_input": forged["workflow_inputs"][0]}
    with pytest.raises(ValueError, match="feature head|canonical"):
        operator.validate_confirmation_dispatch_bundle(bundle, expected_head="e" * 40)


def test_confirmation_reconciler_cli_consumes_packet_wrappers_and_seals_ledger(tmp_path: Path) -> None:
    plan = _plan(); request, ledger = tmp_path / "request.json", tmp_path / "ledger.json"
    request.write_text(json.dumps(plan["confirmation_request"]), encoding="utf-8")
    wrappers = [{"dispatch_bundle": {"confirmation_request": plan["confirmation_request"], "workflow_input": slot},
                 "packet": _packet(slot, run_id=index + 1)} for index, slot in enumerate(plan["workflow_inputs"])]
    ledger.write_text(json.dumps({"packets": wrappers}), encoding="utf-8")
    result = subprocess.run([sys.executable, "eval/reconcile_ann_gate.py", "--confirmation-request", str(request),
                             "--confirmation-ledger", str(ledger), "--mode", "confirmation"],
                            cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    sealed = json.loads(ledger.read_text())
    assert len(sealed["eligible_evidence_runs"]) == 6 and sealed["record_self_sha256"] == reconcile.canonical_digest(sealed)


def test_confirmation_exporter_and_postdownload_reconciler_run_real_cli_path(tmp_path: Path) -> None:
    """The success path is campaign CLI -> exporter -> zip/provenance -> reconciler.

    Unlike the legacy unit fixtures above, this deliberately creates no hand-written
    packet or wrapper.  The only numeric evidence comes from the production LanceDB
    confirmation campaign invoked through its CLI with a tiny trusted test config.
    """
    root = Path(__file__).resolve().parent.parent
    plan = _plan()
    request = tmp_path / "confirmation-request.json"
    request.write_text(json.dumps(plan["confirmation_request"]), encoding="utf-8")
    artifact_dirs: list[Path] = []
    for index, slot in enumerate(plan["workflow_inputs"], start=1):
        slot_root = tmp_path / f"slot-{index}"
        bundle = slot_root / "dispatch-bundle.json"
        bundle.parent.mkdir()
        dispatch_bundle = {"confirmation_request": plan["confirmation_request"], "workflow_input": slot}
        bundle.write_text(json.dumps(dispatch_bundle), encoding="utf-8")
        allocation_path = slot_root / "allocation.json"
        assert operator.seal_confirmation_allocation(
            workflow_inputs=dispatch_bundle, output=allocation_path, repository="owner/repo",
            run_id=index, run_attempt=1, job_key="phase07-confirmation", head_sha=HEAD,
            token="test-token", client=_FakeActions([{
                "id": 100 + index, "name": "Phase 07 independent confirmation campaign",
                "run_id": index, "run_attempt": 1,
            }]),
        ) == 0
        allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
        assert allocation["workflow_inputs_sha256"] == slot["record_self_sha256"]
        campaign_request = {
            "schema_version": 1, "stage": "confirmation", "request_id": f"tiny-{index}",
            "environment": {"head_sha": HEAD}, "model_manifest_sha256": "a" * 64,
            "corpus_manifest_sha256": "b" * 64, "workflow_inputs": slot,
            "run_identity": {name: allocation["allocation"][name] for name in ("run_id", "run_attempt", "job_id", "job_allocation_nonce")},
        }
        campaign_request_path = slot_root / "campaign-request.json"
        campaign_request_path.write_text(json.dumps(campaign_request), encoding="utf-8")
        output = slot_root / "campaign-output"
        campaign = subprocess.run([
            sys.executable, "eval/phase07_ann_campaign.py", "--request-file", str(campaign_request_path),
            "--output-dir", str(output), "--trusted-test-config", json.dumps({"rows": 32, "dimensions": 384, "probes": 2, "work_dir": str(slot_root / "builds")}),
        ], cwd=root, capture_output=True, text=True, check=False)
        assert campaign.returncode == 0, campaign.stderr
        if index == 1:
            for name, invalid_digest in (
                ("null", None),
                ("outer", operator.canonical_digest(dispatch_bundle)),
                ("tampered", "0" * 64),
            ):
                invalid = dict(allocation)
                invalid["workflow_inputs_sha256"] = invalid_digest
                invalid["record_self_sha256"] = operator.canonical_digest(invalid)
                invalid_allocation = slot_root / f"allocation-{name}.json"
                invalid_allocation.write_text(json.dumps(invalid), encoding="utf-8")
                rejected_artifact = slot_root / f"artifact-{name}"
                rejected = subprocess.run([
                    sys.executable, "eval/phase07_ann_campaign.py", "export-confirmation-packet",
                    "--campaign-output-dir", str(output), "--dispatch-bundle", str(bundle),
                    "--allocation-ledger", str(invalid_allocation), "--artifact-dir", str(rejected_artifact),
                ], cwd=root, capture_output=True, text=True, check=False)
                assert rejected.returncode == 1
                assert not (rejected_artifact / "confirmation-packet.json").exists()
                finalized = subprocess.run([
                    sys.executable, "eval/phase07_operator_gate.py", "finalize",
                    "--output-dir", str(rejected_artifact), "--stage", "confirmation",
                    "--head-sha", HEAD, "--run-id", str(index), "--run-attempt", "1",
                    "--job-key", "phase07-confirmation", "--job-status", "failure",
                ], cwd=root, capture_output=True, text=True, check=False)
                assert finalized.returncode == 0, finalized.stderr
                rejection = json.loads((rejected_artifact / "confirmation-pipeline-rejection.json").read_text())
                assert rejection["status"] == "reject-evidence"
                assert rejection["record_self_sha256"] == operator.canonical_digest(rejection)
        artifact = slot_root / "artifact"
        exported = subprocess.run([
            sys.executable, "eval/phase07_ann_campaign.py", "export-confirmation-packet",
            "--campaign-output-dir", str(output), "--dispatch-bundle", str(bundle),
            "--allocation-ledger", str(allocation_path), "--artifact-dir", str(artifact),
        ], cwd=root, capture_output=True, text=True, check=False)
        assert exported.returncode == 0, exported.stderr
        assert {path.name for path in artifact.iterdir()} == {
            "confirmation-request.json", "confirmation-ledger.json", "confirmation-result.json",
            "dispatch-bundle.json", "allocation.json", "confirmation-packet.json",
        }
        archive = slot_root / "archive.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for path in artifact.iterdir():
                zip_file.write(path, path.name)
        archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        extracted = slot_root / "extracted"
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(extracted)
        provenance = {
            "schema_version": 1, "evidence": [
                {"run_id": index, "run_attempt": 1, "job_id": 100 + index, "artifact_id": 1000 + index,
                 "artifact_name": f"phase07-confirmation-{index}", "status": "completed", "conclusion": "success",
                 "runner": {"name": "GitHub Actions test", "group": "GitHub Actions", "labels": ["ubuntu-latest"], "os": "Linux", "image": "ubuntu", "architecture": "X64"},
                 "run_created_at": "2026-08-20T00:00:00Z", "artifact_expires_at": "2026-11-18T00:00:00Z",
                 "api_archive_sha256": archive_sha256, "local_archive_sha256": archive_sha256,
                 "archive": str(archive), "extracted_dir": str(extracted)}
            ],
        }
        provenance["record_self_sha256"] = reconcile.canonical_digest(provenance)
        provenance_path = slot_root / "provenance.json"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        artifact_dirs.append(extracted)
        # Keep a per-slot manifest rather than a packet fixture; the reconciler consumes it below.
        (slot_root / "evidence.json").write_text(json.dumps({"artifact_dir": str(extracted), "provenance": str(provenance_path)}), encoding="utf-8")
    evidence_manifest = tmp_path / "evidence-manifest.json"
    manifest_payload = {"schema_version": 1, "evidence": [json.loads((path.parent / "evidence.json").read_text()) for path in artifact_dirs]}
    manifest_payload["record_self_sha256"] = reconcile.canonical_digest(manifest_payload)
    evidence_manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    reconciled = tmp_path / "reconciled-ledger.json"
    result = subprocess.run([
        sys.executable, "eval/reconcile_ann_gate.py", "--confirmation-request", str(request),
        "--confirmation-evidence-manifest", str(evidence_manifest), "--output", str(reconciled), "--mode", "confirmation-postdownload",
    ], cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    ledger = json.loads(reconciled.read_text())
    assert len(ledger["eligible_evidence_runs"]) == 6
    assert [family["family_size"] for family in ledger["d20_ordinal_families"]] == [4, 4, 4]

    first = artifact_dirs[0]
    provenance_path = first.parent / "provenance.json"
    archive_path = first.parent / "archive.zip"
    originals = {path: path.read_bytes() for path in (
        archive_path, provenance_path, first / "confirmation-result.json", first / "dispatch-bundle.json",
        first / "allocation.json", first / "confirmation-packet.json",
    )}

    def assert_rejected(mutator) -> None:
        mutator()
        rejected = subprocess.run([
            sys.executable, "eval/reconcile_ann_gate.py", "--confirmation-request", str(request),
            "--confirmation-evidence-manifest", str(evidence_manifest), "--output", str(reconciled), "--mode", "confirmation-postdownload",
        ], cwd=root, capture_output=True, text=True, check=False)
        assert rejected.returncode == 1
        for path, content in originals.items():
            path.write_bytes(content)

    assert_rejected(lambda: archive_path.write_bytes(originals[archive_path] + b"tamper"))
    def tamper_api_metadata() -> None:
        value = json.loads(originals[provenance_path])
        value["evidence"][0]["api_archive_sha256"] = "0" * 64
        value["record_self_sha256"] = reconcile.canonical_digest(value)
        provenance_path.write_text(json.dumps(value), encoding="utf-8")
    assert_rejected(tamper_api_metadata)
    for target in (first / "confirmation-result.json", first / "dispatch-bundle.json", first / "allocation.json"):
        assert_rejected(lambda target=target: target.write_bytes(originals[target] + b"tamper"))
    assert_rejected(lambda: (first / "confirmation-packet.json").write_text("{}", encoding="utf-8"))
