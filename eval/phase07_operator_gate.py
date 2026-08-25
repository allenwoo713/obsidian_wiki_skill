"""Fail-closed local operator gates for the Phase 07 hosted evidence workflow.

The program deliberately accepts only fixed JSON request/ledger paths.  It never
contacts GitHub and it rejects unsealed or unsafe operator state before a caller
can push, dispatch, or rely on an artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# ``python eval/phase07_operator_gate.py`` puts only ``eval/`` on sys.path.
# Establish the checkout root before any delayed ``eval.*`` import so the
# direct-file command used by both hosted runner families is deterministic.
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


SHA = re.compile(r"^[0-9a-f]{40}$")
STAGES = frozenset({"preflight", "screening", "confirmation", "continuation", "pr-acceptance"})
INFRA_FAILURES = frozenset({"github_infrastructure", "hosted_runner_unavailable", "artifact_service"})
SECRET_MARKERS = ("token", "secret", "password", "authorization", "private_key", "ghp_")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
STAGE1_RUNTIME_FIELDS = frozenset({
    "branch", "workflow_path", "head_sha", "run_id", "run_attempt", "job_key", "job_allocation_nonce", "runtime",
})
STAGE1_RUNTIME_IDENTITY = {
    "python": "3.13", "lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0",
    "omp_num_threads": 2,
}
STAGE1_LEDGER_DIGEST = "6c424135ec4db8983136826575d9436b6ba88da5029384ada47fadb2d1918e33"
STAGE1_NUMERIC_HEAD = "c45934d7fb8e699faa389c1dc3e80bb9dcc774c8"
CONFIRMATION_IMPLEMENTATION_BASE = "105dab9764cb58fb10610b78345b85d7282ef4d6"
CANONICAL_STAGE1_AUTHORITY_PATH = _REPOSITORY_ROOT / "eval" / "phase07-stage1-authority.json"
CANONICAL_STAGE1_AUTHORITY_REFERENCE = "eval/phase07-stage1-authority.json"
STAGE1_AUTHORITY_DIGEST = "a983ac4be1bb992e237474fc237cd927529ef9dd60e9330de7b959e6159bb319"
CONFIRMATION_SLOTS = (1, 2, 3)
D25_BASELINE = {"index_type": "hnsw_sq", "m": 16, "ef_construction": 300, "query_ef": 100}
D25_CANDIDATES = (
    {"index_type": "hnsw_sq", "m": 20, "ef_construction": 300, "query_ef": 300},
    {"index_type": "hnsw_sq", "m": 32, "ef_construction": 300, "query_ef": 300},
)
CONFIRMATION_WORKFLOW_INPUT_FIELDS = frozenset({
    "schema_version", "campaign_stage", "confirmation_request_sha256",
    "slot", "post_task0_head", "d25_binding",
    "d25_binding_sha256", "dispatch_identity", "record_self_sha256",
})
JOB_DISPLAY_NAMES = {"phase07-confirmation": "Phase 07 independent confirmation campaign"}
CONFIRMATION_DOWNLOAD_FIELDS = frozenset({"run_id", "run_attempt", "archive", "extracted_dir"})


def canonical_digest(payload: dict[str, Any]) -> str:
    """Digest a record without its recursive self-digest field."""
    value = dict(payload)
    value.pop("record_self_sha256", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sealed(payload: dict[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    record["record_self_sha256"] = canonical_digest(record)
    return record


def d25_confirmation_binding(ordinal: int) -> dict[str, Any]:
    """Return the complete immutable D-25 build/query allowlist for one run."""
    if isinstance(ordinal, bool) or ordinal not in CONFIRMATION_SLOTS:
        raise ValueError("D-25 confirmation ordinal must be 1, 2, or 3")
    return {
        "kind": "phase07-d25-paired-confirmation/v1",
        "prior_screening_sha256": STAGE1_LEDGER_DIGEST,
        "run_ordinal": ordinal,
        "baseline": dict(D25_BASELINE),
        "candidates": [dict(candidate) for candidate in D25_CANDIDATES],
    }


def confirmation_input_membership_digest(record: dict[str, Any]) -> str:
    """Stable member identity deliberately excludes the request/self digest cycle."""
    value = dict(record)
    value.pop("record_self_sha256", None)
    value.pop("confirmation_request_sha256", None)
    return canonical_digest(value)


def _build_confirmation_plan_from_immutable(*, stage1_ledger_path: str, post_task0_head: str) -> dict[str, Any]:
    if not SHA.fullmatch(post_task0_head):
        raise ValueError("confirmation requires an exact post-Task-0 head")
    seed_records = []
    for ordinal in CONFIRMATION_SLOTS:
        slot = {"ordinal": ordinal}
        binding = d25_confirmation_binding(ordinal)
        seed_records.append({"schema_version": 1, "campaign_stage": "confirmation", "slot": slot,
                             "post_task0_head": post_task0_head, "d25_binding": binding,
                             "d25_binding_sha256": canonical_digest(binding),
                             "dispatch_identity": f"phase07-confirmation/ordinal/{ordinal}"})
    request = _sealed({
        "schema_version": 1, "campaign_stage": "confirmation",
        "stage1_ledger_path": stage1_ledger_path, "stage1_ledger_sha256": STAGE1_LEDGER_DIGEST,
        "artifact_reported_nominated_m": [16, 20], "reconciled_nominated_m": [32, 20],
        "authoritative_nominated_m": [32, 20], "stage1_numeric_head": STAGE1_NUMERIC_HEAD,
        "confirmation_implementation_base": CONFIRMATION_IMPLEMENTATION_BASE,
        "post_task0_head": post_task0_head,
        "workflow_input_membership_sha256": [confirmation_input_membership_digest(record) for record in seed_records],
    })
    inputs = []
    for record in seed_records:
        inputs.append(_sealed({**record, "confirmation_request_sha256": request["record_self_sha256"]}))
    return _sealed({
        "schema_version": 1, "confirmation_request": request, "workflow_inputs": inputs,
        "artifact_reported_nominated_m": [16, 20], "authoritative_nominated_m": [32, 20],
    })


def build_confirmation_plan(stage1_ledger: Path, *, post_task0_head: str) -> dict[str, Any]:
    """Derive all confirmation dispatch authority from the immutable Stage 1 ledger."""
    if stage1_ledger.resolve() != CANONICAL_STAGE1_AUTHORITY_PATH.resolve():
        raise ValueError("confirmation requires canonical Stage 1 authority path")
    ledger = _read_object(stage1_ledger)
    required = {"schema_version", "kind", "original_ledger_sha256", "repository", "branch", "workflow_path", "head_sha", "run", "artifact", "runner", "runtime", "source_digests", "model_manifest_sha256", "corpus_manifest_sha256", "lock_identity", "stress_identity", "artifact_reported_nominated_m", "authoritative_nominated_m", "authorization", "record_self_sha256"}
    if set(ledger) != required or ledger.get("schema_version") != 1 or ledger.get("kind") != "phase07-sealed-stage1-authority/v1":
        raise ValueError("unrecognized compact Stage 1 authority schema")
    recomputed_digest = canonical_digest(ledger)
    if (
        ledger.get("record_self_sha256") != STAGE1_AUTHORITY_DIGEST
        or recomputed_digest != STAGE1_AUTHORITY_DIGEST
        or ledger.get("original_ledger_sha256") != STAGE1_LEDGER_DIGEST
    ):
        raise ValueError("unrecognized immutable Stage 1 authority")
    if ledger.get("artifact_reported_nominated_m") != [16, 20] or ledger.get("authoritative_nominated_m") != [32, 20] or ledger.get("head_sha") != STAGE1_NUMERIC_HEAD:
        raise ValueError("immutable Stage 1 nominee/head provenance")
    return _build_confirmation_plan_from_immutable(
        stage1_ledger_path=CANONICAL_STAGE1_AUTHORITY_REFERENCE,
        post_task0_head=post_task0_head,
    )


def validate_confirmation_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "confirmation_request", "workflow_inputs", "artifact_reported_nominated_m", "authoritative_nominated_m", "record_self_sha256"} or plan.get("schema_version") != 1:
        raise ValueError("sealed confirmation plan schema")
    if plan.get("record_self_sha256") != canonical_digest(plan):
        raise ValueError("confirmation plan self-digest")
    request = plan["confirmation_request"]
    required_request = {"schema_version", "campaign_stage", "stage1_ledger_path", "stage1_ledger_sha256", "artifact_reported_nominated_m", "reconciled_nominated_m", "authoritative_nominated_m", "stage1_numeric_head", "confirmation_implementation_base", "post_task0_head", "workflow_input_membership_sha256", "record_self_sha256"}
    if not isinstance(request, dict) or set(request) != required_request or request.get("record_self_sha256") != canonical_digest(request):
        raise ValueError("sealed confirmation request")
    if request["campaign_stage"] != "confirmation" or request["stage1_ledger_sha256"] != STAGE1_LEDGER_DIGEST or request["stage1_numeric_head"] != STAGE1_NUMERIC_HEAD or request["confirmation_implementation_base"] != CONFIRMATION_IMPLEMENTATION_BASE or not SHA.fullmatch(request["post_task0_head"]):
        raise ValueError("confirmation request provenance")
    if plan["artifact_reported_nominated_m"] != [16, 20] or plan["authoritative_nominated_m"] != [32, 20] or request["artifact_reported_nominated_m"] != [16, 20] or request["reconciled_nominated_m"] != [32, 20] or request["authoritative_nominated_m"] != [32, 20]:
        raise ValueError("confirmation nominees")
    inputs = plan["workflow_inputs"]
    if not isinstance(inputs, list) or len(inputs) != 3:
        raise ValueError("exactly three generated confirmation ordinal inputs")
    slots = []
    for record in inputs:
        if not isinstance(record, dict) or set(record) != CONFIRMATION_WORKFLOW_INPUT_FIELDS or record.get("record_self_sha256") != canonical_digest(record):
            raise ValueError("sealed generated workflow input")
        slot = record.get("slot")
        if not isinstance(slot, dict) or set(slot) != {"ordinal"} or slot["ordinal"] not in CONFIRMATION_SLOTS:
            raise ValueError("immutable confirmation slot")
        binding = d25_confirmation_binding(slot["ordinal"])
        if record.get("campaign_stage") != "confirmation" or record.get("confirmation_request_sha256") != request["record_self_sha256"] or record.get("post_task0_head") != request["post_task0_head"] or record.get("d25_binding") != binding or record.get("d25_binding_sha256") != canonical_digest(binding) or record.get("dispatch_identity") != f"phase07-confirmation/ordinal/{slot['ordinal']}":
            raise ValueError("confirmation input binding")
        slots.append(slot["ordinal"])
    if slots != list(CONFIRMATION_SLOTS):
        raise ValueError("duplicate, missing, or replayed confirmation slot")
    if request["workflow_input_membership_sha256"] != [confirmation_input_membership_digest(record) for record in inputs] or len(set(request["workflow_input_membership_sha256"])) != 3:
        raise ValueError("confirmation request bundle membership")
    return plan


def validate_confirmation_dispatch_bundle(bundle: dict[str, Any], *, expected_head: str) -> dict[str, Any]:
    """Require an input to carry the sealed request that proves bundle membership."""
    allowed = {"confirmation_request", "workflow_input", "replacement_for_run_id"}
    if not isinstance(bundle, dict) or set(bundle) != allowed:
        raise ValueError("typed confirmation dispatch bundle")
    request, record = bundle["confirmation_request"], bundle["workflow_input"]
    if not SHA.fullmatch(expected_head) or _git("rev-parse", "HEAD") != expected_head:
        raise ValueError("confirmation feature head is not the exact generated head")
    expected = build_confirmation_plan(CANONICAL_STAGE1_AUTHORITY_PATH, post_task0_head=expected_head)
    # The request is not a self-declared authority: it must byte-for-byte match
    # the deterministic immutable lineage and all three generated members.
    if request != expected["confirmation_request"]:
        raise ValueError("noncanonical confirmation request authority")
    validate_confirmation_plan(expected)
    slot = record.get("slot") if isinstance(record, dict) else None
    expected_record = next((item for item in expected["workflow_inputs"] if item["slot"] == slot), None)
    if expected_record is None or record != expected_record:
        raise ValueError("confirmation input is not a canonical generated member")
    if bundle.get("replacement_for_run_id") is not None:
        raise ValueError("D-25 confirmation replacements are not authorized")
    return record


def validate_confirmation_workflow_input(record: dict[str, Any], *, expected_head: str) -> dict[str, Any]:
    """Fail closed before build when a dispatched input is not one canonical slot."""
    if not SHA.fullmatch(expected_head) or not isinstance(record, dict) or set(record) != CONFIRMATION_WORKFLOW_INPUT_FIELDS or record.get("record_self_sha256") != canonical_digest(record):
        raise ValueError("sealed generated workflow input")
    slot = record.get("slot")
    if not isinstance(slot, dict) or set(slot) != {"ordinal"} or slot["ordinal"] not in CONFIRMATION_SLOTS:
        raise ValueError("immutable confirmation slot")
    binding = d25_confirmation_binding(slot["ordinal"])
    if record.get("campaign_stage") != "confirmation" or record.get("post_task0_head") != expected_head or not isinstance(record.get("confirmation_request_sha256"), str) or not HEX64.fullmatch(record["confirmation_request_sha256"]) or record.get("d25_binding") != binding or record.get("d25_binding_sha256") != canonical_digest(binding) or record.get("dispatch_identity") != f"phase07-confirmation/ordinal/{slot['ordinal']}":
        raise ValueError("confirmation input/request/head/binding mismatch")
    return record


def allocate_confirmation_job(client: Any, *, repository: str, run_id: int, run_attempt: int,
                              job_key: str, token: str) -> dict[str, Any]:
    """Bind the hosted allocation before any build without retaining the token."""
    if not repository or not isinstance(run_id, int) or run_id <= 0 or not isinstance(run_attempt, int) or run_attempt <= 0 or not job_key or not token:
        raise ValueError("invalid attempt-scoped allocation identity")
    try:
        response = client.get_json(f"/repos/{repository}/actions/runs/{run_id}/attempts/{run_attempt}/jobs", token)
        jobs = response.get("jobs") if isinstance(response, dict) else None
        expected_name = JOB_DISPLAY_NAMES.get(job_key, job_key)
        matches = [job for job in jobs if isinstance(job, dict) and job.get("name") == expected_name and job.get("run_id") == run_id and job.get("run_attempt") == run_attempt] if isinstance(jobs, list) else []
        if len(matches) != 1 or not isinstance(matches[0].get("id"), int) or matches[0]["id"] <= 0:
            raise ValueError("attempt-scoped job must match exactly once")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("attempt-scoped job lookup failed") from exc
    try:
        nonce = secrets.token_hex(16)
    except Exception as exc:
        raise ValueError("allocation nonce generation failed") from exc
    if not isinstance(nonce, str) or len(nonce) != 32:
        raise ValueError("allocation nonce generation failed")
    return {"run_id": run_id, "run_attempt": run_attempt, "job_id": matches[0]["id"], "job_key": job_key, "job_allocation_nonce": nonce}


class GitHubActionsClient:
    """Minimal fakeable stdlib client; its token is header-only and never serialized."""
    def get_json(self, url: str, token: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"https://api.github.com{url}", headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                if response.status != 200:
                    raise ValueError("attempt-scoped job API status")
                value = json.loads(response.read().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("attempt-scoped job API unavailable") from exc
        if not isinstance(value, dict):
            raise ValueError("attempt-scoped job API JSON")
        return value


def seal_confirmation_allocation(*, workflow_inputs: dict[str, Any], output: Path, repository: str,
                                 run_id: int, run_attempt: int, job_key: str, head_sha: str, token: str,
                                 client: Any | None = None) -> int:
    """Seal lookup/entropy rejection before a workflow is permitted to build."""
    record: dict[str, Any] = {"schema_version": 1, "campaign_stage": "confirmation"}
    try:
        workflow_input = validate_confirmation_dispatch_bundle(workflow_inputs, expected_head=head_sha)
        record["workflow_inputs_sha256"] = workflow_input["record_self_sha256"]
        allocation = allocate_confirmation_job(client or GitHubActionsClient(), repository=repository, run_id=run_id,
                                                run_attempt=run_attempt, job_key=job_key, token=token)
        record.update(status="success", allocation=allocation)
        _write_ledger(output, record)
        return 0
    except Exception:
        record.update(status="reject-evidence", failure_class="attempt_scoped_job_or_nonce")
        _write_ledger(output, record)
        return 1


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("request and ledger must be JSON objects")
    return value


def _reject_secrets(value: Any, *, location: str = "payload") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in SECRET_MARKERS):
                raise ValueError(f"secret-like field is forbidden: {location}.{key}")
            _reject_secrets(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, location=f"{location}[{index}]")
    elif isinstance(value, str) and any(marker in value.lower() for marker in ("ghp_", "github_pat_", "bearer ")):
        raise ValueError(f"secret-like value is forbidden: {location}")


def validate_stage1_screening_runtime(environment: object) -> dict[str, Any]:
    """Reject a hosted Stage 1 request that is not bound to its allocation/runtime."""
    if not isinstance(environment, dict) or set(environment) != STAGE1_RUNTIME_FIELDS:
        raise ValueError("strict Stage 1 hosted runtime binding")
    if not isinstance(environment["branch"], str) or environment["branch"] in {"", "main", "master"} or environment["workflow_path"] != ".github/workflows/eval.yml":
        raise ValueError("Stage 1 branch/workflow path")
    if not isinstance(environment["head_sha"], str) or not SHA.fullmatch(environment["head_sha"]):
        raise ValueError("Stage 1 immutable head")
    if not isinstance(environment["run_id"], int) or environment["run_id"] <= 0:
        raise ValueError("Stage 1 run identity")
    if not isinstance(environment["run_attempt"], int) or environment["run_attempt"] <= 0:
        raise ValueError("Stage 1 run attempt")
    if not isinstance(environment["job_key"], str) or not environment["job_key"]:
        raise ValueError("Stage 1 job key")
    if environment["job_allocation_nonce"] != f"{environment['run_id']}-{environment['run_attempt']}-{environment['job_key']}":
        raise ValueError("Stage 1 job allocation nonce")
    if environment["runtime"] != STAGE1_RUNTIME_IDENTITY:
        raise ValueError("Stage 1 locked runtime/OMP identity")
    return environment


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()


def validate_feature_worktree_preflight(request: dict[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    """Validate exact feature worktree identity and its explicit dirty policy."""
    allowed = {"repository", "branch", "worktree_root", "head_sha", "allowed_dirty_paths", "workflow_name", "campaign_stage", "continuation_binding", "require_upstream_head", "ledger_path", "evidence_packet"}
    unknown = set(request) - allowed
    if unknown:
        raise ValueError(f"unknown request fields: {sorted(unknown)}")
    required = {"repository", "branch", "worktree_root", "head_sha", "allowed_dirty_paths", "workflow_name", "campaign_stage"}
    if set(request) & required != required:
        raise ValueError("missing required preflight fields")
    if request["campaign_stage"] not in STAGES:
        raise ValueError("unknown campaign_stage")
    if not isinstance(request["branch"], str) or request["branch"] in {"master", "main"}:
        raise ValueError("unsafe feature branch")
    if not isinstance(request["head_sha"], str) or not SHA.fullmatch(request["head_sha"]):
        raise ValueError("invalid immutable head SHA")
    if not isinstance(request["allowed_dirty_paths"], list) or not all(isinstance(item, str) and item and not Path(item).is_absolute() and ".." not in Path(item).parts for item in request["allowed_dirty_paths"]):
        raise ValueError("invalid allowed_dirty_paths")
    actual_root = (root or Path(_git("rev-parse", "--show-toplevel"))).resolve()
    if actual_root != Path(request["worktree_root"]).resolve():
        raise ValueError("wrong worktree root")
    if _git("branch", "--show-current") != request["branch"]:
        raise ValueError("wrong feature branch")
    if _git("rev-parse", "HEAD") != request["head_sha"]:
        raise ValueError("wrong immutable head")
    status = _git("status", "--porcelain=v1").splitlines()
    dirty = sorted(line[3:] for line in status if len(line) >= 4)
    if set(dirty) - set(request["allowed_dirty_paths"]):
        raise ValueError("disallowed dirty worktree state")
    return {"worktree_root": str(actual_root), "branch": request["branch"], "head_sha": request["head_sha"], "status": dirty}


def validate_phase07_evidence_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Validate untrusted hosted-artifact metadata at the local trust boundary."""
    required = {"repository", "run_id", "run_attempt", "job_id", "job_allocation_nonce", "artifact_id", "artifact_name", "archive_sha256", "content_sha256", "record_self_sha256", "retention_days_requested", "retention_days_accepted", "head_sha", "runner", "lock_identity", "build_id", "retry_lineage"}
    if set(packet) != required:
        raise ValueError("untrusted artifact packet fields")
    _reject_secrets(packet)
    if not isinstance(packet["repository"], str) or not packet["repository"]:
        raise ValueError("repository binding")
    for key in ("run_id", "run_attempt", "job_id", "artifact_id", "retention_days_requested", "retention_days_accepted"):
        if not isinstance(packet[key], int) or packet[key] <= 0:
            raise ValueError(f"invalid {key}")
    if not isinstance(packet["job_allocation_nonce"], str) or len(packet["job_allocation_nonce"]) < 16:
        raise ValueError("job allocation nonce")
    if not SHA.fullmatch(packet["head_sha"]) or not all(isinstance(packet[key], str) and HEX64.fullmatch(packet[key]) for key in ("archive_sha256", "content_sha256")):
        raise ValueError("immutable digest or head binding")
    if packet["retention_days_requested"] != 90 or packet["retention_days_accepted"] != 90:
        raise ValueError("exact 90-day requested and accepted retention is mandatory")
    if not isinstance(packet["artifact_name"], str) or not packet["artifact_name"]:
        raise ValueError("artifact name binding")
    if not isinstance(packet["runner"], dict) or not {"os", "image", "architecture"} <= set(packet["runner"]):
        raise ValueError("runner variance metadata")
    retry = packet["retry_lineage"]
    if not isinstance(retry, dict) or set(retry) != {"failure_class", "original_run_id", "replacement_run_id"}:
        raise ValueError("retry lineage")
    if retry["replacement_run_id"] is not None and retry["failure_class"] not in INFRA_FAILURES:
        raise ValueError("only classified GitHub infrastructure failures are retryable")
    if packet["record_self_sha256"] != canonical_digest(packet):
        raise ValueError("artifact self-digest")
    return packet


def validate_phase07_evidence_set(packets: list[dict[str, Any]], *, expected_repository: str, expected_head: str) -> list[dict[str, Any]]:
    """Require distinct hosted allocations and fresh builds; runner variance is data, not equality."""
    if not packets:
        raise ValueError("hosted evidence set is empty")
    tuples: set[tuple[int, int, int, str]] = set()
    builds: set[str] = set()
    for packet in packets:
        validate_phase07_evidence_packet(packet)
        if packet["repository"] != expected_repository or packet["head_sha"] != expected_head:
            raise ValueError("repository or head binding")
        allocation = (packet["run_id"], packet["run_attempt"], packet["job_id"], packet["job_allocation_nonce"])
        if allocation in tuples:
            raise ValueError("reused hosted allocation tuple")
        tuples.add(allocation)
        build_id = packet["build_id"]
        if not isinstance(build_id, str) or not HEX64.fullmatch(build_id) or build_id in builds:
            raise ValueError("fresh build identity")
        builds.add(build_id)
    return packets


def _write_ledger(path: Path, record: dict[str, Any]) -> None:
    record["record_self_sha256"] = canonical_digest(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def seal_hosted_preflight(*, output: Path, stage: str, continuation: str, repository: str,
                          run_id: int, run_attempt: int, head_sha: str, job_key: str,
                          runner_os: str, runner_architecture: str, root: Path = Path(".")) -> int:
    """Stdlib-only hosted preflight: always seal success or rejection before exit."""
    record: dict[str, Any] = {
        "schema_version": 2, "repository": repository, "run_id": run_id,
        "run_attempt": run_attempt, "head_sha": head_sha, "campaign_stage": "preflight",
        "job_key": job_key, "job_allocation_nonce": f"{run_id}-{run_attempt}-{job_key}",
        "runner": {"os": runner_os, "architecture": runner_architecture},
        "retention_days_requested": 90, "authorization": "none",
    }
    try:
        if stage != "preflight" or continuation:
            raise ValueError("preflight accepts only empty continuation input")
        if not SHA.fullmatch(head_sha) or not repository or not job_key:
            raise ValueError("invalid immutable hosted identity")
        lock = (root / "requirements.txt").read_text(encoding="utf-8")
        manifest = json.loads((root / "eval" / "model-manifest.json").read_text(encoding="utf-8"))
        if not all(value in lock for value in ("lancedb==0.34.0", "numpy==2.2.6", "pyarrow==25.0.0")):
            raise ValueError("lock syntax")
        from eval.ann_corpus_manifest import validate_model_manifest
        validate_model_manifest(manifest)
        record["status"] = "success"
        record["asvs_l1"] = {"input_validation": "pass", "secret_handling": "pass", "access_control": "pending-artifact-upload", "artifact_trust_boundary": "pending-artifact-upload"}
        code = 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        record["status"] = "reject-evidence"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        code = 1
    _write_ledger(output, record)
    return code


def finalize_pipeline_artifact(*, output_dir: Path, stage: str, head_sha: str,
                               run_id: int, run_attempt: int, job_key: str,
                               job_status: str) -> int:
    """Preserve campaign output or seal a no-secret rejection for every Python-visible failure."""
    if stage == "confirmation":
        if job_status == "success":
            try:
                from eval.phase07_ann_campaign import validate_confirmation_artifact_tree
                validate_confirmation_artifact_tree(
                    output_dir, expected_head=head_sha, expected_run_id=run_id,
                    expected_run_attempt=run_attempt, expected_job_key=job_key,
                )
                return 0
            except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        _seal_confirmation_pipeline_rejection(
            output_dir=output_dir, stage=stage, head_sha=head_sha, run_id=run_id,
            run_attempt=run_attempt, job_key=job_key, job_status=job_status,
        )
        return 0
    # A caller cannot relabel a confirmation tree as another campaign stage to
    # bypass its strict finalizer.  This check is lexical and never traverses a
    # symlinked root.
    if output_dir.is_symlink() or (output_dir.is_dir() and (output_dir / "confirmation-packet.json").exists()):
        _seal_confirmation_pipeline_rejection(
            output_dir=output_dir, stage=stage, head_sha=head_sha, run_id=run_id,
            run_attempt=run_attempt, job_key=job_key, job_status=job_status,
        )
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*-result.json")) or any(output_dir.glob("*-rejection.json")):
        return 0
    _write_ledger(output_dir / f"{stage}-pipeline-rejection.json", {
        "schema_version": 1, "stage": stage, "status": "reject-evidence",
        "head_sha": head_sha, "run_id": run_id, "run_attempt": run_attempt,
        "job_key": job_key, "job_status": job_status, "authorization": "none",
    })
    return 0


def _seal_confirmation_pipeline_rejection(*, output_dir: Path, stage: str, head_sha: str,
                                          run_id: int, run_attempt: int, job_key: str,
                                          job_status: str) -> None:
    """Replace only our output root; never traverse an external symlink target."""
    if output_dir.is_symlink():
        output_dir.unlink()
    elif output_dir.exists() and not output_dir.is_dir():
        output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            # ``rmtree`` refuses a symlink argument; the branch is regular and
            # remains rooted inside this just-created/owned output directory.
            shutil.rmtree(path)
        else:
            path.unlink()
    _write_ledger(output_dir / "confirmation-pipeline-rejection.json", {
        "schema_version": 1, "stage": stage, "status": "reject-evidence",
        "head_sha": head_sha, "run_id": run_id, "run_attempt": run_attempt,
        "job_key": job_key, "job_status": job_status, "authorization": "none",
    })


def reconcile_hosted(binding_file: Path, output: Path) -> int:
    """Seal strict operator-supplied hosted evidence; no download or authorization occurs here."""
    try:
        binding = _read_object(binding_file); _reject_secrets(binding)
        if set(binding) != {"repository", "head_sha", "packets"} or not isinstance(binding["packets"], list):
            raise ValueError("strict reconciliation binding")
        packets = validate_phase07_evidence_set(binding["packets"], expected_repository=binding["repository"], expected_head=binding["head_sha"])
        output.mkdir(parents=True, exist_ok=True)
        _write_ledger(output / "reconciliation-result.json", {"schema_version": 1, "status": "success", "authorization": "none", "repository": binding["repository"], "head_sha": binding["head_sha"], "packets": packets})
        return 0
    except (ValueError, OSError, json.JSONDecodeError, KeyError) as exc:
        output.mkdir(parents=True, exist_ok=True)
        _write_ledger(output / "reconciliation-rejection.json", {"schema_version": 1, "status": "reject-evidence", "authorization": "none", "reason": f"{type(exc).__name__}: {exc}"})
        return 1


def collect_confirmation_provenance(*, request_file: Path, output: Path,
                                    provenance_dir: Path, token: str,
                                    client: Any | None = None) -> int:
    """Collect API-bound provenance for downloaded confirmation artifacts.

    The request supplies only immutable run/attempt identities and local download
    paths.  Every status, job, runner label, artifact identity, expiry, and API
    digest comes from GitHub; locked host details come from the already-sealed
    production packet and are cross-checked again by the post-download reconciler.
    """
    request = _read_object(request_file)
    _reject_secrets(request)
    if set(request) != {"schema_version", "repository", "head_sha", "downloads", "record_self_sha256"} \
            or request.get("schema_version") != 1 \
            or request.get("record_self_sha256") != canonical_digest(request):
        raise ValueError("sealed confirmation download collection request")
    repository, head_sha, downloads = request["repository"], request["head_sha"], request["downloads"]
    if not isinstance(repository, str) or not repository or not isinstance(head_sha, str) or not SHA.fullmatch(head_sha):
        raise ValueError("confirmation collection repository/head binding")
    if not isinstance(downloads, list) or len(downloads) != 3:
        raise ValueError("confirmation collection requires exactly three successful downloads")
    if not token:
        raise ValueError("GitHub actions read token unavailable")
    if output.exists() or provenance_dir.is_symlink() or (provenance_dir.exists() and any(provenance_dir.iterdir())):
        raise ValueError("confirmation provenance output must be new and empty")

    from eval.phase07_ann_campaign import validate_confirmation_artifact_tree
    from eval.reconcile_ann_gate import _validate_confirmation_provenance

    github = client or GitHubActionsClient()
    records: list[dict[str, Any]] = []
    seen_runs: set[tuple[int, int]] = set()
    seen_paths: set[tuple[Path, Path]] = set()
    for download in downloads:
        if not isinstance(download, dict) or set(download) != CONFIRMATION_DOWNLOAD_FIELDS:
            raise ValueError("strict confirmation download record")
        run_id, run_attempt = download["run_id"], download["run_attempt"]
        if not isinstance(run_id, int) or run_id <= 0 or not isinstance(run_attempt, int) or run_attempt <= 0:
            raise ValueError("confirmation download run identity")
        archive, extracted = Path(download["archive"]), Path(download["extracted_dir"])
        identity, paths = (run_id, run_attempt), (archive.resolve(), extracted.resolve())
        if identity in seen_runs or paths in seen_paths or archive.is_symlink() or not archive.is_file() \
                or extracted.is_symlink() or not extracted.is_dir():
            raise ValueError("duplicate or unavailable confirmation download")
        seen_runs.add(identity); seen_paths.add(paths)

        validated = validate_confirmation_artifact_tree(
            extracted, expected_head=head_sha, expected_run_id=run_id,
            expected_run_attempt=run_attempt, expected_job_key="phase07-confirmation",
        )
        packet, allocation = validated["packet"], validated["allocation"]["allocation"]
        run_url = f"/repos/{repository}/actions/runs/{run_id}"
        jobs_url = f"{run_url}/attempts/{run_attempt}/jobs"
        artifacts_url = f"{run_url}/artifacts"
        try:
            run = github.get_json(run_url, token)
            jobs_payload = github.get_json(jobs_url, token)
            artifacts_payload = github.get_json(artifacts_url, token)
        except Exception as exc:
            raise ValueError("GitHub confirmation provenance lookup failed") from exc

        if not isinstance(run, dict) or run.get("id") != run_id or run.get("run_attempt") != run_attempt \
                or run.get("head_sha") != head_sha or not isinstance(run.get("head_branch"), str) \
                or run["head_branch"] in {"", "main", "master"} or run.get("event") != "workflow_dispatch" \
                or run.get("status") != "completed" or run.get("conclusion") != "success" \
                or not isinstance(run.get("created_at"), str):
            raise ValueError("confirmation workflow-run API binding")
        jobs = jobs_payload.get("jobs") if isinstance(jobs_payload, dict) else None
        job_name = JOB_DISPLAY_NAMES["phase07-confirmation"]
        matches = [job for job in jobs if isinstance(job, dict) and job.get("id") == allocation["job_id"]
                   and job.get("run_id") == run_id and job.get("run_attempt") == run_attempt
                   and job.get("name") == job_name and job.get("status") == "completed"
                   and job.get("conclusion") == "success"] if isinstance(jobs, list) else []
        if len(matches) != 1:
            raise ValueError("confirmation attempt-job API binding")
        job = matches[0]
        if not isinstance(job.get("runner_name"), str) or not job["runner_name"] \
                or not isinstance(job.get("runner_group_name"), str) or not job["runner_group_name"] \
                or not isinstance(job.get("labels"), list) or not job["labels"] \
                or not all(isinstance(label, str) and label for label in job["labels"]):
            raise ValueError("confirmation runner API binding")
        artifacts = artifacts_payload.get("artifacts") if isinstance(artifacts_payload, dict) else None
        artifact_name = f"phase07-confirmation-{run_id}-{run_attempt}"
        artifact_matches = [artifact for artifact in artifacts if isinstance(artifact, dict)
                            and artifact.get("name") == artifact_name] if isinstance(artifacts, list) else []
        if len(artifact_matches) != 1:
            raise ValueError("confirmation artifact API binding")
        artifact = artifact_matches[0]
        workflow_run = artifact.get("workflow_run")
        api_digest = artifact.get("digest")
        local_archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        if not isinstance(artifact.get("id"), int) or artifact["id"] <= 0 or artifact.get("expired") is not False \
                or not isinstance(api_digest, str) or api_digest != f"sha256:{local_archive_sha256}" \
                or not isinstance(artifact.get("expires_at"), str) \
                or not isinstance(workflow_run, dict) or workflow_run.get("id") != run_id \
                or workflow_run.get("head_branch") != run["head_branch"] or workflow_run.get("head_sha") != head_sha:
            raise ValueError("confirmation artifact API identity/digest")
        host = packet.get("locked_execution", {}).get("host", {})
        if not isinstance(host, dict) or not all(isinstance(host.get(name), str) and host[name]
                                                for name in ("os", "image", "architecture")):
            raise ValueError("confirmation locked host provenance")
        provenance = {
            "run_id": run_id, "run_attempt": run_attempt, "job_id": job["id"],
            "job_key": allocation["job_key"], "job_name": job_name,
            "artifact_id": artifact["id"], "artifact_name": artifact_name,
            "status": run["status"], "conclusion": run["conclusion"],
            "head_branch": run["head_branch"], "head_sha": head_sha, "event": run["event"],
            "runner": {"name": job["runner_name"], "group": job["runner_group_name"],
                       "labels": job["labels"], "os": host["os"], "image": host["image"],
                       "architecture": host["architecture"]},
            "run_created_at": run["created_at"], "artifact_expires_at": artifact["expires_at"],
            "api_archive_sha256": local_archive_sha256, "local_archive_sha256": local_archive_sha256,
            "archive": str(archive.resolve()), "extracted_dir": str(extracted.resolve()),
        }
        _validate_confirmation_provenance(provenance)
        records.append(provenance)

    provenance_dir.mkdir(parents=True, exist_ok=True)
    evidence = []
    for provenance in records:
        document = {"schema_version": 1, "evidence": [provenance]}
        path = provenance_dir / f"confirmation-{provenance['run_id']}-{provenance['run_attempt']}-provenance.json"
        _write_ledger(path, document)
        evidence.append({"artifact_dir": provenance["extracted_dir"], "provenance": str(path.resolve())})
    _write_ledger(output, {"schema_version": 1, "evidence": evidence})
    return 0


def run_preflight(request_file: Path, ledger_file: Path) -> int:
    request = _read_object(request_file)
    _reject_secrets(request)
    if request.get("ledger_path") != str(ledger_file.resolve()):
        raise ValueError("request must bind the exact ledger output path")
    evidence = validate_feature_worktree_preflight(request)
    record = {"schema_version": 1, "mode": "preflight", "request": request, "branch_head_status": evidence, "asvs_l1": {"input_validation": "pass", "access_control": "pending-hosted-proof", "secret_handling": "pass", "artifact_trust_boundary": "pending-hosted-proof"}}
    _write_ledger(ledger_file, record)
    return 0


def run_campaign(request_file: Path, ledger_file: Path) -> int:
    request = _read_object(request_file)
    _reject_secrets(request)
    evidence = validate_feature_worktree_preflight(request)
    packet = request.get("evidence_packet")
    if not isinstance(packet, dict):
        raise ValueError("campaign requires one evidence_packet object")
    validate_phase07_evidence_packet(packet)
    _write_ledger(ledger_file, {"schema_version": 1, "mode": "campaign", "branch_head_status": evidence, "evidence_packet": packet})
    return 0


def run_decision(request_file: Path, ledger_file: Path) -> int:
    return run_campaign(request_file, ledger_file)


def run_pr_gates(request_file: Path, ledger_file: Path) -> int:
    return run_campaign(request_file, ledger_file)


def run_confirmation_plan(*, stage1_ledger: Path, request_file: Path, workflow_inputs_dir: Path,
                          preflight_request: Path) -> int:
    """Materialize the only three dispatchable ordinal records from the exact head."""
    root = Path(_git("rev-parse", "--show-toplevel"))
    head = _git("rev-parse", "HEAD")
    plan = build_confirmation_plan(stage1_ledger, post_task0_head=head)
    validate_confirmation_plan(plan)
    request_file.parent.mkdir(parents=True, exist_ok=True)
    request_file.write_text(json.dumps(plan["confirmation_request"], sort_keys=True, indent=2) + "\n", encoding="utf-8")
    workflow_inputs_dir.mkdir(parents=True, exist_ok=True)
    for record in plan["workflow_inputs"]:
        slot = record["slot"]
        (workflow_inputs_dir / f"confirmation-ordinal{slot['ordinal']}.json").write_text(
            json.dumps({"confirmation_request": plan["confirmation_request"], "workflow_input": record,
                        "replacement_for_run_id": None}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    ledger_name = preflight_request.name.replace("-request.json", "-ledger.json")
    if ledger_name == preflight_request.name:
        ledger_name = f"{preflight_request.stem}-ledger{preflight_request.suffix}"
    preflight = {
        "repository": _read_object(stage1_ledger)["repository"], "branch": _git("branch", "--show-current"),
        "worktree_root": str(root), "head_sha": head, "allowed_dirty_paths": [],
        "workflow_name": "eval", "campaign_stage": "confirmation", "continuation_binding": "",
        "require_upstream_head": True, "ledger_path": str(preflight_request.with_name(ledger_name).resolve()),
    }
    preflight_request.parent.mkdir(parents=True, exist_ok=True)
    preflight_request.write_text(json.dumps(preflight, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None, *, github_client: Any | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "campaign", "decision", "pr-gates", "hosted-preflight", "finalize", "reconcile-hosted", "confirmation-allocation", "confirmation-plan", "confirmation-provenance"))
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--ledger-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--continuation", default="")
    parser.add_argument("--repository")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--run-attempt", type=int)
    parser.add_argument("--head-sha")
    parser.add_argument("--job-key")
    parser.add_argument("--runner-os", default="")
    parser.add_argument("--runner-architecture", default="")
    parser.add_argument("--job-status", default="unknown")
    parser.add_argument("--stage1-ledger", type=Path)
    parser.add_argument("--workflow-inputs-dir", type=Path)
    parser.add_argument("--preflight-request", type=Path)
    parser.add_argument("--provenance-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "hosted-preflight":
            if args.ledger_file is None:
                raise ValueError("hosted preflight requires --ledger-file")
            return seal_hosted_preflight(output=args.ledger_file, stage=args.stage or "", continuation=args.continuation,
                repository=args.repository or "", run_id=args.run_id or 0, run_attempt=args.run_attempt or 0,
                head_sha=args.head_sha or "", job_key=args.job_key or "", runner_os=args.runner_os,
                runner_architecture=args.runner_architecture)
        if args.command == "finalize":
            return finalize_pipeline_artifact(output_dir=args.output_dir or Path("."), stage=args.stage or "",
                head_sha=args.head_sha or "", run_id=args.run_id or 0, run_attempt=args.run_attempt or 0,
                job_key=args.job_key or "", job_status=args.job_status)
        if args.command == "reconcile-hosted":
            if args.request_file is None or args.ledger_file is None:
                raise ValueError("reconcile-hosted requires binding and output")
            return reconcile_hosted(args.request_file, args.ledger_file)
        if args.command == "confirmation-allocation":
            if args.request_file is None or args.ledger_file is None:
                raise ValueError("confirmation allocation requires sealed input and output")
            token = os.environ.get("GITHUB_TOKEN", "")
            return seal_confirmation_allocation(workflow_inputs=_read_object(args.request_file), output=args.ledger_file,
                                                repository=args.repository or "", run_id=args.run_id or 0,
                                                run_attempt=args.run_attempt or 0, job_key=args.job_key or "", head_sha=args.head_sha or "", token=token)
        if args.command == "confirmation-plan":
            if None in (args.stage1_ledger, args.request_file, args.workflow_inputs_dir, args.preflight_request):
                raise ValueError("confirmation-plan requires immutable ledger and all generated output paths")
            return run_confirmation_plan(stage1_ledger=args.stage1_ledger, request_file=args.request_file,
                                         workflow_inputs_dir=args.workflow_inputs_dir, preflight_request=args.preflight_request)
        if args.command == "confirmation-provenance":
            if None in (args.request_file, args.ledger_file, args.provenance_dir):
                raise ValueError("confirmation provenance requires request, manifest, and provenance directory")
            return collect_confirmation_provenance(
                request_file=args.request_file, output=args.ledger_file,
                provenance_dir=args.provenance_dir, token=os.environ.get("GITHUB_TOKEN", ""),
                client=github_client,
            )
        if args.request_file is None or args.ledger_file is None:
            raise ValueError("operator command requires request and ledger files")
        return {"preflight": run_preflight, "campaign": run_campaign, "decision": run_decision, "pr-gates": run_pr_gates}[args.command](args.request_file, args.ledger_file)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"[FAIL] Phase 07 operator gate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
