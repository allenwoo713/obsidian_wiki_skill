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
import subprocess
import sys
from pathlib import Path
from typing import Any


SHA = re.compile(r"^[0-9a-f]{40}$")
STAGES = frozenset({"preflight", "screening", "confirmation", "continuation", "pr-acceptance"})
INFRA_FAILURES = frozenset({"github_infrastructure", "hosted_runner_unavailable", "artifact_service"})
SECRET_MARKERS = ("token", "secret", "password", "authorization", "private_key", "ghp_")


def canonical_digest(payload: dict[str, Any]) -> str:
    """Digest a record without its recursive self-digest field."""
    value = dict(payload)
    value.pop("record_self_sha256", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()


def validate_feature_worktree_preflight(request: dict[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    """Validate exact feature worktree identity and its explicit dirty policy."""
    allowed = {"repository", "branch", "worktree_root", "head_sha", "allowed_dirty_paths", "workflow_name", "campaign_stage", "continuation_binding", "require_upstream_head"}
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
    required = {"repository", "run_id", "run_attempt", "job_id", "job_allocation_nonce", "artifact_id", "archive_sha256", "record_self_sha256", "retention_days", "head_sha", "runner", "lock_identity", "retry_lineage"}
    if set(packet) != required:
        raise ValueError("untrusted artifact packet fields")
    _reject_secrets(packet)
    if not isinstance(packet["repository"], str) or not packet["repository"]:
        raise ValueError("repository binding")
    for key in ("run_id", "run_attempt", "job_id", "artifact_id", "retention_days"):
        if not isinstance(packet[key], int) or packet[key] <= 0:
            raise ValueError(f"invalid {key}")
    if not isinstance(packet["job_allocation_nonce"], str) or len(packet["job_allocation_nonce"]) < 16:
        raise ValueError("job allocation nonce")
    if not SHA.fullmatch(packet["head_sha"]) or not isinstance(packet["archive_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", packet["archive_sha256"]):
        raise ValueError("immutable digest or head binding")
    if packet["retention_days"] != 90:
        raise ValueError("90-day retention is mandatory")
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


def _write_ledger(path: Path, record: dict[str, Any]) -> None:
    record["record_self_sha256"] = canonical_digest(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def run_preflight(request_file: Path, ledger_file: Path) -> int:
    request = _read_object(request_file)
    _reject_secrets(request)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "campaign", "decision", "pr-gates"))
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--ledger-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        return {"preflight": run_preflight, "campaign": run_campaign, "decision": run_decision, "pr-gates": run_pr_gates}[args.command](args.request_file, args.ledger_file)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"[FAIL] Phase 07 operator gate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
