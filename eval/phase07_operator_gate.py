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
HEX64 = re.compile(r"^[0-9a-f]{64}$")


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
        if not all(value in lock for value in ("lancedb==0.34.0", "numpy==2.2.6", "pyarrow==25.0.0")) or set(manifest) != {"schema_version", "model_id", "revision", "runtime", "files", "record_self_sha256"}:
            raise ValueError("lock or model manifest syntax")
        if manifest["schema_version"] != 1 or manifest["model_id"] != "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" or manifest["runtime"] != {"python": "3.13", "scipy": "1.15.3", "lancedb": "0.34.0"}:
            raise ValueError("model manifest identity/runtime")
        if not SHA.fullmatch(manifest["revision"]) or manifest["revision"] == "0" * 40:
            raise ValueError("immutable provider revision")
        if manifest["record_self_sha256"] != canonical_digest(manifest):
            raise ValueError("model manifest self digest")
        if not isinstance(manifest["files"], list) or not manifest["files"]:
            raise ValueError("model manifest file allowlist")
        paths = set()
        for item in manifest["files"]:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"} or not isinstance(item["path"], str) or not item["path"] or item["path"].startswith("/") or "\\" in item["path"] or ":" in item["path"] or ".." in Path(item["path"]).parts or item["path"] in paths or not isinstance(item["sha256"], str) or not HEX64.fullmatch(item["sha256"]):
                raise ValueError("model manifest file record")
            paths.add(item["path"])
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
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*-result.json")) or any(output_dir.glob("*-rejection.json")):
        return 0
    _write_ledger(output_dir / f"{stage}-pipeline-rejection.json", {
        "schema_version": 1, "stage": stage, "status": "reject-evidence",
        "head_sha": head_sha, "run_id": run_id, "run_attempt": run_attempt,
        "job_key": job_key, "job_status": job_status, "authorization": "none",
    })
    return 0


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "campaign", "decision", "pr-gates", "hosted-preflight", "finalize", "reconcile-hosted"))
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
    args = parser.parse_args()
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
        if args.request_file is None or args.ledger_file is None:
            raise ValueError("operator command requires request and ledger files")
        return {"preflight": run_preflight, "campaign": run_campaign, "decision": run_decision, "pr-gates": run_pr_gates}[args.command](args.request_file, args.ledger_file)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"[FAIL] Phase 07 operator gate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
