"""Fail-closed PR reconciliation for held-out ANN decision artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import re
from pathlib import Path

from benchmark_ann_build import validate_evidence
from run_eval import validate_candidate_decision_records


PHASE07_PACKET_FIELDS = frozenset({
    "repository", "run_id", "run_attempt", "job_id", "job_allocation_nonce",
    "artifact_id", "artifact_name", "archive_sha256", "content_sha256",
    "record_self_sha256", "retention_days_requested", "retention_days_accepted",
    "head_sha", "runner", "lock_identity", "build_id", "retry_lineage",
})
PHASE07_INFRA_FAILURES = frozenset({
    "github_infrastructure", "hosted_runner_unavailable", "artifact_service",
})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SECRET_MARKERS = ("token", "secret", "password", "authorization", "private_key", "ghp_")


def _reject_secrets(value, location="packet") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if any(marker in str(key).lower() for marker in _SECRET_MARKERS):
                raise ValueError(f"secret-like packet field: {location}.{key}")
            _reject_secrets(item, f"{location}.{key}")
    elif isinstance(value, list):
        for item in value:
            _reject_secrets(item, location)
    elif isinstance(value, str) and any(marker in value.lower() for marker in ("ghp_", "github_pat_", "bearer ")):
        raise ValueError("secret-like packet value")


ARCHITECTURE_REQUIRED_CHECKS = (
    "Architecture (ubuntu-latest, Python 3.10)",
    "Architecture (ubuntu-latest, Python 3.13)",
    "Architecture (windows-latest, Python 3.10)",
    "Architecture (windows-latest, Python 3.13)",
)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"artifact unavailable or invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_feature_worktree_preflight(record: dict) -> dict:
    """Reject an operator ledger that does not seal a non-default feature checkout."""
    if not isinstance(record, dict):
        raise ValueError("feature worktree preflight must be an object")
    branch_state = record.get("branch_head_status")
    if not isinstance(branch_state, dict):
        raise ValueError("missing branch/head/status evidence")
    branch = branch_state.get("branch")
    head = branch_state.get("head_sha")
    root = branch_state.get("worktree_root")
    status = branch_state.get("status")
    if not isinstance(branch, str) or branch in {"master", "main"}:
        raise ValueError("unsafe feature branch evidence")
    if not isinstance(head, str) or len(head) != 40:
        raise ValueError("invalid feature head evidence")
    if not isinstance(root, str) or not root:
        raise ValueError("missing worktree root evidence")
    if not isinstance(status, list) or not all(isinstance(item, str) for item in status):
        raise ValueError("invalid worktree status evidence")
    return branch_state


def validate_phase07_evidence_packet(packet: dict) -> dict:
    """Validate remote artifact identity, retention, digest, and retry lineage."""
    if not isinstance(packet, dict) or set(packet) != PHASE07_PACKET_FIELDS:
        raise ValueError("incomplete or unknown Phase 07 artifact fields")
    _reject_secrets(packet)
    for key in ("run_id", "run_attempt", "job_id", "artifact_id", "retention_days_requested", "retention_days_accepted"):
        if not isinstance(packet[key], int) or packet[key] <= 0:
            raise ValueError(f"invalid {key}")
    if packet["retention_days_requested"] != 90 or packet["retention_days_accepted"] != 90:
        raise ValueError("requested and accepted retention must be exactly 90 days")
    if not all(isinstance(packet[key], str) and _HEX64.fullmatch(packet[key]) for key in ("archive_sha256", "content_sha256", "record_self_sha256", "build_id")):
        raise ValueError("invalid artifact digest")
    if not isinstance(packet["head_sha"], str) or len(packet["head_sha"]) != 40:
        raise ValueError("invalid artifact head")
    if not isinstance(packet["job_allocation_nonce"], str) or len(packet["job_allocation_nonce"]) < 16:
        raise ValueError("invalid job allocation nonce")
    runner = packet["runner"]
    if not isinstance(runner, dict) or not {"os", "image", "architecture"} <= set(runner):
        raise ValueError("incomplete runner variance metadata")
    retry = packet["retry_lineage"]
    if not isinstance(retry, dict) or set(retry) != {"failure_class", "original_run_id", "replacement_run_id"}:
        raise ValueError("invalid retry lineage")
    if retry["replacement_run_id"] is not None and retry["failure_class"] not in PHASE07_INFRA_FAILURES:
        raise ValueError("non-infrastructure failures cannot be retried")
    expected = dict(packet)
    expected.pop("record_self_sha256")
    if packet["record_self_sha256"] != _canonical_sha256(expected):
        raise ValueError("artifact self-digest mismatch")
    return packet


def validate_phase07_evidence_set(packets: list[dict], *, expected_repository: str, expected_head: str) -> list[dict]:
    """Cross-run reconciliation rejects stale identity, allocation reuse, and fresh-build reuse."""
    allocations, builds = set(), set()
    for packet in packets:
        validate_phase07_evidence_packet(packet)
        if packet["repository"] != expected_repository or packet["head_sha"] != expected_head:
            raise ValueError("repository/head identity")
        allocation = tuple(packet[key] for key in ("run_id", "run_attempt", "job_id", "job_allocation_nonce"))
        if allocation in allocations or packet["build_id"] in builds:
            raise ValueError("reused allocation or build identity")
        allocations.add(allocation); builds.add(packet["build_id"])
    return packets


def reconcile(
    *, scale_evidence: Path, model_records: Path, conclusions: dict[str, str],
    expected_head: str, expected_pr_head: str,
) -> dict:
    """Validate one same-head numeric-success evidence set before reporting it."""
    for name in ("test-and-eval", "scale", "model-backed"):
        if conclusions.get(name) != "success":
            raise ValueError(f"required job {name} != \"success\"")
    if not isinstance(expected_head, str) or len(expected_head) != 40:
        raise ValueError("expected head")
    if not isinstance(expected_pr_head, str) or len(expected_pr_head) != 40:
        raise ValueError("expected PR head")

    scale = validate_evidence(_read_json(scale_evidence))
    source = scale["source"]
    if source["head_sha"] != expected_head:
        raise ValueError("scale artifact head")
    model = validate_candidate_decision_records(_read_json(model_records), scale)
    if model.get("head_sha") != expected_head:
        raise ValueError("model artifact head")
    if model.get("actions_merge_checkout_sha") != expected_head:
        raise ValueError("model artifact Actions merge checkout")
    if model.get("pr_head_sha") != expected_pr_head:
        raise ValueError("model artifact PR head")

    return {
        "schema_version": 1,
        "all_required_jobs_numeric_success": True,
        "head_sha": expected_head,
        "actions_merge_checkout_sha": expected_head,
        "pr_head_sha": expected_pr_head,
        "comparator_sha256": _canonical_sha256(scale),
        "scale_lock_identity": source["lock_identity"],
        "scale_runtime_identity": scale["environment"]["runtime"],
        "job_conclusions": conclusions,
        "branch_protection_requirements": list(ARCHITECTURE_REQUIRED_CHECKS),
        "artifacts": {
            "scale_evidence": str(scale_evidence),
            "model_records": str(model_records),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-evidence", type=Path, required=True)
    parser.add_argument("--model-records", type=Path, required=True)
    parser.add_argument("--test-and-eval", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--model-backed", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-pr-head", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = reconcile(
            scale_evidence=args.scale_evidence,
            model_records=args.model_records,
            conclusions={
                "test-and-eval": args.test_and_eval,
                "scale": args.scale,
                "model-backed": args.model_backed,
            },
            expected_head=args.expected_head,
            expected_pr_head=args.expected_pr_head,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[FAIL] ANN reconciliation: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("[PASS] ANN reconciliation", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
