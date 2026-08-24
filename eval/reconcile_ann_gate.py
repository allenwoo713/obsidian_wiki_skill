"""Fail-closed PR reconciliation for held-out ANN decision artifacts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import re
import stat
import zipfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    # Direct-file entry points start with ``eval/`` on sys.path, whereas
    # run_eval imports the repository ``eval`` package.
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval import benchmark_ann_build as benchmark
from eval.benchmark_ann_build import validate_evidence
from eval.run_eval import validate_candidate_decision_records
from eval.phase07_ann_campaign import (
    Phase07AnnCampaignRunner,
    canonical_digest as campaign_digest,
    select_stage1_nominees,
    validate_confirmation_execution,
)
from eval.ann_frontier_statistics import holm_adjust, paired_basic_effect, paired_permutation_p


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
_STAGE1_FILES = frozenset({
    "archive.zip", "extracted",
})
_STAGE1_EXTRACTED_FILES = frozenset({
    "screening-request.json", "screening-ledger.json", "screening-result.json",
})
_STAGE1_REQUEST_FIELDS = frozenset({
    "schema_version", "mode", "authorization", "repository", "branch", "workflow_path", "head_sha", "run_id", "run_attempt",
    "job_id", "job_key", "job_allocation_nonce", "runtime", "model_manifest_sha256",
    "corpus_manifest_sha256", "workflow_name", "event", "status", "conclusion", "runner",
    "lock_identity", "retry_lineage", "artifact", "campaign_result_sha256",
    "campaign_request_sha256", "campaign_ledger_sha256", "run_created_at",
    "api_provenance", "workflow_retention_days", "record_self_sha256",
})
_STAGE1_API_PROVENANCE_FIELDS = frozenset({"workflow_run", "job", "artifact"})
_STAGE1_API_WORKFLOW_RUN_FIELDS = frozenset({
    "run_id", "run_attempt", "head_branch", "head_sha", "event", "status",
    "conclusion", "created_at",
})
_STAGE1_API_JOB_FIELDS = frozenset({
    "job_id", "run_id", "name", "status", "conclusion", "runner_name",
    "runner_group_name", "labels",
})
_STAGE1_API_ARTIFACT_FIELDS = frozenset({
    "artifact_id", "job_id", "job_key", "run_id", "name", "created_at",
    "expires_at", "expired",
})


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


def canonical_digest(payload: dict) -> str:
    """Public canonical digest for newly sealed confirmation packets."""
    value = dict(payload)
    value.pop("record_self_sha256", None)
    return _canonical_sha256(value)


def validate_confirmation_packet(packet: dict, workflow_inputs: dict) -> dict:
    """Validate one non-pooled physical confirmation packet before reconciliation."""
    required = {
        "schema_version", "campaign_stage", "workflow_inputs_sha256", "slot", "run_id", "run_attempt",
        "job_id", "job_key", "job_allocation_nonce", "status", "failure_class", "replacement_for_run_id",
        "builds", "d04", "d20", "raw_tree_sha256", "retention_days", "record_self_sha256",
    }
    extended = {"locked_execution", "measurements"}
    if not isinstance(packet, dict) or set(packet) not in (required, required | extended) \
            or packet.get("record_self_sha256") != canonical_digest(packet):
        raise ValueError("sealed confirmation packet")
    if packet["campaign_stage"] != "confirmation" or packet["workflow_inputs_sha256"] != workflow_inputs.get("record_self_sha256") or packet["slot"] != workflow_inputs.get("slot"):
        raise ValueError("confirmation input/request binding")
    if not all(isinstance(packet[key], int) and packet[key] > 0 for key in ("run_id", "run_attempt", "job_id")) or not isinstance(packet["job_key"], str) or not packet["job_key"] or not isinstance(packet["job_allocation_nonce"], str) or len(packet["job_allocation_nonce"]) < 32:
        raise ValueError("confirmation allocation identity")
    if packet["retention_days"] != 90 or not isinstance(packet["raw_tree_sha256"], str) or not _HEX64.fullmatch(packet["raw_tree_sha256"]):
        raise ValueError("confirmation artifact retention/digest")
    failure = packet["failure_class"]
    if packet["status"] == "numeric-success":
        if failure is not None:
            raise ValueError("numeric success cannot carry a failure")
    elif failure not in PHASE07_INFRA_FAILURES | {"numeric", "recall", "hybrid", "watchdog", "malformed", "reconciliation"}:
        raise ValueError("typed confirmation failure class")
    builds = packet["builds"]
    if not isinstance(builds, list) or len(builds) != 3:
        raise ValueError("confirmation requires exactly three fresh builds")
    by_m = {}
    build_ids = set()
    for build in builds:
        if not isinstance(build, dict) or set(build) != {"build_id", "m", "ef_construction", "query_ef"} or build.get("m") not in {16, 20, 32} or build.get("ef_construction") != 300 or not isinstance(build.get("build_id"), str) or not _HEX64.fullmatch(build["build_id"]) or build["build_id"] in build_ids:
            raise ValueError("fresh HNSW-SQ build identity")
        if build["m"] in by_m or not isinstance(build["query_ef"], list):
            raise ValueError("duplicate build m")
        by_m[build["m"]] = build; build_ids.add(build["build_id"])
    if set(by_m) != {16, 20, 32} or by_m[16]["query_ef"] != [100, 200, 300] or any(by_m[m]["query_ef"] != [200, 300] for m in (20, 32)):
        raise ValueError("D-04/D-20 build query allocation")
    for name, size in (("d04", 6), ("d20", 4)):
        family = packet[name]
        member_d20 = name == "d20" and isinstance(family, dict) and family.get("family_name") == "d20_current_baseline_member" and family.get("family_size") == 2
        if not isinstance(family, dict) or (not member_d20 and (family.get("family_size") != size or not isinstance(family.get("holm_adjusted_p_values"), list) or len(family["holm_adjusted_p_values"]) != size)) or not isinstance(family.get("raw_p_values"), list) or len(family["raw_p_values"]) != (2 if member_d20 else size) or not isinstance(family.get("basic_ci_95"), list) or len(family["basic_ci_95"]) != (2 if member_d20 else size):
            raise ValueError("separate confirmation statistical family")
    if packet["d04"].get("family_name") != "d04_ef_300_vs_200" or packet["d20"].get("family_name") not in {"d20_current_baseline", "d20_current_baseline_member"} or packet["d20"].get("baseline_build_id") != by_m[16]["build_id"]:
        raise ValueError("D-20 must reference this packet m=16 baseline")
    for family in (packet["d04"], packet["d20"]):
        for value in [*family["raw_p_values"], *(family.get("holm_adjusted_p_values") or []), *(item for interval in family["basic_ci_95"] for item in interval)]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("non-finite confirmation statistic")
        comparisons = family.get("comparisons")
        if not isinstance(comparisons, list) or len(comparisons) != family["family_size"]:
            raise ValueError("missing numeric confirmation family members")
        canonical_keys = set()
        for comparison in comparisons:
            rows = comparison.get("paired_rows") if isinstance(comparison, dict) else None
            if not isinstance(rows, list) or not rows:
                raise ValueError("missing paired confirmation samples")
            for row in rows:
                if not isinstance(row, list) or len(row) != 2 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in row):
                    raise ValueError("invalid paired confirmation samples")
        if any(comparison.get("raw_permutation_p") != declared for comparison, declared in zip(comparisons, family["raw_p_values"], strict=True)) or any(comparison.get("basic_ci_95") != declared for comparison, declared in zip(comparisons, family["basic_ci_95"], strict=True)):
            raise ValueError("declared confirmation statistic mismatch")
        if family.get("holm_adjusted_p_values") is not None and family["holm_adjusted_p_values"] != holm_adjust(family["raw_p_values"]):
            raise ValueError("declared confirmation Holm mismatch")
        for comparison in comparisons:
            declared = comparison.get("comparison")
            if not isinstance(declared, dict):
                raise ValueError("missing confirmation comparison binding")
            if family["family_name"] == "d04_ef_300_vs_200":
                key = (declared.get("m"), declared.get("metric"), declared.get("baseline_ef"), declared.get("candidate_ef"))
                if key not in {(m, metric, 200, 300) for m in (16, 20, 32) for metric in ("recall_at_10", "recall_at_20")}:
                    raise ValueError("noncanonical D-04 member")
            else:
                key = (declared.get("candidate_m"), declared.get("metric"), declared.get("baseline_ef"), declared.get("candidate_ef"))
                if key not in {(workflow_inputs["slot"]["m"], metric, 100, 300) for metric in ("recall_at_10", "recall_at_20")} or declared.get("baseline_build_id") != by_m[16]["build_id"] or declared.get("candidate_build_id") != by_m[workflow_inputs["slot"]["m"]]["build_id"]:
                    raise ValueError("D-20 comparison cross-build identity")
            if key in canonical_keys:
                raise ValueError("duplicate canonical confirmation member")
            canonical_keys.add(key)
            expected_effect = paired_basic_effect(comparison["paired_rows"], comparison=declared)
            expected_p = paired_permutation_p(comparison["paired_rows"], comparison=declared)
            if comparison.get("mean_effect") != expected_effect["mean_effect"] or comparison.get("basic_ci_95") != expected_effect["basic_ci_95"] or comparison.get("raw_permutation_p") != expected_p:
                raise ValueError("recomputed confirmation statistic mismatch")
    if extended <= set(packet):
        validate_confirmation_execution(packet["locked_execution"])
        measurements = packet["measurements"]
        raw_builds = measurements.get("builds") if isinstance(measurements, dict) else None
        expected_builds = [
            {"build_id": build.get("build_id"), "m": build.get("build", {}).get("m"),
             "ef_construction": build.get("build", {}).get("ef_construction"),
             "query_ef": [group.get("query_ef") for group in build.get("queries", [])]}
            for build in raw_builds
        ] if isinstance(raw_builds, list) else None
        if not isinstance(measurements, dict) or set(measurements) != {"builds", "d04_statistics", "d20_member_statistics"} \
                or expected_builds != packet["builds"] \
                or measurements["d04_statistics"] != {"schema_version": 1, "family_name": packet["d04"]["family_name"], "family_size": packet["d04"]["family_size"], "comparisons": packet["d04"]["comparisons"], "authorization": "none", "record_self_sha256": campaign_digest({"schema_version": 1, "family_name": packet["d04"]["family_name"], "family_size": packet["d04"]["family_size"], "comparisons": packet["d04"]["comparisons"], "authorization": "none"})}:
            raise ValueError("confirmation raw measurement binding")
        d20 = measurements["d20_member_statistics"]
        if not isinstance(d20, dict) or d20.get("family_name") != packet["d20"]["family_name"] \
                or d20.get("comparisons") != packet["d20"]["comparisons"] \
                or d20.get("authorization") != "none":
            raise ValueError("confirmation D-20 raw measurement binding")
    return packet


def reconcile_confirmation(plan: dict, packets: list[dict]) -> dict:
    """Seal six logical slots while retaining, but excluding, typed infra origins."""
    from eval.phase07_operator_gate import validate_confirmation_plan
    validate_confirmation_plan(plan)
    if not isinstance(packets, list) or not packets:
        raise ValueError("confirmation packets required")
    inputs = {record["record_self_sha256"]: record for record in plan["workflow_inputs"]}
    physical = []
    per_slot: dict[str, list[dict]] = {key: [] for key in inputs}
    seen_runs = set()
    for packet in packets:
        key = packet.get("workflow_inputs_sha256") if isinstance(packet, dict) else None
        if key not in inputs:
            raise ValueError("hand-authored or unknown confirmation input")
        validate_confirmation_packet(packet, inputs[key])
        allocation = (packet["run_id"], packet["run_attempt"], packet["job_id"], packet["job_allocation_nonce"])
        if allocation in seen_runs:
            raise ValueError("replayed confirmation physical run")
        seen_runs.add(allocation); per_slot[key].append(packet)
    eligible = []
    for key, slot_packets in per_slot.items():
        success = [packet for packet in slot_packets if packet["status"] == "numeric-success"]
        rejected = [packet for packet in slot_packets if packet["status"] != "numeric-success"]
        if not rejected:
            if len(success) != 1:
                raise ValueError("each confirmation slot requires one eligible numeric result")
        else:
            if len(rejected) != 1 or rejected[0]["failure_class"] not in PHASE07_INFRA_FAILURES or len(success) != 1 or success[0]["replacement_for_run_id"] != rejected[0]["run_id"]:
                raise ValueError("non-infrastructure failure has no replacement or invalid retry lineage")
            physical.append({**rejected[0], "eligible": False})
        physical.append({**success[0], "eligible": True})
        eligible.append(success[0])
    if len(eligible) != 6 or len({record["workflow_inputs_sha256"] for record in eligible}) != 6:
        raise ValueError("exact six eligible confirmation records")
    ordinal_families = []
    for ordinal in (1, 2, 3):
        pair = [record for record in eligible if record["slot"]["ordinal"] == ordinal]
        if {record["slot"]["m"] for record in pair} != {20, 32}:
            raise ValueError("D-20 requires both primary m values per ordinal")
        members = [comparison for record in pair for comparison in record["d20"].get("comparisons", [])]
        if members:
            if len(members) != 4:
                raise ValueError("D-20 ordinal family must contain four paired members")
            raw = [member["raw_permutation_p"] for member in members]
            ordinal_families.append({"ordinal": ordinal, "family_name": "d20_current_baseline", "family_size": 4, "comparisons": members, "raw_p_values": raw, "holm_adjusted_p_values": holm_adjust(raw), "basic_ci_95": [member["basic_ci_95"] for member in members], "baseline_build_ids": [record["d20"]["baseline_build_id"] for record in pair]})
    ledger = {"schema_version": 1, "campaign_stage": "confirmation", "confirmation_plan_sha256": plan["record_self_sha256"], "eligible_evidence_runs": eligible, "all_physical_workflow_runs": physical, "d20_ordinal_families": ordinal_families}
    ledger["record_self_sha256"] = canonical_digest(ledger)
    return ledger


def reconcile_confirmation_request(request: dict, source: dict) -> dict:
    """Consume downloaded packet wrappers and recompute the six-slot cross-run ledger."""
    from eval.phase07_operator_gate import canonical_digest as operator_digest, validate_confirmation_dispatch_bundle
    if not isinstance(request, dict) or request.get("record_self_sha256") != operator_digest(request):
        raise ValueError("sealed confirmation request")
    wrappers = source.get("packets") if isinstance(source, dict) else None
    if not isinstance(wrappers, list) or not wrappers:
        raise ValueError("downloaded confirmation packets required")
    inputs, packets = [], []
    for wrapper in wrappers:
        if not isinstance(wrapper, dict) or set(wrapper) != {"dispatch_bundle", "packet"}:
            raise ValueError("downloaded packet wrapper schema")
        bundle = wrapper["dispatch_bundle"]
        record = validate_confirmation_dispatch_bundle(bundle, expected_head=request.get("post_task0_head", ""))
        if bundle["confirmation_request"] != request:
            raise ValueError("cross-request confirmation replay")
        inputs.append(record); packets.append(wrapper["packet"])
    if len({record["record_self_sha256"] for record in inputs}) != 6:
        raise ValueError("exact six unique downloaded confirmation inputs")
    unique_inputs = {record["record_self_sha256"]: record for record in inputs}
    plan = {"schema_version": 1, "confirmation_request": request, "workflow_inputs": sorted(unique_inputs.values(), key=lambda row: (-row["slot"]["m"], row["slot"]["ordinal"])),
            "artifact_reported_nominated_m": request["artifact_reported_nominated_m"], "authoritative_nominated_m": request["authoritative_nominated_m"]}
    plan["record_self_sha256"] = operator_digest(plan)
    return reconcile_confirmation(plan, packets)


_CONFIRMATION_RAW_FILES = frozenset({
    "confirmation-request.json", "confirmation-ledger.json", "confirmation-result.json",
    "dispatch-bundle.json", "allocation.json",
})
_CONFIRMATION_ARTIFACT_FILES = _CONFIRMATION_RAW_FILES | {"confirmation-packet.json"}
_CONFIRMATION_PROVENANCE_FIELDS = frozenset({
    "run_id", "run_attempt", "job_id", "artifact_id", "artifact_name", "status", "conclusion",
    "runner", "run_created_at", "artifact_expires_at", "api_archive_sha256", "local_archive_sha256",
    "archive", "extracted_dir",
})


def _confirmation_tree_sha256(root: Path) -> str:
    """Compatibility facade for test tooling; production validation is shared."""
    from eval.phase07_ann_campaign import confirmation_raw_tree_sha256
    return confirmation_raw_tree_sha256(root, require_wrapper=True)


def _confirmation_timestamp(value: Any, label: str) -> dt.datetime:
    parsed = _stage1_timestamp(value, label=f"confirmation {label}")
    return parsed


def _validate_confirmation_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CONFIRMATION_PROVENANCE_FIELDS:
        raise ValueError("strict confirmation API provenance schema")
    if not all(isinstance(value[name], int) and value[name] > 0 for name in ("run_id", "run_attempt", "job_id", "artifact_id")) \
            or not isinstance(value["artifact_name"], str) or not value["artifact_name"] \
            or value["status"] != "completed" or value["conclusion"] != "success":
        raise ValueError("confirmation API run/job/artifact status")
    if value["artifact_name"] != f"phase07-confirmation-{value['run_id']}-{value['run_attempt']}":
        raise ValueError("confirmation artifact name/run-attempt binding")
    runner = value["runner"]
    if not isinstance(runner, dict) or set(runner) != {"name", "group", "labels", "os", "image", "architecture"} \
            or not isinstance(runner["name"], str) or not runner["name"] \
            or not isinstance(runner["group"], str) or not runner["group"] \
            or not isinstance(runner["labels"], list) or not runner["labels"] \
            or runner["os"] != "Linux" or runner["architecture"] != "X64" \
            or not isinstance(runner["image"], str) or not runner["image"]:
        raise ValueError("confirmation runner identity")
    created, expires = _confirmation_timestamp(value["run_created_at"], "created"), _confirmation_timestamp(value["artifact_expires_at"], "expiry")
    retention = expires - created
    if not dt.timedelta(days=89, hours=23, minutes=59, seconds=30) <= retention <= dt.timedelta(days=90, seconds=30):
        raise ValueError("confirmation artifact retention interval")
    for name in ("api_archive_sha256", "local_archive_sha256"):
        if not isinstance(value[name], str) or not _HEX64.fullmatch(value[name]):
            raise ValueError("confirmation archive digest schema")
    archive, extracted = Path(value["archive"]), Path(value["extracted_dir"])
    if archive.is_symlink() or not archive.is_file() or extracted.is_symlink() or not extracted.is_dir():
        raise ValueError("confirmation downloaded archive/extraction unavailable")
    local = hashlib.sha256(archive.read_bytes()).hexdigest()
    if value["api_archive_sha256"] != local or value["local_archive_sha256"] != local:
        raise ValueError("confirmation API/local archive digest mismatch")
    try:
        with zipfile.ZipFile(archive) as compressed:
            members = compressed.infolist()
            names = [member.filename for member in members]
            if len(names) != len(_CONFIRMATION_ARTIFACT_FILES) \
                    or len(set(names)) != len(names) \
                    or set(names) != _CONFIRMATION_ARTIFACT_FILES \
                    or any(member.is_dir() or stat.S_ISLNK(member.external_attr >> 16) for member in members):
                raise ValueError("strict confirmation archive allowlist")
            for name in _CONFIRMATION_ARTIFACT_FILES:
                extracted_member = extracted / name
                if extracted_member.is_symlink() or not extracted_member.is_file() \
                        or compressed.read(name) != extracted_member.read_bytes():
                    raise ValueError("confirmation archive/extracted content mismatch")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("invalid confirmation downloaded archive") from exc
    return value


def _read_confirmation_artifact(root: Path, provenance: dict[str, Any], request: dict[str, Any]) -> tuple[dict, dict, dict]:
    """Use the shared artifact validator, then attach immutable API provenance."""
    from eval.phase07_ann_campaign import validate_confirmation_artifact_tree

    validated = validate_confirmation_artifact_tree(root, expected_head=request.get("post_task0_head", ""))
    generated, packet = validated["workflow_input"], validated["packet"]
    identity = validated["allocation"]["allocation"]
    if any(identity[name] != provenance[name] for name in ("run_id", "run_attempt", "job_id")) \
            or packet["run_id"] != provenance["run_id"] or packet["run_attempt"] != provenance["run_attempt"] \
            or packet["job_id"] != provenance["job_id"] or packet["job_allocation_nonce"] != identity["job_allocation_nonce"]:
        raise ValueError("confirmation packet/API allocation identity mismatch")
    host = packet["locked_execution"]["host"]
    if {name: provenance["runner"][name] for name in ("os", "image", "architecture")} != {
        name: host[name] for name in ("os", "image", "architecture")
    }:
        raise ValueError("confirmation runner/locked-host provenance mismatch")
    validated_provenance = {
        **provenance,
        "content_sha256": validated["content_tree_sha256"],
        "raw_tree_sha256": validated["raw_tree_sha256"],
        "packet_self_sha256": packet["record_self_sha256"],
        "wrapper_self_sha256": validated["wrapper"]["record_self_sha256"],
        "raw_result_sha256": validated["raw_file_sha256"]["confirmation-result.json"],
    }
    return generated, packet, {
        "validated_provenance": validated_provenance,
        "validated_measurements": validated["result"],
    }


def reconcile_confirmation_postdownload(request: dict, manifest: dict) -> dict:
    """Reconcile six downloaded, API-provenanced confirmation artifacts fail-closed."""
    from eval.phase07_operator_gate import canonical_digest as operator_digest

    if not isinstance(request, dict) or request.get("record_self_sha256") != operator_digest(request):
        raise ValueError("sealed confirmation request")
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "evidence", "record_self_sha256"} \
            or manifest.get("schema_version") != 1 or manifest.get("record_self_sha256") != canonical_digest(manifest) \
            or not isinstance(manifest.get("evidence"), list) or len(manifest["evidence"]) != 6:
        raise ValueError("strict six-record confirmation provenance manifest")
    inputs, packets, provenance_by_allocation, seen_archives = [], [], {}, set()
    artifact_ids, job_ids, run_identities = set(), set(), set()
    for row in manifest["evidence"]:
        if not isinstance(row, dict) or set(row) != {"artifact_dir", "provenance"}:
            raise ValueError("strict confirmation evidence record")
        provenance_document = _read_json(Path(row["provenance"]))
        if set(provenance_document) != {"schema_version", "evidence", "record_self_sha256"} \
                or provenance_document.get("schema_version") != 1 \
                or provenance_document.get("record_self_sha256") != canonical_digest(provenance_document) \
                or not isinstance(provenance_document.get("evidence"), list) or len(provenance_document["evidence"]) != 1:
            raise ValueError("strict per-artifact API provenance document")
        provenance = _validate_confirmation_provenance(provenance_document["evidence"][0])
        if provenance["archive"] in seen_archives:
            raise ValueError("replayed confirmation archive")
        seen_archives.add(provenance["archive"])
        if Path(row["artifact_dir"]).resolve() != Path(provenance["extracted_dir"]).resolve():
            raise ValueError("manifest/extracted artifact mismatch")
        generated, packet, metadata = _read_confirmation_artifact(Path(row["artifact_dir"]), provenance, request)
        artifact_id, job_id = provenance["artifact_id"], provenance["job_id"]
        run_identity = (provenance["run_id"], provenance["run_attempt"])
        if artifact_id in artifact_ids or job_id in job_ids or run_identity in run_identities:
            raise ValueError("duplicate confirmation artifact/job/run provenance")
        artifact_ids.add(artifact_id); job_ids.add(job_id); run_identities.add(run_identity)
        allocation = tuple(packet[name] for name in ("run_id", "run_attempt", "job_id", "job_allocation_nonce"))
        inputs.append(generated); packets.append(packet); provenance_by_allocation[allocation] = metadata
    if len({item["record_self_sha256"] for item in inputs}) != 6:
        raise ValueError("duplicate confirmation input provenance")
    plan = {"schema_version": 1, "confirmation_request": request,
            "workflow_inputs": sorted(inputs, key=lambda row: (-row["slot"]["m"], row["slot"]["ordinal"])),
            "artifact_reported_nominated_m": request["artifact_reported_nominated_m"],
            "authoritative_nominated_m": request["authoritative_nominated_m"]}
    plan["record_self_sha256"] = operator_digest(plan)
    ledger = reconcile_confirmation(plan, packets)
    for record in [*ledger["eligible_evidence_runs"], *ledger["all_physical_workflow_runs"]]:
        allocation = tuple(record[name] for name in ("run_id", "run_attempt", "job_id", "job_allocation_nonce"))
        metadata = provenance_by_allocation.get(allocation)
        if metadata is None:
            raise ValueError("missing validated confirmation provenance")
        record.update(metadata)
    ledger["record_self_sha256"] = canonical_digest(ledger)
    return ledger


def _stage1_reject_secrets(value: Any, location: str = "request") -> None:
    """Allow the explicit no-authority marker, reject every other credential-shaped value."""
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered == "authorization" and item == "none":
                continue
            if any(marker in lowered for marker in _SECRET_MARKERS):
                raise ValueError(f"secret-like Stage 1 field: {location}.{key}")
            _stage1_reject_secrets(item, f"{location}.{key}")
    elif isinstance(value, list):
        for item in value:
            _stage1_reject_secrets(item, location)
    elif isinstance(value, str) and any(marker in value.lower() for marker in ("ghp_", "github_pat_", "bearer ")):
        raise ValueError("secret-like Stage 1 value")


def _stage1_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage1_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Stage 1 symlinked artifact content")
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        elif not path.is_dir():
            raise ValueError("unknown Stage 1 artifact entry")
    return digest.hexdigest()


def _stage1_read(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("record_self_sha256") != campaign_digest(payload):
        raise ValueError(f"Stage 1 artifact self-digest mismatch: {path.name}")
    return payload


def _stage1_stress_digest(exact_ids: list[list[str]]) -> str:
    return hashlib.sha256(json.dumps(exact_ids, separators=(",", ":")).encode()).hexdigest()


def _stage1_timestamp(value: Any, *, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Stage 1 {label} timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Stage 1 {label} timestamp timezone")
    return parsed.astimezone(dt.timezone.utc)


def _validate_stage1_workflow_retention(request: dict[str, Any]) -> None:
    """Tie the claimed API retention to the exact screening workflow contract."""
    if request["workflow_retention_days"] != 90:
        raise ValueError("Stage 1 workflow retention must be exactly 90 days")
    workflow = (REPOSITORY_ROOT / request["workflow_path"]).resolve()
    if workflow.parent != (REPOSITORY_ROOT / ".github" / "workflows").resolve():
        raise ValueError("Stage 1 workflow path containment")
    try:
        section = workflow.read_text(encoding="utf-8").split("  phase07-screening:", 1)[1].split(
            "  phase07-confirmation:", 1
        )[0]
    except (OSError, IndexError) as exc:
        raise ValueError("Stage 1 screening workflow unavailable") from exc
    if "retention-days: 90" not in section:
        raise ValueError("Stage 1 screening workflow retention declaration")


def _validate_stage1_runner(runner: Any) -> dict[str, Any]:
    if not isinstance(runner, dict) or set(runner) != {"name", "group", "labels", "os", "image", "architecture"}:
        raise ValueError("Stage 1 runner identity schema")
    if not isinstance(runner["name"], str) or not runner["name"].startswith("GitHub Actions ") \
            or runner["group"] != "GitHub Actions" or runner["labels"] != ["ubuntu-latest"] \
            or runner["os"] != "Linux" or runner["architecture"] != "X64" \
            or not isinstance(runner["image"], str) or not runner["image"]:
        raise ValueError("Stage 1 must use hosted ubuntu-latest Linux/X64")
    return runner


def _validate_stage1_api_provenance(request: dict[str, Any], runner: dict[str, Any]) -> dt.datetime:
    provenance = request["api_provenance"]
    if not isinstance(provenance, dict) or set(provenance) != _STAGE1_API_PROVENANCE_FIELDS:
        raise ValueError("strict Stage 1 API provenance schema")
    workflow_run, job, artifact = (
        provenance["workflow_run"], provenance["job"], provenance["artifact"],
    )
    if not isinstance(workflow_run, dict) or set(workflow_run) != _STAGE1_API_WORKFLOW_RUN_FIELDS \
            or not isinstance(job, dict) or set(job) != _STAGE1_API_JOB_FIELDS \
            or not isinstance(artifact, dict) or set(artifact) != _STAGE1_API_ARTIFACT_FIELDS:
        raise ValueError("strict Stage 1 API provenance members")
    if workflow_run != {
        "run_id": request["run_id"], "run_attempt": request["run_attempt"],
        "head_branch": request["branch"], "head_sha": request["head_sha"],
        "event": request["event"], "status": request["status"],
        "conclusion": request["conclusion"], "created_at": request["run_created_at"],
    }:
        raise ValueError("Stage 1 workflow API provenance binding")
    if job != {
        "job_id": request["job_id"], "run_id": request["run_id"],
        "name": "Phase 07 bounded SQ screening campaign", "status": "completed",
        "conclusion": "success", "runner_name": runner["name"],
        "runner_group_name": runner["group"], "labels": runner["labels"],
    }:
        raise ValueError("Stage 1 job API provenance binding")
    request_artifact = request["artifact"]
    if artifact != {
        "artifact_id": request_artifact["artifact_id"], "job_id": request["job_id"],
        "job_key": request["job_key"], "run_id": request["run_id"],
        "name": request_artifact["name"], "created_at": request_artifact["created_at"],
        "expires_at": request_artifact["expires_at"], "expired": request_artifact["expired"],
    }:
        raise ValueError("Stage 1 artifact API provenance binding")
    return _stage1_timestamp(workflow_run["created_at"], label="workflow-run creation")


def _validate_stage1_request(request: dict[str, Any], artifact_dir: Path) -> None:
    _stage1_reject_secrets(request)
    if set(request) != _STAGE1_REQUEST_FIELDS or request["schema_version"] != 1 or request["mode"] != "screening" or request["authorization"] != "none":
        raise ValueError("strict Stage 1 post-download request schema")
    if request["record_self_sha256"] != _canonical_sha256({key: value for key, value in request.items() if key != "record_self_sha256"}):
        raise ValueError("Stage 1 request self-digest mismatch")
    if not isinstance(request["repository"], str) or not request["repository"] or not isinstance(request["branch"], str) or request["branch"] in {"", "main", "master"} or request["workflow_path"] != ".github/workflows/eval.yml" or not isinstance(request["head_sha"], str) or not re.fullmatch(r"[0-9a-f]{40}", request["head_sha"]):
        raise ValueError("Stage 1 repository/head identity")
    if request["workflow_name"] != "eval.yml" or request["event"] != "workflow_dispatch" or request["status"] != "completed" or request["conclusion"] != "success":
        raise ValueError("Stage 1 run must be completed workflow-dispatch success")
    if not all(isinstance(request[name], int) and request[name] > 0 for name in ("run_id", "run_attempt", "job_id")):
        raise ValueError("Stage 1 run/attempt/job identity")
    if not isinstance(request["job_key"], str) or not request["job_key"] or request["job_allocation_nonce"] != f"{request['run_id']}-{request['run_attempt']}-{request['job_key']}":
        raise ValueError("Stage 1 job allocation binding")
    if request["runtime"] != {"python": "3.13", "lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0", "omp_num_threads": 2}:
        raise ValueError("Stage 1 locked runtime/OMP identity")
    runner = _validate_stage1_runner(request["runner"])
    run_created = _validate_stage1_api_provenance(request, runner)
    _validate_stage1_workflow_retention(request)
    if not all(isinstance(request[name], str) and _HEX64.fullmatch(request[name]) for name in ("model_manifest_sha256", "corpus_manifest_sha256", "lock_identity", "campaign_result_sha256", "campaign_request_sha256", "campaign_ledger_sha256")):
        raise ValueError("Stage 1 digest identity")
    retry = request["retry_lineage"]
    if not isinstance(retry, dict) or set(retry) != {"failure_class", "original_run_id", "replacement_run_id"} or retry != {"failure_class": None, "original_run_id": None, "replacement_run_id": None}:
        raise ValueError("Stage 1 retry lineage")
    artifact = request["artifact"]
    required_artifact = {"artifact_id", "name", "job_id", "job_key", "retention_days_requested", "retention_days_accepted", "created_at", "expires_at", "expired", "api_archive_sha256", "local_archive_path", "local_archive_sha256", "content_tree_sha256"}
    if not isinstance(artifact, dict) or set(artifact) != required_artifact or not isinstance(artifact["artifact_id"], int) or artifact["artifact_id"] <= 0 or artifact["job_id"] != request["job_id"] or artifact["job_key"] != request["job_key"] or not isinstance(artifact["name"], str) or not artifact["name"] or artifact["retention_days_requested"] != 90 or artifact["retention_days_accepted"] != 90 or artifact["expired"] is not False:
        raise ValueError("Stage 1 artifact identity/retention")
    created = _stage1_timestamp(artifact["created_at"], label="artifact creation")
    expires = _stage1_timestamp(artifact["expires_at"], label="artifact expiry")
    if created < run_created or not dt.timedelta(days=89, hours=23, minutes=59, seconds=30) <= expires - run_created <= dt.timedelta(days=90, seconds=30):
        raise ValueError("Stage 1 artifact retention interval")
    if not isinstance(artifact["local_archive_path"], str) or Path(artifact["local_archive_path"]).name != artifact["local_archive_path"]:
        raise ValueError("Stage 1 local archive path")
    archive = artifact_dir / artifact["local_archive_path"]
    if archive.is_symlink() or not archive.is_file() or _stage1_file_sha256(archive) != artifact["local_archive_sha256"] or artifact["api_archive_sha256"] != artifact["local_archive_sha256"]:
        raise ValueError("Stage 1 archive digest binding")
    extracted = artifact_dir / "extracted"
    if extracted.is_symlink() or not extracted.is_dir() or not all(isinstance(artifact[name], str) and _HEX64.fullmatch(artifact[name]) for name in ("api_archive_sha256", "local_archive_sha256", "content_tree_sha256")) or _stage1_tree_sha256(extracted) != artifact["content_tree_sha256"]:
        raise ValueError("Stage 1 content-tree digest binding")


def _validate_stage1_nominee_list(value: Any, *, label: str) -> list[int]:
    """Accept only the small, explicit Stage 1 nominee domain."""
    if not isinstance(value, list) or len(value) > 2 \
            or len(set(value)) != len(value) \
            or any(not isinstance(m, int) or isinstance(m, bool) or m not in {16, 20, 32} for m in value):
        raise ValueError(f"Stage 1 {label} nomination boundary")
    return list(value)


def _validate_stage1_result(result_record: dict[str, Any], request_record: dict[str, Any], *, expected_shape: tuple[int, int, int]) -> tuple[dict[str, Any], list[int]]:
    if set(result_record) != {"schema_version", "stage", "request_sha256", "result", "authorization", "record_self_sha256"} or result_record["schema_version"] != 1 or result_record["stage"] != "screening" or result_record["authorization"] != "none" or result_record["request_sha256"] != campaign_digest(request_record):
        raise ValueError("Stage 1 result/request binding")
    result = result_record["result"]
    if not isinstance(result, dict) or result.get("authorization") != "none" or result.get("build_count") != 3 or result.get("exact_truth_computed_once") is not True:
        raise ValueError("Stage 1 screening result shape")
    plan = result.get("plan")
    if not isinstance(plan, dict) or plan.get("index") != {"type": "hnsw_sq", "m": [16, 20, 32], "ef_construction": 300} or plan.get("query_ef") != [100, 150, 200, 300] or plan.get("per_build_max_seconds") != 180:
        raise ValueError("Stage 1 SQ-first plan")
    stress = result.get("stress_identity")
    expected_stress_shape = {"rows": expected_shape[0], "dimensions": expected_shape[1], "queries": expected_shape[2]}
    expected_algorithm = {
        "vectors": "benchmark_ann_build._vectors/v1",
        "exact_truth": "LanceDbIndexRepository.search_dense_exact_batch/cosine/limit20",
    }
    # The generator is pinned by source head, seed, shape, algorithm, locked
    # runtime, OMP setting, hosted runner and sealed artifact.  Re-generating
    # normalized float32 vectors on a different BLAS/OS is not a trustworthy
    # equality oracle: it can change bytes despite identical locked inputs.
    # The runner-produced corpus/query digests are therefore immutable artifact
    # identities; the exact IDs and recalls below are independently recomputed.
    if not isinstance(stress, dict) or set(stress) != {"schema_version", "corpus_sha256", "query_sha256", "exact_truth_sha256", "corpus_seed", "query_seed", "shape", "algorithm"} or stress.get("schema_version") != 1 or stress.get("corpus_seed") != benchmark.CORPUS_SEED or stress.get("query_seed") != benchmark.QUERY_SEED or stress.get("shape") != expected_stress_shape or stress.get("algorithm") != expected_algorithm or not all(isinstance(stress.get(name), str) and _HEX64.fullmatch(stress[name]) for name in ("corpus_sha256", "query_sha256", "exact_truth_sha256")) or stress["corpus_sha256"] == stress["query_sha256"]:
        raise ValueError("Stage 1 stress identity")
    if plan.get("corpus") != {"rows": expected_shape[0], "dimensions": expected_shape[1], "queries": expected_shape[2], "truth": "seeded_vector_exact"}:
        raise ValueError("Stage 1 plan/stress shape binding")
    builds = result.get("builds")
    if not isinstance(builds, list) or len(builds) != 3:
        raise ValueError("Stage 1 requires exactly three builds")
    observed_m, build_ids, truth_identity = set(), set(), None
    for build in builds:
        if not isinstance(build, dict) or set(build) != {"build_id", "build", "queries"} or not isinstance(build["build_id"], str) or not _HEX64.fullmatch(build["build_id"]) or build["build_id"] in build_ids:
            raise ValueError("Stage 1 fresh build identity")
        build_ids.add(build["build_id"])
        card = build["build"]
        watchdog = card.get("watchdog", {}) if isinstance(card, dict) else {}
        if not isinstance(card, dict) or card.get("candidate") != "ivf-hnsw-sq" or card.get("m") not in {16, 20, 32} or card.get("ef_construction") != 300 or card.get("normal_ann_request_count") != 4 * stress["shape"]["queries"] or watchdog.get("cap_seconds") != 180 or watchdog.get("owner") != "parent" or watchdog.get("child_exitcode") != 0 or card.get("reopen_verified") is not True or card.get("unindexed_dense_rows") != 0 or not isinstance(card.get("index_build_ms"), (int, float)) or not math.isfinite(card["index_build_ms"]) or not 0 <= card["index_build_ms"] <= 180_000 or not isinstance(card.get("index_bytes"), (int, float)) or not math.isfinite(card["index_bytes"]) or card["index_bytes"] <= 0:
            raise ValueError("Stage 1 build card/watchdog")
        observed_m.add(card["m"])
        groups = build["queries"]
        if not isinstance(groups, list) or [group.get("query_ef") for group in groups if isinstance(group, dict)] != [100, 150, 200, 300]:
            raise ValueError("Stage 1 query EF matrix")
        for group in groups:
            samples = group.get("queries") if isinstance(group, dict) else None
            if not isinstance(samples, list) or len(samples) != stress["shape"]["queries"]:
                raise ValueError("Stage 1 query sample cardinality")
            exact_ids = []
            for ordinal, sample in enumerate(samples):
                if not isinstance(sample, dict) or sample.get("query_index") != ordinal:
                    raise ValueError("Stage 1 query sample identity")
                exact10, exact20 = sample.get("exact_top_10"), sample.get("exact_top_20")
                candidate10, candidate20 = sample.get("candidate_top_10"), sample.get("candidate_top_20")
                if not all(isinstance(values, list) for values in (exact10, exact20, candidate10, candidate20)) or len(exact10) != 10 or len(exact20) != 20 or len(candidate10) != 10 or len(candidate20) != 20:
                    raise ValueError("Stage 1 exact truth sample")
                expected10 = len(set(exact10) & set(candidate10[:10])) / 10
                expected20 = len(set(exact20) & set(candidate20[:20])) / 20
                if sample.get("recall_at_10") != expected10 or sample.get("recall_at_20") != expected20:
                    raise ValueError("Stage 1 per-query recall recomputation")
                exact_ids.append(exact20)
            expected_group10 = sum(sample["recall_at_10"] for sample in samples) / len(samples)
            expected_group20 = sum(sample["recall_at_20"] for sample in samples) / len(samples)
            if group.get("recall_at_10") != expected_group10 or group.get("recall_at_20") != expected_group20 or not all(isinstance(group.get(name), (int, float)) and math.isfinite(group[name]) and group[name] >= 0 for name in ("build_time_ms", "exact_time_ms", "total_bytes", "index_delta_bytes", "latency_p50_ms", "latency_p95_ms")):
                raise ValueError("Stage 1 group aggregate/measurement recomputation")
            digest = _stage1_stress_digest(exact_ids)
            if truth_identity is None:
                truth_identity = digest
            elif digest != truth_identity:
                raise ValueError("Stage 1 exact truth differs across build/query records")
    if observed_m != {16, 20, 32} or truth_identity != stress["exact_truth_sha256"]:
        raise ValueError("Stage 1 distinct m/truth identity")
    statistics = result.get("d04_statistics")
    if statistics != Phase07AnnCampaignRunner._screening_statistics(builds):
        raise ValueError("Stage 1 D-04 six-member statistics recomputation")
    reported_nominees = _validate_stage1_nominee_list(
        result.get("nominated_m"), label="artifact-reported",
    )
    # The screening artifact records the campaign's own provisional list, but
    # the reconciler is its authority: recompute the D-04-qualified ranking
    # from sealed numeric observations before publishing any nominee.
    reconciled = dict(result)
    reconciled["nominated_m"] = _validate_stage1_nominee_list(
        select_stage1_nominees(builds, statistics), label="reconciled",
    )
    return reconciled, reported_nominees


def reconcile_stage1_screening(*, stage1_request: Path, artifact_dir: Path, output: Path, mode: str, expected_shape: tuple[int, int, int] = (77_348, 384, 256)) -> dict[str, Any]:
    """Recompute and seal one complete, non-authorizing Stage 1 screening artifact."""
    if mode != "screening" or artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise ValueError("strict Stage 1 reconciliation mode/artifact directory")
    found = {path.name for path in artifact_dir.iterdir()}
    if found != _STAGE1_FILES or any(path.is_symlink() for path in artifact_dir.iterdir()):
        raise ValueError("Stage 1 artifact has extra, missing, or symlinked files")
    extracted = artifact_dir / "extracted"
    extracted_found = {path.name for path in extracted.iterdir()}
    if any(name.endswith("-rejection.json") for name in extracted_found):
        raise ValueError("Stage 1 rejection artifact is never successful evidence")
    if extracted_found != _STAGE1_EXTRACTED_FILES or any(path.is_symlink() or not path.is_file() for path in extracted.iterdir()):
        raise ValueError("Stage 1 extracted content has extra, missing, or symlinked files")
    request = _read_json(stage1_request)
    _validate_stage1_request(request, artifact_dir)
    request_record = _stage1_read(extracted / "screening-request.json")
    ledger_record = _stage1_read(extracted / "screening-ledger.json")
    result_record = _stage1_read(extracted / "screening-result.json")
    if _stage1_file_sha256(extracted / "screening-request.json") != request["campaign_request_sha256"] or _stage1_file_sha256(extracted / "screening-ledger.json") != request["campaign_ledger_sha256"] or _stage1_file_sha256(extracted / "screening-result.json") != request["campaign_result_sha256"]:
        raise ValueError("Stage 1 post-download file digest binding")
    environment = request_record.get("environment", {})
    if environment.get("branch") != request["branch"] or environment.get("workflow_path") != request["workflow_path"] or environment.get("head_sha") != request["head_sha"] or environment.get("run_id") != request["run_id"] or environment.get("run_attempt") != request["run_attempt"] or environment.get("job_key") != request["job_key"] or environment.get("job_allocation_nonce") != request["job_allocation_nonce"] or environment.get("runtime") != request["runtime"] or request_record.get("model_manifest_sha256") != request["model_manifest_sha256"] or request_record.get("corpus_manifest_sha256") != request["corpus_manifest_sha256"] or request_record.get("lock_identity") != request["lock_identity"]:
        raise ValueError("Stage 1 campaign/API provenance binding")
    if set(ledger_record) != {"schema_version", "stage", "authorization", "request_sha256", "record_self_sha256"} \
            or ledger_record.get("schema_version") != 1 or ledger_record.get("stage") != "screening" \
            or ledger_record.get("authorization") != "none" \
            or ledger_record.get("request_sha256") != campaign_digest(request_record):
        raise ValueError("Stage 1 campaign ledger")
    result, artifact_reported_nominees = _validate_stage1_result(
        result_record, request_record, expected_shape=expected_shape,
    )
    ledger = {
        "schema_version": 1, "mode": "screening", "status": "success", "authorization": "none",
        "repository": request["repository"], "branch": request["branch"], "workflow_path": request["workflow_path"], "head_sha": request["head_sha"],
        "run": {name: request[name] for name in ("run_id", "run_attempt", "job_id", "job_key", "job_allocation_nonce")},
        "artifact": request["artifact"], "runtime": request["runtime"], "runner": request["runner"],
        "model_manifest_sha256": request["model_manifest_sha256"], "corpus_manifest_sha256": request["corpus_manifest_sha256"], "lock_identity": request["lock_identity"],
        "stress_identity": result["stress_identity"],
        "builds": result["builds"],
        "d04_statistics": result["d04_statistics"],
        "artifact_reported_nominated_m": artifact_reported_nominees,
        "nominated_m": result["nominated_m"],
        "source_digests": {"post_download_request": request["record_self_sha256"], "campaign_result": request["campaign_result_sha256"], "campaign_request": request["campaign_request_sha256"], "campaign_ledger": request["campaign_ledger_sha256"]},
    }
    ledger["record_self_sha256"] = _canonical_sha256(ledger)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ledger, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return ledger


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
    parser.add_argument("--scale-evidence", type=Path)
    parser.add_argument("--model-records", type=Path)
    parser.add_argument("--test-and-eval")
    parser.add_argument("--scale")
    parser.add_argument("--model-backed")
    parser.add_argument("--expected-head")
    parser.add_argument("--expected-pr-head")
    parser.add_argument("--stage1-request", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--mode")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirmation-request", type=Path)
    parser.add_argument("--confirmation-ledger", type=Path)
    parser.add_argument("--confirmation-evidence-manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.confirmation_evidence_manifest is not None:
            if args.confirmation_request is None or args.output is None or args.mode != "confirmation-postdownload":
                raise ValueError("post-download confirmation reconciliation requires request, manifest, output, and mode")
            result = reconcile_confirmation_postdownload(
                _read_json(args.confirmation_request), _read_json(args.confirmation_evidence_manifest),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("[PASS] post-download confirmation ANN reconciliation", file=sys.stderr)
            return 0
        if args.confirmation_request is not None or args.confirmation_ledger is not None:
            if args.confirmation_request is None or args.confirmation_ledger is None or args.mode != "confirmation":
                raise ValueError("confirmation reconciliation requires request, downloaded ledger, and confirmation mode")
            result = reconcile_confirmation_request(_read_json(args.confirmation_request), _read_json(args.confirmation_ledger))
            args.confirmation_ledger.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("[PASS] confirmation ANN reconciliation", file=sys.stderr)
            return 0
        if args.stage1_request is not None or args.artifact_dir is not None or args.mode is not None:
            if args.stage1_request is None or args.artifact_dir is None or args.output is None:
                raise ValueError("Stage 1 reconciliation requires request, artifact directory, and output")
            reconcile_stage1_screening(
                stage1_request=args.stage1_request, artifact_dir=args.artifact_dir,
                output=args.output, mode=args.mode or "",
            )
            print("[PASS] Stage 1 ANN reconciliation", file=sys.stderr)
            return 0
        if None in (args.scale_evidence, args.model_records, args.test_and_eval, args.scale, args.model_backed, args.expected_head, args.expected_pr_head):
            raise ValueError("PR reconciliation requires all legacy evidence inputs")
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
