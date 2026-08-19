"""Fail-closed public-route comparison of snapshot and online incremental builds.

This evaluator deliberately uses the same :class:`WikiIndex` facade and
``query.hybrid_search`` orchestration as production.  It never initializes an
eval baseline, changes ANN policy, or reaches the exact-retrieval diagnostic
API.  The output is an evidence artifact, not a baseline replacement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SKILL_ROOT = HERE.parent
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_index import WikiIndex  # noqa: E402
from obsidian_wiki.application.incremental_policy import (  # noqa: E402
    compatibility_identity_from_manifest,
)
from obsidian_wiki.domain.index_models import FtsIndexConfig  # noqa: E402
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository  # noqa: E402
from query import hybrid_search  # noqa: E402
from query_planner import DefaultQueryPlanner  # noqa: E402
from run_eval import (  # noqa: E402
    _citation_violations,
    _evidence_recall,
    _fixture_digest,
    _norm_chunk_key,
    _page_hit,
    _recall_at_5,
    _stable_json_digest,
    write_graph_artifact,
)


SCHEMA_VERSION = 1
REQUIRED_SCENARIOS = (
    "page_edit",
    "split_merge",
    "page_deletion",
    "unchanged_rebuild",
    "configuration_drift",
    "failure_recovery",
)
_APPROVED_ANN = {
    "selected_index_type": "ivf-hnsw-sq",
    "lancedb_index_type": "hnsw_sq",
    "metric": "cosine",
    "query_ef": 100,
    "recall_at_10_floor": 0.19,
    "recall_at_20_floor": 0.17,
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))


def _page(title: str, body: str) -> str:
    return f"---\ntitle: {title}\ntype: concept\n---\n{body.strip()}\n"


def _initial_files() -> dict[str, str]:
    return {
        "alpha.md": _page(
            "Alpha Sensor",
            "# Measurement\nAlpha sensor measures 72 units. [[Beta Sensor]] confirms the reading.",
        ),
        "beta.md": _page(
            "Beta Sensor", "# Evidence\nBeta Sensor records the Alpha Sensor measurement.",
        ),
        "obsolete.md": _page(
            "Obsolete Sensor", "# Retired\nobsolete-token-zqxwp must never remain after deletion.",
        ),
    }


def _final_files(scenario: str) -> dict[str, str]:
    files = _initial_files()
    if scenario == "page_edit":
        files["alpha.md"] = _page(
            "Alpha Sensor", "# Measurement\nAlpha sensor measures 84 units after calibration. [[Beta Sensor]] confirms the reading.",
        )
    elif scenario == "split_merge":
        files["alpha.md"] = _page(
            "Alpha Sensor",
            "# Measurement\nAlpha sensor measures 72 units.\n\n# Calibration\nCalibration keeps Alpha stable.\n\n# Evidence\n[[Beta Sensor]] confirms the reading.",
        )
    elif scenario == "page_deletion":
        del files["obsolete.md"]
    elif scenario == "unchanged_rebuild":
        pass
    elif scenario in {"configuration_drift", "failure_recovery"}:
        files["alpha.md"] = _page(
            "Alpha Sensor", "# Measurement\nAlpha sensor measures 84 units after recovery. [[Beta Sensor]] confirms the reading.",
        )
    else:
        raise ValueError(f"unknown comparison scenario: {scenario}")
    return files


def _write_files(wiki: Path, files: dict[str, str]) -> None:
    if wiki.exists():
        shutil.rmtree(wiki)
    wiki.mkdir(parents=True)
    for relative, text in sorted(files.items()):
        (wiki / relative).write_text(text, encoding="utf-8")


def _relative_page_id(value: str, wiki: Path) -> str:
    path = Path(value)
    try:
        return f"Wiki/{path.resolve().relative_to(wiki.resolve()).as_posix()}"
    except (OSError, ValueError):
        return str(value).replace("\\", "/")


def _normal_path(value: str, wiki: Path) -> str:
    """Remove only the isolated build root from a public citation/path value."""
    text = str(value).replace("\\", "/")
    root = str(wiki.resolve()).replace("\\", "/")
    root_without_leading = root.lstrip("/")
    text = text.replace(root, "Wiki").replace(root_without_leading, "Wiki")
    marker = "/Wiki/"
    if marker in text:
        return "Wiki/" + text.rsplit(marker, 1)[1]
    return text


def _normal_hit(hit: object, wiki: Path) -> dict[str, object]:
    return {
        "page_id": _relative_page_id(str(getattr(hit, "page_id", "")), wiki),
        "chunk_id": _norm_chunk_key(str(getattr(hit, "chunk_id", ""))),
        "channel": str(getattr(hit, "channel", "")),
        "rank": int(getattr(hit, "rank", 0)),
    }


def _result_observation(result: object, query: dict[str, object], wiki: Path) -> dict[str, object]:
    bundle = result.bundle
    citations = []
    for item in bundle.items:
        citations.append({
            "page_id": _relative_page_id(str(item.page_id), wiki),
            "path": _normal_path(str(item.path), wiki),
            "evidence": [_norm_chunk_key(hit.chunk_id) for hit in item.evidence],
            "graph_paths": [{
                "source": _relative_page_id(path.source_id, wiki),
                "target": _relative_page_id(path.target_id, wiki),
                "signals": list(path.edge_signals),
            } for path in item.graph_paths],
        })
    sparse = [_normal_hit(hit, wiki) for hit in result._mode_comparison_sparse]
    dense = [_normal_hit(hit, wiki) for hit in result._mode_comparison_dense]
    context_text = _normal_path(bundle.context_text or "", wiki)
    return {
        "page_recall_at_5": _recall_at_5(result.candidates, query["relevant_pages"]),
        "evidence_recall_at_10": _evidence_recall(bundle, query.get("required_facts", [])),
        "candidate_pages": [_relative_page_id(candidate.page_id, wiki)
                            for candidate in result.candidates[:10]],
        "citations": citations,
        "citation_violations": _citation_violations(bundle),
        "context": {
            "sha256": _sha256_bytes(context_text.encode("utf-8")),
            "text": context_text,
            "token_count": bundle.token_count,
            "budget": bundle.budget_to_json(),
        },
        "graph": {
            "validated_count": result.graph_validated_count,
            "items": citations,
        },
        "sparse": sparse,
        "dense": dense,
    }


def _manifest_observation(wi: WikiIndex) -> tuple[dict[str, object], dict[str, object]]:
    manifest = json.loads(wi._resolve_active_manifest().read_text(encoding="utf-8"))
    telemetry = manifest["build_telemetry"]
    ann = manifest["ann_policy"]
    if any(ann.get(name) != value for name, value in _APPROVED_ANN.items()):
        raise ValueError("fixed Phase-6 ANN contract changed")
    sparse = telemetry["sparse_rows"]
    dense = telemetry["dense_rows"]
    written = sparse["physically_written"] + dense["physically_written"]
    validation = manifest["validation"]
    unindexed = max(
        int(validation["fts_index"].get("unindexed_rows", -1)),
        int(validation["vector_index"].get("unindexed_dense_rows", -1)),
    )
    return ({
        "build_id": manifest["build_id"],
        "manifest_sha256": _sha256_bytes(wi._resolve_active_manifest().read_bytes()),
        "layout": manifest["layout"],
        "tables": ["dense_chunks", "sparse_chunks"],
        "build_mode_requested": telemetry["mode_requested"],
        "build_mode_selected": telemetry["mode_selected"],
        "selection_reason": telemetry["selection_reason"],
        "build_mode_policy_sha256": telemetry.get("build_mode_policy_sha256"),
        "compatibility_digest": telemetry["compatibility_digest"],
        "unindexed_rows": unindexed,
        "ann": {name: ann[name] for name in _APPROVED_ANN},
        "candidate_publication_evidence": manifest.get("candidate_publication_evidence"),
    }, {
        "sparse_rows": sparse,
        "dense_rows": dense,
        "written_rows": written,
    })


def _query_index(wi: WikiIndex, project: Path, specification: dict[str, object]) -> dict[str, object]:
    wiki = project / "Wiki"
    planner = DefaultQueryPlanner(project_root=project)
    result = hybrid_search(
        wi, str(specification["query"]), planner, k=10, max_tokens=512,
        wiki_dir=wiki, intent_override="auto", allow_local_fallback=True,
    )
    # Read both public search channels in addition to hybrid_search's own use,
    # so the artifact proves split sparse/dense observations independently.
    result._mode_comparison_sparse = wi.search_fts(str(specification["query"]), k=10)
    result._mode_comparison_dense = wi.search_vector(str(specification["query"]), k=10)
    return _result_observation(result, specification, wiki)


def _build(project: Path, files: dict[str, str], *, mode: str,
           policy_path: Path | None = None) -> tuple[WikiIndex, dict[str, object], dict[str, object]]:
    wiki = project / "Wiki"
    _write_files(wiki, files)
    index = WikiIndex(project / ".index")
    index.build(wiki, build_mode=mode, build_mode_policy_path=policy_path)
    index.load()
    write_graph_artifact(wiki, project / ".index")
    manifest, telemetry = _manifest_observation(index)
    return index, manifest, telemetry


def _comparison_query(scenario: str) -> dict[str, object]:
    fact = "84 units" if scenario in {"page_edit", "configuration_drift", "failure_recovery"} else "72 units"
    return {
        "query": "What measurement does Alpha Sensor record with Beta Sensor evidence?",
        "relevant_pages": ["alpha.md"],
        "required_facts": [fact],
    }


def _equivalence(snapshot: dict[str, object], incremental: dict[str, object]) -> dict[str, bool]:
    fields = {
        "page_recall_at_5": snapshot["page_recall_at_5"] == incremental["page_recall_at_5"],
        "evidence_recall_at_10": snapshot["evidence_recall_at_10"] == incremental["evidence_recall_at_10"],
        "candidates": snapshot["candidate_pages"] == incremental["candidate_pages"],
        "citations": snapshot["citations"] == incremental["citations"],
        "context": snapshot["context"] == incremental["context"],
        "graph": snapshot["graph"] == incremental["graph"],
        "sparse": snapshot["sparse"] == incremental["sparse"],
        "dense": snapshot["dense"] == incremental["dense"],
    }
    return fields


def _compatibility_policy(project: Path, manifest: dict[str, object], *,
                          fts_config: FtsIndexConfig | None = None) -> Path:
    telemetry = manifest["build_telemetry"]
    contract = compatibility_identity_from_manifest(manifest)
    if fts_config is not None:
        contract["fts_config"] = fts_config.to_json()
    policy = {
        "schema_version": 1,
        "enabled": True,
        "compatibility_digest": _canonical_sha256(contract),
        "compatibility_contract": contract,
        "evidence_observation_ids": [telemetry["observation_id"]],
        "minimum_compatible_observations": 1,
        "max_evidence_age_seconds": 3600.0,
        "match": "all",
        "criteria": [{"metric": "snapshot_p95_ms", "operator": "gte", "threshold": 0.001}],
    }
    path = project / "build-mode-policy.json"
    path.write_text(json.dumps(policy, sort_keys=True), encoding="utf-8")
    return Path("build-mode-policy.json")


def _run_configuration_drift(project: Path, initial: dict[str, str], final: dict[str, str]) -> tuple[WikiIndex, dict[str, object], dict[str, object], dict[str, int]]:
    wi, _, _ = _build(project, initial, mode="snapshot")
    manifest_path = wi._resolve_active_manifest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy_path = _compatibility_policy(
        project, manifest, fts_config=FtsIndexConfig(max_token_length=128),
    )
    _write_files(project / "Wiki", final)
    calls = {"clone_table": 0, "delta": 0, "catch_up": 0}
    real_clone, real_delta, real_catch_up = (
        LanceDbIndexRepository.clone_tables,
        LanceDbIndexRepository.apply_delta,
        LanceDbIndexRepository.catch_up,
    )

    def clone(*args, **kwargs):
        calls["clone_table"] += 1
        return real_clone(*args, **kwargs)

    def delta(*args, **kwargs):
        calls["delta"] += 1
        return real_delta(*args, **kwargs)

    def catch_up(*args, **kwargs):
        calls["catch_up"] += 1
        return real_catch_up(*args, **kwargs)

    with patch.object(LanceDbIndexRepository, "clone_tables", clone), \
            patch.object(LanceDbIndexRepository, "apply_delta", delta), \
            patch.object(LanceDbIndexRepository, "catch_up", catch_up):
        wi = WikiIndex(project / ".index")
        wi.build(project / "Wiki", build_mode="auto", build_mode_policy_path=policy_path)
        wi.load()
    write_graph_artifact(project / "Wiki", project / ".index")
    manifest_observation, telemetry = _manifest_observation(wi)
    return wi, manifest_observation, telemetry, calls


def _run_failure_recovery(project: Path, initial: dict[str, str], final: dict[str, str]) -> tuple[WikiIndex, dict[str, object], dict[str, object], bool]:
    wi, _, _ = _build(project, initial, mode="snapshot")
    prior_pointer = (project / ".index" / "ACTIVE_INDEX").read_bytes()
    _write_files(project / "Wiki", final)
    failed = False
    with patch.object(
        LanceDbIndexRepository, "catch_up",
        side_effect=RuntimeError("comparison injected catch-up failure"),
    ):
        try:
            WikiIndex(project / ".index").build(project / "Wiki", build_mode="incremental")
        except Exception:  # The public pre-pointer failure is the behavior under test.
            failed = True
    if not failed:
        raise RuntimeError("injected incremental catch-up failure unexpectedly published")
    preserved = (project / ".index" / "ACTIVE_INDEX").read_bytes() == prior_pointer
    wi = WikiIndex(project / ".index")
    wi.build(project / "Wiki", build_mode="snapshot")
    wi.load()
    write_graph_artifact(project / "Wiki", project / ".index")
    manifest, telemetry = _manifest_observation(wi)
    return wi, manifest, telemetry, preserved


def _scenario_record(name: str, root: Path) -> dict[str, object]:
    initial, final = _initial_files(), _final_files(name)
    snapshot_project = root / "snapshot"
    incremental_project = root / "incremental"
    snapshot, snapshot_manifest, snapshot_telemetry = _build(snapshot_project, final, mode="snapshot")
    if name == "configuration_drift":
        incremental, incremental_manifest, incremental_telemetry, calls = _run_configuration_drift(
            incremental_project, initial, final,
        )
    elif name == "failure_recovery":
        incremental, incremental_manifest, incremental_telemetry, preserved = _run_failure_recovery(
            incremental_project, initial, final,
        )
        calls = None
    else:
        _build(incremental_project, initial, mode="snapshot")
        incremental, incremental_manifest, incremental_telemetry = _build(
            incremental_project, final, mode="incremental",
        )
        calls = None
    query = _comparison_query(name)
    snapshot_result = _query_index(snapshot, snapshot_project, query)
    incremental_result = _query_index(incremental, incremental_project, query)
    equivalence = _equivalence(snapshot_result, incremental_result)
    record: dict[str, object] = {
        "verdict": "pass" if all(equivalence.values()) else "fail",
        "input_sha256": _canonical_sha256(final),
        "query_sha256": _canonical_sha256(query),
        "snapshot": {"manifest": snapshot_manifest, "telemetry": snapshot_telemetry, "result": snapshot_result},
        "incremental": {"manifest": incremental_manifest, "telemetry": incremental_telemetry, "result": incremental_result},
        "equivalence": equivalence,
    }
    if name == "page_deletion":
        deleted = "obsolete.md"
        rendered = json.dumps(record["incremental"], ensure_ascii=False)
        record["deleted_page_absent"] = deleted not in rendered and "obsolete-token-zqxwp" not in rendered
    if name == "configuration_drift":
        record["incremental_calls"] = calls
    if name == "failure_recovery":
        record["recovery_preserved_active"] = preserved
    return record


def _artifact_digest(artifact: dict[str, object]) -> str:
    payload = copy.deepcopy(artifact)
    payload.pop("artifact_sha256", None)
    return _canonical_sha256(payload)


def validate_comparison_artifact(artifact: dict[str, object]) -> dict[str, object]:
    """Reject stale, partial, policy-drifted, or non-equivalent evidence."""
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("comparison artifact schema")
    inputs = artifact.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "fixture_sha256", "queries_sha256", "baselines_sha256", "ann_policy_sha256",
    } or not all(isinstance(value, str) and len(value) == 64 for value in inputs.values()):
        raise ValueError("comparison artifact input bindings")
    if inputs["baselines_sha256"] != _sha256_bytes((HERE / "baselines.json").read_bytes()):
        raise ValueError("comparison artifact baseline binding")
    if inputs["ann_policy_sha256"] != _sha256_bytes((HERE / "ann-policy.json").read_bytes()):
        raise ValueError("comparison artifact ANN policy binding")
    scenarios = artifact.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios or not set(scenarios).issubset(REQUIRED_SCENARIOS):
        raise ValueError("comparison artifact scenarios")
    for name, scenario in scenarios.items():
        if not isinstance(scenario, dict) or scenario.get("verdict") != "pass":
            raise ValueError(f"comparison scenario failed: {name}")
        if not isinstance(scenario.get("input_sha256"), str) or not isinstance(scenario.get("query_sha256"), str):
            raise ValueError("comparison scenario digest")
        for mode in ("snapshot", "incremental"):
            route = scenario.get(mode)
            if not isinstance(route, dict) or not isinstance(route.get("manifest"), dict) or not isinstance(route.get("result"), dict):
                raise ValueError("comparison route evidence")
            manifest = route["manifest"]
            if manifest.get("layout") != "sparse_chunks+dense_chunks" or manifest.get("tables") != ["dense_chunks", "sparse_chunks"]:
                raise ValueError("comparison split-table layout")
            if manifest.get("unindexed_rows") != 0 or manifest.get("ann") != _APPROVED_ANN:
                raise ValueError("comparison index coverage or ANN contract")
            if not manifest.get("candidate_publication_evidence"):
                raise ValueError("comparison publication evidence")
            result = route["result"]
            if result.get("citation_violations") or result.get("context", {}).get("budget", {}).get("effective_budget_tokens", 0) < result.get("context", {}).get("token_count", 0):
                raise ValueError("comparison citation or context budget contract")
        if not all(scenario.get("equivalence", {}).get(field) is True for field in (
            "page_recall_at_5", "evidence_recall_at_10", "candidates", "citations", "context", "graph", "sparse", "dense",
        )):
            raise ValueError("comparison equivalence")
        if name == "page_deletion" and scenario.get("deleted_page_absent") is not True:
            raise ValueError("comparison deletion residue")
        if name == "unchanged_rebuild" and scenario["incremental"]["telemetry"].get("written_rows") != 0:
            raise ValueError("comparison unchanged writes")
        if name == "configuration_drift":
            calls = scenario.get("incremental_calls")
            manifest = scenario["incremental"]["manifest"]
            if calls != {"clone_table": 0, "delta": 0, "catch_up": 0} or manifest.get("build_mode_requested") != "auto" or manifest.get("build_mode_selected") != "snapshot" or manifest.get("selection_reason") != "incompatible_active_contract:fts_config" or not manifest.get("build_mode_policy_sha256"):
                raise ValueError("comparison configuration fallback")
        if name == "failure_recovery" and scenario.get("recovery_preserved_active") is not True:
            raise ValueError("comparison failure recovery")
    if artifact.get("verdict") != "pass" or artifact.get("artifact_sha256") != _artifact_digest(artifact):
        raise ValueError("comparison artifact verdict or digest")
    return artifact


def run_mode_comparison(*, work_dir: Path, output: Path,
                        scenarios: Iterable[str] = REQUIRED_SCENARIOS) -> dict[str, object]:
    """Run isolated public builds and persist a digest-bound JSON artifact."""
    selected = tuple(scenarios)
    if not selected or len(set(selected)) != len(selected) or any(name not in REQUIRED_SCENARIOS for name in selected):
        raise ValueError("comparison scenarios must be unique supported names")
    baseline_before = (HERE / "baselines.json").read_bytes()
    policy_before = (HERE / "ann-policy.json").read_bytes()
    work_dir = Path(work_dir)
    output = Path(output)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    records = {name: _scenario_record(name, work_dir / name) for name in selected}
    if (HERE / "baselines.json").read_bytes() != baseline_before or (HERE / "ann-policy.json").read_bytes() != policy_before:
        raise RuntimeError("comparison attempted to modify an accepted baseline or ANN policy")
    artifact: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "fixture_sha256": _canonical_sha256(_initial_files()),
            "queries_sha256": _canonical_sha256([_comparison_query(name) for name in selected]),
            "baselines_sha256": _sha256_bytes(baseline_before),
            "ann_policy_sha256": _sha256_bytes(policy_before),
        },
        "scenarios": records,
        "verdict": "pass" if all(item["verdict"] == "pass" for item in records.values()) else "fail",
    }
    artifact["artifact_sha256"] = _artifact_digest(artifact)
    validate_comparison_artifact(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", action="append", choices=REQUIRED_SCENARIOS)
    args = parser.parse_args()
    artifact = run_mode_comparison(
        work_dir=args.work_dir, output=args.output,
        scenarios=tuple(args.scenario) if args.scenario else REQUIRED_SCENARIOS,
    )
    print(json.dumps({"verdict": artifact["verdict"], "artifact_sha256": artifact["artifact_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
