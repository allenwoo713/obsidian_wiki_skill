"""Public build-mode contract gates for online incremental indexing."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
from build_index import WikiIndex


def test_public_facade_exposes_explicit_incremental_mode() -> None:
    """The public facade must expose online incremental as a distinct request."""
    parameters = inspect.signature(WikiIndex.build).parameters

    assert parameters["build_mode"].default == "snapshot"
    assert "build_mode_policy_path" in parameters


class _Embedder:
    """Model-free approved-dimension embedder for the real public facade."""

    class _Tokenizer:
        def encode(self, text: str):
            return text.split() or ["_"]

    tokenizer = _Tokenizer()

    def get_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts, **_kwargs):
        rows = []
        for offset, _text in enumerate(texts):
            raw = [float((offset + item) % 17 + 1) for item in range(384)]
            norm = math.sqrt(sum(value * value for value in raw))
            rows.append([value / norm for value in raw])
        return rows


def _write_page(wiki_dir: Path, text: str) -> None:
    (wiki_dir / "online.md").write_text(
        f"---\ntitle: Online\ntype: concept\n---\n{text}\n", encoding="utf-8"
    )


def _tiny_chunks(text: str):
    from obsidian_wiki.domain.index_models import SparseChunk

    common = dict(
        page_id="online.md", path="online.md", title="Online", text=text,
        content_hash=text, end_char=len(text),
    )
    return (
        SparseChunk(**common, chunk_id="online::sparse", fts_text=text, chunk_kind="sparse"),
        SparseChunk(**common, chunk_id="online::dense", fts_text=text, chunk_kind="dense"),
    )


def _enable_auto_incremental(index_dir: Path, manifest_path: Path) -> str:
    from obsidian_wiki.application.incremental_policy import compatibility_digest_from_manifest

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    telemetry = manifest["build_telemetry"]
    policy = {
        "schema_version": 1,
        "enabled": True,
        "compatibility_digest": compatibility_digest_from_manifest(manifest),
        "evidence_observation_ids": [telemetry["observation_id"]],
        "minimum_compatible_observations": 1,
        "max_evidence_age_seconds": 3600.0,
        "match": "all",
        "criteria": [{"metric": "snapshot_p95_ms", "operator": "gte", "threshold": 0.001}],
    }
    path = index_dir / "build-mode-policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    return path.read_text(encoding="utf-8")


def test_explicit_incremental_public_facade_reaches_staged_publication(tmp_path, monkeypatch) -> None:
    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    _write_page(wiki_dir, "original public facade payload")
    monkeypatch.setattr(WikiIndex, "_get_embedder", lambda self: _Embedder())

    WikiIndex(index_dir).build(wiki_dir)
    prior_pointer = (index_dir / "ACTIVE_INDEX").read_bytes()
    _write_page(wiki_dir, "edited public facade payload")

    outcome = WikiIndex(index_dir).build(wiki_dir, build_mode="incremental")

    assert outcome.published
    assert (index_dir / "ACTIVE_INDEX").read_bytes() != prior_pointer
    telemetry = json.loads(outcome.artifact.manifest_path.read_text(encoding="utf-8"))["build_telemetry"]
    assert telemetry["mode_requested"] == telemetry["mode_selected"] == "incremental"
    assert telemetry["selection_reason"] == "explicit_incremental"


def test_default_snapshot_is_reported_by_the_public_facade(tmp_path, monkeypatch) -> None:
    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    _write_page(wiki_dir, "default snapshot payload")
    monkeypatch.setattr(WikiIndex, "_get_embedder", lambda self: _Embedder())

    outcome = WikiIndex(tmp_path / ".index").build(wiki_dir)

    telemetry = json.loads(outcome.artifact.manifest_path.read_text(encoding="utf-8"))["build_telemetry"]
    assert telemetry["mode_requested"] == telemetry["mode_selected"] == "snapshot"
    assert telemetry["selection_reason"] == "explicit_snapshot"


def test_single_lock_covers_public_incremental_dispatch(tmp_path, monkeypatch) -> None:
    import obsidian_wiki.application.build_lock as lock_module
    import obsidian_wiki.application.incremental_index_service as incremental_module
    import obsidian_wiki.application.index_build_service as service_module

    calls: list[str] = []
    real_lock = lock_module.BuildLock

    class _TrackingBuildLock(real_lock):
        def acquire(self, *args, **kwargs):
            calls.append(self.ctx.build_id)
            return super().acquire(*args, **kwargs)

    monkeypatch.setattr(lock_module, "BuildLock", _TrackingBuildLock)
    monkeypatch.setattr(service_module, "BuildLock", _TrackingBuildLock)
    monkeypatch.setattr(incremental_module, "BuildLock", _TrackingBuildLock)
    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    _write_page(wiki_dir, "single writer baseline")
    monkeypatch.setattr(WikiIndex, "_get_embedder", lambda self: _Embedder())
    WikiIndex(index_dir).build(wiki_dir)
    _write_page(wiki_dir, "single writer edit")
    calls.clear()

    WikiIndex(index_dir).build(wiki_dir, build_mode="incremental")

    assert len(calls) == 1


def test_auto_without_policy_is_a_truthful_snapshot(tmp_path, monkeypatch) -> None:
    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    _write_page(wiki_dir, "auto missing policy baseline")
    monkeypatch.setattr(WikiIndex, "_get_embedder", lambda self: _Embedder())
    WikiIndex(index_dir).build(wiki_dir)
    _write_page(wiki_dir, "auto missing policy edit")

    outcome = WikiIndex(index_dir).build(wiki_dir, build_mode="auto")

    telemetry = json.loads(outcome.artifact.manifest_path.read_text(encoding="utf-8"))["build_telemetry"]
    assert telemetry["mode_requested"] == "auto"
    assert telemetry["mode_selected"] == "snapshot"
    assert telemetry["selection_reason"] == "policy_missing"
    assert telemetry["build_mode_policy_sha256"] is None


def test_auto_with_compatible_policy_reaches_incremental(tmp_path, monkeypatch) -> None:
    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    _write_page(wiki_dir, "auto compatible baseline")
    monkeypatch.setattr(WikiIndex, "_get_embedder", lambda self: _Embedder())
    baseline = WikiIndex(index_dir).build(wiki_dir)
    _enable_auto_incremental(index_dir, baseline.artifact.manifest_path)
    _write_page(wiki_dir, "auto compatible edit")

    outcome = WikiIndex(index_dir).build(wiki_dir, build_mode="auto")

    telemetry = json.loads(outcome.artifact.manifest_path.read_text(encoding="utf-8"))["build_telemetry"]
    assert telemetry["mode_requested"] == "auto"
    assert telemetry["mode_selected"] == "incremental"
    assert telemetry["selection_reason"] == "policy_criteria_met"
    assert telemetry["build_mode_policy_sha256"]


@pytest.mark.parametrize("fault", ["clone", "catch_up"])
def test_auto_pre_pointer_fault_falls_back_to_queryable_snapshot(tmp_path, monkeypatch, fault) -> None:
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    _write_page(wiki_dir, f"auto {fault} baseline")
    monkeypatch.setattr(WikiIndex, "_get_embedder", lambda self: _Embedder())
    baseline = WikiIndex(index_dir).build(wiki_dir)
    policy_bytes = _enable_auto_incremental(index_dir, baseline.artifact.manifest_path)
    old_pointer = (index_dir / "ACTIVE_INDEX").read_bytes()
    _write_page(wiki_dir, f"auto {fault} edit")

    method = "clone_tables" if fault == "clone" else "catch_up"
    monkeypatch.setattr(
        LanceDbIndexRepository, method,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(f"{fault} fault")),
    )
    outcome = WikiIndex(index_dir).build(wiki_dir, build_mode="auto")

    telemetry = json.loads(outcome.artifact.manifest_path.read_text(encoding="utf-8"))["build_telemetry"]
    assert telemetry["mode_requested"] == "auto"
    assert telemetry["mode_selected"] == "snapshot"
    assert telemetry["selection_reason"] == f"incremental_runtime_fallback:{'shallow_clone_unavailable' if fault == 'clone' else 'index_catch_up_unproven'}"
    assert telemetry["build_mode_policy_sha256"]
    assert (index_dir / "ACTIVE_INDEX").read_bytes() != old_pointer
    assert outcome.artifact.lance_dir.exists()
    assert (index_dir / "build-mode-policy.json").read_text(encoding="utf-8") == policy_bytes


def test_legacy_incremental_cannot_relabel_a_storage_mode_before_model_load(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_index.py", str(tmp_path), "--incremental", "--build-mode", "incremental"],
        cwd=Path(__file__).parent.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "legacy embedding-cache reuse" in result.stderr


def test_explicit_incremental_contract_mismatch_fails_closed_without_snapshot_alias(tmp_path, monkeypatch) -> None:
    from obsidian_wiki.application.incremental_index_service import (
        IncrementalFallbackEligible,
        IncrementalIndexService,
    )

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    _write_page(wiki_dir, "explicit mismatch baseline")
    monkeypatch.setattr(WikiIndex, "_get_embedder", lambda self: _Embedder())
    baseline = WikiIndex(index_dir).build(wiki_dir)
    pointer = (index_dir / "ACTIVE_INDEX").read_bytes()
    _write_page(wiki_dir, "explicit mismatch edit")
    monkeypatch.setattr(
        IncrementalIndexService, "_assert_current_manifest",
        staticmethod(lambda _path: (_ for _ in ()).throw(IncrementalFallbackEligible("incompatible_active_contract"))),
    )

    with pytest.raises(IncrementalFallbackEligible, match="incompatible_active_contract"):
        WikiIndex(index_dir).build(wiki_dir, build_mode="incremental")

    assert (index_dir / "ACTIVE_INDEX").read_bytes() == pointer


def test_auto_does_not_swallow_commit_uncertainty(tmp_path, monkeypatch) -> None:
    import obsidian_wiki.application.incremental_index_service as incremental_module
    from obsidian_wiki.application.durable_filesystem import CommitUncertainError

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    _write_page(wiki_dir, "uncertain baseline")
    monkeypatch.setattr(WikiIndex, "_get_embedder", lambda self: _Embedder())
    baseline = WikiIndex(index_dir).build(wiki_dir)
    _enable_auto_incremental(index_dir, baseline.artifact.manifest_path)
    _write_page(wiki_dir, "uncertain edit")
    monkeypatch.setattr(
        incremental_module, "publish_pointer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CommitUncertainError("pointer uncertain")),
    )

    with pytest.raises(CommitUncertainError, match="pointer uncertain"):
        WikiIndex(index_dir).build(wiki_dir, build_mode="auto")


def test_snapshot_direct_service_does_not_require_incremental_executor_factory(tmp_path) -> None:
    from obsidian_wiki.application.index_build_service import IndexBuildService
    from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest
    from obsidian_wiki.infrastructure.filesystem_post_commit_journal import FilesystemPostCommitJournal
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    service = IndexBuildService(
        LanceDbIndexRepository(index_dir), reopen_storage=LanceDbIndexRepository,
        manifest_store=FilesystemIndexManifest(),
        post_commit_journal=FilesystemPostCommitJournal(index_dir),
    )
    artifact = service.build(
        wiki_dir, index_dir, embed=_Embedder().encode,
        sparse_chunks=_tiny_chunks("direct snapshot"), build_mode="snapshot",
    )
    assert artifact.lance_dir.is_dir()


def test_explicit_incremental_without_executor_factory_fails_before_mutation(tmp_path, monkeypatch) -> None:
    from obsidian_wiki.application.index_build_service import IndexBuildService
    from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest
    from obsidian_wiki.infrastructure.filesystem_post_commit_journal import FilesystemPostCommitJournal
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    storage = LanceDbIndexRepository(index_dir)
    calls: list[str] = []
    monkeypatch.setattr(storage, "persist", lambda *_args, **_kwargs: calls.append("persist"))
    service = IndexBuildService(
        storage, reopen_storage=LanceDbIndexRepository,
        manifest_store=FilesystemIndexManifest(),
        post_commit_journal=FilesystemPostCommitJournal(index_dir),
    )
    with pytest.raises(RuntimeError, match="^incremental_executor_unavailable$"):
        service.build(
            wiki_dir, index_dir, embed=_Embedder().encode,
            sparse_chunks=_tiny_chunks("explicit unavailable"), build_mode="incremental",
        )
    assert calls == []
    assert not (index_dir / "ACTIVE_INDEX").exists()


def test_auto_incremental_without_executor_factory_persists_snapshot_reason(tmp_path, monkeypatch) -> None:
    from obsidian_wiki.application.index_build_service import IndexBuildService
    from obsidian_wiki.domain.incremental_models import BuildModeSelection
    from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest
    from obsidian_wiki.infrastructure.filesystem_post_commit_journal import FilesystemPostCommitJournal
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    service = IndexBuildService(
        LanceDbIndexRepository(index_dir), reopen_storage=LanceDbIndexRepository,
        manifest_store=FilesystemIndexManifest(),
        post_commit_journal=FilesystemPostCommitJournal(index_dir),
    )
    selection = BuildModeSelection("incremental", "policy: measured", "a" * 64, "b" * 64, ("snapshot-a",))
    monkeypatch.setattr(service, "_select_build_mode", lambda *_args, **_kwargs: selection)
    artifact = service.build(
        wiki_dir, index_dir, embed=_Embedder().encode,
        sparse_chunks=_tiny_chunks("auto unavailable"), build_mode="auto",
    )
    telemetry = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))["build_telemetry"]
    assert telemetry["mode_requested"] == "auto"
    assert telemetry["mode_selected"] == "snapshot"
    assert telemetry["selection_reason"] == "incremental_executor_unavailable"
    assert telemetry["build_mode_policy_sha256"] == "a" * 64


def test_public_snapshot_and_incremental_share_one_publication_collaborator_per_composition(tmp_path, monkeypatch) -> None:
    import build_index as module

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    composed = module._compose_storage_services(index_dir)
    executor = composed.incremental_executor_factory()
    assert composed.service._publication_service is composed.publication_service
    assert executor._publication_service is composed.publication_service
    calls: list[str] = []
    for name in ("allocate_generation", "validate_candidate", "construct_manifest"):
        original = getattr(composed.publication_service, name)
        monkeypatch.setattr(
            composed.publication_service, name,
            lambda *args, _name=name, _original=original, **kwargs: (
                calls.append(_name), _original(*args, **kwargs)
            )[1],
        )
    monkeypatch.setattr(module, "_compose_storage_services", lambda *_args, **_kwargs: composed)
    module.build_storage_contract(wiki_dir, index_dir, embed=_Embedder().encode, sparse_chunks=_tiny_chunks("first"))
    module.build_storage_contract(
        wiki_dir, index_dir, embed=_Embedder().encode,
        sparse_chunks=_tiny_chunks("second"), build_mode="incremental",
    )
    assert calls.count("allocate_generation") == 2
    assert calls.count("validate_candidate") == 2
    assert calls.count("construct_manifest") >= 4
