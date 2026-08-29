"""Real-LanceDB gates for the staged online incremental path."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_index import build_storage_contract  # noqa: E402
from obsidian_wiki.domain.index_models import SparseChunk  # noqa: E402
from obsidian_wiki.domain.index_policy import load_ann_policy_file  # noqa: E402


def _active_rows(index_dir: Path) -> tuple[dict[str, object], ...]:
    from obsidian_wiki.application.active_index_pointer import resolve_active_lance_dir
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    return tuple(LanceDbIndexRepository(resolve_active_lance_dir(index_dir)).table_rows("sparse_chunks"))


def _embed(texts):
    """Small deterministic, approved-dimension test embedder."""
    vectors = []
    for offset, _text in enumerate(texts):
        raw = [float((offset + index) % 17 + 1) for index in range(384)]
        norm = math.sqrt(sum(value * value for value in raw))
        vectors.append([value / norm for value in raw])
    return vectors


def _chunks(*, text: str) -> tuple[SparseChunk, ...]:
    page_id = "concepts/online.md"
    common = dict(
        page_id=page_id,
        path="concepts/online.md", title="Online", text=text,
        content_hash=text, end_char=len(text),
    )
    return (
        SparseChunk(**common, chunk_id=f"{page_id}::sparse-stable", fts_text=text, chunk_kind="sparse"),
        SparseChunk(**common, chunk_id=f"{page_id}::dense-stable", fts_text=text, chunk_kind="dense"),
    )


def _page_chunks(page: str, *, text: str, suffix: str = "stable") -> tuple[SparseChunk, ...]:
    page_id = f"concepts/{page}.md"
    common = dict(
        page_id=page_id, path=f"concepts/{page}.md", title=page.title(), text=text,
        content_hash=text, end_char=len(text),
    )
    return (
        SparseChunk(**common, chunk_id=f"{page_id}::sparse-{suffix}", fts_text=text, chunk_kind="sparse"),
        SparseChunk(**common, chunk_id=f"{page_id}::dense-{suffix}", fts_text=text, chunk_kind="dense"),
    )


def _pointer(index_dir: Path) -> dict[str, object]:
    return json.loads((index_dir / "ACTIVE_INDEX").read_text(encoding="utf-8"))


def test_single_page_edit_uses_staged_shallow_clone_and_atomic_pointer(tmp_path, monkeypatch):
    """An edit mutates only a cloned generation before the sole pointer commit."""
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService
    import obsidian_wiki.application.incremental_index_service as service_module
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    original = _chunks(text="original unique payload")
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=original)
    old_pointer_bytes = (index_dir / "ACTIVE_INDEX").read_bytes()
    old_pointer = _pointer(index_dir)
    old_lance = index_dir / str(old_pointer["active_lance"])
    old_repository = LanceDbIndexRepository(old_lance)
    old_sparse_rows = old_repository.context_rows("page_id = 'concepts/online.md'")

    publish_calls: list[tuple[Path, Path]] = []
    real_publish = service_module.publish_pointer

    def _publish_after_all_validation(index_path, build_path, **kwargs):
        # Every source observation remains unchanged until the actual commit point.
        assert (index_dir / "ACTIVE_INDEX").read_bytes() == old_pointer_bytes
        assert LanceDbIndexRepository(old_lance).context_rows(
            "page_id = 'concepts/online.md'"
        ) == old_sparse_rows
        publish_calls.append((index_path, build_path))
        return real_publish(index_path, build_path, **kwargs)

    monkeypatch.setattr(service_module, "publish_pointer", _publish_after_all_validation)
    outcome = IncrementalIndexService().build(
        wiki_dir, index_dir, canonical_chunks=_chunks(text="edited unique payload"), embed=_embed,
    )

    assert len(publish_calls) == 1
    assert outcome.artifact.lance_dir != old_lance
    from obsidian_wiki.domain.index_publication_models import ActiveIndexPointerV4

    pointer = ActiveIndexPointerV4.from_json(_pointer(index_dir))
    assert pointer.active_lance == outcome.artifact.lance_dir.relative_to(index_dir).as_posix()
    assert "\\" not in pointer.active_lance
    manifest = json.loads(outcome.artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["layout"] == "sparse_chunks+dense_chunks"
    assert manifest["ann_policy"]["selected_index_type"] == "ivf-hnsw-sq"
    approved_ann = load_ann_policy_file()
    assert manifest["vector_config"]["m"] == approved_ann.m
    assert manifest["vector_config"]["ef_construction"] == approved_ann.ef_construction
    assert manifest["ann_policy"]["query_ef"] == approved_ann.query_ef


def test_unchanged_population_has_no_mutation_rows(tmp_path):
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    plan = _page_chunks("steady", text="unchanged alpha") + _page_chunks("also_steady", text="unchanged beta")
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=plan)

    outcome = IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=plan, embed=_embed)

    for delta in (outcome.sparse_delta, outcome.dense_delta):
        assert not delta.added_ids and not delta.updated_ids and not delta.deleted_ids
        assert len(delta.unchanged_ids) == 2
        assert not delta.physically_written_ids


def test_split_merge_retains_stable_ids_and_reports_only_new_rows(tmp_path):
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    retained = _page_chunks("split", text="retained block")
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=retained)
    expanded = retained + _page_chunks("split", text="new split block", suffix="new")

    outcome = IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=expanded, embed=_embed)

    assert len(outcome.sparse_delta.unchanged_ids) == 1
    assert len(outcome.dense_delta.unchanged_ids) == 1
    assert len(outcome.sparse_delta.added_ids) == 1
    assert len(outcome.dense_delta.added_ids) == 1
    assert not outcome.sparse_delta.updated_ids and not outcome.dense_delta.updated_ids


def test_page_deletion_removes_all_table_and_manifest_residue(tmp_path):
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    keeper = _page_chunks("keeper", text="retain this content")
    doomed = _page_chunks("doomed", text="unique deleted-token-zqxwp")
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=keeper + doomed)

    outcome = IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=keeper, embed=_embed)
    repository = LanceDbIndexRepository(outcome.artifact.lance_dir)
    for table_name in ("sparse_chunks", "dense_chunks"):
        assert not [row for row in repository.table_rows(table_name) if row["page_id"] == "concepts/doomed.md"]
    manifest = json.loads(outcome.artifact.manifest_path.read_text(encoding="utf-8"))
    assert all(page["page_id"] != "concepts/doomed.md" for page in manifest["pages"])
    assert len(outcome.sparse_delta.deleted_ids) == len(outcome.dense_delta.deleted_ids) == 1


def test_row_accounting_reports_only_physical_upserts(tmp_path):
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    original = _page_chunks("accounted", text="original accounting")
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=original)
    changed = _page_chunks("accounted", text="changed accounting")

    outcome = IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=changed, embed=_embed)

    for delta, mutation in ((outcome.sparse_delta, outcome.sparse_mutation), (outcome.dense_delta, outcome.dense_mutation)):
        assert mutation.inserted == len(delta.added_ids)
        assert mutation.updated == len(delta.updated_ids)
        assert mutation.deleted == len(delta.deleted_ids)
        assert mutation.physically_written == len(delta.physically_written_ids)


def test_journal_requires_monotonic_identity_bound_durable_transitions(tmp_path):
    """Journal records survive reconstruction but reject malformed or skipped states."""
    from obsidian_wiki.domain.incremental_models import (
        IncrementalJournalRecord,
        IncrementalJournalState,
        SourceTableIdentity,
    )
    from obsidian_wiki.infrastructure.filesystem_incremental_journal import FilesystemIncrementalJournal

    index_dir = tmp_path / ".index"
    build_id = "build_20260819T000000000000_" + "a" * 32
    prepared = IncrementalJournalRecord(
        schema_version=1, build_id=build_id, generation=2,
        state=IncrementalJournalState.PREPARED,
        prior_pointer_sha256="b" * 64,
        source_build_id="build_20260819T000000000000_" + "f" * 32,
        source_tables=(
            SourceTableIdentity("sparse_chunks", 3, 2),
            SourceTableIdentity("dense_chunks", 4, 2),
        ),
        plan_sha256="c" * 64, config_sha256="d" * 64, policy_sha256="e" * 64,
        target_build=f"builds/{build_id}", last_completed_boundary="prepared",
    )
    journal = FilesystemIncrementalJournal(index_dir)
    journal.prepare(prepared)
    journal.transition(build_id, IncrementalJournalState.CLONED, boundary="clone")

    reconstructed = FilesystemIncrementalJournal(index_dir).load(build_id)
    assert reconstructed is not None
    assert reconstructed.state is IncrementalJournalState.CLONED
    assert reconstructed.last_completed_boundary == "clone"
    with pytest.raises(ValueError, match="illegal incremental journal transition"):
        journal.transition(build_id, IncrementalJournalState.VALIDATED, boundary="skip")

    path = index_dir / "incremental_journal" / f"{build_id}.json"
    path.write_text('{"foreign": true}', encoding="utf-8")
    assert FilesystemIncrementalJournal(index_dir).load(build_id) is None


def test_journal_resume_after_sparse_mutation_preserves_active_query_state(tmp_path, monkeypatch):
    """A restart resumes the same staged build after its recorded sparse boundary."""
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService
    from obsidian_wiki.infrastructure.filesystem_incremental_journal import FilesystemIncrementalJournal

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=_chunks(text="old restart payload"))
    old_pointer = (index_dir / "ACTIVE_INDEX").read_bytes()
    old_rows = _active_rows(index_dir)
    changed = _chunks(text="new restart payload")
    real_checkpoint = FilesystemIncrementalJournal.checkpoint
    failed = False

    def _interrupt_after_sparse(self, build_id, *, boundary):
        nonlocal failed
        result = real_checkpoint(self, build_id, boundary=boundary)
        if boundary == "sparse_mutated" and not failed:
            failed = True
            raise KeyboardInterrupt("interrupt after sparse mutation")
        return result

    monkeypatch.setattr(FilesystemIncrementalJournal, "checkpoint", _interrupt_after_sparse)
    with pytest.raises(KeyboardInterrupt, match="interrupt after sparse mutation"):
        IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=changed, embed=_embed)
    assert (index_dir / "ACTIVE_INDEX").read_bytes() == old_pointer
    assert _active_rows(index_dir) == old_rows
    record = FilesystemIncrementalJournal(index_dir).nonterminal()[0]
    assert record.last_completed_boundary == "sparse_mutated"

    monkeypatch.setattr(FilesystemIncrementalJournal, "checkpoint", real_checkpoint)
    result = IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=changed, embed=_embed)
    assert result.artifact.build_id == record.build_id
    assert (index_dir / "ACTIVE_INDEX").read_bytes() != old_pointer
    assert any(row["text"] == "new restart payload" for row in _active_rows(index_dir))
    assert FilesystemIncrementalJournal(index_dir).load(record.build_id).state.value == "published"


def test_journal_identity_mismatch_aborts_candidate_and_requires_snapshot(tmp_path, monkeypatch):
    """A changed canonical plan cannot resume foreign staged state or alter ACTIVE_INDEX."""
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService
    from obsidian_wiki.infrastructure.filesystem_incremental_journal import FilesystemIncrementalJournal

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=_chunks(text="identity source payload"))
    old_pointer = (index_dir / "ACTIVE_INDEX").read_bytes()
    real_checkpoint = FilesystemIncrementalJournal.checkpoint

    def _crash_after_checkpoint(self, build_id, *, boundary):
        real_checkpoint(self, build_id, boundary=boundary)
        raise KeyboardInterrupt("simulated process exit")

    monkeypatch.setattr(FilesystemIncrementalJournal, "checkpoint", _crash_after_checkpoint)
    with pytest.raises(KeyboardInterrupt, match="process exit"):
        IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=_chunks(text="first candidate"), embed=_embed)
    pending = FilesystemIncrementalJournal(index_dir).nonterminal()[0]
    monkeypatch.setattr(FilesystemIncrementalJournal, "checkpoint", real_checkpoint)

    with pytest.raises(RuntimeError, match="snapshot required"):
        IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=_chunks(text="different candidate"), embed=_embed)
    assert (index_dir / "ACTIVE_INDEX").read_bytes() == old_pointer
    assert FilesystemIncrementalJournal(index_dir).load(pending.build_id).state.value == "aborted"


@pytest.mark.parametrize("seam", ["clone", "sparse", "dense", "catch_up", "validation", "manifest"])
def test_fault_before_pointer_preserves_old_active_generation(tmp_path, monkeypatch, seam):
    """Every pre-pointer production seam aborts staging without exposing mixed tables."""
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService
    import obsidian_wiki.application.incremental_index_service as service_module
    from obsidian_wiki.infrastructure.filesystem_incremental_journal import FilesystemIncrementalJournal
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=_chunks(text="old fault payload"))
    old_pointer = (index_dir / "ACTIVE_INDEX").read_bytes()
    old_rows = _active_rows(index_dir)

    if seam == "clone":
        monkeypatch.setattr(LanceDbIndexRepository, "clone_tables", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("clone fault")))
    elif seam in {"sparse", "dense"}:
        real_apply = LanceDbIndexRepository.apply_delta
        monkeypatch.setattr(
            LanceDbIndexRepository, "apply_delta",
            lambda self, table_name, **kwargs: (_ for _ in ()).throw(OSError(f"{seam} fault"))
            if table_name == f"{seam}_chunks" else real_apply(self, table_name, **kwargs),
        )
    elif seam == "catch_up":
        monkeypatch.setattr(LanceDbIndexRepository, "catch_up", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("catch_up fault")))
    elif seam == "validation":
        monkeypatch.setattr(LanceDbIndexRepository, "validate_reopened", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("validation fault")))
    else:
        from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest
        monkeypatch.setattr(FilesystemIndexManifest, "write", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("manifest fault")))

    with pytest.raises(OSError, match="fault"):
        IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=_chunks(text="new fault payload"), embed=_embed)
    assert (index_dir / "ACTIVE_INDEX").read_bytes() == old_pointer
    assert _active_rows(index_dir) == old_rows
    assert FilesystemIncrementalJournal(index_dir).nonterminal() == ()


def test_zero_coverage_blocks_validated_and_pointer_publication(tmp_path, monkeypatch):
    """Unavailable/zero FTS coverage is never promoted to VALIDATED or PUBLISHED."""
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService
    from obsidian_wiki.domain.incremental_models import CoverageObservation
    from obsidian_wiki.infrastructure.filesystem_incremental_journal import FilesystemIncrementalJournal
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=_chunks(text="coverage source payload"))
    old_pointer = (index_dir / "ACTIVE_INDEX").read_bytes()
    monkeypatch.setattr(
        LanceDbIndexRepository, "catch_up",
        lambda *_args, **_kwargs: CoverageObservation("sparse_chunks", 1, 0, 0),
    )

    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=_chunks(text="coverage candidate"), embed=_embed)
    assert (index_dir / "ACTIVE_INDEX").read_bytes() == old_pointer
    assert FilesystemIncrementalJournal(index_dir).nonterminal() == ()


def test_commit_uncertainty_reconciles_published_journal_on_restart(tmp_path, monkeypatch):
    """A pointer replace that may have committed is reconciled, never called rollback."""
    from obsidian_wiki.application.durable_filesystem import CommitUncertainError
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService
    import obsidian_wiki.application.incremental_index_service as service_module
    from obsidian_wiki.infrastructure.filesystem_incremental_journal import FilesystemIncrementalJournal

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=_chunks(text="old uncertain payload"))
    real_publish = service_module.publish_pointer

    def _uncertain_after_replace(*args, **kwargs):
        real_publish(*args, **kwargs)
        raise CommitUncertainError("simulated post-replace uncertainty")

    monkeypatch.setattr(service_module, "publish_pointer", _uncertain_after_replace)
    with pytest.raises(CommitUncertainError, match="post-replace uncertainty"):
        IncrementalIndexService().build(wiki_dir, index_dir, canonical_chunks=_chunks(text="new uncertain payload"), embed=_embed)
    journal = FilesystemIncrementalJournal(index_dir)
    pending = journal.nonterminal()[0]
    assert pending.state.value == "validated"

    monkeypatch.setattr(service_module, "publish_pointer", real_publish)
    assert IncrementalIndexService().recover(index_dir) == (pending.build_id,)
    assert journal.load(pending.build_id).state.value == "published"


def test_lineage_retention_guard_blocks_source_cleanup_after_real_shallow_clone(tmp_path):
    """A shallow descendant keeps its source generation reopenable and queryable."""
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    build_storage_contract(wiki_dir, index_dir, embed=_embed, sparse_chunks=_chunks(text="lineage source payload"))
    source_build_id = str(_pointer(index_dir)["build_id"])
    source_lance = index_dir / str(_pointer(index_dir)["active_lance"])
    result = IncrementalIndexService().build(
        wiki_dir, index_dir, canonical_chunks=_chunks(text="lineage descendant payload"), embed=_embed,
    )

    service = IncrementalIndexService()
    assert source_build_id in service.required_ancestor_build_ids(index_dir)
    with pytest.raises(RuntimeError, match="lineage retention guard"):
        service.assert_cleanup_allowed(index_dir, source_build_id, probe_verified=True)
    with pytest.raises(RuntimeError, match="probe evidence is missing"):
        service.assert_cleanup_allowed(index_dir, "unrelated", probe_verified=False)
    assert LanceDbIndexRepository(source_lance).context_rows("page_id = 'concepts/online.md'")
    assert LanceDbIndexRepository(result.artifact.lance_dir).context_rows("page_id = 'concepts/online.md'")
