"""Candidate-neutral Phase 07 frozen corpus preparation and private ANN clones.

The module deliberately sits beside the evaluation orchestration.  It is not a
new production build mode: normal ``WikiIndex.build`` continues to create and
publish a complete active index.  This path seals only the reusable 30k source
boundary (two Lance tables, FTS, graph and page identities), then requires every
ANN role to create a separate writable Lance clone.
"""
from __future__ import annotations

import hashlib
import datetime as dt
import json
import math
import os
import re
import shutil
import stat
import struct
import tarfile
import unicodedata
import argparse
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from build_index import SKILL_EMBEDDER_DIR, page_id_of, scan_wiki
from chunk_plan import chunk_records_to_sparse
from chunking import EmbeddingTokenizer, chunk_page
from lexical_tokenizer import load_lexicon
from obsidian_wiki.domain.index_models import (
    CandidateQueryPolicy,
    DenseChunk,
    FtsIndexConfig,
    VectorIndexConfig,
)
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository
from obsidian_wiki.application.active_index_pointer import (
    publish_pointer,
    record_building,
    record_validated,
)
from obsidian_wiki.application.durable_filesystem import CommitUncertainError
from obsidian_wiki.application.build_lock import new_build_context
from obsidian_wiki.domain.index_models import INDEX_LAYOUT_VERSION, INDEX_MANIFEST_FORMAT_VERSION
from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest


SCHEMA_VERSION = 1
FROZEN_TARGET_SIZE = 30_000
_TOP_LEVEL = frozenset({"Wiki", "lance_db", ".index", "graph.json", "pages.json", "frozen-base.json"})
_DESCRIPTOR_FIELDS = frozenset({
    "schema_version", "kind", "authorization", "resolved_wiki_root", "pages_sha256",
    "graph_sha256", "source_tree_sha256", "lance_tree_sha256", "frozen_tree_sha256",
    "target_size", "corpus_manifest_sha256", "generator_recipe_sha256",
    "model_manifest_sha256", "runtime", "fts_config", "fts_stats",
    "expected_corpus_identity", "canonical_chunk_plan_sha256", "dense_vectors_sha256",
    "frozen_file_inventory", "record_self_sha256",
})
_FORBIDDEN_NAMES = frozenset({"ACTIVE_INDEX", "manifest.json", ".failed"})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_FROZEN_PAGE_FIELDS = frozenset({"page_id", "path", "title", "page_type", "sources", "links", "aliases", "sha256"})
_EXPECTED_CORPUS_IDENTITY_FIELDS = frozenset({"expanded_content_tree_sha256", "expanded_member_count"})
FROZEN_FTS_CONFIG = {
    "column": "fts_text", "base_tokenizer": "whitespace", "lower_case": False,
    "stem": False, "remove_stop_words": False, "ascii_folding": False,
    "max_token_length": 256,
}
FROZEN_PREPARE_IDENTITY_FIELDS = frozenset({
    "repository", "head_sha", "run_id", "run_attempt", "job_id", "artifact_id",
    "artifact_name", "archive_sha256", "archive_size_bytes", "descriptor_self_sha256",
    "base_tree_sha256", "model_manifest_sha256", "corpus_manifest_sha256",
    "generator_recipe_sha256", "runtime", "artifact_created_at", "artifact_expires_at",
    "retention_days", "replacement_for_run_id", "status",
})


class FrozenBaseError(ValueError):
    """Raised before an untrusted frozen source can reach a private clone."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_utc_timestamp(value: object, *, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise FrozenBaseError(label)
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FrozenBaseError(label) from exc
    if parsed.tzinfo is None:
        raise FrozenBaseError(label)
    return parsed.astimezone(dt.timezone.utc)


def validate_frozen_prepare_identity_shape(
    identity: object, *, expected_repository: str, expected_head: str,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Validate the shared frozen-prepare collector shape, not API authenticity.

    The operator collector owns API retrieval.  This deliberately only makes
    its result safe to share with prepare/role code: no optional fields, future
    evidence, replacement, or second attempt can be smuggled through.
    """
    if not isinstance(identity, dict) or set(identity) != FROZEN_PREPARE_IDENTITY_FIELDS:
        raise FrozenBaseError("strict frozen prepare identity")
    if (not isinstance(expected_repository, str) or not expected_repository
            or not _GIT_SHA.fullmatch(expected_head)
            or identity.get("repository") != expected_repository
            or identity.get("head_sha") != expected_head
            or identity.get("run_attempt") != 1
            or identity.get("retention_days") != 90
            or identity.get("replacement_for_run_id") is not None
            or identity.get("status") != "success"):
        raise FrozenBaseError("frozen prepare identity binding")
    for name in ("run_id", "job_id", "artifact_id", "archive_size_bytes"):
        value = identity.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise FrozenBaseError("frozen prepare API identity")
    run_id = identity["run_id"]
    if identity.get("artifact_name") != f"phase07-frozen-base-{run_id}-1":
        raise FrozenBaseError("frozen prepare artifact name")
    for name in (
        "archive_sha256", "descriptor_self_sha256", "base_tree_sha256",
        "model_manifest_sha256", "corpus_manifest_sha256", "generator_recipe_sha256",
    ):
        if not isinstance(identity.get(name), str) or not _HEX64.fullmatch(identity[name]):
            raise FrozenBaseError("frozen prepare digest identity")
    runtime = identity.get("runtime")
    if not isinstance(runtime, dict) or not runtime:
        raise FrozenBaseError("frozen prepare runtime identity")
    try:
        _canonical_json(runtime)
    except (TypeError, ValueError) as exc:
        raise FrozenBaseError("frozen prepare runtime identity") from exc
    created = _parse_utc_timestamp(identity.get("artifact_created_at"), label="prepare artifact creation")
    expires = _parse_utc_timestamp(identity.get("artifact_expires_at"), label="prepare artifact expiry")
    present = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    if created > present or present >= expires or created >= expires:
        raise FrozenBaseError("frozen prepare artifact time")
    if not dt.timedelta(days=89, hours=23, minutes=59, seconds=30) <= expires - created <= dt.timedelta(days=90, seconds=30):
        raise FrozenBaseError("frozen prepare artifact retention")
    return identity


def _canonical_relative(raw: str) -> str:
    """Return one safe POSIX spelling; aliases are rejected rather than collapsed."""
    if not isinstance(raw, str) or not raw or "\\" in raw or raw.startswith("/"):
        raise FrozenBaseError("archive member")
    pure = PurePosixPath(raw)
    canonical = pure.as_posix()
    if raw != canonical or canonical in {".", ""} or ".." in pure.parts:
        raise FrozenBaseError("archive member")
    return canonical


def _tree_inventory(root: Path, *, exclude: frozenset[str] = frozenset()) -> tuple[list[dict[str, object]], str]:
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise FrozenBaseError("frozen tree root")
    inventory: list[dict[str, object]] = []
    folded: set[str] = set()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = _canonical_relative(path.relative_to(root).as_posix())
        if relative in exclude:
            continue
        normalized = unicodedata.normalize("NFC", relative)
        folded_name = normalized.casefold()
        if normalized != relative or folded_name in folded:
            raise FrozenBaseError("normalized/case-fold collision")
        folded.add(folded_name)
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise FrozenBaseError("frozen tree special member")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise FrozenBaseError("frozen tree special member")
        if path.stat().st_nlink != 1:
            raise FrozenBaseError("frozen tree hardlink member")
        inventory.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256_file(path)})
    return inventory, _sha256_bytes(_canonical_json(inventory))


def _graph_payload(wiki_dir: Path) -> dict[str, object]:
    import build_graph

    graph = build_graph.build_graph(wiki_dir)
    stats = build_graph.compute_4_signals(graph)
    communities = build_graph.detect_communities(graph)
    return {
        "nodes": [
            {"id": node, **{key: value for key, value in data.items() if key != "signals"}}
            for node, data in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source, "target": target, "weight": round(data.get("weight", 1.0), 4),
                "signal": sorted(data.get("signals", set()))[0] if data.get("signals") else "unknown",
                "signals": sorted(data.get("signals", set())),
            }
            for source, target, data in graph.edges(data=True)
        ],
        "signals": stats,
        "communities": communities,
    }


def _exact_term(chunks: list[object]) -> str:
    for chunk in chunks:
        for token in str(getattr(chunk, "fts_text", "")).split():
            if token:
                return token
    raise FrozenBaseError("frozen FTS term")


def _canonical_markdown_plan(wiki_dir: Path, *, tokenizer: object | None) -> tuple[list[object], list[object], list[dict[str, object]]]:
    """Rebuild the source-derived chunk plan without reading either Lance table."""
    wiki_dir = Path(wiki_dir)
    project_root = wiki_dir.parent
    pages = scan_wiki(wiki_dir, project_root)
    if not pages:
        raise FrozenBaseError("frozen corpus has no pages")
    lexicon = load_lexicon(project_root)
    token_counter = EmbeddingTokenizer(tokenizer).count if tokenizer is not None else None
    canonical = []
    page_rows = []
    for page in pages:
        page_id = page_id_of(page.path)
        records = list(chunk_page(
            page_id=page_id, path=page.path, title=page.title, page_type=page.page_type,
            content=page.content, tokenizer=token_counter,
        ))
        if page.content.strip() and {record.chunk_kind for record in records} != {"sparse", "dense"}:
            raise FrozenBaseError("canonical frozen chunk plan")
        canonical.extend(chunk_records_to_sparse(records, lexicon))
        page_rows.append({
            "page_id": page_id, "path": str(page.path), "title": page.title,
            "page_type": page.page_type, "sources": page.sources, "links": page.links,
            "aliases": page.aliases, "sha256": page.sha256,
        })
    dense_sources = [chunk for chunk in canonical if chunk.chunk_kind == "dense"]
    return [chunk for chunk in canonical if chunk.chunk_kind == "sparse"], dense_sources, page_rows


def _dense_from_canonical(chunk: object, vector: tuple[float, ...]) -> DenseChunk:
    return DenseChunk(
        chunk_id=chunk.chunk_id, page_id=chunk.page_id, path=chunk.path, title=chunk.title,
        text=chunk.text, vector=vector, page_type=chunk.page_type,
        section_path=chunk.section_path, heading=chunk.heading, chunk_kind=chunk.chunk_kind,
        chunk_index=chunk.chunk_index, parent_section_id=chunk.parent_section_id,
        token_count=chunk.token_count, content_hash=chunk.content_hash,
        forced_split=chunk.forced_split, continuation_index=chunk.continuation_index,
        start_char=chunk.start_char, end_char=chunk.end_char,
    )


def _make_chunks(wiki_dir: Path, embed: Callable[[list[str]], list[list[float]]], *, tokenizer: object) -> tuple[list[object], list[DenseChunk], list[dict[str, object]]]:
    sparse, dense_sources, page_rows = _canonical_markdown_plan(wiki_dir, tokenizer=tokenizer)
    vectors = embed([chunk.text for chunk in dense_sources])
    if len(vectors) != len(dense_sources):
        raise FrozenBaseError("frozen embedding count")
    dense = [_dense_from_canonical(chunk, tuple(float(value) for value in vector))
             for chunk, vector in zip(dense_sources, vectors)]
    if not dense:
        raise FrozenBaseError("frozen corpus has no dense chunks")
    return sparse, dense, page_rows


def _default_expected_corpus_identity() -> dict[str, object]:
    """The committed 30k generator is the production authority, never a descriptor label."""
    from eval.run_eval import FIXTURES_WIKI, expected_phase07_expanded_corpus_identity

    return expected_phase07_expanded_corpus_identity(
        fixture_root=FIXTURES_WIKI, target_size=FROZEN_TARGET_SIZE,
    )


def _expected_corpus_identity(value: Mapping[str, object] | None, *, allow_explicit: bool = False) -> dict[str, object]:
    """Bind a descriptor to the committed recipe, except an explicit test seam."""
    authority = dict(_default_expected_corpus_identity())
    identity = authority if value is None else dict(value)
    if set(identity) != _EXPECTED_CORPUS_IDENTITY_FIELDS \
            or not isinstance(identity.get("expanded_content_tree_sha256"), str) \
            or not _HEX64.fullmatch(str(identity["expanded_content_tree_sha256"])) \
            or isinstance(identity.get("expanded_member_count"), bool) \
            or not isinstance(identity.get("expanded_member_count"), int) \
            or identity["expanded_member_count"] <= 0:
        raise FrozenBaseError("frozen corpus identity")
    if authority["expanded_member_count"] != FROZEN_TARGET_SIZE:
        raise FrozenBaseError("frozen corpus authority")
    if identity != authority and not allow_explicit:
        raise FrozenBaseError("frozen corpus identity")
    return identity


def _actual_corpus_identity(wiki_dir: Path) -> dict[str, object]:
    from eval.ann_corpus_manifest import canonical_content_tree_sha256

    wiki_dir = Path(wiki_dir)
    markdown = [path for path in wiki_dir.rglob("*.md") if path.is_file()]
    return {
        "expanded_content_tree_sha256": canonical_content_tree_sha256(wiki_dir),
        "expanded_member_count": len(markdown),
    }


def _canonical_non_vector_rows(rows: list[object] | tuple[Mapping[str, object], ...]) -> list[dict[str, object]]:
    """Normalize every persisted non-vector field for source-plan comparison."""
    payload = []
    for row in rows:
        raw = dict(row) if isinstance(row, Mapping) else dict(vars(row))
        raw.pop("vector", None)
        payload.append(raw)
    return sorted(payload, key=lambda row: str(row["chunk_id"]))


def _canonical_chunk_plan_sha256(sparse: list[object] | tuple[Mapping[str, object], ...],
                                 dense: list[object] | tuple[Mapping[str, object], ...]) -> str:
    """Bind every canonical chunk field except the separately sealed float32 bytes."""
    payload = {
        "sparse": _canonical_non_vector_rows(sparse),
        "dense": _canonical_non_vector_rows(dense),
    }
    return _sha256_bytes(_canonical_json(payload))


def _dense_vectors_sha256(dense: list[object] | tuple[Mapping[str, object], ...]) -> str:
    """Hash canonical chunk IDs and the exact persisted little-endian float32 vector bytes."""
    digest = hashlib.sha256()
    for row in sorted(dense, key=lambda item: str((dict(item) if isinstance(item, Mapping) else vars(item))["chunk_id"])):
        raw = dict(row) if isinstance(row, Mapping) else vars(row)
        vector = raw.get("vector")
        if not isinstance(vector, (list, tuple)) or len(vector) != 384:
            raise FrozenBaseError("frozen dense vector identity")
        digest.update(str(raw.get("chunk_id", "")).encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(struct.pack("<384f", *(float(value) for value in vector)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise FrozenBaseError("frozen dense vector identity") from exc
    return digest.hexdigest()


def _page_rows(wiki_dir: Path) -> list[dict[str, object]]:
    wiki_dir = Path(wiki_dir)
    return [{
        "page_id": page_id_of(page.path), "path": str(page.path), "title": page.title,
        "page_type": page.page_type, "sources": page.sources, "links": page.links,
        "aliases": page.aliases, "sha256": page.sha256,
    } for page in scan_wiki(wiki_dir, wiki_dir.parent)]


def _default_descriptor_identity() -> dict[str, object]:
    """The exact committed identities used by local/tiny storage seams."""
    from eval.ann_corpus_manifest import load_manifest, public_distractor_recipe_sha256

    here = Path(__file__).resolve().parent
    model_path = here / "model-manifest.json"
    return {
        "target_size": FROZEN_TARGET_SIZE,
        "corpus_manifest_sha256": _sha256_file(here / "personal-wiki-corpus-manifest.json"),
        "generator_recipe_sha256": public_distractor_recipe_sha256(),
        "model_manifest_sha256": _sha256_file(model_path),
        "runtime": dict(load_manifest(model_path)["runtime"]),
    }


def _descriptor_identity(value: Mapping[str, object] | None, *,
                         expected_corpus_identity: Mapping[str, object] | None = None) -> dict[str, object]:
    authority = _default_descriptor_identity()
    identity = authority if value is None else dict(value)
    required = {"target_size", "corpus_manifest_sha256", "generator_recipe_sha256", "model_manifest_sha256", "runtime"}
    if set(identity) != required or identity.get("target_size") != FROZEN_TARGET_SIZE:
        raise FrozenBaseError("frozen descriptor identity")
    for key in required - {"target_size", "runtime"}:
        if not isinstance(identity.get(key), str) or not _HEX64.fullmatch(str(identity[key])):
            raise FrozenBaseError("frozen descriptor identity")
    if not isinstance(identity.get("runtime"), dict) or not identity["runtime"]:
        raise FrozenBaseError("frozen descriptor runtime")
    try:
        _canonical_json(identity["runtime"])
    except (TypeError, ValueError) as exc:
        raise FrozenBaseError("frozen descriptor runtime") from exc
    if identity != authority:
        raise FrozenBaseError("frozen descriptor identity")
    identity["expected_corpus_identity"] = _expected_corpus_identity(
        expected_corpus_identity, allow_explicit=expected_corpus_identity is not None,
    )
    return identity


def prepare_frozen_base(*, wiki_dir: Path, frozen_dir: Path, embed: Callable[[list[str]], list[list[float]]],
                        tokenizer: object = None, descriptor_identity: Mapping[str, object] | None = None,
                        expected_corpus_identity: Mapping[str, object] | None = None) -> dict[str, object]:
    """Persist reusable data once, deliberately stopping before HNSW/publication."""
    wiki_dir, frozen_dir = Path(wiki_dir).resolve(), Path(frozen_dir).resolve()
    identity = _descriptor_identity(
        descriptor_identity, expected_corpus_identity=expected_corpus_identity,
    )
    canonical_wiki = frozen_dir / "Wiki"
    if frozen_dir.exists():
        # The production corpus is materialized directly at its final artifact
        # root.  It is safe only while that root contains the sole source tree.
        if wiki_dir != canonical_wiki.resolve() or not canonical_wiki.is_dir() \
                or canonical_wiki.is_symlink() or any(path.name != "Wiki" for path in frozen_dir.iterdir()):
            raise FrozenBaseError("frozen target must be empty canonical root")
    else:
        frozen_dir.mkdir(parents=True)
        shutil.copytree(wiki_dir, canonical_wiki)
    # The tables and page IDs must be created from the exact tree which is
    # sealed into the archive.  The canonical final root is never relocated.
    wiki_dir = canonical_wiki.resolve()
    _validate_wiki_semantics(wiki_dir)
    if _actual_corpus_identity(wiki_dir) != identity["expected_corpus_identity"]:
        raise FrozenBaseError("frozen corpus identity")
    sparse, dense, pages = _make_chunks(wiki_dir, embed, tokenizer=tokenizer)
    if pages != _page_rows(wiki_dir):
        raise FrozenBaseError("canonical frozen page metadata")
    pages_path = frozen_dir / "pages.json"
    pages_path.write_bytes(_canonical_json(pages))
    graph_bytes = _canonical_json(_graph_payload(wiki_dir))
    (frozen_dir / "graph.json").write_bytes(graph_bytes)
    # Public hybrid search resolves graph state from ``Wiki/../.index``.  Copy
    # the already-built sealed graph there instead of rebuilding it per role.
    (frozen_dir / ".index").mkdir()
    (frozen_dir / ".index" / "graph.json").write_bytes(graph_bytes)
    repository = LanceDbIndexRepository(frozen_dir / "lance_db")
    repository.persist(frozen_dir / "lance_db", sparse, dense, FtsIndexConfig())
    repository.validate_reopened(
        dimension=len(dense[0].vector), exact_term=_exact_term(sparse), vector_index_name=None,
    )
    if repository._dense_table().list_indices():  # Lance enumeration is the primary no-HNSW assertion.
        raise FrozenBaseError("frozen dense vector index")
    repository.seal(frozen_dir / "lance_db")
    source_tree = _tree_inventory(frozen_dir / "Wiki")[1]
    lance_tree = _tree_inventory(frozen_dir / "lance_db")[1]
    inventory, frozen_tree = _tree_inventory(frozen_dir, exclude=frozenset({"frozen-base.json"}))
    descriptor: dict[str, object] = {
        "schema_version": SCHEMA_VERSION, "kind": "phase07-frozen-base", "authorization": "none",
        "resolved_wiki_root": str(wiki_dir), "pages_sha256": _sha256_file(pages_path),
        "graph_sha256": _sha256_file(frozen_dir / "graph.json"), "source_tree_sha256": source_tree,
        "lance_tree_sha256": lance_tree, "frozen_tree_sha256": frozen_tree,
        "target_size": identity["target_size"], "corpus_manifest_sha256": identity["corpus_manifest_sha256"],
        "generator_recipe_sha256": identity["generator_recipe_sha256"],
        "model_manifest_sha256": identity["model_manifest_sha256"], "runtime": identity["runtime"],
        "expected_corpus_identity": identity["expected_corpus_identity"],
        "canonical_chunk_plan_sha256": _canonical_chunk_plan_sha256(sparse, dense),
        "dense_vectors_sha256": _dense_vectors_sha256(dense),
        "fts_config": dict(FROZEN_FTS_CONFIG),
        "fts_stats": {"index_name": "fts_text_idx", "indexed_rows": len(sparse), "unindexed_rows": 0},
        "frozen_file_inventory": inventory,
    }
    descriptor["record_self_sha256"] = _sha256_bytes(_canonical_json(descriptor))
    (frozen_dir / "frozen-base.json").write_bytes(_canonical_json(descriptor))
    return descriptor


def _read_descriptor(frozen_dir: Path, *, expected_corpus_identity: Mapping[str, object] | None = None) -> dict[str, object]:
    try:
        descriptor = json.loads((Path(frozen_dir) / "frozen-base.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenBaseError("frozen descriptor") from exc
    if not isinstance(descriptor, dict) or set(descriptor) != _DESCRIPTOR_FIELDS:
        raise FrozenBaseError("frozen descriptor allowlist")
    sealed = {key: value for key, value in descriptor.items() if key != "record_self_sha256"}
    if descriptor.get("schema_version") != SCHEMA_VERSION or descriptor.get("kind") != "phase07-frozen-base" \
            or descriptor.get("authorization") != "none" or descriptor.get("record_self_sha256") != _sha256_bytes(_canonical_json(sealed)):
        raise FrozenBaseError("frozen descriptor identity")
    identity = _descriptor_identity({key: descriptor[key] for key in (
        "target_size", "corpus_manifest_sha256", "generator_recipe_sha256", "model_manifest_sha256", "runtime",
    )}, expected_corpus_identity=expected_corpus_identity)
    if descriptor.get("expected_corpus_identity") != identity["expected_corpus_identity"]:
        raise FrozenBaseError("frozen corpus identity")
    for name in ("canonical_chunk_plan_sha256", "dense_vectors_sha256"):
        if not isinstance(descriptor.get(name), str) or not _HEX64.fullmatch(str(descriptor[name])):
            raise FrozenBaseError("frozen descriptor identity")
    if descriptor.get("fts_config") != FROZEN_FTS_CONFIG:
        raise FrozenBaseError("frozen FTS config")
    stats = descriptor.get("fts_stats")
    if not isinstance(stats, dict) or set(stats) != {"index_name", "indexed_rows", "unindexed_rows"} \
            or stats.get("index_name") != "fts_text_idx" \
            or not isinstance(stats.get("indexed_rows"), int) or stats["indexed_rows"] <= 0 \
            or stats.get("unindexed_rows") != 0:
        raise FrozenBaseError("frozen FTS stats")
    inventory = descriptor.get("frozen_file_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise FrozenBaseError("frozen file inventory")
    for entry in inventory:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"} \
                or _canonical_relative(str(entry.get("path", ""))) != entry.get("path") \
                or not isinstance(entry.get("size"), int) or entry["size"] < 0 \
                or not isinstance(entry.get("sha256"), str) or not _HEX64.fullmatch(entry["sha256"]):
            raise FrozenBaseError("frozen file inventory")
    return descriptor


def _validate_wiki_semantics(wiki_dir: Path) -> None:
    """A frozen Wiki is source Markdown only, never a nested decision packet."""
    root = Path(wiki_dir)
    if not root.is_dir() or root.is_symlink():
        raise FrozenBaseError("frozen Wiki layout")
    markdown: list[Path] = []
    forbidden = {name.casefold() for name in _FORBIDDEN_NAMES} | {
        "active", "candidate", "policy", "verdict", "authorization",
    }
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part.casefold() in forbidden or any(token in part.casefold() for token in (
                "candidate", "policy", "verdict", "authorization")) for part in relative.parts):
            raise FrozenBaseError("frozen Wiki sidecar")
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise FrozenBaseError("frozen Wiki layout")
        if path.is_file():
            if path.suffix.lower() != ".md":
                raise FrozenBaseError("frozen Wiki sidecar")
            markdown.append(path)
    if not markdown:
        raise FrozenBaseError("frozen corpus has no pages")


def _validate_frozen_semantic_layout(frozen_dir: Path) -> None:
    """Reject any policy/evidence sidecar before opening Lance or cloning it."""
    root = Path(frozen_dir)
    if {path.name for path in root.iterdir()} != _TOP_LEVEL:
        raise FrozenBaseError("frozen top-level allowlist")
    _validate_wiki_semantics(root / "Wiki")
    index_files = [path.relative_to(root / ".index").as_posix() for path in (root / ".index").rglob("*") if path.is_file()]
    if index_files != ["graph.json"]:
        raise FrozenBaseError("frozen index sidecar")
    lance = root / "lance_db"
    if not lance.is_dir() or lance.is_symlink() \
            or {path.name for path in lance.iterdir()} != {"sparse_chunks.lance", "dense_chunks.lance", "__manifest"}:
        raise FrozenBaseError("frozen Lance layout")
    for path in lance.rglob("*"):
        relative = path.relative_to(lance).as_posix()
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise FrozenBaseError("frozen Lance layout")
        if any(token in part.casefold() for part in Path(relative).parts for token in (
                "candidate", "policy", "verdict", "authorization", "active")):
            raise FrozenBaseError("frozen Lance sidecar")
        if path.is_file() and path.suffix.lower() == ".json" \
                and not relative.endswith("/_versions/latest_version_hint.json"):
            raise FrozenBaseError("frozen Lance sidecar")


def validate_frozen_base(frozen_dir: Path, *, expected_wiki_root: Path,
                         expected_corpus_identity: Mapping[str, object] | None = None) -> str:
    """Validate data, FTS, graph/page identities and the fixed absolute-root contract."""
    frozen_dir = Path(frozen_dir)
    descriptor = _read_descriptor(frozen_dir, expected_corpus_identity=expected_corpus_identity)
    _validate_frozen_semantic_layout(frozen_dir)
    expected = Path(expected_wiki_root).resolve()
    if expected != (frozen_dir / "Wiki").resolve() or descriptor["resolved_wiki_root"] != str(expected):
        raise FrozenBaseError("frozen resolved root")
    if _actual_corpus_identity(frozen_dir / "Wiki") != descriptor["expected_corpus_identity"]:
        raise FrozenBaseError("frozen corpus identity")
    if _tree_inventory(frozen_dir / "Wiki")[1] != descriptor["source_tree_sha256"] \
            or _tree_inventory(frozen_dir / "lance_db")[1] != descriptor["lance_tree_sha256"]:
        raise FrozenBaseError("frozen tree digest")
    inventory, frozen_tree = _tree_inventory(frozen_dir, exclude=frozenset({"frozen-base.json"}))
    if inventory != descriptor["frozen_file_inventory"] or frozen_tree != descriptor["frozen_tree_sha256"]:
        raise FrozenBaseError("frozen tree digest")
    if _sha256_file(frozen_dir / "pages.json") != descriptor["pages_sha256"] \
            or _sha256_file(frozen_dir / "graph.json") != descriptor["graph_sha256"]:
        raise FrozenBaseError("frozen sidecar digest")
    if _sha256_file(frozen_dir / ".index" / "graph.json") != descriptor["graph_sha256"]:
        raise FrozenBaseError("frozen public graph identity")
    try:
        pages = json.loads((frozen_dir / "pages.json").read_text(encoding="utf-8"))
        graph = json.loads((frozen_dir / "graph.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenBaseError("frozen sidecar") from exc
    if not isinstance(pages, list) or not isinstance(graph, dict):
        raise FrozenBaseError("frozen sidecar shape")
    page_ids = _validate_pages(pages, wiki_dir=frozen_dir / "Wiki")
    if _canonical_json(graph) != _canonical_json(_graph_payload(frozen_dir / "Wiki")):
        raise FrozenBaseError("frozen graph identity")
    repository = LanceDbIndexRepository(frozen_dir / "lance_db")
    sparse = repository.table_rows("sparse_chunks")
    dense = repository.table_rows("dense_chunks")
    _validate_table_rows(sparse, dense, page_ids=page_ids)
    expected_sparse, expected_dense_sources, expected_pages = _canonical_markdown_plan(
        frozen_dir / "Wiki", tokenizer=None,
    )
    expected_dense = [_dense_from_canonical(chunk, ()) for chunk in expected_dense_sources]
    if pages != expected_pages or _canonical_non_vector_rows(sparse) != _canonical_non_vector_rows(expected_sparse) \
            or _canonical_non_vector_rows(dense) != _canonical_non_vector_rows(expected_dense):
        raise FrozenBaseError("frozen canonical Markdown chunk plan")
    if _canonical_chunk_plan_sha256(sparse, dense) != descriptor["canonical_chunk_plan_sha256"] \
            or _dense_vectors_sha256(dense) != descriptor["dense_vectors_sha256"]:
        raise FrozenBaseError("frozen canonical table identity")
    if repository._dense_table().list_indices():
        raise FrozenBaseError("frozen dense vector index")
    fts_indices = {index.name for index in repository._sparse_table().list_indices()}
    if fts_indices != {"fts_text_idx"}:
        raise FrozenBaseError("frozen sparse FTS")
    if descriptor["fts_stats"] != {
        "index_name": "fts_text_idx", "indexed_rows": len(sparse), "unindexed_rows": 0,
    }:
        raise FrozenBaseError("frozen sparse FTS")
    repository.validate_reopened(
        dimension=len(dense[0]["vector"]), exact_term=_exact_term([type("R", (), row) for row in sparse]),
        vector_index_name=None,
    )
    return str(descriptor["frozen_tree_sha256"])


def _validate_pages(pages: list[object], *, wiki_dir: Path) -> set[str]:
    root = Path(wiki_dir).resolve()
    page_ids: set[str] = set()
    paths: set[str] = set()
    if not pages:
        raise FrozenBaseError("frozen page identity")
    if pages != _page_rows(root):
        raise FrozenBaseError("frozen page metadata identity")
    for page in pages:
        if not isinstance(page, dict) or set(page) != _FROZEN_PAGE_FIELDS:
            raise FrozenBaseError("frozen page schema")
        if not all(isinstance(page[name], str) for name in ("page_id", "path", "title", "page_type", "sha256")) \
                or not all(isinstance(page[name], list) and all(isinstance(item, str) for item in page[name])
                           for name in ("sources", "links", "aliases")) \
                or not _HEX64.fullmatch(page["sha256"]):
            raise FrozenBaseError("frozen page schema")
        path = Path(page["path"])
        if not path.is_absolute() or path.is_symlink() or path.suffix.lower() != ".md":
            raise FrozenBaseError("frozen page identity")
        resolved = path.resolve()
        if resolved != path or not resolved.is_relative_to(root) or page["page_id"] != str(resolved) \
                or not resolved.is_file() or _sha256_file(resolved) != page["sha256"]:
            raise FrozenBaseError("frozen page identity")
        if page["page_id"] in page_ids or page["path"] in paths:
            raise FrozenBaseError("frozen page identity")
        page_ids.add(page["page_id"])
        paths.add(page["path"])
    actual_paths = {
        str(path.resolve()) for path in root.rglob("*.md")
        if path.is_file() and not path.is_symlink()
    }
    if paths != actual_paths:
        raise FrozenBaseError("frozen Markdown page set")
    return page_ids


def _validate_table_rows(sparse: tuple[Mapping[str, object], ...], dense: tuple[Mapping[str, object], ...], *, page_ids: set[str]) -> None:
    if not sparse or not dense:
        raise FrozenBaseError("frozen table rows")
    all_ids: set[str] = set()
    table_pages: list[set[str]] = []
    dimension: int | None = None
    for expected_kind, rows in (("sparse", sparse), ("dense", dense)):
        current_pages: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping) or row.get("chunk_kind") != expected_kind \
                    or not isinstance(row.get("chunk_id"), str) or not row["chunk_id"] \
                    or row["chunk_id"] in all_ids or row.get("page_id") not in page_ids \
                    or row.get("path") != row.get("page_id"):
                raise FrozenBaseError("frozen table chunk identity")
            if expected_kind == "dense":
                vector = row.get("vector")
                if not isinstance(vector, (list, tuple)) or not vector \
                        or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
                    raise FrozenBaseError("frozen dense vector identity")
                if len(vector) != 384:
                    raise FrozenBaseError("frozen dense vector identity")
                if dimension is None:
                    dimension = len(vector)
                elif len(vector) != dimension:
                    raise FrozenBaseError("frozen dense vector identity")
            elif "vector" in row:
                raise FrozenBaseError("frozen sparse vector identity")
            all_ids.add(row["chunk_id"])
            current_pages.add(str(row["page_id"]))
        table_pages.append(current_pages)
    if table_pages[0] != page_ids or table_pages[1] != page_ids:
        raise FrozenBaseError("frozen table page identity")


def _candidate_manifest(*, pages: list[dict[str, object]], build_id: str, generation: int,
                        candidate_query_policy: CandidateQueryPolicy,
                        vector_config: VectorIndexConfig) -> dict[str, object]:
    """Construct the smallest normal load manifest for an already-validated clone.

    Frozen data deliberately has no manifest.  This manifest is created only
    after the private HNSW is reopened and validated, immediately before the
    existing lifecycle's ``record_validated`` / ``publish_pointer`` commit.
    """
    return {
        "format_version": INDEX_MANIFEST_FORMAT_VERSION,
        "index_layout_version": INDEX_LAYOUT_VERSION,
        "layout": "sparse_chunks+dense_chunks",
        "build_id": build_id,
        "generation": generation,
        "pages": pages,
        "candidate_query_policy": {
            "candidate": candidate_query_policy.candidate,
            "query_ef": candidate_query_policy.query_ef,
            "build_policy": {
                "candidate": candidate_query_policy.build_policy.candidate,
                "m": candidate_query_policy.build_policy.m,
                "ef_construction": candidate_query_policy.build_policy.ef_construction,
            },
        },
        "ann_policy": {
            "selected_index_type": candidate_query_policy.candidate,
            "query_ef": candidate_query_policy.query_ef,
        },
        "vector_config": {
            "index_type": vector_config.index_type,
            "metric": vector_config.metric,
            "num_partitions": vector_config.num_partitions,
            "m": vector_config.m,
            "ef_construction": vector_config.ef_construction,
            "dense_chunks_count": vector_config.dense_chunks_count,
            "index_name": vector_config.index_name,
        },
    }


def finalize_private_role(*, frozen_dir: Path, target_dir: Path, expected_wiki_root: Path,
                          candidate_query_policy: CandidateQueryPolicy,
                          publish_index_dir: Path | None = None,
                          expected_corpus_identity: Mapping[str, object] | None = None) -> dict[str, object]:
    """Clone validated tables and build exactly one candidate HNSW in that clone."""
    source_digest = validate_frozen_base(
        frozen_dir, expected_wiki_root=expected_wiki_root,
        expected_corpus_identity=expected_corpus_identity,
    )
    target_dir = Path(target_dir)
    if target_dir.exists():
        raise FrozenBaseError("private clone target must be new")
    index_dir = Path(publish_index_dir) if publish_index_dir is not None else None
    build_id: str | None = None
    generation: int | None = None
    build_dir: Path | None = None
    manifest_path = None
    lance_dir: Path | None = None
    policy = candidate_query_policy.build_policy
    if policy is None:
        raise FrozenBaseError("private role requires candidate build policy")
    pointer_committed = False
    failure_dir: Path | None = None
    try:
        target_dir.mkdir(parents=True)
        failure_dir = target_dir
        # A caller which needs the public WikiIndex path supplies an empty private
        # index root.  The clone then lives in a normal staged build directory; the
        # default retains the small direct-Lance seam used by storage tests.
        if index_dir is not None:
            if index_dir.exists() and any(index_dir.iterdir()):
                raise FrozenBaseError("private publication target must be new")
            index_dir.mkdir(parents=True, exist_ok=True)
            build_id = new_build_context().build_id
            generation = 1
            build_dir = index_dir / "builds" / build_id
            build_dir.mkdir(parents=True, exist_ok=False)
            failure_dir = build_dir
            # The full pre-pointer lifecycle is one failure window.  This
            # includes the first durable BUILDING record, clone, HNSW, reopen,
            # seal, manifest and VALIDATED transition.
            record_building(build_dir, build_id=build_id, generation=generation)
            lance_dir = build_dir / "lance_db"
        else:
            lance_dir = target_dir / "lance_db"
        source = LanceDbIndexRepository(Path(frozen_dir) / "lance_db")
        identities = source.clone_tables(lance_dir)
        shutil.copy2(Path(frozen_dir) / "pages.json", target_dir / "pages.json")
        shutil.copy2(Path(frozen_dir) / "graph.json", target_dir / "graph.json")
        dense_count = next(identity.row_count for identity in identities if identity.table_name == "dense_chunks")
        target = LanceDbIndexRepository(lance_dir, eval_candidate_policy=candidate_query_policy)
        config = VectorIndexConfig(
            index_type="hnsw_sq", metric="cosine", num_partitions=1, m=policy.m,
            ef_construction=policy.ef_construction, dense_chunks_count=dense_count,
        )
        target.create_eval_candidate_index(config)
        target.validate_reopened(
            dimension=len(target.table_rows("dense_chunks")[0]["vector"]),
            exact_term=_exact_term([type("R", (), row) for row in target.table_rows("sparse_chunks")]),
            vector_index_name=config.index_name,
        )
        target.seal(lance_dir)
        if validate_frozen_base(
            frozen_dir, expected_wiki_root=expected_wiki_root,
            expected_corpus_identity=expected_corpus_identity,
        ) != source_digest:
            raise FrozenBaseError("frozen source mutated by private clone")
        if build_dir is not None and index_dir is not None and build_id is not None and generation is not None:
            pages = json.loads((frozen_dir / "pages.json").read_text(encoding="utf-8"))
            if not isinstance(pages, list):
                raise FrozenBaseError("frozen page manifest")
            manifest = _candidate_manifest(
                pages=pages, build_id=build_id, generation=generation,
                candidate_query_policy=candidate_query_policy, vector_config=config,
            )
            manifest_path = build_dir / "manifest.json"
            # Use the same durable tmp/fsync/replace collaborator as a normal
            # build.  A frozen clone is only an alternate input, never an
            # alternate publication protocol.
            FilesystemIndexManifest().write(manifest_path, manifest)
            record_validated(
                build_dir, build_id=build_id, generation=generation,
                manifest_sha256=_sha256_file(manifest_path),
            )
            publish_pointer(index_dir, build_dir, build_id=build_id, generation=generation)
            pointer_committed = True
    except CommitUncertainError:
        # A durable replace may already have happened.  Never contradict an
        # uncertain/post-pointer state with a pre-publication marker.
        raise
    except Exception:
        if not pointer_committed and failure_dir is not None:
            try:
                (failure_dir / ".failed").write_text(
                    "frozen private clone failed before publication\n", encoding="utf-8",
                )
            except OSError:
                pass
        raise
    assert lance_dir is not None
    return {
        "source_tree_sha256": source_digest, "role_m": policy.m,
        "lance_dir": str(lance_dir), "index_dir": str(index_dir) if index_dir is not None else None,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
    }


def safe_extract_frozen_base(archive: Path, destination: Path) -> None:
    """Extract a tar archive only into a new empty root, rejecting links and traversal."""
    archive, destination = Path(archive), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FrozenBaseError("archive destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as handle:
        members = handle.getmembers()
        names: set[str] = set()
        for member in members:
            normalized = unicodedata.normalize("NFC", member.name)
            try:
                canonical = _canonical_relative(member.name)
            except FrozenBaseError:
                raise FrozenBaseError("archive member") from None
            if member.name != normalized or canonical != normalized or member.issym() or member.islnk() \
                    or not (member.isdir() or member.isfile()) or normalized.casefold() in names:
                raise FrozenBaseError("archive member")
            names.add(normalized.casefold())
        for member in members:
            handle.extract(member, destination, set_attrs=False, numeric_owner=False)
    # Treat an extracted artifact as hostile until its exact sealed inventory,
    # regular-file topology, and descriptor have all been revalidated.
    validate_frozen_base(destination, expected_wiki_root=destination / "Wiki")


def validate_frozen_role_provenance(records: list[dict[str, Any]], *, expected_head: str) -> list[dict[str, Any]]:
    """Accept only the complete, independent, first-attempt role set for one base."""
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("complete frozen role batch")
    expected = {("baseline", 16), ("m20", 20), ("m32", 32)}
    common_fields = {
        "prepare_run_id", "prepare_run_attempt", "prepare_job_id", "prepare_artifact_id",
        "prepare_archive_sha256", "prepare_descriptor_sha256", "prepare_tree_sha256",
        "retention_days", "head_sha", "runtime",
        "corpus_sha256", "model_manifest_sha256",
    }
    seen = set()
    run_ids: set[int] = set()
    job_ids: set[int] = set()
    artifact_ids: set[int] = set()
    common = None
    for record in records:
        if not isinstance(record, dict) or (record.get("role"), record.get("m")) not in expected:
            raise ValueError("frozen role identity")
        if record.get("run_attempt") != 1 or record.get("prepare_run_attempt") != 1 \
                or record.get("retention_days") != 90 or record.get("head_sha") != expected_head:
            raise ValueError("frozen attempt/head/retention")
        identity = (record.get("run_id"), record.get("job_id"), record.get("artifact_id"))
        if not all(isinstance(value, int) and value > 0 for value in identity) or identity in seen:
            raise ValueError("frozen roles must be distinct")
        seen.add(identity)
        run_ids.add(identity[0]); job_ids.add(identity[1]); artifact_ids.add(identity[2])
        current = {field: record.get(field) for field in common_fields}
        if not all(isinstance(current[name], str) and _HEX64.fullmatch(current[name])
                   for name in ("prepare_archive_sha256", "prepare_descriptor_sha256", "prepare_tree_sha256",
                                "corpus_sha256", "model_manifest_sha256")):
            raise ValueError("frozen prepare identity")
        if common is None:
            common = current
        elif common != current:
            raise ValueError("mixed frozen base")
    if {(record["role"], record["m"]) for record in records} != expected:
        raise ValueError("frozen role cardinality")
    if len(run_ids) != 3 or len(job_ids) != 3 or len(artifact_ids) != 3:
        raise ValueError("frozen roles must be distinct")
    return records


def validate_frozen_prepare_bundle(value: object, *, expected_head: str) -> dict[str, Any]:
    """Local import seam so prepare validation stays target-free and testable."""
    from eval.phase07_operator_gate import validate_frozen_prepare_bundle as validate_bundle
    return validate_bundle(value, expected_head=expected_head)


def load_verified_frozen_embedder(model_dir: Path):
    """Verify and load only the caller-selected local model, outside frozen data."""
    from eval.ann_corpus_manifest import load_manifest, validate_model_tree
    from obsidian_wiki.infrastructure.sentence_transformer_embedder import SentenceTransformerEmbedder

    model_dir = Path(model_dir).resolve()
    validate_model_tree(model_dir, load_manifest(Path(__file__).resolve().parent / "model-manifest.json"))
    embedder = SentenceTransformerEmbedder(model_dir)
    # Force local-only model load now, before any frozen-root mutation.  This
    # prevents a missing model from leaving a misleading partial artifact.
    _ = embedder.tokenizer
    return embedder


def main(argv: list[str] | None = None) -> int:
    """Hosted-only prepare entry point; it creates no candidate or role evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",))
    parser.add_argument("--wiki-dir", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--prepare-bundle", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=SKILL_EMBEDDER_DIR)
    parser.add_argument("--prepare-identity", type=Path)
    parser.add_argument("--repository")
    args = parser.parse_args(argv)

    try:
        bundle = json.loads(args.prepare_bundle.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenBaseError("prepare bundle") from exc
    expected_head = os.popen("git rev-parse HEAD").read().strip()
    validate_frozen_prepare_bundle(bundle, expected_head=expected_head)
    # This call deliberately precedes ``prepare_frozen_base``: no target,
    # probe directory, or copied corpus may exist when model validation fails.
    embedder = load_verified_frozen_embedder(args.model_dir)
    descriptor_identity = None
    if args.prepare_identity is not None:
        try:
            prepare_identity = json.loads(args.prepare_identity.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenBaseError("prepare identity") from exc
        if not args.repository:
            raise FrozenBaseError("prepare identity repository")
        prepared = validate_frozen_prepare_identity_shape(
            prepare_identity, expected_repository=args.repository, expected_head=expected_head,
        )
        descriptor_identity = {
            "target_size": FROZEN_TARGET_SIZE,
            "corpus_manifest_sha256": prepared["corpus_manifest_sha256"],
            "generator_recipe_sha256": prepared["generator_recipe_sha256"],
            "model_manifest_sha256": prepared["model_manifest_sha256"],
            "runtime": prepared["runtime"],
        }
    descriptor = prepare_frozen_base(
        wiki_dir=args.wiki_dir, frozen_dir=args.frozen_dir,
        embed=lambda texts: embedder.embed(list(texts)),
        tokenizer=embedder.tokenizer,
        descriptor_identity=descriptor_identity,
    )
    print(json.dumps({"authorization": "none", "descriptor": descriptor}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
