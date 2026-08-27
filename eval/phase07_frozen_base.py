"""Candidate-neutral Phase 07 frozen corpus preparation and private ANN clones.

The module deliberately sits beside the evaluation orchestration.  It is not a
new production build mode: normal ``WikiIndex.build`` continues to create and
publish a complete active index.  This path seals only the reusable 30k source
boundary (two Lance tables, FTS, graph and page identities), then requires every
ANN role to create a separate writable Lance clone.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import unicodedata
from pathlib import Path
from typing import Any, Callable

from build_index import page_id_of, scan_wiki
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


SCHEMA_VERSION = 1
_TOP_LEVEL = frozenset({"Wiki", "lance_db", "graph.json", "pages.json", "frozen-base.json"})
_DESCRIPTOR_FIELDS = frozenset({
    "schema_version", "kind", "authorization", "resolved_wiki_root", "pages_sha256",
    "graph_sha256", "source_tree_sha256", "lance_tree_sha256", "frozen_tree_sha256",
    "record_self_sha256",
})
_FORBIDDEN_NAMES = frozenset({"ACTIVE_INDEX", "manifest.json", ".failed"})


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


def _tree_inventory(root: Path, *, exclude: frozenset[str] = frozenset()) -> tuple[list[dict[str, object]], str]:
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise FrozenBaseError("frozen tree root")
    inventory: list[dict[str, object]] = []
    folded: set[str] = set()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
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


def _make_chunks(wiki_dir: Path, embed: Callable[[list[str]], list[list[float]]], *, tokenizer: object) -> tuple[list[object], list[DenseChunk], list[dict[str, object]]]:
    wiki_dir = Path(wiki_dir)
    project_root = wiki_dir.parent
    pages = scan_wiki(wiki_dir, project_root)
    if not pages:
        raise FrozenBaseError("frozen corpus has no pages")
    lexicon = load_lexicon(project_root)
    tokenizer = EmbeddingTokenizer(tokenizer)
    canonical = []
    page_rows = []
    for page in pages:
        page_id = page_id_of(page.path)
        records = list(chunk_page(
            page_id=page_id, path=page.path, title=page.title, page_type=page.page_type,
            content=page.content, tokenizer=tokenizer.count,
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
    vectors = embed([chunk.text for chunk in dense_sources])
    if len(vectors) != len(dense_sources):
        raise FrozenBaseError("frozen embedding count")
    dense = [
        DenseChunk(
            chunk_id=chunk.chunk_id, page_id=chunk.page_id, path=chunk.path, title=chunk.title,
            text=chunk.text, vector=tuple(float(value) for value in vector), page_type=chunk.page_type,
            section_path=chunk.section_path, heading=chunk.heading, chunk_kind=chunk.chunk_kind,
            chunk_index=chunk.chunk_index, parent_section_id=chunk.parent_section_id,
            token_count=chunk.token_count, content_hash=chunk.content_hash,
            forced_split=chunk.forced_split, continuation_index=chunk.continuation_index,
            start_char=chunk.start_char, end_char=chunk.end_char,
        )
        for chunk, vector in zip(dense_sources, vectors)
    ]
    if not dense:
        raise FrozenBaseError("frozen corpus has no dense chunks")
    return [chunk for chunk in canonical if chunk.chunk_kind == "sparse"], dense, page_rows


def prepare_frozen_base(*, wiki_dir: Path, frozen_dir: Path, embed: Callable[[list[str]], list[list[float]]],
                        tokenizer: object) -> dict[str, object]:
    """Persist reusable data once, deliberately stopping before HNSW/publication."""
    wiki_dir, frozen_dir = Path(wiki_dir).resolve(), Path(frozen_dir)
    if frozen_dir.exists():
        raise FrozenBaseError("frozen target must be new")
    sparse, dense, pages = _make_chunks(wiki_dir, embed, tokenizer=tokenizer)
    frozen_dir.mkdir(parents=True)
    shutil.copytree(wiki_dir, frozen_dir / "Wiki")
    pages_path = frozen_dir / "pages.json"
    pages_path.write_bytes(_canonical_json(pages))
    (frozen_dir / "graph.json").write_bytes(_canonical_json(_graph_payload(wiki_dir)))
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
    frozen_tree = _tree_inventory(frozen_dir)[1]
    descriptor: dict[str, object] = {
        "schema_version": SCHEMA_VERSION, "kind": "phase07-frozen-base", "authorization": "none",
        "resolved_wiki_root": str(wiki_dir), "pages_sha256": _sha256_file(pages_path),
        "graph_sha256": _sha256_file(frozen_dir / "graph.json"), "source_tree_sha256": source_tree,
        "lance_tree_sha256": lance_tree, "frozen_tree_sha256": frozen_tree,
    }
    descriptor["record_self_sha256"] = _sha256_bytes(_canonical_json(descriptor))
    (frozen_dir / "frozen-base.json").write_bytes(_canonical_json(descriptor))
    return descriptor


def _read_descriptor(frozen_dir: Path) -> dict[str, object]:
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
    return descriptor


def validate_frozen_base(frozen_dir: Path, *, expected_wiki_root: Path) -> str:
    """Validate data, FTS, graph/page identities and the fixed absolute-root contract."""
    frozen_dir = Path(frozen_dir)
    descriptor = _read_descriptor(frozen_dir)
    if {path.name for path in frozen_dir.iterdir()} != _TOP_LEVEL or any(
        name in _FORBIDDEN_NAMES for name in (path.name for path in frozen_dir.rglob("*"))
    ):
        raise FrozenBaseError("frozen top-level allowlist")
    expected = Path(expected_wiki_root).resolve()
    if descriptor["resolved_wiki_root"] != str(expected):
        raise FrozenBaseError("frozen resolved root")
    if _tree_inventory(frozen_dir / "Wiki")[1] != descriptor["source_tree_sha256"] \
            or _tree_inventory(frozen_dir / "lance_db")[1] != descriptor["lance_tree_sha256"]:
        raise FrozenBaseError("frozen tree digest")
    if _sha256_file(frozen_dir / "pages.json") != descriptor["pages_sha256"] \
            or _sha256_file(frozen_dir / "graph.json") != descriptor["graph_sha256"]:
        raise FrozenBaseError("frozen sidecar digest")
    try:
        pages = json.loads((frozen_dir / "pages.json").read_text(encoding="utf-8"))
        graph = json.loads((frozen_dir / "graph.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenBaseError("frozen sidecar") from exc
    if not isinstance(pages, list) or not isinstance(graph, dict):
        raise FrozenBaseError("frozen sidecar shape")
    page_ids = {str(page.get("page_id", "")) for page in pages if isinstance(page, dict)}
    if not page_ids or any(Path(str(page.get("path", ""))).resolve() != Path(str(page["page_id"])) for page in pages):
        raise FrozenBaseError("frozen page identity")
    if page_ids != {str(Path(page["page_id"])) for page in pages}:
        raise FrozenBaseError("frozen page identity")
    node_ids = {str(node.get("id", "")) for node in graph.get("nodes", []) if isinstance(node, dict)}
    if not node_ids.issubset(page_ids):
        raise FrozenBaseError("frozen graph identity")
    repository = LanceDbIndexRepository(frozen_dir / "lance_db")
    sparse = repository.table_rows("sparse_chunks")
    dense = repository.table_rows("dense_chunks")
    if {str(row["page_id"]) for row in (*sparse, *dense)} - page_ids:
        raise FrozenBaseError("frozen table page identity")
    if repository._dense_table().list_indices():
        raise FrozenBaseError("frozen dense vector index")
    fts_indices = {index.name for index in repository._sparse_table().list_indices()}
    if fts_indices != {"fts_text_idx"}:
        raise FrozenBaseError("frozen sparse FTS")
    repository.validate_reopened(
        dimension=len(dense[0]["vector"]), exact_term=_exact_term([type("R", (), row) for row in sparse]),
        vector_index_name=None,
    )
    return str(descriptor["frozen_tree_sha256"])


def finalize_private_role(*, frozen_dir: Path, target_dir: Path, expected_wiki_root: Path,
                          candidate_query_policy: CandidateQueryPolicy) -> dict[str, object]:
    """Clone validated tables and build exactly one candidate HNSW in that clone."""
    source_digest = validate_frozen_base(frozen_dir, expected_wiki_root=expected_wiki_root)
    target_dir = Path(target_dir)
    if target_dir.exists():
        raise FrozenBaseError("private clone target must be new")
    target_dir.mkdir(parents=True)
    source = LanceDbIndexRepository(Path(frozen_dir) / "lance_db")
    identities = source.clone_tables(target_dir / "lance_db")
    shutil.copy2(Path(frozen_dir) / "pages.json", target_dir / "pages.json")
    shutil.copy2(Path(frozen_dir) / "graph.json", target_dir / "graph.json")
    dense_count = next(identity.row_count for identity in identities if identity.table_name == "dense_chunks")
    policy = candidate_query_policy.build_policy
    target = LanceDbIndexRepository(target_dir / "lance_db", eval_candidate_policy=candidate_query_policy)
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
    target.seal(target_dir / "lance_db")
    if validate_frozen_base(frozen_dir, expected_wiki_root=expected_wiki_root) != source_digest:
        raise FrozenBaseError("frozen source mutated by private clone")
    return {"source_tree_sha256": source_digest, "role_m": policy.m, "lance_dir": str(target_dir / "lance_db")}


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
            path = Path(member.name)
            normalized = unicodedata.normalize("NFC", member.name)
            if member.name != normalized or path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() \
                    or not (member.isdir() or member.isfile()) or normalized.casefold() in names:
                raise FrozenBaseError("archive member")
            names.add(normalized.casefold())
        for member in members:
            handle.extract(member, destination, set_attrs=False, numeric_owner=False)


def validate_frozen_role_provenance(records: list[dict[str, Any]], *, expected_head: str) -> list[dict[str, Any]]:
    """Accept only the complete, independent, first-attempt role set for one base."""
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("complete frozen role batch")
    expected = {("baseline", 16), ("m20", 20), ("m32", 32)}
    common_fields = {
        "prepare_run_id", "prepare_run_attempt", "prepare_job_id", "prepare_artifact_id",
        "archive_sha256", "tree_sha256", "retention_days", "head_sha", "runtime",
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
        if common is None:
            common = current
        elif common != current:
            raise ValueError("mixed frozen base")
    if {(record["role"], record["m"]) for record in records} != expected:
        raise ValueError("frozen role cardinality")
    if len(run_ids) != 3 or len(job_ids) != 3 or len(artifact_ids) != 3:
        raise ValueError("frozen roles must be distinct")
    return records
