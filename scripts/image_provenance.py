"""Shared, read-only Obsidian image-reference and Raw-source provenance resolver."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

WIKILINK = re.compile(r"!\[\[([^\]]+)\]\]")
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


@dataclass(frozen=True)
class ImageResolution:
    canonical_key: str
    referring_wiki_pages: tuple[str, ...]
    parent_page_id: str
    source_doc: str
    source_candidates: tuple[str, ...]
    status: str

    def to_json(self):
        return asdict(self)


def normalize_embed(reference: str) -> Optional[str]:
    """Return a case-preserving, vault-relative ``assets/...`` key for an embed."""
    raw = reference.split("|", 1)[0].split("#", 1)[0].strip().replace("\\", "/")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
        return None
    while raw.startswith("./"):
        raw = raw[2:]
    if raw.lower().startswith("wiki/"):
        raw = raw[5:]
    if any(part == ".." for part in raw.split("/")):
        return None
    if not raw.lower().startswith("assets/"):
        raw = f"assets/{raw}"
    key = "/".join(part for part in raw.split("/") if part not in ("", "."))
    return key if key and not any(part == ".." for part in key.split("/")) else None


def resolve_asset_path(wiki_root: Path, canonical_key: str) -> Optional[Path]:
    """Return a real asset only when symlink resolution stays inside Wiki/assets."""
    root = (Path(wiki_root) / "assets").resolve()
    try:
        resolved = (Path(wiki_root) / canonical_key).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _frontmatter_sources(text: str) -> List[str]:
    match = _FRONTMATTER.match(text)
    if not match:
        return []
    lines = match.group(1).splitlines()
    sources, in_sources = [], False
    for line in lines:
        if line.startswith("sources:"):
            value = line.split(":", 1)[1].strip().strip("[]")
            if value:
                sources.extend(item.strip().strip("'\"") for item in value.split(",") if item.strip())
            in_sources = True
            continue
        if in_sources and re.match(r"^\s*-\s+", line):
            sources.append(re.sub(r"^\s*-\s+", "", line).strip().strip("'\""))
            continue
        if line and not line.startswith((" ", "\t")):
            in_sources = False
    return [source.replace("\\", "/") for source in sources if source.lower().startswith("raw/")]


def resolve_image_references(project_root: Path, manifest_images: Iterable[dict] = ()) -> Dict[str, ImageResolution]:
    """Resolve every Wiki image reference without guessing a Raw source document."""
    root = Path(project_root)
    refs: Dict[str, List[str]] = {}
    candidates: Dict[str, set[str]] = {}
    wiki = root / "Wiki"
    for page in sorted(wiki.rglob("*.md")):
        if ".obsidian" in page.parts or ".graph" in page.parts:
            continue
        try:
            content = page.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        page_key = page.relative_to(root).as_posix()
        sources = _frontmatter_sources(content)
        for raw in WIKILINK.findall(content):
            key = normalize_embed(raw)
            if key is None:
                continue
            refs.setdefault(key, []).append(page_key)
            candidates.setdefault(key, set()).update(sources)
    registered = {key: entry for entry in manifest_images
                  if (key := normalize_embed(entry.get("rel_path") or entry.get("filename", ""))) is not None}
    resolved = {}
    for key in sorted(refs):
        referrers = tuple(sorted(set(refs[key])))
        source_candidates = tuple(sorted(candidates.get(key, set())))
        if len(source_candidates) == 1:
            source_doc, status = source_candidates[0], "resolved"
        elif source_candidates:
            source_doc, status = "", "ambiguous_provenance"
        else:
            source_doc, status = "", "unresolved_provenance"
        if key in registered:
            status = registered[key].get("status") or "registered"
        resolved[key] = ImageResolution(key, referrers, referrers[0], source_doc, source_candidates, status)
    return resolved
