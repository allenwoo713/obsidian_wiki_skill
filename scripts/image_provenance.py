"""Shared, read-only Obsidian image-reference and Raw-source provenance resolver."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from pathlib import Path
from typing import Dict, Iterable, List

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


def normalize_embed(reference: str) -> str:
    """Return a case-preserving, vault-relative ``assets/...`` key for an embed."""
    raw = reference.split("|", 1)[0].split("#", 1)[0].strip().replace("\\", "/")
    raw = raw.lstrip("/")
    while raw.startswith("./"):
        raw = raw[2:]
    if raw.lower().startswith("wiki/"):
        raw = raw[5:]
    if not raw.lower().startswith("assets/"):
        raw = f"assets/{raw}"
    return "/".join(part for part in raw.split("/") if part not in ("", "."))


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
            refs.setdefault(key, []).append(page_key)
            candidates.setdefault(key, set()).update(sources)
    registered = {normalize_embed(entry.get("rel_path") or entry.get("filename", "")): entry
                  for entry in manifest_images}
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
