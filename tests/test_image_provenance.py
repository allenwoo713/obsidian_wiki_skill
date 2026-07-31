"""Cross-path image provenance and read-only audit tests (issue #19)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from image_provenance import normalize_embed, resolve_image_references  # noqa: E402


def _page(root, name, sources, embeds):
    page = root / "Wiki" / "concepts" / name
    page.parent.mkdir(parents=True, exist_ok=True)
    source_lines = "\n".join(f"  - {source}" for source in sources)
    page.write_text(f"---\nsources:\n{source_lines}\n---\n\n" + "\n".join(embeds), encoding="utf-8")


def test_normalizer_and_shared_resolution_keep_raw_source_distinct(tmp_path):
    _page(tmp_path, "one.md", ["Raw/sources/a.pdf"], ["![[assets/sub/Figure.PNG|caption]]"])
    _page(tmp_path, "two.md", ["Raw/sources/a.pdf"], ["![[sub\\Figure.PNG]]"])
    resolution = resolve_image_references(tmp_path)["assets/sub/Figure.PNG"]
    assert normalize_embed("Figure.PNG|alias") == "assets/Figure.PNG"
    assert resolution.source_doc == "Raw/sources/a.pdf"
    assert resolution.parent_page_id == "Wiki/concepts/one.md"
    assert resolution.referring_wiki_pages == ("Wiki/concepts/one.md", "Wiki/concepts/two.md")


def test_ambiguous_and_unresolved_sources_are_never_guessed(tmp_path):
    _page(tmp_path, "one.md", ["Raw/sources/a.pdf"], ["![[same.png]]"])
    _page(tmp_path, "two.md", ["Raw/sources/b.pdf"], ["![[same.png]]"])
    _page(tmp_path, "three.md", [], ["![[missing.png]]"])
    resolved = resolve_image_references(tmp_path)
    assert resolved["assets/same.png"].status == "ambiguous_provenance"
    assert resolved["assets/same.png"].source_doc == ""
    assert resolved["assets/missing.png"].status == "unresolved_provenance"


def test_audit_is_byte_identical_without_explicit_write_mode(tmp_path, monkeypatch):
    _page(tmp_path, "one.md", ["Raw/sources/a.pdf"], ["![[fig.png]]"])
    assets = tmp_path / "Wiki" / "assets"
    assets.mkdir(parents=True)
    (assets / "fig.png").write_bytes(b"image")
    index = tmp_path / ".index"
    index.mkdir()
    manifest = index / "manifest.json"
    manifest.write_text(json.dumps({"images": [{"filename": "fig.png", "caption_text": ""}]}), encoding="utf-8")
    before = manifest.read_bytes()
    import audit_images
    monkeypatch.setattr(sys, "argv", ["audit_images.py", str(tmp_path)])
    audit_images.main()
    assert manifest.read_bytes() == before
