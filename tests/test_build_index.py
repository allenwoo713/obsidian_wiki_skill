"""build_index.py 测试。"""
import json
from pathlib import Path
import pytest
import sys
import types

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_index import WikiIndex
from obsidian_wiki.infrastructure.sentence_transformer_embedder import SentenceTransformerEmbedder


def test_wikiindex_propagates_requested_vector_mode(monkeypatch, tmp_path):
    """Eval's exact control must cross the production facade unchanged."""
    import build_index as build_module

    requested = []

    class FakeEmbedder:
        tokenizer = object()

        def get_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(build_module, "scan_wiki", lambda *_args: [])
    monkeypatch.setattr(build_module, "EmbeddingTokenizer", lambda _tokenizer: types.SimpleNamespace(count=len))

    def fake_build(*_args, **kwargs):
        mode = kwargs["vector_index_mode"]
        requested.append(mode)
        manifest = tmp_path / f"{mode}.json"
        manifest.write_text(
            json.dumps({"policy": {"selected_mode": "exact" if mode == "exact" else "ann"}}),
            encoding="utf-8",
        )
        return types.SimpleNamespace(artifact=types.SimpleNamespace(manifest_path=manifest))

    monkeypatch.setattr(build_module, "build_storage_contract", fake_build)
    index = WikiIndex(tmp_path / ".index")
    index._embedder = FakeEmbedder()

    index._build(tmp_path / "Wiki", vector_index_mode="exact")
    index._build(tmp_path / "Wiki", vector_index_mode="ivf-hnsw-flat")

    assert requested == ["exact", "ivf-hnsw-flat"]


def _write_page(wiki: Path, name: str, title: str, body: str, sources=None):
    d = wiki / "concepts"
    d.mkdir(parents=True, exist_ok=True)
    fm = '---\ntype: concept\ntitle: "%s"\nsources: %s\ntags: []\nrelated: []\nupdated: 2026-06-29\n---\n\n' % (title, json.dumps(sources or []))
    (d / name).write_text(fm + body, encoding="utf-8")


def test_build_and_search(tmp_path):
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "Acme Front Radar", "频率 60fps 探测距离 200m", ["raw/a.docx"])
    _write_page(wiki, "b.md", "Vega Radar", "频率 76GHz 探测距离 150m", ["raw/b.docx"])
    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    results = wi.search("Acme 60fps", k=2)
    assert len(results) > 0
    assert "Acme" in results[0].title or "Acme" in results[0].path.name


def test_manifest_written(tmp_path):
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "T1", "body text here", ["raw/a.docx"])
    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    # #11 指针方案：manifest 落在 builds/<id>/manifest.json（顶层仅作 images/entries 合并源）
    builds = list((idx_dir / "builds").glob("build_*/manifest.json"))
    assert builds, "build manifest 未写入 builds/<id>/"
    data = json.loads(builds[0].read_text(encoding="utf-8"))
    assert len(data["pages"]) >= 1
    assert data["pages"][0]["sha256"]


def test_vector_search(tmp_path):
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "Radar Calibration", "radar calibration procedure angle alignment", ["raw/a.docx"])
    _write_page(wiki, "b.md", "UDP Protocol", "udp packet format diagnostic interface", ["raw/b.docx"])
    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    results = wi.search_vector("calibration alignment", k=2)
    assert len(results) > 0


def test_sentence_transformer_embedder_loads_only_local_assets(monkeypatch, tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"local")
    calls = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            calls.append((list(texts), kwargs))
            return [[1.0, 2.0] for _ in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(
        SentenceTransformer=lambda path, local_files_only: FakeModel()
    ))
    embedder = SentenceTransformerEmbedder(model_dir)

    assert embedder.embed(["dense text"]) == [(1.0, 2.0)]
    assert calls == [(["dense text"], {"show_progress_bar": False, "normalize_embeddings": False})]

    with pytest.raises(RuntimeError, match="(?i)local embedding model"):
        SentenceTransformerEmbedder(tmp_path / "missing").embed(["dense text"])
