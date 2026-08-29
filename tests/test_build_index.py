"""build_index.py 测试。"""
import json
from pathlib import Path
import pytest
import sys
import types
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_index import WikiIndex
from obsidian_wiki.domain.index_policy import load_ann_policy_file
from obsidian_wiki.infrastructure.sentence_transformer_embedder import SentenceTransformerEmbedder


class _RecordingDenseEmbedder:
    """Production-seam fake that records each outer encode request."""

    def __init__(self):
        self.calls = []

    def get_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return [[float(text.rsplit("-", 1)[1]), float(len(text))] for text in texts]


def _dense_chunk(page_id, position):
    return types.SimpleNamespace(
        page_id=page_id,
        content_hash=f"hash-{page_id}-{position}",
        text=f"dense-{position}",
    )


def test_cached_dense_vectors_empty_input_never_encodes(tmp_path):
    index = WikiIndex(tmp_path / ".index")
    embedder = _RecordingDenseEmbedder()
    index._embedder = embedder

    assert index._cached_dense_vectors([], embedder, full_rebuild=False) == []
    assert embedder.calls == []


def test_cached_dense_vectors_at_most_outer_bound_preserves_arguments_and_order(tmp_path):
    index = WikiIndex(tmp_path / ".index")
    embedder = _RecordingDenseEmbedder()
    index._embedder = embedder
    chunks = [_dense_chunk("page-a", position) for position in range(3)]

    vectors = index._cached_dense_vectors(chunks, embedder, full_rebuild=False)

    assert embedder.calls == [(
        ["dense-0", "dense-1", "dense-2"],
        {"show_progress_bar": False, "normalize_embeddings": False},
    )]
    assert vectors == [[0.0, 7.0], [1.0, 7.0], [2.0, 7.0]]


def test_cached_dense_vectors_batches_1025_misses_and_preserves_page_cache_order(tmp_path):
    index = WikiIndex(tmp_path / ".index")
    embedder = _RecordingDenseEmbedder()
    index._embedder = embedder
    chunks = (
        [_dense_chunk("page-a", position) for position in range(1023)]
        + [_dense_chunk("page-b", position) for position in range(1023, 1025)]
    )

    vectors = index._cached_dense_vectors(chunks, embedder, full_rebuild=False)

    assert [len(texts) for texts, _kwargs in embedder.calls] == [1024, 1]
    assert embedder.calls[0][0] == [f"dense-{position}" for position in range(1024)]
    assert embedder.calls[1][0] == ["dense-1024"]
    assert all(kwargs == {
        "show_progress_bar": False,
        "normalize_embeddings": False,
    } for _texts, kwargs in embedder.calls)
    assert vectors == [[float(position), float(len(f"dense-{position}"))]
                       for position in range(1025)]

    cache_arrays = [np.load(path).tolist() for path in (tmp_path / ".index").glob("vec_cache/**/*.npy")]
    assert sorted(cache_arrays, key=len) == [
        [[1023.0, 10.0], [1024.0, 10.0]],
        [[float(position), float(len(f"dense-{position}"))] for position in range(1023)],
    ]


def test_wikiindex_build_has_no_runtime_mode_selection(monkeypatch, tmp_path):
    """Phase 06（issue #49）：facade 不传播 auto/exact/FLAT/SQ 运行时选择。"""
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
        assert "vector_index_mode" not in kwargs, "runtime mode selection must be removed"
        requested.append(sorted(kwargs))
        approved_ann = load_ann_policy_file()
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps({
                "policy": {"selected_mode": "ann"},
                "ann_policy": {
                    "selected_index_type": approved_ann.selected_index_type,
                    "query_ef": approved_ann.query_ef,
                },
            }),
            encoding="utf-8",
        )
        return types.SimpleNamespace(artifact=types.SimpleNamespace(manifest_path=manifest))

    monkeypatch.setattr(build_module, "build_storage_contract", fake_build)
    index = WikiIndex(tmp_path / ".index")
    index._embedder = FakeEmbedder()

    index._build(tmp_path / "Wiki")

    assert requested, "facade must still route through build_storage_contract"
    assert index._vector_index_mode == "ivf-hnsw-sq"  # 来自 manifest 的固定策略


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


def test_sentence_transformer_embedder_can_surface_batch_progress(monkeypatch, tmp_path):
    """Long operator runs must be able to expose the encoder's batch counter."""
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

    assert SentenceTransformerEmbedder(model_dir).embed(
        ["dense text"], show_progress_bar=True,
    ) == [(1.0, 2.0)]
    assert calls == [(["dense text"], {"show_progress_bar": True, "normalize_embeddings": False})]
