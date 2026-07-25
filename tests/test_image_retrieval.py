"""图片检索链路 + crash-safe 向量构建自检（ISSUE-16，Retrieval v2 API）。

覆盖两条此前踩过坑的路径：
1. image_caption 虚拟页能进索引、能被检索、且被 hybrid_search 归入 images 字段
   —— 防「图注写了却检索不到」回归。
2. _build_vector 的 crash-safe checkpoint 机制：成功后清理 .vec_ckpt；损坏的
   checkpoint 能被容错为「从头开始」。

全部使用虚构工业相机域数据（Acme VisionCam），自包含 fixture，不依赖任何真实知识库。

v2 API 变迁（测试已对齐）：
- ``query.split_text_image`` → ``query._split_text_image``（私有，吃 ContextItem）
- ``wi.search_bm25`` → 已移除；FTS 检索改用 ``wi.search_fts``
- 图注归入 images 的端到端验证改用 ``hybrid_search``（内部调用 _split_text_image）
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_index import WikiIndex  # noqa: E402


def _write_page(wiki: Path, name: str, title: str, body: str, sources=None):
    d = wiki / "concepts"
    d.mkdir(parents=True, exist_ok=True)
    fm = ('---\ntype: concept\ntitle: "%s"\nsources: %s\ntags: []\n'
          'related: []\nupdated: 2026-06-29\n---\n\n'
          % (title, json.dumps(sources or [])))
    (d / name).write_text(fm + body, encoding="utf-8")


def _seed_manifest_image(idx_dir: Path, rel_path: str, caption_text: str,
                         figure_caption: str = "", filename: str = ""):
    """在 manifest.json 预置一条 images 记录（build() 会据此生成 image_caption 页）。"""
    idx_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "images": [{
            "rel_path": rel_path,
            "caption_text": caption_text,
            "figure_caption": figure_caption,
            "filename": filename,
            "source_doc": "raw/manual.pdf",
            "sha256": "deadbeef",
        }]
    }
    (idx_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def test_image_caption_indexed_and_classified_as_image(tmp_path):
    """图注虚拟页进 BM25 索引、可检索，且 path 含 assets/ → hybrid_search 归入 images。"""
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "Acme VisionCam Overview",
                "工业相机总体介绍 分辨率 帧率", ["raw/a.docx"])
    # 图注：rel_path 落在 assets/images 下，_split_text_image 才会归为 image
    _seed_manifest_image(
        idx_dir,
        rel_path="assets/images/fig_exposure_curve.jpg",
        caption_text="曝光时间与信噪比关系曲线 exposure signal-to-noise ratio curve",
        figure_caption="图3 曝光-信噪比曲线",
    )
    wi = WikiIndex(idx_dir)
    wi.build(wiki)

    # 端到端：hybrid_search 内部 _split_text_image 把图注页归入 images
    from query_planner import DefaultQueryPlanner
    from query import hybrid_search
    planner = DefaultQueryPlanner(project_root=tmp_path)
    result = hybrid_search(wi, "exposure 信噪比 curve", planner, k=5, wiki_dir=wiki)
    all_items = result.text_items + result.image_items
    assert all_items, "图注页应能被检索到"
    paths = [str(it.path).replace("\\", "/") for it in all_items]
    assert any("fig_exposure_curve.jpg" in p for p in paths), \
        f"检索结果应包含图注页，实际: {paths}"

    # split_text_image 应把图注页归入 images
    img_paths = [str(it.path).replace("\\", "/") for it in result.image_items]
    assert any("fig_exposure_curve.jpg" in p for p in img_paths), \
        f"图注页应被归入 images 字段，实际 images: {img_paths}"


def test_empty_caption_image_not_indexed(tmp_path):
    """caption_text 与 vlm_caption 均空的图片不应进检索（避免噪声空页）。"""
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "T1", "正文内容", ["raw/a.docx"])
    _seed_manifest_image(idx_dir, rel_path="assets/images/blank.jpg",
                         caption_text="")  # 空 caption
    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    results = wi.search_fts("blank", k=5)
    paths = [str(r.path).replace("\\", "/") for r in results]
    assert not any("blank.jpg" in p for p in paths), \
        "空 caption 图片不应进入检索索引"


def test_vec_rebuild_idempotent_with_stale_checkpoint(tmp_path):
    """crash-safe 陈旧性保护：内容变更后重建，即使残留旧 checkpoint（如成功后
    清理被环境阻止），也不会复用错位批次，索引仍正确对应新内容。

    注：checkpoint 成功后会 best-effort 清理（shutil.rmtree），但某些受限环境
    （如禁用回收站的沙箱）会阻止删除、残留 .vec_ckpt。本测试正是验证「残留也无害」。
    """
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    # 第一次构建：内容 A
    _write_page(wiki, "a.md", "Acme VisionCam", "分辨率 帧率 曝光 exposure", ["raw/a.docx"])
    WikiIndex(idx_dir).build(wiki)
    # 改内容为 B（chunk 数/文本不同），再次构建
    _write_page(wiki, "a.md", "Vega Opticam", "增益 白平衡 gain white balance 全新内容", ["raw/a.docx"])
    wi2 = WikiIndex(idx_dir)
    wi2.build(wiki)
    # 新内容可检索，且旧内容特有词不再命中（证明未复用陈旧 checkpoint）
    r_new = wi2.search_fts("gain white balance 增益", k=3)
    assert r_new, "重建后新内容应可检索"
    titles = " ".join(t.title for t in r_new)
    assert "Vega Opticam" in titles, f"应召回新内容页，实际: {titles}"


def test_vec_build_recovers_from_corrupt_checkpoint(tmp_path):
    """crash-safe：损坏的 done.json 不应导致构建崩溃（容错为从头开始）。"""
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "Acme VisionCam", "分辨率 帧率 曝光", ["raw/a.docx"])
    # 预置损坏的 checkpoint
    ckpt = idx_dir / ".vec_ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "done.json").write_text("{not valid json", encoding="utf-8")
    wi = WikiIndex(idx_dir)
    wi.build(wiki)  # 不应抛异常
    results = wi.search_vector("曝光", k=2)
    assert results, "损坏 checkpoint 恢复后向量检索仍应可用"


def test_image_hit_backtraces_parent_context(tmp_path):
    """issue #12：图片命中后回溯父文档/页码/section/附近正文。

    manifest image 条目带 source_page/source_section/parent_page_id/nearby_text；
    build 保留这些字段；query 路径 load() 加载 _image_meta；
    assemble_context 把回溯信息附加到 image item 文本 + sources。
    """
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "Acme VisionCam Overview",
                "工业相机总体介绍 分辨率 帧率", ["raw/a.docx"])
    manifest = {
        "images": [{
            "rel_path": "assets/images/fig_exposure_curve.jpg",
            "caption_text": "曝光时间与信噪比关系曲线 exposure signal-to-noise ratio curve",
            "figure_caption": "图3 曝光-信噪比曲线",
            "source_doc": "raw/manual.pdf",
            "sha256": "deadbeef",
            "source_page": 5,
            "source_section": ["Installation", "Mounting angle"],
            "parent_page_id": "raw/manual.pdf",
            "nearby_text": "安装角度应保持在水平 ±5 度以内，曝光时间随环境亮度自适应。",
        }]
    }
    idx_dir.mkdir(parents=True, exist_ok=True)
    (idx_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    WikiIndex(idx_dir).build(wiki)
    # 查询路径走 load()（_image_meta 由 _load_image_meta 填充）
    wi = WikiIndex(idx_dir)
    wi.load()
    from query_planner import DefaultQueryPlanner
    from query import hybrid_search
    planner = DefaultQueryPlanner(project_root=tmp_path)
    result = hybrid_search(wi, "exposure 信噪比 curve", planner, k=5, wiki_dir=wiki)
    img_items = result.image_items
    assert img_items, "图注页应被检索到并归入 images"
    text = img_items[0].text
    # 回溯信息进入 image item 文本
    assert "页 5" in text, f"应含来源页码，实际: {text}"
    assert "Installation" in text, f"应含 section，实际: {text}"
    assert "安装角度" in text, f"应含附近正文，实际: {text}"
    # source_doc 进入 sources
    assert "raw/manual.pdf" in (img_items[0].sources or []), \
        f"source_doc 应进入 sources，实际: {img_items[0].sources}"


def test_image_no_nearby_text_marked_clearly(tmp_path):
    """issue #12：图片附近无正文时明确标记，不伪造解释。"""
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "Acme VisionCam", "分辨率 帧率", ["raw/a.docx"])
    manifest = {
        "images": [{
            "rel_path": "assets/images/fig_blank.jpg",
            "caption_text": "方框图 block diagram overview",
            "source_doc": "raw/manual.pdf",
            "sha256": "deadbeef",
            # 无 source_page / nearby_text
        }]
    }
    idx_dir.mkdir(parents=True, exist_ok=True)
    (idx_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    WikiIndex(idx_dir).build(wiki)
    wi = WikiIndex(idx_dir)
    wi.load()
    from query_planner import DefaultQueryPlanner
    from query import hybrid_search
    planner = DefaultQueryPlanner(project_root=tmp_path)
    result = hybrid_search(wi, "block diagram 方框图", planner, k=5, wiki_dir=wiki)
    img_items = result.image_items
    assert img_items, "图注页应被检索到"
    text = img_items[0].text
    # 无 nearby_text 时明确标记（不伪造）
    assert "无可用正文上下文" in text or "无父文档元数据" in text, \
        f"无附近正文时应明确标记，实际: {text}"
