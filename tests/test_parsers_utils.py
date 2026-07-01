"""parsers/utils.py 测试：slugify、图片命名与图注绑定。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from models import ImageRef
from parsers.utils import attach_captions, image_filename, slugify


def test_slugify_basic():
    assert slugify("Acme Front Radar Datasheet v1.6") == "acme-visioncam-front-datasheet-v1-6"


def test_slugify_chinese():
    # 中文保留，仅标点转 -
    assert slugify("Acme前雷达校准规范V1.0") == "Acme前雷达校准规范v1-0"


def test_slugify_strips_special():
    assert slugify("Vega Radar Tools User Manual (ClientX) -v1.1") == "vega-radar-tools-user-manual-clientx-v1-1"


def test_image_filename_basic():
    name = image_filename("acme-visioncam-front-datasheet-v1-6", 3, "png")
    assert name == "acme-visioncam-front-datasheet-v1-6_img03.png"


def test_image_filename_two_digit():
    name = image_filename("some-doc", 12, "jpeg")
    assert name == "some-doc_img12.jpeg"


def test_image_filename_preserves_ext():
    assert image_filename("d", 1, "jpg") == "d_img01.jpg"


def _make_image(seq: int = 1) -> ImageRef:
    return ImageRef(
        filename=f"doc_img{seq:02d}.png",
        rel_path=f"assets/doc_img{seq:02d}.png",
        caption="",
        source_media_name=f"xref={seq}",
        sha256="a" * 64,
        page_or_section=f"page {seq}",
    )


def test_attach_captions_chinese():
    img = _make_image(1)
    text = "some text\n{{IMG|assets/doc_img01.png|图注: 待补}}\n图1 系统架构图"
    new_text, new_images = attach_captions(text, [img])
    assert new_images[0].caption == "图1 系统架构图"
    assert "图注: 图1 系统架构图" in new_text


def test_attach_captions_figure():
    img = _make_image(1)
    text = "{{IMG|assets/doc_img01.png|图注: 待补}}\nFigure 1 System Architecture"
    new_text, new_images = attach_captions(text, [img])
    assert new_images[0].caption == "Figure 1 System Architecture"
    assert "图注: Figure 1 System Architecture" in new_text


def test_attach_captions_fig_dot():
    img = _make_image(1)
    text = "{{IMG|assets/doc_img01.png|图注: 待补}}\nFig. 1 System Architecture"
    new_text, new_images = attach_captions(text, [img])
    assert new_images[0].caption == "Fig. 1 System Architecture"
    assert "图注: Fig. 1 System Architecture" in new_text


def test_attach_captions_no_caption_within_window():
    img = _make_image(1)
    text = "{{IMG|assets/doc_img01.png|图注: 待补}}\nline1\nline2\nline3\nline4\nline5\n图1 too far"
    new_text, new_images = attach_captions(text, [img])
    assert new_images[0].caption == ""
    assert "图注: [无图注]" in new_text


def test_attach_captions_multiple_in_order():
    img1 = _make_image(1)
    img2 = _make_image(2)
    text = (
        "{{IMG|assets/doc_img01.png|图注: 待补}}\n图1 架构\n"
        "{{IMG|assets/doc_img02.png|图注: 待补}}\n图2 流程"
    )
    _, new_images = attach_captions(text, [img1, img2])
    assert new_images[0].caption == "图1 架构"
    assert new_images[1].caption == "图2 流程"


def test_attach_captions_in_place_modification():
    """契约：images 列表原地修改——返回对象身份相同，原列表元素被写入 caption。"""
    img1 = _make_image(1)
    img2 = _make_image(2)
    imgs = [img1, img2]
    text = (
        "{{IMG|assets/doc_img01.png|图注: 待补}}\n图1 架构\n"
        "{{IMG|assets/doc_img02.png|图注: 待补}}\n图2 流程"
    )
    _, returned = attach_captions(text, imgs)
    assert returned is imgs  # 同一列表对象
    assert imgs[0].caption == "图1 架构"  # 原列表元素被修改
    assert imgs[1].caption == "图2 流程"
    assert img1.caption == "图1 架构"  # 传入的 ImageRef 对象本身被修改
    assert img2.caption == "图2 流程"


def test_attach_captions_window_5th_line_hits():
    """契约：占位符后第 5 行（含）命中的图注应被采纳。"""
    img = _make_image(1)
    text = (
        "{{IMG|assets/doc_img01.png|图注: 待补}}\n"
        "line1\nline2\nline3\nline4\n"
        "图1 第五行命中"
    )
    new_text, new_images = attach_captions(text, [img])
    assert new_images[0].caption == "图1 第五行命中"
    assert "图注: 图1 第五行命中" in new_text


def test_attach_captions_window_6th_line_misses():
    """契约：占位符后第 6 行命中的图注不应被采纳。"""
    img = _make_image(1)
    text = (
        "{{IMG|assets/doc_img01.png|图注: 待补}}\n"
        "line1\nline2\nline3\nline4\nline5\n"
        "图1 第六行未命中"
    )
    new_text, new_images = attach_captions(text, [img])
    assert new_images[0].caption == ""
    assert "图注: [无图注]" in new_text


def test_attach_captions_more_placeholders_than_images():
    img = _make_image(1)
    text = (
        "{{IMG|assets/doc_img01.png|图注: 待补}}\n图1 命中\n"
        "{{IMG|assets/doc_img02.png|图注: 待补}}\n图2 未分配"
    )
    new_text, new_images = attach_captions(text, [img])
    assert len(new_images) == 1
    assert new_images[0].caption == "图1 命中"
    assert "图注: 图1 命中" in new_text
    assert "图注: 待补" in new_text


def test_attach_captions_empty_images():
    text = "{{IMG|assets/doc_img01.png|图注: 待补}}\n图1 说明"
    new_text, new_images = attach_captions(text, [])
    assert new_images == []
    assert new_text == text  # 无图片时 text 完全未变
    assert "图注: 待补" in new_text
