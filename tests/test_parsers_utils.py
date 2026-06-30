"""parsers/utils.py 测试：slugify 与图片命名。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from parsers.utils import slugify, image_filename


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
