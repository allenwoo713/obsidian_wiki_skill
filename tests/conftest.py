import sys
from pathlib import Path

# Make the sibling `scripts/` importable for tests (no package install).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


import pytest  # noqa: E402


@pytest.fixture
def tiny_kb(tmp_path):
    """构建一个最小脱敏 KB（虚构 Acme VisionCam / Columbus 雷达）并返回
    ``(WikiIndex, wiki_dir, tmp_path)``。两台 Columbus 雷达共享同一 source，
    便于图谱源重叠测试。"""
    from build_index import WikiIndex

    wiki = tmp_path / "Wiki"
    wiki.mkdir()

    def _w(name, title, body, sources=None):
        fm = "---\ntitle: " + title + "\n"
        if sources:
            fm += "sources:\n" + "".join(f"  - {s}\n" for s in sources)
        fm += "---\n\n"
        (wiki / name).write_text(fm + body, encoding="utf-8")

    _w("cam_x200.md", "Acme VisionCam X200",
       "# Acme VisionCam X200\n\n分辨率 1920×1080，帧率 60 fps，接口 GigE Vision。\n\n"
       "## 错误码\n\n| 码 | 说明 |\n|---|---|\n| 0x0102 | magicWord 校验失败 |\n",
       ["raw/cam.docx"])
    _w("radar_cfr100.md", "Columbus Front Radar CFR-100",
       "# Columbus Front Radar CFR-100\n\n探测距离 250 m，频段 77 GHz，接口 CAN FD。",
       ["raw/radar.docx"])
    _w("radar_ccr100.md", "Columbus Corner Radar CCR-100",
       "# Columbus Corner Radar CCR-100\n\n探测距离 150 m，视场角 ±75°，接口 CAN FD。",
       ["raw/radar.docx"])

    idx = tmp_path / ".index"
    wi = WikiIndex(idx)
    wi.build(wiki)
    wi.load()
    return wi, wiki, tmp_path
