from obsidian_wiki.query_result import load_hits


def _sample():
    return {
        "text": [{"score": 0.9, "title": "a", "path": "Wiki/a.md"}],
        "images": [{"score": 0.8, "title": "b", "path": "Wiki/b.png"}],
        "query_plan": {"intent": "x"},
    }


def test_merge_text_and_images_with_kind():
    hits = load_hits(_sample())
    assert len(hits) == 2
    assert {h["kind"] for h in hits} == {"text", "image"}
    assert hits[0]["title"] == "a" and hits[1]["title"] == "b"


def test_accepts_dict_or_path(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(__import__("json").dumps(_sample()), encoding="utf-8")
    assert len(load_hits(p)) == 2
    assert len(load_hits(_sample())) == 2
