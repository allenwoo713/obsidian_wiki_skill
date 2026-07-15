"""Wiki 索引构建：BM25 + LanceDB 向量 + manifest.json。
用法：python build_index.py <project_root>
"""
from __future__ import annotations
import json
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models import WikiPage, RetrievedPage, ManifestEntry


_FM_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def parse_wiki_page(path: Path, project_root: Path) -> Optional[WikiPage]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    m = _FM_RE.match(raw)
    if not m:
        return None
    fm_text, body = m.group(1), m.group(2)
    import yaml
    fm = yaml.safe_load(fm_text) or {}
    links = [l.strip() for l in _LINK_RE.findall(body)]
    import hashlib
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    sources = fm.get("sources", []) or []
    if isinstance(sources, str):
        sources = [sources]
    return WikiPage(
        path=path,
        title=fm.get("title", path.stem),
        page_type=fm.get("type", "concept"),
        content=body.strip(),
        sources=[str(s) for s in sources],
        links=links,
        sha256=sha,
    )


def scan_wiki(wiki_dir: Path, project_root: Path) -> List[WikiPage]:
    pages = []
    for md in sorted(wiki_dir.rglob("*.md")):
        if ".graph" in md.parts:
            continue
        p = parse_wiki_page(md, project_root)
        if p:
            pages.append(p)
    return pages


class WikiIndex:
    """BM25 + LanceDB 向量索引，支持增量构建。"""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.bm25 = None
        self.pages: List[WikiPage] = []
        self._page_paths: List[str] = []
        self._embedder = None
        self._lance_table = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            # 优先本地模型路径（已从 modelscope 预下载），回退 HF 在线下载
            local_path = os.environ.get("WIKI_EMBEDDER_LOCAL_PATH") or \
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))),
                    "binaries", "python", "envs", "default", "models",
                    "paraphrase-multilingual-MiniLM-L12-v2")
            # 也检查 venv 标准位置（跨平台）
            candidate_paths = [
                local_path,
                os.path.expanduser("~/.workbuddy/binaries/python/envs/default/models/paraphrase-multilingual-MiniLM-L12-v2"),
                "<home>/.workbuddy/binaries/python/envs/default/models/paraphrase-multilingual-MiniLM-L12-v2",
            ]
            for p in candidate_paths:
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "model.safetensors")):
                    self._embedder = SentenceTransformer(p)
                    return self._embedder
            # 回退：HF 在线下载（需网络）
            self._embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return self._embedder

    def _get_lance(self):
        if self._lance_table is None:
            import lancedb
            db = lancedb.connect(str(self.index_dir / "lance_db"))
            try:
                self._lance_table = db.open_table("wiki_pages")
            except Exception:
                self._lance_table = db.create_table("wiki_pages", data=[
                    {"path": "", "title": "", "page_type": "", "content": "",
                     "sources": "", "vector": [0.0] * 384}
                ])
                self._lance_table.delete('path = ""')
        return self._lance_table

    def build(self, wiki_dir: Path):
        self.pages = scan_wiki(wiki_dir, wiki_dir.parent)
        # 追加图片 caption 作为虚拟页（page_type=image_caption）
        idx_dir = wiki_dir.parent / ".index"
        image_pages = self._load_image_caption_pages(idx_dir)
        self.pages.extend(image_pages)
        self._page_paths = [str(p.path) for p in self.pages]
        self._build_bm25()
        self._build_vector()
        self._write_manifest()

    def _load_image_caption_pages(self, idx_dir: Path) -> List[WikiPage]:
        """从 manifest.json 读 images，caption_text 非空的转为 WikiPage。"""
        manifest_file = idx_dir / "manifest.json"
        if not manifest_file.exists():
            return []
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            return []
        wiki_dir = idx_dir.parent / "Wiki"
        pages = []
        for img in manifest.get("images", []):
            # caption_text 是主检索字段；为空时兜底读 vlm_caption.description，
            # 确保已生成 VLM 描述的图不会因 caption_text 漏填而缺席检索（defense-in-depth）。
            caption = (img.get("caption_text") or "").strip()
            if not caption:
                vlm = img.get("vlm_caption") or {}
                caption = (vlm.get("description") or "").strip()
            if not caption:
                continue
            img_path = wiki_dir / img["rel_path"]
            pages.append(WikiPage(
                path=img_path,
                title=img.get("figure_caption") or img["filename"],
                page_type="image_caption",
                content=caption,
                sources=[img.get("source_doc", "")],
                links=[],
                sha256=img.get("sha256", ""),
            ))
        return pages

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"[^\w\u4e00-\u9fff]", " ", text.lower())
        tokens = re.split(r"\s+", text.strip())
        cjk = re.findall(r"[\u4e00-\u9fff]", text)
        bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
        return [t for t in tokens if t] + bigrams

    def _build_bm25(self):
        from rank_bm25 import BM25Plus
        corpus = [self._tokenize(p.title + " " + p.content) for p in self.pages]
        self.bm25 = BM25Plus(corpus) if corpus else None
        with open(self.index_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "paths": self._page_paths}, f)

    def _build_vector(self):
        embedder = self._get_embedder()
        texts = [p.title + "\n" + p.content for p in self.pages]
        vectors = embedder.encode(texts, show_progress_bar=False).tolist()
        table = self._get_lance()
        try:
            table.delete("path != ''")
        except Exception:
            pass
        data = [
            {
                "path": str(p.path),
                "title": p.title,
                "page_type": p.page_type,
                "content": p.content[:2000],
                "sources": json.dumps(p.sources, ensure_ascii=False),
                "vector": v,
            }
            for p, v in zip(self.pages, vectors)
        ]
        if data:
            table.add(data)

    def _write_manifest(self):
        manifest_file = self.index_dir / "manifest.json"
        # 读取现有 manifest（保留 images/entries 等非 pages 字段）
        existing = {}
        if manifest_file.exists():
            try:
                existing = json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing["built_at"] = datetime.now().isoformat()
        existing["page_count"] = len(self.pages)
        existing["pages"] = [
            {
                "path": str(p.path),
                "sha256": p.sha256,
                "page_type": p.page_type,
                "title": p.title,
                "sources": p.sources,
                "links": p.links,
            }
            for p in self.pages
        ]
        manifest_file.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self):
        with open(self.index_dir / "bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self._page_paths = data["paths"]
        self._get_lance()
        # 从 manifest 重建 pages（供 search_bm25 获取 title/sources/snippet）
        manifest_file = self.index_dir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            page_map = {p["path"]: p for p in manifest.get("pages", [])}
            self.pages = []
            for path in self._page_paths:
                p = page_map.get(path, {})
                self.pages.append(WikiPage(
                    path=Path(path),
                    title=p.get("title", Path(path).stem),
                    page_type=p.get("page_type", "concept"),
                    content="",  # snippet 从文件读取
                    sources=p.get("sources", []),
                    links=p.get("links", []),
                    sha256=p.get("sha256", ""),
                ))

    def search_bm25(self, query: str, k: int = 20) -> List[RetrievedPage]:
        if self.bm25 is None:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:k]
        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            p = self.pages[idx] if idx < len(self.pages) else None
            path = self._page_paths[idx]
            title = p.title if p else Path(path).stem
            sources = p.sources if p else []
            snippet = (p.content[:200] if p and p.content else "")[:200]
            if not snippet and p:
                # 从文件读取 snippet
                try:
                    raw = p.path.read_text(encoding="utf-8", errors="replace")
                    import re as _re
                    m = _re.search(r"^---\n.*?\n---\n(.*)$", raw, _re.DOTALL)
                    snippet = (m.group(1).strip()[:200] if m else raw[:200])
                except Exception:
                    snippet = ""
            results.append(RetrievedPage(
                path=Path(path), title=title, score=float(score),
                snippet=snippet, sources=sources, retrieval_method="bm25",
            ))
        return results

    def search_vector(self, query: str, k: int = 20) -> List[RetrievedPage]:
        embedder = self._get_embedder()
        qv = embedder.encode([query], show_progress_bar=False)[0].tolist()
        table = self._get_lance()
        rows = table.search(qv).limit(k).to_list()
        results = []
        for r in rows:
            results.append(RetrievedPage(
                path=Path(r["path"]), title=r["title"], score=1.0 / (1.0 + float(r.get("_distance", 1.0))),
                snippet=r["content"][:200], sources=json.loads(r.get("sources", "[]")),
                retrieval_method="vector",
            ))
        return results


def main():
    if len(sys.argv) < 2:
        print("用法: python build_index.py <project_root>")
        sys.exit(1)
    proj = Path(sys.argv[1])
    wiki = proj / "Wiki"
    idx_dir = proj / ".index"
    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    print(f"索引构建完成: {len(wi.pages)} 页, manifest → {idx_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
