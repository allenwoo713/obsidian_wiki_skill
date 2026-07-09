# Obsidian Wiki Skill

> 把一堆散乱的产品文档（datasheet / 安装规范 / 校准规范 / 工具手册 / 接口文档）变成可被 AI agent **混合检索**的知识库：Obsidian vault（双链 + frontmatter）作存储与人工浏览后端，BM25 + 向量 + 图谱三路融合作检索后端，答案强制标注出处。

专为 WorkBuddy agent（Box）设计——`query.py` 是 agent 检索知识库的唯一入口，返回带出处的结构化结果。

---

## 为什么不是纯向量检索

小 corpus（< 100 文档）+ 中英混合（datasheet 英文 / 校准规范中文）场景下，纯向量召回准确率不够：

| 痛点 | 本方案 |
|---|---|
| 产品型号 / 具体数值（60fps、±45°、120m）向量检索不到 | BM25 关键词精确命中 |
| 同义不同形（"前雷达" vs "Front Radar"）词袋错过 | 多语言向量 + 查询预处理中英互译 |
| 孤立页面无上下文 | 4 信号图谱 2 跳扩展 |
| 三路结果难取舍 | RRF（Reciprocal Rank Fusion, k=60）融合 |

## 架构

```
Raw/sources/ ──parse──▶ Wiki/*.md ──index──▶ .index/ ──query──▶ 带出处的答案
 (不可变)      (Box LLM)   (双链+frontmatter)   (BM25+LanceDB+graph)   (RRF 融合)
```

### 检索管道

```
Phase 1    BM25 召回 (top-20)        ← rank-bm25 (BM25Plus，小 corpus 友好)
Phase 1.5  向量召回 (top-20)         ← LanceDB + paraphrase-multilingual-MiniLM-L12-v2
Phase 2    图谱扩展 (2 跳, top-10)   ← networkx，4 信号加权
Phase 3    RRF 融合三路              ← Reciprocal Rank Fusion (k=60)
Phase 4    预算控制 (4K→1M tokens)
```

### 4 信号图谱

| 信号 | 计算 | 权重 |
|---|---|---|
| 直接链接 | `[[wikilink]]` 解析 | 1.0 |
| 源重叠 | `sources[]` Jaccard | 0.6 |
| Adamic-Adar | 共同邻居 | 0.4 |
| 类型亲和力 | 同 `type` | 0.3 |

Louvain 社区检测用于可视化聚类。

## 目录结构

```
obsidian_wiki_skill/
├── SKILL.md                  # agent 工作流规范（何时触发/检索礼仪/出处标注）
├── scripts/
│   ├── parse_sources.py      # Raw/sources/ → Wiki/*.md（路由到各 parser）
│   ├── update_wiki.py        # 增量更新（manifest SHA256 追踪，append 不全量重扫）
│   ├── build_index.py        # 建 BM25 + LanceDB 索引
│   ├── build_graph.py        # 建 4 信号图谱 + pyvis 可视化 HTML
│   ├── query.py              # 检索入口（agent 调用）
│   ├── extract_assets.py     # 提取文档内嵌图片到 Wiki/assets/
│   ├── picture_caption.py    # 图片 caption 管理（list/apply）
│   ├── models.py             # 数据模型（WikiPage / ParsedDoc / ParseResult）
│   └── parsers/
│       ├── base.py
│       ├── mineru_cloud.py   # MinerU Cloud API parser（PDF/PPT/DOC/XLSX/HTML）
│       ├── mineru_local.py   # MinerU Local parser（敏感文档，本地不外发）
│       ├── mineru_common.py  # MinerU markdown → ParseResult 适配器
│       ├── docx_parser.py    # python-docx 本地解析
│       └── pdf_split.py      # PyMuPDF 仅拆页（>200 页分批），不解析内容
├── lib/                      # 前端库（vis-network / tom-select，图谱可视化用）
├── tests/                    # pytest 测试套件
├── requirements.txt          # 核心依赖
├── requirements-mineru.txt   # 本地 MinerU venv 依赖锁定（132 包，含 torch/transformers）
├── conftest.py               # pytest 配置（LanceDB basetemp 改到项目内）
└── .env.example              # 配置模板
```

### 项目侧目录（skill 运行时操作的工作区）

```
<project_root>/
├── Raw/sources/              # 源文档（不可变，脚本只读）
├── Raw/assets/               # 二进制资产
├── Wiki/                     # Obsidian vault root（LLM 生成，双链 + frontmatter）
│   ├── .obsidian/            # ⚠️ PROTECTED：Obsidian 独占，脚本绝不触碰
│   ├── entities/ concepts/ sources/ comparisons/
│   └── .graph/index.html     # pyvis 图谱
├── .index/                   # BM25 + LanceDB + graph.json + manifest.json
├── purpose.md                # 知识库目标
└── schema.md                 # 页面类型与 frontmatter 规范
```

## 安装

### 1. 核心依赖（轻量，CPU 即可）

```bash
python -m venv .venv
# 国内推荐清华源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

核心依赖：`rank-bm25`、`lancedb`、`sentence-transformers`、`networkx`、`python-docx`、`PyMuPDF`、`pyvis`。

### 2. Embedding 模型

`paraphrase-multilingual-MiniLM-L12-v2`（多语言，~50MB，CPU 友好）。

国内网络推荐从 [ModelScope](https://modelscope.cn) 下载（HuggingFace 镜像对权重文件 xet 分发不可靠）：

```python
from modelscope import snapshot_download
snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
```

### 3. 文档解析后端（可选但推荐）

默认按扩展名路由：

| 扩展名 | 默认后端 | 敏感文档 | 说明 |
|---|---|---|---|
| `.pdf` | MinerU Cloud API | MinerU Local | >200 页自动拆分 |
| `.pptx` | MinerU Cloud API | MinerU Local | 需 mineru ≥ 3.1.0 |
| `.doc`/`.ppt`/`.xls`/`.xlsx`/`.html` | MinerU Cloud API | — | 旧二进制格式 |
| `.docx` | python-docx | — | 结构化 XML，本地足够 |

**MinerU Cloud**（非敏感文档默认）：需 `MINERU_API_TOKEN`（从 https://mineru.net/apiManage 获取）。

**MinerU Local**（敏感文档，数据不外发）：可选安装，独立 venv 隔离重型依赖：

```bash
python -m venv <mineru_venv>
<mineru_venv>/Scripts/pip install -r requirements-mineru.txt
```

> 未安装本地 MinerU 不影响非敏感文档（走 Cloud）和 `.docx`（走 python-docx）的解析。敏感文档若标记本地但 venv 缺失，会直接抛 `FileNotFoundError`——**不静默回退 Cloud**，避免敏感内容外发。

### 4. 配置

复制 `.env.example` 为 `.env` 并填入 token：

```bash
cp .env.example .env
```

或直接用系统环境变量：`MINERU_API_TOKEN` / `MINERU_PYTHON_EXE` / `MINERU_PDF_SENSITIVE`。

## 使用

### 首次构建

```bash
# 1. 起草 purpose.md / schema.md，重组源文档到 Raw/sources/
# 2. 解析文档 → Wiki/*.md（由 agent/LLM 按 schema 生成）
# 3. 建索引
PYTHONDONTWRITEBYTECODE=1 python scripts/build_index.py <project_root>
# 4. 建图谱
PYTHONDONTWRITEBYTECODE=1 python scripts/build_graph.py <project_root>
# 5. 用 Obsidian 打开 Wiki/ 目录（必须指向 Wiki/，不是 project_root）
```

### 增量更新

```bash
# 扫描变更（manifest SHA256 追踪，未变更零开销跳过）
PYTHONDONTWRITEBYTECODE=1 python scripts/update_wiki.py <project_root>
# 输出 new / modified / deleted / unchanged 列表
# 对 new/modified 由 agent 生成 frontmatter/摘要/衍生页
# 重建索引与图谱（秒级）
```

### 检索（agent 核心）

```bash
# snippet 模式（默认，每结果 200 字片段）
PYTHONDONTWRITEBYTECODE=1 python scripts/query.py <project_root> "<query>" --k 5 --json

# 全文模式（问具体数值/流程/对比时用）
PYTHONDONTWRITEBYTECODE=1 python scripts/query.py <project_root> "<query>" --k 5 --read-full --json
```

**查询预处理**（agent 调用前必做）：提取产品名/术语 → 中英互译扩展 → 拼接增强查询。例：`"Acme 前雷达的频率"` → `"Acme 前雷达 Front Radar 频率 frequency 60fps"`。

**score 解读**：RRF 融合后典型范围 0.015–0.035，看相对 gap 不看绝对值——top1 是 top2 的 2 倍以上为高置信。`method` 字段：`fused`（三路融合）/ `bm25`（仅关键词）/ `vector`（仅语义）/ `graph`（图谱邻居，置信度低）。

### 图谱邻域查询

```bash
python scripts/query.py <project_root> "<X>" --k 5 --json   # 检索自带图谱扩展
# 或直接读 .index/graph.json 手动遍历 edges
```

## 脚本写边界（强制安全约束）

所有脚本的读写范围严格隔离，防止破坏源数据或 Obsidian 配置：

| 脚本 | 可写 | 禁止触碰 |
|---|---|---|
| `parse_sources.py` | `Wiki/*.md` | `.obsidian/` / `Raw/` / `.index/` |
| `build_index.py` | `.index/` | `.obsidian/` / `Wiki/*.md` / `Raw/` |
| `build_graph.py` | `Wiki/.graph/` + `.index/graph.json` | `.obsidian/` / `Wiki/*.md` / `Raw/` |
| `update_wiki.py` | `Wiki/*.md` + `.index/manifest.json` | `.obsidian/` / `Raw/` |
| `query.py` | 无（只读） | 全部 |

`.obsidian/` 目录由 Obsidian 独占管理，含 vault 级配置，任何脚本强制绕过。

## 已知约束

- **Windows + WorkBuddy 沙箱**：`.pyc` / `.pytest_cache` / junction 路径需特殊处理；pytest 加 `-p no:cacheprovider`，Python 运行设 `PYTHONDONTWRITEBYTECODE=1`
- **LanceDB basetemp**：沙箱拦截系统 Temp 目录，`conftest.py` 将 pytest basetemp 改到项目内
- **Obsidian vault 根**：必须指向 `Wiki/`，不可设为 `project_root`（否则 `Raw/sources/` 下 `.md` 文件混入图谱成孤立幽灵节点）
- **小 corpus BM25**：默认 BM25Plus（BM25Okapi 的 IDF 在 < 100 文档时易产生 0 值）
- **前端资源本地化**：图谱 HTML 在 `file://` 协议下 CORS 拦截 CDN，前端库已本地化到 `lib/`

## 出处标注规范

agent 用本 skill 回答问题时**强制标注出处**：

```
<直接回答，结论先行>

<支撑细节，每个事实后标 [来源: Wiki/xxx.md]>

---
**引用来源：**
- [1] Wiki/entities/acme-visioncam-front.md — 产品实体页
  - 原始文档: Raw/sources/Datasheet/.../xxx.docx
```

无结果时诚实说"知识库中未找到"，**绝不编造**。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider
```

覆盖：索引构建、图谱构建、BM25/向量/融合检索、各 parser、manifest 增量、图片 caption。

## 技术栈

- **检索**：rank-bm25 (BM25Plus) + LanceDB + sentence-transformers (MiniLM-L12-v2)
- **图谱**：networkx（4 信号 + Louvain 社区）+ pyvis 可视化
- **融合**：RRF (Reciprocal Rank Fusion, k=60)
- **解析**：MinerU Cloud/Local + python-docx + PyMuPDF（仅拆页）
- **存储**：Obsidian vault（Markdown + 双链 + frontmatter）
