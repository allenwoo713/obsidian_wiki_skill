# Obsidian Wiki Skill

> 把一堆散乱的文档变成可被 AI agent **混合检索**的知识库：Obsidian vault（双链 + frontmatter）作存储与人工浏览后端，BM25 + 向量 + 图谱三路融合作检索后端，答案强制标注出处。

专为 AI agent 设计（兼容 WorkBuddy / Claude Code / 其他 agent 框架）——`query.py` 是 agent 检索知识库的唯一入口，返回带出处的结构化结果。

---

## 为什么不是纯向量检索

小 corpus（< 100 文档）+ 中英混合（datasheet 英文 / 校准规范中文）场景下，纯向量召回准确率不够：

| 痛点 | 本方案 |
|---|---|
| 产品型号 / 具体数值（60fps、±45°、0.1mm）向量检索不到 | BM25 关键词精确命中 |
| 同义不同形（"前向相机" vs "Front Camera"）词袋错过 | 多语言向量 + 查询预处理中英互译 |
| 孤立页面无上下文 | 4 信号图谱 2 跳扩展 |
| 三路结果难取舍 | RRF（Reciprocal Rank Fusion, k=60）融合 |

## 架构

```
Raw/sources/ ──parse──▶ Wiki/*.md ──index──▶ .index/ ──query──▶ 带出处的答案
 (不可变)      (LLM)     (双链+frontmatter)   (BM25+LanceDB+graph)   (RRF 融合)
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
| 源重叠 | sources[] Jaccard | 0.6 |
| Adamic-Adar | 共同邻居 | 0.4 |
| 类型亲和力 | 同 `type` | 0.3 |

Louvain 社区检测用于可视化聚类。

## 目录结构

```
obsidian_wiki_skill/
├── SKILL.md                  # agent 工作流规范（何时触发/检索礼仪/出处标注）
├── CHANGELOG.md              # 变更记录
├── scripts/
│   ├── _config.py           # 集中配置加载（import 即 load .env，自推导 SKILL_DIR）
│   ├── wiki / wiki.cmd      # wrapper 脚本（bash / Windows），免手拼路径
│   ├── parse_sources.py      # Raw/sources/ → Wiki/*.md（路由到各 parser）
│   ├── update_wiki.py        # 增量更新（manifest SHA256 追踪，append 不全量重扫；末尾自动重建 index.md）
│   ├── build_index.py        # 建 BM25 + LanceDB 索引
│   ├── build_graph.py        # 建 4 信号图谱 + pyvis 可视化 HTML
│   ├── build_index_md.py     # 自动重建 Wiki/index.md（MOC，按 type 分组）
│   ├── check_tags.py         # 检测并修复 Obsidian 非法标签（含空格→连字符）
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
├── references/               # 首次 setup 模板
│   ├── purpose.template.md  # 知识库目标模板
│   └── schema.template.md   # 页面类型与 frontmatter 规范模板
├── lib/                      # 前端库（vis-network / tom-select，图谱可视化用）
├── .github/workflows/ci.yml # GitHub Actions CI（语法检查 + import 健康 + .env 未入库）
├── requirements.txt          # 核心依赖
├── requirements-mineru.txt   # 本地 MinerU venv 依赖锁定（含 torch/transformers）
├── conftest.py               # pytest 配置（LanceDB basetemp 改到项目内）
└── .env.example              # 配置模板
```

> **tests/** 目录为本地开发用，**不在公开发布的 skill 仓库中包含**（已在 `.gitignore` 中排除）。如需测试用例，请联系作者。

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
python -m venv <venv>
# 国内推荐清华源
<venv>/Scripts/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
# Linux/macOS: <venv>/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

核心依赖：`rank-bm25`、`lancedb`、`sentence-transformers`、`networkx`、`python-docx`、`PyMuPDF`、`pyvis`。

### 2. Embedding 模型

`paraphrase-multilingual-MiniLM-L12-v2`（多语言，~50MB，CPU 友好）。

国内网络推荐从 [ModelScope](https://modelscope.cn) 下载（HuggingFace 镜像对权重文件 xet 分发不可靠）：

```python
from modelscope import snapshot_download
snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
```

下载后放到 `<venv>/models/paraphrase-multilingual-MiniLM-L12-v2/` 下，`build_index.py` 会优先从该位置加载。

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
# Windows: <mineru_venv>/Scripts/pip install -r requirements-mineru.txt
# Linux/macOS: <mineru_venv>/bin/pip install -r requirements-mineru.txt
```

`MINERU_PYTHON_EXE` 是**可选**配置：留空时按约定位置自动探测（`~/.workbuddy/binaries/python/envs/mineru/Scripts|bin/python`），命中就用；探测不到且未显式设置时，才会抛 `FileNotFoundError`（**不静默回退 Cloud**，避免敏感内容外发）。venv 装在其他位置时显式设置该变量覆盖自动探测。

模型下载路径由 MinerU 自身的 `mineru.json`（默认 `~/mineru.json`）管理，与本 skill 解耦——每个人的硬盘/挂载盘不同，模型存放位置不应硬编码在 skill 里。若 `mineru.json` 不在默认位置，用 `MINERU_TOOLS_CONFIG_JSON` 指定；该变量会被自动透传给 MinerU 子进程。

> 未安装本地 MinerU 不影响非敏感文档（走 Cloud）和 `.docx`（走 python-docx）的解析。

### 4. 配置

复制 `.env.example` 为 `.env` 并填入实际值：

```bash
cp .env.example .env
```

或直接用系统环境变量：`MINERU_API_TOKEN` / `MINERU_PYTHON_EXE` / `MINERU_PDF_SENSITIVE`。

## 路径变量

本 skill 的所有命令模板使用以下占位符，**调用方需按本机实际路径替换**：

| 占位符 | 含义 |
|---|---|
| `<venv_python>` | skill 核心 venv 的 python 可执行文件 |
| `<skill_dir>` | 本 skill 根目录（含 `SKILL.md`） |
| `<project_root>` | 目标知识库项目根目录 |
| `<mineru_python>` | MinerU Local venv 的 python（可选，仅敏感文档需要） |

## 使用

### wrapper 脚本（推荐，免手拼路径）

配好 `.env`（至少 `WIKI_VENV_PYTHON`）后，用 wrapper 免去每次手拼 venv_python / skill_dir：

```bash
# bash（Linux/macOS/Git Bash on Windows）
./scripts/wiki <project_root> "<query>" --k 5 --json
./scripts/wiki build-index <project_root>
./scripts/wiki build-graph <project_root>
./scripts/wiki build-index-md <project_root>       # 重建 Wiki/index.md（MOC）
./scripts/wiki check-tags <project_root> [--check] # 检查/修复非法标签
./scripts/wiki update <project_root> [--apply]

# Windows cmd.exe
scripts\wiki.cmd <project_root> "<query>" --k 5 --json
scripts\wiki.cmd build-index <project_root>
scripts\wiki.cmd build-graph <project_root>
scripts\wiki.cmd build-index-md <project_root>       # 重建 Wiki/index.md（MOC）
scripts\wiki.cmd check-tags <project_root> [--check] # 检查/修复非法标签
```

wrapper 自动定位 skill_dir，从 `.env` 读 `WIKI_VENV_PYTHON` 决定用哪个 python。`<venv_python>` 和 `<skill_dir>` 都不用再手填。

### 直接调用脚本（显式路径）

不用 wrapper 时，按以下模板手填路径：

```bash
# 首次构建
PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/build_index.py <project_root>
PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/build_graph.py <project_root>
# 用 Obsidian 打开 Wiki/ 目录（必须指向 Wiki/，不是 project_root）
```

### 增量更新

```bash
# 扫描变更（manifest SHA256 追踪，未变更零开销跳过）
PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/update_wiki.py <project_root>
# 输出 new / modified / deleted / unchanged 列表
# 对 new/modified 由 agent 生成 frontmatter/摘要/衍生页
# 重建索引与图谱（秒级）；update_wiki.py 末尾已自动重建 Wiki/index.md（MOC）
```

### 检索（agent 核心）

```bash
# snippet 模式（默认，每结果 200 字片段）
PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/query.py <project_root> "<query>" --k 5 --json

# 全文模式（问具体数值/流程/对比时用，必须 --out 落盘）
PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/query.py <project_root> "<query>" --k 5 --read-full --json --out <project_root>/tmp/rf_out.json
```

**查询预处理**（agent 调用前必做）：提取产品名/术语 → 中英互译扩展 → 拼接增强查询。例（虚构的工业相机知识库）：`"Acme 前向相机的帧率"` → `"Acme 前向相机 VisionCam Front 帧率 frame rate fps"`。

**score 解读**：RRF 融合后典型范围 0.015–0.035，看相对 gap 不看绝对值——top1 是 top2 的 2 倍以上为高置信。`method` 字段：`fused`（三路融合）/ `bm25`（仅关键词）/ `vector`（仅语义）/ `graph`（图谱邻居，置信度低）。

### 图谱邻域查询

```bash
<venv_python> <skill_dir>/scripts/query.py <project_root> "<X>" --k 5 --json   # 检索自带图谱扩展
# 或直接读 .index/graph.json 手动遍历 edges
```

### 维护命令（MOC 与标签自检）

`Wiki/index.md` 是**自动生成**的 MOC（按 `type` 分组的页面地图），禁止手改；Obsidian 标签值也**禁止含空格**（否则报"不被允许的标签名"）。

```bash
# 手动增删 Wiki 页面后，重建 index.md（会顺带自动修复非法标签）
PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/build_index_md.py <project_root>

# 仅检查非法标签（只报告不修改）
PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/check_tags.py <project_root> --check

# 自动修复非法标签（含空格/# 的标签值→连字符；c-ncap→C-NCAP 等别名归一）
PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/check_tags.py <project_root>
```

> 这两个脚本已**自动接入**：`update_wiki.py` 末尾调用 `build_index_md.py`，`build_index_md.py` 重建前调用 `check_tags.fix_invalid_tags()`。增量流程通常无需手动触发；仅当直接手改 `Wiki/*.md` 时才需补跑一次。也可用 wrapper 调用：`./scripts/wiki build-index-md <project_root>` / `./scripts/wiki check-tags <project_root> [--check]`。

## 脚本写边界（强制安全约束）

所有脚本的读写范围严格隔离，防止破坏源数据或 Obsidian 配置：

| 脚本 | 可写 | 禁止触碰 |
|---|---|---|
| `parse_sources.py` | `Wiki/*.md` | `.obsidian/` / `Raw/` / `.index/` |
| `build_index.py` | `.index/` | `.obsidian/` / `Wiki/*.md` / `Raw/` |
| `build_graph.py` | `Wiki/.graph/` + `.index/graph.json` | `.obsidian/` / `Wiki/*.md` / `Raw/` |
| `update_wiki.py` | `Wiki/*.md` + `.index/manifest.json` | `.obsidian/` / `Raw/` |
| `query.py` | 无（只读） | 全部 |
| `build_index_md.py` | `Wiki/index.md` | `.obsidian/` / `Raw/` / `Wiki/*.md`（除 index.md 外不重写） |
| `check_tags.py` | `Wiki/*.md`（仅 `tags:` 行） | `.obsidian/` / `Raw/` / 正文与标题 |

`.obsidian/` 目录由 Obsidian 独占管理，含 vault 级配置，任何脚本强制绕过。

## 已知约束

- **Windows + WorkBuddy 沙箱**：`.pyc` / `.pytest_cache` / junction 路径需特殊处理；pytest 加 `-p no:cacheprovider`，Python 运行设 `PYTHONDONTWRITEBYTECODE=1`
- **stdout 大输出段错误**：沙箱对 managed-python 的 stdout 拦截层在 >~20KB 时非确定性触发 access-violation。`--read-full` 必须用 `--out` 落盘，禁用 `| head` 等管道
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
- [1] Wiki/entities/<entity-page>.md — <实体页简述>
  - 原始文档: Raw/sources/.../xxx.<ext>
```

无结果时诚实说"知识库中未找到"，**绝不编造**。

## 测试

测试代码本地保留（`tests/` 目录），但**不在公开发布的 skill 仓库中包含**（`.gitignore` 已排除）。本地运行：

```bash
PYTHONDONTWRITEBYTECODE=1 <venv_python> -m pytest -p no:cacheprovider
```

覆盖：索引构建、图谱构建、BM25/向量/融合检索、各 parser、manifest 增量、图片 caption。

## 技术栈

- **检索**：rank-bm25 (BM25Plus) + LanceDB + sentence-transformers (MiniLM-L12-v2)
- **图谱**：networkx（4 信号 + Louvain 社区）+ pyvis 可视化
- **融合**：RRF (Reciprocal Rank Fusion, k=60)
- **解析**：MinerU Cloud/Local + python-docx + PyMuPDF（仅拆页）
- **存储**：Obsidian vault（Markdown + 双链 + frontmatter）
