---
title: "Obsidian Wiki Builder"
summary: "将文档目录转化为结构化 Obsidian wiki + hybrid FTS+RAG 检索 + 4 信号图谱，支持增量更新。Box agent 检索后端。"
read_when:
  - 用户要把一批文档整理成知识库
  - 用户要增量更新已有知识库
  - Box 需要检索知识库回答问题
---

# Obsidian Wiki Builder

将 `Raw/sources/` 下的文档转化为结构化 Obsidian wiki，具备 hybrid FTS+RAG 检索（BM25 + LanceDB 向量 + networkx 4 信号图谱，RRF 融合）、增量 append 更新（SHA256 manifest diff）。Box agent 可通过 `query.py` 检索此知识库回答问题。

## 前置条件

- Python venv: `<home>\.workbuddy\binaries\python\envs\default`
- ⚠️ 所有 Python 运行必须设 `PYTHONDONTWRITEBYTECODE=1`（沙箱限制 .pyc 写入）
- ⚠️ pytest 运行加 `-p no:cacheprovider`（沙箱限制 .pytest_cache 写入）
- ⚠️ git commit 在 junction 路径需 `dangerouslyDisableSandbox: true`
- embedding 模型: `paraphrase-multilingual-MiniLM-L12-v2`（已从 modelscope 预下载到 venv/models/）
- 依赖: python-docx, rank-bm25, lancedb, sentence-transformers, networkx, python-louvain, pyvis, pyyaml, pdfplumber

## 目录约定

```
<project_root>/
├── Raw/sources/          # 源文档（不可变）
├── Raw/assets/           # 二进制资产
├── Wiki/                 # LLM 生成（双链 + frontmatter）
│   ├── index.md
│   ├── overview.md
│   ├── log.md
│   ├── entities/
│   ├── concepts/
│   ├── sources/
│   ├── comparisons/
│   └── .graph/index.html
├── .index/               # 索引层（BM25 + LanceDB + graph.json）
├── purpose.md
└── schema.md
```

## 工作流

### 1. 初始化新知识库

1. 确认 `purpose.md` / `schema.md`（用 `references/` 模板起草，用户审阅）
2. 重组源文档到 `Raw/sources/`（备份后移动）
3. Box 解析文档 → 生成 `Wiki/*.md`（按 schema 的页面类型与 frontmatter 规范）
4. 构建索引：`PYTHONDONTWRITEBYTECODE=1 <venv_python> scripts/build_index.py <project_root>`
5. 构建图谱：`PYTHONDONTWRITEBYTECODE=1 <venv_python> scripts/build_graph.py <project_root>`
6. 用 Obsidian 打开 vault

### 2. 增量更新（append，不全量重扫）

1. 扫描变更：`<venv_python> scripts/update_wiki.py <project_root>`
   - 输出 new / modified / deleted / unchanged 列表
2. 对 new / modified 文档：Box 解析 → 生成/更新 `Wiki/*.md`
3. 对 deleted 文档：Box 清理关联 `Wiki/*.md`，更新 `index.md`
4. 重建索引：`<venv_python> scripts/build_index.py <project_root>`
5. 重建图谱：`<venv_python> scripts/build_graph.py <project_root>`
6. 更新 `Wiki/log.md`（append 操作记录）

**增量原理：** `manifest.json` 记录每个源文件 SHA256。未变更文件零开销跳过。仅索引/图谱全量重建（秒级）。

### 3. Box agent 检索

```bash
PYTHONDONTWRITEBYTECODE=1 <venv_python> scripts/query.py <project_root> "<query>" --k 5 --max-tokens 4096 --json
```

返回 JSON 数组，每项含 path / title / score / snippet / sources / method。Box 据此组装上下文回答用户。

### 4. 人查阅

- Obsidian 打开 vault → graph view 看双链图谱
- 浏览器打开 `Wiki/.graph/index.html` 看 4 信号交互图谱

## frontmatter 规范

```yaml
---
type: product | specs | installation | calibration | diagnostics | interface | source-summary | comparison | concept
title: "页面标题"
sources: ["Raw/sources/相对路径"]
products: ["产品实体名"]
tags: ["radar", "front"]
related: ["[[其他页标题]]"]
updated: 2026-06-29
---
```

## 4 信号图谱边定义

| 信号 | 计算 | 权重 |
|---|---|---|
| 直接链接 | `[[wikilink]]` 解析 | 1.0 |
| 源重叠 | sources[] Jaccard | 0.6 |
| Adamic-Adar | 共同邻居 | 0.4 |
| 类型亲和力 | 同 type | 0.3 |

## hybrid 检索管道

```
Phase 1:   BM25 召回 (top-20)         ← rank-bm25 (BM25Plus)
Phase 1.5: 向量召回 (top-20)          ← LanceDB + paraphrase-multilingual-MiniLM-L12-v2
Phase 2:   图谱扩展 (2跳, top-10)     ← networkx graph.json
Phase 3:   RRF 融合三路               ← Reciprocal Rank Fusion (k=60)
Phase 4:   预算控制 (4K→1M tokens)
```

## 已知环境约束

- Windows + WorkBuddy 沙箱：.pyc / .pytest_cache / junction 路径需特殊处理
- embedding 模型从 modelscope.cn 下载（HF 镜像对权重文件分发有问题）
- LanceDB 在系统 Temp 目录被沙箱拦截，conftest.py 将 pytest basetemp 改到项目内
