---
title: "<知识库名称> Wiki Schema"
version: 1.0
updated: YYYY-MM-DD
status: draft
---

# Schema

## 页面类型（frontmatter: type）

| type | 说明 | 存放目录 |
|---|---|---|
| entity | 实体页（一个产品/主题一页） | Wiki/entities/ |
| concept | 通用技术概念页 | Wiki/concepts/ |
| specs | 规格参数详细页 | Wiki/concepts/ |
| installation | 安装/部署规范页 | Wiki/concepts/ |
| calibration | 校准/配置规范页 | Wiki/concepts/ |
| diagnostics | 诊断/排错页 | Wiki/concepts/ |
| interface | 接口协议页（通信/协议） | Wiki/concepts/ |
| source-summary | 源文档完整解析镜像（1 文档 1 页，含脚本管理的全文区 + LLM 管理的摘要区） | Wiki/sources/ |
| comparison | 跨实体对比页 | Wiki/comparisons/ |

> **按需调整**：以上是建议的页面类型。根据你的知识库领域，可增删 type。例如非工程领域可能不需要 `calibration`，而需要 `tutorial` / `faq` 等。

## Frontmatter 规范

```yaml
---
type: entity
title: "<实体标题>"
sources: ["Raw/sources/<path>/<file>.<ext>"]
tags: ["<tag1>", "<tag2>"]
related: ["[[<相关实体 1>]]", "[[<相关实体 2>]]"]
updated: YYYY-MM-DD
---
```

| 字段 | 必填 | 说明 |
|---|---|---|
| type | 是 | 见页面类型表 |
| title | 是 | 页面标题 |
| sources[] | 是 | 源文件相对路径，保证可追溯 |
| tags[] | 是 | 检索标签 |
| related[] | 是 | 双链，驱动图谱边 |
| updated | 是 | ISO 日期 |

> **可选字段**：根据领域可加 `products[]`（涉及的产品实体名）、`version`（文档版本）等。

## 命名规范

- 文件名：kebab-case，如 `acme-visioncam-front.md`
- 实体页：`{series}-{role}.md`，如 `acme-visioncam-front.md`
- 源摘要页：`{original-filename}.md`，如 `acme-visioncam-front-datasheet-v1-6.md`
- 概念页：`{concept}.md`，如 `camera-calibration.md`

## source-summary 页分区（脚本 vs LLM 管理区）

source-summary 页是源文档的**确定性解析镜像**，数据完整性由脚本保证，不由 LLM 转写。页面分两个管理区：

| 区域 | 管理者 | 内容 | 标记 |
|---|---|---|---|
| LLM 管理区 | AI agent | frontmatter（tags/related）、`## 核心内容摘要`（导航性概述）、`## 关联 Wiki 页` | 无标记，原样保留 |
| 脚本管理区 | `update_wiki.py` `write_source_fulltext()` | `## 全文内容`（`ParsedDoc.text` 机械写入）、`## 文档内嵌图片` | `<!-- BEGIN AUTO-GENERATED FULLTEXT -->` / `<!-- END AUTO-GENERATED FULLTEXT -->`，`<!-- BEGIN AUTO-GENERATED IMAGES -->` / `<!-- END AUTO-GENERATED IMAGES -->` |

**规则：**
- 脚本管理区每次解析覆盖写入，LLM 绝不触碰
- LLM 管理区首次生成后只在文档语义变更时由 AI agent 更新
- 增量更新时，`update_wiki.py` 对 new/modified 文档自动调用 `write_source_fulltext()`
- concept/entities/comparisons 等衍生页的数值**必须**从 source-summary 全文区提取，不得凭摘要臆测

## 链接规范

- 使用 `[[wikilink]]` 双链，Obsidian 与图谱引擎均可解析
- 概念页与实体页双向链接
- 对比页显式 link 参与对比的实体
- 源摘要页 link 到其衍生的实体/概念页

## 目录结构

```
Wiki/
├── index.md              # 全部页面目录（可选，自动生成）
├── entities/             # 实体页
├── concepts/            # 规格/安装/校准/诊断/接口/概念
├── sources/              # 源文档摘要
├── comparisons/          # 跨实体对比
└── .graph/
    └── index.html        # 4信号交互图谱
```

## 索引层（.index/）

```
.index/
├── manifest.json          # {path, sha256, mtime, status, wiki_pages[], images[]}
├── lance_db/             # LanceDB 向量表（增量 upsert）
├── bm25_index.pkl        # BM25 索引（pickle）
└── graph.json            # networkx 图谱序列化（全量重建，秒级）
```

## 图谱边定义（4 信号）

| 信号 | 计算方式 | 权重 |
|---|---|---|
| 直接链接 | `[[wikilink]]` 解析 | 1.0 |
| 源重叠 | frontmatter sources[] Jaccard | 0.6 |
| Adamic-Adar | 共同邻居倒数对数和（top-N per node） | 0.4 |
| 类型亲和力 | 同 type 加权 | 0.3 |

社区检测：Louvain 算法，自动发现知识聚类。孤立节点统一归入"未分类"社区，灰色着色。
