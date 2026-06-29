---
name: obsidian_wiki_skill
description: "管理 Obsidian 知识库：setup（首次构建）、增量更新、hybrid FTS+RAG 检索、4 信号图谱。Box 作为专业知识 agent 通过 query.py 检索知识库回答用户问题，答案必须标注出处。触发词：知识库 / wiki / 检索 / 查知识库 / my wiki / 我的知识库。"
read_when:
  - 用户要把一批文档整理成知识库
  - 用户要增量更新已有知识库
  - Box 需要检索知识库回答问题
  - 用户问"我的知识库里有没有 X" / "查一下 wiki"
---

# Obsidian Wiki Skill

管理 Obsidian 知识库全生命周期：setup（首次构建）、增量更新、hybrid 检索、4 信号图谱。Box agent 通过 `query.py` 检索知识库回答用户问题。

## 何时触发

- 用户说"知识库 / wiki / 我的知识库 / 查一下 wiki / 检索知识库"
- 用户问的产品/技术问题可能已在知识库中有答案
- 用户要新增文档到知识库或更新现有文档
- 用户要看知识图谱 / wiki 结构

**不触发：** 用户说"搜我的笔记 / Obsidian vault / Notion"——那是其他工具。

## 前置条件

- Python venv: `<home>\.workbuddy\binaries\python\envs\default`
- ⚠️ 所有 Python 运行必须设 `PYTHONDONTWRITEBYTECODE=1`
- ⚠️ pytest 运行加 `-p no:cacheprovider`
- ⚠️ git commit 在 junction 路径需 `dangerouslyDisableSandbox: true`
- embedding 模型: `paraphrase-multilingual-MiniLM-L12-v2`（已从 modelscope 预下载到 venv/models/）
- Skill 路径: `~/.workbuddy/skills/obsidian_wiki_skill/`

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

---

## 工作流 1：首次 Setup（Builder）

1. 确认 `purpose.md` / `schema.md`（用 `references/` 模板起草，用户审阅）
2. 备份后重组源文档到 `Raw/sources/`
3. Box 解析文档 → 生成 `Wiki/*.md`（按 schema 的页面类型与 frontmatter 规范）
4. 构建索引：`PYTHONDONTWRITEBYTECODE=1 <venv_python> scripts/build_index.py <project_root>`
5. 构建图谱：`PYTHONDONTWRITEBYTECODE=1 <venv_python> scripts/build_graph.py <project_root>`
6. 用 Obsidian 打开 vault

## 工作流 2：增量更新（append，不全量重扫）

1. 扫描变更：`<venv_python> scripts/update_wiki.py <project_root>`
   - 输出 new / modified / deleted / unchanged 列表
2. 对 new / modified 文档：Box 解析 → 生成/更新 `Wiki/*.md`
3. 对 deleted 文档：Box 清理关联 `Wiki/*.md`，更新 `index.md`
4. 重建索引：`<venv_python> scripts/build_index.py <project_root>`
5. 重建图谱：`<venv_python> scripts/build_graph.py <project_root>`
6. 更新 `Wiki/log.md`（append 操作记录）

**增量原理：** `manifest.json` 记录每个源文件 SHA256。未变更文件零开销跳过。仅索引/图谱全量重建（秒级）。

---

## 工作流 3：Box agent 检索（核心——回答用户问题）

### 标准检索流程（4 步）

#### 步骤 1：查询预处理（关键词提取与扩展）

**BM25 是词袋模型，query 与 doc 的 token 重叠决定得分。关键词提取质量直接影响检索结果。** Box 在调用 query.py 前必须做查询预处理：

1. **识别查询语言**（中文 / 英文 / 混合）
2. **提取产品名/型号**（Acme / Vega / Vega / Front Radar / Corner Radar / Traffic Radar）
3. **提取技术术语**（FOV / FMCW / MIMO / 校准 / 安装 / 诊断 / UDP / CAN / PoE / 60fps）
4. **中英文互译扩展**：
   - "前雷达" → 加 "Front Radar"
   - "校准" → 加 "calibration"
   - "安装" → 加 "installation"
   - "诊断" → 加 "diagnostics"
   - "接口" → 加 "interface"
   - "规格" → 加 "specs"
5. **构造增强查询**：原文 + 扩展词，用空格连接

**示例：**
- 用户问"Acme 前雷达的频率是多少" → 增强查询：`Acme 前雷达 Front Radar 频率 frequency 60fps`
- 用户问"Vega 的 UDP 诊断怎么用" → 增强查询：`Vega Vega UDP 诊断 diagnostics interface`
- 用户问"校准流程" → 增强查询：`校准 calibration 流程 角度 对齐`

#### 步骤 2：执行 hybrid 检索

```bash
PYTHONDONTWRITEBYTECODE=1 <venv_python> scripts/query.py <project_root> "<增强查询>" --k 5 --json
```

默认返回 snippet 模式（每结果 200 字片段）。

**何时加 `--read-full`（Box 判断规则）：**

| 场景 | 用 --read-full | 理由 |
|---|---|---|
| 用户问具体规格数值（频率/距离/FOV/分辨率） | ✅ 是 | 数值在表格中，snippet 可能截断 |
| 用户问流程/步骤（校准/安装/诊断流程） | ✅ 是 | 流程是多步骤，snippet 不够 |
| 用户问对比分析（A vs B 差异） | ✅ 是 | 需要完整对比表 |
| 用户问"知识库里有没有 X" | ❌ 否 | snippet 足够判断有无 |
| 用户问"X 的概要" | ❌ 否 | snippet 通常够 |
| 检索结果 score 很接近（gap < 2x） | ✅ 是 | 需要更多上下文区分 |
| 只需要快速定位页面路径 | ❌ 否 | snippet 够 |

**决策原则：** snippet 能否支撑你给出完整、准确的答案？如果不确定，先用 snippet，发现不够再补 `--read-full`。

#### 步骤 3：解读检索结果

**score 读取规则：**
- RRF 融合后 score 典型范围 0.015–0.035
- **看相对 gap，不看绝对值**：top1 score 是 top2 的 2 倍以上 → 高置信；gap 小 → 多读几条
- `method` 字段：`fused` = 三路融合，`bm25` = 仅关键词命中，`vector` = 仅语义命中，`graph` = 图谱扩展
- 如果 top 结果全是 `graph` 方法 → 说明关键词和向量都没直接命中，是图谱邻居推荐，置信度低

**无结果处理：**
- 返回空结果 → 诚实说"知识库中未找到相关内容"，**绝不编造**
- 返回结果但 score 都很低 → 说"找到一些弱相关内容"，列出但标注低置信
- 返回结果但与问题不匹配 → 说"检索结果与问题匹配度低"，尝试改写查询重试一次

#### 步骤 4：合成答案 + 标注出处

**出处标注规范（强制）：**

1. **每个事实后标注来源**，格式：`[来源: Wiki/entities/acme-visioncam-front.md]`
2. **答案末尾列引用清单**：
   ```
   ---
   **引用来源：**
   - [1] Wiki/entities/acme-visioncam-front.md — Acme Front Radar 产品实体页
   - [2] Wiki/concepts/radar-specs-overview.md — 雷达规格参数总览
   ```
3. **引用原始源文档**（追溯到 Raw/sources/）：
   如果 wiki 页的 frontmatter 有 `sources: ["Raw/sources/..."]`，在引用清单中同时标注

**答案模板：**

```
<直接回答用户问题，结论先行>

<支撑细节，每个事实后标 [来源: Wiki/xxx.md]>

<如有对比/补充，继续展开>

---
**引用来源：**
- [1] Wiki/entities/acme-visioncam-front.md — Acme Front Radar 产品实体页
- [2] Wiki/concepts/radar-specs-overview.md — 雷达规格参数总览
  - 原始文档: Raw/sources/Datasheet/Acme/Acme_VisionCam_Front_Datasheet_v1.6.docx
```

### 检索礼仪

1. **标注出处**——每个事实后标 `[来源: Wiki/xxx.md]`，绝不省略
2. **只读不写**——查询不修改 wiki，只有 update_wiki.py 才写
3. **不堆砌全文**——snippet 够就别 dump 全文，除非用户要求
4. **诚实**——无结果说无结果，不编造；不确定说不确定
5. **尊重项目边界**——不要混用不同项目的知识库

---

## 工作流 4：图谱邻域查询

用户问"哪些页面链接到 X" / "X 的关联概念有哪些"：

```bash
# 方式 1：query.py 检索 + 图谱扩展（已在 hybrid_search 内置）
<venv_python> scripts/query.py <project_root> "<X>" --k 5 --json

# 方式 2：直接读 graph.json 手动遍历
<venv_python> -c "
import json
from pathlib import Path
g = json.loads(Path('<project_root>/.index/graph.json').read_text(encoding='utf-8'))
target = '<X 的标题>'
for e in g['edges']:
    if e['source'] == target or e['target'] == target:
        nbr = e['target'] if e['source'] == target else e['source']
        print(f'{nbr} (weight={e.get(\"weight\",1):.2f}, signals={e.get(\"signals\",[])})')
"
```

---

## 端到端示例

**用户：** "Acme 前雷达的探测距离和 FOV 是多少？"

**Box 内部执行：**

1. **查询预处理：**
   - 语言：中文
   - 产品名：Acme 前雷达 → Acme Front Radar
   - 术语：探测距离 → range，FOV
   - 增强查询：`Acme 前雷达 Front Radar 探测距离 range FOV`

2. **判断 --read-full：** 用户问具体数值 → ✅ 需要 --read-full（数值在表格中）

3. **执行检索：**
   ```bash
   PYTHONDONTWRITEBYTECODE=1 <venv_python> scripts/query.py <project_root> "Acme 前雷达 Front Radar 探测距离 range FOV" --k 3 --read-full --json
   ```

4. **解读结果：** top1 = Acme Front Radar（score=0.03x, fused），读全文找到规格表

**Box 回答用户：**

> Acme Front Radar 的核心规格 [来源: Wiki/entities/acme-visioncam-front.md]：
>
> | 参数 | 值 | 单位 |
> |---|---|---|
> | 最大探测距离 | 250 | m |
> | 方位角 FOV | ±60 | ° |
> | 俯仰角 FOV | ±15（远距离） | ° |
> | 距离精度 | 0.06 | m |
> | 距离分辨率 | 0.25 | m |
>
> ---
> **引用来源：**
> - [1] Wiki/entities/acme-visioncam-front.md — Acme Front Radar 产品实体页
>   - 原始文档: Raw/sources/Datasheet/Acme/Acme_VisionCam_Front_Datasheet_v1.6.docx

---

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
