---
name: obsidian_wiki_skill
description: "管理 Obsidian 知识库：setup（首次构建）、增量更新、hybrid FTS+RAG 检索、4 信号图谱。AI agent 通过 query.py 检索知识库回答用户问题，答案必须标注出处。触发词：知识库 / wiki / 检索 / 查知识库 / my wiki / 我的知识库。"
read_when:
  - 用户要把一批文档整理成知识库
  - 用户要增量更新已有知识库
  - agent 需要检索知识库回答问题
  - 用户问"我的知识库里有没有 X" / "查一下 wiki"
---

# Obsidian Wiki Skill

管理 Obsidian 知识库全生命周期：setup（首次构建）、增量更新、hybrid 检索、4 信号图谱。AI agent 通过 `query.py` 检索知识库回答用户问题。

## 何时触发

- 用户说"知识库 / wiki / 我的知识库 / 查一下 wiki / 检索知识库"
- 用户问的产品/技术问题可能已在知识库中有答案
- 用户要新增文档到知识库或更新现有文档
- 用户要看知识图谱 / wiki 结构

**不触发：** 用户说"搜我的笔记 / Obsidian vault / Notion"——那是其他工具。

## 路径变量（首次使用前必读）

本 skill 所有调用模板使用以下占位符，**调用方需按本机实际路径替换**：

| 占位符 | 含义 | 示例（Windows / macOS / Linux） |
|---|---|---|
| `<venv_python>` | skill 核心 venv 的 python 可执行文件 | `C:/path/to/venv/Scripts/python.exe` / `/path/to/venv/bin/python` |
| `<skill_dir>` | 本 skill 根目录（含 `SKILL.md`） | `C:/Users/<you>/.workbuddy/skills/obsidian_wiki_skill` / `~/.workbuddy/skills/obsidian_wiki_skill` |
| `<project_root>` | 目标知识库项目根目录 | `D:/path/to/MyKnowledgeBase` / `/home/<you>/projects/my-wiki` |
| `<mineru_python>` | MinerU Local venv 的 python（**可选**，仅敏感文档需要） | `C:/path/to/mineru_venv/Scripts/python.exe` / `/path/to/mineru_venv/bin/python` |

> ⚠️ **首次使用前请按本机环境把上述占位符替换为真实路径**，不要直接 copy-paste 模板。

## 前置条件

- Python venv（含本 skill 核心依赖：`rank-bm25` / `lancedb` / `sentence-transformers` / `networkx` / `python-docx` / `PyMuPDF` / `pyvis`），可执行文件路径即 `<venv_python>`
  - ⚠️ **Windows 下 venv 用 `Scripts/python.exe`，不是 Linux 布局的 `bin/python`**。后者在此环境不存在，直跑会报 `No such file or directory`（exit 127）。
  - 在 skill 正文与示例中统一用占位符 `<venv_python>` 表示上述完整路径。
- ⚠️ 所有 Python 运行必须设 `PYTHONDONTWRITEBYTECODE=1`
- ⚠️ pytest 运行加 `-p no:cacheprovider`
- ⚠️ git commit 在 junction 路径需 `dangerouslyDisableSandbox: true`
- embedding 模型: `paraphrase-multilingual-MiniLM-L12-v2`（多语言，~50MB，CPU 友好，可从 [ModelScope](https://modelscope.cn) 或 HuggingFace 下载到 venv/models/ 下）
- Skill 路径: `<skill_dir>`（即本 `SKILL.md` 所在目录）

## 标准调用模板（按本机路径替换占位符）

所有 `scripts/*.py` 调用都用以下模板。关键三点已固化：**venv python 路径、设 `PYTHONDONTWRITEBYTECODE=1`、query 大输出走 `--out` 落盘**。

```bash
# ① 按本机实际填入（首次使用时设置一次）
VENV_PY="<venv_python>"
SKILL_DIR="<skill_dir>"
PROJ="<project_root>"

# ② snippet 检索（小输出可直走管道）
PYTHONDONTWRITEBYTECODE=1 "$VENV_PY" "$SKILL_DIR/scripts/query.py" "$PROJ" "<增强查询>" --k 6 --json

# ③ 全文检索（>~20KB 必须 --out 落盘，禁直走 stdout 管道）
PYTHONDONTWRITEBYTECODE=1 "$VENV_PY" "$SKILL_DIR/scripts/query.py" "$PROJ" "<增强查询>" --k 5 --read-full --json --out "$PROJ/tmp/rf_out.json"
```

> ⚠️ 常见踩坑：Windows 下首次调用若写成 `<venv_python>` 实际指向 `.../venv/bin/python`（Linux 布局）会 `No such file or directory`(exit 127)。Windows venv 只有 `Scripts/python.exe`。见上方「前置条件」警告。

## 目录约定

```
<project_root>/
├── Raw/sources/          # 源文档（不可变）
├── Raw/assets/           # 二进制资产
├── Wiki/                 # LLM 生成（双链 + frontmatter）—— Obsidian vault root
│   ├── .obsidian/         # ⚠️ PROTECTED：Obsidian 自动创建，脚本绝不触碰
│   ├── index.md
│   ├── overview.md
│   ├── log.md
│   ├── entities/
│   ├── concepts/
│   ├── sources/
│   ├── comparisons/
│   └── .graph/index.html  # pyvis 图谱（脚本生成）
├── .index/               # 索引层（BM25 + LanceDB + graph.json）
├── purpose.md
└── schema.md
```

### ⚠️ 脚本操作边界（强制）

所有脚本的读写范围严格限于以下路径，**绝对禁止**操作 `.obsidian/`、`Raw/` 及其他目录：

| 脚本 | 可写范围 | 禁止触碰 |
|---|---|---|
| `parse_sources.py` | `Wiki/*.md` | `.obsidian/` / `Raw/` / `.index/` |
| `build_index.py` | `.index/` | `.obsidian/` / `Wiki/*.md` / `Raw/` |
| `build_graph.py` | `Wiki/.graph/` + `.index/graph.json` | `.obsidian/` / `Wiki/*.md` / `Raw/` |
| `update_wiki.py` | `Wiki/*.md` + `.index/manifest.json` | `.obsidian/` / `Raw/` |
| `query.py` | 无（只读） | 全部 |

**Obsidian vault 设置：** 必须指向 `Wiki/` 目录（不是 `project_root`），否则 Raw/sources/ 下的原始 `.md` 文件会被 Obsidian 索引，导致图谱中出现孤立幽灵节点。

**`.obsidian/` 保护：** 该目录由 Obsidian 首次打开 vault 时自动创建，含 `app.json` / `graph.json` / `workspace.json` 等 vault 级配置。所有脚本强制绕过此目录。若新增脚本需要清理 `Wiki/`，必须显式排除 `.obsidian/` 和 `.graph/`。

---

## 工作流 1：首次 Setup（Builder）

1. 确认 `purpose.md` / `schema.md`（用 `references/` 模板起草，用户审阅）
2. 备份后重组源文档到 `Raw/sources/`
3. agent 解析文档 → 生成 `Wiki/*.md`（按 schema 的页面类型与 frontmatter 规范）
4. 构建索引：`PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/build_index.py <project_root>`
5. 构建图谱：`PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/build_graph.py <project_root>`
6. 用 Obsidian 打开 vault（必须指向 `Wiki/` 目录，**不是** project_root）

## 工作流 2：增量更新（append，不全量重扫）

1. 扫描变更：`<venv_python> <skill_dir>/scripts/update_wiki.py <project_root>`
   - 输出 new / modified / deleted / unchanged 列表
2. 对 new / modified 文档：
   - **source-summary 全文区**（`## 全文内容` + `## 文档内嵌图片`）：`update_wiki.py` 步骤 1 已自动落盘 `ParsedDoc.text` 到 `<!-- BEGIN AUTO-GENERATED -->` 标记区，**不经 LLM**，保证数据完整性
   - **source-summary 的 frontmatter/摘要/related** + **entities/concepts/comparisons 衍生页**：agent 生成/更新。衍生页数值必须从 source-summary 全文区提取，不得凭摘要臆测
3. 对 deleted 文档：agent 清理关联 `Wiki/*.md`，更新 `index.md`
4. 重建索引：`<venv_python> <skill_dir>/scripts/build_index.py <project_root>`
5. 重建图谱：`<venv_python> <skill_dir>/scripts/build_graph.py <project_root>`
6. 更新 `Wiki/log.md`（append 操作记录）

**增量原理：** `manifest.json` 记录每个源文件 SHA256。未变更文件零开销跳过。仅索引/图谱全量重建（秒级）。

---

## 工作流 3：agent 检索（核心——回答用户问题）

### 标准检索流程（4 步）

#### 步骤 1：查询预处理（关键词提取与扩展）

**BM25 是词袋模型，query 与 doc 的 token 重叠决定得分。关键词提取质量直接影响检索结果。** agent 在调用 query.py 前必须做查询预处理：

1. **识别查询语言**（中文 / 英文 / 混合）
2. **提取产品名/型号**——对照知识库 `purpose.md` 中的产品实体清单（例如某知识库的产品是 `Acme VisionCam` / `Vega Opticam`，就把这些名词加入查询）
3. **提取技术术语**——按知识库领域提取（示例：相机领域常见术语 `FOV` / `帧率` / `分辨率` / `GigE` / `PoE` / `Global Shutter` / `HDR` / `校准` / `安装` / `诊断` / `接口`）
4. **中英文互译扩展**：
   - "<中文产品名>" → 加对应英文（如"前向相机" → 加 "Front Camera"）
   - "校准" → 加 "calibration"
   - "安装" → 加 "installation"
   - "诊断" → 加 "diagnostics"
   - "接口" → 加 "interface"
   - "规格" → 加 "specs"
5. **构造增强查询**：原文 + 扩展词，用空格连接

**示例（以虚构的 Acme 工业相机知识库为例）：**
- 用户问"Acme 前向相机的帧率是多少" → 增强查询：`Acme 前向相机 Front Camera 帧率 frame rate fps`
- 用户问"Vega 的 GigE 接口怎么配置" → 增强查询：`Vega Opticam GigE Vision 接口 interface configuration`
- 用户问"校准流程" → 增强查询：`校准 calibration 流程 角度 对齐`

#### 步骤 2：执行 hybrid 检索

```bash
PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/query.py <project_root> "<增强查询>" --k 5 --json
```

默认返回 snippet 模式（每结果 200 字片段）。

> ⚠️ **`--read-full` 必须用 `--out` 落盘，禁止直走 stdout 管道**：沙箱对 managed-python 的 stdout 拦截层在大输出（≈20KB+）时存在非确定性 access-violation（exit -1073741819）。正确写法：
> ```bash
> PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/query.py <project_root> "<增强查询>" --k 5 --read-full --json --out <project_root>/tmp/rf_out.json
> ```
> stdout 仅返回一行 `wrote <path> (N bytes)`（管道安全）；agent 用 Read 工具读 `tmp/rf_out.json` 取大 payload。无 `--out` 时行为不变（小输出可直走管道）。详见"已知环境约束"。

**何时加 `--read-full`（agent 判断规则）：**

| 场景 | 用 --read-full | 理由 |
|---|---|---|
| 用户问具体规格数值（帧率/分辨率/FOV/精度） | ✅ 是 | 数值在表格中，snippet 可能截断 |
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

1. **每个事实后标注来源**，格式：`[来源: Wiki/entities/<page>.md]`
2. **答案末尾列引用清单**：
   ```
   ---
   **引用来源：**
   - [1] Wiki/entities/<page>.md — <实体页简述>
   - [2] Wiki/concepts/<page>.md — <概念页简述>
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
- [1] Wiki/entities/<entity-page>.md — <实体页简述>
- [2] Wiki/concepts/<concept-page>.md — <概念页简述>
  - 原始文档: Raw/sources/<subdir>/<original-file>.<ext>
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
<venv_python> <skill_dir>/scripts/query.py <project_root> "<X>" --k 5 --json

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

> 以下示例使用虚构的 Acme 工业相机知识库，仅用于示范工作流。请按你自己知识库的实际产品/术语替换。

**用户：** "Acme 前向相机的帧率和 FOV 是多少？"

**agent 内部执行：**

1. **查询预处理：**
   - 语言：中文
   - 产品名：Acme 前向相机 → Acme VisionCam Front
   - 术语：帧率 → frame rate / fps，FOV
   - 增强查询：`Acme 前向相机 VisionCam Front 帧率 frame rate FOV`

2. **判断 --read-full：** 用户问具体数值 → ✅ 需要 --read-full（数值在表格中）

3. **执行检索：**
   ```bash
   PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/query.py <project_root> "Acme 前向相机 VisionCam Front 帧率 frame rate FOV" --k 3 --read-full --json --out <project_root>/tmp/rf_out.json
   ```
   （stdout 仅 `wrote ... (N bytes)`；agent 随后 Read `tmp/rf_out.json`）

4. **解读结果：** top1 = Acme VisionCam Front（score=0.03x, fused），读全文找到规格表

**agent 回答用户：**

> Acme VisionCam Front 的核心规格 [来源: Wiki/entities/acme-visioncam-front.md]：
>
> | 参数 | 值 | 单位 |
> |---|---|---|
> | 最大帧率 | 60 | fps |
> | 水平 FOV | ±45 | ° |
> | 垂直 FOV | ±30 | ° |
> | 像素精度 | 0.1 | mm |
> | 空间分辨率 | 0.4 | mm |
>
> ---
> **引用来源：**
> - [1] Wiki/entities/acme-visioncam-front.md — Acme VisionCam Front 产品实体页
>   - 原始文档: Raw/sources/Datasheet/Acme/Acme_VisionCam_Front_Datasheet_v2.0.pdf

---

## frontmatter 规范

```yaml
---
type: product | specs | installation | calibration | diagnostics | interface | source-summary | comparison | concept
title: "页面标题"
sources: ["Raw/sources/相对路径"]
products: ["产品实体名"]
tags: ["camera", "front"]
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
- **stdout 大输出段错误（2026-07-14 定位）**：沙箱对 managed-python 的 stdout 拦截层在进程经捕获管道写出较大数据（query.py `--read-full` JSON ≈29KB+、trivial 写 50KB 均复现）时，非确定性触发 access-violation（exit -1073741819 / 0xC0000005）。**非尺寸阈值**（64KB 成功而 50KB 崩溃），是拦截层时序竞态；纯 Python 脚本（仅 json/pathlib）也复现 → 非 torch/LanceDB 原生库问题。**规避**：`--read-full` 及任何 >~20KB stdout 一律 `> tmp/out.json 2> tmp/err.log` 重定向到文件再 Read；小输出可直走管道；禁用 `| head` 等管道。
- embedding 模型从 modelscope.cn 下载（HF 镜像对权重文件分发有问题）
- LanceDB 在系统 Temp 目录被沙箱拦截，conftest.py 将 pytest basetemp 改到项目内
- Obsidian vault 根目录 = `Wiki/`，不可设为 project_root（否则 Raw/sources/ 中 `.md` 文件混入图谱）
- 脚本绝不触碰 `Wiki/.obsidian/`，该目录由 Obsidian 独占管理

## 文档解析后端配置

默认路由（按扩展名）：

| 扩展名 | 默认后端 | 敏感时 | 说明 |
|--------|----------|--------|------|
| `.pdf` | MinerU Cloud API | MinerU Local | >200 页自动拆分；Local 原生支持 |
| `.pptx` | MinerU Cloud API | MinerU Local | Cloud/Local 均完整支持（需 mineru ≥3.1.0） |
| `.doc` / `.ppt` / `.xls` / `.xlsx` / `.html` | MinerU Cloud API | — | 旧二进制或本地不支持的格式 |
| `.docx` | python-docx | — | 结构化 XML，本地解析足够准确 |
| 其他 | 不支持 | — | 抛 `UnsupportedFormat` |

敏感文档处理（PDF / PPTX）：
- 交互模式下，新增 PDF/PPTX 会询问用户是否敏感；敏感则走 `MinerU Local`。
- 非交互模式可设置环境变量 `MINERU_PDF_SENSITIVE=1` 强制 PDF/PPTX 走本地。
- 本地解析使用独立 MinerU venv，路径由 `MINERU_PYTHON_EXE` 环境变量指定（必须指向已安装 MinerU 的 venv python 可执行文件，对应占位符 `<mineru_python>`）。

### MinerU Local（可选组件）

本地 MinerU 是**可选**的，仅在解析「敏感文档」时需要：
- **非敏感文档默认走 MinerU Cloud API**（需 `MINERU_API_TOKEN`），不依赖本地 MinerU。
- 若标记敏感但本地 venv 缺失（即 `MINERU_PYTHON_EXE` 未设置或路径不存在），`MineruLocalPdfParser` 会直接抛 `FileNotFoundError`，**不会静默回退到 Cloud**——避免敏感内容外发，符合安全边界。
- **格式支持**（取决于 mineru 版本）：
  - ≥3.1.0：PDF / 图片 / DOCX / PPTX / XLSX 全格式
  - 3.0.x：PDF / 图片 / DOCX（PPTX/XLSX 未实现，`_process_office_doc` 仅打 warning 跳过）
- 用户可完全不安装本地 MinerU，代价是丧失敏感文档的本地解析能力；非敏感文档（Cloud API）与 docx 等本地解析均不受影响。

重建本地 MinerU venv（从锁定文件）：
```bash
# 1. 创建隔离 venv（用本机 Python 3.10+）
python -m venv <mineru_venv>
# Windows: <mineru_venv>/Scripts/pip install -r <skill_dir>/requirements-mineru.txt
# Linux/macOS: <mineru_venv>/bin/pip install -r <skill_dir>/requirements-mineru.txt

# 2. 配置模型源（local 模式，避免联网下载）
# 创建 mineru.json 配置文件指向本地模型，内容形如：{"model-source": "local"}
# 详见 MinerU 官方文档

# 3. 设置环境变量
# Windows (PowerShell): $env:MINERU_PYTHON_EXE = "<mineru_venv>/Scripts/python.exe"
# Linux/macOS: export MINERU_PYTHON_EXE="<mineru_venv>/bin/python"
```
依赖版本锁定见 `<skill_dir>/requirements-mineru.txt`（含 torch / transformers / opencv 等重型依赖，刻意与 skill 的核心 venv 隔离）。

配置方式（二选一）：

1. 复制 `.env.example` 为 `.env`，填入实际路径与 token：
   ```bash
   cp <skill_dir>/.env.example <skill_dir>/.env
   ```
2. 填入 `MINERU_API_TOKEN`（从 https://mineru.net/apiManage 获取）和 `MINERU_PYTHON_EXE`（指向本机 MinerU venv 的 python）。
3. 或直接使用系统环境变量：
   - `MINERU_API_TOKEN=<your_token>`
   - `MINERU_PYTHON_EXE=<path_to_mineru_venv_python>`
   - `MINERU_PDF_SENSITIVE=0/1`

⚠️ 涉及商业敏感信息的文档（如厂商私有 datasheet），启用 Cloud 前需用户明确授权。

## 图片提取与 caption 检索（路线 B）

### 写边界表

| 脚本 | 可写范围 | 禁止触碰 |
|---|---|---|
| `extract_assets.py` | `Wiki/assets/` | `.obsidian/` / `Raw/` / `Wiki/*.md` / `.index/` |
| `parsers/*.py` | 仅返回对象，不写盘 | 任何文件 |
| `parse_sources.py` | `Wiki/assets/`（经 extract_assets） | `Wiki/*.md` / `Raw/` / `.index/` |
| `update_wiki.py` | `Wiki/assets/` / `.index/manifest.json` | `Wiki/*.md` / `Raw/` |
| `picture_caption.py` | `.index/manifest.json` | `Wiki/assets/` / `Raw/` / `Wiki/*.md` |
| `build_index.py` | `.index/`（merge 模式） | `Wiki/assets/` / `Raw/` / `Wiki/*.md` |
| `query.py` | 只读检索 | 任何文件 |

### agent caption 生成工作流

1. **列出待标注图片**：
   ```bash
   PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/picture_caption.py <project_root> list --limit N > captions.json
   ```
   stdout 输出 pending JSON（每项含 `filename`/`rel_path`/`figure_caption`/`source_doc`/空 `vlm_caption`）；
   **stderr 输出 total/done/pending 统计 + 按 source_doc 分组**。务必先读统计，勿把 pending 切片当成全集汇报。

2. **agent 逐张 Read 图片**：`Read Wiki/assets/<filename>`，结合 `figure_caption` 生成结构化描述。

3. **填充 captions.json 的 vlm_caption 与 caption_text**：
   ```yaml
   vlm_caption:
     description: "1-3 句话描述图片内容（机械尺寸图/光学原理图/接线拓扑等）"
     key_values: ["图中标注的关键数值或术语，如 ±45°, 60fps, 12V, GigE"]
     category: "图片类型分类，如 '光学规格/FOV 覆盖图'"
   caption_text: "{figure_caption}。{description} 关键数值: {key_values}。所属: {category}。"
   ```

   > **关键约束**：`caption_text` 是 `build_index._load_image_caption_pages` **唯一读取的检索字段**；`vlm_caption` 仅作 metadata 存储、检索不读。caption_text 为空的图**不会进 BM25/LanceDB**。apply 已加自愈（caption_text 空时回退 vlm_caption.description），build_index 也有兜底，但仍应按上方模板显式填好 caption_text。

   agent generate caption prompt（内部用，按知识库领域调整）：
   > 你是知识库的图片标注员。请阅读图片（源文档: {source_doc}，原图题: "{figure_caption}"）。
   > 生成：1) description（1-3 句中文描述）2) key_values（图中原值，忠于图片不臆造）
   > 3) category（图片类型）。中英文混合输出，key_values 必须忠于图中原值。

4. **写回 manifest**：
   ```bash
   PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/picture_caption.py <project_root> apply captions.json
   ```

5. **重建索引**：
   ```bash
   PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/build_index.py <project_root>
   ```

### agent 检索答案合成工作流

`query.py` 返回 text 和 images 两组。agent 合成答案时：
1. **text** → 提取段落作 prose
2. **images** → 嵌入 `![[xxx.png]]` + 引用 caption
3. 必要时 agent 直接 `Read` 图片确认细节（多模态读图）
4. **出处标注**：
   - text: `[来源: Wiki/xxx.md]`
   - image: `[来源: Wiki/assets/xxx.png, 源文档: Raw/.../xxx.<ext>]`

## 开发与维护规则（从实战教训沉淀）

### 代码修改后必须重建产物
修改 `build_graph.py` / `build_index.py` 等生成器脚本后，**立即运行该脚本**重建 HTML / 索引。不要假设"代码改了就行"——产物是用户看到的。

### 内容生成后交叉验证关键实体
生成 Wiki 内容后，对产品名、供应商、客户等关键事实做交叉验证：frontmatter `sources[]` 指向的原始文档 vs 生成内容中的声明。不一致时以原始文档为准。**云记忆或全局背景中的项目信息不得作为事实来源。**

### 中国网络环境依赖策略
- 模型权重：优先 modelscope.cn（HF 镜像 xet 机制对权重文件不可靠）
- 前端 CDN 资源：本地化到 `Wiki/.graph/lib/`（file:// 协议下 CORS 拦截 CDN）
- pip 包：设 `-i https://pypi.tuna.tsinghua.edu.cn/simple`

### 小 corpus BM25 选型
corpus < 100 文档时，默认用 BM25Plus（非 BM25Okapi）。BM25Okapi 的 IDF 公式在小 corpus 上易产生 0 值。
