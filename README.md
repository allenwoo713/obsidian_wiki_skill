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
# 默认：把用户原始问题原样传入，Query Planner 自动做查询预处理（禁止调用前改写）
PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/query.py <project_root> "<用户原始问题>" --k 5 --json

# 全文模式（问具体数值/流程/对比时用，必须 --out 落盘）
PYTHONDONTWRITEBYTECODE=1 <venv_python> <skill_dir>/scripts/query.py <project_root> "<用户原始问题>" --mode full --k 5 --json --out <project_root>/tmp/rf_out.json
```

**查询预处理已内置于 Query Planner**（issue #6）：agent **无需、也不得**在调用前手工提取关键词、做中英互译或拼接增强查询——直接把用户原始问题原样传入 `query.py` 即可。`query.py` 内部会生成 FTS 词项（`lexical_terms`+`exact_terms`，型号/错误码/数字单位原样保留）、向量语义查询（`semantic_queries`，原始问题恒为第 0 条）、图谱实体（`entities`），并按意图选择 `context_mode`。无 LLM 时仍可确定性规划与检索。

**score 解读**：RRF 融合后典型范围 0.015–0.035，看相对 gap 不看绝对值——top1 是 top2 的 2 倍以上为高置信。`method`(inclusion_reason) 字段：`rrf`（FTS+向量 page-level RRF 融合）/ `graph_expansion`（图谱 1-hop 扩展，置信度低）/ `image`（图片命中）。

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
- **pyarrow 必须早于 torch 导入（重要）**：在已加载 torch 的进程里再 `import pyarrow`（经 lancedb）会触发 Windows access violation 段错误（RC=139）。`build_index.py` 已在模块导入期先 `import lancedb` 再配置 torch 以固定顺序。若你在别处编写「同进程既 encode 又写 lance」的脚本，务必让 `import lancedb`/`pyarrow` 出现在任何 torch 导入之前。
- **CPU 线程数 `WIKI_TORCH_THREADS`**：向量 encode 的 torch intra-op 线程数，默认 `1`（受限/沙箱环境唯一稳定值；多线程易触发 OpenMP race）。稳定的大机器可设 `WIKI_TORCH_THREADS=4` 等提速——但对本 skill 用的小模型（MiniLM）+ 短切片，收益有限（many-small-ops，线程同步开销常抵消收益）。
- **crash-safe 向量重建**：`_build_vector` 逐批 encode 后落盘到 `.index/.vec_ckpt`（`.npy` + `done.json` + `meta.json` 签名），崩溃/超时重跑自动断点续；内容变更（chunk 签名不符）则丢弃陈旧 checkpoint 从头 encode，避免向量与元数据错位。成功后 best-effort 清理（禁删回收站的沙箱里可能残留 `.vec_ckpt`，无害）。
- **超大库兜底（极端情况）**：正常情况下导入顺序修复已让「同进程 encode + 写 lance」稳定。万一在超大库上 lance 写入仍崩，可从 `.index/.vec_ckpt` 的 `.npy` 用一个**完全不 import torch** 的独立脚本单独执行 `table.add`（仅 `numpy.load` + lancedb 写入），彻底隔离原生库冲突。

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

## 进阶：用项目级 hook 强制先检索（可选）

`query.py` 是 agent 检索知识库的唯一入口，但 agent 有时会"自作主张"跳过检索、直接用模型自身知识回答。有两道防线可以强制"先 query 再答"：

| 防线 | 位置 | 作用域 | 依赖 skill 激活 |
|---|---|---|---|
| **强制检索规则**（本 skill 内置） | `SKILL.md` body 指令块 | 该 skill 被加载的任何会话 | 是（inject 型 skill 全文注入） |
| **项目级 hook**（可选，使用者自建） | `<project_root>/.codebuddy/settings.json` | 仅该项目 | 否（框架级，每条消息都触发） |

第一道是本 skill 自带的（`SKILL.md` 里的「强制检索规则」段）。第二道是**框架级 hook**——不依赖 skill 是否被激活，只要在这个项目里提问就生效，是更硬的兜底。

> 该机制基于 CodeBuddy / WorkBuddy 的 `UserPromptSubmit` hook（在用户提交消息后、模型处理前运行，可向 prompt 注入额外指令）。其他 agent 框架若有等价的 prompt 预处理钩子，同理适用。

### 配置步骤

在**知识库项目根**（不是 skill 目录）建两个文件：

**1. `<project_root>/.codebuddy/settings.json`**

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "bash .codebuddy/hooks/force_kb_query.sh", "timeout": 10 } ] }
    ]
  }
}
```

**2. `<project_root>/.codebuddy/hooks/force_kb_query.sh`**

```bash
#!/usr/bin/env bash
# UserPromptSubmit hook：命中知识库相关问题时，强制 agent 先 query 再回答。
# 对本项目每条消息都触发；仅当命中知识库词且非「写/入库」意图时才注入指令，否则输出 {}（no-op）。
INPUT="$(cat)"
# 触发词表按你的知识库领域自行增删（示例为虚构工业相机知识库）
KB_RE='知识库|wiki|检索|查知识库|资料库|根据文档|根据知识库|根据wiki|datasheet|规格|参数|校准|安装|接口|诊断|Acme|VisionCam'
# 负向守卫：命中知识库词但属「写/入库」意图（导入、转换、构建知识库等）时，
# 属入库任务而非查询，不强制先检索。
WRITE_RE='导入|转换|入库|构建|重建|建库|建索引|更新知识库|写入知识库|加入知识库|加进知识库|添加.*知识库|重新建立|重新构建|新增.*知识库'
if printf '%s' "$INPUT" | grep -qiE "$KB_RE"; then
  if printf '%s' "$INPUT" | grep -qiE "$WRITE_RE"; then
    printf '{}'   # 入库任务，不强制查询
  else
    cat <<'JSON'
{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"【强制检索指令】本问题与知识库相关：必须先调用本 skill 的 query.py，并把用户本轮原始问题原样传入；禁止在调用前自行改写、翻译或拼接关键词。query.py 内部 Query Planner 会生成 FTS、向量和图谱所需的通道专用查询。回答时必须使用 query.py 返回的 original_query、query_plan 和 context_text，并按要求标注来源。若检索为空，明确说明未找到。"}}
JSON
  fi
else
  printf '{}'
fi
```

### 注意事项

- **无 matcher**：`UserPromptSubmit` 不支持按工具过滤，对本项目**每条消息**都触发；是否注入由脚本里的关键词判断。`KB_RE` 为触发词表、`WRITE_RE` 为写意图负向守卫，按你的领域自行增删。
- **写/入库任务不触发**：用户意图若是「导入 / 转换 / 构建知识库」等写操作，hook 不注入检索指令（此时库中可能尚无该内容），由 `WRITE_RE` 守卫控制。
- **JSON 必须配平**：heredoc 里的注入 JSON 是 `{...{...}}` 两层，漏一个闭合 `}` 会让宿主解析失败（静默不注入）。改完用 `echo '{"prompt":"知识库测试"}' | bash force_kb_query.sh | python -m json.tool` 自测。
- **cwd 假设**：命令用相对路径 `bash .codebuddy/hooks/...`，前提是宿主调用 hook 时工作目录为项目根；不确定就改用绝对路径。
- **首次启用**：部分宿主需在设置界面「信任 / 批准」该 hook 后才生效。
- **可移植性**：此 hook 属于**使用者的项目**，不随本 skill 分发；不同项目按需各自配置。
- **旧 hook 迁移（Query Planner 上线后）**：本 skill 的 `query.py` 已内置 Query Planner，**不再要求 agent 在调用前手工构造增强查询**。若你已按旧版 README 部署过 hook，请把上面 `additionalContext` 模板替换为新版本（核心变化：要求"把用户本轮原始问题**原样**传入 query.py，禁止调用前改写/翻译/拼接关键词"，并改为使用返回的 `original_query` / `query_plan` / `context_text`）。**仓库 README 更新不会自动改动你已部署的 hook**——需你手动替换注入文本。保留一个版本兼容：旧模板仍可用，但会多一道 agent 层冗余改写（不影响正确性，仅损失可复现性与评测一致性）。

> **要不要在 `SKILL.md` 里也写 hook？——不需要。** `SKILL.md` 是给模型读的行为指令，而 hook 由宿主框架在对话外触发、不由模型执行；在 `SKILL.md` 写 hook 配置模型既不会也无法执行。"加载后必须先 query" 的行为约束已由 `SKILL.md` 的「强制检索规则」段覆盖，hook 只是框架层再加固一道，两者互补、无需重复。

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

## Roadmap（规划中，尚未实现）

- **本地小 VLM 离线补全空 caption**：目前图片 caption 依赖解析阶段（MinerU Cloud/Local）产出，若源文档未给出图注则该图 `caption_text` 为空、不进检索。规划引入一个**本地小型视觉语言模型**（如 [Florence-2](https://huggingface.co/microsoft/Florence-2-base) / [Moondream](https://huggingface.co/vikhyatk/moondream2)，均可 CPU 离线运行、体积小），在导入/建索引阶段对空 caption 的图片自动生成描述性 caption，从而把「无图注图片」也纳入检索。
  - 设计约束：与 embedding 模型一致走「env var 指定本地路径 → `~/.workbuddy/...` → HF 在线下载」三级回退；默认关闭，通过开关（如 `WIKI_VLM_CAPTION=1`）显式启用，避免拖慢常规建索引。
  - 触发范围：仅对 `caption_text` 为空的图片调用，已有图注的图片不重复生成，保证幂等与增量友好。
  - 现状：**暂不实现**，先记录为将来方向。
