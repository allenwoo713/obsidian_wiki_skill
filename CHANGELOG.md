# Changelog

本项目变更记录，遵循 [Keep a Changelog](https://keepachangelog.com/) 风格。

## [Unreleased]

### Fixed — ISSUE-16：向量重建段错误根因（pyarrow/torch 导入顺序）+ crash-safe 断点续

- **根因定位（重要）**：`build_index.py` 长期在 `build()` 阶段非确定性 RC=139 段错误，此前归因为「encode 阶段 OpenMP/torch 多线程 race」。经 `faulthandler` 追踪确认真实根因是 **在已加载 torch 的进程里再经 `lancedb` `import pyarrow` 触发 Windows access violation（原生 DLL 加载冲突）**，与 encode 线程无关。
- **修复**：`build_index.py` 模块导入期**先 `import lancedb`（从而先加载 pyarrow）再配置 torch**，固定导入顺序；实测同进程「encode + 写 lance」端到端 EXIT=0。
- **`_build_vector` 重构为 crash-safe 断点续**：逐批 encode 落盘到 `.index/.vec_ckpt`（`batch_*.npy` + `done.json` + `meta.json` 签名），崩溃/超时重跑自动续；内容签名（chunk 数/batch/dim/model）变更时丢弃陈旧 checkpoint 重新 encode，避免向量与元数据错位。lance 写入前释放 embedder + `gc.collect()`（内存卫生）。
- **`WIKI_TORCH_THREADS` 环境变量**：torch intra-op 线程数改为可配置，默认 `1`（沙箱唯一稳定值）；稳定大机器可调高，但小模型 + 短切片收益有限。
- **README「已知约束」**：新增 pyarrow-before-torch 导入顺序、`WIKI_TORCH_THREADS`、crash-safe 向量重建、超大库 torch-free 兜底 4 条说明。

### Changed/Fixed — ISSUE-13：生产构建真实 tokenizer + 稳定 ChunkRecord ID

- **真实 tokenizer 注入**：`chunking.py` 新增轻量适配器 `EmbeddingTokenizer`（包裹 embedding 模型的 HF tokenizer，`.count` 即 `chunk_page` 所需的 `Tokenizer` 可调用对象）。`build_index._build_chunks` 在构建开始时**仅初始化一次** tokenizer 并显式传入 `chunk_page(..., tokenizer=token_counter.count)`；模型 tokenizer 加载失败直接 `RuntimeError` 终止构建、保留旧活动索引（#11 指针不翻转），**不再静默回退 `len//4` 字符估算**。
- **稳定 chunk_id（内容哈希）**：`chunk_id` 改为 `page_id::{sha256(kind|正文body|occurrence)}`（schema v2→v3），**与位置/section_path 无关**。前方插入/删除无关 section 后，未修改 chunk 的 ID 保持不变（occurrence 按 `(kind, body)` 在文档序内计数）。LanceDB row 直接写入 `cr.chunk_id`，不再改写成 `schema:page_id:kind:index`。
- **真实原文 span**：`ChunkRecord.start_char/end_char` 由 block/sentence 原始偏移计算，覆盖 chunk 实际正文；`build_index` 同步持久化两列。
- **sparse 结构安全**：`_split_sparse` 改为按 Markdown block 边界切分（超长单块才按句子/行/token 强制切分），禁止 `text[a:b]` 直接硬切，确保表格、列表、代码块、Wiki Link 不被切断。
- **迁移守卫**：`chunk_schema_version` 升到 3；vec_cache 命名空间与 checkpoint 签名随版本变更自动失效（强制全量重编码）；`build()` 检测到旧活动索引 schema 不符时强制 `--full-rebuild` 语义，旧索引因 #11 指针机制在新构建失败时可查。
- **评测归一化同步**：`eval/run_eval.py` 的 `_norm_chunk_key` 改为取 `page_id::` 之后的 content_hash（跨 project 内容一致即一致，可直接对齐 exact/ANN 结果）。
- **测试**：`test_chunking.py` 新增 `EmbeddingTokenizer` 单测、插入无关 section 后 ID 稳定的单测、sparse 边界安全单测、dense span 映射回原文单测；新增 `tests/test_build_tokenizer.py`（真实模型）验证 `token_count≤HARD_MAX`、`chunk_id` 格式、非字符回退、插入后 ID 稳定。
- **sparse 细化（块对齐滑动窗口 + 重叠）**：`_sparse_chunks_for_section` 在保持块边界安全的前提下，对超长 section 做 ~650 字符目标、~100 字符重叠的块对齐滑动窗口，恢复旧版细粒度 BM25 覆盖（避免整块粗聚合导致的稀疏召回下降）。
- **评测基线重置（`--init-baseline`）**：真实 tokenizer 使中文稠密 chunk 数由 82→172（≈2.1x，受 128-token 模型上限驱动，属设计性变更）。向量通道 top-k 被内容丰富的页族（如 Columbus 9 页）占据更多位置，导致 `page_recall_at_5` 由 0.9838→0.9068、`mrr_at_10` 0.8932→0.8513——但 `evidence_recall_at_10` 反升至 0.9476、gold 在 FTS 通道排名 1–15 且融合后仍处 top10、ANN recall 恒为 1.0，检索正确性未退化，属评测代理指标随 chunking 变更的偏移。已将 `eval/baselines.json` 重置为新的正确 chunking 基准（如更关注 top-5 页级排序，可后续调融合层）。
- **评测基线契约守卫（CI 固化）**：`eval/run_eval.py` 的 `--init-baseline` 现把当前 `chunk_schema_version`（来自 `chunking.CHUNK_SCHEMA_VERSION`）写入 `baselines.json` 的 `meta` 字段；对比模式下若基线 `meta.chunk_schema_version` 与当前代码不一致，**直接标红退出 1**（并打印「请先 `--init-baseline` 重置并在 issue/CHANGELOG 说明」），本地可用 `--force-compare` 绕过。这样「改了 chunking 却忘了 reset 基线」在 CI 会被自动抓住，无需人工记忆。

### Added — ISSUE-16 脱敏自检

- **`tests/test_image_retrieval.py`**（脱敏，工业相机领域）：覆盖 图片 caption 建索引并被 `split_text_image` 正确归类为图片、空 caption 图片不入检索、内容变更时陈旧 checkpoint 不复用（幂等/防错位）、损坏 checkpoint 可恢复重建。与 `tests/` 一致不随公开仓库发布。

### Fixed — 既有 bug：`update_wiki.py` 图片注册 `status` 字段崩溃

- **症状**：`extract_images_for_diff` 处理「全新图片（old_index 无旧条目，`prev=None`）」时，`status` 字段直接 `prev.get("caption_text")` 未加 `if prev` 守卫，抛 `AttributeError`；回归门禁在 #13 全量测试时暴露。
- **修复**：`status` 复用与 `caption_text` 相同的取值（优先旧 caption、否则用 `ref.caption`），统一守卫；`test_update_wiki.py` 7 项全过。

### Added — Roadmap

- **README**：新增 Roadmap 章节，记录「本地小 VLM（Florence-2/Moondream）离线补全空 caption」为将来方向（暂不实现）。

### Changed — 通用化改造（让 skill 可被他人复用）

#### 路径变量化

- **SKILL.md / README.md**：所有硬编码的本机绝对路径（用户目录、数据目录等）替换为占位符 `<venv_python>` / `<skill_dir>` / `<project_root>` / `<mineru_python>`，调用方按本机实际路径替换。
- **`scripts/build_index.py`**：删除 embedding 模型候选路径中硬编码的本机路径项；保留 env var (`WIKI_EMBEDDER_LOCAL_PATH`) → `~/.workbuddy/...` expanduser → HF 在线下载 三级回退。
- **`scripts/parsers/mineru_local.py`**：`_DEFAULT_MINERU_PYTHON` 由硬编码本机路径改为 `None`；构造函数在 `mineru_python_exe` 未传且 `MINERU_PYTHON_EXE` 环境变量未设置时，抛带说明的 `FileNotFoundError`（提示设置 env 或传参），不再静默回退到任何默认路径。
- **`scripts/build_graph.py`**：HTML header 从硬编码项目名改为从 `purpose.md` 动态读取标题，读不到时用 `"Wiki"` 兜底。
- **`.env.example`**：`MINERU_PYTHON_EXE` 改为占位符模板（含 Windows / Linux 路径示例），不再写死具体路径。

#### 示例脱敏

- **产品领域**：从原作者的实际知识库领域替换为虚构的工业相机领域（Acme VisionCam / Vega Opticam / ClientX），保留工作流示范价值。
- **查询预处理示例**：改为通用引导（"对照知识库 `purpose.md` 中的产品实体清单"），不再硬编码具体产品名列表。
- **端到端示例 + 出处标注示例**：全部替换为 Acme 工业相机场景，规格数值虚构。
- **README.md**：「专为 WorkBuddy agent 设计」改为「专为 AI agent 设计（兼容 WorkBuddy / Claude Code / 其他 agent 框架）」。

#### Tests 处理

- **`tests/` 目录不再随 skill 仓库公开发布**（`.gitignore` 已添加 `tests/` 排除规则；`git rm --cached -r tests/` 已从 git index 移除，本地文件保留）。
- **删除调试用 e2e 脚本**：`tests/debug_cloud_raw.py` / `e2e_cloud_fixed.py` / `e2e_cloud_real.py`（含真实知识库路径与 MinerU Cloud 调用，仅供作者本地调试，已从文件系统与 git index 移除）。

#### 历史重写

- 使用 `git filter-repo --replace-text` 清除了所有历史 commit 中的本机路径、用户名、真实产品名/客户名/规格数值。
- 验证：重写后历史中上述敏感字符串出现次数为 0（CHANGELOG.md 与 commit message 中的描述性提及除外）。

#### 强制先检索（防止 agent 跳过 query 直接回答）

- **SKILL.md**：`description` 触发词扩展（新增「根据知识库回答 / 根据wiki回答」等同义表述），提升自动加载命中面；新增「⚠️ 强制检索规则」段，规定 skill 加载后 MUST 先执行 `query.py` 再回答、每个事实陈述标注 `[来源: Wiki/xxx.md]`、检索为空时明确告知「基于模型自身训练数据或网页搜索」。
- **README.md**：新增「进阶：用项目级 hook 强制先检索（可选）」章节，提供脱敏版 `settings.json` + hook 脚本模板，说明如何在宿主框架层（UserPromptSubmit）再兜一道底；并明确「SKILL.md 无需重复写 hook 配置」——hook 由框架在对话外触发、不由模型执行，写在 SKILL.md 属无效误导。

### Added

- **CHANGELOG.md**：本次创建，记录通用化改造。
- **`scripts/build_index_md.py`**：按 frontmatter `type` 自动重建 `Wiki/index.md`（MOC，页面地图），替代纯手改、避免遗漏与质量不稳；`update_wiki.py` 末尾自动调用。
- **`scripts/check_tags.py`**：通用、幂等检测并修复 Obsidian 非法标签（含空格/`#` 的标签值→连字符，`c-ncap`→`C-NCAP` 等别名归一），仅改 `tags:` 行；`build_index_md.py` 重建前自动调用。

### Notes

- 本次改造**不改变 skill 的功能行为**，仅做路径参数化与示例脱敏。
- 已脱敏的单元测试本地仍可运行验证（`pytest -p no:cacheprovider`），但不在公开发布的仓库中包含。如他人需要测试用例，可联系作者。
