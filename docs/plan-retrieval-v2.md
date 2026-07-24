# Retrieval v2 — 执行计划（来源：GitHub review issues #1–#12）

> 本计划由 `executing-plans` 技能驱动执行。每个任务单独 commit，commit message 引用 issue 编号。
> 分支：`feature/retrieval-v2` —— **注意**：本沙箱 git 无法创建新分支 ref（已验证 `update-ref`/`symbolic-ref` 写入被静默丢弃），故所有实现直接 commit 到 `master` 并 push；feature 分支由 Derek 在本机创建。

## 0. 现状与已有地基

最近 3 个 commit 已为多个 issue 铺垫地基，本计划在其上"升级为正式契约"而非从零：

- `3c769b4` 向量索引改为 chunk 粒度嵌入 → #1 / #8 基础（已有 `split_into_chunks`）
- `a1fcbc0` crash-safe 断点续跑 + 段错误根治（pyarrow 早于 torch 导入）→ #11 基础（已有 `.vec_ckpt`/`.npy`/`done.json`/`meta.json`）
- `a0dd762` MinerU HTML 表格 `<img>` 图提取 + 保留 alt 图注 → #12 基础

待重写/新增模块：`chunking.py`、`lexical_tokenizer.py`、`vector_index_policy.py`、`query_router.py`、`build_community_reports` 子命令；`build_index.py` / `query.py` / `build_graph.py` 大幅改造。

## 1. 依赖关系（DAG）

```
#1 Chunking ──┬─> #2 FTS ──────────────┐
   (ChunkRecord,    (LanceDB FTS,        │
    tokenizer)       lexical_tokenizer)  │
       ├─> #8 Vector ───────────────────┤
       │   (自适应策略)                   │
       ├─> #11 Index Safety ────────────┤ (原子发布依赖 encode 管线)
       │                                │
       └─> #4 Fusion ──> #3 Context ─────┤ (page-level RRF → ContextBundle)
                     └─> #5 Graph ───────┤ (文本验证扩展)
                                        │
#6 Query Planner ─────────────────────┘ (独立模块，向 #4/#3/#5 输出通道专用查询；依赖 #2 tokenizer)
#7 Incremental ────────────────────────> 依赖 #1/#2/#8 + #11
#12 Multimodal ───────────────────────> 依赖 #1(chunk anchor) + #3(ContextBundle image)
#9 Evaluation ─────────────────────────> 跨阶段门禁（需稳定 API）
#10 GraphRAG Global Search ───────────> 依赖 #5 图模型 + #6 global intent
```

**关键结论：** #1 是全局地基；#2/#8/#11 围绕 chunks 表与 encode 管线；#4→#3→#5 是核心检索链；#6 路由收口；#7/#12 是增量与多模态增强；#9 是质量门禁；#10 是独立的全局检索管道。

## 2. 阶段划分

### Phase 0 — 地基与契约（无行为变更，建立数据契约）
- P0-1 定义统一 LanceDB `chunks` 表 schema（dense + sparse 双列），manifest v2 `index_state`
- P0-2 实现 `scripts/chunking.py`：`ChunkRecord` dataclass + tokenizer-aware 分层分块骨架
- P0-3 实现 `scripts/lexical_tokenizer.py`：Jieba + bigram + exact_terms（索引/查询共用）
- P0-4 固定 LanceDB 版本范围（CI 验证），写入 manifest

### Phase 1 — P0 核心检索重写
- #1 Chunking 完整实现（tokenizer-aware、按 block/sentence overlap、保护代码块/表格/列表/WikiLink）
- #2 FTS 迁移（rank_bm25 → LanceDB 原生 FTS，`bm25_index.pkl` 标记 legacy 停止生成）
- #8 Vector 自适应策略（exact → IVF_HNSW_FLAT → IVF_HNSW_SQ，recall gate ≥0.98）
- #4 Fusion 重构（page-level RRF，EvidenceHit 双路保留，graph 移出主 RRF）
- #3 ContextBundle（tokenizer-aware packing，新 `--mode`/`--max-context-tokens`，废弃 `--read-full`）
- #5 Graph 文本验证扩展（page_id 节点、explicit/inferred 边类、1-hop 默认、文本验证保留）

### Phase 2 — P1 路由 / 安全 / 增量 / 多模态
- #6 Query Planner（**独立低耦合模块**，issue #6 已更新完整 spec）
  - 新增 `scripts/query_plan_models.py`（`QueryPlan`/`PlannerContext`/`RetrievalFeedback` + `QueryPlanner`/`RewriteProvider` Protocol；frozen dataclass，可 JSON 序列化）
  - 新增 `scripts/query_planner.py`（`QueryPlanner` 实现：Level1 确定性规划 + Level2 条件式 LLM rewrite）
  - **不导入** LanceDB / NetworkX / RRF / ContextBundle；只产出 `QueryPlan` 数据契约
  - Level1 必执行：Unicode/空格/标点/大小写规范化 → 型号/错误码/路径/IP/端口/URL/数字/单位提取 → 复用 #2 `lexical_tokenizer` 生成 `lexical_terms` → 加载项目级词典/固定中英术语 → 意图识别(LOOKUP/PROCEDURE/COMPARISON/RELATION/GLOBAL) → 默认 filters/context_mode → 原始查询保留为 `semantic_queries[0]`
  - Level2 条件式 LLM rewrite：仅当指代/口语过短/多对象多跳/意图不确定/首轮低召回时触发；`RewriteProvider` Protocol 注入，`NullRewriteProvider` 默认；超时/非法 JSON/约束校验失败自动回退 Level1；最多 2 条额外 `semantic_queries`，不得覆盖原始
  - 低召回重试 `plan_retry()`：最多 1 次（缩写展开/中英互译/同义上位/去过强约束词/补上下文实体/更通用 query）；禁删型号/数值/否定/版本/比较对象；总检索 ≤2 轮
  - 多轮上下文：CLI `--conversation-context`/`--conversation-context-file`，只消解指代不替代原问；未提供且代词不可解时保留 warning
  - `query.py` 改造：位置参数始终收**原始问题**；调 `QueryPlanner.plan()` 得 `QueryPlan`；FTS 用 `lexical_terms+exact_terms`、向量用 `semantic_queries`、图谱用 `entities/relation_intent`；低召回触发 `plan_retry()`；`--json` 输出完整 `query_plan`
  - 配置 env：`QUERY_PLANNER_REWRITE`(`auto|off|force`)/`MAX_SEMANTIC_QUERIES=3`/`MAX_RETRIES=1`/`REWRITE_TIMEOUT_SECONDS=8`/`MIN_QUERY_CHARS=6`；项目级 `<project_root>/lexicon.txt` + `<project_root>/query_synonyms.yaml`
  - 文档迁移：`SKILL.md` 删除"agent 调用前手工提取关键词/中英互译/拼接增强查询"；`README` hook 模板改为"原始问题原样传入 query.py，禁止调用前改写"；保留旧 hook 一个版本兼容说明 + 明确已部署旧 hook 需手动替换注入文本
  - 验收：同输入 Level1 完全确定可复现；无 LLM 仍可规划+检索；fake provider 离线单测/集成测试覆盖中英/口语/省略/代词/多对象对比/否定/数值
- #11 Index Safety（checkpoint 内容签名、staging/validate/atomic publish、崩溃恢复）
- #7 Incremental（`--incremental`/`--full-rebuild`，upsert/delete，图谱增量，原子发布）
- #12 Multimodal（图片 metadata 回溯父文档/页码/附近正文，ContextBundle image item）

### Phase 3 — 评测与 GraphRAG
- #9 Evaluation（`tests/` + `eval/`，≥100 queries，质量/性能基线，CI 门禁）
- #10 GraphRAG Global Search（`build-community-reports`，community_reports.jsonl，map/reduce，global intent 独立入口）

### Phase 4 — 文档与收尾
- 更新 `SKILL.md`（query.py 新 CLI、`context_text` handoff、强制出处标注）
- `README` 评测运行说明 + CI workflow
- `finishing-a-development-branch` 收尾（合并/PR）

## 3. 任务 → Issue 映射与验收锚点

| Task | Issue | 验收锚点（节选） |
|---|---|---|
| P0-2/P0-3 | #1,#2 | ChunkRecord 字段齐全；tokenizer/词典查询共用 |
| #1 | #1 | dense≤112 token；embedding truncation=0；heading 不误分配 |
| #2 | #2 | 不再生成 `bm25_index.pkl`；错误码/型号返回实际 section；中文 bigram 命中 |
| #8 | #8 | 小 corpus 不建 ANN；ANN Recall@10/20≥0.98；manifest 记录 index type |
| #4 | #4 | 融合页保留 sparse+dense 双路 evidence；graph 不进主 RRF；结果稳定 |
| #3 | #3 | 实际 tokenizer 算预算；相邻 chunk/section 进入 JSON；无 8000 静默截断 |
| #5 | #5 | 种子来自融合结果；默认 1-hop；无文本证据不进上下文；inferred 不表述为事实 |
| #6 | #6 | 模块不导入 LanceDB/NetworkX/RRF/ContextBundle；`original_query` 恒等于 `semantic_queries[0]`；FTS/vector/graph 各得通道专用查询；与 #2 共享同一 tokenizer/词典/schema version；型号/数字/单位/路径/错误码/否定/比较不因 rewrite 丢失；无 LLM 仍可确定性规划+检索；LLM 失败/超时/违约束自动回退；低召回最多 1 次 retry 不无限循环；同输入 Level1 完全确定可复现；`SKILL.md` 不再要求 agent 手工构造查询；`README` hook 要求原样传原始问题；JSON 输出含完整 QueryPlan+rewrite 来源+约束保留+retry 原因 |
| #11 | #11 | 内容同数量变更不复用旧向量；写入中断仍用旧索引；ACTIVE_INDEX 永指向成功版 |
| #7 | #7 | 改一页不重 encode 全库；删页 vector/FTS/graph 无残留；增量与全量结果一致 |
| #12 | #12 | 图片命中含 source_doc/页码/section/parent；附至少一个正文 chunk |
| #9 | #9 | 测试进公开仓库；CI 门禁（Recall 降 >2pp 失败）；参数修改须附评测 |
| #10 | #10 | global/local 独立入口；报告可追溯成员页/源；无报告时明确提示 |

## 4. 风险与决策点

1. **LanceDB 版本漂移**（#2/#8 明确要求固定 CI 验证范围）→ P0-4 锁定版本，FTS API 与 ANN API 均按锁定版本实现。
2. **新分支 ref 创建被沙箱阻断** → 已验证本环境 `git update-ref`/`symbolic-ref` 写入新 ref 被静默丢弃（仅更新既有 `master` ref 正常）。故实现直接 commit+push 到 `master`；`feature/retrieval-v2` 由 Derek 本机创建。所有 commit message 仍引用 issue 编号。
3. **范围巨大** → 单会话无法完成全部 12 issue；按 Phase 推进，每完成 3 任务做一次 checkpoint 回顾。
4. **中文分词可复现** → 应用层统一分词（Jieba + bigram）+ LanceDB whitespace FTS，避免运行时依赖 `LANCE_LANGUAGE_MODEL_HOME`。
5. **既有 chunk 逻辑复用** → `3c769b4` 的 `split_into_chunks` 需升级为 `ChunkRecord` 产出，而非废弃重写。

## 5. 执行纪律

- 每个任务：标记进行中 → 实现 → 跑验证（pytest / 构建 / 查询冒烟）→ commit（引用 issue）→ 标记完成。
- 每 3 任务暂停回顾方向。
- 遇阻塞（依赖缺失/测试反复失败/指令不清）立即停下上报，不猜测。
- 全部完成后走 `finishing-a-development-branch`。
