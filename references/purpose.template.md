---
title: "<知识库名称> 目标"
version: 1.0
updated: YYYY-MM-DD
status: draft
---

# Purpose

## 目标

构建 <领域> 知识库，作为 <团队/个人> 的权威参考与 AI agent 的检索后端。<使用者> 未来任何会话中的 <领域> 技术问题，AI agent 通过 hybrid 检索调用此知识库生成回答。

## 覆盖范围

### 产品线 / 主题域

- **<系列 A>**（<厂商/来源>）
  - <产品/主题 1>
  - <产品/主题 2>
- **<系列 B>**（<客户/项目>）
  - <产品/主题 3>

### 文档类型

| 类型 | 说明 |
|---|---|
| Datasheet | 产品规格书 |
| Installation Guideline | 安装规范 |
| Calibration Specification | 校准规范 |
| Tools User Manual | 工具手册 |
| Interface Protocol | 接口协议 |
| Example Code | 示例代码 |

## 用途

1. **工程查阅**：快速检索某产品/主题的规格/规范
2. **跨产品对比**：<系列 A> vs <系列 B> 差异
3. **技术问答**：AI agent 基于此知识库回答技术问题
4. **变更追踪**：文档版本迭代时增量更新知识库（append，不全量重扫）

## 输出语言

<简体中文 / English>，技术术语保留原文（如 <术语 1> / <术语 2>）。

## 关键问题（驱动知识库建设）

- 各产品的核心规格参数（<参数 1> / <参数 2> / <参数 3>）？
- <规范类问题>？
- <流程类问题>？
- <对比类问题>？

## 消费者

- **<使用者>**：Obsidian 浏览、双链跳转、图谱分析
- **AI agent**：hybrid 检索（BM25 + 向量 + 图谱）后回答问题

## 检索架构

```
Phase 1:   BM25 关键词召回 (top-20)         ← rank-bm25
Phase 1.5: 向量语义召回 (top-20)            ← LanceDB + multilingual-MiniLM
Phase 2:   图谱扩展 (2跳邻居, top-10)       ← networkx 4信号
Phase 3:   RRF 融合三路 → 重排              ← Reciprocal Rank Fusion
Phase 4:   预算控制 (4K→1M tokens)
Phase 5:   输出编号片段给 AI agent
```

## 增量更新机制

- `.index/manifest.json` 记录每个源文件 SHA256 + 处理状态
- 仅处理 new/modified/deleted 文件，未变更零开销
- 图谱轻量，全量重建（秒级）
- 向量索引 LanceDB upsert 增量更新
