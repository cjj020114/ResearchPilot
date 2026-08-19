# 查询扩展与低置信二次检索

> 整理说明：结合 RAG 评测/系统综述中的常见模块描述整理。
> 主要参考：[Evaluation of RAG Survey, arXiv:2405.07437](https://arxiv.org/abs/2405.07437)；[Gao et al. RAG Survey](https://arxiv.org/abs/2312.10997)

## 为什么需要查询侧增强

用户问题常常过短、含糊或带口语。直接用原句检索，容易落入词面不匹配。查询侧增强的目标是：在不改变用户真实意图的前提下，提高召回覆盖面。

## 常见技术

- **Query Rewriting**：把口语问题改写成检索友好的陈述。
- **Multi-Query / Sub-Queries**：拆成若干子问题分别检索，再按 chunk id 合并去重（保留更高分）。
- **Step-back Prompting**：先问更抽象/上位概念，扩大相关背景召回。
- **HyDE**：先让 LLM 写假想答案再检索（成本更高，需防跑偏）。

ANCHOR_FACT_QUERY_EXPAND：查询扩展常见手段包括改写、子查询拆分与 step-back，用于提高语义召回。

## 与置信度触发的二次检索

当融合检索的最佳分数偏低时，可放大候选池（例如 candidate_k 从 20 提到 40）做第二次检索。这属于 **不确定性触发的策略循环**，综述中也常把 uncertainty-aware control 列为 Modular RAG 特征。

ANCHOR_FACT_RETRY_SEARCH：低置信时可扩大候选池做二次检索，属于不确定性触发的检索策略。

## 生成阶段注意点

无论扩展多少查询，最终回答应锚定 **用户原始问题**，避免被改写句带偏。评测时也应以原问题为准则计算 answer relevance。
