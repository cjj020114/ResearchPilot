# Evaluation Guide

ResearchPilot 的评测重点是证明检索优化带来了可量化收益。

## Golden Set 格式

每行一个 JSON：

```json
{"question":"问题","relevant_chunk_ids":["chunk-id-1"],"answer":"参考答案"}
```

流程：

1. 导入论文或笔记。
2. 通过 `/api/ask` 或 index 文件查看候选 chunk id。
3. 人工标注每个问题对应的 relevant chunk id。
4. 运行 `/api/evaluate` 得到指标。

## 指标

- `recall@k`：前 k 个结果中覆盖了多少相关 chunk。
- `MRR`：第一个相关 chunk 排名越靠前，分数越高。
- `context_precision`：前 k 个结果中相关 chunk 的比例。

## 建议实验

| 实验 | 变量 | 目标 |
| --- | --- | --- |
| Dense baseline | 只用向量召回 | 建立基础指标 |
| Hybrid retrieval | 向量 + BM25 | 验证关键词对术语、公式、缩写的帮助 |
| Hybrid + rerank | 增加重排 | 验证 precision 和 MRR 是否提升 |
| Chunk ablation | fixed / recursive / heading | 验证科研章节结构是否提升引用质量 |

最终报告建议写清楚：

- 数据集规模：文档数量、chunk 数量、问题数量。
- 指标变化：baseline 到最终方案的提升。
- 失败案例：检索失败还是生成失败。
- 工程指标：平均延迟、top-k、模型、硬件或 API 成本。
