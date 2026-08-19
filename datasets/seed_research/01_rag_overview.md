# RAG 概览：从 Naive 到 Modular

> 整理说明：本文为公开综述要点的二次转述，供 ResearchPilot 离线评测与检索演示使用，非正式论文全文转载。
> 主要参考：[Lewis et al., 2020 RAG](https://arxiv.org/abs/2005.11401)；[Gao et al., RAG Survey](https://arxiv.org/abs/2312.10997)；[arXiv:2407.13193](https://arxiv.org/abs/2407.13193)；[arXiv:2410.12837](https://arxiv.org/abs/2410.12837)

## 核心定义

**检索增强生成（Retrieval-Augmented Generation, RAG）** 在回答用户问题时，先从外部知识库检索相关段落，再把检索结果与问题一并交给生成模型。与仅依赖参数记忆的 LLM 相比，RAG 更容易更新知识、降低幻觉，并便于给出可追溯引用。

## Naive RAG 的典型流水线

经典 Naive RAG 通常包含三步：

1. **Indexing**：切分文档、编码向量并写入向量库。
2. **Retrieval**：用问题向量做相似度检索，取 Top-k。
3. **Generation**：把 Top-k 上下文拼进 prompt，由 LLM 生成答案。

ANCHOR_FACT_RAG_NAIVE：Naive RAG 的标准三步是 Indexing、Retrieval 与 Generation。

## Advanced / Modular RAG 的常见增强

公开综述普遍指出，生产级系统会在 Naive RAG 上叠加：

- **查询改写 / 多查询**：改写模糊问题，或拆成子问题分别检索。
- **混合检索**：稀疏检索（如 BM25）与稠密向量检索并行，再融合排序。
- **重排序**：对候选段落用 Cross-Encoder 或轻量规则再精排。
- **路由与策略控制**：按问题类型选择知识库、是否二次检索、是否调用工具。
- **评测闭环**：用 Recall@k、MRR、Faithfulness 等指标迭代系统。

ANCHOR_FACT_RAG_MODULAR：Modular RAG 常通过查询改写、混合检索、重排序与策略路由增强 Naive RAG。

## 对科研笔记场景的启示

个人科研助手更适合 **文本知识库 + 引用溯源** 的 RAG：把论文笔记、实验记录、会议纪要切块入库；回答时返回 chunk 级引用。多模态材料可先经 OCR/VLM 转成文本，再进入同一套检索—生成链路。
