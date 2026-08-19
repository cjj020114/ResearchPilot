# Resume Notes

## 项目描述

ResearchPilot 是一个面向个人科研资料的 RAG 知识助手，支持论文、笔记和网页资料的导入、检索增强问答、引用溯源和离线评测。项目重点展示 RAG 系统从文档解析到质量评估的完整工程闭环。

## 简历 Bullet

- 设计并实现面向科研文献的个人 RAG 知识助手，支持 PDF/Markdown/TXT 解析、章节级 metadata 提取、三种 chunk 策略、混合检索、rerank 和引用溯源。
- 构建 retrieval evaluation harness，基于 golden set 计算 `recall@k`、`MRR`、`context precision`，支持对 dense baseline、hybrid retrieval、hybrid + rerank 做消融对比。
- 基于 FastAPI + Streamlit + Docker Compose 实现可部署系统，返回每次请求的召回片段、向量分数、BM25 分数、重排分数和来源 metadata，便于定位检索失败与生成失败。

## 面试高频问题

### 为什么不是只用向量检索？

科研资料中有很多术语、公式名、模型缩写和指标名，纯向量检索容易漏掉精确匹配。BM25 能补足关键词召回，rerank 再提高最终上下文 precision。

### 为什么要做评测？

RAG 的质量不能只靠主观体验判断。`recall@k` 能定位召回问题，`MRR` 能衡量相关证据是否排得足够靠前，`context_precision` 能衡量 prompt 中噪声上下文比例。

### 如何降低幻觉？

答案生成阶段强制基于检索上下文，并返回引用、页码、章节和分数；当检索结果为空或分数较低时返回低置信度提示，而不是编造答案。

### 后续如何接近生产？

接入 Qdrant-backed index、异步索引任务、token/latency tracing、用户反馈闭环、RAGAS/LLM-as-judge，以及 CI 中的评测阈值检查。
