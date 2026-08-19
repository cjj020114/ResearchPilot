# Architecture

ResearchPilot 的核心链路是 `parse -> chunk -> embed -> retrieve -> rerank -> generate -> evaluate`。

## 模块边界

- `backend/app/ingestion/`：负责图片 / 表格 / 文档路由与解析，产出自有 `Element` → `Document`（含 OCR+VLM 富化）。
- `backend/app/indexing/`：负责 chunk、embedding、本地索引和 Qdrant 向量库持久化。
- `backend/app/retrieval/`：负责 BM25、dense retrieval、融合排序和 rerank。
- `backend/app/generation/`：负责基于上下文生成答案和引用信息。
- `backend/app/evaluation/`：负责 golden set 加载和 retrieval metrics 计算。
- `frontend/`：负责上传、提问、展示引用和 trace。

## 关键设计取舍

1. 优先自实现 RAG 核心链路，避免项目看起来只是框架调用。
2. 使用本地 deterministic embedding 作为兜底，使测试和演示不依赖外部模型。
3. 混合检索采用向量分数和 BM25 分数归一化融合，便于做 ablation。
4. 答案默认使用 extractive fallback，优先展示引用溯源和检索质量；后续可接入 DeepSeek、OpenAI、通义千问或 Ollama。
5. 离线知识库支持本地 JSON fallback 和 Qdrant-backed index；正式 RAG 演示推荐使用 Qdrant。
6. 不同领域知识库放在同一个 Qdrant collection 中，通过 `domain` 和 `knowledge_base_id` metadata filter 做逻辑隔离。

## 知识库隔离与查询路由

- 上传时由**文本 LLM**根据「当前知识库清单 + 文档摘要」分配到已有库，或创建新库（写入 `storage/knowledge_bases.json`，实时更新）。
- 提问时由**文本 LLM**路由到 Top2～3 个知识库再检索；不确定时返回「请选择知识库」。
- 提问前做查询扩展：改写、复杂问题分解、退步（step-back）扩展；多路检索合并。
- 低置信（最高融合分 &lt; 0.35）时二次检索并放大 `candidate_k`。
- 检索到的片段作为上下文交给文本 LLM，围绕**原始问题**生成最终答案（未配置 LLM 时回退 extractive）。
- 支持删除单篇文档，以及级联删除整个知识库（注册表 + 库内文档/chunk）。
- 配置：`LLM_API_BASE` / `LLM_API_KEY` / `LLM_MODEL`（与 VLM 分离）。

## 文档路由（image / table / document）

ResearchPilot 文档加载按第一层三路分流（无 MinerU）：

1. **图片**（png/jpg/…）→ 一次云端 VLM 调用同时得到 `ocr_text` + `vlm_caption` → `Document`
2. **表格**（csv/xlsx/xls）→ table parser → `Document`
3. **其它文档**
   - txt/log → text parser → `Document`
   - md → Markdown → `Element[]` → `Document`（文中图片再走 VLM）
   - docx/pptx/html → Unstructured → `Element[]` → `Document`
   - pdf → digital（可提取文字，页内插图仍可 VLM）或 ocr（页图 VLM；`VLM_MAX_CALLS=0` 表示无上限）
   - json/yaml/xml → **不支持**（明确拒绝）

对象模型：`Element` → 组装为 `Document`（保留 `text`）→ 按 Element 边界切 `Chunk`（图片通常 1 chunk；纯文本 `ingest_text` 无 elements 时走旧切分）。

VLM 使用 OpenAI 兼容接口（硅基流动 / 阿里云百炼等），配置 `ENABLE_CLOUD_VLM`、`VLM_API_BASE`、`VLM_API_KEY`、`VLM_MODEL`。未配置或调用失败时局部降级，不中断整份文档索引。

## 后续增强

- 增加 query rewrite、HyDE 或 multi-query。
- 接入 RAGAS 对 faithfulness 和 answer relevance 做 LLM-as-judge。
- 增加 tracing dashboard，记录 latency、token、召回片段和 rerank 分数。
