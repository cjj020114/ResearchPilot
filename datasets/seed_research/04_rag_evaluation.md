# RAG 评测：检索指标与生成指标

> 整理说明：公开 RAG 评测综述要点转述。
> 主要参考：[arXiv:2405.07437](https://arxiv.org/abs/2405.07437)；[RAGAS 相关实践与论文讨论](https://arxiv.org/abs/2309.15217)；MDPI SLR on RAG metrics

## 检索质量指标

- **Recall@k**：Top-k 中命中的相关文档比例（相对全部相关集）。
- **Hit@k（或 Success@k）**：Top-k 中是否至少命中一条相关文档（0/1），再对问题集取平均。
- **MRR（Mean Reciprocal Rank）**：第一条相关结果排名倒数的平均值，强调「相关结果排得有多靠前」。
- **Context Precision**：Top-k 里相关片段占比，关注噪声上下文。

ANCHOR_FACT_HIT_MRR：Hit@k 看 Top-k 是否命中任一相关；MRR 看第一条相关结果的排名倒数。

## 生成质量指标（含 RAGAS 思路）

仅有检索指标不够：还要看答案是否忠于上下文、是否回答问题。

- **Faithfulness（忠实度）**：答案声称的信息是否能被检索上下文支持；惩罚胡编。
- **Answer Relevance（答案相关性）**：答案是否切题、是否回应提问，而不只是堆砌上下文。
- 其他常见还有 Context Relevance、人工 Likert、EM/F1（偏抽取式 QA）。

ANCHOR_FACT_RAGAS_PAIR：Faithfulness 衡量答案是否被上下文支持；Answer Relevance 衡量答案是否切题。

## 评测集（Golden Set）注意点

Golden set 需要真实的 `relevant_chunk_ids`。占位符 id 会让 Recall/MRR/Hit 全部失真。科研演示库应先入库，再根据可定位锚点句反查 chunk id，再写入 jsonl。
