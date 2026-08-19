# 多模态材料如何进入文本 RAG

> 整理说明：结合公开多模态 RAG / 文档理解实践与 ResearchPilot 设计说明整理。
> 主要参考：RAG 多模态讨论（综合综述章节）；文档 OCR/VLM 工程通识

## 两种「多模态 RAG」

1. **联合多模态嵌入**：图文同一向量空间直接检索（实现与数据成本高）。
2. **VLM→文本→RAG**：用视觉语言模型把图片/扫描页转成 OCR 文本与 caption，再走标准文本检索与生成。个人科研助手更常采用后者。

ANCHOR_FACT_VLM_TEXT_RAG：个人科研场景常见路径是 VLM 转文本后再做标准文本 RAG，而非联合多模态嵌入。

## 文档路由直觉

上传文件可按模态分流：纯图走 OCR+VLM；表格走表格解析；普通文档走文本/版面解析。结构化配置文件（json/yaml/xml）通常不适合直接当「阅读型知识」入库。

ANCHOR_FACT_DOC_ROUTE：文档路由可按 image / table / document 分流，配置类 json/yaml/xml 不宜当阅读型知识入库。

## 评测含义

一旦多模态材料进入文本块，评测仍可用同一套 Hit@k、MRR、Faithfulness、Answer Relevance；关键是 golden set 的相关 chunk 必须来自真实入库结果。
