# 混合 EMG–NIRS 上肢假肢人机接口

> 整理说明：基于公开 IEEE / Frontiers 等工作的二次摘要，供检索评测。
> 主要参考：
> - Guo et al., “Toward an Enhanced HMI… Combined EMG and NIRS”, IEEE THMS, 2017. https://doi.org/10.1109/THMS.2016.2641389
> - IEEE Access 2021: Enhancing Classification Accuracy… Hybrid sEMG and fNIRS. https://doi.org/10.1109/ACCESS.2021.3099973
> - Frontiers Neurorobotics 2023: Deep learning framework with combined bio-signals. https://doi.org/10.3389/fnbot.2023.1174613

## 动机

高级肌电假手受限于残肢 **信号源数量不足** 与实时控制稳定性。把 EMG（电生理、响应快）与 NIRS/fNIRS（血氧/皮层血流动力学、另一信息通道）融合，可在不显著增加传感器节点复杂度的前提下提高分类准确率与在线操作表现。

ANCHOR_FACT_HYBRID_MOTIVE：EMG 与 NIRS 融合旨在弥补残肢肌电信号源不足并提升假肢控制分类表现。

## 代表性实验结论（公开报道摘要）

1. **前臂动作识别（Guo et al., 2017）**：在健全者与截肢者上比较 EMG-only、NIRS-only 与 hybrid；离线分类准确率与虚拟假手在线表现均显示 hybrid 显著优于单模态（p < 0.05）。
2. **经肱场景分工（IEEE Access 2021）**：Myo armband 的 sEMG 负责肘/腕类动作；fNIRS 取前额叶血流动力学负责手开合；报道健全者肘腕 sEMG 平均约 94.6%、截肢者约 74%，手部 fNIRS 健全者约 96.9%、截肢者约 94.5%。
3. **深度学习实时框架（Frontiers 2023）**：运动皮层 fNIRS + 肱二头肌 sEMG，CNN 分类八类上肢意图，报道平均准确率约 94.5%。

ANCHOR_FACT_HYBRID_RESULT：多项公开实验报告 hybrid EMG–NIRS/fNIRS 相对单模态可提升分类准确率与在线控制表现。

## 对系统设计的启示

融合可以是 **特征级拼接** 或 **决策级/分工控制**（不同模态负责不同自由度）。工程上还需处理时间尺度差异（EMG 毫秒级窗 vs fNIRS 秒级窗）、标定与用户自适应。ResearchPilot 可将此类笔记作为「假肢人机接口」知识库语料，用 RAG 回答方法对比与指标问题。
