# fNIRS / NIRS 在意图解码与假肢接口中的角色

> 整理说明：公开 fNIRS/BCI 与假肢接口文献要点转述。
> 主要参考：IEEE Access 2021 hybrid sEMG/fNIRS；Frontiers in Neurorobotics 2023 combined bio-signals；hybrid bionic control reviews

## 原理简述

**近红外光谱（NIRS）/功能近红外光谱（fNIRS）** 利用近红外光在组织中的吸收差异，估计氧合血红蛋白（HbO）与还原血红蛋白（HbR）变化，从而间接反映皮层或肌肉局部血流动力学。相对 EEG，fNIRS 对电噪声更不敏感，但时间分辨率更慢（秒级血流动力学延迟）。

ANCHOR_FACT_FNIRS：fNIRS 通过 HbO/HbR 变化反映血流动力学，时间分辨率通常慢于 EEG。

## 在假肢/BCI 中的用法

- **皮层意图**：在前额叶或运动皮层放置光极，解码手开合等高阶意图。
- **肌肉血氧**：也有工作在残肢/前臂用 NIRS 捕捉肌肉激活相关的血氧变化，与 EMG 形成互补。
- **特征窗**：研究中常见数秒滑动窗上的 HbO/HbR 峰值、均值、最小值等特征。

ANCHOR_FACT_FNIRS_USE：假肢接口中 fNIRS 可用于皮层意图或肌肉血氧特征，常与 EMG 互补。

## 局限

血流动力学延迟制约实时闭环；运动伪迹、头发/皮肤光学耦合、环境光都会影响信号质量。因此单模态 fNIRS 很少单独承担全部多自由度控制，更常作为混合接口的一路证据。
