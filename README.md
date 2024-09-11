# RecJourney: 推荐系统算法实现与实验

## 项目简介
RecJourney 是一个专注于推荐系统算法实现与实验的开源项目。本项目旨在提供一个全面的推荐系统学习和研究平台,包含多个主流推荐算法的PyTorch实现,以及在公开数据集上的实验结果。

### 主要特性

- 📚 包含20+种主流推荐算法的PyTorch实现
- 🔬 在多个公开数据集上进行了实验对比
- 🛠 提供了完整的数据预处理、模型训练、评估流程
- 📊 详细的实验结果分析与可视化
- 🔧 模块化设计,易于扩展新的算法


## 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.7+
- CUDA 10.1+ (对于GPU加速)

### 安装

1. 克隆仓库:
```
git clone https://github.com/yourusername/RecJourney.git
```

2. 安装依赖:
```
pip install -r requirements.txt
```

### 使用

TODO


## 支持的算法


### 阿里巴巴

| 论文标题                                                                                       | 发表年份   | URL                              | 仓库        |
| ---------------------------------------------------------------------------------------------- | ---------- | -------------------------------- | ----------- |
| Deep Interest Network for Click-Through Rate Prediction                                        | KDD 2018   | http://arxiv.org/abs/1706.06978  | models/DIN  |
| Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate | SIGIR 2018 | https://arxiv.org/pdf/1804.07931 | models/ESMM |
| Deep Interest Evolution Network for Click-Through Rate Prediction                              | AAAI 2019  | http://arxiv.org/abs/1809.03672  | models/DIEN |
| Deep Session Interest Network for Click-Through Rate Prediction                                | IJCAI 2019 | http://arxiv.org/abs/1905.06482  | models/DSIN |
| Behavior Sequence Transformer for E-commerce Recommendation in Alibaba                         | KDD 2019   | http://arxiv.org/abs/1905.06874  | models/BST  |
| Hybrid Contrastive Constraints for Multi-Scenario Ad Ranking                                   | CIKM 2023  | https://arxiv.org/abs/2302.02636 | models/HC2  |

### 谷歌

| 论文标题                                                                                           | 发表年份 | URL                              | 仓库       |
| -------------------------------------------------------------------------------------------------- | -------- | -------------------------------- | ---------- |
| Deep & Cross Network for Ad Click Predictions                                                      | KDD 2017 | http://arxiv.org/abs/1708.05123  | models/DCN |
| DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems | WWW2021  | https://arxiv.org/pdf/2008.13535 | models/DCN |

### 华为

| 论文标题                                                                                                  | 发表年份 | URL                                                            | 仓库        |
| --------------------------------------------------------------------------------------------------------- | -------- | -------------------------------------------------------------- | ----------- |
| Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models | KDD 2021 | https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf | models/EDCN |

### 腾讯

| 论文标题                                                                                                       | 发表年份    | URL                                                                                                                                                                                                                                                                               | 仓库      |
| -------------------------------------------------------------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations | RecSys 2020 | https://github.com/tangxyw/RecSysPapers/blob/927b56b90a0d7252114131ad08e516c0ad143106/Multi-Task/%5B2020%5D%5BTencent%5D%5BPLE%5D%20Progressive%20Layered%20Extraction%20(PLE)%20-%20A%20Novel%20Multi-Task%20Learning%20(MTL)%20Model%20for%20Personalized%20Recommendations.pdf | modes/PLE |

### 快手

| 论文标题                                                                  | 发表年份 | URL                              | 仓库        |
| ------------------------------------------------------------------------- | -------- | -------------------------------- | ----------- |
| POSO: Personalized Cold Start Modules for Large-scale Recommender Systems | 2021     | https://arxiv.org/pdf/2108.04690 | models/POSO |

### 美团

| 论文标题                                                                                   | 发表年份  | URL                              | 仓库        |
| ------------------------------------------------------------------------------------------ | --------- | -------------------------------- | ----------- |
| HiNet: Novel Multi-Scenario & Multi-Task Learning with Hierarchical Information Extraction | ICDE 2023 | https://arxiv.org/pdf/2303.06095 | model/HiNet |

### 微软

| 论文标题                                                                   | 发表年份  | URL                              | 仓库        |
| -------------------------------------------------------------------------- | --------- | -------------------------------- | ----------- |
| Towards Deeper, Lighter and Interpretable Cross Network for CTR Prediction | CIKM 2023 | https://arxiv.org/pdf/2311.04635 | models/GDCN |

## 3. 模型

| 模型名称 | 标签 | 进度 |
| -------- | ---- | ---- |
| AFM      |      | 完成 |
| AutoInt  |      | 完成 |
| BST      |      | 完成 |
| CAN      |      | 完成 |
| DCN      |      | 完成 |
| DIEN     |      | 完成 |
| DIN      |      | 完成 |
| DSIN     |      |      |
| EDCN     |      | 完成 |
| ESMM     |      | 完成 |
| FiBiNet  |      |      |
| FM       |      |      |
| GDCN     |      | 完成 |
| HC2      |      |      |
| HiNet    |      | 完成 |
| MMOE     |      |      |
| PEPNet   |      |      |
| PLE      |      | 完成 |
| POSO     |      | 完成 |
| WideDeep |      | 完成 |
| xDeepFM  |      | 完成 |

## 4.模型评估

数据集：IJCAI 18

| 模型名称     | AUC     | LogLoss |
| ------------ | ------- | ------- |
| AutoInt      | 0.56431 | 0.09450 |
| SharedBottom | 0.55899 | 0.09820 |
|              |         |         |

## 贡献指南

TODO

## 引用
TODO