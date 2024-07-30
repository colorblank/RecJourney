# RecJourney

## 1.环境依赖

```
pip install -r requirements.txt
```

## 2.论文

### 阿里巴巴

| 论文标题                                                                                       | 发表年份   | URL                              | 仓库        |
| ---------------------------------------------------------------------------------------------- | ---------- | -------------------------------- | ----------- |
| Deep Interest Network for Click-Through Rate Prediction                                        | KDD 2018   | http://arxiv.org/abs/1706.06978  | models/DIN  |
| Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate | SIGIR 2018 | https://arxiv.org/pdf/1804.07931 | models/ESMM |
| Deep Interest Evolution Network for Click-Through Rate Prediction                              | AAAI 2019  | http://arxiv.org/abs/1809.03672  | models/DIEN |
| Deep Session Interest Network for Click-Through Rate Prediction                                | IJCAI 2019 | http://arxiv.org/abs/1905.06482  | models/DSIN |
| Behavior Sequence Transformer for E-commerce Recommendation in Alibaba                         | KDD 2019   | http://arxiv.org/abs/1905.06874  |             |
|                                                                                                |            |                                  |             |

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
 
 ## 3.模型评估
数据集：IJCAI 18

| 模型名称     | AUC     | LogLoss |
| ------------ | ------- | ------- |
| AutoInt      | 0.56431 | 0.09450 |
| SharedBottom | 0.55899 | 0.09820 |
|              |         |         |
