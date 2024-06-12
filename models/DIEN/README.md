# Deep Interest Evolution Network for Click-Through Rate Prediction

公司：阿里巴巴

对比 DIN，DIEN 的改进：

- 增加了 GRU 单元来编码用户历史交互的时序关系；
- 提出 AUGRU，把用户历史交互物品与候选物品编码的权重引入 GRU单元。

## 模型结构

![1717380415372](image/README/1717380415372.png)
