# Deep Interest Network for Click-Through Rate Prediction

公司：阿里巴巴

将用户历史交互的物品与候选物品编码为权重加权在历史物品上。

## 模型结构

![1717379872084](image/README/1717379872084.png)

## 问题

### 为什么计算权重时不使用 softmax?

保留用户兴趣的比例。softmax是在指数空间计算的，会导致量纲变化。
