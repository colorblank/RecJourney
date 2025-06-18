# Factorization Machine (FM)

## 论文

*   **标题**: Factorization Machines
*   **作者**: Steffen Rendle
*   **年份**: 2010
*   **会议/期刊**: ICDM 2010
*   **链接**: [https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

## 公式解读

Factorization Machine (FM) 是一种通用的预测模型，它将稀疏数据中的特征组合考虑在内。其核心思想是通过因子化参数来建模特征之间的二阶交叉。

FM 的预测公式如下：

$$
\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j
$$

其中：
*   $w_0$ 是全局偏置项。
*   $w_i$ 是第 $i$ 个特征的权重。
*   $\mathbf{v}_i \in \mathbb{R}^k$ 是第 $i$ 个特征的 $k$ 维隐向量（因子）。
*   $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ 表示两个隐向量的点积，用于建模特征 $i$ 和特征 $j$ 之间的二阶交叉。

二阶交叉项可以通过以下方式高效计算：

$$
\sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2} \left( \left( \sum_{i=1}^{n} \mathbf{v}_i x_i \right)^2 - \sum_{i=1}^{n} \mathbf{v}_i^2 x_i^2 \right)
$$

这个转换将计算复杂度从 $O(n^2)$ 降低到 $O(nk)$，其中 $k$ 是隐向量的维度。

## 代码解读

### PyTorch 实现 (`models/pytorch/FM/fm.py`)

*   **`second_order_interaction(x: Tensor) -> Tensor` 函数**:
    *   实现了上述公式中二阶交叉项的高效计算。
    *   `square_of_sum = torch.pow(x.sum(dim=1), 2)`: 计算所有特征嵌入向量之和的平方。
    *   `sum_of_square = x.pow(2).sum(dim=1)`: 计算每个特征嵌入向量的平方和。
    *   `interaction = 0.5 * (square_of_sum - sum_of_square)`: 根据公式计算二阶交叉项。
    *   `torch.sum(interaction, dim=1, keepdim=True)`: 对结果求和并保持维度，得到最终的二阶交叉项。

*   **`FactorizationMachine(nn.Module)` 类**:
    *   **`__init__` 方法**:
        *   `num_fields`: 列表，表示每个特征的词表大小，用于创建嵌入层。
        *   `emb_dim`: 嵌入向量的维度 $k$。
        *   `use_bias`: 是否使用全局偏置项 $w_0$。
        *   `unify_embedding`: 是否使用统一的嵌入层。如果为 `True`，则所有特征共享一个大的嵌入矩阵，通过偏移量来区分不同特征的索引；如果为 `False`，则每个特征有一个独立的嵌入层。
        *   `embedding_first`: 用于学习一阶特征权重的嵌入层（维度为1）。
        *   `embedding_second`: 用于学习二阶特征隐向量的嵌入层（维度为 `emb_dim`）。
    *   **`forward` 方法**:
        *   `bias`: 如果 `use_bias` 为 `True`，则获取全局偏置项。
        *   根据 `unify_embedding` 的设置，从 `embedding_first` 和 `embedding_second` 中获取一阶和二阶特征嵌入。
        *   `x_first_order = torch.sum(x1, dim=1)`: 计算一阶项 $\sum w_i x_i$。
        *   `x_second_order = second_order_interaction(x2)`: 调用 `second_order_interaction` 函数计算二阶交叉项。
        *   `output = x_first_order + x_second_order + bias`: 将一阶项、二阶项和偏置项相加得到最终预测结果。

### TensorFlow 实现 (`models/tensorflow/FM/fm.py`)

*   **待实现**

## 测试

*   **待实现**
