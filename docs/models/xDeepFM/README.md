# xDeepFM

## 论文

*   **标题**: xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems
*   **作者**: Jianxun Lian, Xiaoxuan Zhang, Fuzheng Zhang, Chengsheng Mao, Kaiming He, Yi Sun, Bingzheng Wei, Xing Xie
*   **年份**: 2018
*   **会议/期刊**: KDD 2018
*   **链接**: [https://arxiv.org/pdf/1803.05170.pdf](https://arxiv.org/pdf/1803.05170.pdf)

## 公式解读

xDeepFM 模型旨在通过结合显式和隐式特征交互来提高推荐系统的性能。它由三个主要组件组成：**线性部分 (Linear Part)**、**深度神经网络部分 (Deep Part)** 和 **压缩交互网络部分 (Compressed Interaction Network, CIN)**。

xDeepFM 的预测公式如下：

$$
\hat{y} = \sigma(\mathbf{w}_{linear}^T \mathbf{x} + \mathbf{w}_{dnn}^T \mathbf{a} + \mathbf{w}_{cin}^T \mathbf{p} + b)
$$

其中：
*   $\mathbf{x}$ 是原始特征向量。
*   $\mathbf{a}$ 是深度神经网络的输出。
*   $\mathbf{p}$ 是 CIN 的输出。
*   $\mathbf{w}_{linear}$, $\mathbf{w}_{dnn}$, $\mathbf{w}_{cin}$ 是对应部分的权重向量。
*   $b$ 是偏置项。
*   $\sigma$ 是 Sigmoid 激活函数。

### 1. 线性部分 (Linear Part)

线性部分与传统的线性模型类似，用于捕捉一阶特征的重要性：

$$
y_{linear} = \mathbf{w}_{linear}^T \mathbf{x}
$$

### 2. 深度神经网络部分 (Deep Part)

深度神经网络部分用于捕捉特征之间的高阶隐式交互。它将所有特征的嵌入向量拼接起来，然后输入到多层全连接网络中：

$$
\mathbf{a}^{(0)} = [\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_m]
$$
$$
\mathbf{a}^{(k)} = f(\mathbf{W}^{(k)} \mathbf{a}^{(k-1)} + \mathbf{b}^{(k)})
$$

其中 $\mathbf{e}_i$ 是第 $i$ 个特征的嵌入向量，$f$ 是激活函数。

### 3. 压缩交互网络部分 (Compressed Interaction Network, CIN)

CIN 旨在捕捉特征之间的显式高阶交互。它通过在特征维度上应用卷积操作来生成新的特征图。

CIN 的核心思想是逐层构建特征交互。第 $k$ 层的特征图 $\mathbf{X}^k$ 是由第 $k-1$ 层的特征图 $\mathbf{X}^{k-1}$ 和原始输入特征图 $\mathbf{X}^0$ 进行外积操作，然后通过卷积层压缩得到的。

$$
\mathbf{X}_h^k = \sum_{i=1}^{H_{k-1}} \sum_{j=1}^{m} \mathbf{W}_{ij}^{k,h} (\mathbf{x}_i^{k-1} \circ \mathbf{x}_j^0)
$$

其中：
*   $\mathbf{X}^0 = [\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_m]$ 是原始输入特征图。
*   $\mathbf{X}^{k-1}$ 是第 $k-1$ 层的特征图。
*   $\mathbf{x}_i^{k-1}$ 和 $\mathbf{x}_j^0$ 是特征图中的行向量。
*   $\circ$ 表示哈达玛积（Hadamard product）。
*   $\mathbf{W}_{ij}^{k,h}$ 是卷积核。

最终，所有层的特征图会进行池化操作，然后拼接起来作为 CIN 的输出。

## 代码解读

### PyTorch 实现 (`models/pytorch/xDeepFM/xdeepfm.py`)

*   **`CINArgs`, `DNNArgs`, `xDeepFMArgs`**: 使用 `dataclass` 定义模型参数，提高代码可读性和参数管理。
*   **`LinearACT(nn.Module)`**: 封装了线性层、激活函数和 Dropout，方便 DNN 和线性部分复用。
*   **`DNN(nn.Module)`**: 实现了深度神经网络部分，由多个 `LinearACT` 层堆叠而成。
*   **`CompressedInteractionNetwork(nn.Module)`**:
    *   实现了 CIN 的核心逻辑。
    *   `torch.einsum("bmd,bnd->bmnd", x0, h)`: 计算外积，生成交互特征。
    *   `x.view(x.size(0), -1, x.size(-1))`: 将交互特征展平，以便进行 Conv1d 操作。
    *   `nn.Conv1d`: 应用卷积层进行特征压缩。
    *   `torch.split`: 如果 `split_half` 为 `True`，则将特征图减半。
    *   `torch.cat(xs, dim=1)`: 拼接所有层的输出。
    *   `torch.sum(f, 2)`: 对拼接后的特征图在嵌入维度上求和，得到最终的 CIN 输出。
*   **`xDeepFM(nn.Module)`**:
    *   整合了 `CIN`、`DNN` 和 `LinearACT`。
    *   `forward` 方法中，分别计算线性部分、CIN 部分和深度部分，然后将它们的输出拼接起来，最后通过一个 `nn.Sequential` 预测最终结果。

### TensorFlow 实现 (`models/tensorflow/xDeepFM/xdeepfm.py`)

*   **`CINArgs`, `DNNArgs`, `xDeepFMArgs`**: 与 PyTorch 版本类似，使用 `dataclass` 定义参数。
*   **`LinearACT(layers.Layer)`**: TensorFlow 版本的线性层加激活函数和 Dropout。
*   **`DNN(layers.Layer)`**: TensorFlow 版本的深度神经网络层。
*   **`CompressedInteractionNetwork(layers.Layer)`**:
    *   TensorFlow 版本的 CIN 实现。
    *   `tf.einsum("bmd,bnd->bmnd", x0, h)`: 计算外积。
    *   `tf.reshape`: 展平交互特征。
    *   `layers.Conv1D`: 应用卷积层。
    *   `tf.transpose`: 由于 Conv1D 的输入要求，需要进行转置操作。
    *   `tf.split`: 如果 `split_half` 为 `True`，则将特征图减半。
    *   `tf.concat`: 拼接所有层的输出。
    *   `tf.reduce_sum`: 对拼接后的特征图在嵌入维度上求和。
*   **`xDeepFM(Model)`**:
    *   整合了 `CIN`、`DNN` 和 `LinearACT`。
    *   `call` 方法中，分别计算线性部分、CIN 部分和深度部分，然后将它们的输出拼接起来，最后通过一个 `layers.Dense` 预测最终结果。

## 测试

*   **待实现**
