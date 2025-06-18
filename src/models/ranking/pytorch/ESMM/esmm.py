import torch
import torch.nn as nn
from torch import Tensor


class LinearAct(nn.Module):
    """
    定义了一个带有激活函数和可选dropout的线性层。

    参数:
    - dim_in: 输入维度。
    - dim_out: 输出维度。
    - bias: 是否使用偏置。
    - act: 激活函数类型，默认为relu。可选值包括None、'relu'、'sigmoid'。
    - dropout: dropout比例，默认为0.0，即不使用dropout。
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = True,
        act: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # 初始化线性层
        self.fc = nn.Linear(dim_in, dim_out, bias=bias)
        # 根据激活函数类型初始化激活函数模块
        if act is None:
            self.act = nn.Identity()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError(f"Invalid activation function: {act}")
        # 根据dropout比例初始化dropout模块，如果比例为0，则使用Identity替代
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播。

        参数:
        - x: 输入张量。

        返回:
        - 经过线性变换、激活函数和dropout处理后的输出张量。
        """
        # 应用线性变换
        x = self.fc(x)
        # 应用激活函数
        x = self.act(x)
        # 应用dropout
        x = self.dropout(x)
        return x


class DNN(nn.Module):
    """
    深度神经网络类，继承自nn.Module。

    该类用于构建多层感知器（MLP）类型的深度神经网络模型。

    参数:
    - dims: 层维度列表，表示每一层的神经元数量。
    - bias: 是否使用偏置项，默认为True。
    - act: 激活函数类型，默认为'relu'。
    - dropout: 遗传率，默认为0.0，表示不使用dropout。
    - last_act: 最后一层是否使用激活函数，默认为False。
    """

    def __init__(
        self,
        dims: list[int],
        bias: bool = True,
        act: str = "relu",
        dropout: float = 0.0,
        last_act: bool = False,
    ) -> None:
        super().__init__()
        # 初始化层列表
        layers = list()
        # 遍历每一层，构建神经网络层
        for i in range(len(dims) - 1):
            # 根据是否为最后一层决定是否使用激活函数
            activation = None if i == len(dims) - 2 and not last_act else act
            # 添加带有激活函数和dropout的线性层
            layers.append(
                LinearAct(
                    dims[i],
                    dims[i + 1],
                    bias=bias,
                    act=activation,
                    dropout=dropout,
                )
            )
        # 将层列表封装为ModuleList，方便管理和调用
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
        - x: 输入的张量。

        返回:
        - 输出的张量。
        """
        # 遍历每一层，进行前向传播
        for layer in self.layers:
            x = layer(x)
        # 返回最终的输出
        return x


class ESMM(nn.Module):
    """
    ESMM（Embedded Sample Matching Model）类，用于实现嵌入样本匹配模型。

    参数:
    - cvr_dims: 模型转化率（CVR）神经网络的层维度列表。
    - ctr_dims: 模型点击率（CTR）神经网络的层维度列表。
    - bias: 是否在神经网络中使用偏置项，默认为True。
    - act: 神经网络层的激活函数，默认为'relu'。
    - dropout: 神经网络中的dropout比例，默认为0.0（不使用dropout）。
    """

    def __init__(
        self,
        cvr_dims: list[int],
        ctr_dims: list[int],
        bias: bool = True,
        act: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        """
        初始化ESMM模型，包括CVR和CTR的深度神经网络（DNN）模型。
        """
        super().__init__()
        # 初始化转化率（CVR）的深度神经网络
        self.cvr_dnn = DNN(cvr_dims, bias=bias, act=act, dropout=dropout)
        # 初始化点击率（CTR）的深度神经网络
        self.ctr_dnn = DNN(ctr_dims, bias=bias, act=act, dropout=dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        定义前向传播过程。

        参数:
        - x: 输入的张量。

        返回:
        - ctr_logits: 点击率（CTR）的logits张量。
        - ctcvr_logits: 经过CTR和CVR概率相乘后的logits张量。
        """
        # 计算转化率（CVR）的logits
        cvr_logits = self.cvr_dnn(x)
        # 计算点击率（CTR）的logits
        ctr_logits = self.ctr_dnn(x)

        # 将logits转换为概率
        cvr_logits = torch.sigmoid(cvr_logits)
        ctr_logits = torch.sigmoid(ctr_logits)

        # 计算CTR和CVR概率相乘的结果，用于预测最终的嵌入样本匹配概率
        ctcvr_logits = ctr_logits * cvr_logits

        return ctr_logits, ctcvr_logits


if __name__ == "__main__":
    model = ESMM([10, 20, 1], [10, 20, 1])
    x = torch.randn(10, 10)
    ctr_logits, ctcvr_logits = model(x)
    print(ctr_logits.shape, ctcvr_logits.shape)
