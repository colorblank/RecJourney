import torch
import torch.nn as nn


class LinearReLU(nn.Module):
    """
    该类继承自nn.Module，实现了线性层后跟ReLU激活函数的结构。

    参数:
    - dim_in: 输入维度。
    - dim_hidden: 隐藏层维度。
    - bias: 是否使用偏置，默认为True。
    """

    def __init__(self, dim_in: int, dim_hidden: int, bias: bool = True) -> None:
        super(LinearReLU, self).__init__()
        self.linear = nn.Linear(dim_in, dim_hidden, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class CrossNetwork(nn.Module):
    """
    该类实现了一个交叉网络结构，包含多个线性ReLU层。

    参数:
    - dim_in: 输入维度。
    - hidden_layer_num: 隐藏层的数量。
    - bias: 是否在线性层中使用偏置，默认为True。
    """

    def __init__(self, dim_in: int, hidden_layer_num: int, bias: bool = True):
        super(CrossNetwork, self).__init__()
        # 使用ModuleDict来存储多个线性ReLU层
        self.linears = nn.ModuleDict(
            {
                f"fc_{i}": LinearReLU(dim_in, dim_in, bias=bias)
                for i in range(hidden_layer_num)
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): size = (batch_size, dim_in)

        Returns:
            torch.Tensor: _description_
        """
        x_i = x
        # 遍历所有线性ReLU层，进行前向传播
        for i in range(len(self.linears)):
            x_prev = x_i
            # 计算当前层的输出，同时引入前一层的输出进行交叉
            x_i = x * self.linears[f"fc_{i}"](x_i) + x_prev
        # 返回最终的输出
        return x_i


class DCN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class DCNV2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    # 初始化输入张量
    x = torch.randn(2, 10)
    # 创建一个交叉网络实例
    model = CrossNetwork(dim_in=10, hidden_layer_num=2)
    # 打印模型输出的尺寸
    print(model(x).size())
