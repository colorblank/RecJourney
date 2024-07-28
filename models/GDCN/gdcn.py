import torch
import torch.nn as nn
from torch import Tensor


class GatedCrossNetwork(nn.Module):
    """
    Gated Cross Network类，用于实现一种具有门控机制的交叉网络结构。

    参数:
    - dim_in: 输入维度。
    - layer_num: 层数，默认为3。

    input:
    - Tensor (batch_size, dim_in)

    Returns:
    - Tensor (batch_size, dim_in)
    """

    def __init__(self, dim_in: int, layer_num: int = 3) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.W = nn.ModuleList(
            [nn.Linear(dim_in, dim_in, bias=False) for _ in range(layer_num)]
        )
        self.Wg = nn.ModuleList(
            [nn.Linear(dim_in, dim_in, bias=False) for _ in range(layer_num)]
        )
        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim_in)) for _ in range(layer_num)]
        )
        for i in range(layer_num):
            nn.init.uniform_(self.b[i].data)

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Arguments:
            x -- Tensor (batch_size, dim_in)

        Returns:
            Tensor (batch_size, dim_in)
        """
        xi = x
        for i in range(self.layer_num):
            weight = torch.sigmoid(self.Wg[i](xi))
            f = self.W[i](xi) + self.b[i]
            xi = x * f * weight + xi
        return xi


if __name__ == "__main__":
    batch = 2
    dim_in = 40
    x = torch.randn(batch, dim_in)
    model = GatedCrossNetwork(dim_in, layer_num=3)
    print(model(x).shape)
