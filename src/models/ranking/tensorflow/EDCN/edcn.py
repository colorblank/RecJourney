from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


class BridgeLayer(nn.Module):
    """BridgeLayer

    Parameters:
    - dim_in: 输入特征维度。
    - mode: 桥接层的模式，可选值为"add", "product", "concat", "attention"，默认为"product"。

    Input:
    - x_deep: 输入特征，形状为[batch_size, dim_in]。
    - x_cross: 输入特征，形状为[batch_size, dim_in]。

    Output:
    - y: 输出特征，形状为[batch_size, dim_in]。
    """

    def __init__(
        self,
        dim_in: int,
        mode: Literal["add", "product", "concat", "attention"] = "product",
    ):
        super(BridgeLayer, self).__init__()
        self.mode = mode
        if mode == "attention":
            self.fc_deep = nn.Sequential(
                nn.Linear(dim_in, dim_in, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_in, bias=False),
            )
            self.fc_cross = nn.Sequential(
                nn.Linear(dim_in, dim_in, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_in, bias=False),
            )
        elif mode == "concat":
            self.fc = nn.Sequential(
                nn.Linear(dim_in * 2, dim_in, bias=True),
                nn.ReLU(inplace=True),
            )

    def forward(self, x_deep: Tensor, x_cross: Tensor) -> Tensor:
        if self.mode == "add":
            return x_deep + x_cross
        elif self.mode == "product":
            return x_deep * x_cross
        elif self.mode == "concat":
            return self.fc(torch.cat([x_deep, x_cross], dim=-1))
        elif self.mode == "attention":
            deep_weight = torch.softmax(self.fc_deep(x_deep), dim=-1)
            cross_weight = torch.softmax(self.fc_cross(x_cross), dim=-1)
            return deep_weight * x_deep + cross_weight * x_cross


class RegularizationLayer(nn.Module):
    """RegularizationLayer

    Parameters:
    - field_num: 字段数量。
    - tau: 温度参数，默认为0.1。

    Input:
    - x: 输入特征，形状为[batch_size, field_num, dim_hidden]。
    Output:
    - y: 输出特征，形状为[batch_size, dim_hidden * field_num]。
    """

    def __init__(self, field_num: int, tau: float) -> None:
        super().__init__()
        self.field_num = field_num
        self.tau = tau
        self.w = nn.Parameter(torch.randn(field_num, 1))

    def forward(self, x: Tensor) -> Tensor:
        if len(x.size()) == 2:
            x = x.view(x.size(0), self.field_num, -1)
        g = torch.softmax(self.w.unsqueeze(0) / self.tau, dim=1)  # [1, field_num, 1]
        y = g * x  # [batch_size, field_num, dim_hidden]
        return y.view(y.size(0), -1)  # [batch_size, dim_hidden * field_num]


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


class EDCN(nn.Module):
    """Enhanced Deep & Cross Network

    Parameters:
    - field_num: 字段数量。
    - emb_dim: 嵌入维度。
    - total_emb_dim: 总嵌入维度。
    - layer_num: 层数。
    - num_classes: 类别数量，默认为1。
    - bias: 是否使用偏置，默认为True。
    - bridge_mode: 桥接层的模式，可选值为"add", "product", "concat", "attention"，默认为"product"。

    Input:
    - x: 输入的深度特征，形状为[batch_size, field_num, emb_dim]。

    Output:
    - y: 预测结果，形状为[batch_size, num_classes]。
    """

    def __init__(
        self,
        field_num: int,
        emb_dim: int,
        total_emb_dim: int,
        layer_num: int,
        num_classes: int = 1,
        tau: float = 0.1,
        bias: bool = True,
        bridge_mode: Literal["add", "product", "concat", "attention"] = "product",
    ):
        super(EDCN, self).__init__()
        self.field_num = field_num
        self.embedding_dim = emb_dim
        self.layer_num = layer_num
        assert layer_num > 1, "layer_num must be greater than 1"
        self.cross_reg = RegularizationLayer(field_num, tau)
        self.deep_reg = RegularizationLayer(field_num, tau)

        self.deep_layers = nn.ModuleList(
            [LinearReLU(total_emb_dim, total_emb_dim, bias) for _ in range(layer_num)]
        )
        self.cross_layers = nn.ModuleList(
            [LinearReLU(total_emb_dim, total_emb_dim, bias) for _ in range(layer_num)]
        )
        self.bridge_layers = nn.ModuleList(
            [BridgeLayer(total_emb_dim, bridge_mode) for _ in range(layer_num)]
        )
        self.reg_layer_cross = nn.ModuleList(
            [RegularizationLayer(field_num, tau) for _ in range(layer_num)]
        )
        self.reg_layer_deep = nn.ModuleList(
            [RegularizationLayer(field_num, tau) for _ in range(layer_num)]
        )

        self.predict_head = nn.Linear(3 * total_emb_dim, num_classes, bias)

    def forward(self, x: Tensor) -> Tensor:
        x_deep = self.reg_layer_deep[0](x)
        x_cross = self.reg_layer_cross[0](x)
        x_cross_i = x_cross
        for i in range(self.layer_num):
            x_deep = self.deep_layers[i](x_deep)
            x_cross_i = x_cross * self.cross_layers[i](x_cross_i) + x_cross_i
            tmp = self.bridge_layers[i](x_deep, x_cross_i)
            if i != self.layer_num - 1:
                x_deep = self.reg_layer_deep[i + 1](tmp)
                x_cross_i = self.reg_layer_cross[i + 1](tmp)

        f = torch.cat([x_deep, x_cross_i, tmp], dim=-1)
        y = self.predict_head(f)
        y = torch.sigmoid(y)
        return y


if __name__ == "__main__":
    x_deep = torch.randn(2, 25, 8)
    model = EDCN(field_num=25, emb_dim=8, total_emb_dim=200, layer_num=3, num_classes=1)
    y = model(x_deep)
    print(y)
