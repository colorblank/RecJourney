from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class LinearReLUDropOut(nn.Module):
    """
    一个结合了线性层、ReLU激活函数和dropout的模块，用于神经网络中。

    参数:
        dim_in (int): 输入维度。
        dim_out (int): 输出维度。
        bias (bool, 可选): 线性层是否使用偏置，默认为True。
        p (float, 可选): Dropout的概率，默认为0.1。
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = True,
        p: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        # 初始化线性层、ReLU激活函数和dropout层
        self.linear = nn.Linear(dim_in, dim_out, bias)
        if activation is None:
            self.act = nn.Identity()
        elif activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError
        if p == 0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(p)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播过程。

        参数:
            x (Tensor): 输入的张量。

        返回:
            Tensor: 经过线性层、ReLU激活函数和dropout后的输出张量。
        """
        # 应用线性层，然后是ReLU激活函数，最后是dropout
        return self.dropout(self.act(self.linear(x)))


class Expert(nn.Module):
    """
    专家模块，用于创建一个具有多个隐藏层的神经网络专家模型。

    参数:
    - dim_in: int, 输入维度
    - dim_out: int, 输出维度
    - dim_hidden: List[int], 隐藏层维度的列表，默认为None，表示不使用隐藏层
    - dropout: float, Dropout比例，默认为0.1

    返回:
    - None
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: List[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # 如果没有指定隐藏层维度，初始化一个空列表
        dim_hidden = [] if not dim_hidden else dim_hidden
        # 构建层维度列表
        dims = [dim_in, *dim_hidden, dim_out]
        # 使用nn.Sequential创建模型层序列，包含多个Linear层与ReLU激活函数及dropout
        self.expert = nn.Sequential(
            *[
                LinearReLUDropOut(dims[i - 1], dims[i], p=dropout)
                for i in range(1, len(dims))
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
        - x: Tensor, 输入的张量

        返回:
        - Tensor, 经过模型处理后的输出张量
        """
        return self.expert(x)


@dataclass
class HeadArgs:
    dims: List[int]
    activation: Optional[str] = "relu"
    bias: Optional[bool] = True


class PredictHead(nn.Module):
    def __init__(self, dims: List[int], activation: str = "relu", bias: bool = True):
        super().__init__()
        self.layer_num = len(dims) - 1
        layers = list()
        for i in range(self.layer_num):
            if i == (self.layer_num - 1):
                activation = None
            layer = LinearReLUDropOut(
                dim_in=dims[i],
                dim_out=dims[i + 1],
                bias=bias,
                p=0,
                activation=activation,
            )
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Gate(nn.Module):
    """
    创建一个门控类，用于在多专家模型中选择专家。

    参数:
    - dim_in: int, 输入维度
    - expert_num: int, 专家的数量
    - bias: bool = True, 是否在门控线性变换中使用偏置，默认为True
    - dropout: float = 0.1, Dropout比例，默认为0.1，用于防止过拟合

    返回:
    - None
    """

    def __init__(
        self, dim_in: int, expert_num: int, bias: bool = True, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(
            dim_in, expert_num, bias
        )  # 创建一个线性变换，用于生成专家的选择概率
        self.dropout = nn.Dropout(p=dropout)  # 应用dropout防止极化

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        前向传播逻辑。

        参数:
        - x: torch.tensor, 输入的张量

        返回:
        - torch.tensor, 经过softmax激活函数和dropout后的张量，表示专家的选择概率。
        """
        return self.dropout(
            torch.softmax(self.gate(x), dim=-1)
        )  # 应用softmax激活函数，并通过dropout防止过拟合


class SharedBottomModel(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: List[int],
        # prev is bottom, next is tower
        dims: List[int],
        task_num: int,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.bottom = Expert(dim_in, dim_out, dim_hidden, dropout)
        assert dims[0] == dim_out
        self.towers = nn.ModuleDict(
            {
                f"tower_{i}": PredictHead(dims, activation=activation, bias=bias)
                for i in range(task_num)
            }
        )
        self.task_num = task_num

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.bottom(x)
        res = []
        for i in range(self.task_num):
            y = self.towers[f"tower_{i}"](x)
            y = torch.sigmoid(y)
            res.append(y)
        return res


@dataclass
class MMOEArgs:
    dim_in: int
    dim_out: int
    dim_hidden: List[int]
    expert_num: int
    task_num: int
    dropout: Optional[float] = 0.1
    bias: Optional[bool] = True


class Multi_Gate_MOE(nn.Module):
    """
    实现一个多门限专家模型(Multi-Gate Mixture of Experts, MOE)的类。
    paper: Modeling Task Relationships in Multi-task Learning \
             with Multi-gate Mixture-of-Experts
    url: https://dl.acm.org/doi/pdf/10.1145/3219819.3220007

    参数:
    - dim_in: int, 输入维度
    - dim_out: int, 输出维度
    - dim_hidden: List[int], 隐藏层维度列表
    - expert_num: int, 专家网络的数量
    - task_num: int, 任务的数量
    - dropout: float = 0.1, Dropout比例，默认为0.1
    - bias: bool = True, 是否使用偏置，默认为True

    返回:
    - None
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: List[int],
        expert_num: int,
        task_num: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.task_num = task_num  # 任务数量
        self.expert_num = expert_num  # 专家数量

        # 初始化专家网络
        self.experts = nn.ModuleDict(
            {
                f"expert_{i}": Expert(dim_in, dim_out, dim_hidden, dropout)
                for i in range(expert_num)
            }
        )

        # 初始化门控网络
        self.gates = nn.ModuleDict(
            {f"gate_{i}": Gate(dim_in, expert_num, bias) for i in range(task_num)}
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        前向传播函数。

        参数:
        - x: Tensor, 输入张量

        返回:
        - out: List[Tensor], 包含每个任务的加权专家输出的列表
        """
        feats = list()  # 存储每个专家的输出
        for i in range(self.expert_num):
            # 计算每个专家的输出
            f = self.experts[f"expert_{i}"](x).unsqueeze(-1)  # [batch, dim_out, 1]
            feats.append(f)
        feats = torch.cat(feats, dim=-1)  # 将所有专家的输出在最后一个维度上连接起来

        out = list()  # 存储每个任务的输出
        for i in range(self.task_num):
            # 计算每个任务的门控权重
            w = self.gates[f"gate_{i}"](x)  # [batch, expert_num]
            # 应用门控权重
            f = torch.einsum("bde,be->bd", feats, w)
            out.append(f)

        return out


class MMOE(nn.Module):
    def __init__(self, mmoe_args: MMOEArgs, head_args: HeadArgs) -> None:
        super().__init__()
        self.mmoe = Multi_Gate_MOE(**mmoe_args.__dict__)
        self.task_num = mmoe_args.task_num
        self.heads = nn.ModuleDict(
            {
                f"head_{i}": PredictHead(**head_args.__dict__)
                for i in range(self.task_num)
            }
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        feats = self.mmoe(x)
        outs = list()
        for i in range(self.task_num):
            out = self.heads[f"head_{i}"](feats[i])
            out = torch.sigmoid(out)
            outs.append(out)
        return outs


if __name__ == "__main__":
    batch_size = 2
    dim_in = 10
    dim_out = 64
    dim_hidden = [32, 64]
    expert_num = 4
    task_num = 2
    x = torch.randn(batch_size, dim_in)
    # model = Multi_Gate_MOE(
    #     dim_in,
    #     dim_out,
    #     dim_hidden,
    #     expert_num,
    #     task_num,
    # )
    # ys = model(x)
    # for y in ys:
    #     print(y.size())  # [batch_size, dim_out]

    model = SharedBottomModel(
        dim_in, dim_out, dim_hidden, dims=[dim_out, dim_out, 1], task_num=1, dropout=0.1
    )
    print(model)
    ys = model(x)
    for y in ys:
        print(y.size())  # [batch_size, dim_out]
