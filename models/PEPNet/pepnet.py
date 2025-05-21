from typing import List

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
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function")
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


class GateNeuralUnit(nn.Module):
    """
    门控神经单元类，继承自nn.Module，用于创建具有门控机制的神经网络单元。

    参数:
    - dim_in: int, 输入维度
    - dim_hidden: int, 隐藏层维度
    - dim_out: int, 输出维度
    - gamma: float, 门控系数，默认为2.0

    返回:
    - None
    """

    def __init__(
        self, dim_in: int, dim_hidden: int, dim_out: int, gamma: float = 2.0
    ) -> None:
        super().__init__()  # 初始化父类
        # 创建一个多层感知器（MLP），包含两个线性层和一个ReLU激活函数
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_out)
        )
        self.gamma = gamma  # 门控系数

    def forward(self, f_domain: Tensor, f_general: Tensor = None) -> Tensor:
        """
        执行前向传播操作，将特定领域的特征和通用特征结合起来，并通过门控机制产生最终输出。

        参数:
            f_domain (Tensor): 特定领域的特征向量。[batch_size, dim_in_1]
            f_general (Tensor): 通用特征向量。 [batch_size, dim_in_2]

        返回:
            Tensor: 经过门控机制处理后的综合特征向量。 [batch_size, dim_in_2]
        """
        if f_general is None:
            x = f_domain
        else:
            # 将特定领域的特征和通用特征在最后一个维度上连接
            x = torch.cat([f_domain, f_general.detach()], dim=-1)
        # 应用门控机制，并返回结果
        return self.gamma * torch.sigmoid(self.mlp(x))


class EPNet(nn.Module):
    def __init__(
        self,
        gate_dim_in: int,
        gate_dim_hidden: int,
        gate_dim_out,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.gate_nu = GateNeuralUnit(gate_dim_in, gate_dim_hidden, gate_dim_out, gamma)

    def forward(self, f_d: Tensor, f_g: Tensor) -> Tensor:
        f = torch.cat([f_d, f_g.detach()], dim=-1)
        return self.gate_nu(f) * f_g


class PPNet(nn.Module):
    """
    PPNet模型类，用于实现多任务学习中的门控神经网络。

    参数:
        gate_dim_in: 输入维度到门控单元。
        gate_dim_hidden: 隐藏层维度到门控单元。
        task_num: 任务数量。
        mlp_dims: 多层感知器的维度列表。
        gamma: 门控单元的缩放因子，默认为2.0。
        bias: 是否在线性层中使用偏置，默认为True。
        dropout: Dropout比例，默认为0.1。
    """

    def __init__(
        self,
        gate_dim_in: int,
        gate_dim_hidden: int,
        task_num: int,
        mlp_dims: List[int],
        gamma: float = 2.0,
        bias: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layer_num = len(mlp_dims) - 1  # 网络层数
        self.task_num = task_num  # 任务数量

        # 初始化门控单元
        self.gate_nus = nn.ModuleDict(
            {
                f"gate_nu_{layer}": GateNeuralUnit(
                    gate_dim_in, gate_dim_hidden, mlp_dims[layer + 1] * task_num, gamma
                )
                for layer in range(self.layer_num)
            }
        )

        # 初始化任务塔
        tower = dict()
        for t in range(task_num):
            for layer in range(self.layer_num):
                # 最后一层使用sigmoid激活函数，其他层使用relu激活函数
                if layer == self.layer_num - 1:
                    tower[f"task_{t}_layer_{layer}"] = LinearReLUDropOut(
                        mlp_dims[layer],
                        mlp_dims[layer + 1],
                        bias,
                        p=0,
                        activation="sigmoid",
                    )
                else:
                    tower[f"task_{t}_layer_{layer}"] = LinearReLUDropOut(
                        mlp_dims[layer],
                        mlp_dims[layer + 1],
                        bias,
                        p=dropout,
                        activation="relu",
                    )
        self.towers = nn.ModuleDict(tower)

    def forward(
        self,
        f_s: Tensor,
        o_ep: Tensor,
    ) -> List[Tensor]:
        """
        前向传播函数。

        参数:
            f_s: 侧边特征，shape: (batch_size, emb_dim)。
            o_ep: 来自门控单元的特征，shape: (batch_size, dim_user)。

        返回值:
            一个列表，包含每个任务的特征。
        """
        f = torch.cat([f_s, o_ep.detach()], dim=1)  # 合并输入特征
        feat_list = list()
        for i in range(self.layer_num):
            weight = self.gate_nus[f"gate_nu_{i}"](f)  # 通过门控单元处理
            weight = weight.view(f.shape[0], -1, self.task_num)  # 调整形状以适配任务塔
            feat_list.append(weight)

        task_feat = list()
        for t in range(self.task_num):
            x = o_ep
            for layer in range(self.layer_num):
                # 根据门控特征调整任务特征
                x = (
                    self.towers[f"task_{t}_layer_{layer}"](x)
                    * feat_list[layer][:, :, t]
                )
            task_feat.append(x)
        return task_feat


class PEPNet(nn.Module):
    """
    PEPNet模型类，继承自nn.Module，用于实现一种具有领域和侧边信息处理能力的网络结构。

    参数:
    - general_dim: 通用特征维度
    - domain_dim: 领域特征维度
    - side_dim: 侧边特征维度
    - dim_hidden: 隐藏层维度
    - mlp_dims: 多层感知器(MLP)的维度列表
    - task_num: 任务数量
    - gamma: PPNet中的缩放因子
    - dropout: Dropout比例
    - bias: 是否使用偏置

    返回:
    - 无
    """

    def __init__(
        self,
        general_dim: int,
        domain_dim: int,
        side_dim: int,
        dim_hidden: int,
        mlp_dims: List[int],
        task_num: int,
        gamma: float = 2.0,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        # 确保MLP的输入维度与通用维度一致
        assert mlp_dims[0] == general_dim
        # 初始化EPNet模块，用于处理通用和领域特征
        self.epnet = EPNet(
            gate_dim_in=domain_dim + general_dim,
            gate_dim_hidden=dim_hidden,
            gate_dim_out=general_dim,
        )
        # 初始化PPNet模块，用于处理侧边和经过EPNet处理的特征
        self.ppnet = PPNet(
            gate_dim_in=side_dim + general_dim,
            gate_dim_hidden=dim_hidden,
            task_num=task_num,
            mlp_dims=mlp_dims,
            gamma=gamma,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, f_domain: Tensor, f_side: Tensor, f_general: Tensor) -> Tensor:
        """
        前向传播函数，输入领域特征、侧边特征和通用特征，输出任务特征。

        参数:
        - f_domain: 领域特征张量
        - f_side: 侧边特征张量
        - f_general: 通用特征张量

        返回:
        - task_feats: 处理后得到的任务特征张量
        """
        # 通过EPNet处理通用特征和领域特征
        f_g = self.epnet(f_domain, f_general)
        # 通过PPNet处理侧边特征和经过EPNet处理的特征，输出任务特征
        task_feats = self.ppnet(f_side, f_g)
        return task_feats


if __name__ == "__main__":
    batch_size = 2
    general_feat_dim = 64
    domain_feat_dim = 4
    side_feat_dim = 3
    dim_hidden = 128
    mlp_dims = [64, 128, 64]
    model = PEPNet(
        general_dim=general_feat_dim,
        domain_dim=domain_feat_dim,
        side_dim=side_feat_dim,
        dim_hidden=dim_hidden,
        mlp_dims=mlp_dims,
        task_num=2,
        gamma=2,
        dropout=0.1,
        bias=True,
    )
    f_general = torch.randn(batch_size, general_feat_dim)
    f_domain = torch.randn(batch_size, domain_feat_dim)
    f_side = torch.randn(batch_size, side_feat_dim)
    f_out = model(f_domain, f_side, f_general)
    for f in f_out:
        print(f.shape)
