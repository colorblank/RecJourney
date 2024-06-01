import torch
import torch.nn as nn
from typing import List

from dataclasses import dataclass


@dataclass
class SEIlArgs:
    dim_in: int
    dim_out: int
    dim_hidden: List[int]
    expert_num: int


@dataclass
class CGCArgs:
    dim_in: int
    dim_out: int
    dim_hidden: List[int]
    shared_expert_num: int
    unique_expert_num: int
    task_num: int
    bias: bool = True


@dataclass
class ModelArgs:
    sei_args: SEIlArgs
    cgc_args: CGCArgs

    scene_indicator_dim: int
    shared_expert_num: int
    bias: bool = True


class LinearAct(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, bias: bool = True, act: str = "relu"
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias)
        if act.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act.lower() == "sigmoid":
            self.act = nn.Sigmoid()
        elif act.lower() == "prelu":
            self.act = nn.PReLU()
        else:
            raise NotImplementedError(f"{act} is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


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
        bias: bool = True,
    ) -> None:
        super().__init__()
        # 如果没有指定隐藏层维度，初始化一个空列表
        dim_hidden = [] if not dim_hidden else dim_hidden
        # 构建层维度列表
        dims = [dim_in, *dim_hidden, dim_out]
        # 使用nn.Sequential创建模型层序列，包含多个Linear层与ReLU激活函数及dropout
        self.expert = nn.Sequential(
            *[LinearAct(dims[i - 1], dims[i], bias=bias) for i in range(1, len(dims))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        参数:
        - x: torch.Tensor, 输入的张量

        返回:
        - torch.Tensor, 经过模型处理后的输出张量
        """
        return self.expert(x)


class SEI(nn.Module):
    """子专家集成模块

    参数:
        nn -- 模块描述
    """

    def __init__(
        self, dim_in: int, dim_out: int, dim_hidden: List[int], expert_num: int
    ) -> None:
        super().__init__()
        # 初始化专家网络，存储在一个模块字典中
        self.experts = nn.ModuleDict(
            {
                f"expert_{i}": Expert(dim_in, dim_out, dim_hidden)
                for i in range(expert_num)
            }
        )
        # 初始化门控网络，用于权重分配
        self.gate = nn.Linear(dim_in, expert_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程

        参数:
            x -- 输入特征

        返回:
            经过子专家集成后的输出特征
        """
        # 获取每个专家的输出
        expert_outputs = [
            self.experts[f"expert_{i}"](x).unsqueeze(-1) for i in range(self.expert_num)
        ]
        # 将所有专家的输出在最后一个维度上连接
        feat = torch.cat(expert_outputs, dim=-1)  # [batch, dim_out, expert_num]
        # 通过门控网络获取每个专家的权重
        gate_output = self.gate(x).unsqueeze(1)  # [batch, 1, expert_num]
        weight = torch.softmax(gate_output, dim=-1)
        # 根据权重对专家输出进行加权求和
        return torch.sum(feat * weight, dim=-1)  # [batch, dim_out]


class SharedExpertNet(nn.Module):
    """
    共享专家网络类，用于实现一个具有多个共享专家的网络模型。

    参数:
    - feat_dim_in: 输入特征维度。
    - feat_dim_out: 输出特征维度。
    - feat_dim_hidden: 隐藏层特征维度列表。
    - shared_expert_num: 共享专家的数量。
    - bias: 是否在专家网络中使用偏置，默认为True。

    返回:
    - 无
    """

    def __init__(
        self,
        feat_dim_in: int,
        feat_dim_out: int,
        feat_dim_hidden: List[int],
        shared_expert_num: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.shared_expert_num = shared_expert_num
        # 初始化共享专家模块字典
        self.shared_expert = nn.ModuleDict(
            {
                f"expert_{i}": Expert(feat_dim_in, feat_dim_out, feat_dim_hidden, bias)
                for i in range(shared_expert_num)
            }
        )

    def forward(self, f_main: torch.Tensor):
        """
        前向传播函数。

        参数:
        - f_main: 主要特征张量。

        返回:
        - 经过所有共享专家处理后的特征张量。
        """
        # 通过每个共享专家处理输入特征，并将结果集合起来
        feats = [
            self.shared_expert[f"expert_{i}"](f_main).unsqeeze(-1)
            for i in range(self.shared_expert_num)
        ]
        # 在最后一个维度上连接所有专家的输出
        feats = torch.cat(feats, dim=-1)  # (batch, feat_dim_out, shared_expert_num)
        return feats


class SAN(nn.Module):
    def __init__(
        self,
        shared_expert_num: int,
        scene_dim_in: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.gate = nn.Linear(scene_dim_in, shared_expert_num, bias)

    def forward(
        self, feats: torch.Tensor, f_scene: torch.Tensor, scene_indicator: torch.Tensor
    ) -> torch.Tensor:
        """_summary_

        Arguments:
            feats -- main features, size (batch, feat_dim_out, shared_expert_num)
            f_scene -- scene embedding, size (batch_size, scene_dim_in)
            scene_indicator -- a one hot indicator, size=(batch_size, num_scenes)

        Returns:
            _description_
        """

        gate = self.gate(f_scene).unsquezee(1)  # (batch, 1, shared_expert_num)
        weights = torch.softmax(gate, dim=-1)

        mask = scene_indicator.unsqueeze(1).long().detach()

        return torch.sum(feats * weights * mask, dim=-1)


class CustomizedGateControl(nn.Module):
    """
    一个定制的门控模块，用于管理共享和独特的专家网络。
    paper: Progressive Layered Extraction (PLE): \
            A Novel Multi-Task Learning (MTL) Model \
            for Personalized Recommendations

    参数:
    - dim_in: int, 输入维度
    - dim_out: int, 输出维度
    - dim_hidden: List[int], 隐藏层维度列表
    - shared_expert_num: int, 共享专家的数量
    - unique_expert_num: int, 每个任务独特专家的数量
    - task_num: int, 任务数量
    - dropout: float = 0.1, Dropout比例，默认为0.1
    - bias: bool = True, 是否包含偏置，默认为True

    返回:
    - None
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: List[int],
        shared_expert_num: int,
        unique_expert_num: int,
        task_num: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.task_num = task_num
        self.shared_expert_num = shared_expert_num
        self.unique_expert_num = unique_expert_num

        # 初始化每个任务的门控模块
        self.gate = nn.ModuleDict(
            {
                f"gate_{j}": Gate(dim_in, shared_expert_num + unique_expert_num, bias)
                for j in range(task_num)
            }
        )

        # 初始化共享专家网络
        self.shared_experts = nn.ModuleDict(
            {
                f"expert_{i}": Expert(dim_in, dim_out, dim_hidden)
                for i in range(shared_expert_num)
            }
        )

        # 初始化每个任务的独特专家网络
        self.unique_experts = nn.ModuleDict(
            {
                f"expert_{i}_task_{j}": Expert(dim_in, dim_out, dim_hidden)
                for i in range(unique_expert_num)
                for j in range(task_num)
            }
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播函数。

        参数:
        - x: torch.Tensor, 输入张量

        返回:
        - List[torch.Tensor], 包含每个任务特征的列表
        """

        # 计算共享专家的输出
        shared_expert_outputs = [
            self.shared_experts[f"expert_{i}"](x).unsqueeze(-1)
            for i in range(self.shared_expert_num)
        ]
        shared_expert_outputs = torch.cat(
            shared_expert_outputs, dim=-1
        )  # [batch_size, dim_out, shared_expert_num]

        feats = list()
        for j in range(self.task_num):
            # 计算当前任务独特专家的输出
            task_expert_outputs = [
                self.unique_experts[f"expert_{i}_task_{j}"](x).unsqueeze(-1)
                for i in range(self.unique_expert_num)
            ]  # [B, dim_out, 1] x unique_expert_num
            task_expert_outputs = torch.cat(task_expert_outputs, dim=-1)

            # 合并共享和独特专家的输出
            feat_of_task_i = torch.cat(
                [
                    shared_expert_outputs,
                    task_expert_outputs,
                ],
                dim=-1,
            )

            # 应用门控权重
            gate_weight = self.gate[f"gate_{j}"](x)
            f = torch.einsum("bde,be->bd", feat_of_task_i, gate_weight)
            feats.append(f)
        return feats


class HiNet(nn.Module):
    """_summary_
    paper: HiNet: Novel Multi-Scenario & Multi-Task Learning \
          with Hierarchical Information Extraction
    url: http://arxiv.org/abs/2303.06095
    
    Arguments:
        nn -- _description_
    """    
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        # 初始化特定专家网络(SEI)
        self.specific_expert = SEI(
            dim_in=args.sei_args.dim_in,
            dim_out=args.sei_args.dim_out,
            dim_hidden=args.sei_args.dim_hidden,
            expert_num=args.sei_args.expert_num,
        )
        # 初始化共享专家网络(SharedExpertNet)
        self.shared_experts = SharedExpertNet(
            feat_dim_in=args.sei_args.dim_in,
            feat_dim_out=args.sei_args.dim_in,
            feat_dim_hidden=args.sei_args.dim_hidden,
            shared_expert_num=args.shared_expert_num,
            bias=args.bias,
        )
        # 初始化场景注意力网络(SAN)
        self.san = SAN(
            args.shared_expert_num,
            scene_dim_in=args.scene_indicator_dim,
            bias=args.bias,
        )
        # 断言，确保输入维度与SEI的输出维度匹配CGC的输入维度要求
        assert args.cgc_args.dim_in == args.sei_args.dim_out * 3
        # 初始化自定义门控控制(CGC)
        self.cgc = CustomizedGateControl(
            dim_in=args.cgc_args.dim_in,
            dim_hidden=args.cgc_args.dim_hidden,
            dim_out=args.cgc_args.dim_out,
            shared_expert_num=args.cgc_args.shared_expert_num,
            unique_expert_num=args.cgc_args.unique_expert_num,
            task_num=args.cgc_args.task_num,
            bias=args.cgc_args.bias,
        )

    def forward(
        self, x: torch.Tensor, scene_emb: torch.Tensor, scene_indicator: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        前向传播函数，结合特定专家的输出、共享专家的输出、场景嵌入和场景指示器来生成最终结果。

        参数:
        - x: 输入数据的张量。 size = (batch_size, dim_in)
        - scene_emb: 场景嵌入的张量。 size = (batch_size, dim_scene)
        - scene_indicator: 场景指示器的张量，用于指示输入数据属于哪个场景。
            one-hot vector, size = (batch_size, scene_num)

        返回值:
        - ys: 通过自定义门控控制合并后的专家输出列表。
        """
        x_unique = self.specific_expert(x)
        x_share = self.shared_experts(x)
        s_i = x_share[scene_indicator.long()]
        a_i = self.san(x_share, scene_emb, scene_indicator)
        c = torch.cat([x_unique, s_i, a_i], dim=1)
        ys = self.cgc(c)
        return ys


if __name__ == "__main__":
    # user profile, user behavior, item feature, current scenario feature
    pass
