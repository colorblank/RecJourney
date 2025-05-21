import torch
import torch.nn as nn
from typing import List


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
        self, dim_in: int, dim_out: int, bias: bool = True, p: float = 0.1
    ) -> None:
        super().__init__()
        # 初始化线性层、ReLU激活函数和dropout层
        self.linear = nn.Linear(dim_in, dim_out, bias)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程。

        参数:
            x (torch.Tensor): 输入的张量。

        返回:
            torch.Tensor: 经过线性层、ReLU激活函数和dropout后的输出张量。
        """
        # 应用线性层，然后是ReLU激活函数，最后是dropout
        return self.dropout(self.relu(self.linear(x)))


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        参数:
        - x: torch.Tensor, 输入的张量

        返回:
        - torch.Tensor, 经过模型处理后的输出张量
        """
        return self.expert(x)


class Gate(nn.Module):
    """
    创建一个门控类，用于在多专家模型中选择专家。

    参数:
    - dim_in: int, 输入维度
    - expert_num: int, 专家的数量
    - bias: bool = True, 是否在门控线性变换中使用偏置，默认为True
    - dropout: float = 0.1, Dropout比例，默认为0.1，用于防止过拟合


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


class CustomizedGateControl(nn.Module):
    """
    一个定制的门控模块，用于管理共享和独特的专家网络。
    paper: Progressive Layered Extraction (PLE): \
            A Novel Multi-Task Learning (MTL) Model \
            for Personalized Recommendations
    url: https://dl.acm.org/doi/abs/10.1145/3383313.3412236

    参数:
    - dim_in: int, 输入维度
    - dim_out: int, 输出维度
    - dim_hidden: List[int], 隐藏层维度列表
    - shared_expert_num: int, 共享专家的数量
    - unique_expert_num: int, 每个任务独特专家的数量
    - task_num: int, 任务数量
    - dropout: float = 0.1, Dropout比例，默认为0.1
    - bias: bool = True, 是否包含偏置，默认为True

    输入:
    - x: torch.Tensor, 输入张量

    输出:
    - List[torch.Tensor], 包含每个任务特征的列表
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: List[int],
        shared_expert_num: int,
        unique_expert_num: int,
        task_num: int,
        dropout: float = 0.1,
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
                f"expert_{i}": Expert(dim_in, dim_out, dim_hidden, dropout)
                for i in range(shared_expert_num)
            }
        )

        # 初始化每个任务的独特专家网络
        self.unique_experts = nn.ModuleDict(
            {
                f"expert_{i}_task_{j}": Expert(dim_in, dim_out, dim_hidden, dropout)
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


class PLE(nn.Module):
    """Progressive Layered Extraction

    paper: Progressive Layered Extraction (PLE): \
            A Novel Multi-Task Learning (MTL) Model \
            for Personalized Recommendations
    url: https://dl.acm.org/doi/abs/10.1145/3383313.3412236
    Arguments:
        nn -- _description_
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    batch_size = 2
    dim_in = 4
    dim_out = 64
    dim_hidden = [32, 16]
    expert_num = 4
    task_num = 2
    model = CustomizedGateControl(
        dim_in,
        dim_out,
        dim_hidden,
        expert_num,
        expert_num // 2,
        task_num,
        dropout=0.1,
        bias=True,
    )
    x = torch.randn(batch_size, dim_in)
    feats = model(x)
    for f in feats:
        print(f.size())  # [batch, dim_out]
