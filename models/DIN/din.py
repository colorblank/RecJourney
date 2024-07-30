from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class FeatArgs:
    profile_dim: int
    seq_len: int
    candidate_dim: int
    context_dim: int


@dataclass
class AUArgs:
    dim_hidden: int
    dim_out: int = 1


@dataclass
class HeadArgs:
    dim_hidden: int
    dim_out: int


class ActivationUnit(nn.Module):
    """
    ActivationUnit类继承自nn.Module，用于定义一个包含激活函数的神经网络单元。

    参数:
    - dim_in (int): 输入层的维度。
    - dim_hidden (int): 隐藏层的维度。
    - dim_out (int, 可选): 输出层的维度，默认为1。
    - activation (str, 可选): 激活函数的类型，默认为"prelu"，可选值为"prelu"或"relu"。

    Input:
    - x_hist (Tensor): 历史序列的输入特征，形状为[batch_size, seq_len, dim_in]。
    - x_cand (Tensor): 候选项的输入特征，形状为[batch_size, dim_in]。

    Output:
    - x (Tensor): 输出特征，形状为[batch_size, seq_len, 1]。
    """

    def __init__(
        self, dim_in: int, dim_hidden: int, dim_out: int = 1, activation: str = "prelu"
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.PReLU() if activation == "prelu" else nn.ReLU(),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x_hist: Tensor, x_cand: Tensor) -> Tensor:
        """_summary_

        Arguments:
            x_hist -- a list item features of history items.
                each item feature is a tensor.
                shape: [batch_size, seq_len, dim]
            x_cand -- candicate item feature
                shape: [batch_size, dim]

        Returns:
            Tensor, size: [batch_size, seq_len, 1]
        """
        seq_len = x_hist.shape[1]
        x_cand = x_cand.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x_hist, x_cand, x_hist - x_cand, x_cand * x_hist], dim=-1)
        return self.fc(x)


class DIN(nn.Module):
    """
    DIN类继承自nn.Module，用于定义一个深度信息融合网络（Deep Interest Network）模型。

    参数:
    - feat_args (FeatArgs): 特征参数。
    - au_args (AUArgs): ActivationUnit参数。
    - head_args (HeadArgs): 输出层的参数。
    - activation (str, 可选): 激活函数的类型，默认为"prelu"，可选值为"prelu"或"relu"。
    - dropout (float, 可选): 丢弃概率，默认为0.1。
    - bias (bool, 可选): 是否使用偏置项，默认为True。

    Input:
    - x_profile (Tensor): 用户的输入特征，形状为[batch_size, dim]。
    - x_hist (Tensor): 历史序列的输入特征，形状为[batch_size, seq_len, dim]。
    - x_cand (Tensor): 候选项的输入特征，形状为[batch_size, dim]。
    - x_context (Tensor): 上下文的输入特征，形状为[batch_size,dim]。

    Output:
    - y (Tensor): 输出特征，形状为[batch_size, 1]。
    """

    def __init__(
        self,
        feat_args: FeatArgs,
        au_args: AUArgs,
        head_args: HeadArgs,
        activation: str = "prelu",
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.au = ActivationUnit(
            dim_in=feat_args.candidate_dim * 4,
            dim_hidden=au_args.dim_hidden,
            dim_out=au_args.dim_out,
            activation=activation,
        )
        dim_in = feat_args.profile_dim + feat_args.candidate_dim * 2
        self.pred = nn.Sequential(
            nn.Linear(dim_in, head_args.dim_hidden, bias=bias),
            nn.PReLU() if activation == "prelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_args.dim_hidden, head_args.dim_out, bias=bias),
        )

    def forward(
        self,
        x_profile: Tensor,
        x_hist: Tensor,
        x_cand: Tensor,
        x_context: Tensor,
    ) -> Tensor:
        """
        Arguments:
            x_profile -- a user profile feature.
                shape: [batch_size, dim]
            x_hist -- a list item features of history items.
                each item feature is a tensor.
                shape: [batch_size, seq_len, dim]
            x_cand -- candicate item feature
                shape: [batch_size, dim]
            x_context -- context feature
                shape: [batch_size, dim]
        """
        w = self.au(x_hist, x_cand)  # [batch_size, seq_len, 1]

        x = (x_hist * w).sum(dim=1)
        x = torch.cat([x_profile, x, x_context], dim=-1)
        y = self.pred(x)
        return y


if __name__ == "__main__":
    feat_args = FeatArgs(
        profile_dim=10,
        seq_len=5,
        candidate_dim=10,
        context_dim=10,
    )
    au_args = AUArgs(
        dim_hidden=10,
        dim_out=1,
    )
    head_args = HeadArgs(
        dim_hidden=10,
        dim_out=1,
    )
    din = DIN(feat_args, au_args, head_args)
    batch_size = 2
    x_profile = torch.randn(batch_size, feat_args.profile_dim)
    x_context = torch.randn(batch_size, feat_args.context_dim)
    x_hist = torch.randn(batch_size, feat_args.seq_len, feat_args.candidate_dim)
    x_candicate = torch.randn(batch_size, feat_args.candidate_dim)

    y = din(x_profile, x_hist, x_candicate, x_context)

    print(y.shape)
