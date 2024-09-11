import torch
import torch.nn as nn

from torch import Tensor


class Transformer(nn.Module):
    """根据BST定制的Transformer

    paper: Behavior Sequence Transformer for E-commerce Recommendation in Alibaba
    URL: https://arxiv.org/pdf/1905.06874

    Parameters:
    - d_model: int, 输入特征维度
    - nhead: int, 多头注意力的头数
    - dim_feedforward: int, 前馈网络隐藏层维度
    - dropout: float, 丢弃概率

    Input:
    - x: Tensor, 输入特征，形状为(batch_size, seq_len, d_model)

    Output:
    - z: Tensor, 输出特征，形状为(batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.2,
    ):
        super(Transformer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * 3, bias=False)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.linear(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        s = self.attn(q, k, v)[0]
        s_ = self.norm1(x + s)
        z = self.ffn(s_)
        z = self.norm2(s_ + z)
        return z


class BST(nn.Module):
    """
    paper: Behavior Sequence Transformer for E-commerce Recommendation in Alibaba
    URL: https://arxiv.org/pdf/1905.06874

    Parameters:
    - seq_len: int, 序列长度
    - other_feat_dim: int, 其它特征维度
    - transformer_dim: int, transformer隐藏层维度
    - nhead: int, 多头注意力的头数
    - dim_forward: int, 前馈网络隐藏层维度
    - predict_dims: list[int], 预测层隐藏层维度
    - num_classes: int, 预测类别数
    - bias: bool, 是否使用偏置
    - dropout: float, 丢弃概率

    Input:
    - x_seq: Tensor, 序列特征，形状为(batch_size, seq_len, d_model)
    - x_pos_emb: Tensor, 位置嵌入，形状为(batch_size, d_model)
    - x_other: Tensor, 其它特征，形状为(batch_size, other_feat_dim)

    Output:
    - y: Tensor, 预测结果，形状为(batch_size, num_classes)
    """

    def __init__(
        self,
        seq_len: int,
        other_feat_dim: int,
        transformer_dim: int,
        nhead: int,
        dim_forward: int,
        predict_dims: list[int] = [
            1024,
            512,
            256,
        ],
        num_classes: int = 1,
        bias: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.transformer = Transformer(transformer_dim, nhead, dim_forward, dropout)
        dims = (
            [seq_len * transformer_dim + other_feat_dim] + predict_dims + [num_classes]
        )
        fcs = list()
        for i in range(len(dims) - 1):
            fc = nn.Linear(dims[i], dims[i + 1], bias=bias)
            fcs.append(fc)
            if i != len(dims) - 2:
                fcs.append(nn.LeakyReLU())
        self.fcs = nn.Sequential(*fcs)

    def forward(self, x_seq: Tensor, x_pos_emb: Tensor, x_other: Tensor) -> Tensor:
        x_seq = x_seq + x_pos_emb.unsqueeze(1)
        x = self.transformer(x_seq)
        batch = x.size(0)
        x = x.view(batch, -1)
        x = torch.cat([x, x_other], dim=-1)
        y = self.fcs(x)
        return y


if __name__ == "__main__":
    x = torch.randn(10, 10, 128)
    model = Transformer(128, 8, 512)
    print(model(x).shape)

    model = BST(
        seq_len=10,
        other_feat_dim=128,
        transformer_dim=128,
        nhead=8,
        dim_forward=512,
        predict_dims=[1024, 512, 256],
        num_classes=1,
        bias=True,
        dropout=0.2,
    )
    x = torch.randn(10, 10, 128)
    x_pos_emb = torch.randn(10, 128)
    x_other = torch.randn(10, 128)
    print(model(x, x_pos_emb, x_other).shape)
