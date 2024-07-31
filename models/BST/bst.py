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


if __name__ == "__main__":
    x = torch.randn(10, 100, 128)
    model = Transformer(128, 8, 512)
    print(model(x).shape)
