import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class AutoInt(nn.Module):
    """
    AutoInt类继承自nn.Module，用于实现自动整数化模型。

    参数:
    - nfeat: 特征数量。
    - dim_in: 输入维度。
    - dim_qk: 查询和键的维度。
    - dim_v: 值的维度。
    - nhead: 多头注意力的头数。
    - out_dim: 输出维度，默认为1。
    - bias: 是否使用偏置，默认为True。
    """

    def __init__(
        self,
        nfeat: int,
        dim_in: int,
        dim_qk: int,
        dim_v: int,
        nhead: int,
        out_dim: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        # 初始化查询、键、值的线性变换
        self.linear_q = nn.Linear(dim_in, dim_qk, bias=bias)
        self.linear_k = nn.Linear(dim_in, dim_qk, bias=bias)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=bias)
        # 初始化值到输入维度的线性变换
        self.fc1 = nn.Linear(dim_v, dim_in, bias=bias)
        # 初始化最终输出的线性变换
        self.fc2 = nn.Linear(dim_in * nfeat, out_dim, bias=bias)
        # 初始化激活函数为ReLU
        self.act = nn.ReLU(inplace=True)
        # 设置多头注意力的头数
        self.nhead = nhead

    def _self_attention(self, x: Tensor) -> Tensor:
        """
        实现自注意力机制。

        参数:
        - x: 输入的张量。

        返回:
        - 经过自注意力机制处理后的张量。
        """
        # 计算查询、键、值
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        # 对查询、键、值进行多头注意力的排列
        q = rearrange(
            q,
            "b n (nhead d) -> (b nhead) n d",
            nhead=self.nhead,
            d=q.size(2) // self.nhead,
        )
        k = rearrange(
            k,
            "b n (nhead d) -> (b nhead) n d",
            nhead=self.nhead,
            d=k.size(2) // self.nhead,
        )
        v = rearrange(
            v,
            "b n (nhead d) -> (b nhead) n d",
            nhead=self.nhead,
            d=v.size(2) // self.nhead,
        )
        # 计算注意力权重并应用softmax
        att = torch.einsum("bid,bjd->bij", q, k) / (k.size(2) ** 0.5)
        att = torch.softmax(att, dim=-1)
        # 根据注意力权重计算加权和
        z = torch.einsum("bij,bjk->bik", att, v)
        # 将多头注意力的结果重新排列
        z = rearrange(
            z,
            "(b nhead) n d -> b n (nhead d)",
            nhead=self.nhead,
        )
        return z

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
        - x: 输入的张量。

        返回:
        - 模型的输出张量。
        """
        # 应用自注意力机制
        x_att = self._self_attention(x)
        # 经过ReLU激活函数和线性变换后的加权和
        f = self.act(self.fc1(x_att) + x)
        # 将特征展开后通过最终的线性变换得到输出
        y = self.fc2(f.flatten(start_dim=1))
        y = torch.sigmoid(y)
        return y


if __name__ == "__main__":
    batch = 4
    nfeat = 27
    emb_dim = 8
    dim_k = 8
    dim_v = 16
    nhead = 4
    x = torch.randn(batch, nfeat, emb_dim)
    model = AutoInt(nfeat, emb_dim, dim_k, dim_v, nhead, out_dim=1)
    y = model(x)
    print(y.shape)
