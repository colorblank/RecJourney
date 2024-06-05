import torch
import torch.nn as nn
from einops import rearrange


class AutoInt(nn.Module):
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
        self.linear_q = nn.Linear(dim_in, dim_qk, bias=bias)
        self.linear_k = nn.Linear(dim_in, dim_qk, bias=bias)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=bias)
        self.fc1 = nn.Linear(dim_v, dim_in, bias=bias)
        self.fc2 = nn.Linear(dim_in * nfeat, out_dim, bias=bias)
        self.act = nn.ReLU(inplace=True)

        self.nhead = nhead

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # to nhead
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
        # multi-head attention matrix:
        att = torch.einsum("bid,bjd->bij", q, k) / (k.size(2) ** 0.5)
        att = torch.softmax(att, dim=-1)
        z = torch.einsum("bij,bjk->bik", att, v)
        z = rearrange(
            z,
            "(b nhead) n d -> b n (nhead d)",
            nhead=self.nhead,
        )
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Arguments:
            x -- size (batch_size, n_features, dim_in)
        """

        x_att = self._self_attention(x)
        f = self.act(self.fc1(x_att) + x)
        y = self.fc2(f.flatten(start_dim=1))
        return y


if __name__ == "__main__":
    batch = 4
    nfeat = 10
    emb_dim = 8
    dim_k = 8
    dim_v = 16
    nhead = 4
    x = torch.randn(batch, nfeat, emb_dim)
    model = AutoInt(nfeat, emb_dim, dim_k, dim_v, nhead, out_dim=1)
    y = model(x)
    print(y.shape)
