import torch
import torch.nn as nn

from torch import Tensor


class MultiHeadAttention(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, nhead: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.nhead = nhead
        self.W_q = nn.Linear(dim_in, dim_out, bias=bias)
        self.W_k = nn.Linear(dim_in, dim_out, bias=bias)
        self.W_v = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x_non: Tensor, x_seq: Tensor, mask: Tensor = None) -> Tensor:
        """_summary_

        Arguments:
            x_non -- user non-sequential features, size (batch_size, dim_in)
            x_seq -- user sequential features, size (batch_size, seq_len, dim_in)

        Keyword Arguments:
            mask -- _description_ (default: {None})

        Returns:
            _description_
        """
        batch, seq_len, dim = x_seq.size()
        q = self.W_q(x_non).view(batch, self.nhead, dim // self.nhead)
        k = self.W_k(x_seq).view(batch, seq_len, self.nhead, dim // self.nhead)
        v = self.W_v(x_seq).view(batch, seq_len, self.nhead, dim // self.nhead)

        qk = torch.einsum("bnd,bsnd->bns", q, k) // (dim // self.nhead) ** 0.5
        if mask is not None:
            qk = qk.masked_fill(mask == 0, -1e9)
        qk = torch.softmax(qk, dim=-1)
        z = torch.einsum("bns,bsnd->bnd", qk, v).view(batch, dim)

        return z


class FCBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias)
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "prelu":
            self.act = nn.PReLU()
        elif activation == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Invalid activation function: {activation}")

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        if norm == "batch":
            self.norm = nn.BatchNorm1d(dim_out)
        elif norm == "layer":
            self.norm = nn.LayerNorm(dim_out)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Invalid normalization: {norm}")

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class POSOLinear(nn.Module):
    def __init__(self, dim_in: int, hidden_dims: list[int], bias: bool = True) -> None:
        super().__init__()
        self.linears = nn.ModuleList(
            [FCBlock(dim_in, hidden_dims[i], bias) for i in range(len(hidden_dims))]
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        """

        Arguments:
            x -- Tensor. size = (batch_size, dim_in)

        Returns:
            List[Tensor].
             - size = (batch_size, dim_i)
        """
        res = []
        for layer in self.linears:
            f = layer(x)
            res.append(f)
        return res


class BaseModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


if __name__ == "__main__":
    batch = 2
    seq = 5
    dim = 16
    x_non = torch.randn(batch, dim)
    x_seq = torch.randn(batch, seq, dim)
    model = MultiHeadAttention(dim, dim, 4)
    z = model(x_non, x_seq)
    print(z.size())
    model = POSOLinear(dim, [dim // 2, dim // 4])
    z = model(x_non)
    print([f.size() for f in z])
