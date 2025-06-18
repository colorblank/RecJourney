import torch
import torch.nn as nn
from torch import Tensor


class CoActionUnit(nn.Module):
    """
    CoActionUnit类实现了协同作用网络的模型结构。

    参数:
    - emb_dim: int, 表示嵌入维度。
    - hidden_dims: list[int], 表示隐藏层的维度列表。
    - orders: int, 默认为3, 表示使用的序数级别。

    Input:
    - ad: (batch, T)
    - his_items: (batch, seq, D)

    Returns:
    - out: (batch, sum(hidden_dims) * orders)
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dims: list[int],
        orders: int = 3,
        order_indep: bool = False,
    ) -> None:
        super().__init__()

        self.emb_dim = emb_dim
        self.orders = orders
        self.order_indep = order_indep
        dims = [emb_dim] + hidden_dims
        self.total_dims = 0
        for i in range(1, len(dims)):
            self.total_dims += dims[i] * dims[i - 1]
        self.dims = dims

    def forward(
        self, ad: Tensor, his_items: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        """
        Args:
            ad: (batch, T)
            his_items: (batch, seq, D)
        Returns:
            out: (batch, dim)
        """
        assert ad.shape[1] == self.total_dims, (
            f"expected ad to have shape (batch, {self.total_dims}), got {ad.shape[1]}"
        )
        assert his_items.shape[2] == self.emb_dim, (
            f"expected his_items to have shape (batch, seq, emb_dim), got {his_items.shape[2]}"
        )

        out = list()
        for o in range(self.orders):
            hh = his_items ** (o + 1)
            h_i = hh
            prev_dim = 0
            for i in range(1, len(self.dims)):
                start = prev_dim
                end = prev_dim + self.dims[i] * self.dims[i - 1]
                weight = ad[:, start:end]
                weight = weight.view(-1, self.dims[i - 1], self.dims[i])
                h_i = torch.matmul(h_i, weight)  # (batch, seq, dim_out)
                if i != len(self.dims) - 1:
                    h_i = torch.tanh(h_i)
                out.append(h_i)
                prev_dim = end

        res = torch.cat(out, dim=2)  # (batch, seq, (d1+d2)*orders)

        if mask is not None:
            mask = mask.unsqueeze(
                -1
            )  # Add a dimension for broadcasting: (batch, seq, 1)
            res = res * mask  # Apply mask: (batch, seq, sum(w[1]))

        out = torch.sum(res, dim=1)  # (batch, (d1+d2)*orders)

        return out


if __name__ == "__main__":
    ad = torch.randn(2, 160)
    his_items = torch.randn(2, 3, 16)
    mask = torch.tensor([[1, 1, 0], [1, 0, 1]])
    model = CoActionUnit(16, [8, 4], 3, True)
    y = model(ad, his_items, mask)
    print(y.shape)
