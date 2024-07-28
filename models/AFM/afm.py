from itertools import combinations

import torch
import torch.nn as nn
from torch import Tensor


class AttentionalFactorizationMachine(nn.Module):
    def __init__(
        self,
        num_fields: int,
        emb_dim: int,
        attn_dim: int,
        num_classes: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        combs = list(combinations(range(num_fields), 2))
        self.comb_i = [c[0] for c in combs]
        self.comb_j = [c[1] for c in combs]
        self.linear_attn = nn.Sequential(
            nn.Linear(emb_dim, attn_dim, bias),
            nn.ReLU(inplace=True),
            nn.Linear(attn_dim, 1, bias),
        )
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Arguments:
            x -- Tensor. shape: (batch_size, num_fields, emb_dim)

        Returns:
            _description_
        """
        x_i = x[:, self.comb_i]  # (batch_size, num_fields * (num_fields-1)/2, emb_dim)
        x_j = x[:, self.comb_j]  # (batch_size, num_fields * (num_fields-1)/2, emb_dim)
        x_cross = x_i * x_j
        attn_score = self.linear_attn(
            x_cross
        )  # [batch_size, num_fields * (num_fields-1)/2, 1]
        attn_score = torch.softmax(attn_score, dim=1)
        f = torch.sum(attn_score * x_cross, dim=1)  # (batch_size, emb_dim)
        y = self.fc(f)  # (batch_size, num_classes)
        return y


if __name__ == "__main__":
    num_fields = 10
    emb_dim = 8
    x = torch.randn(2, num_fields, emb_dim)
    model = AttentionalFactorizationMachine(
        num_fields=num_fields, emb_dim=emb_dim, attn_dim=32
    )
    y = model(x)
    print(y.size())
