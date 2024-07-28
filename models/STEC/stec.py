from itertools import combinations

import torch
import torch.nn as nn
from torch import Tensor


class STECBlock(nn.Module):
    def __init__(
        self,
        num_fields: int,
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(emb_dim, emb_dim, bias=False)
                for _, _ in combinations(range(num_fields), 2)
            ]
        )
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.num_fields = num_fields

    def forward(self, x: Tensor) -> Tensor:
        f = x.split(1, dim=1)  # (B, 1, E) x F
        qk = [self.linears[i](v[0]) * v[1] for i, v in enumerate(combinations(f, 2))]
        qk = torch.cat(qk, dim=1)  # (B, F * F, E)
        qk = torch.mean(qk, dim=-1)  # (B, F * F)
        v = self.W_v(x)
        z = qk @ v
        # v , (B, F, )
