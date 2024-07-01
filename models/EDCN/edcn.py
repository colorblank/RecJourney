import torch
import torch.nn as nn

from typing import Literal


class BridgeLayer(nn.Module):
    def __init__(
        self,
        dim_in: int,
        mode: Literal["add", "product", "concat", "attention"] = "product",
    ):
        super(BridgeLayer, self).__init__()
        if mode == "attention":
            self.fc_deep = nn.Sequential(
                nn.Linear(dim_in, dim_in, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_in, bias=False),
            )
            self.fc_cross = nn.Sequential(
                nn.Linear(dim_in, dim_in, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_in, bias=False),
            )
        elif mode == "concat":
            self.fc = nn.Sequential(
                nn.Linear(dim_in * 2, dim_in, bias=True),
                nn.ReLU(inplace=True),
            )

    def forward(self, x_deep: torch.Tensor, x_cross: torch.Tensor) -> torch.Tensor:
        if self.mode == "add":
            return x_deep + x_cross
        elif self.mode == "product":
            return x_deep * x_cross
        elif self.mode == "concat":
            return self.fc(torch.cat([x_deep, x_cross], dim=-1))
        elif self.mode == "attention":
            deep_weight = torch.softmax(self.fc_deep(x_deep), dim=-1)
            cross_weight = torch.softmax(self.fc_cross(x_cross), dim=-1)
            return deep_weight * x_deep + cross_weight * x_cross
