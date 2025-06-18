import torch
import torch.nn as nn
from torch import Tensor


class Matryoshka_CE_Loss(nn.Module):
    def __init__(self, relative_importance: list[float] | None = None, **kwargs):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)
        self.relative_importance = relative_importance

    def forward(self, output: tuple[torch.Tensor], target: torch.Tensor):
        losses = torch.stack([self.criterion(output_i, target) for output_i in output])

        # Set relative_importance to 1 if not specified
        rel_importance = (
            torch.ones_like(losses)
            if self.relative_importance is None
            else torch.tensor(self.relative_importance)
        )

        # Apply relative importance weights
        weighted_losses = rel_importance * losses
        return weighted_losses.sum()


class MRL_Linear_Layer(nn.Module):
    def __init__(
        self,
        dims: list[int],
        num_classes: int,
        efficient: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.dims = dims
        self.num_classes = num_classes
        self.efficient = efficient
        if self.efficient:
            self.fc = nn.Linear(dims[-1], self.num_classes, bias=bias)
        else:
            self.fc = nn.ModuleList(
                [nn.Linear(dims[i], num_classes, bias=bias) for i in range(len(dims))]
            )

    def forward(self, x: Tensor) -> Tensor:
        feat = list()
        for i, dim in enumerate(self.dims):
            if isinstance(self.fc, nn.Linear):
                bias = self.fc.bias if self.fc.bias else 0
                z = x[:, :dim] @ self.fc.weight[:, :dim].t() + bias
                # [B, D] x [D, C] = [B, C]
            else:
                z = self.fc[i](x[:, :dim])
            feat.append(z)
        return torch.stack(feat, dim=0)  # [G, B, C]


if __name__ == "__main__":
    x = torch.randn(10, 100)
    nesting_list = [2**i for i in range(3, 12)]
    mrl = MRL_Linear_Layer(nesting_list, num_classes=1000)
    z: tuple[torch.Tensor] = mrl(x)
    print(z)
