import torch
import torch.nn as nn


class LinearReLU(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, bias: bool = True) -> None:
        super(LinearReLU, self).__init__()
        self.linear = nn.Linear(dim_in, dim_hidden, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class CrossNetwork(nn.Module):
    def __init__(self, dim_in: int, hidden_layer_num: int, bias: bool = True):
        super(CrossNetwork, self).__init__()
        self.linears = nn.ModuleDict(
            {
                f"fc_{i}": LinearReLU(dim_in, dim_in, bias=bias)
                for i in range(hidden_layer_num)
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_i = x
        for i in range(len(self.linears)):
            x_prev = x_i
            x_i = x * self.linears[f"fc_{i}"](x_i) + x_prev
            # x_i = x0 * (w_i * x_i + b_i) + x_{i-1}
        return x_i


if __name__ == "__main__":
    x = torch.randn(2, 10)
    model = CrossNetwork(dim_in=10, hidden_layer_num=2)
    print(model(x).size())
