import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim_in: int
    dim_hidden: int
    dim_out: int
    hidden_layer_num: int


class LinearReLU(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int) -> None:
        super(LinearReLU, self).__init__()
        self.linear = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class DNN(nn.Module):
    def __init__(
        self, dim_in: int, dim_hidden: int, dim_out: int, hidden_layer_num: int
    ) -> None:
        super(DNN, self).__init__()
        self.fc_in = LinearReLU(dim_in, dim_hidden)
        self.fc_out = LinearReLU(dim_hidden, dim_out)
        self.fc_hiddens = (
            nn.ModuleDict(
                {
                    f"fc_{i}": LinearReLU(dim_hidden, dim_hidden)
                    for i in range(hidden_layer_num)
                }
            )
            if hidden_layer_num > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        x = self.fc_in(x)
        x = self.fc_hiddens(x)
        x = self.fc_out(x)
        return x


class Predictor(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, use_sigmoid: bool = False) -> None:
        super(Predictor, self).__init__()
        self.pred = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.Sigmoid() if use_sigmoid else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(x)


class LHUC(nn.Module):
    def __init__(
        self,
        item_args: ModelArgs,
        user_args: ModelArgs,
        predict_dim: int,
        use_sigmoid: bool = False,
    ):
        super(LHUC, self).__init__()
        assert (
            item_args.hidden_layer_num == user_args.hidden_layer_num
        ), f"{item_args.hidden_layer_num} != {user_args.hidden_layer_num}"
        self.item_branch = DNN(
            item_args.dim_in,
            item_args.dim_hidden,
            item_args.dim_out,
            item_args.hidden_layer_num,
        )

        self.user_branch = DNN(
            user_args.dim_in,
            user_args.dim_hidden,
            user_args.dim_out,
            user_args.hidden_layer_num,
        )

        self.predict = Predictor(user_args.dim_out, predict_dim, use_sigmoid)

    def _cross(self, x_i: torch.Tensor, x_u: torch.Tensor) -> torch.Tensor:
        # element-wise multiplication
        out = x_i * x_u
        return out

    def forward(
        self, item_input: torch.Tensor, user_input: torch.Tensor
    ) -> torch.Tensor:
        x_i = self.item_branch.fc_in(item_input)
        x_u = self.user_branch.fc_in(user_input)
        x_i = self._cross(x_i, x_u)

        for layer_name, layer in self.item_branch.fc_hiddens.items():
            x_i = layer(x_i)
            x_u = self.user_branch.fc_hiddens[layer_name](x_u)
            x_i = self._cross(x_i, x_u)

        x_i = self.item_branch.fc_out(x_i)
        x_u = self.user_branch.fc_out(x_u)
        x_i = self._cross(x_i, x_u)

        y = self.predict(x_i)
        return y


if __name__ == "__main__":
    user_args = ModelArgs(dim_in=10, dim_hidden=64, dim_out=64, hidden_layer_num=2)
    item_args = ModelArgs(dim_in=5, dim_hidden=64, dim_out=64, hidden_layer_num=2)
    batch_size = 2

    x_user = torch.randn(batch_size, user_args.dim_in)
    x_item = torch.randn(batch_size, item_args.dim_in)
    model = LHUC(item_args, user_args, predict_dim=1, use_sigmoid=True)
    print(model)
    print(model(x_item, x_user))
