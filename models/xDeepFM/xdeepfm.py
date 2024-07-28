from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CINArgs:
    dim_in: int
    dim_hiddens: list[int]
    num_classes: int = 1
    split_half: bool = True
    bias: bool = True


@dataclass
class DNNArgs:
    dim_in: int
    dim_hiddens: list[int]
    bias: bool = True
    dropout: float = 0.0
    activation: str = "relu"


@dataclass
class xDeepFMArgs:
    num_fields: int
    emb_dim: int
    dnn_args: DNNArgs
    cin_args: CINArgs
    num_classes: int = 1

    def __post_init__(self):
        self.dnn_args.dim_in = num_fields * emb_dim
        self.cin_args.dim_in = num_fields


class LinearACT(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=bias)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError("Invalid activation function")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DNN(nn.Module):
    def __init__(
        self,
        args: DNNArgs,
    ):
        super().__init__()
        self.num_layers = len(args.dim_hiddens)
        fcs = list()
        for i in range(self.num_layers):
            fcs.append(
                LinearACT(
                    args.dim_in if i == 0 else args.dim_hiddens[i - 1],
                    args.dim_hiddens[i],
                    bias=args.bias,
                    activation=args.activation if i != self.num_layers - 1 else None,
                    dropout=args.dropout,
                )
            )
        self.fcs = nn.Sequential(*fcs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcs(x)


class CompressedInteractionNetwork(nn.Module):
    """
    压缩交互网络类，用于处理高维特征交互的深度学习模型。

    参数:
    - dim_in: 输入维度, num_fields。
    - dim_hiddens: 隐藏层维度列表。
    - split_half: 是否在每个隐藏层后将维度减半，默认为True。
    - bias: 是否在卷积层中使用偏置，默认为True。

    input:
    - x: 输入特征，形状为(batch_size, num_fields, embed_dim)。

    Returns:
    - torch.Tensor: 输出特征，形状为(batch_size, 1)。

    """

    def __init__(
        self,
        dim_in: int,
        dim_hiddens: list[int],
        num_classes: int = 1,
        split_half: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.num_layers = len(dim_hiddens)
        self.split_half = split_half
        self.conv_layers = nn.ModuleList()
        prev_dim = dim_in
        fc_dim_in = 0
        for i in range(self.num_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    dim_in * prev_dim,
                    dim_hiddens[i],
                    1,
                    bias=bias,
                )
            )
            if self.split_half and i != self.num_layers - 1:
                dim_hiddens[i] //= 2
            prev_dim = dim_hiddens[i]
            fc_dim_in += prev_dim
        self.fc = nn.Linear(fc_dim_in, num_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Arguments:
            x -- torch.Tensor. (batch_size, num_fields, embed_dim)

        Returns:
            torch.Tensor. (batch_size, 1)
        """
        xs = list()
        x0, h = x, x
        for i in range(self.num_layers):
            x = torch.einsum("bmd,bnd->bmnd", x0, h)
            x = x.view(x.size(0), -1, x.size(-1))  # (batch_size, m*n, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        f = torch.cat(xs, dim=1)
        # (batch_size, dim_hidden * layer_num, embed_dim)
        f = torch.sum(f, 2)
        # (batch_size, dim_hidden * layer_num)
        return self.fc(f)


class xDeepFM(nn.Module):
    def __init__(
        self,
        args: xDeepFMArgs,
    ):
        super().__init__()
        self.cin = CompressedInteractionNetwork(
            args.num_fields,
            args.cin_args.dim_hiddens,
            args.cin_args.num_classes,
            args.cin_args.split_half,
            args.cin_args.bias,
        )
        self.deep = DNN(args.dnn_args)
        self.linear = LinearACT(
            args.num_fields * args.emb_dim,
            args.num_classes,
            bias=args.num_classes,
            activation=None,
        )
        self.pred = nn.Sequential(
            LinearACT(
                args.cin_args.num_classes
                + args.dnn_args.dim_hiddens[-1]
                + args.num_classes,
                args.num_classes,
                bias=True,
                activation=None,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Arguments:
            x -- torch.Tensor. (batch_size, num_fields, embed_dim)

        Returns:
            torch.Tensor. (batch_size, 1)
        """
        x_linear = x.view(x.size(0), -1)
        x_linear = self.linear(x_linear)
        x_cin = self.cin(x)
        x_deep = x.view(x.size(0), -1)
        x_deep = self.deep(x_deep)
        y = torch.cat([x_linear, x_cin, x_deep], dim=1)
        y = self.pred(y)
        y = torch.sigmoid(y)
        return y


if __name__ == "__main__":
    num_fields = 10
    emb_dim = 8
    x = torch.randn(2, num_fields, emb_dim)
    args = xDeepFMArgs(
        num_fields=num_fields,
        emb_dim=emb_dim,
        cin_args=CINArgs(
            dim_in=num_fields,
            dim_hiddens=[32, 16],
            num_classes=1,
            split_half=True,
            bias=True,
        ),
        dnn_args=DNNArgs(
            dim_in=num_fields * emb_dim,
            dim_hiddens=[32, 16],
            bias=True,
            activation="relu",
            dropout=0.2,
        ),
    )
    model = xDeepFM(args)
    print(model)
    y = model(x)
    print(y)
