from itertools import combinations

import torch
from torch import Tensor, nn


class BilinearInteraction(nn.Module):
    """
    用于计算特征之间的双线性交互的类。

    支持三种双线性类型：'all', 'each', 'interaction'。
    'all' 类型计算所有特征之间的双线性交互；
    'each' 类型为每个特征单独计算双线性交互；
    'interaction' 类型计算每对特征之间的双线性交互。

    参数:
    - num_fields: 特征的数量。
    - emb_dim: 特征的嵌入维度。
    - bilinear_type: 双线性交互的类型。
    """

    def __init__(self, num_fields: int, emb_dim: int, bilinear_type="interaction"):
        super().__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "all":
            # 用于计算所有特征之间交互的线性层
            self.bilinear_layer = nn.Linear(emb_dim, emb_dim, bias=False)
        elif self.bilinear_type == "each":
            # 为每个特征单独配备一个线性层，用于计算其与其他特征的交互
            self.bilinear_layer = nn.ModuleList(
                [nn.Linear(emb_dim, emb_dim, bias=False) for _ in range(num_fields - 1)]
            )
        elif self.bilinear_type == "interaction":
            # 为每对特征配备一个线性层，用于计算交互
            self.bilinear_layer = nn.ModuleList(
                [
                    nn.Linear(emb_dim, emb_dim, bias=False)
                    for _, _ in combinations(range(num_fields), 2)
                ]
            )
        else:
            raise NotImplementedError()

    def forward(self, feats: Tensor) -> Tensor:
        """
        前向传播方法，根据双线性类型计算特征之间的交互。

        参数:
        - feats: 输入的特征张量，形状为(batch_size, num_fields, emb_dim)。

        返回:
        - 计算得到的双线性交互张量，形状为(batch_size, num_fields*(num_fields-1)/2, emb_dim)。
        """
        # 将特征张量拆分为列表，每个特征对应一个张量
        feats_list = torch.split(feats, 1, dim=1)
        if self.bilinear_type == "all":
            # 计算所有特征两两之间的双线性交互
            bilinear_list = [
                self.bilinear_layer(v_i) * v_j
                for v_i, v_j in combinations(feats_list, 2)
            ]
        elif self.bilinear_type == "each":
            # 对每个特征分别计算与其他特征的双线性交互
            bilinear_list = [
                self.bilinear_layer[i](feats_list[i]) * feats_list[j]
                for i, j in combinations(range(len(feats_list)), 2)
            ]
        elif self.bilinear_type == "interaction":
            # 对每对特征分别计算双线性交互
            bilinear_list = [
                self.bilinear_layer[i](v[0]) * v[1]
                for i, v in enumerate(combinations(feats_list, 2))
            ]
        # 将所有双线性交互张量在维度1上拼接
        return torch.cat(bilinear_list, dim=1)


class SqueezeExcition(nn.Module):
    """
    Squeeze and Excitation block class.

    Parameters:
    - num_fields: The number of input fields.
    - reduction_ratio: The reduction ratio for feature dimension reduction in the bottleneck layer. Default is 8.
    - bias: Whether to use bias in the linear layer. Default is False.

    input:
    - x: The input tensor. Shape: (batch_size, num_fields, emb_dim).

    Returns:
    - The recalibrated feature tensor. Shape: (batch_size, num_fields, emb_dim).
    """

    def __init__(self, num_fields: int, reduction_ratio: int = 8, bias: bool = False):
        super().__init__()
        reduced_dim = max(1, num_fields // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(reduced_dim, num_fields, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters:
        - x: The input tensor. Shape: (batch_size, num_fields, emb_dim).

        Returns:
        - The recalibrated feature tensor.
            Shape: (batch_size, num_fields, emb_dim).
        """
        # 对于特征嵌入的均值作为 squeeze 层输出
        Z = torch.mean(x, dim=-1, out=None)  # (batch_size, num_fields)
        A = self.excitation(Z)  # (batch_size, num_fields)
        V = x * A.unsqueeze(-1)  # (batch_size, num_fields, emb_dim)
        return V


class LinearWithAct(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        act: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "tanh":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError()
        if batch_norm:
            self.bn = nn.BatchNorm1d(dim_out)
        else:
            self.bn = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class FiBiNet(nn.Module):
    def __init__(
        self,
        num_fields: int,
        emb_dim: int,
        dnn_hidden_dims: list[int],
        num_classes: int = 1,
        act: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        bilinear_type: str = "each",
        reduce_ratio: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        self.bilinear_layer_1 = BilinearInteraction(num_fields, emb_dim, bilinear_type)
        self.bilinear_layer_2 = BilinearInteraction(num_fields, emb_dim, bilinear_type)
        self.selayer = SqueezeExcition(num_fields, reduce_ratio, bias=False)
        dims = (
            [num_fields * (num_fields - 1) * emb_dim] + dnn_hidden_dims + [num_classes]
        )
        fcs = []
        for i in range(len(dims) - 1):
            activation = act if i < len(dims) - 2 else "sigmoid"
            fc = LinearWithAct(
                dims[i],
                dims[i + 1],
                act=activation,
                dropout=dropout,
                batch_norm=batch_norm,
                bias=bias,
            )
            fcs.append(fc)
        self.fcs = nn.Sequential(*fcs)

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Arguments:
            x -- shape (batch_size, num_fields, emb_dim)

        Returns:
            Tensor -- shape (batch_size, num_classes)
        """
        f1 = self.bilinear_layer_1(x)  # (batch_size, fields*(fields-1)/2, emb_dim)
        f2 = self.selayer(x)  # (batch_size, fields, emb_dim)
        f2 = self.bilinear_layer_2(f2)  # (batch_size, fields*(fields-1)/2, emb_dim)

        feat = torch.cat([f1, f2], dim=1)  # (batch_size, fields*(fields-1), emb_dim)
        feat = feat.view(x.size(0), -1)  # (batch_size, fields*(fields-1)*emb_dim)

        y = self.fcs(feat)  # (batch_size, num_classes)
        return y


if __name__ == "__main__":
    num_fields = 6
    emb_dim = 4
    bilinear_type = "each"
    bilinear_layer = BilinearInteraction(num_fields, emb_dim, bilinear_type)
    print(bilinear_layer)
    feats = torch.randn(2, num_fields, emb_dim)
    bilinear_output = bilinear_layer(feats)
    print(bilinear_output.shape)
    fields = num_fields * (num_fields - 1) // 2
    selayer = SqueezeExcition(fields)
    y = selayer(bilinear_output)
    print(y.shape)

    model = FiBiNet(
        num_fields,
        emb_dim,
        dnn_hidden_dims=[32, 16],
        num_classes=1,
        act="relu",
        dropout=0.0,
        batch_norm=False,
        bilinear_type="each",
        reduce_ratio=8,
        bias=True,
    )
    y = model(feats)
    print(y.shape)
