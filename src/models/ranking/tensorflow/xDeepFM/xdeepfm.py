from dataclasses import dataclass

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model, layers


@dataclass
class CINArgs:
    """
    CIN (Compressed Interaction Network) 参数。

    属性:
        dim_in: 输入维度。
        dim_hiddens: 隐藏层维度列表。
        num_classes: 输出类别数量，默认为1。
        split_half: 是否在每个隐藏层后将维度减半，默认为True。
        bias: 是否在卷积层中使用偏置，默认为True。
    """

    dim_in: int
    dim_hiddens: list[int]
    num_classes: int = 1
    split_half: bool = True
    bias: bool = True


@dataclass
class DNNArgs:
    """
    DNN (Deep Neural Network) 参数。

    属性:
        dim_in: 输入维度。
        dim_hiddens: 隐藏层维度列表。
        bias: 是否在全连接层中使用偏置，默认为True。
        dropout: Dropout 比率，默认为0.0。
        activation: 激活函数名称，默认为"relu"。
    """

    dim_in: int
    dim_hiddens: list[int]
    bias: bool = True
    dropout: float = 0.0
    activation: str = "relu"


@dataclass
class xDeepFMArgs:
    """
    xDeepFM 模型参数。

    属性:
        num_fields: 特征字段数量。
        emb_dim: 嵌入维度。
        dnn_args: DNNArgs 实例。
        cin_args: CINArgs 实例。
        num_classes: 输出类别数量，默认为1。
    """

    num_fields: int
    emb_dim: int
    dnn_args: DNNArgs
    cin_args: CINArgs
    num_classes: int = 1

    def __post_init__(self):
        """
        初始化后处理，设置 DNN 和 CIN 的输入维度。
        """
        self.dnn_args.dim_in = self.num_fields * self.emb_dim
        self.cin_args.dim_in = self.num_fields


class LinearACT(layers.Layer):
    """
    线性层加激活函数和Dropout。

    参数:
        dim_in: 输入维度。
        dim_out: 输出维度。
        bias: 是否使用偏置，默认为True。
        activation: 激活函数名称，默认为"relu"。
        dropout: Dropout 比率，默认为0.0。
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = True,
        activation: str | None = "relu",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fc = layers.Dense(dim_out, use_bias=bias, input_shape=(dim_in,))

        if dropout > 0:
            self.dropout_layer = layers.Dropout(dropout)
        else:
            self.dropout_layer = layers.Identity()

        if activation == "relu":
            self.activation_layer = layers.ReLU()
        elif activation == "sigmoid":
            self.activation_layer = layers.Activation("sigmoid")
        elif activation == "tanh":
            self.activation_layer = layers.Activation("tanh")
        elif activation is None:
            self.activation_layer = layers.Identity()
        else:
            raise ValueError("Invalid activation function")

    def call(self, x: Tensor) -> Tensor:
        """
        前向传播。

        参数:
            x: 输入张量。

        返回:
            输出张量。
        """
        x = self.fc(x)
        x = self.activation_layer(x)
        x = self.dropout_layer(x)
        return x


class DNN(layers.Layer):
    """
    深度神经网络层。

    参数:
        args: DNNArgs 实例。
    """

    def __init__(
        self,
        args: DNNArgs,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = len(args.dim_hiddens)
        self.fcs = []
        for i in range(self.num_layers):
            self.fcs.append(
                LinearACT(
                    args.dim_in if i == 0 else args.dim_hiddens[i - 1],
                    args.dim_hiddens[i],
                    bias=args.bias,
                    activation=args.activation if i != self.num_layers - 1 else None,
                    dropout=args.dropout,
                )
            )

    def call(self, x: Tensor) -> Tensor:
        """
        前向传播。

        参数:
            x: 输入张量。

        返回:
            输出张量。
        """
        for fc_layer in self.fcs:
            x = fc_layer(x)
        return x


class CompressedInteractionNetwork(layers.Layer):
    """
    压缩交互网络类，用于处理高维特征交互的深度学习模型。

    参数:
        dim_in: 输入维度, num_fields。
        dim_hiddens: 隐藏层维度列表。
        num_classes: 输出类别数量，默认为1。
        split_half: 是否在每个隐藏层后将维度减半，默认为True。
        bias: 是否在卷积层中使用偏置，默认为True。
    """

    def __init__(
        self,
        dim_in: int,
        dim_hiddens: list[int],
        num_classes: int = 1,
        split_half: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = len(dim_hiddens)
        self.split_half = split_half
        self.conv_layers = []
        prev_dim = dim_in
        fc_dim_in = 0
        for i in range(self.num_layers):
            self.conv_layers.append(
                layers.Conv1D(
                    dim_hiddens[i],
                    1,
                    use_bias=bias,
                    input_shape=(None, dim_in * prev_dim),
                )
            )
            if self.split_half and i != self.num_layers - 1:
                dim_hiddens[i] //= 2
            prev_dim = dim_hiddens[i]
            fc_dim_in += prev_dim
        self.fc = layers.Dense(num_classes, use_bias=bias, input_shape=(fc_dim_in,))

    def call(self, x: Tensor) -> Tensor:
        """
        前向传播。

        参数:
            x: 输入特征，形状为(batch_size, num_fields, embed_dim)。

        返回:
            Tensor: 输出特征，形状为(batch_size, 1)。
        """
        xs = []
        x0, h = x, x
        for i in range(self.num_layers):
            # (batch_size, num_fields, h_dim, embed_dim)
            x_outer = tf.einsum("bmd,bnd->bmnd", x0, h)
            # (batch_size, num_fields * h_dim, embed_dim)
            x_reshaped = tf.reshape(x_outer, (x_outer.shape[0], -1, x_outer.shape[-1]))

            # Conv1D expects (batch, steps, channels)
            # Here, steps = embed_dim, channels = num_fields * h_dim
            # So we need to transpose to (batch_size, embed_dim, num_fields * h_dim)
            x_transposed = tf.transpose(x_reshaped, perm=[0, 2, 1])

            x_conv = self.conv_layers[i](x_transposed)
            x_relu = tf.nn.relu(x_conv)

            # Transpose back to (batch_size, num_fields * h_dim, embed_dim)
            x_processed = tf.transpose(x_relu, perm=[0, 2, 1])

            if self.split_half and i != self.num_layers - 1:
                # Split along the second dimension (num_fields * h_dim)
                splits = tf.split(x_processed, num_or_size_splits=2, axis=1)
                x = splits[0]
                h = splits[1]
            else:
                h = x_processed
                x = x_processed  # Keep x updated for the next iteration's x_outer calculation
            xs.append(x)

        # (batch_size, sum(dim_hiddens), embed_dim)
        f = tf.concat(xs, axis=1)
        # (batch_size, sum(dim_hiddens))
        f = tf.reduce_sum(f, axis=2)
        return self.fc(f)


class xDeepFM(Model):
    """
    xDeepFM 模型。

    参数:
        args: xDeepFMArgs 实例。
    """

    def __init__(
        self,
        args: xDeepFMArgs,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
            bias=bool(args.num_classes),  # Convert int to bool
            activation=None,
        )
        self.pred = layers.Dense(
            args.num_classes,
            use_bias=True,
            activation="sigmoid",
        )

    def call(self, x: Tensor) -> Tensor:
        """
        前向传播。

        参数:
            x: 输入张量，形状为(batch_size, num_fields, embed_dim)。

        返回:
            Tensor: 输出张量，形状为(batch_size, 1)。
        """
        # Linear part
        x_linear = tf.reshape(x, (x.shape[0], -1))
        x_linear = self.linear(x_linear)

        # CIN part
        x_cin = self.cin(x)

        # Deep part
        x_deep = tf.reshape(x, (x.shape[0], -1))
        x_deep = self.deep(x_deep)

        # Concatenate and predict
        y = tf.concat([x_linear, x_cin, x_deep], axis=1)
        y = self.pred(y)
        return y


if __name__ == "__main__":
    num_fields = 10
    emb_dim = 8
    # Create a dummy input tensor
    x = tf.random.normal((2, num_fields, emb_dim))

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
    # Build the model with a dummy input to initialize weights
    model.build(input_shape=(None, num_fields, emb_dim))
    model.summary()
    y = model(x)
    print(y)
