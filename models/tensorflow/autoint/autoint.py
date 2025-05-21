import tensorflow as tf


class AutoInt(tf.keras.Model):
    def __init__(
        self,
        nfeat: int,
        dim_in: int,
        dim_qk: int,
        dim_v: int,
        nhead: int,
        out_dim: int = 1,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.nfeat = nfeat
        self.dim_in = dim_in
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.nhead = nhead
        self.out_dim = out_dim
        self.bias = bias

        # Initialize linear transformations
        self.linear_q = tf.keras.layers.Dense(dim_qk, use_bias=bias, name="linear_q")
        self.linear_k = tf.keras.layers.Dense(dim_qk, use_bias=bias, name="linear_k")
        self.linear_v = tf.keras.layers.Dense(dim_v, use_bias=bias, name="linear_v")

        # Initialize linear transformation from value to input dimension
        self.fc1 = tf.keras.layers.Dense(dim_in, use_bias=bias, name="fc1")

        # Initialize final output linear transformation
        self.fc2 = tf.keras.layers.Dense(out_dim, use_bias=bias, name="fc2")

        # Initialize activation function as ReLU
        self.act = tf.keras.layers.ReLU()

        # Sigmoid for output
        self.sigmoid_output = tf.keras.layers.Activation("sigmoid")

    def _self_attention(self, x: tf.Tensor) -> tf.Tensor:
        """
        Implement self-attention mechanism in TensorFlow.

        Args:
            x: Input tensor with shape (batch, nfeat, dim_in).

        Returns:
            Tensor after self-attention with shape (batch, nfeat, dim_in).
        """
        # Calculate query, key, value
        q = self.linear_q(x)  # (batch, nfeat, dim_qk)
        k = self.linear_k(x)  # (batch, nfeat, dim_qk)
        v = self.linear_v(x)  # (batch, nfeat, dim_v)

        # Reshape for multi-head attention
        # PyTorch: rearrange(q, "b n (nhead d) -> (b nhead) n d", nhead=self.nhead, d=q.size(2) // self.nhead)
        # TensorFlow equivalent: reshape and transpose
        batch_size = tf.shape(q)[0]
        dim_qk_per_head = self.dim_qk // self.nhead
        dim_v_per_head = self.dim_v // self.nhead

        q = tf.reshape(q, (batch_size, self.nfeat, self.nhead, dim_qk_per_head))
        q = tf.transpose(q, perm=[0, 2, 1, 3])  # (batch, nhead, nfeat, dim_qk_per_head)
        q = tf.reshape(
            q, (-1, self.nfeat, dim_qk_per_head)
        )  # (batch * nhead, nfeat, dim_qk_per_head)

        k = tf.reshape(k, (batch_size, self.nfeat, self.nhead, dim_qk_per_head))
        k = tf.transpose(k, perm=[0, 2, 1, 3])  # (batch, nhead, nfeat, dim_qk_per_head)
        k = tf.reshape(
            k, (-1, self.nfeat, dim_qk_per_head)
        )  # (batch * nhead, nfeat, dim_qk_per_head)

        v = tf.reshape(v, (batch_size, self.nfeat, self.nhead, dim_v_per_head))
        v = tf.transpose(v, perm=[0, 2, 1, 3])  # (batch, nhead, nfeat, dim_v_per_head)
        v = tf.reshape(
            v, (-1, self.nfeat, dim_v_per_head)
        )  # (batch * nhead, nfeat, dim_v_per_head)

        # Calculate attention weights and apply softmax
        # PyTorch: att = torch.einsum("bid,bjd->bij", q, k) / (k.size(2) ** 0.5)
        # TensorFlow equivalent: tf.matmul
        # q shape: (batch * nhead, nfeat, dim_qk_per_head)
        # k shape: (batch * nhead, nfeat, dim_qk_per_head)
        # Need to transpose k for matmul: (batch * nhead, dim_qk_per_head, nfeat)
        att = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            tf.cast(dim_qk_per_head, tf.float32)
        )  # (batch * nhead, nfeat, nfeat)
        att = tf.nn.softmax(att, axis=-1)  # (batch * nhead, nfeat, nfeat)

        # Calculate weighted sum
        # PyTorch: z = torch.einsum("bij,bjk->bik", att, v)
        # TensorFlow equivalent: tf.matmul
        # att shape: (batch * nhead, nfeat, nfeat)
        # v shape: (batch * nhead, nfeat, dim_v_per_head)
        z = tf.matmul(att, v)  # (batch * nhead, nfeat, dim_v_per_head)

        # Reshape back to original batch and feature dimensions
        # PyTorch: rearrange(z, "(b nhead) n d -> b n (nhead d)", nhead=self.nhead)
        # TensorFlow equivalent: reshape
        z = tf.reshape(z, (batch_size, self.nhead, self.nfeat, dim_v_per_head))
        z = tf.transpose(z, perm=[0, 2, 1, 3])  # (batch, nfeat, nhead, dim_v_per_head)
        z = tf.reshape(
            z, (batch_size, self.nfeat, self.nhead * dim_v_per_head)
        )  # (batch, nfeat, dim_v)

        return z

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass in TensorFlow.

        Args:
            inputs: Input tensor with shape (batch, nfeat, dim_in).

        Returns:
            Model output tensor with shape (batch, out_dim).
        """
        # Apply self-attention mechanism
        x_att = self._self_attention(inputs)  # (batch, nfeat, dim_v)

        # Apply ReLU activation and linear transformation with residual connection
        # PyTorch: f = self.act(self.fc1(x_att) + x)
        # TensorFlow:
        f = self.act(self.fc1(x_att) + inputs)  # (batch, nfeat, dim_in)

        # Flatten features and pass through final linear transformation
        # PyTorch: y = self.fc2(f.flatten(start_dim=1))
        # TensorFlow:
        f_flat = tf.keras.layers.Flatten()(f)  # (batch, nfeat * dim_in)
        y = self.fc2(f_flat)  # (batch, out_dim)

        # Apply sigmoid for CTR prediction
        y = self.sigmoid_output(y)  # (batch, out_dim)

        return y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nfeat": self.nfeat,
                "dim_in": self.dim_in,
                "dim_qk": self.dim_qk,
                "dim_v": self.dim_v,
                "nhead": self.nhead,
                "out_dim": self.out_dim,
                "bias": self.bias,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Example usage (for testing purposes, similar to PyTorch __main__ block)
if __name__ == "__main__":
    batch_size = 4
    nfeat = 27
    dim_in = 8  # Corresponds to emb_dim in PyTorch example
    dim_qk = 8
    dim_v = 16
    nhead = 4
    out_dim = 1

    # Create dummy input data
    # Input shape is (batch, nfeat, dim_in)
    dummy_input = tf.random.normal(shape=(batch_size, nfeat, dim_in))

    # Initialize the TensorFlow AutoInt model
    model = AutoInt(
        nfeat=nfeat,
        dim_in=dim_in,
        dim_qk=dim_qk,
        dim_v=dim_v,
        nhead=nhead,
        out_dim=out_dim,
    )

    # Call the model with dummy input
    output = model(dummy_input)

    # Print output shape
    print(f"输出形状: {output.shape}")

    # Check if output is within [0, 1] range (sigmoid output)
    assert tf.reduce_all(output >= 0.0) and tf.reduce_all(output <= 1.0), (
        "输出不是有效的概率值"
    )

    print("测试通过!")
