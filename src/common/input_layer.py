import tensorflow as tf
from tensorflow.keras import layers


class InputLayer(layers.Layer):
    """
    输入层，用于处理原始特征并生成适合模型输入的张量。

    TODO: 根据实际特征处理逻辑完善此层。
    目前仅作为 WideAndDeep 模型的占位符，返回模拟的深层和宽层特征。
    """

    def __init__(self, feature_specs: dict, **kwargs):
        """
        初始化 InputLayer。

        Args:
            feature_specs (dict): 描述特征的字典，包含类型、词汇大小、嵌入维度等。
        """
        super().__init__(**kwargs)
        self.feature_specs = feature_specs
        # 这里可以根据 feature_specs 定义嵌入层、归一化层等
        # 例如: self.embedding_layers = {}
        # for feature_name, spec in feature_specs.items():
        #     if spec['type'] == 'categorical':
        #         self.embedding_layers[feature_name] = layers.Embedding(...)

    def call(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """
        前向传播，处理输入特征。

        Args:
            inputs (dict): 原始输入特征的字典，键为特征名，值为原始张量。

        Returns:
            dict: 包含处理后的深层特征和宽层特征的字典。
                'deep_features' (tf.Tensor): 适合深层网络的特征张量。
                'wide_features' (tf.Tensor): 适合宽层网络的特征张量。
        """
        # 这是一个占位符实现，实际应根据 feature_specs 处理 inputs
        # 并生成正确的 deep_features 和 wide_features
        batch_size = tf.shape(list(inputs.values())[0])[0]

        # 模拟 deep_features 的维度计算
        deep_input_dim_calc = 0
        for feature_name, spec in self.feature_specs.items():
            if spec.get("use_in_deep", False):
                if spec["type"] == "categorical":
                    deep_input_dim_calc += spec.get("embedding_dim", 16)
                elif spec["type"] == "numerical":
                    deep_input_dim_calc += 1

        # 模拟 wide_features 的维度计算
        wide_input_dim_calc = 0
        for feature_name, spec in self.feature_specs.items():
            if spec.get("use_in_wide", False):
                if spec["type"] == "categorical":
                    wide_input_dim_calc += spec["vocab_size"]
                elif spec["type"] == "numerical":
                    wide_input_dim_calc += 1

        # 返回模拟的张量
        return {
            "deep_features": tf.random.normal(shape=(batch_size, deep_input_dim_calc)),
            "wide_features": tf.random.normal(shape=(batch_size, wide_input_dim_calc)),
        }

    def get_config(self):
        config = super().get_config()
        config.update({"feature_specs": self.feature_specs})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    # 示例用法
    dummy_feature_specs = {
        "user_id": {
            "type": "categorical",
            "vocab_size": 1000,
            "embedding_dim": 32,
            "use_in_deep": True,
            "use_in_wide": False,
        },
        "item_id": {
            "type": "categorical",
            "vocab_size": 5000,
            "embedding_dim": 32,
            "use_in_deep": True,
            "use_in_wide": False,
        },
        "age": {"type": "numerical", "use_in_deep": True, "use_in_wide": True},
        "gender": {
            "type": "categorical",
            "vocab_size": 2,
            "embedding_dim": 8,
            "use_in_deep": True,
            "use_in_wide": True,
        },
        "wide_feature_1": {
            "type": "numerical",
            "use_in_deep": False,
            "use_in_wide": True,
        },
        "wide_feature_2": {
            "type": "categorical",
            "vocab_size": 10,
            "use_in_deep": False,
            "use_in_wide": True,
        },
    }
    batch_size = 4
    dummy_inputs = {
        "user_id": tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=1000, dtype=tf.int32
        ),
        "item_id": tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=5000, dtype=tf.int32
        ),
        "age": tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=100, dtype=tf.float32
        ),
        "gender": tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=2, dtype=tf.int32
        ),
        "wide_feature_1": tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=1, dtype=tf.float32
        ),
        "wide_feature_2": tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=10, dtype=tf.int32
        ),
    }

    input_layer = InputLayer(dummy_feature_specs)
    processed_features = input_layer(dummy_inputs)

    print("Deep Features Shape:", processed_features["deep_features"].shape)
    print("Wide Features Shape:", processed_features["wide_features"].shape)
    print("InputLayer 测试通过!")
