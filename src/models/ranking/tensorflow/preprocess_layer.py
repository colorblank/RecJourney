"""
通用的 TensorFlow Keras 数据预处理层。
"""

import tensorflow as tf
from typing import Any, Dict, Optional


class KerasPreprocessorLayer(tf.keras.layers.Layer):
    """
    通用的 TensorFlow Keras 数据预处理层。

    封装了缺失值处理、特征转换（分桶、归一化、编码）等步骤。
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        初始化 KerasPreprocessorLayer。

        Args:
            config: 预处理配置字典。
            **kwargs: 其他 Keras 层参数。
        """
        super().__init__(**kwargs)
        self.config = config
        self.normalization_layers: Dict[str, tf.keras.layers.Normalization] = {}
        self.string_lookup_layers: Dict[str, tf.keras.layers.StringLookup] = {}
        self.integer_lookup_layers: Dict[str, tf.keras.layers.IntegerLookup] = {}
        self.binning_boundaries: Dict[str, tf.Tensor] = {}
        self.cross_feature_lookup: Optional[tf.keras.layers.StringLookup] = None

        # 初始化归一化层
        normalization_features = self.config.get("normalization_features", [])
        for col in normalization_features:
            self.normalization_layers[col] = tf.keras.layers.Normalization(
                axis=None, name=f"norm_{col}"
            )

        # 初始化类别特征查找层
        categorical_features = self.config.get("categorical_features", [])
        for col in categorical_features:
            # 假设类别特征是字符串类型，如果不是，需要调整
            self.string_lookup_layers[col] = tf.keras.layers.StringLookup(
                output_mode="int", name=f"lookup_{col}"
            )

        # 初始化交叉特征查找层 (如果启用)
        cross_feature_config = self.config.get("cross_feature_config", {})
        if cross_feature_config.get("enabled", False):
            # 交叉特征的词汇表需要在 adapt 阶段构建
            self.cross_feature_lookup = tf.keras.layers.StringLookup(
                output_mode="int", name="cross_feature_lookup"
            )

    def adapt(self, dataset: tf.data.Dataset) -> None:
        """
        根据输入数据集拟合预处理层，计算统计信息和训练编码器/归一化器。

        Args:
            dataset: 用于拟合的 tf.data.Dataset。
        """
        print("开始拟合 Keras 预处理层...")

        # 拟合归一化层
        normalization_features = self.config.get("normalization_features", [])
        for col in normalization_features:
            if col in self.normalization_layers:
                # 收集该列的所有值
                feature_data = dataset.map(lambda x: x[col])
                self.normalization_layers[col].adapt(feature_data)
            else:
                print(f"警告: 列 '{col}' 没有对应的归一化层，跳过 adapt。")

        # 拟合类别特征查找层
        categorical_features = self.config.get("categorical_features", [])
        for col in categorical_features:
            if col in self.string_lookup_layers:
                feature_data = dataset.map(lambda x: x[col])
                self.string_lookup_layers[col].adapt(feature_data)
            else:
                print(f"警告: 列 '{col}' 没有对应的字符串查找层，跳过 adapt。")

        # 拟合数值特征分桶的边界 (这里简化为基于数据集的 min/max 或分位数)
        binning_config = self.config.get("binning_config", {})
        if binning_config.get("enabled", False):
            for col in binning_config.get("features", []):
                num_bins = binning_config.get("num_bins", 10)
                strategy = binning_config.get("strategy", "quantile")

                feature_data = dataset.map(lambda x: x[col])
                if strategy == "quantile":
                    # TensorFlow 没有直接的 qcut，需要手动计算分位数
                    # 这是一个简化的实现，实际可能需要更复杂的逻辑来精确匹配 Pandas 的 qcut
                    # 这里使用 tf.raw_ops.Quantile 可能会更准确，但需要处理输入张量
                    # 暂时使用 min/max 均匀分桶作为替代，或者需要收集所有数据计算分位数
                    min_val = tf.reduce_min(feature_data)
                    max_val = tf.reduce_max(feature_data)
                    boundaries = tf.linspace(min_val, max_val, num_bins + 1)[1:-1]
                    self.binning_boundaries[col] = boundaries
                elif strategy == "width":
                    min_val = tf.reduce_min(feature_data)
                    max_val = tf.reduce_max(feature_data)
                    boundaries = tf.linspace(min_val, max_val, num_bins + 1)[1:-1]
                    self.binning_boundaries[col] = boundaries
                else:
                    raise ValueError("分桶策略必须是 'quantile' 或 'width'。")

        # 拟合交叉特征查找层
        cross_feature_config = self.config.get("cross_feature_config", {})
        if cross_feature_config.get("enabled", False) and self.cross_feature_lookup:
            for cross_features in cross_feature_config.get("features", []):
                # 假设输入是字典形式的张量
                # 这是一个简化的方法，实际需要迭代数据集并构建所有交叉组合的字符串
                # 这里仅为示例，实际需要从 dataset 中提取并组合
                # 例如: dataset.map(lambda x: tf.strings.join([x[f] for f in cross_features], separator="_"))
                pass  # 实际的 adapt 逻辑会在这里

            # self.cross_feature_lookup.adapt(tf.constant(cross_feature_combinations))

        print("Keras 预处理层拟合完成。")

    def call(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        执行前向传播，转换输入张量。

        Args:
            inputs: 包含特征张量的字典。

        Returns:
            转换后的特征张量字典。
        """
        outputs = inputs.copy()
        print("开始转换 Keras 数据...")

        # 1. 处理缺失值 (这里假设数值特征用 0 填充，类别特征用 'UNKNOWN' 填充)
        missing_value_fill_strategy = self.config.get("missing_value_fill_strategy", {})
        for col, strategy in missing_value_fill_strategy.items():
            if col in outputs:
                if outputs[col].dtype.is_floating or outputs[col].dtype.is_integer:
                    if strategy == "zero":
                        outputs[col] = tf.where(
                            tf.math.is_nan(outputs[col]), 0.0, outputs[col]
                        )
                    # mean/median 填充需要在 adapt 阶段计算并保存
                    # 这里为了简化，只处理 'zero'
                elif outputs[col].dtype == tf.string:
                    if strategy == "zero":  # 对于字符串，'zero' 可以映射到特殊字符串
                        outputs[col] = tf.where(
                            tf.equal(outputs[col], ""),
                            tf.constant(""),
                            outputs[col],
                        )
                    elif strategy == "mode":
                        # mode 填充需要在 adapt 阶段计算并保存
                        pass

        # 2. 处理时间特征 (简化，仅提取 dayofweek 和 hour)
        time_features = self.config.get("time_features", [])
        for col in time_features:
            if col == "time_ms" and col in outputs:
                # 假设 time_ms 是 int64
                timestamp_s = outputs[col] // 1000
                # TensorFlow 没有直接的 datetime 对象，通常通过时间戳进行操作
                # 这里需要更复杂的逻辑来提取 dayofweek 和 hour
                # 例如：tf.timestamp() 或 tf.io.decode_csv 中的时间解析
                # 暂时跳过精确的时间特征提取，只做占位符
                outputs["dayofweek"] = tf.zeros_like(timestamp_s, dtype=tf.int32)
                outputs["hour"] = tf.zeros_like(timestamp_s, dtype=tf.int32)

        # 3. 处理布尔特征
        boolean_features = self.config.get("boolean_features", [])
        for col in boolean_features:
            if col in outputs:
                outputs[col] = tf.cast(outputs[col], tf.int32)

        # 4. 处理多值类别特征 (简化，仅处理 'tag')
        multivalued_features = self.config.get("multivalued_features", [])
        if "tag" in multivalued_features and "tag" in outputs:
            # 假设 tag 是字符串，例如 "1,2,3"
            # 需要将其转换为 RaggedTensor
            outputs["tag"] = tf.strings.to_number(
                tf.strings.split(outputs["tag"], ","), out_type=tf.int32
            ).to_tensor(default_value=0)  # 转换为 dense tensor，填充 0

        # 5. 对数值特征进行分桶
        binning_config = self.config.get("binning_config", {})
        if binning_config.get("enabled", False):
            for col in binning_config.get("features", []):
                if col in outputs and col in self.binning_boundaries:
                    # tf.histogram_fixed_width 适用于等宽分桶
                    # 对于等频分桶，需要更复杂的逻辑
                    outputs[f"{col}_bin"] = tf.histogram_fixed_width(
                        outputs[col],
                        self.binning_boundaries[col][0],  # min
                        self.binning_boundaries[col][-1],  # max
                        nbins=len(self.binning_boundaries[col]) + 1,
                    )
                elif col in outputs:
                    print(f"警告: 列 '{col}' 没有对应的分桶边界，跳过分桶。")

        # 6. 对数值特征进行归一化
        normalization_features = self.config.get("normalization_features", [])
        for col in normalization_features:
            if col in outputs and col in self.normalization_layers:
                outputs[col] = self.normalization_layers[col](outputs[col])
            elif col in outputs:
                print(f"警告: 列 '{col}' 没有对应的归一化层，跳过归一化。")

        # 7. 处理类别特征（标签编码）
        categorical_features = self.config.get("categorical_features", [])
        for col in categorical_features:
            if col in outputs and col in self.string_lookup_layers:
                outputs[col] = self.string_lookup_layers[col](outputs[col])
            elif col in outputs:
                print(f"警告: 列 '{col}' 没有对应的字符串查找层，跳过标签编码。")

        # 8. 创建交叉特征
        cross_feature_config = self.config.get("cross_feature_config", {})
        if cross_feature_config.get("enabled", False) and self.cross_feature_lookup:
            for cross_features in cross_feature_config.get("features", []):
                if all(f in outputs for f in cross_features):
                    cross_feature_name = "_".join(cross_features) + "_cross"
                    # 将多个特征拼接成一个字符串
                    combined_string = tf.strings.join(
                        [tf.cast(outputs[f], tf.string) for f in cross_features],
                        separator="_",
                    )
                    outputs[cross_feature_name] = self.cross_feature_lookup(
                        combined_string
                    )
                else:
                    print(
                        f"警告: 交叉特征 {cross_features} 中的某些列不存在，跳过交叉。"
                    )

        # 9. 清理不再需要的原始列 (Keras 层通常不直接删除输入张量，而是返回新的字典)
        # 这里为了保持与 Pandas 预处理器的一致性，可以从 outputs 中移除
        cols_to_drop = []
        for col in time_features:
            if col in outputs:
                cols_to_drop.append(col)
        # 假设 'datetime', 'timestamp_s', 'date', 'hourmin' 也是需要清理的原始时间相关列
        for col in ["datetime", "timestamp_s", "date", "hourmin"]:
            if col in outputs:
                cols_to_drop.append(col)

        for col in list(set(cols_to_drop)):
            if col in outputs:
                outputs.pop(col)

        print("Keras 数据转换完成。")
        return outputs

    def get_config(self) -> Dict[str, Any]:
        """
        返回层的配置，以便 Keras 可以序列化它。
        """
        config = super().get_config()
        config.update({"config": self.config})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """
        从配置字典创建层实例。
        """
        return cls(**config)
