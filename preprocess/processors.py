from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from .operations import (
    get_hour,
    get_minute,
    get_month,
    int_to_date,
    isoweekday,
    list_hash,
    log1p,
    padding,
    str_hash,
    str_to_date,
    str_to_list,
)


class BasePandasProcessor(ABC):
    """
    Pandas 预处理器的抽象基类。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化基类。

        Args:
            config: 预处理配置字典。
        """
        self.config = config

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """
        根据输入数据拟合预处理器，计算统计信息和训练编码器/归一化器。

        Args:
            df: 用于拟合的 DataFrame。
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据拟合的参数转换输入数据。

        Args:
            df: 需要转换的 DataFrame。

        Returns:
            转换后的 DataFrame。
        """
        pass


class MissingValueFiller(BasePandasProcessor):
    """
    处理 Pandas DataFrame 中的缺失值。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.fill_values: Dict[str, Union[int, float, str]] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """
        根据数据拟合缺失值填充策略。
        """
        missing_value_fill_strategy = self.config.get("missing_value_fill_strategy", {})
        for col, strategy in missing_value_fill_strategy.items():
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if strategy == "mean":
                        self.fill_values[col] = df[col].mean()
                    elif strategy == "median":
                        self.fill_values[col] = df[col].median()
                    elif strategy == "zero":
                        self.fill_values[col] = 0
                elif isinstance(
                    df[col].dtype, pd.CategoricalDtype
                ) or pd.api.types.is_object_dtype(df[col]):
                    if strategy == "mode":
                        self.fill_values[col] = df[col].mode()[0]
                    elif strategy == "zero":
                        self.fill_values[col] = 0  # 对于类别特征，0可能表示一个特殊类别
                elif strategy == "none":
                    pass  # 不处理

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据拟合的填充值处理缺失值。
        """
        df_transformed = df.copy()
        missing_value_fill_strategy = self.config.get("missing_value_fill_strategy", {})
        for col, strategy in missing_value_fill_strategy.items():
            if col in df_transformed.columns and col in self.fill_values:
                df_transformed[col] = df_transformed[col].fillna(self.fill_values[col])
        return df_transformed


class TimeFeatureExtractor(BasePandasProcessor):
    """
    从时间戳中提取时间特征（如周几、小时）。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 TimeFeatureExtractor。

        Args:
            config: 预处理配置字典，包含 'time_features_config' 列表。
                    每个字典包含 'col_in', 'col_out', 'unit', 'format', 'feature'。
        """
        super().__init__(config)
        self.time_features_config = self.config.get("time_features_config", [])

    def fit(self, df: pd.DataFrame) -> None:
        """
        时间特征提取不需要拟合。

        Args:
            df: 用于拟合的 DataFrame。
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取时间特征。

        Args:
            df: 需要转换的 DataFrame。

        Returns:
            转换后的 DataFrame。
        """
        df_transformed = df.copy()
        for feature_config in self.time_features_config:
            col_in = feature_config["col_in"]
            col_out = feature_config["col_out"]
            unit = feature_config["unit"]
            date_format = feature_config.get("format")
            feature_to_extract = feature_config["feature"]

            if col_in not in df_transformed.columns:
                print(f"警告: 列 '{col_in}' 不存在，跳过时间特征提取。")
                continue

            temp_series = df_transformed[col_in].copy()

            if unit == "ms":
                temp_series = temp_series // 1000
                dt_series = temp_series.apply(
                    lambda x: int_to_date(x) if pd.notna(x) else pd.NaT
                )
            elif unit == "str":
                dt_series = temp_series.apply(
                    lambda x: str_to_date(str(x), date_format)
                    if pd.notna(x)
                    else pd.NaT
                )
            elif unit == "int_to_hour":
                # hourmin 是一个整数，例如 1030 表示 10:30
                dt_series = temp_series.apply(
                    lambda x: datetime(1970, 1, 1, x // 100, x % 100)
                    if pd.notna(x)
                    else pd.NaT
                )
            elif unit == "int_to_minute":
                dt_series = temp_series.apply(
                    lambda x: datetime(1970, 1, 1, x // 100, x % 100)
                    if pd.notna(x)
                    else pd.NaT
                )
            else:
                print(f"警告: 不支持的时间单位 '{unit}'。")
                continue

            if feature_to_extract == "dayofweek":
                df_transformed[col_out] = dt_series.apply(
                    lambda x: isoweekday(x) if pd.notna(x) else -1
                )
            elif feature_to_extract == "hour":
                df_transformed[col_out] = dt_series.apply(
                    lambda x: get_hour(x) if pd.notna(x) else -1
                )
            elif feature_to_extract == "minute":
                df_transformed[col_out] = dt_series.apply(
                    lambda x: get_minute(x) if pd.notna(x) else -1
                )
            elif feature_to_extract == "month":
                df_transformed[col_out] = dt_series.apply(
                    lambda x: get_month(x) if pd.notna(x) else -1
                )
            else:
                print(f"警告: 不支持的时间特征 '{feature_to_extract}'。")
                continue

            # 填充缺失值，例如用 -1
            df_transformed[col_out] = df_transformed[col_out].fillna(-1).astype(int)

        return df_transformed


class BooleanConverter(BasePandasProcessor):
    """
    将布尔特征转换为整数（0或1）。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.boolean_features = self.config.get("boolean_features", [])

    def fit(self, df: pd.DataFrame) -> None:
        """
        布尔特征转换不需要拟合。
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换布尔特征。
        """
        df_transformed = df.copy()
        for col in self.boolean_features:
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].astype(int)
        return df_transformed


class CategoricalEncoder(BasePandasProcessor):
    """
    对类别特征进行标签编码。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.categorical_features = self.config.get("categorical_features", [])
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """
        拟合 LabelEncoder。
        """
        for col in self.categorical_features:
            if col in df.columns:
                if isinstance(
                    df[col].dtype, pd.CategoricalDtype
                ) or pd.api.types.is_object_dtype(df[col]):
                    unique_values = df[col].dropna().unique().tolist()
                    le = LabelEncoder()
                    le.fit(unique_values)
                    self.label_encoders[col] = le
                else:
                    print(
                        f"警告: 列 '{col}' 在配置中被列为类别特征，"
                        f"但其数据类型{df[col].dtype}不适合标签编码。"
                    )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换类别特征。
        """
        df_transformed = df.copy()
        missing_value_fill_strategy = self.config.get("missing_value_fill_strategy", {})
        for col in self.categorical_features:
            if col in df_transformed.columns and col in self.label_encoders:
                # 使用 LabelEncoder 转换，处理未知类别和 NaN
                # 对于 LabelEncoder 未见过的类别，映射为 -1
                df_transformed[col] = df_transformed[col].map(
                    lambda s: self.label_encoders[col].transform([s])[0]
                    if s in self.label_encoders[col].classes_
                    else -1
                )
                # 对于标签编码后可能出现的-1（表示未知类别或NaN），根据配置进行处理
                if (df_transformed[col] == -1).any():
                    fill_strategy = missing_value_fill_strategy.get(col, "zero")
                    if fill_strategy == "zero":
                        df_transformed[col] = df_transformed[col].replace(-1, 0)
            elif col in df_transformed.columns:
                print(f"警告: 列 '{col}' 没有对应的标签编码器，跳过标签编码。")
        return df_transformed


class MultivaluedProcessor(BasePandasProcessor):
    """
    处理多值类别特征（例如将逗号分隔的字符串转换为整数列表）。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 MultivaluedProcessor。

        Args:
            config: 预处理配置字典，包含 'multivalued_features_config' 列表。
                    每个字典包含 'col_in', 'col_out', 'sep', 'hash_method',
                    'vocabulary_size', 'max_len', 'padding_value'。
        """
        super().__init__(config)
        self.multivalued_features_config = self.config.get(
            "multivalued_features_config", []
        )

    def fit(self, df: pd.DataFrame) -> None:
        """
        多值特征处理不需要拟合。

        Args:
            df: 用于拟合的 DataFrame。
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换多值特征。

        Args:
            df: 需要转换的 DataFrame。

        Returns:
            转换后的 DataFrame。
        """
        df_transformed = df.copy()
        for feature_config in self.multivalued_features_config:
            col_in = feature_config["col_in"]
            col_out = feature_config["col_out"]
            sep = feature_config["sep"]
            vocabulary_size = feature_config["vocabulary_size"]
            max_len = feature_config["max_len"]
            padding_value = feature_config["padding_value"]

            if col_in not in df_transformed.columns:
                print(f"警告: 列 '{col_in}' 不存在，跳过多值特征处理。")
                continue

            # 将字符串按分隔符转换为列表
            # 对于 NaN 值，转换为空列表
            list_series = df_transformed[col_in].apply(
                lambda x: str_to_list(str(x), sep) if pd.notna(x) else []
            )

            # 对列表中的每个值进行哈希
            hashed_list_series = list_series.apply(
                lambda x: list_hash(x, vocabulary_size)
            )

            # 对列表进行填充
            df_transformed[col_out] = hashed_list_series.apply(
                lambda x: padding(x, max_len, padding_value)
            )
        return df_transformed


class NumericalBinner(BasePandasProcessor):
    """
    对数值特征进行分桶。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.binning_config = self.config.get("binning_config", {})

    def fit(self, df: pd.DataFrame) -> None:
        """
        分桶不需要拟合，因为分桶策略（等频或等宽）是固定的。
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对数值特征进行分桶。
        """
        df_transformed = df.copy()
        if self.binning_config.get("enabled", False):
            for col in self.binning_config.get("features", []):
                if (
                    col not in df_transformed.columns
                    or df_transformed[col].isnull().all()
                ):
                    print(f"警告: 列 '{col}' 不存在或全为 NaN，跳过分桶。")
                    continue

                temp_series = df_transformed[col].fillna(df_transformed[col].median())
                num_bins = self.binning_config.get("num_bins", 10)
                strategy = self.binning_config.get("strategy", "quantile")

                if strategy == "quantile":
                    try:
                        df_transformed[f"{col}_bin"] = pd.qcut(
                            temp_series,
                            q=num_bins,
                            labels=False,
                            duplicates="drop",
                            precision=0,
                        )
                    except ValueError as e:
                        print(
                            f"警告: 对列 '{col}' 进行等频分桶时出错: {e}。尝试使用等宽分桶。"
                        )
                        df_transformed[f"{col}_bin"] = pd.cut(
                            temp_series, bins=num_bins, labels=False, precision=0
                        )
                elif strategy == "width":
                    df_transformed[f"{col}_bin"] = pd.cut(
                        temp_series, bins=num_bins, labels=False, precision=0
                    )
                else:
                    raise ValueError("分桶策略必须是 'quantile' 或 'width'。")

                df_transformed[f"{col}_bin"] = (
                    df_transformed[f"{col}_bin"].fillna(-1).astype(int)
                )
        return df_transformed


class NumericalNormalizer(BasePandasProcessor):
    """
    对数值特征进行归一化（MinMaxScaler）。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.normalization_features = self.config.get("normalization_features", [])
        self.scalers: Dict[str, MinMaxScaler] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """
        拟合 MinMaxScaler。
        """
        for col in self.normalization_features:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if not df[col].isnull().all():
                    scaler = MinMaxScaler()
                    temp_series = df[col].fillna(df[col].median())
                    scaler.fit(temp_series.values.reshape(-1, 1))
                    self.scalers[col] = scaler
                else:
                    print(f"警告: 数值列 '{col}' 全为 NaN，跳过归一化器拟合。")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换数值特征。
        """
        df_transformed = df.copy()
        for col in self.normalization_features:
            if col in df_transformed.columns and pd.api.types.is_numeric_dtype(
                df_transformed[col]
            ):
                if col in self.scalers:
                    temp_series = df_transformed[col].fillna(
                        df_transformed[col].median()
                    )
                    df_transformed[col] = self.scalers[col].transform(
                        temp_series.values.reshape(-1, 1)
                    )
                else:
                    print(f"警告: 列 '{col}' 没有对应的归一化器，跳过归一化。")
            elif col in df_transformed.columns:
                print(
                    f"警告: 列 '{col}' 在配置中被列为数值特征，但其数据类型不是数值类型。"
                )
        return df_transformed


class CrossFeatureCreator(BasePandasProcessor):
    """
    创建特征交叉。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cross_feature_config = self.config.get("cross_feature_config", {})
        self.cross_feature_encoders: Dict[str, LabelEncoder] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """
        拟合交叉特征的 LabelEncoder。
        """
        if self.cross_feature_config.get("enabled", False):
            for cross_features in self.cross_feature_config.get("features", []):
                if all(f in df.columns for f in cross_features):
                    cross_feature_name = "_".join(cross_features) + "_cross"
                    combined_series = df[cross_features[0]].astype(str)
                    for i in range(1, len(cross_features)):
                        combined_series += "_" + df[cross_features[i]].astype(str)

                    le = LabelEncoder()
                    le.fit(combined_series.unique())
                    self.cross_feature_encoders[cross_feature_name] = le
                else:
                    print(
                        f"警告: 交叉特征 {cross_features} 中的某些列不存在，跳过拟合。"
                    )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建交叉特征。
        """
        df_transformed = df.copy()
        if self.cross_feature_config.get("enabled", False):
            for cross_features in self.cross_feature_config.get("features", []):
                if all(f in df_transformed.columns for f in cross_features):
                    cross_feature_name = "_".join(cross_features) + "_cross"
                    combined_string = df_transformed[cross_features[0]].astype(str)
                    for i in range(1, len(cross_features)):
                        combined_string += "_" + df_transformed[
                            cross_features[i]
                        ].astype(str)

                    if cross_feature_name in self.cross_feature_encoders:
                        le = self.cross_feature_encoders[cross_feature_name]
                        # 处理未知交叉特征组合
                        df_transformed[cross_feature_name] = combined_string.map(
                            lambda s: le.transform([s])[0] if s in le.classes_ else -1
                        )
                        # 对于未知组合，可以根据需要填充为 0 或其他值
                        if (df_transformed[cross_feature_name] == -1).any():
                            df_transformed[cross_feature_name] = df_transformed[
                                cross_feature_name
                            ].replace(-1, 0)
                    else:
                        print(
                            f"警告: 交叉特征 '{cross_feature_name}' 没有对应的编码器，跳过转换。"
                        )
                else:
                    print(
                        f"警告: 交叉特征 {cross_features} 中的某些列不存在，跳过交叉。"
                    )
        return df_transformed


class LogTransformer(BasePandasProcessor):
    """
    对数值特征进行对数变换（log1p）。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 LogTransformer。

        Args:
            config: 预处理配置字典。
        """
        super().__init__(config)
        self.log_features = self.config.get("log_features", [])

    def fit(self, df: pd.DataFrame) -> None:
        """
        对数变换不需要拟合。

        Args:
            df: 用于拟合的 DataFrame。
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对数值特征进行对数变换。

        Args:
            df: 需要转换的 DataFrame。

        Returns:
            转换后的 DataFrame。
        """
        df_transformed = df.copy()
        for col in self.log_features:
            if col in df_transformed.columns and pd.api.types.is_numeric_dtype(
                df_transformed[col]
            ):
                # 确保所有值都大于等于 0，因为 log1p(x) 要求 x >= -1
                # 并且通常我们对非负数进行对数变换
                df_transformed[col] = df_transformed[col].apply(
                    lambda x: log1p(x) if x >= 0 else np.nan
                )
                # 填充可能由负数或 NaN 产生的 NaN 值，例如用 0 填充
                df_transformed[col] = df_transformed[col].fillna(0)
            else:
                print(
                    f"警告: 列 '{col}' 在配置中被列为对数变换特征，但其数据类型不是数值类型。"
                )
        return df_transformed


class ColumnCleaner(BasePandasProcessor):
    """
    删除不再需要的原始列。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 ColumnCleaner。

        Args:
            config: 预处理配置字典，包含 'original_columns' 和 'generated_columns'。
        """
        super().__init__(config)
        self.original_columns = set(self.config.get("original_columns", []))
        self.generated_columns = set(self.config.get("generated_columns", []))

    def fit(self, df: pd.DataFrame) -> None:
        """
        列清理不需要拟合。

        Args:
            df: 用于拟合的 DataFrame。
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        删除不再需要的原始列。

        Args:
            df: 需要转换的 DataFrame。

        Returns:
            转换后的 DataFrame。
        """
        df_transformed = df.copy()

        # 确定需要删除的列
        # 如果原始列经过转换后，其名称没有改变，则不删除
        # 如果原始列被转换成新的列名，则删除原始列
        cols_to_drop = []
        for col in self.original_columns:
            if col not in self.generated_columns and col in df_transformed.columns:
                cols_to_drop.append(col)

        # 额外删除 TimeFeatureExtractor 可能生成的中间列
        if "datetime" in df_transformed.columns:
            cols_to_drop.append("datetime")
        if "timestamp_s" in df_transformed.columns:
            cols_to_drop.append("timestamp_s")

        df_transformed = df_transformed.drop(
            columns=list(set(cols_to_drop)), errors="ignore"
        )
        return df_transformed


class HashTransformer(BasePandasProcessor):
    """
    对特征进行哈希转换。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 HashTransformer。

        Args:
            config: 预处理配置字典，包含 'hash_features_config' 列表。
                    每个字典包含 'col_in', 'col_out', 'hash_method', 'vocabulary_size'。
        """
        super().__init__(config)
        self.hash_features_config = self.config.get("hash_features_config", [])

    def fit(self, df: pd.DataFrame) -> None:
        """
        哈希转换不需要拟合。

        Args:
            df: 用于拟合的 DataFrame。
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对指定列进行哈希转换。

        Args:
            df: 需要转换的 DataFrame。

        Returns:
            转换后的 DataFrame。
        """
        df_transformed = df.copy()
        for feature_config in self.hash_features_config:
            col_in = feature_config["col_in"]
            col_out = feature_config["col_out"]
            vocabulary_size = feature_config["vocabulary_size"]
            # hash_method 可以根据需要使用，但 str_hash 默认使用 xxh32
            # 这里简化处理，直接使用 str_hash
            if col_in in df_transformed.columns:
                df_transformed[col_out] = df_transformed[col_in].apply(
                    lambda x: str_hash(str(x), vocabulary_size)
                )
            else:
                print(f"警告: 列 '{col_in}' 不存在，跳过哈希转换。")
        return df_transformed
