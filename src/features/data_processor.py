import yaml
import pandas as pd
from typing import Any, List
from preprocess.processors import (
    MissingValueFiller,
    TimeFeatureExtractor,
    BooleanConverter,
    CategoricalEncoder,
    MultivaluedProcessor,
    LogTransformer,
    ColumnCleaner,
    HashTransformer,
)
from preprocess.config import PreprocessConfig


class DataProcessor:
    """
    数据预处理器，根据配置文件执行数据清洗和特征工程。
    """

    def __init__(self, config_path: str):
        """
        初始化数据处理器。

        Args:
            config_path: 配置文件目录路径。
        """
        base_config = self._load_config(f"{config_path}/base.yml")
        feature_config = self._load_config(f"{config_path}/feature.yml")
        na_config = self._load_config(f"{config_path}/na.yml")

        # 合并所有配置到一个字典中
        merged_config = {
            "dataset": base_config.get("dataset", {}),
            "NA_Processing": na_config.get("NA_Processing", {}),
            "feature_processing": feature_config.get("feature_processing", []),
        }
        self.preprocess_config: PreprocessConfig = PreprocessConfig(**merged_config)

        self.processors: List[Any] = []
        self._initialize_processors()

    def _load_config(self, file_path: str) -> dict:
        """
        加载 YAML 配置文件。

        Args:
            file_path: YAML 文件路径。

        Returns:
            配置字典。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _initialize_processors(self) -> None:
        """
        根据配置文件初始化各种预处理器。
        """
        # 1. 缺失值处理
        na_config = self.preprocess_config.NA_Processing
        if na_config.enabled:
            na_strategy = na_config.strategy
            fill_values_map = {f.feature_name: f.fill_value for f in na_config.features}

            # 构建 MissingValueFiller 的配置
            mvf_config = {"missing_value_fill_strategy": {}}
            for feature in self.preprocess_config.dataset.column_names:
                if feature in fill_values_map:
                    mvf_config["missing_value_fill_strategy"][feature] = (
                        fill_values_map[feature]
                    )
                else:
                    # 对于没有在 na.yml 中明确指定填充值的列，使用默认策略
                    mvf_config["missing_value_fill_strategy"][feature] = na_strategy

            self.processors.append(MissingValueFiller(mvf_config))

        # 2. 特征处理 (根据 feature.yml)
        # 收集不同类型的特征，以便传递给相应的处理器
        time_features_config = []
        boolean_features_config = []
        categorical_features_config = []
        multivalued_features_config = []
        log_features_config = []
        hash_features_config = []  # 用于 HashTransform

        for feature_def in self.preprocess_config.feature_processing:
            for process_step in feature_def.feature_process:
                function_name = process_step.function_name
                col_in = process_step.col_in
                col_out = process_step.col_out
                function_parameters = (
                    process_step.function_parameters.dict()
                    if process_step.function_parameters
                    else {}
                )

                if function_name == "TimeFeatureExtractTransform":
                    time_features_config.append(
                        {
                            "col_in": col_in,
                            "col_out": col_out,
                            "unit": function_parameters.get("unit"),
                            "format": function_parameters.get("format"),
                            "feature": function_parameters.get("feature"),
                        }
                    )
                elif function_name == "BooleanTransform":
                    boolean_features_config.append(col_in)
                elif function_name == "LabelEncodeTransform":
                    categorical_features_config.append(col_in)
                elif function_name == "MultiValueTransform":
                    multivalued_features_config.append(
                        {
                            "col_in": col_in,
                            "col_out": col_out,
                            "sep": function_parameters.get("sep"),
                            "hash_method": function_parameters.get("hash_method"),
                            "vocabulary_size": function_parameters.get(
                                "vocabulary_size"
                            ),
                            "max_len": function_parameters.get("max_len"),
                            "padding_value": function_parameters.get("padding_value"),
                        }
                    )
                elif function_name == "LogTransform":
                    log_features_config.append(col_in)
                elif function_name == "HashTransform":
                    hash_features_config.append(
                        {
                            "col_in": col_in,
                            "col_out": col_out,
                            "hash_method": function_parameters.get("hash_method"),
                            "vocabulary_size": function_parameters.get(
                                "vocabulary_size"
                            ),
                        }
                    )

        # 收集所有原始列名和预期生成的新列名
        original_cols = set(self.preprocess_config.dataset.column_names)
        generated_cols = set()

        for feature_def in self.preprocess_config.feature_processing:
            for process_step in feature_def.feature_process:
                col_out = process_step.col_out
                generated_cols.add(col_out)
                # 如果 col_in 和 col_out 不同，那么 col_in 可能是需要清理的原始列
                if process_step.col_in != col_out:
                    original_cols.add(process_step.col_in)  # 确保原始列被记录

        # 添加时间特征提取器
        if time_features_config:
            self.processors.append(
                TimeFeatureExtractor({"time_features_config": time_features_config})
            )

        # 添加布尔转换器
        if boolean_features_config:
            self.processors.append(
                BooleanConverter({"boolean_features": boolean_features_config})
            )

        # 添加对数转换器
        if log_features_config:
            self.processors.append(
                LogTransformer({"log_features": log_features_config})
            )

        # 添加类别编码器
        if categorical_features_config:
            self.processors.append(
                CategoricalEncoder(
                    {"categorical_features": categorical_features_config}
                )
            )

        # 添加多值处理器
        if multivalued_features_config:
            self.processors.append(
                MultivaluedProcessor(
                    {"multivalued_features_config": multivalued_features_config}
                )
            )

        # 添加哈希转换器
        if hash_features_config:
            self.processors.append(
                HashTransformer({"hash_features_config": hash_features_config})
            )

        # 3. 列清理 (在所有特征处理之后)
        self.processors.append(
            ColumnCleaner(
                {
                    "original_columns": list(original_cols),
                    "generated_columns": list(generated_cols),
                }
            )
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行数据预处理流程。

        Args:
            df: 原始 DataFrame。

        Returns:
            预处理后的 DataFrame。
        """
        processed_df = df.copy()
        for processor in self.processors:
            processed_df = processor.transform(processed_df)
        return processed_df

    def fit(self, df: pd.DataFrame) -> None:
        """
        拟合数据处理器中的所有可拟合组件。

        Args:
            df: 用于拟合的 DataFrame。
        """
        for processor in self.processors:
            processor.fit(df)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        拟合并转换数据。

        Args:
            df: 原始 DataFrame。

        Returns:
            预处理后的 DataFrame。
        """
        self.fit(df)
        return self.preprocess(df)
