"""
数据预处理的配置类。
"""

from typing import List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """
    数据集配置模型。
    """

    name: str
    version: str
    description: str
    data_path: str
    column_names: List[str]
    label_column_names: List[str]


class NAProcessingFeature(BaseModel):
    """
    NA 处理特征配置模型。
    """

    feature_name: str
    feature_type: Literal["sparse", "dense", "varlen_sparse"]
    feature_side: Literal["user", "item", "context"]
    data_type: str
    fill_value: Optional[Union[int, float, str]] = None


class NAProcessingConfig(BaseModel):
    """
    NA 处理配置模型。
    """

    enabled: bool
    strategy: str
    default_fill_value: str
    features: List[NAProcessingFeature]


class FunctionParameters(BaseModel):
    """
    函数参数配置模型。
    """

    hash_method: Optional[str] = None
    vocabulary_size: Optional[int] = None


class FeatureProcessStep(BaseModel):
    """
    特征处理步骤配置模型。
    """

    col_in: str
    col_out: str
    function_name: str
    function_parameters: Optional[FunctionParameters] = None
    out_data_type: str
    out_data_shape: List[int]


class FeatureProcessingConfig(BaseModel):
    """
    特征处理配置模型。
    """

    feature_name: str
    feature_type: str
    feature_side: Literal["user", "item", "context"]
    data_type: str
    feature_process: List[FeatureProcessStep]


class PreprocessConfig(BaseModel):
    """
    预处理主配置模型。
    """

    dataset: DatasetConfig
    NA_Processing: NAProcessingConfig = Field(..., alias="NA_Processing")
    feature_processing: List[FeatureProcessingConfig]


def load_preprocess_config(config_path: str) -> PreprocessConfig:
    """
    从 YAML 文件加载预处理配置。

    Args:
        config_path: YAML 配置文件的路径。

    Returns:
        PreprocessConfig 实例。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return PreprocessConfig(**config_data)
