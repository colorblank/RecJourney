import torch.nn as nn

from .dtypes import Config


def build_emb_dict(cfg: Config, emb_dim: int, device: str = "cpu") -> nn.ModuleDict:
    """
    根据配置信息构建嵌入字典。

    该函数遍历配置文件中定义的管道（pipelines），为每个稀疏特征创建一个嵌入层。
    嵌入层用于将离散的特征值映射到连续的向量表示。

    参数:
    cfg: DataConfig类型，包含数据处理管道的配置信息。
    emb_dim: int类型，表示嵌入向量的维度。

    返回值:
    nn.ModuleDict类型，一个包含所有嵌入层的字典，键为特征输出列名，值为对应的嵌入层。
    """
    # 初始化一个模块字典，用于存储嵌入层
    embedding_dict = nn.ModuleDict()
    # 获取配置文件中定义的处理管道列表
    pipelines = cfg.pipelines
    # 遍历每个处理管道
    for pipeline in pipelines:
        # 获取当前管道处理的特征类型
        feature_type = pipeline.feature_type
        # 如果特征类型以"sparse"结尾，表示该特征是稀疏的，需要创建嵌入层
        if feature_type.endswith("sparse") and pipeline.source != "label":
            # 为该稀疏特征创建一个嵌入层，并添加到嵌入字典中
            embedding_dict[pipeline.col_out] = nn.Embedding(
                pipeline.num_embeddings, emb_dim
            ).to(device=device)
    # 返回包含所有嵌入层的字典
    return embedding_dict


def cal_feat_dim_in(cfg: Config, emb_dim: int):
    dim = 0
    for pipeline in cfg.pipelines:
        # 获取当前管道处理的特征类型
        feature_type = pipeline.feature_type
        # 如果特征类型以"sparse"结尾，表示该特征是稀疏的，需要创建嵌入层
        if feature_type.endswith("sparse") and pipeline.source != "label":
            dim += emb_dim
        elif feature_type == "dense":
            dim += 1
    return dim
