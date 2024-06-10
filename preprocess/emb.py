import torch.nn as nn

from .dtypes import DataConfig


def build_emb_dict(cfg: DataConfig, emb_dim: int) -> nn.ModuleDict:
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
            )
    # 返回包含所有嵌入层的字典
    return embedding_dict


def cal_feat_dim(cfg: DataConfig, emb_dim: int) -> int:
    """
    计算特征维度的总和。

    根据配置中的处理管道（pipelines），累加稀疏特征的嵌入维度和稠密特征的维度。
    稀疏特征的维度通过给定的嵌入维度进行累加，稠密特征的维度固定为1。

    参数:
    cfg: DataConfig - 配置对象，包含处理管道的信息。
    emb_dim: int - 稀疏特征的嵌入维度。

    返回:
    int - 所有特征维度的总和。
    """
    # 初始化特征维度总和为0
    feat_dim = 0
    # 遍历配置中的每个处理管道
    for pipeline in cfg.pipelines:
        if pipeline.source == "label":
            continue
        # 获取当前管道的特征类型
        feature_type = pipeline.feature_type
        # 如果特征类型以"sparse"结尾，视为稀疏特征
        if feature_type.endswith("sparse"):
            # 稀疏特征的维度累加嵌入维度
            feat_dim += emb_dim
        else:
            # 否则，视为稠密特征，维度累加1
            feat_dim += 1
    # 返回计算得到的特征维度总和
    return feat_dim
