import importlib
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Union, Tuple

DTYPES = Literal["int", "float"]
SOURCES = Literal["user", "item", "context", "label"]
FEATURE_TYPES = Literal["sparse", "dense", "varlen_sparse"]


@dataclass
class Pipeline:
    """
    数据处理管道的定义类。

    该类用于定义一个数据处理流程，包括输入列、输出列、缺失值填充方式、数据来源、特征类型、操作序列等属性。
    操作序列（ops）中的每个元素可以是一个操作函数或者带参数的操作函数字典。

    参数:
    col_in: 输入列的名称。
    col_out: 输出列的名称。
    fillna: 缺失值填充的值，可以是字符串、整数或浮点数。
    source: 数据来源的枚举值。
    feature_type: 特征类型的枚举值。
    ops: 数据处理操作的序列，每个操作可以是一个函数或带参数的函数字典。
    dtype: 输出列的数值类型，可选。
    tensor_type: 输出列的张量类型，可选。
    num_embeddings: 用于嵌入操作的向量维度，如果有的话。
    emb_dim: 嵌入层的维度，如果有的话。

    __post_init__ 方法用于在实例化后初始化 ops 属性，将操作函数字典转换为可调用的对象。
    """

    col_in: str
    col_out: str
    fillna: Union[str, int, float]
    source: SOURCES
    feature_type: FEATURE_TYPES
    ops: List[Union[Callable, Dict[Any, Any]]]
    dtype: Optional[DTYPES] = None
    tensor_type: Optional[DTYPES] = None
    num_embeddings: Optional[int] = None
    emb_dim: Optional[int] = None

    def __post_init__(self):
        """
        初始化 ops 属性，将操作函数字典转换为可调用对象。

        对 ops 中的每个操作进行处理，如果操作是一个字典，则尝试从字典中获取函数名和参数，
        并通过 partial 函数预设参数，将操作转换为可调用对象。
        """
        for i, op in enumerate(self.ops):
            # 遍历操作列表，对每个操作进行处理
            # funcs = list()
            for func_name, param_dict in op.items():
                # 导入操作模块，并根据函数名获取具体函数
                op_hub = importlib.import_module(".operation", package="preprocess")
                func = getattr(op_hub, func_name)
                # 根据是否提供参数，决定是否使用 partial 包装函数
                if len(param_dict) == 0:
                    func_with_param = func
                else:
                    func_with_param = partial(func, **param_dict)
                # 更新 ops 列表中的操作为可调用对象
            self.ops[i] = func_with_param


@dataclass
class DataSetConfig:
    """
    数据集配置类，用于存储数据集相关配置信息。

    属性:
        set_name (str): 数据集名称。
        file_type (Literal["csv", "excel"]): 文件类型，限定为 "csv" 或 "excel"。
        data_path (Union[str, List[str]]): 数据路径，可以是单个文件路径或文件路径列表。
        sep (str): 数据文件列分隔符，默认为 ","。
        chunksize (Optional[int]): 可选，用于指定读取大文件时的分块大小。
        names (Optional[List[str]]): 可选，用于指定数据文件的列名。
        label_columns (Optional[List[str]]): 可选，用于指定标签列名。
        join_type (Literal["concat", "merge"]): 可选，用于指定多个数据集的合并方式，默认为 None。
        join_on (Optional[List[str]]): 可选，用于指定合并数据集时的键列，默认为 None。
        how (Literal["inner", "left"]): 可选，用于指定合并数据集时的连接方法，默认为 None。
        join_names (Optional[List[str]]): 可选，用于指定另一个待合并数据集的名称，默认为 None。
    """

    set_name: str
    file_type: Literal["csv", "excel"]
    data_path: Union[str, List[str]]
    sep: str  # 数据文件分隔符，默认为 ","
    header: Optional[int] = 0
    chunksize: Optional[int] = None
    names: Optional[List[str]] = None  # 数据文件列名
    label_columns: Optional[List[str]] = None  # 标签列名
    join_type: Literal["concat", "merge"] = None
    join_on: Optional[List[str]] = None
    how: Literal["inner", "left"] = None
    join_names: Optional[List[str]] = None  # 另一个数据集的名称，用于合并


@dataclass
class Config:
    """
    配置类，用于存储训练、验证、测试集的配置信息以及特征管道的设置。

    Attributes:
        train_set: 训练集的配置信息。
        pipelines: 特征处理管道的列表。
        val_set: 验证集的配置信息，可选。
        test_set: 测试集的配置信息，可选。
        combo_set: 组合集的配置信息，可选。
        sparse_dim: 稀疏特征的维度，初始化为0。
        dense_dim: 密集特征的维度，初始化为0。
        total_dim: 总特征维度，由稀疏和密集特征维度组成，
                    初始化为0。
        defaul_emb_dim: 默认嵌入维度，用于未指定嵌入维度的特征
        emb_param_dict: 特征嵌入参数列表，键为特征名，
                值为嵌入维度和是否使用默认嵌入维度的元组
    """

    train_set: DataSetConfig
    pipelines: List[Pipeline]

    val_set: Optional[DataSetConfig] = None
    test_set: Optional[DataSetConfig] = None
    combo_set: Optional[DataSetConfig] = None

    sparse_dim: Optional[int] = 0
    dense_dim: Optional[int] = 0
    total_dim: Optional[int] = 0
    defaul_emb_dim: Optional[int] = 8
    emb_param_dict: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self):
        """
        在实例初始化后执行的特殊方法，用于进一步配置特征管道和计算特征维度。
        """
        self.train_set = DataSetConfig(**self.train_set)
        self.val_set = DataSetConfig(**self.val_set)
        self.test_set = DataSetConfig(**self.test_set)
        # 遍历特征处理管道列表，对每个管道进行实例化并根据特征类型更新稀疏和密集特征维度
        for i, pipe in enumerate(self.pipelines):
            cur_pipe = Pipeline(**pipe)  # 使用解包操作符**实例化Pipeline对象
            self.pipelines[i] = cur_pipe  # 更新列表中的元素为实例化的Pipeline对象
            if cur_pipe.feature_type.endswith("sparse") and cur_pipe.source != "label":
                if cur_pipe.emb_dim:
                    cur_emb_dim = cur_pipe.emb_dim
                elif self.defaul_emb_dim:
                    cur_emb_dim = self.defaul_emb_dim
                else:
                    cur_emb_dim = 0
                self.sparse_dim += (
                    cur_emb_dim  # 如果特征类型为稀疏，累加嵌入维度到稀疏特征维度
                )
                self.emb_param_dict[cur_pipe.col_out] = (
                    cur_pipe.num_embeddings,  # 获取特征名称对应的嵌入维度和嵌入向量维度
                    cur_emb_dim,
                )
            elif cur_pipe.feature_type.endswith("dense"):
                self.dense_dim += 1  # 如果特征类型为密集，累加1到密集特征维度
        # 计算总特征维度，即稀疏和密集特征维度之和
        self.total_dim = self.sparse_dim + self.dense_dim
