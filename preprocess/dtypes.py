import importlib
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Union

DTYPES = Literal["int", "float"]
SOURCES = Literal["user", "item", "context", "label"]
FEATURE_TYPES = Literal["sparse", "dense", "varlen_sparse"]


@dataclass
class Pipeline:
    col_in: str
    col_out: str
    fillna: Union[str, int, float]
    source: SOURCES
    feature_type: FEATURE_TYPES
    ops: List[Union[Callable, Dict[Any, Any]]]
    dtype: Optional[DTYPES] = None
    tensor_type: Optional[DTYPES] = None
    num_embeddings: Optional[int] = None

    def __post_init__(self):
        for i, op in enumerate(self.ops):
            # funcs = list()
            for func_name, param_dict in op.items():
                op_hub = importlib.import_module(".operation", package="preprocess")
                func = getattr(op_hub, func_name)
                if len(param_dict) == 0:
                    func_with_param = func
                else:
                    func_with_param = partial(func, **param_dict)
                # funcs.append(func)
            self.ops[i] = func_with_param


@dataclass
class DataConfig:
    train_dir: str
    val_dir: str
    data_columns: List[str]
    pipelines: List[Pipeline]
    label_columns: Optional[List[str]] = None
    chunksize: Optional[int] = None
    sep: Optional[str] = " "

    def __post_init__(self):
        for i, pipe in enumerate(self.pipelines):
            self.pipelines[i] = Pipeline(**pipe)


@dataclass
class DataSetConfig:
    set_name: str
    file_type: Literal["csv", "excel"]
    data_path: Union[str, List[str]]
    sep: str  # data file separator, default is ","
    chunksize: Optional[int] = None
    names: Optional[List[str]] = None  # data file column names
    label_columns: Optional[List[str]] = None  # label column names

    join_type: Literal["concat", "merge"] = None
    join_on: Optional[List[str]] = None
    how: Literal["inner", "left"] = None
    join_names: Optional[List[str]] = None  # aonther data set names


@dataclass
class Config:
    train_set: DataSetConfig
    pipelines: List[Pipeline]

    val_set: Optional[DataSetConfig] = None
    test_set: Optional[DataSetConfig] = None

    combo_set: Optional[DataSetConfig] = None

    def __post_init__(self):
        for i, pipe in enumerate(self.pipelines):
            self.pipelines[i] = Pipeline(**pipe)
