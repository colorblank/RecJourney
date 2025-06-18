from collections.abc import Callable

import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import Tensor

from src.features.dtypes import Config, Pipeline


class BaseTrainer(nn.Module):
    """
    训练器的基本类，用于设置和执行模型的训练。

    参数:
    - cfg: 配置对象，包含训练相关的配置信息。
    - base_model: 基础模型，继承自nn.Module，是待训练的具体模型。
    - emb_dict: 字典形式的嵌入层模块，用于存储和管理不同类型的嵌入层。
    - epochs: 训练的轮数。
    - lr: 学习率，用于控制优化器的步长。
    - loss_fn: 损失函数，默认为二元交叉熵损失函数(nn.BCELoss())，用于计算模型预测与实际标签之间的差异。
    - optimizer: 优化器，默认为Adam优化器(torch.optim.Adam)，用于更新模型参数。
    - device: 训练设备，默认为cpu，也可以是cuda设备。
    - log_interval: 训练日志的打印间隔，默认为10，表示每训练10个iter打印一次日志。
    """

    def __init__(
        self,
        cfg: Config,
        base_model: nn.Module,
        emb_dict: nn.ModuleDict,
        epochs: int,
        lr: float,
        loss_fn: Callable = nn.BCELoss(),
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        device: str = "cpu",
        log_interval: int = 10,
    ) -> None:
        super().__init__()
        self.epochs = epochs  # 训练轮数
        self.cfg = cfg  # 配置对象
        self.loss_fn = loss_fn  # 损失函数
        self.running_loss = 0  # 运行时损失，用于记录训练过程中的损失

        self.device = device  # 训练设备

        self.emb_dict = emb_dict  # 嵌入层字典
        self.base_model = base_model.to(device)  # 基础模型
        self.optimizer = optimizer(
            list(self.parameters()), lr=lr
        )  # 优化器 # type: ignore

        self.log_interval = log_interval  # 日志打印间隔

        self.val_x_dict = dict()
        self.val_y_dict = dict()
        self._load_validation_data()

    def _load_validation_data(self) -> None:
        if self.cfg.val_set is None:
            logger.warning("验证集配置为空，跳过加载验证数据。")
            return

        data_path = self.cfg.val_set.data_path
        if isinstance(data_path, list):
            data_path = data_path[0]

        read_csv_params = {
            "filepath_or_buffer": data_path,
            "sep": self.cfg.val_set.sep,
            "header": self.cfg.val_set.header,
            "names": self.cfg.val_set.names,
        }
        if self.cfg.val_set.chunksize is not None:
            read_csv_params["chunksize"] = self.cfg.val_set.chunksize

        df = pd.read_csv(**read_csv_params)

        for feat_idx, pipe in enumerate(self.cfg.pipelines):
            # each column
            tensor = self._chunk_to_tensor(df, pipe)
            # self.val_tenosr_dict[pipe.col_out] = tensor
            if pipe.source == "label":
                self.val_y_dict[pipe.col_out] = tensor
            else:
                self.val_x_dict[pipe.col_out] = tensor

    def evaluation_one_predict(
        self, y_pred: Tensor, y_true: Tensor, epoch: int
    ) -> None:
        """
        用于评估模型在验证集上的预测结果。
        """
        acc, auc, logloss = self.cal_metric(y_pred, y_true)
        logger.info(
            f"[Val] Epoch: {epoch + 1}, Acc: {acc:.5f}, AUC: {auc:.5f}, Logloss: {logloss:.5f}."
        )

    def evaluation(
        self,
        epoch: int,
    ):
        with torch.no_grad():
            y_pred = self.forward(self.val_x_dict)
            y_true_names = sorted(list(self.val_y_dict.keys()))
            if isinstance(y_pred, list):
                for i in range(len(y_pred)):
                    y_t = self.val_y_dict[y_true_names[i]]
                    self.evaluation_one_predict(y_pred[i], y_t, epoch)
            elif isinstance(y_pred, Tensor):
                y_t = self.val_y_dict[y_true_names[0]]
                self.evaluation_one_predict(y_pred, y_t, epoch)

    def _chunk_to_tensor(
        self, chunk: pd.DataFrame, pipe: Pipeline, reduce: str = "sum"
    ):
        x = chunk[pipe.col_in]
        x = x.fillna(pipe.fillna)  # fillna should be reassigned
        for op in pipe.ops:
            if callable(op):  # 确保op是可调用的
                x = x.apply(op)
            else:
                # 如果op不是可调用的，这表示在dtypes.py中转换失败，或者ops中存在非预期的元素
                # 可以在这里添加日志或错误处理
                logger.error(f"操作 {op} 不可调用，跳过。")
        tensor = torch.tensor(x.to_list()).to(self.device)

        if pipe.feature_type.endswith("sparse") and pipe.source != "label":
            tensor = tensor.long()
            tensor = self.emb_dict[pipe.col_out](tensor)
            if len(tensor.shape) == 3:
                if reduce == "sum":
                    tensor = tensor.sum(dim=1)
                elif reduce == "mean":
                    tensor = tensor.mean(dim=1)
                else:
                    raise NotImplementedError
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(1)

        return tensor.to(self.device)

    def _preprocess(
        self, x: dict[str, Tensor], tensor_op: Callable | None = None
    ) -> Tensor:
        if tensor_op is not None:
            return tensor_op(x)
        return torch.cat(list(x.values()), dim=1).float()

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        processed_x = self._preprocess(x)
        return self.base_model(processed_x)

    def cal_metric(self, y_pred: Tensor, y_true: Tensor):
        y_pred_np = y_pred.detach().flatten().cpu().numpy()
        y_true_np = y_true.detach().flatten().cpu().numpy()
        auc = roc_auc_score(y_true_np.astype(int), y_pred_np)
        accuracy = accuracy_score(y_true_np.astype(int), (y_pred_np > 0.5).astype(int))
        logloss = log_loss(y_true_np.astype(int), y_pred_np)
        return accuracy, auc, logloss

    def _calculate_loss(
        self,
        y_pred: list[Tensor] | Tensor,
        y_true: dict[str, Tensor],
    ) -> Tensor:
        """
        根据预测值与真实标签计算并返回损失值。

        此函数支持两种输入格式：单个张量预测值及预测值的张量列表。
        若y_pred为单个张量，则依据y_true的第一个元素计算损失。
        若y_pred为张量列表，则对列表中的每一项分别计算损失后求和。

        参数:
        y_pred: 预测值，可以是PyTorch张量或张量列表。
        y_true: 真实值，格式与y_pred相同。

        返回:
        Tensor: 总损失值，以单个张量形式返回。

        异常:
        NotImplementedError: 若y_pred为不支持的类型。
        """
        """根据预测与真实标签计算并返回损失。"""
        # 判断y_pred是否为张量
        if isinstance(y_pred, Tensor):
            # 直接使用损失函数计算损失
            return self.loss_fn(y_pred, y_true[list(y_true.keys())[0]])
        # 判断y_pred是否为张量列表
        elif isinstance(y_pred, list):
            # 对y_pred中每一项计算损失后累加得到总损失
            loss = torch.tensor(0.0, device=self.device)  # 初始化为Tensor
            i = 0
            for _, y_t in y_true.items():
                loss = loss + self.loss_fn(y_pred[i], y_t)
                i += 1
            return loss
        else:
            # 抛出异常，表示y_pred类型不受支持
            raise NotImplementedError("不支持的y_pred类型。")

    def _log_metrics(
        self,
        epoch: int,  # 当前训练轮次
        chunk_index: int,  # 当前数据块的索引
        sample_count: int,  # 目前为止处理的样本总数
        loss: float,  # 当前数据块的损失值
        metrics: tuple[float, float, float]
        | None = None,  # 额外指标元组（准确率, AUC, 对数损失）, 默认为None
    ) -> None:
        """
        记录训练过程中的评估指标。

        此函数记录训练过程中的损失值以及额外的评估指标（如准确率、AUC、对数损失），
        用以监控训练进程。如果未提供具体指标，相关部分将默认不显示具体数值。

        参数:
            - epoch: 当前训练轮次。
            - chunk_index: 当前处理数据块的索引。
            - sample_count: 目前为止在当前轮次中处理的样本数量。
            - loss: 当前数据块计算出的损失值。
            - metrics: 一个包含额外评估指标（准确率、AUC、对数损失）的元组，默认为None。

        返回:
            - 无返回值。
        """
        # 若提供了metrics参数，则解包；否则，各指标默认设为None以便记录日志
        acc, auc, logloss = metrics if metrics else (None, None, None)
        # 打印训练信息，包括轮次、数据块索引、样本计数、损失值及各项评估指标（如有）
        logger.info(
            f"[训练] 轮次: {epoch + 1}, 数据块: {chunk_index}, 样本数: {sample_count}, 损失: {loss:.5f}, \
准确率: {acc:.5f}, AUC: {auc:.5f} 对数损失: {logloss:.5f}."
        )

    def train_one_chunk(
        self,
        x: dict[str, Tensor],
        y: dict[str, Tensor],
        chunk_index: int,
        chunk_size: int,
        epoch: int,
    ) -> None:
        """
        训练模型的一个数据块。

        参数:
        x: 输入数据，以字典形式存储，键为特征名称，值为Tensor。
        y: 目标数据，以字典形式存储，键为标签名称，值为Tensor。
        chunk_index: 当前数据块的索引。
        chunk_size: 数据块的大小。
        epoch: 当前训练的epoch数。

        返回:
        无
        """
        # 计算当前数据块的样本总数
        sample_count = chunk_index * chunk_size + chunk_size
        # 清零梯度
        self.optimizer.zero_grad()
        # 前向传播，获取预测结果
        y_pred = self.forward(x)
        # 计算损失
        loss = self._calculate_loss(y_pred, y)
        # 反向传播，计算梯度
        loss.backward()

        # 如果预测结果是列表且当前数据块是日志记录间隔的倍数，则计算并记录指标
        if (chunk_index + 1) % self.log_interval == 0:
            # 从预测结果和目标中提取需要计算指标的部分
            if isinstance(y_pred, list) and len(y_pred) == 1:
                y_pred = y_pred[0]
            acc, auc, logloss = self.cal_metric(y_pred, y[list(y.keys())[0]])
            # 记录当前数据块的指标
            self._log_metrics(
                epoch,
                chunk_index,
                sample_count,
                loss.item(),
                (float(acc), float(auc), float(logloss)),
            )

        # 更新参数
        self.optimizer.step()
        # 更新运行损失
        self.running_loss = loss.item()

    def train_one_epoch(self, epoch: int):
        """
        训练模型的一个epoch。

        参数:
        epoch: int, 当前训练的epoch数。
        """
        # 从CSV文件中读取训练数据，按指定的分隔符和块大小进行分块读取
        data_path = self.cfg.train_set.data_path
        if isinstance(data_path, list):
            data_path = data_path[0]  # 如果是列表，只取第一个路径

        read_csv_params = {
            "filepath_or_buffer": data_path,
            "sep": self.cfg.train_set.sep,
            "header": self.cfg.train_set.header,
            "names": self.cfg.train_set.names,
        }
        if self.cfg.train_set.chunksize is not None:
            read_csv_params["chunksize"] = self.cfg.train_set.chunksize

        df = pd.read_csv(**read_csv_params)
        # 遍历每个数据块进行训练
        for chunk_idx, chunk in enumerate(df):
            # 初始化特征和标签的字典
            x_dict = dict()
            y_dict = dict()
            # 遍历配置中的每个处理管道，对数据块进行处理
            for feat_idx, pipe in enumerate(self.cfg.pipelines):
                # 将数据块通过管道转换为张量
                tensor = self._chunk_to_tensor(chunk, pipe)
                # 根据管道类型（特征或标签），将张量添加到相应的字典中
                if pipe.source == "label":
                    y_dict[pipe.col_out] = tensor
                else:
                    x_dict[pipe.col_out] = tensor
            # 使用当前数据块进行模型训练
            self.train_one_chunk(x_dict, y_dict, chunk_idx, chunk.shape[0], epoch)

    def run_train(self):
        for e in range(self.epochs):
            logger.info(f"epoch {e + 1}")
            self.train_one_epoch(e)
            self.evaluation(e)


# if __name__ == "__main__":
#     from pathlib import Path

#     import yaml

#     from models.MMOE.mmoe import MMOE, HeadArgs, MMOEArgs, SharedBottomModel
#     from preprocess.dtypes import DataConfig
#     from src.features.emb import build_emb_dict, cal_feat_dim

#     work_dir = Path.cwd()
#     with open(work_dir / "config" / "ijcai18.yml", "r") as f:
#         cfg = yaml.safe_load(f)
#     data_cfg = DataConfig(**cfg)
#     logger.info(f"data_cfg: {data_cfg}")

#     emb_dim = 8
#     dim_out = 64
#     dim_hidden = [32, 32]
#     dims = [dim_out, dim_out, 1]
#     task_num = 1
#     drop_out = 0.1
#     device = "cuda"
#     emb_dict = build_emb_dict(data_cfg, emb_dim=emb_dim)
#     feat_dims = cal_feat_dim(data_cfg, emb_dim=emb_dim)
#     # mmoe_args = MMOEArgs(
#     #     dim_in=feat_dims,
#     #     dim_out=dim_out,
#     #     dim_hidden=dim_hidden,
#     #     expert_num=4,
#     #     task_num=1,
#     # )
#     # head_args = HeadArgs(dims=[dim_out, dim_out, 1])
#     # base_model = MMOE(mmoe_args, head_args)
#     base_model = SharedBottomModel(
#         dim_in=feat_dims,
#         dim_out=dim_out,
#         dim_hidden=dim_hidden,
#         dims=dims,
#         task_num=task_num,
#         dropout=drop_out,
#     )

#     trainer = Trainer(data_cfg, base_model, emb_dict, epochs=10, device=device, lr=1e-4)
#     trainer.to(device)
#     trainer.train()
