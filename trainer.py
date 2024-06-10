from typing import Callable, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import accuracy_score, roc_auc_score

from preprocess.dtypes import DataConfig, Pipeline


class Trainer(nn.Module):
    def __init__(
        self,
        cfg: DataConfig,
        base_model: nn.Module,
        emb_dict: nn.ModuleDict,
        epochs: int,
        lr: float = 1e-4,
        device: str = "cpu",
        log_interval: int = 10,
    ) -> None:
        super().__init__()
        self.epochs = epochs
        self.cfg = cfg
        self.loss_fn = nn.BCELoss()
        self.running_loss = 0

        self.device = device

        self.emb_dict = emb_dict
        self.base_model = base_model
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.log_interval = log_interval

    def _chunk_to_tensor(
        self, chunk: pd.DataFrame, pipe: Pipeline, reduce: str = "sum"
    ):
        x = chunk[pipe.col_in]
        x.fillna(pipe.fillna)
        for op in pipe.ops:
            # a list of operation
            x = x.apply(op)
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

        return tensor

    def _preprocess(
        self, x: Dict[str, torch.Tensor], tensor_op: Optional[Callable] = None
    ) -> torch.Tensor:
        if tensor_op is not None:
            x = tensor_op(x)
        else:
            x = torch.cat(list(x.values()), dim=1).float()

        return x

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self._preprocess(x)
        return self.base_model(x)

    def cal_metric(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = y_pred.detach().flatten().cpu().numpy()
        y_true = y_true.detach().flatten().cpu().numpy().astype(int)
        auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, (y_pred > 0.5).astype(int))
        return accuracy, auc

    def train_one_chunk(
        self,
        x: Dict[str, torch.Tensor],
        y: Dict[str, torch.Tensor],
        chunk_index: int,
        chunk_size: int,
        epoch: int,
    ):
        sample_count = chunk_index * chunk_size + chunk_size
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        # TODO train
        if isinstance(y_pred, torch.Tensor):
            assert len(y) == 1, "only support single label"
            loss = self.loss_fn(y_pred, y[list(y.keys())[0]])
            loss.backward()
            self.optimizer.step()
        elif isinstance(y_pred, list):
            assert len(y) == len(
                y_pred
            ), f"label num {len(y)} must equal to pred num {len(y_pred)}"
            total_loss = 0
            # TODO: support weighted loss sum
            for i in range(len(y_pred)):
                y_p = y_pred[i]
                y_t = y[list(y.keys())[i]].float()
                loss = self.loss_fn(y_p, y_t)
                total_loss += loss
                if (chunk_index + 1) % self.log_interval == 0:
                    acc, auc = self.cal_metric(y_p, y_t)
                    # .5f
                    logger.info(
                        f"[Train] Epoch: {epoch}, Chunk: {chunk_index},sample: {sample_count},\
Loss: {loss:.5f}, Acc: {acc:.5f}, AUC: {auc:.5f}"
                    )
            total_loss.backward()
            self.optimizer.step()
            # cal metrics

        else:
            raise NotImplementedError

        self.running_loss = loss

    def _load_data(self):
        root_dir = Path(self.cfg.train_dir)
        if root_dir.is_file():
            df = pd.read_csv(
                self.cfg.train_dir, sep=self.cfg.sep, chunksize=self.cfg.chunksize
            )
            return df
        elif root_dir.is_dir():
            # TODO: multi file
            pass
        else:
            raise NotImplementedError

    def train_one_epoch(self, epoch: int):
        df = pd.read_csv(
            self.cfg.train_dir, sep=self.cfg.sep, chunksize=self.cfg.chunksize
        )
        for chunk_idx, chunk in enumerate(df):
            # each chunk
            x_dict = dict()
            y_dict = dict()
            for feat_idx, pipe in enumerate(self.cfg.pipelines):
                # each column
                tensor = self._chunk_to_tensor(chunk, pipe)
                if pipe.source == "label":
                    y_dict[pipe.col_out] = tensor
                else:
                    x_dict[pipe.col_out] = tensor
            self.train_one_chunk(x_dict, y_dict, chunk_idx, chunk.shape[0], epoch)

    def train(self):
        for e in range(self.epochs):
            logger.info(f"epoch {e+1}")
            self.train_one_epoch(e)


if __name__ == "__main__":
    from pathlib import Path

    import yaml

    from models.MMOE.mmoe import SharedBottomModel
    from preprocess.dtypes import DataConfig
    from preprocess.emb import build_emb_dict, cal_feat_dim

    work_dir = Path.cwd()
    with open(work_dir / "config" / "ijcai18.yml", "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = DataConfig(**cfg)
    logger.info(f"data_cfg: {data_cfg}")

    emb_dim = 8
    dim_out = 64
    dim_hidden = [32, 32]
    dims = [dim_out, dim_out, 1]
    task_num = 1
    drop_out = 0.1
    device = "mps"
    emb_dict = build_emb_dict(data_cfg, emb_dim=emb_dim)
    feat_dims = cal_feat_dim(data_cfg, emb_dim=emb_dim)
    base_model = SharedBottomModel(
        dim_in=feat_dims,
        dim_out=dim_out,
        dim_hidden=dim_hidden,
        dims=dims,
        task_num=task_num,
        dropout=drop_out,
    )
    trainer = Trainer(data_cfg, base_model, emb_dict, epochs=10, device=device, lr=1e-4)
    trainer.to(device)
    trainer.train()
