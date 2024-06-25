from typing import List

import torch
import torch.nn as nn


def second_order_interaction(x: torch.Tensor) -> torch.Tensor:
    """
    计算输入特征的二阶交叉效应。

    Args:
        x (torch.Tensor): 输入的特征嵌入，
        形状为(batch_size, num_fields, embedding_dim)。

    Returns:
        torch.Tensor: 二阶交叉项的和，形状为(batch_size, 1)
    """
    square_of_sum = torch.pow(x.sum(dim=1), 2)
    sum_of_square = x.pow(2).sum(dim=1)
    interaction = 0.5 * (square_of_sum - sum_of_square)

    interaction = torch.sum(interaction, dim=1, keepdim=True)

    return interaction


class FactorizationMachine(nn.Module):
    """
    Factorization Machine模型，用于处理稀疏特征。

    Args:
        num_fields (list(int)): 每个特征的词表大小。
        emb_dim (int): 嵌入向量的维度。
        use_bias (bool): 是否使用偏置项。
        unify_embedding (bool): 是否使用统一嵌入。
    """

    def __init__(
        self,
        num_fields: List[int],
        emb_dim: int,
        use_bias: bool = True,
        unify_embedding: bool = False,
    ):
        super(FactorizationMachine, self).__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.use_bias = use_bias
        self.unify_embedding = unify_embedding

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
            self.bias.requires_grad = True

        if unify_embedding:
            self.embedding_first = nn.Embedding(sum(num_fields), 1)
            self.embedding_second = nn.Embedding(sum(num_fields), emb_dim)

            self.offsets = torch.tensor(
                [0] + list(torch.cumsum(torch.tensor(num_fields), dim=0)[:-1])
            )
        else:
            self.embedding_first = nn.ModuleList(
                [nn.Embedding(num_field, 1) for num_field in num_fields]
            )
            self.embedding_second = nn.ModuleList(
                [nn.Embedding(num_field, emb_dim) for num_field in num_fields]
            )

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.LongTensor): size = (batch_size, num_fields)

        Returns:
            torch.Tensor: size = (batch_size, 1)
        """
        if self.use_bias:
            bias = self.bias.expand(x.size(0), 1)
        else:
            bias = 0

        if self.unify_embedding:
            # 根据输入的x，将每个特征映射到对应的偏移量上
            x = x + self.offsets.unsqueeze(0)
            x1 = self.embedding_first(x)
            x2 = self.embedding_second(x)
        else:
            x1 = torch.stack(
                [emb(x[:, i]) for i, emb in enumerate(self.embedding_first)], dim=1
            )
            x2 = torch.stack(
                [emb(x[:, i]) for i, emb in enumerate(self.embedding_second)], dim=1
            )

        x_first_order = torch.sum(x1, dim=1)
        x_second_order = second_order_interaction(x2)
        output = x_first_order + x_second_order + bias
        return output


if __name__ == "__main__":
    num_fields = [3, 4, 5]
    fm = FactorizationMachine(
        num_fields, emb_dim=16, use_bias=True, unify_embedding=True
    )
    x = torch.randint(0, 3, (2, 3))
    print(fm(x))
