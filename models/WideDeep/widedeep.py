import torch
import torch.nn as nn
from typing import List


class WideAndDeep(nn.Module):
    """
    宽深度学习模型(Wide & Deep Learning)

    该模型结合了线性模型的记忆能力和深度神经网络的泛化能力。

    属性:
        cat_features (Dict[str, int]): 分类特征及其取值数量的字典
        emb_dims (List[int]): 深度网络各层的嵌入维度
        encode_dim (int): 特征编码的维度
        deep_fea_nums (int): 深度网络的特征数量
        wide_fea_nums (int): 宽度部分的特征数量
        embs (nn.ModuleDict): 存储不同特征的嵌入层
        dlayers (nn.ModuleList): 深度网络层
        wlayer (nn.Linear): 宽度部分的线性层
        softmax (nn.Softmax): 用于输出概率的softmax层

    方法:
        encoder: 对输入特征进行编码
        deep_layer: 构建深度网络层
        wide_layer: 构建宽度部分的线性层
        forward: 模型的前向传播
    """

    def __init__(self, deep_input_dim: int, wide_input_dim: int, emb_dims: List[int]):
        super(WideAndDeep, self).__init__()
        self.emb_dims = emb_dims
        self.dlayers = self._deep_layer(deep_input_dim)
        self.wlayer = nn.Linear(wide_input_dim, emb_dims[-1])
        self.softmax = nn.Softmax(dim=1)

    def _deep_layer(self, input_dim: int) -> nn.Sequential:
        layers = [
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, self.emb_dims[0]),
            nn.PReLU(),
        ]
        for i in range(1, len(self.emb_dims)):
            layers.extend(
                [nn.Linear(self.emb_dims[i - 1], self.emb_dims[i]), nn.PReLU()]
            )
        return nn.Sequential(*layers)

    def forward(
        self, deep_features: torch.Tensor, wide_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播方法

        参数:
        deep_features: torch.Tensor, 形状为 (batch_size, deep_input_dim) 的深度特征张量
        wide_features: torch.Tensor, 形状为 (batch_size, wide_input_dim) 的宽度特征张量

        返回:
        torch.Tensor: 形状为 (batch_size, emb_dims[-1]) 的输出概率张量
        """
        y_deep = self.dlayers(deep_features)
        y_wide = self.wlayer(wide_features)
        final_embd = y_deep + y_wide
        return self.softmax(final_embd)


if __name__ == "__main__":
    # 测试用例
    deep_input_dim = 64
    wide_input_dim = 32
    emb_dims = [128, 64, 32]
    batch_size = 16

    # 初始化模型
    model = WideAndDeep(deep_input_dim, wide_input_dim, emb_dims)

    # 生成随机输入数据
    deep_features = torch.rand(batch_size, deep_input_dim)
    wide_features = torch.rand(batch_size, wide_input_dim)

    # 前向传播
    output = model(deep_features, wide_features)

    # 打印输出形状
    print(f"输出形状: {output.shape}")

    # 检查输出是否为概率分布
    assert torch.allclose(output.sum(dim=1), torch.ones(batch_size)), "输出不是有效的概率分布"

    print("测试通过!")

