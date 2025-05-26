import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):
    """残差量化器。

    Args:
        num_layers (int): 量化器的层数。
        codebook_size (int): 每个码本的大小。
        latent_dim (int): 潜在空间的维度。
    """

    def __init__(
        self, num_layers: int = 3, codebook_size: int = 2048, latent_dim: int = 64
    ):
        super().__init__()
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, latent_dim) for _ in range(num_layers)]
        )

    def forward(self, z: torch.Tensor):
        """前向传播。

        Args:
            z (torch.Tensor): 输入的潜在向量。

        Returns:
            tuple: 包含以下元素的元组：
                - codes (list[torch.Tensor]): 每层量化后的码本索引。
                - quantized (torch.Tensor): 最终量化后的向量。
                - residuals (list[torch.Tensor]): 每层量化后的残差。
        """
        residuals = []
        codes = []
        # 确保 quantized 与 z 在同一设备和数据类型
        quantized = torch.zeros_like(z, device=z.device)

        for i in range(self.num_layers):
            # 停止梯度，避免梯度回传到之前的量化结果
            residual = z - quantized.detach()
            residuals.append(residual)

            # 找到最近的码本向量
            # self.codebooks[i].weight 是 nn.Embedding 的权重，类型为 torch.Tensor
            # 显式将权重转换为 Tensor 类型，解决 Pylance 警告
            distances = torch.cdist(residual, torch.as_tensor(self.codebooks[i].weight))
            code = torch.argmin(distances, dim=-1)
            codes.append(code)

            # 量化并累加
            quantized = self.codebooks[i](code) + quantized.detach()

        return codes, quantized, residuals


class RQVAE(nn.Module):
    """残差量化变分自编码器 (RQ-VAE)。

    Args:
        input_dim (int): 输入数据的维度。
        latent_dim (int): 潜在空间的维度。
        num_layers (int): 残差量化器的层数。
        codebook_size (int): 残差量化器中每个码本的大小。
    """

    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 64,
        num_layers: int = 3,
        codebook_size: int = 2048,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, latent_dim)
        )

        self.quantizer = ResidualQuantizer(num_layers, codebook_size, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, input_dim)
        )

    def forward(self, x: torch.Tensor):
        """前向传播。

        Args:
            x (torch.Tensor): 输入数据。

        Returns:
            tuple: 包含以下元素的元组：
                - codes (list[torch.Tensor]): 量化后的码本索引。
                - x_recon (torch.Tensor): 重构后的数据。
                - quantized (torch.Tensor): 量化后的潜在向量。
                - residuals (list[torch.Tensor]): 量化过程中的残差。
        """
        z = self.encoder(x)
        codes, quantized, residuals = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return codes, x_recon, quantized, residuals

    def loss(
        self, x: torch.Tensor, x_recon: torch.Tensor, residuals: list[torch.Tensor]
    ):
        """计算 RQ-VAE 的损失。

        Args:
            x (torch.Tensor): 原始输入数据。
            x_recon (torch.Tensor): 重构后的数据。
            residuals (list[torch.Tensor]): 量化过程中的残差。

        Returns:
            torch.Tensor: 总损失，包括重构损失和码本损失。
        """
        # 重构损失
        recon_loss = F.mse_loss(x_recon, x)

        # 码本损失
        # 鼓励残差趋近于零，使量化器更好地近似原始潜在向量
        codebook_loss = 0.0
        for residual in residuals:
            codebook_loss += torch.mean(residual**2)

        return recon_loss + 0.5 * codebook_loss


class PrefixNgramEmbedding(nn.Module):
    """前缀 N-gram 嵌入模块。

    用于从分层编码生成语义 ID 嵌入。

    Args:
        num_layers (int): 分层编码的层数。
        codebook_size (int): 每个码本的大小。
        embed_dim (int): 嵌入维度。
        hash_size (float): 哈希表的大小。
    """

    def __init__(
        self,
        num_layers: int = 3,
        codebook_size: int = 2048,
        embed_dim: int = 64,
        hash_size: float = 1e6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size  # 添加 codebook_size 属性
        self.hash_size = int(hash_size)

        # 初始化嵌入表
        self.embed_table = nn.Embedding(self.hash_size, embed_dim)

    def get_prefix_ngrams(self, codes: list[torch.Tensor]):
        """从分层编码生成前缀 N-gram。

        Args:
            codes (list[torch.Tensor]): 来自 RQ-VAE 的分层编码列表。

        Returns:
            list[torch.Tensor]: 生成的前缀 N-gram 列表。
        """
        prefixes = []
        for layer_index in range(1, self.num_layers + 1):
            # 通过连接前 l 个编码生成 N-gram
            # 注意：这里假设 codes 中的每个元素都是一个标量或单元素张量
            # 如果 codes 是批量的，需要调整计算方式
            prefix = torch.zeros_like(codes[0], dtype=torch.long)
            for i, c in enumerate(codes[:layer_index]):
                prefix += c * (self.codebook_size ** (layer_index - 1 - i))
            prefixes.append(prefix)
        return prefixes

    def forward(self, codes: list[torch.Tensor]):
        """前向传播。

        Args:
            codes (list[torch.Tensor]): 来自 RQ-VAE 的分层编码列表。

        Returns:
            torch.Tensor: 汇总后的嵌入向量。
        """
        # 生成所有前缀 N-gram
        prefixes = self.get_prefix_ngrams(codes)

        # 哈希到嵌入表索引
        indices = [p % self.hash_size for p in prefixes]

        # 查找嵌入
        embeddings = [self.embed_table(i) for i in indices]

        # 求和池化嵌入
        return sum(embeddings)
