import unittest

import torch

from .rqvae import RQVAE, ResidualQuantizer


class TestResidualQuantizer(unittest.TestCase):
    """测试 ResidualQuantizer 类。"""

    def test_forward(self):
        """测试 ResidualQuantizer 的前向传播。"""
        num_layers = 3
        codebook_size = 2048
        latent_dim = 64
        quantizer = ResidualQuantizer(num_layers, codebook_size, latent_dim)

        # 创建一个随机输入张量
        z = torch.randn(10, latent_dim)  # 批大小为 10

        codes, quantized, residuals = quantizer(z)

        # 验证输出类型和形状
        self.assertIsInstance(codes, list)
        self.assertEqual(len(codes), num_layers)
        for code in codes:
            self.assertIsInstance(code, torch.Tensor)
            self.assertEqual(code.shape, (10,))  # 批大小

        self.assertIsInstance(quantized, torch.Tensor)
        self.assertEqual(quantized.shape, z.shape)

        self.assertIsInstance(residuals, list)
        self.assertEqual(len(residuals), num_layers)
        for residual in residuals:
            self.assertIsInstance(residual, torch.Tensor)
            self.assertEqual(residual.shape, z.shape)

        # 验证量化结果的合理性
        # 在没有训练的情况下，量化后的向量不一定接近原始向量，因此移除此断言。
        # 更严格的测试需要检查量化误差，但这通常在训练后进行。


class TestRQVAE(unittest.TestCase):
    """测试 RQVAE 类。"""

    def test_forward_and_loss(self):
        """测试 RQVAE 的前向传播和损失计算。"""
        input_dim = 512
        latent_dim = 64
        num_layers = 3
        codebook_size = 2048
        rqvae = RQVAE(input_dim, latent_dim, num_layers, codebook_size)

        # 创建一个随机输入张量
        x = torch.randn(10, input_dim)  # 批大小为 10

        codes, x_recon, quantized, residuals = rqvae(x)

        # 验证前向传播输出
        self.assertIsInstance(codes, list)
        self.assertEqual(len(codes), num_layers)
        self.assertIsInstance(x_recon, torch.Tensor)
        self.assertEqual(x_recon.shape, x.shape)
        self.assertIsInstance(quantized, torch.Tensor)
        self.assertEqual(quantized.shape, (10, latent_dim))
        self.assertIsInstance(residuals, list)
        self.assertEqual(len(residuals), num_layers)

        # 测试损失计算
        loss = rqvae.loss(x, x_recon, residuals)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
