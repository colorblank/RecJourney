import torch
import pytest

from models.pytorch.xDeepFM.xdeepfm import xDeepFM, xDeepFMArgs, CINArgs, DNNArgs


def test_xdeepfm_forward():
    """
    测试 xDeepFM 模型的前向传播。
    """
    num_fields = 10
    emb_dim = 8
    batch_size = 2

    # 创建模型参数
    args = xDeepFMArgs(
        num_fields=num_fields,
        emb_dim=emb_dim,
        cin_args=CINArgs(
            dim_in=num_fields,
            dim_hiddens=[32, 16],
            num_classes=1,
            split_half=True,
            bias=True,
        ),
        dnn_args=DNNArgs(
            dim_in=num_fields * emb_dim,
            dim_hiddens=[32, 16],
            bias=True,
            activation="relu",
            dropout=0.2,
        ),
    )

    # 创建模型实例
    model = xDeepFM(args)

    # 创建模拟输入数据 (batch_size, num_fields, emb_dim)
    x = torch.randn(batch_size, num_fields, emb_dim)

    # 执行前向传播
    output = model(x)

    # 检查输出的形状
    assert output.shape == (batch_size, 1), "xDeepFM模型输出形状不正确"

    # 检查输出是否为浮点类型
    assert output.dtype == torch.float32, "xDeepFM模型输出数据类型不正确"

    # 检查输出值是否在 [0, 1] 之间 (因为最后有 Sigmoid 激活)
    assert torch.all(output >= 0) and torch.all(output <= 1), "xDeepFM模型输出值不在[0, 1]之间"

    print(f"xDeepFM模型输出形状: {output.shape}")
    print(f"xDeepFM模型输出数据类型: {output.dtype}")


if __name__ == "__main__":
    pytest.main([__file__])
