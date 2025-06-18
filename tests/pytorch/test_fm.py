import torch
import pytest

from models.pytorch.FM.fm import FactorizationMachine


def test_fm_forward():
    """
    测试 FactorizationMachine 模型的前向传播。
    """
    num_fields = [3, 4, 5]
    emb_dim = 16
    batch_size = 2

    # 创建模型实例
    fm_model = FactorizationMachine(
        num_fields=num_fields, emb_dim=emb_dim, use_bias=True, unify_embedding=True
    )

    # 创建模拟输入数据
    # 输入x的每一列对应一个特征字段的索引
    x = torch.randint(
        0, max(num_fields), (batch_size, len(num_fields)), dtype=torch.long
    )

    # 执行前向传播
    output = fm_model(x)

    # 检查输出的形状
    assert output.shape == (batch_size, 1), "FM模型输出形状不正确"

    # 检查输出是否为浮点类型
    assert output.dtype == torch.float32, "FM模型输出数据类型不正确"

    print(f"FM模型输出形状: {output.shape}")
    print(f"FM模型输出数据类型: {output.dtype}")


def test_second_order_interaction():
    """
    测试 second_order_interaction 函数。
    """
    from models.pytorch.FM.fm import second_order_interaction

    # 模拟输入数据 (batch_size, num_fields, embedding_dim)
    x = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]], dtype=torch.float32
    )  # batch_size=2, num_fields=2, embedding_dim=2

    # 预期计算:
    # batch 1:
    # x.sum(dim=1) = [4.0, 6.0]
    # square_of_sum = [16.0, 36.0]
    # x.pow(2) = [[[1.0, 4.0], [9.0, 16.0]], [[0.25, 2.25], [6.25, 12.25]]]
    # sum_of_square = [[5.0, 25.0], [2.5, 18.5]]
    # interaction = 0.5 * ([16.0, 36.0] - [5.0, 25.0]) = 0.5 * ([11.0, 11.0]) = [5.5, 5.5]
    # torch.sum(interaction, dim=1, keepdim=True) = [[11.0], [11.0]] (incorrect, should be sum over last dim)
    # Let's re-evaluate the formula: 0.5 * ((sum_i v_i x_i)^2 - sum_i (v_i x_i)^2)
    # Here x is already the embedding, so it's 0.5 * ((sum_f x_f)^2 - sum_f (x_f)^2)
    # For batch 1, field 1: [1.0, 2.0], field 2: [3.0, 4.0]
    # sum_f x_f = [1.0+3.0, 2.0+4.0] = [4.0, 6.0]
    # (sum_f x_f)^2 = [16.0, 36.0]
    # sum_f (x_f)^2 = [1.0^2+3.0^2, 2.0^2+4.0^2] = [1.0+9.0, 4.0+16.0] = [10.0, 20.0]
    # 0.5 * ([16.0, 36.0] - [10.0, 20.0]) = 0.5 * ([6.0, 16.0]) = [3.0, 8.0]
    # Sum over embedding_dim: 3.0 + 8.0 = 11.0
    # So for batch 1, the interaction should be 11.0

    # For batch 2, field 1: [0.5, 1.5], field 2: [2.5, 3.5]
    # sum_f x_f = [0.5+2.5, 1.5+3.5] = [3.0, 5.0]
    # (sum_f x_f)^2 = [9.0, 25.0]
    # sum_f (x_f)^2 = [0.5^2+2.5^2, 1.5^2+3.5^2] = [0.25+6.25, 2.25+12.25] = [6.5, 14.5]
    # 0.5 * ([9.0, 25.0] - [6.5, 14.5]) = 0.5 * ([2.5, 10.5]) = [1.25, 5.25]
    # Sum over embedding_dim: 1.25 + 5.25 = 6.5

    expected_output = torch.tensor([[11.0], [6.5]], dtype=torch.float32)

    output = second_order_interaction(x)

    assert torch.allclose(output, expected_output), (
        "second_order_interaction 函数计算不正确"
    )
    assert output.shape == (x.size(0), 1), "second_order_interaction 函数输出形状不正确"
    assert output.dtype == torch.float32, (
        "second_order_interaction 函数输出数据类型不正确"
    )

    print(f"second_order_interaction 输出: {output}")


if __name__ == "__main__":
    pytest.main([__file__])
