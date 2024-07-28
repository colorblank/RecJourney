import torch
from can import CoActionUnit


class TestCoActionUnit:
    def test_init(self):
        # 测试初始化方法
        emb_dim = 10
        hidden_dims = [20, 30]
        orders = 3
        order_indep = False
        cau = CoActionUnit(emb_dim, hidden_dims, orders, order_indep)
        assert cau.emb_dim == emb_dim
        assert cau.orders == orders
        assert cau.order_indep == order_indep
        assert cau.total_dims == emb_dim * 20 + 20 * 30

    def test_forward(self):
        # 测试前向传播方法
        emb_dim = 10
        hidden_dims = [20, 30]
        orders = 3
        order_indep = False
        cau = CoActionUnit(emb_dim, hidden_dims, orders, order_indep)

        batch_size = 5
        seq_len = 7
        ad_shape = (
            batch_size,
            emb_dim * hidden_dims[0] + hidden_dims[0] * hidden_dims[1],
        )
        his_items_shape = (batch_size, seq_len, emb_dim)

        ad = torch.rand(ad_shape)
        his_items = torch.rand(his_items_shape)
        mask = torch.rand(his_items_shape) > 0.5  # 创建一个随机的二值掩码
        out = cau.forward(ad, his_items, None)

        assert out.shape == (
            batch_size,
            sum(hidden_dims) * orders,
        )


# 运行测试
if __name__ == "__main__":
    test = TestCoActionUnit()
    test.test_init()
    test.test_forward()
