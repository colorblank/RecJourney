import tensorflow as tf
from src.models.recall.tensorflow.two_tower import UserTower, ItemTower


def export_two_tower_models():
    """
    构建并保存用户塔和物品塔模型。
    """
    user_tower = UserTower()
    item_tower = ItemTower()

    # 示例：构建模型 (需要示例输入数据来构建图)
    # 由于使用了 Subclassing API，模型在第一次调用时构建。
    # 提供示例输入以构建模型并检查形状。
    # 示例输入数据 (需要根据实际数据格式调整)
    user_features_example = {
        "user_id": tf.constant(
            ["user_placeholder_1", "user_placeholder_2"], dtype=tf.string
        ),
        "age": tf.constant([25, 30], dtype=tf.int32),
        "gender": tf.constant(["male", "female"], dtype=tf.string),
        "recent_clicks": tf.constant(
            [
                ["item_placeholder_1", "item_placeholder_2"],
                ["item_placeholder_1", "item_placeholder_2"],
            ],
            dtype=tf.string,
        ),
    }
    item_features_example = {
        "item_id": tf.constant(
            ["item_placeholder_1", "item_placeholder_2"], dtype=tf.string
        ),
        "category": tf.constant(
            ["category_placeholder_1", "category_placeholder_2"], dtype=tf.string
        ),
        "title": tf.constant(
            [
                ["title_placeholder_1", "title_placeholder_2"],
                ["title_placeholder_1", "title_placeholder_2"],
            ],
            dtype=tf.string,
        ),
    }

    # 调用模型以构建图
    user_tower(user_features_example)
    item_tower(item_features_example)

    # 假设训练完成，保存模型
    # TODO: 根据实际训练情况加载权重
    user_tower.export("saved_models/user_tower_tf2")
    item_tower.export("saved_models/item_tower_tf2")

    print("User and item towers built and saved using TensorFlow 2.0 Subclassing API.")


if __name__ == "__main__":
    export_two_tower_models()
