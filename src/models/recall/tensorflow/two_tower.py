import tensorflow as tf
from tensorflow.keras import Model, layers


# 占位符函数，需要根据实际数据加载词汇表
def get_user_vocab():
    """
    获取用户词汇表。

    TODO: 根据实际数据加载用户词汇表。

    Returns:
        list: 用户词汇列表。
    """
    return ["user_placeholder_1", "user_placeholder_2"]


def get_item_vocab():
    """
    获取物品词汇表。

    TODO: 根据实际数据加载物品词汇表。

    Returns:
        list: 物品词汇列表。
    """
    return ["item_placeholder_1", "item_placeholder_2"]


def get_category_vocab():
    """
    获取类别词汇表。

    TODO: 根据实际数据加载类别词汇表。

    Returns:
        list: 类别词汇列表。
    """
    return ["category_placeholder_1", "category_placeholder_2"]


def get_title_vocab():
    """
    获取标题词汇表。

    TODO: 根据实际数据加载标题词汇表。

    Returns:
        list: 标题词汇列表。
    """
    return ["title_placeholder_1", "title_placeholder_2"]


class MeanPoolingLayer(layers.Layer):
    """
    平均池化层。
    """

    def call(self, inputs):
        """
        执行平均池化操作。

        Args:
            inputs (tf.Tensor): 输入张量，形状为 (batch_size, sequence_length, embedding_dim)。

        Returns:
            tf.Tensor: 输出张量，形状为 (batch_size, embedding_dim)。
        """
        return tf.reduce_mean(inputs, axis=1)


class UserTower(Model):
    """
    用户塔模型。
    """

    def __init__(self, embedding_dim=8):
        """
        初始化用户塔。

        Args:
            embedding_dim (int): 嵌入向量的维度。
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        user_vocab_size = len(get_user_vocab()) + 1
        item_vocab_size = len(get_item_vocab()) + 1
        gender_vocab_size = len(["male", "female"]) + 1

        self.user_id_lookup = layers.StringLookup(vocabulary=get_user_vocab())
        self.gender_lookup = layers.StringLookup(vocabulary=["male", "female"])
        self.recent_clicks_lookup = layers.StringLookup(vocabulary=get_item_vocab())

        self.user_embedding = layers.Embedding(user_vocab_size, embedding_dim)
        self.age_embedding = layers.Embedding(100, embedding_dim)
        self.gender_embedding = layers.Embedding(gender_vocab_size, embedding_dim)
        self.recent_clicks_embedding = layers.Embedding(item_vocab_size, embedding_dim)

        self.mean_pooling = MeanPoolingLayer()

        self.dense = layers.Dense(64, activation="relu")

    def call(self, inputs):
        """
        用户塔的前向传播。

        Args:
            inputs (dict): 用户特征张量字典。
                'user_id' (tf.Tensor): 用户 ID 张量 (形状: batch_size)。
                'age' (tf.Tensor): 年龄张量 (形状: batch_size)。
                'gender' (tf.Tensor): 性别张量 (形状: batch_size)。
                'recent_clicks' (tf.Tensor): 最近点击张量 (形状: batch_size, sequence_length)。

        Returns:
            tf.Tensor: 输出张量，形状为 (batch_size, 64)。
        """
        user_id = inputs["user_id"]
        age = inputs["age"]
        gender = inputs["gender"]
        recent_clicks = inputs["recent_clicks"]

        user_indices = self.user_id_lookup(user_id)
        gender_indices = self.gender_lookup(gender)
        recent_clicks_indices = self.recent_clicks_lookup(recent_clicks)

        user_emb = self.user_embedding(user_indices)
        age_emb = self.age_embedding(age)
        gender_emb = self.gender_embedding(gender_indices)
        recent_clicks_emb = self.recent_clicks_embedding(recent_clicks_indices)

        recent_clicks_pooled = self.mean_pooling(recent_clicks_emb)

        concat = tf.concat(
            [user_emb, age_emb, gender_emb, recent_clicks_pooled], axis=1
        )

        output = self.dense(concat)
        return output


class ItemTower(Model):
    """
    物品塔模型。
    """

    def __init__(self, embedding_dim=8):
        """
        初始化物品塔。

        Args:
            embedding_dim (int): 嵌入向量的维度。
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        item_vocab_size = len(get_item_vocab()) + 1
        category_vocab_size = len(get_category_vocab()) + 1
        title_vocab_size = len(get_title_vocab()) + 1

        self.item_id_lookup = layers.StringLookup(vocabulary=get_item_vocab())
        self.category_lookup = layers.StringLookup(vocabulary=get_category_vocab())
        self.title_lookup = layers.StringLookup(vocabulary=get_title_vocab())

        self.item_embedding = layers.Embedding(item_vocab_size, embedding_dim)
        self.category_embedding = layers.Embedding(category_vocab_size, embedding_dim)
        self.title_embedding = layers.Embedding(title_vocab_size, embedding_dim)

        self.mean_pooling = MeanPoolingLayer()

        self.dense = layers.Dense(64, activation="relu")

    def call(self, inputs):
        """
        物品塔的前向传播。

        Args:
            inputs (dict): 物品特征张量字典。
                'item_id' (tf.Tensor): 物品 ID 张量 (形状: batch_size)。
                'category' (tf.Tensor): 类别张量 (形状: batch_size)。
                'title' (tf.Tensor): 标题张量 (形状: batch_size, sequence_length)。

        Returns:
            tf.Tensor: 输出张量，形状为 (batch_size, 64)。
        """
        item_id = inputs["item_id"]
        category = inputs["category"]
        title = inputs["title"]

        item_indices = self.item_id_lookup(item_id)
        category_indices = self.category_lookup(category)
        title_indices = self.title_lookup(title)

        item_emb = self.item_embedding(item_indices)
        category_emb = self.category_embedding(category_indices)
        title_emb = self.title_embedding(title_indices)

        title_pooled = self.mean_pooling(title_emb)

        concat = tf.concat([item_emb, category_emb, title_pooled], axis=1)

        output = self.dense(concat)
        return output


class TwoTowerModel(Model):
    """
    双塔模型。
    """

    def __init__(self, user_tower, item_tower):
        """
        初始化双塔模型。

        Args:
            user_tower (UserTower): 用户塔模型。
            item_tower (ItemTower): 物品塔模型。
        """
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def call(self, inputs):
        """
        双塔模型的前向传播。

        Args:
            inputs (dict): 包含用户和物品特征字典的字典。
                'user' (dict): 用户特征张量字典。
                'item' (dict): 物品特征张量字典。

        Returns:
            tf.Tensor: Logits 张量，形状为 (batch_size, num_items)。
        """
        user_features = inputs["user"]
        item_features = inputs["item"]

        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)

        logits = tf.matmul(user_emb, item_emb, transpose_b=True)
        return logits
