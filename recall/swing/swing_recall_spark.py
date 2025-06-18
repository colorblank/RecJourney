"""
@Author: [您的姓名/公司名称]
@Date: 2025-05-27
@Description: 考虑时间窗口的 Swing 召回算法的 PySpark 实现。
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import Optional


class SwingRecall:
    """
    Swing 召回算法的 PySpark 实现。

    Attributes:
        spark (SparkSession): SparkSession 实例。
        user_col (str): 用户 ID 列的名称。
        item_col (str): 物品 ID 列的名称。
        timestamp_col (str): 时间戳列的名称。
    """

    def __init__(
        self,
        spark: SparkSession,
        user_col: str = "user_id",
        item_col: str = "item_id",
        timestamp_col: str = "timestamp",
    ):
        """
        初始化 SwingRecall 类。

        Args:
            spark (SparkSession): SparkSession 实例。
            user_col (str): 用户 ID 列的名称。
            item_col (str): 物品 ID 列的名称。
            timestamp_col (str): 时间戳列的名称。
        """
        self.spark = spark
        self.user_col = user_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col

    def calculate_co_occurrence(self, data: F.DataFrame) -> F.DataFrame:
        """
        根据用户交互计算物品的共同出现。

        Args:
            data (F.DataFrame): 包含用户、物品和时间戳列的输入 DataFrame。

        Returns:
            F.DataFrame: 包含物品对共同出现计数的 DataFrame。
        """
        # 按用户分组并收集每个用户交互过的所有物品
        user_items_df = data.groupBy(self.user_col).agg(
            F.collect_set(self.item_col).alias("items")
        )

        # 展开物品列表，为每个用户生成所有唯一的物品对
        # 这种方法避免了大型自连接和潜在的 distinct() 问题
        co_occurrence_pairs = user_items_df.withColumn(
            "item1", F.explode("items")
        ).withColumn("item2", F.explode("items"))

        # 过滤掉自配对，并确保 item1 < item2 以避免重复的 (i,j) 和 (j,i)
        co_occurrence_pairs = co_occurrence_pairs.filter(
            F.col("item1") < F.col("item2")
        )

        # 统计每个物品对的共同出现次数
        co_occurrence = co_occurrence_pairs.groupBy("item1", "item2").agg(
            F.count(self.user_col).alias("co_occurrence_count")
        )
        return co_occurrence

    def calculate_swing_score(
        self, data: F.DataFrame, co_occurrence_df: F.DataFrame
    ) -> F.DataFrame:
        """
        计算物品对的 Swing 分数。

        Swing(i, j) = sum_{u in U(i) & U(j)} 1 / (|U(i)| + |U(j)| - |U(i) & U(j)|)

        Args:
            data (F.DataFrame): 包含用户、物品和时间戳列的原始输入 DataFrame。
                                用于计算 |U(i)| 和 |U(j)|。
            co_occurrence_df (F.DataFrame): 包含物品对及其共同出现计数的 DataFrame。

        Returns:
            F.DataFrame: 包含物品对及其 Swing 分数的 DataFrame。
        """
        # 获取与每个物品交互的唯一用户数 (|U(i)|)
        item_user_count = data.groupBy(self.item_col).agg(
            F.countDistinct(self.user_col).alias("user_count")
        )

        # 将共同出现计数与物品用户计数连接
        swing_df = (
            co_occurrence_df.alias("a")
            .join(
                item_user_count.alias("b"),
                F.col("a.item1") == F.col(f"b.{self.item_col}"),
                "inner",
            )
            .join(
                item_user_count.alias("c"),
                F.col("a.item2") == F.col(f"c.{self.item_col}"),
                "inner",
            )
            .select(
                F.col("a.item1"),
                F.col("a.item2"),
                F.col("a.co_occurrence_count"),
                F.col("b.user_count").alias("item1_user_count"),
                F.col("c.user_count").alias("item2_user_count"),
            )
        )

        # 计算 Swing 分数
        swing_df = swing_df.withColumn(
            "swing_score",
            F.col("co_occurrence_count")
            / (
                F.col("item1_user_count")
                + F.col("item2_user_count")
                - F.col("co_occurrence_count")
            ),
        )
        return swing_df

    def recommend_items(
        self, swing_scores: F.DataFrame, top_k: int = 10
    ) -> F.DataFrame:
        """
        根据 Swing 分数生成推荐。

        Args:
            swing_scores (F.DataFrame): 包含物品对及其 Swing 分数的 DataFrame。
            top_k (int): 为每个物品生成的 Top-K 推荐数量。

        Returns:
            F.DataFrame: 包含每个物品推荐物品的 DataFrame。
        """
        window_spec = Window.partitionBy("item1").orderBy(F.desc("swing_score"))
        ranked_recommendations = swing_scores.withColumn(
            "rank", F.rank().over(window_spec)
        ).filter(F.col("rank") <= top_k)

        return ranked_recommendations.select("item1", "item2", "swing_score", "rank")

    def get_swing_recommendations(
        self, data: F.DataFrame, time_window_days: Optional[int] = None, top_k: int = 10
    ) -> F.DataFrame:
        """
        获取 Swing 推荐的主方法，考虑时间窗口。

        Args:
            data (F.DataFrame): 包含用户、物品和时间戳列的输入 DataFrame。
            time_window_days (int, optional): 时间窗口的天数。
                                              如果为 None，则使用所有数据。
            top_k (int): 为每个物品生成的 Top-K 推荐数量。

        Returns:
            F.DataFrame: 包含每个物品推荐物品的 DataFrame。
        """
        processed_data = data

        if time_window_days is not None:
            # 过滤指定时间窗口内的数据
            max_timestamp = data.agg(F.max(self.timestamp_col)).collect()[0][0]
            start_timestamp = F.to_timestamp(F.lit(max_timestamp)) - F.expr(
                f"INTERVAL {time_window_days} DAYS"
            )
            processed_data = data.filter(F.col(self.timestamp_col) >= start_timestamp)

        co_occurrence_df = self.calculate_co_occurrence(processed_data)
        swing_scores = self.calculate_swing_score(processed_data, co_occurrence_df)
        recommendations = self.recommend_items(swing_scores, top_k)
        return recommendations


if __name__ == "__main__":
    # 示例用法：
    spark = (
        SparkSession.builder.appName("SwingRecallExample")
        .master("local[*]")
        .getOrCreate()
    )

    # 示例数据
    # 在实际场景中，这将从 HDFS、S3 等加载。
    data = [
        (1, 101, "2024-05-01 10:00:00"),
        (1, 102, "2024-05-01 11:00:00"),
        (1, 103, "2024-05-02 12:00:00"),
        (2, 101, "2024-05-03 13:00:00"),
        (2, 102, "2024-05-03 14:00:00"),
        (3, 102, "2024-05-04 15:00:00"),
        (3, 104, "2024-05-04 16:00:00"),
        (4, 101, "2024-05-20 10:00:00"),
        (4, 105, "2024-05-21 11:00:00"),
        (5, 101, "2024-05-25 12:00:00"),
        (5, 102, "2024-05-26 13:00:00"),
    ]
    columns = ["user_id", "item_id", "timestamp"]
    df = spark.createDataFrame(data, columns)
    df = df.withColumn("timestamp", F.to_timestamp(F.col("timestamp")))

    swing_recall = SwingRecall(spark)

    print("--- Swing Recommendations (All Time) ---")
    all_time_recommendations = swing_recall.get_swing_recommendations(df, top_k=2)
    all_time_recommendations.show()

    print("--- Swing Recommendations (Last 7 Days) ---")
    last_7_days_recommendations = swing_recall.get_swing_recommendations(
        df, time_window_days=7, top_k=2
    )
    last_7_days_recommendations.show()

    print("--- Swing Recommendations (Last 3 Days) ---")
    last_3_days_recommendations = swing_recall.get_swing_recommendations(
        df, time_window_days=3, top_k=2
    )
    last_3_days_recommendations.show()

    spark.stop()
