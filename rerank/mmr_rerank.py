"""最大边缘相关性(Maximal Marginal Relevance)重排算法实现。

该模块实现了MMR算法，用于在推荐系统中平衡相关性和多样性。
"""

import random
from typing import Dict, List, Optional


class Item:
    """表示推荐系统中的物品。

    属性:
        item_id: 物品唯一标识符
        score: 物品相关性分数
        author: 物品作者
        topic: 物品主题分类
    """

    def __init__(self, item_id: str, score: float, author: str, topic: str) -> None:
        self.item_id = item_id
        self.score = score
        self.author = author
        self.topic = topic

    def calculate_similarity(self, other_item: "Item") -> float:
        """计算与另一个物品的相似度分数。

        基于作者和主题的简单相似度计算:
        - 相同作者: +0.5
        - 相同主题: +0.5

        参数:
            other_item: 要比较的物品对象

        返回:
            相似度分数(0.0到1.0之间)
        """
        similarity = 0.0
        if self.author == other_item.author:
            similarity += 0.5
        if self.topic == other_item.topic:
            similarity += 0.5
        return similarity

    def __repr__(self) -> str:
        return (
            f"Item(id='{self.item_id}', score={self.score:.2f}, "
            f"author='{self.author}', topic='{self.topic}')"
        )


class MMRReRanker:
    """实现MMR重排算法。

    属性:
        lambda_diversity: 多样性权重系数(0.0-1.0)
    """

    def __init__(self, lambda_diversity: float = 0.5) -> None:
        if not 0 <= lambda_diversity <= 1:
            raise ValueError("lambda_diversity must be between 0 and 1.")
        self.lambda_diversity = lambda_diversity

    def re_rank(self, items: List[Item], num_results: int) -> List[Item]:
        """对物品列表进行MMR重排。

        参数:
            items: 待重排的物品列表
            num_results: 最终返回的物品数量

        返回:
            重排后的物品列表

        异常:
            TypeError: 如果输入包含非Item对象
        """
        if not items:
            return []

        if not all(isinstance(item, Item) for item in items):
            raise TypeError("All elements must be Item instances.")

        candidate_items = items.copy()
        re_ranked_items: List[Item] = []

        while len(re_ranked_items) < num_results and candidate_items:
            best_score = -float("inf")
            best_item: Optional[Item] = None

            for current in candidate_items:
                relevance = current.score
                diversity = self._calculate_diversity(current, re_ranked_items)
                mmr_score = self._calculate_mmr(relevance, diversity)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = current

            if best_item:
                re_ranked_items.append(best_item)
                candidate_items.remove(best_item)
            else:
                break

        return re_ranked_items

    def _calculate_diversity(self, item: Item, selected: List[Item]) -> float:
        """计算当前物品相对于已选物品集的多样性分数。"""
        if not selected:
            return 1.0

        max_similarity = max(item.calculate_similarity(sel) for sel in selected)
        return 1.0 - max_similarity

    def _calculate_mmr(self, relevance: float, diversity: float) -> float:
        """计算MMR分数。"""
        return (self.lambda_diversity * diversity) + (
            (1 - self.lambda_diversity) * relevance
        )


def generate_sample_items(num_items: int) -> List[Item]:
    """生成模拟物品数据用于测试。

    参数:
        num_items: 要生成的物品数量

    返回:
        生成的物品列表
    """
    authors = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"]
    topics = ["Tech", "Science", "Sports", "Art", "History", "Food"]
    return [
        Item(
            item_id=f"item_{i:03d}",
            score=round(random.uniform(0.1, 1.0), 2),
            author=random.choice(authors),
            topic=random.choice(topics),
        )
        for i in range(num_items)
    ]


def recall_by_author(
    author_name: str, num_recall: int, item_db: List[Item]
) -> List[Item]:
    """根据作者名称召回物品。

    参数:
        author_name: 作者名称
        num_recall: 召回数量
        item_db: 物品数据库

    返回:
        召回的物品列表
    """
    recalled = [item for item in item_db if item.author == author_name]
    random.shuffle(recalled)
    return recalled[:num_recall]


def recall_by_topic(
    topic_name: str, num_recall: int, item_db: List[Item]
) -> List[Item]:
    """根据主题召回物品。

    参数:
        topic_name: 主题名称
        num_recall: 召回数量
        item_db: 物品数据库

    返回:
        召回的物品列表
    """
    recalled = [item for item in item_db if item.topic == topic_name]
    random.shuffle(recalled)
    return recalled[:num_recall]


def deduplicate_and_score(
    author_recalled: List[Item], topic_recalled: List[Item]
) -> List[Item]:
    """对多路召回结果进行去重和重新评分。

    参数:
        author_recalled: 作者召回结果
        topic_recalled: 主题召回结果

    返回:
        去重后按新分数排序的物品列表
    """
    unique_items: Dict[str, Item] = {}

    # 处理作者召回列表
    for idx, item in enumerate(author_recalled):
        if item.item_id not in unique_items:
            rank_score = 1.0 - (idx / len(author_recalled))
            item.score = rank_score * 10
            unique_items[item.item_id] = item

    # 处理主题召回列表
    for idx, item in enumerate(topic_recalled):
        if item.item_id not in unique_items:
            rank_score = 1.0 - (idx / len(topic_recalled))
            item.score = rank_score * 10
            unique_items[item.item_id] = item

    # 转换为列表并排序
    dedup_items = list(unique_items.values())
    dedup_items.sort(key=lambda x: x.score, reverse=True)
    return dedup_items


def main() -> None:
    """主函数，演示MMR重排流程。"""
    # 创建模拟数据库
    item_db = generate_sample_items(200)

    # 召回阶段
    author_recall = recall_by_author("Alice", 30, item_db)
    topic_recall = recall_by_topic("Tech", 30, item_db)

    print(f"作者召回数量: {len(author_recall)}")
    print(f"主题召回数量: {len(topic_recall)}")

    # 去重和重新评分
    candidates = deduplicate_and_score(author_recall, topic_recall)

    print(f"\n去重后物品数量: {len(candidates)}")
    print("去重后物品 (前10个):")
    for item in candidates[:10]:
        print(item)

    # MMR重排
    reranker = MMRReRanker(lambda_diversity=0.7)
    final_items = reranker.re_rank(candidates, 6)

    print("\nMMR重排结果 (lambda=0.7):")
    for item in final_items:
        print(item)


if __name__ == "__main__":
    main()


# MMR 重排阶段
num_final_results = 6

# 尝试不同的多样性参数
print("\n--- MMR 重排结果 (lambda_diversity = 0.7，更偏向多样性) ---")
mmr_re_ranker_diversity = MMRReRanker(lambda_diversity=0.7)
final_ranked_items_diversity = mmr_re_ranker_diversity.re_rank(
    candidate_for_rerank, num_final_results
)

print("最终推荐结果:")
for item in final_ranked_items_diversity:
    print(item)

print("\n--- MMR 重排结果 (lambda_diversity = 0.3，更偏向准确性/召回顺序) ---")
mmr_re_ranker_accuracy = MMRReRanker(lambda_diversity=0.3)
final_ranked_items_accuracy = mmr_re_ranker_accuracy.re_rank(
    candidate_for_rerank, num_final_results
)

print("最终推荐结果:")
for item in final_ranked_items_accuracy:
    print(item)
