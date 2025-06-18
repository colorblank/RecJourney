"""
MMR (Maximal Marginal Relevance) 重排的 Python 实现。

该模块提供了 MMR 算法，用于对推荐系统中的候选物品进行排序，
旨在平衡物品的原始相关性得分和多样性。多样性根据指定的多个物品标签计算。
"""

import numpy as np
from typing import List, Dict, Any, Optional


class MMRReranker:
    """
    MMR (Maximal Marginal Relevance) 重排器类。

    该类实现了 MMR 算法，用于在推荐系统中平衡物品的相关性（原始得分）和多样性。
    多样性根据指定的物品标签计算。
    """

    def __init__(self, diversity_fields: List[str], lambda_param: float = 0.5):
        """
        初始化 MMRReranker。

        Args:
            diversity_fields: 用于计算多样性的物品属性字段列表（例如：["author", "topic", "category"]）。
                              这些字段的值可以是单个字符串/数字，也可以是字符串/数字列表。
            lambda_param: 平衡因子，介于 0.0 和 1.0 之间。
                          lambda_param 越大，越侧重于原始得分；越小，越侧重于多样性。
        """
        if not (0.0 <= lambda_param <= 1.0):
            raise ValueError("lambda_param 必须在 0.0 和 1.0 之间。")
        if not diversity_fields:
            raise ValueError("diversity_fields 不能为空，MMR 需要指定多样性字段。")

        self.diversity_fields = diversity_fields
        self.lambda_param = lambda_param

    def _calculate_item_similarity(
        self, item1: Dict[str, Any], item2: Dict[str, Any]
    ) -> float:
        """
        计算两个物品之间的相似度。

        相似度基于 diversity_fields 中指定标签的重叠程度。
        对于每个 diversity_field，计算其值的 Jaccard 相似度（如果值是列表）
        或简单匹配（如果值是单个）。然后对所有字段的相似度取平均。

        Args:
            item1: 第一个物品字典。
            item2: 第二个物品字典。

        Returns:
            两个物品之间的相似度得分（0.0 到 1.0 之间）。
        """
        total_similarity = 0.0
        num_fields = 0

        for field in self.diversity_fields:
            value1 = item1.get(field)
            value2 = item2.get(field)

            if value1 is None or value2 is None:
                continue

            num_fields += 1

            if isinstance(value1, list) and isinstance(value2, list):
                set1 = set(value1)
                set2 = set(value2)
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                if union > 0:
                    total_similarity += intersection / union
                # else: 两个集合都为空，或者其中一个为空，相似度为0
            elif not isinstance(value1, list) and not isinstance(value2, list):
                # 对于非列表类型，如果值相同则相似度为1，否则为0
                total_similarity += 1.0 if value1 == value2 else 0.0
            # 如果类型不匹配（一个列表一个非列表），则视为不相似，不增加相似度

        if num_fields == 0:
            return 0.0  # 没有可用于计算相似度的字段

        return total_similarity / num_fields

    def rerank(
        self, candidate_items: List[Dict[str, Any]], k: int
    ) -> List[Dict[str, Any]]:
        """
        使用 MMR 算法对候选物品进行重排。

        Args:
            candidate_items: 候选物品列表，每个物品是一个字典，必须包含 'item_id' 和 'score' 字段，
                             以及 diversity_fields 中指定的标签字段。
            k: 需要选择的最终物品数量。

        Returns:
            重排后的物品列表，数量为 k。
        """
        if not candidate_items or k <= 0:
            return []

        # 确保所有物品都有 'score' 字段，如果没有则默认为 0.0
        for item in candidate_items:
            if "score" not in item:
                item["score"] = 0.0

        reranked_list: List[Dict[str, Any]] = []
        candidate_pool: List[Dict[str, Any]] = list(candidate_items)

        # 第一次选择：选择原始得分最高的物品
        # 假设原始分数在 'score' 字段中
        first_item = max(candidate_pool, key=lambda x: x["score"])
        reranked_list.append(first_item)
        candidate_pool.remove(first_item)

        while len(reranked_list) < k and candidate_pool:
            max_mmr_score = -np.inf
            best_item_for_this_iteration: Optional[Dict[str, Any]] = None

            for candidate_item in candidate_pool:
                # Sim1: 物品的原始得分
                sim1 = candidate_item["score"]

                # Sim2: 物品与已选物品集中最相似物品的相似度
                # 如果 reranked_list 为空（理论上不会，因为第一个物品已选），则 Sim2 为 0
                if not reranked_list:
                    sim2 = 0.0
                else:
                    max_sim_with_selected = 0.0
                    for selected_item in reranked_list:
                        sim = self._calculate_item_similarity(
                            candidate_item, selected_item
                        )
                        if sim > max_sim_with_selected:
                            max_sim_with_selected = sim
                    sim2 = max_sim_with_selected

                # 计算 MMR
                mmr_score = (self.lambda_param * sim1) - (
                    (1 - self.lambda_param) * sim2
                )

                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    best_item_for_this_iteration = candidate_item

            if best_item_for_this_iteration:
                reranked_list.append(best_item_for_this_iteration)
                candidate_pool.remove(best_item_for_this_iteration)
            else:
                # 如果没有找到合适的物品（例如，所有物品都被过滤或相似度过高），则退出循环
                break

        # 如果最终列表数量不足k，则从剩余的候选中补充，不再考虑多样性
        # 这一步确保即使多样性限制导致物品不足，也能凑够 k 个
        if len(reranked_list) < k:
            selected_item_ids = {item["item_id"] for item in reranked_list}
            # 将剩余的候选物品按原始分数降序排序
            remaining_sorted_candidates = sorted(
                candidate_pool, key=lambda x: x["score"], reverse=True
            )
            for item in remaining_sorted_candidates:
                if len(reranked_list) >= k:
                    break
                if item["item_id"] not in selected_item_ids:
                    reranked_list.append(item)
                    selected_item_ids.add(item["item_id"])

        return reranked_list


def test_consecutive_diversity(
    reranker_instance: MMRReranker,
    reranked_items: List[Dict[str, Any]],
    fields_to_check: List[str],
) -> None:
    """
    测试重排后的列表中是否存在连续两个物品在指定字段上相同的情况。

    Args:
        reranker_instance: MMRReranker 实例。
        reranked_items: 重排后的物品列表。
        fields_to_check: 需要检查的字段列表（例如：["author", "topic"]）。
    """
    print("\n--- 连续多样性测试 ---")
    print("正在执行连续多样性测试...")
    has_consecutive_duplicate = False
    for i in range(len(reranked_items) - 1):
        item1 = reranked_items[i]
        item2 = reranked_items[i + 1]
        for field in fields_to_check:
            value1 = item1.get(field)
            value2 = item2.get(field)

            # 检查值是否相同，并确保它们不是 None
            if value1 is not None and value2 is not None and value1 == value2:
                print(
                    f"警告: 物品 {item1['item_id']} 和 {item2['item_id']} 在字段 '{field}' 上连续相同 (值: {value1})"
                )
                has_consecutive_duplicate = True
                # 如果找到一个连续重复，可以跳出内层循环，检查下一个物品对
                break
    if not has_consecutive_duplicate:
        print("重排后的列表中没有发现连续的作者或主题重复。")
    else:
        print("重排后的列表中发现连续的作者或主题重复。")


if __name__ == "__main__":
    # 示例用法
    # 1. 准备候选物品列表，包含多样性字段
    candidate_items_example = [
        {
            "item_id": "item_1",
            "score": 0.95,
            "author": "Author A",
            "topic": "Topic X",
            "stock": ["AAPL", "MSFT"],
        },
        {
            "item_id": "item_2",
            "score": 0.92,
            "author": "Author A",
            "topic": "Topic Y",
            "stock": ["GOOG"],
        },
        {
            "item_id": "item_3",
            "score": 0.88,
            "author": "Author A",
            "topic": "Topic X",
            "stock": ["MSFT", "AMZN"],
        },  # 与item_1作者主题相同，股票有重叠
        {
            "item_id": "item_4",
            "score": 0.98,
            "author": "Author C",
            "topic": "Topic Z",
            "stock": ["TSLA"],
        },
        {
            "item_id": "item_5",
            "score": 0.90,
            "author": "Author B",
            "topic": "Topic X",
            "stock": ["AMZN"],
        },  # 与item_2作者相同，与item_1/3主题相同
        {
            "item_id": "item_6",
            "score": 0.80,
            "author": "Author A",
            "topic": "Topic Y",
            "stock": ["NVDA"],
        },  # 与item_1/3作者相同，与item_2主题相同
        {
            "item_id": "item_7",
            "score": 0.87,
            "author": "Author D",
            "topic": "Topic X",
            "stock": ["FB"],
        },  # 与item_1/3/5主题相同
    ]

    print("原始候选物品:")
    for item in candidate_items_example:
        print(
            f"  Item ID: {item['item_id']}, Score: {item['score']:.2f}, Author: {item.get('author')}, Topic: {item.get('topic')}, Stock: {item.get('stock')}"
        )

    # 2. 初始化 MMRReranker，并指定多样性字段和 lambda 参数
    diversity_fields_example = [
        "author",
        "topic",
        "stock",
    ]  # 示例：对作者、主题和股票进行多样性控制
    lambda_param_example = 0.01  # 平衡因子，0.1 表示更侧重多样性
    k_example = 5  # 选择5个物品

    reranker = MMRReranker(
        diversity_fields=diversity_fields_example, lambda_param=lambda_param_example
    )

    # 3. 执行重排
    reranked_results = reranker.rerank(candidate_items_example, k=k_example)

    print(f"\n重排后的物品 (考虑多样性，选择前 {k_example} 个):")
    for item in reranked_results:
        print(
            f"  Item ID: {item['item_id']}, Original Score: {item['score']:.2f}, Author: {item.get('author')}, Topic: {item.get('topic')}, Stock: {item.get('stock')}"
        )

    print(
        "\n注意: 多样性处理后，物品的最终顺序可能不再严格按照原始分数降序，而是优先保证多样性。"
    )
    print("最终列表数量:", len(reranked_results))

    # 运行测试用例
    test_consecutive_diversity(reranker, reranked_results, ["author", "topic"])
