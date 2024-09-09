from collections import defaultdict
from typing import Any, List


def calculate_auc(y_true: List[int], y_pred: List[float]) -> float:
    """
    Calculate AUC (Area Under Curve) for binary classification.
    Args:
        y_true (List[int]): List of true labels.
        y_pred (List[float]): List of predicted probabilities.

    Returns:
        float: AUC value.
    """
    rank_info = enumerate(
        sorted(
            zip(y_true, y_pred),
            key=lambda x: x[1],
        ),
        start=1,
    )
    rank_true = [rank for rank, y in rank_info if y[0] == 1]  # index of true label
    M = sum(y_true)
    N = len(y_true) - M

    if M == 0 or N == 0:  # no true label or no false label
        return 0.0

    auc = (sum(rank_true) - M * (M + 1) / 2) / (M * N)
    return auc


def calculdate_gauc(
    y_true: List[int], y_pred: List[float], user_ids: List[Any]
) -> float:
    """
    Calculate Group AUC (Area Under Curve) for binary classification.
    Args:
        y_true (List[int]): List of true labels.
        y_pred (List[float]): List of predicted probabilities.
        user_ids (List[any]): List of user IDs.

    Returns:
        float: Group AUC value.
    """

    groups = defaultdict(lambda: {"scores": [], "truths": []})
    for user_id, score, truth in zip(user_ids, y_pred, y_true):
        groups[user_id]["scores"].append(score)
        groups[user_id]["truths"].append(truth)

    return (
        sum(
            calculate_auc(truths, scores) * len(truths)
            for _, (scores, truths) in groups.items()
        )
        / len(groups)
        / len(y_true)
    )


if __name__ == "__main__":
    y_true = [0, 1, 0, 1, 0]
    y_scores = [0.1, 0.2, 0.35, 0.8, 0.7]
    print(calculate_auc(y_true, y_scores))
    from sklearn.metrics import roc_auc_score

    print(roc_auc_score(y_true, y_scores))
