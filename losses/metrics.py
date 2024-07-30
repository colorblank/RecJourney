from typing import List


def calculate_auc(y_true: List[int], y_pred: List[float]) -> float:
    """
    Calculate AUC (Area Under Curve) for binary classification.
    Args:
        y_true (List[int]): List of true labels.
        y_pred (List[float]): List of predicted probabilities.

    Returns:
        float: AUC value.
    """
    y_score_idx = sorted(range(len(y_pred)), key=lambda i: y_pred[i], reverse=True)
    y_true = [y_true[i] for i in y_score_idx]

    pos, neg = 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            pos += 1
        else:
            neg += 1
    if pos == 0 or neg == 0:
        return 0.0

    cum_pos, auc = 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            cum_pos += 1
        else:
            auc += cum_pos
    auc = auc / (pos * neg)
    return auc


if __name__ == "__main__":
    y_true = [0, 1, 0, 1, 0]
    y_scores = [0.1, 0.2, 0.35, 0.8, 0.7]
    print(calculate_auc(y_true, y_scores))
    from sklearn.metrics import roc_auc_score

    print(roc_auc_score(y_true, y_scores))
