import numpy as np
from sklearn.metrics import average_precision_score


def compute_map(targets: np.ndarray, preds: np.ndarray) -> float:
    """Mean Average Precision across classes (macro, label-wise)."""
    return average_precision_score(targets, preds, average="macro")
