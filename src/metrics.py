from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> Dict:
    """
    Calculate paper-ready classification metrics.
    """

    labels = list(range(len(class_names)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "per_class": report,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
    }


def format_metrics(metrics: Dict) -> str:
    return (
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Balanced accuracy: {metrics['balanced_accuracy']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )
