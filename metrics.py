"""
Metrics: a running-average tracker plus macro precision/recall/F1/accuracy
and per-class accuracy computations, evaluated only over each task's
"active classes" so retention on old tasks and progress on new tasks can
be reported separately.
"""

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class MulticlassMetrics:
    """Buffers per-batch metric values and reports their running mean."""

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.metrics = defaultdict(list)

    def update_state(self, **kwargs):
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().item()
            self.metrics[key].append(value)

    def result(self):
        return {k: np.mean(v) for k, v in self.metrics.items()}


def calculate_epoch_metrics(y_true, y_pred, num_classes, active_classes=None):
    """
    y_true, y_pred: numpy arrays of class ids (already argmax'd).
    Returns macro precision/recall/F1/accuracy plus per-class accuracy,
    restricted to `active_classes` if given.
    """
    metrics = {}

    if active_classes is None:
        active_classes = np.unique(y_true)

    metrics["acc"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(
        y_true, y_pred, labels=active_classes, average="macro", zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred, labels=active_classes, average="macro", zero_division=0
    )
    metrics["f1_score"] = f1_score(
        y_true, y_pred, labels=active_classes, average="macro", zero_division=0
    )

    for cls in active_classes:
        mask = y_true == cls
        if mask.sum() == 0:
            metrics[f"class_{cls}_accuracy"] = 0.0
        else:
            metrics[f"class_{cls}_accuracy"] = (y_pred[mask] == cls).mean()

    return metrics


def calculate_f1_precision_recall_accuracy(y_true, y_pred, average="macro", active_classes=None):
    """
    Tensor-based macro/weighted/micro F1/precision/recall + overall accuracy,
    used inside the training loop (y_pred are logits, not class ids).
    """
    y_pred = torch.argmax(y_pred, dim=1)

    if active_classes is None:
        active_classes = torch.unique(y_true).tolist()
    active_classes = list(active_classes)
    num_active = len(active_classes)

    tp = torch.zeros(num_active, device=y_true.device)
    fp = torch.zeros(num_active, device=y_true.device)
    fn = torch.zeros(num_active, device=y_true.device)
    support = torch.zeros(num_active, device=y_true.device)

    for idx, cls in enumerate(active_classes):
        tp[idx] = ((y_pred == cls) & (y_true == cls)).sum()
        fp[idx] = ((y_pred == cls) & (y_true != cls)).sum()
        fn[idx] = ((y_pred != cls) & (y_true == cls)).sum()
        support[idx] = (y_true == cls).sum()

    precision_per_class = tp / (tp + fp + 1e-8)
    recall_per_class = tp / (tp + fn + 1e-8)
    f1_per_class = (
        2 * precision_per_class * recall_per_class
        / (precision_per_class + recall_per_class + 1e-8)
    )
    total_support = support.sum()

    if average == "macro":
        precision = precision_per_class.mean()
        recall = recall_per_class.mean()
        f1 = f1_per_class.mean()
    elif average == "weighted":
        precision = (precision_per_class * support).sum() / (total_support + 1e-8)
        recall = (recall_per_class * support).sum() / (total_support + 1e-8)
        f1 = (f1_per_class * support).sum() / (total_support + 1e-8)
    elif average == "micro":
        tp_total, fp_total, fn_total = tp.sum(), fp.sum(), fn.sum()
        precision = tp_total / (tp_total + fp_total + 1e-8)
        recall = tp_total / (tp_total + fn_total + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    else:
        raise ValueError(f"Unsupported average type: {average}")

    acc = (y_pred == y_true).float().mean()

    return f1, precision, recall, acc


def calculate_class_wise_accuracy(y_true, y_pred, active_classes=None):
    y_pred = torch.argmax(y_pred, dim=1)

    if active_classes is None:
        active_classes = torch.unique(y_true).tolist()

    class_wise_accuracy = {}
    for cls in active_classes:
        mask = y_true == cls
        total_true = mask.sum().item()
        if total_true == 0:
            accuracy = 0.0
        else:
            accuracy = (y_pred[mask] == cls).float().mean().item()
        class_wise_accuracy[f"class_{cls}_accuracy"] = accuracy

    return class_wise_accuracy
