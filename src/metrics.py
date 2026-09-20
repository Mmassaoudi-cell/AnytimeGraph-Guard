from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, log_loss, recall_score,
)
from sklearn.preprocessing import label_binarize


def ece_score(y: np.ndarray, proba: np.ndarray, bins: int = 15) -> float:
    confidence = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = pred == y
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = proba.argmax(axis=1)
    labels = np.arange(proba.shape[1])
    recalls = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    ybin = label_binarize(y, classes=labels)
    if proba.shape[1] == 2 and ybin.shape[1] == 1:
        ybin = np.column_stack([1 - ybin[:, 0], ybin[:, 0]])
    try:
        pr = average_precision_score(ybin, proba, average="macro")
    except ValueError:
        pr = float("nan")
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "minority_recall": float(recalls.min()),
        "macro_pr_auc": float(pr),
        "ece": ece_score(y, proba),
        "nll": float(log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=labels)),
        "brier": float(np.mean(np.sum((proba - np.eye(proba.shape[1])[y]) ** 2, axis=1))),
    }

