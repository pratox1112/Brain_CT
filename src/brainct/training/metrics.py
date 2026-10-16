"""Evaluation metrics: per-label ROC AUC, macro AUC, F1 at val-chosen thresholds, confusion
matrices, and patient-level bootstrap confidence intervals.

Macro AUC and macro F1 average the 5 subtypes; `any` is reported separately.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

from brainct.data.labels import LABELS, SUBTYPES

SUBTYPE_COLUMNS = [LABELS.index(s) for s in SUBTYPES]


def auc_per_label(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """ROC AUC for each of the 6 labels (NaN if a label has only one class)."""
    aucs = {}
    for j, label in enumerate(LABELS):
        column = y_true[:, j]
        both_classes = 0 < column.sum() < len(column)
        aucs[label] = float(roc_auc_score(column, y_prob[:, j])) if both_classes else float("nan")
    return aucs


def macro(values: dict[str, float]) -> float:
    return float(np.mean([values[s] for s in SUBTYPES]))


def best_f1_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Per label, the probability threshold that maximizes F1 (choose on val, apply to test)."""
    thresholds = {}
    for j, label in enumerate(LABELS):
        precision, recall, cut = precision_recall_curve(y_true[:, j], y_prob[:, j])
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        thresholds[label] = float(cut[int(np.argmax(f1[:-1]))])  # the last point has no threshold
    return thresholds


def confusion(y_true: np.ndarray, y_prob: np.ndarray, thresholds: dict[str, float]) -> dict:
    """Per-label 2x2 counts and F1 at the given thresholds."""
    out = {}
    for j, label in enumerate(LABELS):
        truth = y_true[:, j].astype(bool)
        pred = y_prob[:, j] >= thresholds[label]
        tp, fp = int((truth & pred).sum()), int((~truth & pred).sum())
        fn, tn = int((truth & ~pred).sum()), int((~truth & ~pred).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        out[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": f1}
    return out


def bootstrap_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile CIs for each label's AUC and the macro AUC, resampling *patients*.

    Slices of one patient are correlated, so resampling images would understate uncertainty.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    _, starts, counts = np.unique(groups[order], return_index=True, return_counts=True)
    members = [order[s : s + c] for s, c in zip(starts, counts, strict=True)]

    samples: dict[str, list[float]] = {label: [] for label in [*LABELS, "macro"]}
    for _ in range(n_boot):
        picked = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in picked])
        aucs = auc_per_label(y_true[idx], y_prob[idx])
        for label, value in aucs.items():
            samples[label].append(value)
        samples["macro"].append(macro(aucs))

    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {
        label: tuple(float(v) for v in np.nanpercentile(values, q))
        for label, values in samples.items()
    }


def report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict[str, float],
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Everything evaluate.py records for one split."""
    aucs = auc_per_label(y_true, y_prob)
    matrices = confusion(y_true, y_prob, thresholds)
    result: dict[str, Any] = {
        "images": int(len(y_true)),
        "positives": {label: int(y_true[:, j].sum()) for j, label in enumerate(LABELS)},
        "auc": aucs,
        "macro_auc": macro(aucs),
        "thresholds": thresholds,
        "f1": {label: matrices[label]["f1"] for label in LABELS},
        "macro_f1": macro({label: matrices[label]["f1"] for label in LABELS}),
        "confusion": matrices,
    }
    if groups is not None and n_boot > 0:
        result["auc_ci95"] = bootstrap_auc(y_true, y_prob, groups, n_boot=n_boot, seed=seed)
        result["bootstrap"] = {"unit": "patient", "resamples": n_boot, "seed": seed}
    return result
