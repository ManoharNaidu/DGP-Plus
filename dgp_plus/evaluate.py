"""Metrics, the Macro-F1 threshold, seed aggregates, and comparison statistics.

The Macro-F1 threshold comes from validation only, then applies to the test set: 19 thresholds in [0.05, 0.95],
predict fraud if p > t, first best wins. Values are percentages, and std is the population std (ddof=0).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

METRICS = ("macro_f1", "auroc", "auprc")
THRESHOLDS = np.linspace(0.05, 0.95, 19)
BOOTSTRAP_ROUNDS = 2000


def macro_f1_at(labels, probabilities, threshold: float) -> float:
    predictions = (np.asarray(probabilities) > threshold).astype(int)
    return 100 * f1_score(np.asarray(labels).astype(int), predictions, average="macro", zero_division=0)


def select_threshold(val_labels, val_probabilities) -> float:
    best_f1, best_threshold = -1.0, 0.5
    for threshold in THRESHOLDS:
        score = macro_f1_at(val_labels, val_probabilities, threshold)
        if score > best_f1:
            best_f1, best_threshold = score, float(threshold)
    return best_threshold


def compute_metrics(labels, probabilities, threshold: float = 0.5) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("labels must be 0 or 1; unlabelled nodes must not reach evaluation")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    out = {"macro_f1": macro_f1_at(labels, probabilities, threshold), "auroc": float("nan"), "auprc": float("nan")}
    if len(np.unique(labels)) == 2:
        out["auroc"] = 100 * roc_auc_score(labels, probabilities)
        out["auprc"] = 100 * average_precision_score(labels, probabilities)
    return out


def evaluate_split(val_labels, val_probabilities, test_labels, test_probabilities) -> dict[str, dict[str, float]]:
    """Threshold from validation, applied to both splits. Macro-F1 at 0.5 is kept as a secondary number."""
    threshold = select_threshold(val_labels, val_probabilities)
    val = compute_metrics(val_labels, val_probabilities, threshold)
    test = compute_metrics(test_labels, test_probabilities, threshold)
    val["macro_f1_at_0.5"] = macro_f1_at(val_labels, val_probabilities, 0.5)
    test["macro_f1_at_0.5"] = macro_f1_at(test_labels, test_probabilities, 0.5)
    val["threshold"] = test["threshold"] = threshold
    return {"val": val, "test": test}


def aggregate(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and std of each metric over seed runs."""
    return {m: {"mean": float(np.mean([r[m] for r in runs])), "std": float(np.std([r[m] for r in runs]))} for m in METRICS}


def paired_bootstrap(labels: np.ndarray, runs_a: list[np.ndarray], runs_b: list[np.ndarray],
                     thresholds_a: list[float] | None = None, thresholds_b: list[float] | None = None) -> dict[str, list[float]]:
    """95% CI of the seed-averaged difference a - b, resampling test nodes.

    The same indices are used for every seed and both arms, so the comparison stays paired. Runs must be in seed order.
    Given each run's validation threshold, Macro-F1 is covered too, with the thresholds held fixed.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(0)
    differences: dict[str, list[float]] = {"auroc": [], "auprc": []}
    scorers = {"auroc": roc_auc_score, "auprc": average_precision_score}
    for _ in range(BOOTSTRAP_ROUNDS):
        index = rng.integers(0, len(labels), len(labels))
        if labels[index].min() == labels[index].max():
            continue
        for metric, score in scorers.items():
            differences[metric].append(100 * float(np.mean([score(labels[index], a[index]) - score(labels[index], b[index])
                                                            for a, b in zip(runs_a, runs_b)])))
        if thresholds_a is not None and thresholds_b is not None:
            differences.setdefault("macro_f1", []).append(float(np.mean(
                [macro_f1_at(labels[index], a[index], ta) - macro_f1_at(labels[index], b[index], tb)
                 for a, b, ta, tb in zip(runs_a, runs_b, thresholds_a, thresholds_b)])))
    return {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in differences.items()}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values for one family of tests."""
    adjusted, running = {}, 0.0
    ordered = sorted(p_values, key=p_values.get)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted
