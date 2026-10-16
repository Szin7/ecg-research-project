import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Binary classification metrics. y_prob is the positive-class
    (arrhythmia) probability, used for the threshold-independent scores."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        # only one class present in this split/batch -- AUC is undefined
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return metrics


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Sweep candidate thresholds and return the one maximizing F1.

    With a pos_weight-reweighted loss (needed for the class imbalance here),
    the raw sigmoid output is no longer calibrated around 0.5 -- a fixed 0.5
    cutoff turned out to make validation F1 swing wildly epoch to epoch even
    though ROC-AUC/PR-AUC stayed stable, because small shifts in BatchNorm's
    eval-mode calibration moved thousands of borderline negatives across the
    threshold. Call this on validation data only and freeze the result;
    reusing it unchanged on test/adversarial data avoids tuning a threshold
    against the same data it gets reported on.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    # precision_recall_curve sorts by score internally (O(n log n)) instead
    # of the O(n^2) you'd get scoring every unique probability with a fresh
    # f1_score call -- that difference is the gap between seconds and
    # minutes once this runs once per epoch on tens of thousands of beats.
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]  # drop the "predict nothing" endpoint

    if len(thresholds) == 0:
        return 0.5

    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[np.argmax(f1)])


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
    """Multi-class analog of compute_metrics: no single "positive class"
    to define precision/recall/AUC against, so this reports accuracy,
    macro/weighted F1, and a per-class breakdown instead. Macro-F1 (an
    unweighted average across classes) is the early-stopping/model-
    selection metric for the multi-class experiment, the same role F1
    plays for the binary task -- it doesn't let the majority class hide
    poor minority-class performance the way accuracy would.
    """
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
