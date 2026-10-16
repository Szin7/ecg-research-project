import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.torch_dataset import ECGBeatDataset
from src.evaluation.metrics import best_f1_threshold, compute_metrics, compute_multiclass_metrics
from src.evaluation.predict import predict_classes
from src.evaluation.robustness import attack_success_rate
from src.models.cnn1d import ECGCNN


def test_perfect_predictions_score_maximally():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8])

    metrics = compute_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_confusion_matrix_matches_known_counts():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])

    metrics = compute_metrics(y_true, y_pred, y_pred.astype(float))

    assert metrics["confusion_matrix"] == [[1, 1], [1, 2]]


def test_recall_exposes_false_negatives_that_accuracy_hides():
    # 90 easy normals right, all 10 arrhythmia beats missed -- accuracy
    # still looks good, recall/F1 should not.
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.array([0] * 100)
    y_prob = np.zeros(100)

    metrics = compute_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.9
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_single_class_split_does_not_crash_auc():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3])

    metrics = compute_metrics(y_true, y_pred, y_prob)

    assert np.isnan(metrics["pr_auc"])
    assert np.isnan(metrics["roc_auc"])
    assert metrics["accuracy"] == 1.0


def test_best_f1_threshold_recovers_a_clean_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])

    threshold = best_f1_threshold(y_true, y_prob)

    preds = (y_prob >= threshold).astype(int)
    assert compute_metrics(y_true, preds, y_prob)["f1"] == 1.0


def test_best_f1_threshold_beats_a_fixed_bad_default():
    # probabilities shifted well above 0.5 for every sample (as a
    # pos_weight-reweighted model's outputs can be) -- a fixed 0.5 cutoff
    # would call everything positive, calibration should do better.
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.55, 0.58, 0.6, 0.62, 0.9, 0.95])

    fixed_f1 = compute_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob)["f1"]
    threshold = best_f1_threshold(y_true, y_prob)
    calibrated_f1 = compute_metrics(y_true, (y_prob >= threshold).astype(int), y_prob)["f1"]

    assert calibrated_f1 >= fixed_f1


def test_best_f1_threshold_handles_single_class():
    assert best_f1_threshold(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) == 0.5


def test_multiclass_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 2, 3])
    y_pred = np.array([0, 0, 1, 2, 3])

    metrics = compute_multiclass_metrics(y_true, y_pred, class_names=["N", "S", "V", "F"])

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["per_class"]["V"]["recall"] == 1.0


def test_multiclass_metrics_macro_f1_penalizes_ignoring_rare_classes():
    # a classifier that always predicts the majority class N: accuracy
    # looks fine, macro-F1 should not, since it weights every class equally
    y_true = np.array([0] * 90 + [1] * 5 + [2] * 3 + [3] * 2)
    y_pred = np.array([0] * 100)

    metrics = compute_multiclass_metrics(y_true, y_pred, class_names=["N", "S", "V", "F"])

    assert metrics["accuracy"] == 0.9
    assert metrics["macro_f1"] < 0.5
    assert metrics["per_class"]["F"]["recall"] == 0.0


def test_multiclass_metrics_confusion_matrix_shape_and_support():
    y_true = np.array([0, 1, 1, 2, 3, 3, 3])
    y_pred = np.array([0, 1, 0, 2, 3, 3, 2])

    metrics = compute_multiclass_metrics(y_true, y_pred, class_names=["N", "S", "V", "F"])

    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (4, 4)
    assert cm.sum() == len(y_true)
    assert metrics["per_class"]["F"]["support"] == 3


def test_predict_classes_returns_argmax_over_logits():
    torch.manual_seed(0)
    model = ECGCNN(num_classes=4)
    model.eval()

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 250)).astype(np.float32)
    y = rng.integers(0, 4, size=20).astype(np.int64)
    loader = DataLoader(ECGBeatDataset(X, y, label_dtype=torch.long), batch_size=8, shuffle=False)

    preds, targets, loss = predict_classes(model, loader, torch.device("cpu"))

    assert preds.shape == (20,)
    assert set(preds.tolist()) <= {0, 1, 2, 3}
    assert np.array_equal(targets, y)
    assert loss is None


def test_predict_classes_reports_loss_when_given_a_criterion():
    torch.manual_seed(0)
    model = ECGCNN(num_classes=4)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 250)).astype(np.float32)
    y = rng.integers(0, 4, size=8).astype(np.int64)
    loader = DataLoader(ECGBeatDataset(X, y, label_dtype=torch.long), batch_size=4, shuffle=False)

    _, _, loss = predict_classes(model, loader, torch.device("cpu"), loss_fn=torch.nn.CrossEntropyLoss())

    assert loss is not None and loss > 0


def test_attack_success_rate_is_zero_when_nothing_flips():
    y_true = np.array([0, 1, 1, 0])
    clean_preds = np.array([0, 1, 1, 0])
    adv_preds = np.array([0, 1, 1, 0])
    assert attack_success_rate(y_true, clean_preds, adv_preds) == 0.0


def test_attack_success_rate_counts_only_originally_correct_flips():
    y_true = np.array([0, 1, 1, 0])
    clean_preds = np.array([0, 1, 0, 0])  # sample 2 already wrong before the attack
    adv_preds = np.array([1, 0, 1, 0])  # samples 0 and 1 flip, sample 2 "flips" but wasn't correct to begin with

    # 3 originally-correct samples (0, 1, 3), 2 of them flip -> 2/3
    assert attack_success_rate(y_true, clean_preds, adv_preds) == pytest.approx(2 / 3)


def test_attack_success_rate_handles_no_correct_predictions():
    y_true = np.array([0, 1])
    clean_preds = np.array([1, 0])
    adv_preds = np.array([0, 1])
    assert attack_success_rate(y_true, clean_preds, adv_preds) == 0.0
