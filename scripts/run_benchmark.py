import argparse

from src.utils.tags import resolve_tag
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.attacks.pgd import pgd_attack
from src.data.cache import load_or_build_split
from src.data.dataset import class_distribution
from src.data.splits import get_split
from src.data.torch_dataset import ECGBeatDataset
from src.defenses.input_smoothing import SmoothedModel
from src.evaluation.metrics import best_f1_threshold, compute_metrics
from src.evaluation.predict import predict_probs
from src.evaluation.robustness import run_attack_sweep
from src.models.baseline_features import extract_feature_matrix
from src.models.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.paths import ARTIFACTS_DIR


def evaluate_cnn_on_split(model, threshold, X, y, device) -> dict:
    loader = DataLoader(ECGBeatDataset(X, y), batch_size=256, shuffle=False)
    probs, targets, _ = predict_probs(model, loader, device)
    return compute_metrics(targets, (probs >= threshold).astype(int), probs)


def run_baseline_comparison(X_train, y_train, X_val, y_val, X_test, y_test, seed=42) -> dict:
    """LogisticRegression + RandomForest on hand-engineered features, for
    comparison against the CNN's learned waveform representation. Threshold
    is calibrated on val the same way as the CNN, for a fair comparison."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    feat_train = extract_feature_matrix(X_train)
    feat_val = extract_feature_matrix(X_val)
    feat_test = extract_feature_matrix(X_test)

    scaler = StandardScaler().fit(feat_train)
    logreg = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    logreg.fit(scaler.transform(feat_train), y_train)

    forest = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)
    forest.fit(feat_train, y_train)

    results = {}
    for name, clf, transform in [
        ("logistic_regression", logreg, scaler.transform),
        ("random_forest", forest, lambda x: x),
    ]:
        val_probs = clf.predict_proba(transform(feat_val))[:, 1]
        threshold = best_f1_threshold(y_val, val_probs)
        test_probs = clf.predict_proba(transform(feat_test))[:, 1]
        metrics = compute_metrics(y_test, (test_probs >= threshold).astype(int), test_probs)
        metrics["threshold"] = threshold
        results[name] = metrics

    return results


def run_clean_evaluation(mode: str, tag: str = "baseline_cnn") -> dict:
    """Experiment 1: baseline CNN on clean, held-out test data, plus the
    classical-feature baselines for comparison."""
    dataset_cfg = load_config("dataset.yaml")
    device = get_device()

    split = get_split(mode, dataset_cfg)
    X_train, y_train, _ = load_or_build_split(mode, "train", split["train"], dataset_cfg)
    X_val, y_val, _ = load_or_build_split(mode, "val", split["val"], dataset_cfg)
    X_test, y_test, _ = load_or_build_split(mode, "test", split["test"], dataset_cfg)

    ckpt_path = ARTIFACTS_DIR / "models" / f"{tag}.pt"
    loaded = load_checkpoint(ckpt_path)

    loaded.model.to(device)

    print(f"test beats: {len(y_test)} {class_distribution(y_test)}")
    cnn_metrics = evaluate_cnn_on_split(loaded.model, loaded.threshold, X_test, y_test, device)
    print(
        f"CNN clean test: acc={cnn_metrics['accuracy']:.3f} f1={cnn_metrics['f1']:.3f} "
        f"precision={cnn_metrics['precision']:.3f} recall={cnn_metrics['recall']:.3f} "
        f"pr_auc={cnn_metrics['pr_auc']:.3f} roc_auc={cnn_metrics['roc_auc']:.3f}"
    )

    baseline_metrics = run_baseline_comparison(X_train, y_train, X_val, y_val, X_test, y_test)
    for name, m in baseline_metrics.items():
        print(f"{name}: f1={m['f1']:.3f} precision={m['precision']:.3f} recall={m['recall']:.3f}")

    result = {
        "mode": mode,
        "tag": tag,
        "threshold": loaded.threshold,
        "n_test_beats": len(y_test),
        "test_class_distribution": class_distribution(y_test),
        "cnn": cnn_metrics,
        "baselines": baseline_metrics,
    }

    out_dir = ARTIFACTS_DIR / "experiments" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clean_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"saved -> {out_dir / 'clean_metrics.json'}")
    return result


def run_robustness_benchmark(mode: str, tag: str = "baseline_cnn") -> list:
    """Experiments 2/3: FGSM and PGD across the configured epsilon sweep.
    Aggregate curves only -- individual adversarial waveforms live in
    attack_examples.csv/attack_waveforms.npz from generate_attacks.py."""
    dataset_cfg = load_config("dataset.yaml")
    exp_cfg = load_config("experiments.yaml")
    device = get_device()

    split = get_split(mode, dataset_cfg)
    X_test, y_test, _ = load_or_build_split(mode, "test", split["test"], dataset_cfg)

    loaded = load_checkpoint(ARTIFACTS_DIR / "models" / f"{tag}.pt")

    loaded.model.to(device)

    print(f"robustness sweep for {tag} on {len(y_test)} test beats")
    all_results = []
    for attack in ["fgsm", "pgd"]:
        attack_cfg = exp_cfg["attack"][attack]
        results = run_attack_sweep(
            loaded.model, loaded.threshold, X_test, y_test, attack=attack,
            epsilons=attack_cfg["epsilons"], device=device,
            alpha_ratio=attack_cfg.get("alpha_ratio", 0.25),
            iterations=attack_cfg.get("iterations", 1),
        )
        for r in results:
            print(
                f"  {attack:4s} eps={r['epsilon']:.3f}  robust_acc={r['robust_accuracy']:.3f}  "
                f"adv_f1={r['adversarial_f1']:.3f}  attack_success={r['attack_success_rate']:.3f}"
            )
        all_results.extend(results)

    out_dir = ARTIFACTS_DIR / "experiments" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(out_dir / "robustness_curve.csv", index=False)
    print(f"saved -> {out_dir / 'robustness_curve.csv'}")
    return all_results


def run_secondary_defense_comparison(mode: str, tag: str = "baseline_cnn", epsilons=(0.1, 0.25, 0.4)) -> list:
    """Attack the plain and smoothed pipelines independently; include clean cost."""
    dataset_cfg = load_config("dataset.yaml")
    exp_cfg = load_config("experiments.yaml")
    device = get_device()
    kernel_size = exp_cfg["defense_secondary"]["kernel_size"]
    pgd_cfg = exp_cfg["attack"]["pgd"]

    split = get_split(mode, dataset_cfg)
    X_test, y_test, _ = load_or_build_split(mode, "test", split["test"], dataset_cfg)
    loaded = load_checkpoint(ARTIFACTS_DIR / "models" / f"{tag}.pt")
    loaded.model.to(device)
    defended = SmoothedModel(loaded.model, kernel_size).to(device).eval()
    # epsilon=0 measures the clean cost of the same complete pipeline.
    budgets = sorted({0.0, *epsilons})
    rows = []
    for name, pipeline in [("plain", loaded.model), ("adaptive_smoothing", defended)]:
        results = run_attack_sweep(
            pipeline, loaded.threshold, X_test, y_test, "pgd", budgets, device,
            alpha_ratio=pgd_cfg["alpha_ratio"], iterations=pgd_cfg["iterations"],
        )
        rows.extend({"pipeline": name, **result} for result in results)
    print("Adaptive PGD attacks each complete pipeline; threshold remains frozen.")

    out_dir = ARTIFACTS_DIR / "experiments" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "secondary_defense_adaptive.csv", index=False)
    print(f"saved -> {out_dir / 'secondary_defense_adaptive.csv'}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run ECG Research Project evaluation experiments")
    parser.add_argument("--mode", choices=["demo", "research"], default="demo")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--secondary-defense", action="store_true")
    args = parser.parse_args()
    args.tag = resolve_tag(args.mode, args.tag, 'baseline_cnn')

    run_clean_evaluation(mode=args.mode, tag=args.tag)
    if not args.skip_robustness:
        run_robustness_benchmark(mode=args.mode, tag=args.tag)
    if args.secondary_defense:
        run_secondary_defense_comparison(mode=args.mode, tag=args.tag)


if __name__ == "__main__":
    main()
