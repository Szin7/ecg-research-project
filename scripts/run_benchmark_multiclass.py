import argparse

from src.utils.tags import resolve_tag
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.attacks.pgd import pgd_attack
from src.data.cache import load_or_build_split
from src.data.dataset import class_distribution
from src.data.splits import get_split, map_symbol_to_aami_class
from src.data.torch_dataset import ECGBeatDataset
from src.evaluation.metrics import compute_multiclass_metrics
from src.evaluation.predict import predict_classes
from src.evaluation.robustness import attack_success_rate, perturbation_stats
from src.models.checkpoint import build_model
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.paths import ARTIFACTS_DIR

CACHE_SUFFIX = "aami"


def _load_aami_checkpoint(tag: str):
    ckpt = torch.load(ARTIFACTS_DIR / "models" / f"{tag}.pt", map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"]


def run_clean_evaluation(mode: str, tag: str = "aami_cnn") -> dict:
    """Experiment 1 (multi-class variant): AAMI CNN on clean, held-out test data."""
    dataset_cfg = load_config("dataset.yaml")
    device = get_device()

    split = get_split(mode, dataset_cfg)
    X_test, y_test, _ = load_or_build_split(
        mode, "test", split["test"], dataset_cfg, label_fn=map_symbol_to_aami_class, cache_suffix=CACHE_SUFFIX
    )

    model, class_names = _load_aami_checkpoint(tag)
    model.to(device)

    loader = DataLoader(ECGBeatDataset(X_test, y_test, label_dtype=torch.long), batch_size=256, shuffle=False)
    preds, targets, _ = predict_classes(model, loader, device)
    metrics = compute_multiclass_metrics(targets, preds, class_names)

    print(f"test beats: {len(y_test)} {class_distribution(y_test)}")
    print(f"AAMI clean test: acc={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f} weighted_f1={metrics['weighted_f1']:.3f}")
    for name in class_names:
        pc = metrics["per_class"][name]
        print(f"  {name}: precision={pc['precision']:.3f} recall={pc['recall']:.3f} f1={pc['f1']:.3f} support={pc['support']}")

    result = {
        "mode": mode,
        "tag": tag,
        "classes": class_names,
        "n_test_beats": len(y_test),
        "test_class_distribution": class_distribution(y_test),
        "metrics": metrics,
    }

    out_dir = ARTIFACTS_DIR / "experiments" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clean_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"saved -> {out_dir / 'clean_metrics.json'}")
    return result


def run_robustness_check(mode: str, tag: str = "aami_cnn", epsilons=(0.0, 0.1, 0.25, 0.4)) -> list:
    """A handful of PGD epsilon points, not the full 8-point binary sweep --
    this is a secondary experiment. Per-class recall is reported alongside
    macro-F1 and attack success rate, since an attack could plausibly wipe
    out one rare class's recall while barely denting overall accuracy.
    """
    dataset_cfg = load_config("dataset.yaml")
    exp_cfg = load_config("experiments.yaml")
    device = get_device()

    split = get_split(mode, dataset_cfg)
    X_test, y_test, _ = load_or_build_split(
        mode, "test", split["test"], dataset_cfg, label_fn=map_symbol_to_aami_class, cache_suffix=CACHE_SUFFIX
    )

    model, class_names = _load_aami_checkpoint(tag)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    pgd_cfg = exp_cfg["attack"]["pgd"]
    loader = DataLoader(ECGBeatDataset(X_test, y_test, label_dtype=torch.long), batch_size=256, shuffle=False)

    print(f"AAMI robustness check for {tag} on {len(y_test)} test beats")
    rows = []
    for epsilon in epsilons:
        clean_preds_all, adv_preds_all, targets_all, stat_batches = [], [], [], []

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            with torch.no_grad():
                clean_preds = model(xb).argmax(dim=-1)

            x_adv = xb if epsilon == 0.0 else pgd_attack(model, xb, yb, epsilon, pgd_cfg["alpha_ratio"] * epsilon, pgd_cfg["iterations"], loss_fn)

            with torch.no_grad():
                adv_preds = model(x_adv).argmax(dim=-1)

            clean_preds_all.append(clean_preds.cpu().numpy())
            adv_preds_all.append(adv_preds.cpu().numpy())
            targets_all.append(yb.cpu().numpy())
            if epsilon > 0.0:
                stat_batches.append(perturbation_stats(xb, x_adv))

        clean_preds = np.concatenate(clean_preds_all)
        adv_preds = np.concatenate(adv_preds_all)
        targets = np.concatenate(targets_all)

        adv_metrics = compute_multiclass_metrics(targets, adv_preds, class_names)
        agg_stats = {k: float(np.mean([s[k] for s in stat_batches])) for k in stat_batches[0]} if stat_batches else {}

        row = {
            "epsilon": epsilon,
            "robust_accuracy": adv_metrics["accuracy"],
            "robust_macro_f1": adv_metrics["macro_f1"],
            "attack_success_rate": attack_success_rate(targets, clean_preds, adv_preds),
            **{f"recall_{name}": adv_metrics["per_class"][name]["recall"] for name in class_names},
            **agg_stats,
        }
        rows.append(row)
        print(
            f"  eps={epsilon:.2f}  robust_acc={row['robust_accuracy']:.3f}  "
            f"robust_macro_f1={row['robust_macro_f1']:.3f}  attack_success={row['attack_success_rate']:.3f}"
        )

    out_dir = ARTIFACTS_DIR / "experiments" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "robustness_curve.csv", index=False)
    print(f"saved -> {out_dir / 'robustness_curve.csv'}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate the ECG Research Project multi-class AAMI CNN")
    parser.add_argument("--mode", choices=["demo", "research"], default="demo")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    args.tag = resolve_tag(args.mode, args.tag, 'aami_cnn')
    run_clean_evaluation(mode=args.mode, tag=args.tag)
    run_robustness_check(mode=args.mode, tag=args.tag)


if __name__ == "__main__":
    main()
