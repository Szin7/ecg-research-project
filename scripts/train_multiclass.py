import argparse

from src.utils.tags import resolve_tag
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.cache import load_or_build_split
from src.data.dataset import class_distribution, compute_class_weights
from src.data.splits import get_split, map_symbol_to_aami_class
from src.data.torch_dataset import ECGBeatDataset
from src.evaluation.metrics import compute_multiclass_metrics
from src.evaluation.predict import predict_classes
from src.models.checkpoint import build_model
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.paths import ARTIFACTS_DIR
from src.utils.seed import set_seed

CACHE_SUFFIX = "aami"


def evaluate(model, loader, device, loss_fn, class_names):
    preds, targets, loss = predict_classes(model, loader, device, loss_fn)
    metrics = compute_multiclass_metrics(targets, preds, class_names)
    metrics["loss"] = loss
    return metrics


def train(mode: str = "demo", tag: str = "aami_cnn") -> dict:
    """Secondary experiment: full 4-class AAMI classification (N/S/V/F,
    Q excluded) instead of the binary Normal-vs-Arrhythmia task. Deliberately
    a separate script rather than a --task flag on scripts/train.py: the
    loss, prediction rule (argmax vs. a calibrated threshold), and tracked
    metric (macro-F1 vs. F1) differ enough that branching one function
    for both would read worse than two short, direct ones.
    """
    dataset_cfg = load_config("dataset.yaml")
    training_cfg = load_config("training_multiclass.yaml")
    class_names = training_cfg["classes"]
    set_seed(training_cfg["seed"])
    device = get_device()

    split = get_split(mode, dataset_cfg)
    X_train, y_train, _ = load_or_build_split(
        mode, "train", split["train"], dataset_cfg, label_fn=map_symbol_to_aami_class, cache_suffix=CACHE_SUFFIX
    )
    X_val, y_val, _ = load_or_build_split(
        mode, "val", split["val"], dataset_cfg, label_fn=map_symbol_to_aami_class, cache_suffix=CACHE_SUFFIX
    )

    print(f"train beats: {len(y_train)} {class_distribution(y_train)}")
    print(f"val beats:   {len(y_val)} {class_distribution(y_val)}")

    train_cfg = training_cfg["train"]
    train_loader = DataLoader(
        ECGBeatDataset(X_train, y_train, label_dtype=torch.long), batch_size=train_cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(ECGBeatDataset(X_val, y_val, label_dtype=torch.long), batch_size=256, shuffle=False)

    model = build_model(training_cfg["model"]).to(device)
    class_weights = torch.tensor(compute_class_weights(y_train, len(class_names)), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optim_cfg = training_cfg["optim"]
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=optim_cfg["scheduler_patience"], factor=optim_cfg["scheduler_factor"]
    )

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    ckpt_dir = ARTIFACTS_DIR / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{tag}.pt"

    start = time.time()
    for epoch in range(train_cfg["max_epochs"]):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, device, loss_fn, class_names)
        scheduler.step(val_metrics["macro_f1"])

        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), **val_metrics})
        print(
            f"epoch {epoch:02d}  train_loss={np.mean(train_losses):.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}  val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save({"state_dict": model.state_dict(), "model_config": training_cfg["model"], "classes": class_names}, ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start
    (ckpt_dir / f"{tag}_history.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "tag": tag,
                "task": "aami_multiclass",
                "classes": class_names,
                "seed": training_cfg["seed"],
                "best_val_macro_f1": best_val_macro_f1,
                "epochs_trained": len(history),
                "train_seconds": elapsed,
                "n_params": sum(p.numel() for p in model.parameters()),
                "train_class_distribution": class_distribution(y_train),
                "val_class_distribution": class_distribution(y_val),
                "history": history,
            },
            indent=2,
        )
    )

    print(f"done in {elapsed:.1f}s, best val_macro_f1={best_val_macro_f1:.4f}, checkpoint -> {ckpt_path}")
    return {"best_val_macro_f1": best_val_macro_f1, "checkpoint": str(ckpt_path)}


def main():
    parser = argparse.ArgumentParser(description="Train the ECG Research Project multi-class AAMI CNN")
    parser.add_argument("--mode", choices=["demo", "research"], default="demo")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    args.tag = resolve_tag(args.mode, args.tag, 'aami_cnn')
    train(mode=args.mode, tag=args.tag)


if __name__ == "__main__":
    main()
