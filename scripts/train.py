import argparse

from src.utils.tags import resolve_tag
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.cache import load_or_build_split
from src.data.dataset import class_distribution, compute_pos_weight
from src.data.splits import get_split
from src.data.torch_dataset import ECGBeatDataset
from src.evaluation.metrics import best_f1_threshold, compute_metrics
from src.evaluation.predict import predict_probs
from src.models.checkpoint import build_model
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.paths import ARTIFACTS_DIR
from src.utils.seed import set_seed


def evaluate(model, loader, device, loss_fn):
    """Threshold is calibrated on whatever data is passed in -- callers
    must only ever pass validation data here, never test data, or the
    threshold stops being an honest out-of-sample choice."""
    probs, targets, loss = predict_probs(model, loader, device, loss_fn)
    threshold = best_f1_threshold(targets, probs)
    metrics = compute_metrics(targets, (probs >= threshold).astype(int), probs)
    metrics["loss"] = loss
    metrics["threshold"] = threshold
    return metrics


def train(mode: str = "demo", tag: str = "baseline_cnn", extra_batches=None) -> dict:
    """extra_batches: optional callable(xb, yb, model, loss_fn) -> (xb2, yb2)
    used by adversarial training to mix in adversarial examples each step,
    without duplicating this whole loop in src/defenses/adversarial_training.py.
    Receives loss_fn so attacks are crafted against the same pos_weight-
    reweighted objective the model is actually being trained on."""
    dataset_cfg = load_config("dataset.yaml")
    training_cfg = load_config("training.yaml")
    set_seed(training_cfg["seed"])
    device = get_device()

    split = get_split(mode, dataset_cfg)
    X_train, y_train, _ = load_or_build_split(mode, "train", split["train"], dataset_cfg)
    X_val, y_val, _ = load_or_build_split(mode, "val", split["val"], dataset_cfg)

    print(f"train beats: {len(y_train)} {class_distribution(y_train)}")
    print(f"val beats:   {len(y_val)} {class_distribution(y_val)}")

    train_cfg = training_cfg["train"]
    train_loader = DataLoader(ECGBeatDataset(X_train, y_train), batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(ECGBeatDataset(X_val, y_val), batch_size=256, shuffle=False)

    model = build_model(training_cfg["model"]).to(device)
    pos_weight = torch.tensor(compute_pos_weight(y_train), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim_cfg = training_cfg["optim"]
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=optim_cfg["scheduler_patience"], factor=optim_cfg["scheduler_factor"]
    )

    best_val_f1 = -1.0
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
            if extra_batches is not None:
                xb, yb = extra_batches(xb, yb, model, loss_fn)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, device, loss_fn)
        scheduler.step(val_metrics["f1"])

        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), **val_metrics})
        print(
            f"epoch {epoch:02d}  train_loss={np.mean(train_losses):.4f}  "
            f"val_f1={val_metrics['f1']:.4f}  val_loss={val_metrics['loss']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": training_cfg["model"],
                    "threshold": val_metrics["threshold"],
                },
                ckpt_path,
            )
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
                "seed": training_cfg["seed"],
                "best_val_f1": best_val_f1,
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

    print(f"done in {elapsed:.1f}s, best val_f1={best_val_f1:.4f}, checkpoint -> {ckpt_path}")
    return {"best_val_f1": best_val_f1, "checkpoint": str(ckpt_path)}


def main():
    parser = argparse.ArgumentParser(description="Train the ECG Research Project baseline CNN")
    parser.add_argument("--mode", choices=["demo", "research"], default="demo")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--defense", choices=["none", "adversarial_training"], default="none")
    args = parser.parse_args()
    args.tag = resolve_tag(args.mode, args.tag, ("defended_cnn" if args.defense == "adversarial_training" else "baseline_cnn"))

    extra_batches = None
    if args.defense == "adversarial_training":
        from src.defenses.adversarial_training import make_adversarial_batch_fn

        adv_cfg = load_config("experiments.yaml")["adversarial_training"]
        extra_batches = make_adversarial_batch_fn(
            attack=adv_cfg["attack"],
            epsilon=adv_cfg["epsilon"],
            clean_ratio=adv_cfg["clean_ratio"],
            alpha_ratio=adv_cfg["alpha_ratio"],
            iterations=adv_cfg["iterations"],
        )

    train(mode=args.mode, tag=args.tag, extra_batches=extra_batches)


if __name__ == "__main__":
    main()
