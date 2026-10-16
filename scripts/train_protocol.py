"""Train one immutable run, using only training and validation data."""

import argparse
import json
import time
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from src.attacks.pgd import pgd_attack
from src.data.protocol import PROTOCOL_ID, assert_subject_disjoint, sha256_file
from src.evaluation.metrics import (
    best_f1_threshold,
    compute_metrics,
    compute_multiclass_metrics,
)
from src.models.checkpoint import build_model
from src.utils.experiment import begin_run, finish_run
from src.utils.paths import DATA_DIR, ARTIFACTS_DIR, CONFIGS_DIR
from src.utils.seed import set_seed


def train(variant, seed, run_id):
    cfg = yaml.safe_load((CONFIGS_DIR / "study_v2.yaml").read_text())
    model_cfg = yaml.safe_load((CONFIGS_DIR / "training.yaml").read_text())["model"]
    multiclass = variant == "aami_context"
    model_cfg.update(
        waveform_samples=250,
        rr_features=0 if variant == "morphology" else 4,
        num_classes=4 if multiclass else 1,
    )
    root = DATA_DIR / "processed" / PROTOCOL_ID
    manifest = json.loads((root / "manifest.json").read_text())
    assert_subject_disjoint(manifest["config"]["records"])
    arrays = {}
    for split in ["train", "val"]:
        if sha256_file(root / f"{split}.npz") != manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split} digest mismatch")
        with np.load(root / f"{split}.npz", allow_pickle=False) as data:
            arrays[split] = (
                torch.from_numpy(data["X"]).unsqueeze(1),
                torch.from_numpy(data["y_aami"] if multiclass else data["y"]).to(
                    torch.long if multiclass else torch.float32
                ),
            )
    set_seed(seed)
    out = ARTIFACTS_DIR / "runs" / run_id
    record = begin_run(
        out,
        {
            "study": cfg,
            "variant": variant,
            "seed": seed,
            "model": model_cfg,
            "multiclass_weighting": "inverse_square_root_frequency"
            if multiclass
            else None,
        },
        manifest,
    )
    loader = DataLoader(
        TensorDataset(*arrays["train"]), batch_size=cfg["batch_size"], shuffle=True
    )
    validation = DataLoader(TensorDataset(*arrays["val"]), batch_size=512)
    model = build_model(model_cfg)
    labels = arrays["train"][1]
    if multiclass:
        counts = torch.bincount(labels, minlength=4).clamp_min(1)
        weights = (counts.sum() / counts).sqrt()
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights / weights.mean())
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=(len(labels) - labels.sum()) / labels.sum()
        )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    best, stale, history = -1, 0, []
    start = time.perf_counter()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        total = 0
        for xb, yb in loader:
            if variant == "context_defended":
                n = len(xb) // 2
                attack = cfg["training_pgd"]
                adv = pgd_attack(
                    model,
                    xb[n:],
                    yb[n:],
                    attack["epsilon"],
                    attack["epsilon"] / 3,
                    attack["steps"],
                    loss_fn,
                )
                xb = torch.cat([xb[:n], adv])
                model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
        model.eval()
        parts = []
        with torch.inference_mode():
            for xb, _ in validation:
                parts.append(model(xb))
        logits = torch.cat(parts)
        truth = arrays["val"][1].numpy()
        if multiclass:
            metrics = compute_multiclass_metrics(
                truth, logits.argmax(1).numpy(), ["N", "S", "V", "F"]
            )
            score, threshold = metrics["macro_f1"], None
        else:
            probs = logits.sigmoid().numpy()
            threshold = best_f1_threshold(truth, probs)
            metrics = compute_metrics(truth, (probs >= threshold).astype(int), probs)
            score = metrics["f1"]
        scheduler.step(score)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(labels),
                "val_score": score,
                "threshold": threshold,
                "val_metrics": metrics,
            }
        )
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(
            run_id,
            epoch,
            round(score, 4),
            round(time.perf_counter() - start, 1),
            flush=True,
        )
        if score > best:
            best, stale = score, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": model_cfg,
                    "threshold": threshold,
                    "protocol_id": PROTOCOL_ID,
                    "seed": seed,
                    "variant": variant,
                    "best_epoch": epoch,
                    "best_val_score": best,
                    "run_id": run_id,
                },
                out / "model.pt",
            )
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    finish_run(
        out,
        record,
        {
            "best_val_score": best,
            "epochs": len(history),
            "train_seconds": time.perf_counter() - start,
            "n_params": sum(p.numel() for p in model.parameters()),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["morphology", "context", "context_defended", "aami_context"],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    train(args.variant, args.seed, args.run_id)
