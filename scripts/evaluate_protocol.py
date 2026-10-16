"""Report every seed; publish aliases selected using validation scores only."""

import argparse
import json
import shutil
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
import yaml
from src.data.protocol import PROTOCOL_ID, sha256_file
from src.evaluation.metrics import compute_metrics, compute_multiclass_metrics
from src.evaluation.uncertainty import subject_bootstrap
from src.models.checkpoint import load_checkpoint
from src.utils.paths import ARTIFACTS_DIR, DATA_DIR, CONFIGS_DIR


def evaluate(name, run_prefix):
    output = ARTIFACTS_DIR / "studies" / name
    output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((CONFIGS_DIR / "study_v2.yaml").read_text())
    (output / "frozen_evaluation_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False)
    )
    root = DATA_DIR / "processed" / PROTOCOL_ID
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["splits"]["test"]["sha256"] != sha256_file(root / "test.npz"):
        raise ValueError("Test digest mismatch")
    shutil.copy2(root / "manifest.json", output / "dataset_manifest.json")
    data = np.load(root / "test.npz", allow_pickle=False)
    X, y, ya, groups = (data[k] for k in ["X", "y", "y_aami", "groups"])
    rows = []
    selected = {}
    for variant in cfg["variants"]:
        candidates = []
        for seed in cfg["seeds"]:
            directory = ARTIFACTS_DIR / "runs" / f"{run_prefix}_{variant}_s{seed}"
            record = json.loads((directory / "manifest.json").read_text())
            if record["status"] != "completed" or record[
                "checkpoint_sha256"
            ] != sha256_file(directory / "model.pt"):
                raise ValueError("Invalid run")
            candidates.append((record["result"]["best_val_score"], directory, seed))
        selected[variant] = max(candidates, key=lambda row: row[0])[1].name
        for val, directory, seed in candidates:
            loaded = load_checkpoint(directory / "model.pt")
            parts = []
            with torch.inference_mode():
                for start in range(0, len(y), 512):
                    parts.append(
                        loaded.model(
                            torch.from_numpy(X[start : start + 512]).unsqueeze(1)
                        ).numpy()
                    )
            logits = np.concatenate(parts)
            result = {
                "mode": "research",
                "tag": directory.name,
                "n_test_beats": len(y),
                "protocol_id": PROTOCOL_ID,
                "selected_run": selected[variant],
            }
            row = {"variant": variant, "seed": seed, "val_score": val}
            if variant == "aami_context":
                pred = logits.argmax(1)
                metrics = compute_multiclass_metrics(ya, pred, ["N", "S", "V", "F"])
                result.update(classes=["N", "S", "V", "F"], metrics=metrics)
                row.update(accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"])
                target = ya
            else:
                probs = torch.from_numpy(logits).sigmoid().numpy()
                pred = (probs >= loaded.threshold).astype(int)
                metrics = compute_metrics(y, pred, probs)
                result.update(
                    cnn=metrics,
                    threshold=loaded.threshold,
                    test_class_distribution={
                        str(c): int(np.sum(y == c)) for c in [0, 1]
                    },
                    subject_bootstrap_95=subject_bootstrap(
                        y, pred, groups, cfg["evaluation"]["bootstrap_repetitions"]
                    ),
                )
                row.update(
                    {k: v for k, v in metrics.items() if k != "confusion_matrix"}
                )
                target = y
            rows.append(row)
            (output / f"{directory.name}_clean.json").write_text(
                json.dumps(result, indent=2)
            )
            np.savez_compressed(
                output / f"{directory.name}_predictions.npz",
                y=y,
                y_aami=ya,
                pred=pred,
                logits=logits,
                groups=groups,
            )
            errors = []
            for group in np.unique(groups):
                for cls in np.unique(target[groups == group]):
                    mask = (groups == group) & (target == cls)
                    errors.append(
                        {
                            "record": int(group),
                            "class": int(cls),
                            "support": int(mask.sum()),
                            "correct": int(np.sum(pred[mask] == cls)),
                            "recall": float(np.mean(pred[mask] == cls)),
                        }
                    )
            pd.DataFrame(errors).to_csv(
                output / f"{directory.name}_errors.csv", index=False
            )
            if directory.name == selected[variant]:
                alias = f"v2_{variant}"
                shutil.copy2(
                    directory / "model.pt", ARTIFACTS_DIR / "models" / f"{alias}.pt"
                )
                destination = ARTIFACTS_DIR / "experiments" / alias
                destination.mkdir(exist_ok=True)
                result["tag"] = alias
                (destination / "clean_metrics.json").write_text(
                    json.dumps(result, indent=2)
                )
            print(row, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "seed_results.csv", index=False)
    frame.groupby("variant").agg(
        {
            k: ["mean", "std"]
            for k in ["f1", "recall", "precision", "accuracy", "pr_auc", "macro_f1"]
        }
    ).to_csv(output / "seed_summary.csv")
    rng = np.random.default_rng(cfg["evaluation"]["subset_seed"])
    indices = np.concatenate(
        [
            rng.choice(
                np.where(groups == g)[0],
                size=min(
                    cfg["evaluation"]["audit_beats_per_record"], np.sum(groups == g)
                ),
                replace=False,
            )
            for g in np.unique(groups)
        ]
    )
    np.save(output / "audit_indices.npy", indices)
    preview = DATA_DIR / "preview"
    preview.mkdir(exist_ok=True)
    np.savez_compressed(
        preview / "v2_test.npz", **{k: data[k][indices] for k in data.files}
    )
    selection = {
        "protocol_id": PROTOCOL_ID,
        "selected_runs": selected,
        "selection_rule": "highest validation score; test scores not used",
        "evaluation_config_sha256": sha256_file(
            output / "frozen_evaluation_config.yaml"
        ),
        "dataset_manifest_sha256": sha256_file(root / "manifest.json"),
        "audit_indices_sha256": sha256_file(output / "audit_indices.npy"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2))
    (ARTIFACTS_DIR / "active_study.json").write_text(
        json.dumps(
            {"study_dir": str(output.relative_to(ARTIFACTS_DIR)), **selection}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="v2_20261016")
    parser.add_argument("--run-prefix", default="v2")
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate(args.name, args.run_prefix)
