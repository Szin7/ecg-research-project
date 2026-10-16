"""Verify saved-study checksums and independently recompute reported metrics."""

import json

import numpy as np
import pandas as pd
import torch

from src.data.protocol import sha256_file
from src.evaluation.metrics import compute_metrics, compute_multiclass_metrics
from src.utils.paths import ARTIFACTS_DIR


def verify():
    active = json.loads((ARTIFACTS_DIR / "active_study.json").read_text())
    study = ARTIFACTS_DIR / active["study_dir"]

    for filename, key in [
        ("dataset_manifest.json", "dataset_manifest_sha256"),
        ("frozen_evaluation_config.yaml", "evaluation_config_sha256"),
        ("audit_indices.npy", "audit_indices_sha256"),
    ]:
        assert sha256_file(study / filename) == active[key], filename

    seeds = pd.read_csv(study / "seed_results.csv")
    assert len(seeds) == 12

    for row in seeds.itertuples():
        # Use an exact filename. A wildcard for context can also match
        # aami_context, which is a different model with no binary threshold.
        selected_run = active["selected_runs"][row.variant]
        prefix, selected_seed = selected_run.rsplit(
            f"_{row.variant}_s", 1
        )

        if not prefix or not selected_seed.isdigit():
            raise ValueError(f"Invalid selected run: {selected_run}")

        run = f"{prefix}_{row.variant}_s{row.seed}"
        prediction = study / f"{run}_predictions.npz"

        record = json.loads(
            (ARTIFACTS_DIR / "runs" / run / "manifest.json").read_text()
        )

        assert record["status"] == "completed"
        assert record["config"]["variant"] == row.variant
        assert record["config"]["seed"] == row.seed
        assert (
            sha256_file(ARTIFACTS_DIR / "runs" / run / "model.pt")
            == record["checkpoint_sha256"]
        )

        with np.load(prediction, allow_pickle=False) as data:
            if row.variant == "aami_context":
                actual = compute_multiclass_metrics(
                    data["y_aami"],
                    data["pred"],
                    ["N", "S", "V", "F"],
                )
                assert abs(actual["macro_f1"] - row.macro_f1) < 1e-10

            else:
                clean = json.loads(
                    (study / f"{run}_clean.json").read_text()
                )
                probabilities = (
                    torch.from_numpy(data["logits"]).sigmoid().numpy()
                )

                assert np.array_equal(
                    data["pred"],
                    (probabilities >= clean["threshold"]).astype(int),
                )

                actual = compute_metrics(
                    data["y"], data["pred"], probabilities
                )
                assert abs(actual["f1"] - row.f1) < 1e-10
                assert abs(actual["pr_auc"] - row.pr_auc) < 1e-10

    for variant, run in active["selected_runs"].items():
        checkpoint = ARTIFACTS_DIR / "runs" / run / "model.pt"
        assert sha256_file(checkpoint) == sha256_file(
            ARTIFACTS_DIR / "models" / f"v2_{variant}.pt"
        )

    n_attack_rows = 0

    for pipeline in [
        "morphology",
        "context",
        "context_defended",
        "context_smoothed",
    ]:
        folder = study / f"attacks_{pipeline}"
        record = json.loads((folder / "manifest.json").read_text())

        assert record["status"] == "completed"
        assert record["results_sha256"] == sha256_file(
            folder / "metrics.csv"
        )

        for row in pd.read_csv(folder / "metrics.csv").itertuples():
            prediction_path = (
                folder / f"{row.scope}_eps{row.epsilon:g}.npz"
            )

            with np.load(prediction_path, allow_pickle=False) as data:
                pred = (
                    data["probabilities"] >= record["threshold"]
                ).astype(int)
                clean = (
                    data["clean_probabilities"] >= record["threshold"]
                ).astype(int)

                actual = compute_metrics(
                    data["y"], pred, data["probabilities"]
                )

                assert abs(
                    actual["f1"] - row.adversarial_f1
                ) < 1e-10
                assert abs(
                    actual["accuracy"] - row.robust_accuracy
                ) < 1e-10
                assert abs(
                    actual["recall"] - row.adversarial_recall
                ) < 1e-10
                assert abs(
                    actual["precision"] - row.adversarial_precision
                ) < 1e-10

                assert not np.any(
                    (clean != data["y"]) & (pred == data["y"])
                ), "Attack repaired a clean error"

                assert row.max_linf <= row.epsilon + 1e-5
                assert len(data["y"]) == row.n_beats

            n_attack_rows += 1

    result = {
        "training_runs_verified": len(seeds),
        "attack_rows_verified": n_attack_rows,
        "status": "passed",
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()