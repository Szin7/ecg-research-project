"""Adaptive full-test attack point and separate record-balanced strength audit."""

import argparse
import json
import shutil
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
import yaml
from src.attacks.strong_pgd import worst_case_pgd
from src.data.protocol import PROTOCOL_ID, sha256_file
from src.defenses.input_smoothing import SmoothedModel
from src.evaluation.metrics import compute_metrics
from src.evaluation.robustness import attack_success_rate
from src.models.checkpoint import load_checkpoint
from src.utils.paths import ARTIFACTS_DIR, DATA_DIR


def run(variant, smoothing=False):
    active = json.loads((ARTIFACTS_DIR / "active_study.json").read_text())
    study = ARTIFACTS_DIR / active["study_dir"]
    cfg = yaml.safe_load((study / "frozen_evaluation_config.yaml").read_text())[
        "evaluation"
    ]
    name = variant + ("_smoothed" if smoothing else "")
    out = study / f"attacks_{name}"
    out.mkdir(exist_ok=False)
    tag = f"v2_{variant}"
    checkpoint = ARTIFACTS_DIR / "models" / f"{tag}.pt"
    loaded = load_checkpoint(checkpoint)
    model = SmoothedModel(loaded.model, 5) if smoothing else loaded.model
    model.eval()
    data = np.load(
        DATA_DIR / "processed" / PROTOCOL_ID / "test.npz", allow_pickle=False
    )
    X, y = data["X"], data["y"]
    audit = np.load(study / "audit_indices.npy")
    spec = {
        "variant": variant,
        "smoothing": smoothing,
        "selected_run": active["selected_runs"][variant],
        "checkpoint_sha256": sha256_file(checkpoint),
        "threshold": loaded.threshold,
        "config": cfg,
        "audit_indices_sha256": sha256_file(study / "audit_indices.npy"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    (out / "manifest.json").write_text(json.dumps(spec, indent=2))
    rows = []
    for scope, indices, budgets in [
        ("full_test", np.arange(len(y)), [0.0, *cfg["full_test_epsilons"]]),
        ("record_balanced_audit", audit, cfg["epsilons"]),
    ]:
        clean_parts = []
        probs = {eps: [] for eps in budgets}
        strong_parts = []
        norms = {eps: 0.0 for eps in budgets}
        strong_norm = 0.0
        for start in range(0, len(indices), 512):
            idx = indices[start : start + 512]
            xb = torch.from_numpy(X[idx]).unsqueeze(1)
            yb = torch.from_numpy(y[idx]).float()
            with torch.inference_mode():
                clean = torch.sigmoid(model(xb)).numpy()
            clean_parts.append(clean)
            previous = None
            for eps in budgets:
                if eps == 0:
                    adv = xb.clone()
                else:
                    adv = worst_case_pgd(
                        model,
                        xb,
                        yb,
                        eps,
                        eps * 0.1,
                        cfg["pgd_steps"],
                        cfg["pgd_starts"],
                        loaded.threshold,
                        cfg["attack_seed"] + start,
                        initial=previous,
                    )
                    adv = worst_case_pgd(
                        model,
                        xb,
                        yb,
                        eps,
                        eps * 0.1,
                        cfg["pgd_steps"],
                        1,
                        loaded.threshold,
                        cfg["attack_seed"] + start,
                        initial=adv,
                        objective="bce",
                    )
                previous = adv
                with torch.inference_mode():
                    prob = torch.sigmoid(model(adv)).numpy()
                probs[eps].append(prob)
                norms[eps] = max(
                    norms[eps], float((adv[..., :250] - xb[..., :250]).abs().max())
                )
                assert norms[eps] <= eps + 1e-5 and torch.equal(
                    adv[..., 250:], xb[..., 250:]
                )
                if scope == "record_balanced_audit" and eps == 0.1:
                    strong = worst_case_pgd(
                        model,
                        xb,
                        yb,
                        eps,
                        eps * 0.04,
                        cfg["strength_steps"],
                        cfg["pgd_starts"],
                        loaded.threshold,
                        cfg["attack_seed"] + 100000 + start,
                        initial=adv,
                    )
                    strong_norm = max(strong_norm, float((strong - xb).abs().max()))
                    assert strong_norm <= eps + 1e-5 and torch.equal(
                        strong[..., 250:], xb[..., 250:]
                    )
                    with torch.inference_mode():
                        strong_parts.append(torch.sigmoid(model(strong)).numpy())
                    if start == 0:
                        np.savez_compressed(
                            out / "example_waveforms.npz",
                            original=xb[:12, 0, :250].numpy(),
                            adversarial=adv[:12, 0, :250].numpy(),
                        )
                        pd.DataFrame(
                            [
                                {
                                    "beat_index": int(i),
                                    "true_label": int(y[i]),
                                    "attack": "pgd",
                                    "epsilon": eps,
                                    "clean_prob": float(c),
                                    "clean_pred": int(c >= loaded.threshold),
                                    "adv_prob": float(p),
                                    "adv_pred": int(p >= loaded.threshold),
                                    "flipped": bool(
                                        (c >= loaded.threshold)
                                        != (p >= loaded.threshold)
                                    ),
                                }
                                for i, c, p in zip(idx[:12], clean[:12], prob[:12])
                            ]
                        ).to_csv(out / "attack_examples.csv", index=False)
            if start % 5120 == 0:
                print(name, scope, start, "/", len(indices), flush=True)
        clean = np.concatenate(clean_parts)
        truth = y[indices]
        clean_pred = (clean >= loaded.threshold).astype(int)

        def record(scope_name, eps, prob, steps, maxnorm):
            pred = (prob >= loaded.threshold).astype(int)
            m = compute_metrics(truth, pred, prob)
            rows.append(
                {
                    "scope": scope_name,
                    "n_beats": len(indices),
                    "attack": "pgd",
                    "epsilon": eps,
                    "robust_accuracy": m["accuracy"],
                    "adversarial_f1": m["f1"],
                    "adversarial_recall": m["recall"],
                    "adversarial_precision": m["precision"],
                    "attack_success_rate": attack_success_rate(truth, clean_pred, pred),
                    "max_linf": maxnorm,
                    "steps": steps,
                    "starts": cfg["pgd_starts"],
                    "objectives": "margin + BCE"
                    if steps == cfg["pgd_steps"]
                    else "20-step union + 50-step margin",
                }
            )
            np.savez_compressed(
                out / f"{scope_name}_eps{eps:g}.npz",
                indices=indices,
                probabilities=prob,
                clean_probabilities=clean,
                y=truth,
            )

        for eps in budgets:
            record(scope, eps, np.concatenate(probs[eps]), cfg["pgd_steps"], norms[eps])
        if strong_parts:
            record(
                "strength_check",
                0.1,
                np.concatenate(strong_parts),
                cfg["strength_steps"],
                strong_norm,
            )
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    spec.update(
        status="completed",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        results_sha256=sha256_file(out / "metrics.csv"),
    )
    (out / "manifest.json").write_text(json.dumps(spec, indent=2))
    if not smoothing:
        destination = ARTIFACTS_DIR / "experiments" / tag
        frame = pd.DataFrame(rows)
        frame[frame.scope == "record_balanced_audit"].to_csv(
            destination / "robustness_curve.csv", index=False
        )
        shutil.copy2(out / "attack_examples.csv", destination / "attack_examples.csv")
        shutil.copy2(
            out / "example_waveforms.npz", destination / "attack_waveforms.npz"
        )
    print(name, "completed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["morphology", "context", "context_defended"],
        required=True,
    )
    parser.add_argument("--smoothing", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.variant, args.smoothing)
