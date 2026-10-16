import argparse

from src.utils.tags import resolve_tag

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.data.cache import load_or_build_split
from src.data.splits import get_split
from src.evaluation.robustness import perturbation_stats
from src.models.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.paths import ARTIFACTS_DIR


def craft_examples(model, threshold, X, y, device, attack, epsilon, alpha_ratio, iterations, n_examples, seed):
    """Pick a random, class-balanced-ish sample of correctly classified
    test beats and attack each individually -- a small curated gallery for
    the dashboard and README, distinct from the full aggregate sweep in
    run_benchmark.py."""
    rng = np.random.default_rng(seed)
    sample_size = min(2000, len(y))
    idx = rng.choice(len(y), size=sample_size, replace=False)
    X_sub, y_sub = X[idx], y[idx]

    x_tensor = torch.from_numpy(X_sub).float().unsqueeze(1).to(device)
    y_tensor = torch.from_numpy(y_sub).float().to(device)

    with torch.no_grad():
        clean_probs = torch.sigmoid(model(x_tensor)).cpu().numpy()
    clean_preds = (clean_probs >= threshold).astype(int)
    correct_idx = np.where(clean_preds == y_sub)[0]
    chosen = rng.choice(correct_idx, size=min(n_examples, len(correct_idx)), replace=False)

    loss_fn = nn.BCEWithLogitsLoss()
    rows, waveforms = [], []
    for i in chosen:
        xi, yi = x_tensor[i : i + 1], y_tensor[i : i + 1]

        if attack == "fgsm":
            x_adv = fgsm_attack(model, xi, yi, epsilon, loss_fn)
        elif attack == "pgd":
            x_adv = pgd_attack(model, xi, yi, epsilon, alpha_ratio * epsilon, iterations, loss_fn)
        else:
            raise ValueError(f"unknown attack: {attack!r}")

        with torch.no_grad():
            adv_prob = torch.sigmoid(model(x_adv)).item()
        adv_pred = int(adv_prob >= threshold)
        stats = perturbation_stats(xi, x_adv)

        rows.append(
            {
                "beat_index": int(idx[i]),
                "true_label": int(y_sub[i]),
                "attack": attack,
                "epsilon": epsilon,
                "clean_prob": float(clean_probs[i]),
                "clean_pred": int(clean_preds[i]),
                "adv_prob": adv_prob,
                "adv_pred": adv_pred,
                "flipped": adv_pred != int(clean_preds[i]),
                **stats,
            }
        )
        waveforms.append((xi.squeeze().cpu().numpy(), x_adv.squeeze().cpu().numpy()))

    return rows, waveforms


def run(mode: str, tag: str, attack: str, epsilon: float, n_examples: int = 12, seed: int = 0) -> dict:
    dataset_cfg = load_config("dataset.yaml")
    exp_cfg = load_config("experiments.yaml")
    device = get_device()

    split = get_split(mode, dataset_cfg)
    X_test, y_test, _ = load_or_build_split(mode, "test", split["test"], dataset_cfg)

    loaded = load_checkpoint(ARTIFACTS_DIR / "models" / f"{tag}.pt")
    loaded.model.to(device)
    alpha_ratio = exp_cfg["attack"]["pgd"]["alpha_ratio"]
    iterations = exp_cfg["attack"]["pgd"]["iterations"]

    rows, waveforms = craft_examples(
        loaded.model, loaded.threshold, X_test, y_test, device,
        attack=attack, epsilon=epsilon, alpha_ratio=alpha_ratio, iterations=iterations,
        n_examples=n_examples, seed=seed,
    )

    n_flipped = sum(r["flipped"] for r in rows)
    print(f"{tag}: {n_flipped}/{len(rows)} correctly-classified examples flipped by {attack} at eps={epsilon}")

    out_dir = ARTIFACTS_DIR / "experiments" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "attack_examples.csv", index=False)

    originals = np.stack([w[0] for w in waveforms]) if waveforms else np.zeros((0, X_test.shape[1]))
    adversarials = np.stack([w[1] for w in waveforms]) if waveforms else np.zeros((0, X_test.shape[1]))
    np.savez(out_dir / "attack_waveforms.npz", original=originals, adversarial=adversarials)

    print(f"saved -> {out_dir / 'attack_examples.csv'} and attack_waveforms.npz")
    return {"n_examples": len(rows), "n_flipped": n_flipped}


def main():
    parser = argparse.ArgumentParser(description="Generate a curated set of adversarial examples")
    parser.add_argument("--mode", choices=["demo", "research"], default="demo")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--attack", choices=["fgsm", "pgd"], default="pgd")
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--n-examples", type=int, default=12)
    args = parser.parse_args()
    args.tag = resolve_tag(args.mode, args.tag, 'baseline_cnn')
    run(mode=args.mode, tag=args.tag, attack=args.attack, epsilon=args.epsilon, n_examples=args.n_examples)


if __name__ == "__main__":
    main()
