import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.data.torch_dataset import ECGBeatDataset
from src.evaluation.metrics import compute_metrics


def perturbation_stats(x: torch.Tensor, x_adv: torch.Tensor) -> dict:
    """L-infinity, L2, and RMS perturbation magnitude, plus signal-to-noise
    ratio (in dB) between original and adversarial batches. Averaged over
    the batch -- for a batch of size 1 this is just that example's stats.
    """
    delta = (x_adv - x).detach()
    linf = delta.abs().amax(dim=(1, 2))
    l2 = delta.flatten(1).norm(dim=1)
    rms = torch.sqrt((delta**2).mean(dim=(1, 2)))

    signal_power = (x**2).mean(dim=(1, 2)).clamp_min(1e-12)
    noise_power = (delta**2).mean(dim=(1, 2)).clamp_min(1e-12)
    snr_db = 10 * torch.log10(signal_power / noise_power)

    return {
        "linf_mean": float(linf.mean()),
        "l2_mean": float(l2.mean()),
        "rms_mean": float(rms.mean()),
        "snr_db_mean": float(snr_db.mean()),
    }


def attack_success_rate(y_true: np.ndarray, clean_preds: np.ndarray, adv_preds: np.ndarray) -> float:
    """Untargeted definition: the fraction of samples the model got right
    *before* the attack whose predicted class changed afterward. Samples
    the model already had wrong aren't counted -- there's nothing for the
    attack to flip."""
    correct_mask = clean_preds == y_true
    if correct_mask.sum() == 0:
        return 0.0
    flipped = (adv_preds != clean_preds) & correct_mask
    return float(flipped.sum() / correct_mask.sum())


def run_attack_sweep(
    model, threshold, X, y, attack, epsilons, device, alpha_ratio=0.25, iterations=10, batch_size=256
) -> list:
    """Evaluate one model against one attack across a list of epsilon
    budgets. epsilon=0.0 is treated as the clean (no-attack) case rather
    than running the attack with a zero budget."""
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(ECGBeatDataset(X, y), batch_size=batch_size, shuffle=False)

    results = []
    for epsilon in epsilons:
        clean_preds_all, adv_preds_all, targets_all, adv_probs_all = [], [], [], []
        stat_batches = []

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            with torch.no_grad():
                clean_probs = torch.sigmoid(model(xb))
            clean_preds = (clean_probs >= threshold).float()

            if epsilon == 0.0:
                x_adv = xb
            elif attack == "fgsm":
                x_adv = fgsm_attack(model, xb, yb, epsilon, loss_fn)
            elif attack == "pgd":
                x_adv = pgd_attack(model, xb, yb, epsilon, alpha_ratio * epsilon, iterations, loss_fn)
            else:
                raise ValueError(f"unknown attack: {attack!r}")

            with torch.no_grad():
                adv_probs = torch.sigmoid(model(x_adv))
            adv_preds = (adv_probs >= threshold).float()

            clean_preds_all.append(clean_preds.cpu().numpy())
            adv_preds_all.append(adv_preds.cpu().numpy())
            targets_all.append(yb.cpu().numpy())
            adv_probs_all.append(adv_probs.detach().cpu().numpy())
            if epsilon > 0.0:
                stat_batches.append(perturbation_stats(xb, x_adv))

        clean_preds = np.concatenate(clean_preds_all)
        adv_preds = np.concatenate(adv_preds_all)
        targets = np.concatenate(targets_all)
        adv_probs = np.concatenate(adv_probs_all)

        adv_metrics = compute_metrics(targets, adv_preds.astype(int), adv_probs)
        agg_stats = (
            {k: float(np.mean([s[k] for s in stat_batches])) for k in stat_batches[0]} if stat_batches else {}
        )

        results.append(
            {
                "attack": attack,
                "epsilon": epsilon,
                "robust_accuracy": adv_metrics["accuracy"],
                "adversarial_f1": adv_metrics["f1"],
                "adversarial_precision": adv_metrics["precision"],
                "adversarial_recall": adv_metrics["recall"],
                "attack_success_rate": attack_success_rate(targets, clean_preds, adv_preds),
                **agg_stats,
            }
        )

    return results
