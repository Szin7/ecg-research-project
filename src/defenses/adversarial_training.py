import torch

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack


def make_adversarial_batch_fn(attack: str, epsilon: float, clean_ratio: float, clip_min=-6.0, clip_max=6.0,
                               alpha_ratio: float = 0.25, iterations: int = 7):
    """Builds the extra_batches hook scripts/train.py expects: replaces a
    (1 - clean_ratio) fraction of each batch with adversarial examples
    crafted against the model's current weights, mixed back in with the
    untouched clean fraction. PGD by default -- a stronger adversary than
    FGSM, the standard choice for adversarial training since Madry et al.
    """

    def batch_fn(xb, yb, model, loss_fn):
        n = xb.size(0)
        n_clean = int(round(n * clean_ratio))
        if n_clean >= n:
            return xb, yb

        x_to_attack, y_to_attack = xb[n_clean:], yb[n_clean:]

        if attack == "fgsm":
            x_adv = fgsm_attack(model, x_to_attack, y_to_attack, epsilon, loss_fn, clip_min, clip_max)
        elif attack == "pgd":
            alpha = alpha_ratio * epsilon
            x_adv = pgd_attack(model, x_to_attack, y_to_attack, epsilon, alpha, iterations, loss_fn, clip_min, clip_max)
        else:
            raise ValueError(f"unknown attack: {attack!r}")

        model.train()  # fgsm/pgd switch to eval() internally while crafting the attack
        return torch.cat([xb[:n_clean], x_adv], dim=0), yb

    return batch_fn
