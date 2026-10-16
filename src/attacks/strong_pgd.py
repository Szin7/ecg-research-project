"""Threshold-aware PGD retaining each example's worst tested candidate."""

import math
import torch
import torch.nn.functional as F


def worst_case_pgd(
    model,
    x,
    y,
    epsilon,
    alpha,
    steps=20,
    starts=3,
    threshold=0.5,
    seed=42,
    initial=None,
    objective="margin",
):
    if epsilon < 0 or alpha < 0 or steps < 1 or starts < 1:
        raise ValueError("Invalid attack parameters")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one")
    if objective not in {"margin", "bce"}:
        raise ValueError("Unknown objective")
    model.eval()
    original = x.detach()
    direction = 1 - 2 * y
    mask = model.attack_mask(x) if hasattr(model, "attack_mask") else torch.ones_like(x)
    boundary = math.log(threshold / (1 - threshold))
    generator = torch.Generator(device=x.device).manual_seed(seed)
    with torch.no_grad():
        best = original.clone()
        best_margin = direction * (model(best) - boundary)

    def retain(candidate, logits=None):
        nonlocal best, best_margin
        with torch.no_grad():
            score = direction * (
                (model(candidate) if logits is None else logits.detach()) - boundary
            )
            better = score > best_margin
            best[better] = candidate.detach()[better]
            best_margin = torch.maximum(best_margin, score)

    if initial is not None:
        if (initial - original).abs().max() > epsilon + 1e-5 or not torch.equal(
            initial * (1 - mask), original * (1 - mask)
        ):
            raise ValueError("Invalid initial candidate")
        retain(initial)
    if epsilon == 0:
        return original.clone()
    for start in range(starts):
        delta = (
            torch.zeros_like(x)
            if start == 0
            else (torch.rand(x.shape, generator=generator, device=x.device) * 2 - 1)
            * epsilon
            * mask
        )
        candidate = torch.where(mask.bool(), (original + delta).clamp(-6, 6), original)
        for _ in range(steps):
            candidate = candidate.detach().requires_grad_(True)
            logits = model(candidate)
            retain(candidate, logits)
            loss = (
                direction * (logits - boundary)
                if objective == "margin"
                else F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            )
            gradient = torch.autograd.grad(loss.sum(), candidate)[0]
            delta = (
                candidate.detach() + alpha * gradient.sign() * mask - original
            ).clamp(-epsilon, epsilon) * mask
            candidate = torch.where(
                mask.bool(), (original + delta).clamp(-6, 6), original
            )
        retain(candidate)
    return best.detach()
