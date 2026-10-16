import torch
import torch.nn as nn


def pgd_attack(
    model,
    x,
    y,
    epsilon,
    alpha,
    iterations,
    loss_fn=None,
    clip_min=-6.0,
    clip_max=6.0,
    targeted=False,
    random_start=True,
):
    """Projected Gradient Descent (Madry et al., 2018): repeated FGSM-style
    steps of size alpha, with the perturbation re-projected onto the
    epsilon L-infinity ball around the original input after every step --
    that projection is what keeps many small steps from drifting past the
    same budget a single FGSM step would respect.
    """
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    model.eval()
    x_orig = x.clone().detach()
    mask = model.attack_mask(x) if hasattr(model, "attack_mask") else torch.ones_like(x)

    if random_start:
        delta = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(x_orig)
    x_adv = torch.where(
        mask.bool(), torch.clamp(x_orig + delta * mask, clip_min, clip_max), x_orig
    ).detach()

    for _ in range(iterations):
        x_adv.requires_grad_(True)
        loss = loss_fn(model(x_adv), y)
        if targeted:
            loss = -loss

        grad = torch.autograd.grad(loss, x_adv)[0]
        step = x_adv.detach() + alpha * grad.sign()

        delta = torch.clamp(step - x_orig, -epsilon, epsilon)
        x_adv = torch.where(
            mask.bool(), torch.clamp(x_orig + delta * mask, clip_min, clip_max), x_orig
        ).detach()

    return x_adv
