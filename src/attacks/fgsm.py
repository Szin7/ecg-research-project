import torch
import torch.nn as nn


def fgsm_attack(
    model, x, y, epsilon, loss_fn=None, clip_min=-6.0, clip_max=6.0, targeted=False
):
    """Fast Gradient Sign Method (Goodfellow et al., 2015): a single step
    in the sign of the loss gradient with respect to the input.

    Untargeted (default): y is the true label and the step moves toward
    *higher* loss, away from the correct prediction. Targeted: y is the
    label to push toward, and the step moves toward *lower* loss for that
    label instead. For this binary task the two coincide -- there's only
    one other class to flip to -- the distinction only really matters once
    there's more than two classes to pick a target from.

    clip_min/clip_max bound the result to the same range beats are
    normalized into (see configs/dataset.yaml normalization.clip_std), so
    an attack can't produce a signal outside what the model was ever
    trained to see.
    """
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    model.eval()
    x = x.clone().detach().requires_grad_(True)

    loss = loss_fn(model(x), y)
    if targeted:
        loss = -loss

    grad = torch.autograd.grad(loss, x)[0]
    mask = model.attack_mask(x) if hasattr(model, "attack_mask") else torch.ones_like(x)
    x_adv = x + epsilon * grad.sign() * mask
    return torch.where(mask.bool(), torch.clamp(x_adv, clip_min, clip_max), x).detach()
