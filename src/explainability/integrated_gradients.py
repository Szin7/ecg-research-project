import torch
import torch.nn as nn


def integrated_gradients(model, x, y=None, baseline=None, steps=50, loss_fn=None):
    """Integrated Gradients (Sundararajan et al., 2017) for a single
    example (x has shape (1, C, L)): accumulate gradients along a
    straight-line path from a baseline -- the flat isoelectric line by
    default -- to the actual input, approximated with a Riemann sum over
    `steps` interpolation points. Attribution = (x - baseline) times the
    average gradient along that path.
    """
    assert x.shape[0] == 1, "integrated_gradients expects a single example"
    if steps < 2:
        raise ValueError("steps must be at least 2")
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)

    alphas = torch.linspace(0, 1, steps, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    x_interp = (baseline + alphas * (x - baseline)).clone().detach().requires_grad_(True)

    if y is not None:
        criterion = loss_fn or nn.BCEWithLogitsLoss(reduction="sum")
        loss = criterion(model(x_interp), y.expand(steps))
        # A batch mean scales each path gradient by 1/steps; undo that
        # before integrating. Otherwise attribution shrinks with resolution.
        if getattr(criterion, "reduction", None) == "mean":
            loss = loss * steps
        loss = loss.sum()
    else:
        loss = model(x_interp).sum()

    grad = torch.autograd.grad(loss, x_interp)[0]
    avg_grad = ((grad[:-1] + grad[1:]) / 2).mean(dim=0, keepdim=True)
    return ((x - baseline) * avg_grad).detach()
