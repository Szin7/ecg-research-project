import torch
import torch.nn as nn


def compute_saliency(model, x, y=None, loss_fn=None):
    """Vanilla gradient saliency: d(output)/d(input). If y is given,
    differentiates the loss w.r.t. the true label instead of the raw
    logit sum -- showing what drove the model toward (or away from) the
    correct answer, rather than just what the logit responds to.
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    if y is not None:
        output = (loss_fn or nn.BCEWithLogitsLoss())(model(x), y)
    else:
        output = model(x).sum()

    return torch.autograd.grad(output, x)[0].detach()
