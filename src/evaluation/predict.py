import numpy as np
import torch


@torch.inference_mode()
def predict_array(model, X, batch_size=256):
    """Binary sigmoid scores without constructing a whole-dataset graph."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(X) == 0:
        return np.empty(0, dtype=np.float32)
    model.eval()
    device = next(model.parameters()).device
    scores = []
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device).unsqueeze(1)
        scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(scores)


@torch.no_grad()
def predict_probs(model, loader, device, loss_fn=None):
    """Run a model over a DataLoader in eval mode. Returns (probs, targets,
    avg_loss); avg_loss is None when loss_fn isn't given, so a plain
    inference pass doesn't need a criterion it has no use for."""
    model.eval()
    all_probs, all_targets, losses = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if loss_fn is not None:
            losses.append(loss_fn(logits, yb).item())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    avg_loss = float(np.mean(losses)) if losses else None
    return probs, targets, avg_loss


@torch.no_grad()
def predict_classes(model, loader, device, loss_fn=None):
    """Multi-class analog of predict_probs: argmax over raw logits instead
    of a sigmoid probability, since there's no single positive class."""
    model.eval()
    all_preds, all_targets, losses = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if loss_fn is not None:
            losses.append(loss_fn(logits, yb).item())
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    avg_loss = float(np.mean(losses)) if losses else None
    return preds, targets, avg_loss
