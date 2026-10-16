import torch
import torch.nn.functional as F


def smooth_signal(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Moving-average smoothing applied at inference time -- a lightweight,
    training-free secondary defense. Averaging over a small window blurs
    out the kind of high-frequency, near-every-sample perturbation FGSM/PGD
    introduce, at the cost of also blurring genuine fine waveform detail,
    which is why this is a secondary comparison point, not the primary
    defense (see src/defenses/adversarial_training.py).
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    pad = kernel_size // 2
    weight = torch.ones(1, 1, kernel_size, device=x.device, dtype=x.dtype) / kernel_size
    return F.conv1d(x, weight, padding=pad)


class SmoothedModel(torch.nn.Module):
    """Differentiable defense pipeline, so attacks include preprocessing."""

    def __init__(self, model, kernel_size=5):
        super().__init__()
        self.model = model
        self.kernel_size = kernel_size

    def forward(self, x):
        length = getattr(self.model, "waveform_samples", None)
        if length:
            x = torch.cat(
                [smooth_signal(x[..., :length], self.kernel_size), x[..., length:]],
                dim=-1,
            )
        else:
            x = smooth_signal(x, self.kernel_size)
        return self.model(x)

    def attack_mask(self, x):
        return (
            self.model.attack_mask(x)
            if hasattr(self.model, "attack_mask")
            else torch.ones_like(x)
        )


def predict_with_smoothing(
    model, x: torch.Tensor, kernel_size: int = 5
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return SmoothedModel(model, kernel_size)(x)
