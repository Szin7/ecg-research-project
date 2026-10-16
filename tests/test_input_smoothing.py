import torch

from src.defenses.input_smoothing import predict_with_smoothing, smooth_signal
from src.models.cnn1d import ECGCNN


def test_smoothing_preserves_shape():
    x = torch.randn(4, 1, 250)
    assert smooth_signal(x).shape == x.shape


def test_smoothing_reduces_high_frequency_noise():
    torch.manual_seed(0)
    clean = torch.sin(torch.linspace(0, 4 * 3.14159, 250)).view(1, 1, -1)
    noisy = clean + 0.5 * torch.randn_like(clean)

    smoothed = smooth_signal(noisy, kernel_size=7)

    error_before = (noisy - clean).pow(2).mean()
    error_after = (smoothed - clean).pow(2).mean()
    assert error_after < error_before


def test_smoothing_kernel_size_one_is_identity():
    x = torch.randn(2, 1, 250)
    assert torch.allclose(smooth_signal(x, kernel_size=1), x, atol=1e-6)


def test_predict_with_smoothing_returns_valid_logits():
    torch.manual_seed(0)
    model = ECGCNN()
    x = torch.randn(8, 1, 250)

    logits = predict_with_smoothing(model, x)

    assert logits.shape == (8,)
    assert not torch.isnan(logits).any()
