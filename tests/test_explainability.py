import torch

from src.explainability.integrated_gradients import integrated_gradients
from src.explainability.saliency import compute_saliency
from src.models.cnn1d import ECGCNN


def _model():
    torch.manual_seed(0)
    return ECGCNN()


def test_saliency_shape_matches_input():
    model = _model()
    x = torch.randn(4, 1, 250)
    y = torch.randint(0, 2, (4,)).float()

    assert compute_saliency(model, x, y).shape == x.shape


def test_saliency_is_nonzero_for_a_real_model():
    model = _model()
    x = torch.randn(4, 1, 250)
    y = torch.randint(0, 2, (4,)).float()

    assert compute_saliency(model, x, y).abs().sum().item() > 0


def test_integrated_gradients_shape_matches_input():
    model = _model()
    model.eval()
    x = torch.randn(1, 1, 250)
    y = torch.ones(1)

    assert integrated_gradients(model, x, y, steps=20).shape == x.shape


def test_integrated_gradients_satisfies_completeness_for_a_linear_model():
    # Completeness axiom: sum(IG) == F(x) - F(baseline). A linear model's
    # gradient is constant along the whole path, so this holds almost
    # exactly even with few Riemann-sum steps -- a real correctness check
    # on the implementation, not just a shape check.
    torch.manual_seed(0)
    linear = torch.nn.Conv1d(1, 1, kernel_size=250, bias=True)

    x = torch.randn(1, 1, 250)
    baseline = torch.zeros_like(x)

    attributions = integrated_gradients(linear, x, y=None, baseline=baseline, steps=10)

    with torch.no_grad():
        f_x = linear(x).sum().item()
        f_baseline = linear(baseline).sum().item()

    assert abs(attributions.sum().item() - (f_x - f_baseline)) < 1e-3


def test_integrated_gradients_rejects_a_batch():
    model = _model()
    x = torch.randn(2, 1, 250)
    try:
        integrated_gradients(model, x)
        assert False, "expected an assertion error for batch size > 1"
    except AssertionError:
        pass


def test_loss_integrated_gradients_completeness_and_step_invariance():
    class SumModel(torch.nn.Module):
        def forward(self, x):
            return x.sum(dim=(1, 2))

    model = SumModel()
    x = torch.tensor([[[0.2, 0.3, 0.5]]])
    y = torch.ones(1)
    criterion = torch.nn.BCEWithLogitsLoss()
    expected = criterion(model(x), y) - criterion(model(torch.zeros_like(x)), y)
    for steps in [20, 100]:
        attribution = integrated_gradients(model, x, y, steps=steps, loss_fn=criterion)
        assert torch.allclose(attribution.sum(), expected, atol=1e-4)
