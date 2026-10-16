import torch
import torch.nn as nn

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.models.cnn1d import ECGCNN


def _model():
    torch.manual_seed(0)
    return ECGCNN()


def test_pgd_respects_epsilon_linf_budget():
    model = _model()
    x = torch.randn(16, 1, 250)
    y = torch.randint(0, 2, (16,)).float()

    x_adv = pgd_attack(model, x, y, epsilon=0.1, alpha=0.03, iterations=10, clip_min=-100, clip_max=100)

    assert (x_adv - x).abs().max().item() <= 0.1 + 1e-4


def test_pgd_budget_holds_even_with_many_iterations():
    model = _model()
    x = torch.randn(16, 1, 250)
    y = torch.randint(0, 2, (16,)).float()

    x_adv = pgd_attack(model, x, y, epsilon=0.05, alpha=0.02, iterations=50, clip_min=-100, clip_max=100)

    assert (x_adv - x).abs().max().item() <= 0.05 + 1e-4


def test_pgd_respects_clip_bounds():
    model = _model()
    x = torch.randn(8, 1, 250) * 5
    y = torch.randint(0, 2, (8,)).float()

    x_adv = pgd_attack(model, x, y, epsilon=2.0, alpha=0.5, iterations=10, clip_min=-6.0, clip_max=6.0)

    assert x_adv.max().item() <= 6.0 + 1e-4
    assert x_adv.min().item() >= -6.0 - 1e-4


def test_pgd_output_shape_matches_input():
    model = _model()
    x = torch.randn(5, 1, 250)
    y = torch.randint(0, 2, (5,)).float()
    assert pgd_attack(model, x, y, epsilon=0.1, alpha=0.03, iterations=5).shape == x.shape


def test_pgd_single_step_matches_fgsm():
    # one PGD iteration with alpha == epsilon and no random start is
    # exactly a single FGSM step
    model = _model()
    x = torch.randn(8, 1, 250)
    y = torch.randint(0, 2, (8,)).float()

    x_fgsm = fgsm_attack(model, x, y, epsilon=0.1, clip_min=-100, clip_max=100)
    x_pgd = pgd_attack(
        model, x, y, epsilon=0.1, alpha=0.1, iterations=1, random_start=False, clip_min=-100, clip_max=100
    )

    assert torch.allclose(x_fgsm, x_pgd, atol=1e-5)


def test_pgd_is_at_least_as_effective_as_fgsm():
    # PGD can only do at least as well as a single FGSM step at the same
    # epsilon, since it optimizes over more steps within the same budget
    model = _model()
    x = torch.randn(128, 1, 250)
    y = torch.randint(0, 2, (128,)).float()
    loss_fn = nn.BCEWithLogitsLoss()

    x_fgsm = fgsm_attack(model, x, y, epsilon=0.15, loss_fn=loss_fn, clip_min=-100, clip_max=100)
    x_pgd = pgd_attack(
        model, x, y, epsilon=0.15, alpha=0.03, iterations=10, loss_fn=loss_fn,
        random_start=False, clip_min=-100, clip_max=100,
    )

    with torch.no_grad():
        fgsm_loss = loss_fn(model(x_fgsm), y).item()
        pgd_loss = loss_fn(model(x_pgd), y).item()

    assert pgd_loss >= fgsm_loss - 1e-3


def test_pgd_works_against_a_multiclass_model_with_cross_entropy():
    torch.manual_seed(0)
    model = ECGCNN(num_classes=4)
    x = torch.randn(8, 1, 250)
    y = torch.randint(0, 4, (8,))
    loss_fn = nn.CrossEntropyLoss()

    x_adv = pgd_attack(model, x, y, epsilon=0.15, alpha=0.03, iterations=10, loss_fn=loss_fn, clip_min=-100, clip_max=100)

    assert x_adv.shape == x.shape
    assert (x_adv - x).abs().max().item() <= 0.15 + 1e-4
