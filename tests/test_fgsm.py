import torch
import torch.nn as nn

from src.attacks.fgsm import fgsm_attack
from src.models.cnn1d import ECGCNN


def _model():
    torch.manual_seed(0)
    return ECGCNN()


def test_fgsm_respects_epsilon_linf_budget():
    model = _model()
    x = torch.randn(8, 1, 250)
    y = torch.randint(0, 2, (8,)).float()

    x_adv = fgsm_attack(model, x, y, epsilon=0.1, clip_min=-100, clip_max=100)

    assert (x_adv - x).abs().max().item() <= 0.1 + 1e-5


def test_fgsm_epsilon_zero_is_a_no_op():
    model = _model()
    x = torch.randn(4, 1, 250)
    y = torch.randint(0, 2, (4,)).float()

    x_adv = fgsm_attack(model, x, y, epsilon=0.0, clip_min=-100, clip_max=100)

    assert torch.allclose(x_adv, x, atol=1e-6)


def test_fgsm_respects_clip_bounds():
    model = _model()
    x = torch.randn(8, 1, 250) * 5
    y = torch.randint(0, 2, (8,)).float()

    x_adv = fgsm_attack(model, x, y, epsilon=2.0, clip_min=-6.0, clip_max=6.0)

    assert x_adv.max().item() <= 6.0 + 1e-5
    assert x_adv.min().item() >= -6.0 - 1e-5


def test_fgsm_output_shape_matches_input():
    model = _model()
    x = torch.randn(5, 1, 250)
    y = torch.randint(0, 2, (5,)).float()
    assert fgsm_attack(model, x, y, epsilon=0.05).shape == x.shape


def test_fgsm_increases_loss_on_average():
    model = _model()
    x = torch.randn(64, 1, 250)
    y = torch.randint(0, 2, (64,)).float()
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        clean_loss = loss_fn(model(x), y).item()

    x_adv = fgsm_attack(model, x, y, epsilon=0.2, loss_fn=loss_fn, clip_min=-100, clip_max=100)
    with torch.no_grad():
        adv_loss = loss_fn(model(x_adv), y).item()

    assert adv_loss > clean_loss


def test_targeted_and_untargeted_agree_in_binary_setting():
    # BCEWithLogitsLoss makes untargeted-away-from-1 and targeted-toward-0
    # the same sign-of-gradient direction: sigmoid(logits) and its
    # complement are both always positive, so the sign of the gradient
    # doesn't depend on which one you differentiate through.
    model = _model()
    x = torch.randn(8, 1, 250)
    y_true = torch.ones(8)
    y_target = torch.zeros(8)

    x_adv_untargeted = fgsm_attack(model, x, y_true, epsilon=0.1, targeted=False, clip_min=-100, clip_max=100)
    x_adv_targeted = fgsm_attack(model, x, y_target, epsilon=0.1, targeted=True, clip_min=-100, clip_max=100)

    assert torch.allclose(x_adv_untargeted, x_adv_targeted, atol=1e-5)


def test_fgsm_works_against_a_multiclass_model_with_cross_entropy():
    # fgsm_attack takes loss_fn as a parameter specifically so it works
    # unchanged against the multi-class AAMI model -- verify that holds,
    # not just assert it in a docstring. A freshly-initialized model's
    # loss sits almost exactly at ln(4) (near-uniform predictions), which
    # is close enough to a saddle point that FGSM's single step doesn't
    # reliably increase loss there -- give it a few real training steps
    # first, both for a meaningful gradient direction and because that's
    # the realistic scenario anyway (FGSM is only ever run against a
    # trained model in practice, not a random one).
    torch.manual_seed(0)
    model = ECGCNN(num_classes=4)
    x = torch.randn(128, 1, 250)
    y = torch.randint(0, 4, (128,))
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(20):
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        clean_loss = loss_fn(model(x), y).item()

    x_adv = fgsm_attack(model, x, y, epsilon=0.2, loss_fn=loss_fn, clip_min=-100, clip_max=100)

    assert x_adv.shape == x.shape
    assert (x_adv - x).abs().max().item() <= 0.2 + 1e-5
    with torch.no_grad():
        adv_loss = loss_fn(model(x_adv), y).item()
    assert adv_loss > clean_loss
