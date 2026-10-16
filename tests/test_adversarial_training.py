import torch
import torch.nn as nn

from src.defenses.adversarial_training import make_adversarial_batch_fn
from src.models.cnn1d import ECGCNN


def _model():
    torch.manual_seed(0)
    return ECGCNN()


def _batch():
    x = torch.randn(16, 1, 250)
    y = torch.randint(0, 2, (16,)).float()
    return x, y


def test_clean_ratio_one_leaves_batch_untouched():
    model = _model()
    x, y = _batch()
    batch_fn = make_adversarial_batch_fn(attack="fgsm", epsilon=0.1, clean_ratio=1.0)

    x_out, y_out = batch_fn(x, y, model, nn.BCEWithLogitsLoss())

    assert torch.equal(x_out, x)
    assert torch.equal(y_out, y)


def test_clean_ratio_zero_perturbs_every_example():
    model = _model()
    x, y = _batch()
    batch_fn = make_adversarial_batch_fn(attack="fgsm", epsilon=0.1, clean_ratio=0.0)

    x_out, y_out = batch_fn(x, y, model, nn.BCEWithLogitsLoss())

    assert x_out.shape == x.shape
    assert torch.equal(y_out, y)
    assert not torch.equal(x_out, x)
    assert (x_out - x).abs().max().item() > 0


def test_partial_ratio_leaves_clean_slice_untouched_and_perturbs_the_rest():
    model = _model()
    x, y = _batch()
    batch_fn = make_adversarial_batch_fn(attack="pgd", epsilon=0.1, clean_ratio=0.5, iterations=3)

    x_out, y_out = batch_fn(x, y, model, nn.BCEWithLogitsLoss())

    n_clean = 8
    assert torch.equal(x_out[:n_clean], x[:n_clean])
    assert not torch.equal(x_out[n_clean:], x[n_clean:])
    assert torch.equal(y_out, y)


def test_labels_stay_aligned_with_examples_after_mixing():
    model = _model()
    x, y = _batch()
    batch_fn = make_adversarial_batch_fn(attack="pgd", epsilon=0.05, clean_ratio=0.25, iterations=2)

    x_out, y_out = batch_fn(x, y, model, nn.BCEWithLogitsLoss())

    assert x_out.shape[0] == y_out.shape[0] == x.shape[0]
    assert torch.equal(y_out, y)


def test_unknown_attack_raises():
    model = _model()
    x, y = _batch()
    batch_fn = make_adversarial_batch_fn(attack="bogus", epsilon=0.1, clean_ratio=0.0)

    try:
        batch_fn(x, y, model, nn.BCEWithLogitsLoss())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_perturbed_slice_respects_epsilon_budget():
    model = _model()
    x, y = _batch()
    epsilon = 0.08
    batch_fn = make_adversarial_batch_fn(attack="pgd", epsilon=epsilon, clean_ratio=0.5, iterations=5, clip_min=-100, clip_max=100)

    x_out, _ = batch_fn(x, y, model, nn.BCEWithLogitsLoss())

    n_clean = 8
    perturbation = (x_out[n_clean:] - x[n_clean:]).abs().max().item()
    assert perturbation <= epsilon + 1e-4
