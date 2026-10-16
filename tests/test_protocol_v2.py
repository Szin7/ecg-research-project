import numpy as np
import pytest
import torch
from src.attacks.strong_pgd import worst_case_pgd
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.data.protocol import assert_subject_disjoint, past_rr_features
from src.defenses.input_smoothing import SmoothedModel
from src.evaluation.peaks import peak_agreement
from src.models.cnn1d import ECGCNN


def test_subject_overlap_is_rejected():
    with pytest.raises(ValueError, match="Subject overlap"):
        assert_subject_disjoint({"train": [100], "val": [201], "test": [202]})
    assert_subject_disjoint({"train": [100], "val": [203], "test": [202]})


def test_rr_context_cannot_see_future_beats():
    peaks = np.array([100, 400, 700, 1100, 1400])
    np.testing.assert_array_equal(
        past_rr_features(peaks, 360)[:3], past_rr_features(peaks[:3], 360)
    )
    np.testing.assert_array_equal(past_rr_features(peaks, 360)[0], np.zeros(4))


def test_context_dimensions_and_attack_bounds():
    torch.manual_seed(4)
    model = ECGCNN(waveform_samples=250, rr_features=4)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 250))
    x = torch.randn(2, 1, 254).clamp(-5, 5)
    y = torch.ones(2)
    for attack in [
        lambda: worst_case_pgd(model, x, y, 0.1, 0.04, steps=2, starts=2),
        lambda: fgsm_attack(model, x, y, 0.1),
        lambda: pgd_attack(model, x, y, 0.1, 0.04, 2),
    ]:
        adv = attack()
        assert torch.equal(x[..., 250:], adv[..., 250:])
        assert (adv - x).abs().max() <= 0.10001
    assert torch.equal(x, worst_case_pgd(model, x, y, 0, 0))


def test_retention_and_random_seed_are_reproducible():
    torch.manual_seed(7)
    model = ECGCNN().eval()
    x = torch.randn(5, 1, 250)
    y = torch.randint(0, 2, (5,)).float()
    first = worst_case_pgd(model, x, y, 0.1, 0.04, steps=2, starts=1, seed=99)
    second = worst_case_pgd(
        model, x, y, 0.1, 0.04, steps=5, starts=3, seed=99, initial=first
    )
    with torch.no_grad():
        assert torch.all(
            (1 - 2 * y) * model(second) >= (1 - 2 * y) * model(first) - 1e-6
        )
    assert torch.equal(
        second,
        worst_case_pgd(
            model, x, y, 0.1, 0.04, steps=5, starts=3, seed=99, initial=first
        ),
    )


def test_smoothing_keeps_context_and_gradients():
    model = SmoothedModel(ECGCNN(waveform_samples=250, rr_features=4))
    x = torch.randn(2, 1, 254, requires_grad=True)
    gradient = torch.autograd.grad(model(x).sum(), x)[0]
    assert gradient[..., :250].abs().sum() > 0 and gradient[..., 250:].abs().sum() > 0
    assert torch.equal(model.attack_mask(x)[..., 250:], torch.zeros_like(x[..., 250:]))


def test_peak_matching_cannot_double_count():
    result = peak_agreement([100, 200], [99, 101, 205], 1000, 10)
    assert result["matched"] == 2 and result["recall"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
