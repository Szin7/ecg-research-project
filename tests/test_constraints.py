import numpy as np
import torch

from src.attacks.fgsm import fgsm_attack
from src.evaluation.robustness import perturbation_stats, run_attack_sweep
from src.models.cnn1d import ECGCNN


def _model():
    torch.manual_seed(1)
    return ECGCNN()


def test_perturbation_stats_zero_for_identical_signals():
    x = torch.randn(4, 1, 250)
    stats = perturbation_stats(x, x.clone())
    assert stats["linf_mean"] == 0.0
    assert stats["l2_mean"] == 0.0
    assert stats["rms_mean"] == 0.0


def test_perturbation_stats_matches_a_known_constant_offset():
    x = torch.zeros(2, 1, 4)
    stats = perturbation_stats(x, x + 0.5)
    assert abs(stats["linf_mean"] - 0.5) < 1e-6
    assert abs(stats["rms_mean"] - 0.5) < 1e-6


def test_perturbation_stats_snr_decreases_as_perturbation_grows():
    torch.manual_seed(2)
    x = torch.randn(16, 1, 250)
    small = perturbation_stats(x, x + 0.01 * torch.randn_like(x))
    large = perturbation_stats(x, x + 1.0 * torch.randn_like(x))
    assert small["snr_db_mean"] > large["snr_db_mean"]


def test_fgsm_perturbation_never_exceeds_epsilon_budget():
    model = _model()
    x = torch.randn(32, 1, 250)
    y = torch.randint(0, 2, (32,)).float()

    for epsilon in [0.01, 0.05, 0.1, 0.2]:
        x_adv = fgsm_attack(model, x, y, epsilon=epsilon, clip_min=-100, clip_max=100)
        stats = perturbation_stats(x, x_adv)
        assert stats["linf_mean"] <= epsilon + 1e-5


def test_clip_bounds_are_never_violated_regardless_of_epsilon():
    model = _model()
    x = torch.randn(16, 1, 250) * 3
    y = torch.randint(0, 2, (16,)).float()

    x_adv = fgsm_attack(model, x, y, epsilon=50.0, clip_min=-6.0, clip_max=6.0)

    assert x_adv.max().item() <= 6.0 + 1e-5
    assert x_adv.min().item() >= -6.0 - 1e-5


def test_attack_sweep_epsilon_zero_matches_clean_performance():
    model = _model()
    model.eval()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 250)).astype(np.float32)
    y = rng.integers(0, 2, size=40).astype(np.int64)

    results = run_attack_sweep(model, threshold=0.5, X=X, y=y, attack="fgsm", epsilons=[0.0], device=torch.device("cpu"))

    assert results[0]["epsilon"] == 0.0
    assert results[0]["attack_success_rate"] == 0.0
    assert "linf_mean" not in results[0]  # no attack ran, so no perturbation to report


def test_attack_sweep_perturbation_grows_monotonically_with_epsilon():
    model = _model()
    model.eval()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 250)).astype(np.float32)
    y = rng.integers(0, 2, size=40).astype(np.int64)

    epsilons = [0.05, 0.1, 0.2]
    results = run_attack_sweep(model, threshold=0.5, X=X, y=y, attack="fgsm", epsilons=epsilons, device=torch.device("cpu"))

    linf_values = [r["linf_mean"] for r in results]
    assert linf_values == sorted(linf_values)
    for r, epsilon in zip(results, epsilons):
        assert r["linf_mean"] <= epsilon + 1e-4
