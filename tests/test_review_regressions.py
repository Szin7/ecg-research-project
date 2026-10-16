import numpy as np
import pytest
import torch

from src.attacks.pgd import pgd_attack
from src.defenses.input_smoothing import SmoothedModel, smooth_signal
from src.evaluation.predict import predict_array
from src.models.cnn1d import ECGCNN
from src.utils.tags import resolve_tag


def test_batched_array_prediction_matches_direct_inference():
    torch.manual_seed(42)
    model = ECGCNN().eval()
    X = np.random.default_rng(42).normal(size=(17, 250)).astype(np.float32)
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(X).unsqueeze(1))).numpy()
    np.testing.assert_allclose(predict_array(model, X, batch_size=4), expected, atol=1e-6)
    assert all(p.grad is None for p in model.parameters())


def test_default_tags_cannot_overwrite_research_with_demo():
    assert resolve_tag("demo") == "demo_baseline_cnn"
    assert resolve_tag("research") == "baseline_cnn"
    assert resolve_tag("demo", default="defended_cnn") == "demo_defended_cnn"
    with pytest.raises(ValueError):
        resolve_tag("demo", "baseline_cnn")
    with pytest.raises(ValueError):
        resolve_tag("research", "demo_baseline_cnn")
    with pytest.raises(ValueError):
        resolve_tag("research", "../outside")


def test_adaptive_smoothing_attack_differentiates_through_defense():
    class LinearModel(torch.nn.Module):
        def forward(self, x):
            return (x * torch.tensor([[[1., -2., 3., -4., 5.]]])).sum((1, 2))

    wrapped = SmoothedModel(LinearModel(), kernel_size=3)
    x = torch.zeros(1, 1, 5)
    y = torch.ones(1)
    loss = torch.nn.BCEWithLogitsLoss()
    adv = pgd_attack(wrapped, x, y, epsilon=0.2, alpha=0.05, iterations=10,
                     random_start=False)
    assert loss(wrapped(adv), y) > loss(wrapped(x), y)
    assert (adv - x).abs().max() <= 0.20001


@pytest.mark.parametrize("kernel", [0, 2, -1])
def test_smoothing_rejects_length_changing_kernels(kernel):
    with pytest.raises(ValueError):
        smooth_signal(torch.zeros(1, 1, 10), kernel)


def test_cache_rebuilds_when_configuration_or_records_change(tmp_path, monkeypatch):
    from src.data import cache
    calls = []

    def build(mode, records, cfg, label_fn=None):
        calls.append((records, cfg))
        return np.zeros((1, 3)), np.zeros(1, dtype=int), np.array(records[:1])

    monkeypatch.setattr(cache, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(cache, "build_beats", build)
    cache.load_or_build_split("demo", "test", ["a"], {"clip": 6})
    cache.load_or_build_split("demo", "test", ["a"], {"clip": 6})
    assert len(calls) == 1
    cache.load_or_build_split("demo", "test", ["a"], {"clip": 4})
    cache.load_or_build_split("demo", "test", ["b"], {"clip": 4})
    assert len(calls) == 3
