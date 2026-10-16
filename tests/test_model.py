import numpy as np
import torch

from src.models.baseline_features import FEATURE_NAMES, extract_feature_matrix, extract_features
from src.models.cnn1d import ECGCNN


def _model():
    return ECGCNN(in_channels=1, conv_channels=(16, 32, 64), kernel_sizes=(7, 5, 3), fc_hidden=32, dropout=0.3)


def test_forward_pass_shape():
    model = _model()
    logits = model(torch.randn(8, 1, 250))
    assert logits.shape == (8,)


def test_forward_pass_batch_size_one():
    model = _model()
    logits = model(torch.randn(1, 1, 250))
    assert logits.shape == (1,)


def test_sigmoid_outputs_are_valid_probabilities():
    model = _model()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.randn(16, 1, 250)))
    assert torch.all(probs >= 0) and torch.all(probs <= 1)
    assert not torch.isnan(probs).any()


def test_parameter_count_is_small():
    n_params = sum(p.numel() for p in _model().parameters())
    assert n_params < 100_000


def test_gradients_flow_to_input():
    model = _model()
    x = torch.randn(4, 1, 250, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_eval_mode_does_not_collapse_after_training():
    """Regression test for a real bug found during development: BatchNorm's
    running stats need enough update steps to track the true activation
    scale, and with too few (e.g. a small dataset, few batches/epoch), eval
    mode can settle into predicting a near-constant value regardless of
    input, even though train-mode (live batch stats) looks fine. Trains on
    two obviously-separable clusters and checks eval-mode outputs still
    vary across inputs afterward -- this failed before bumping BatchNorm's
    momentum in ECGConvBlock."""
    torch.manual_seed(0)
    model = _model()

    x_a = torch.randn(32, 1, 250) * 0.1 + 1.0
    x_b = torch.randn(32, 1, 250) * 0.1 - 1.0
    x = torch.cat([x_a, x_b])
    y = torch.cat([torch.ones(32), torch.zeros(32)])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(30):
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x))

    assert probs.std().item() > 0.05


def test_multiclass_forward_pass_shape():
    model = ECGCNN(num_classes=4)
    logits = model(torch.randn(8, 1, 250))
    assert logits.shape == (8, 4)


def test_multiclass_softmax_outputs_are_valid_probabilities():
    model = ECGCNN(num_classes=4)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.randn(16, 1, 250)), dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-5)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


def test_multiclass_gradients_flow_to_input():
    model = ECGCNN(num_classes=4)
    x = torch.randn(4, 1, 250, requires_grad=True)
    y = torch.randint(0, 4, (4,))
    torch.nn.CrossEntropyLoss()(model(x), y).backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_extract_features_shape_and_no_nans():
    rng = np.random.default_rng(0)
    beat = rng.normal(size=250).astype(np.float32)
    features = extract_features(beat)
    assert features.shape == (len(FEATURE_NAMES),)
    assert not np.isnan(features).any()


def test_extract_feature_matrix_shape():
    rng = np.random.default_rng(1)
    beats = rng.normal(size=(10, 250)).astype(np.float32)
    matrix = extract_feature_matrix(beats)
    assert matrix.shape == (10, len(FEATURE_NAMES))


def test_features_discriminate_different_beat_shapes():
    t = np.linspace(0, 1, 250, dtype=np.float32)
    narrow_spike = np.exp(-((t - 0.5) ** 2) / (2 * 0.01**2)).astype(np.float32)
    wide_spike = np.exp(-((t - 0.5) ** 2) / (2 * 0.05**2)).astype(np.float32)

    assert not np.allclose(extract_features(narrow_spike), extract_features(wide_spike))
