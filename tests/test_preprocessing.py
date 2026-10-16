import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.record import Record
from src.data.synthetic import generate_demo_dataset, load_record
from src.preprocessing.filtering import remove_baseline_wander
from src.preprocessing.normalize import normalize_beat, normalize_beats
from src.preprocessing.pipeline import InvalidRecordError, process_record, validate_record
from src.preprocessing.qrs_detect import detect_r_peaks
from src.preprocessing.segmentation import extract_beats, valid_beat_mask
from src.utils.config import load_config


def _demo_record(n_records=1, beats_per_record=40, seed=0, record_id="demo00"):
    tmp = Path(tempfile.mkdtemp())
    generate_demo_dataset(tmp, n_records=n_records, beats_per_record=beats_per_record, seed=seed)
    return load_record(record_id, tmp)


def test_baseline_wander_removal_reduces_slow_drift():
    record = _demo_record(seed=11, beats_per_record=60)
    corrected = remove_baseline_wander(record.signal, record.fs)

    def slow_trend_power(x, fs):
        window = int(1.0 * fs)
        trend = np.convolve(x, np.ones(window) / window, mode="valid")
        return np.var(trend)

    assert slow_trend_power(corrected, record.fs) < 0.5 * slow_trend_power(record.signal, record.fs)


def test_valid_beat_mask_drops_edge_beats():
    r_peaks = np.array([10, 100, 5000, 5990])
    mask = valid_beat_mask(r_peaks, signal_length=6000, pre=90, post=160)
    assert mask.tolist() == [False, True, True, False]


def test_extract_beats_shape_and_no_nans():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=3600).astype(np.float32)
    r_peaks = np.arange(200, 3400, 300)
    symbols = ["N"] * len(r_peaks)

    beats, kept_symbols = extract_beats(signal, r_peaks, symbols, pre=90, post=160)

    assert beats.shape == (len(r_peaks), 250)
    assert len(kept_symbols) == beats.shape[0]
    assert not np.isnan(beats).any()


def test_normalize_beat_is_zero_mean_unit_std():
    rng = np.random.default_rng(1)
    beat = rng.normal(loc=5.0, scale=2.0, size=250).astype(np.float32)

    normalized = normalize_beat(beat)

    assert abs(normalized.mean()) < 1e-3
    assert abs(normalized.std() - 1.0) < 1e-3


def test_normalize_beat_handles_flat_signal():
    flat = np.zeros(250, dtype=np.float32)
    normalized = normalize_beat(flat)
    assert np.isfinite(normalized).all()


def test_normalize_beats_respects_clip_bound():
    rng = np.random.default_rng(2)
    beats = rng.normal(size=(5, 250)).astype(np.float32)
    beats[0, 0] = 1000.0

    normalized = normalize_beats(beats, clip_std=6.0)

    assert np.abs(normalized).max() <= 6.0 + 1e-5


def test_qrs_detector_finds_approximately_right_beat_count():
    record = _demo_record(seed=7, beats_per_record=40)
    detected = detect_r_peaks(record.signal, record.fs)
    assert abs(len(detected) - len(record.ann_samples)) <= 5


def test_qrs_detector_is_not_broken_by_one_outlier_spike():
    # Regression test: a single unusually tall beat (or artifact) anywhere
    # in a long recording used to set a threshold too high for every
    # ordinary beat elsewhere to clear, when the threshold was computed
    # once from the signal's global max. Simulate a long run of ordinary
    # beats plus one 10x-amplitude spike partway through, and check most
    # of the ordinary beats still get detected.
    record = _demo_record(seed=13, beats_per_record=150)
    signal = record.signal.copy()
    spike_center = len(signal) // 2
    signal[spike_center - 5 : spike_center + 5] *= 10

    detected = detect_r_peaks(signal, record.fs)

    assert len(detected) > 0.8 * len(record.ann_samples)


def test_validate_record_rejects_wrong_sample_rate():
    record = Record("x", np.zeros(4000, dtype=np.float32), fs=250, ann_samples=np.array([]), ann_symbols=[])
    with pytest.raises(InvalidRecordError):
        validate_record(record, expected_fs=360)


def test_validate_record_rejects_nan_signal():
    signal = np.zeros(4000, dtype=np.float32)
    signal[10] = np.nan
    record = Record("x", signal, fs=360, ann_samples=np.array([]), ann_symbols=[])
    with pytest.raises(InvalidRecordError):
        validate_record(record, expected_fs=360)


def test_process_record_end_to_end_on_demo_data():
    record = _demo_record(seed=3, beats_per_record=40)
    cfg = load_config("dataset.yaml")

    beats, symbols = process_record(record, cfg)

    expected_len = cfg["beat_window"]["pre_r_samples"] + cfg["beat_window"]["post_r_samples"]
    assert beats.ndim == 2
    assert beats.shape[1] == expected_len
    assert len(symbols) == beats.shape[0]
    assert not np.isnan(beats).any()
