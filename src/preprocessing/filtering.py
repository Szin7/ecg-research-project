import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt


def remove_baseline_wander(
    signal: np.ndarray, fs: int, window_ms_short: int = 200, window_ms_long: int = 600
) -> np.ndarray:
    """Two-stage median filter baseline estimate, subtracted from the signal.

    The short window (~200ms) is wide enough to suppress QRS/P waves; running
    a second, longer filter (~600ms) over that result further suppresses the
    T-wave, leaving an estimate of the slow baseline drift alone. Subtracting
    it removes wander without distorting QRS morphology the way a naive
    high-pass filter would.
    """
    k1 = _to_odd_samples(window_ms_short, fs)
    k2 = _to_odd_samples(window_ms_long, fs)

    stage1 = median_filter(signal, size=k1, mode="nearest")
    baseline = median_filter(stage1, size=k2, mode="nearest")

    return (signal - baseline).astype(np.float32)


def lowpass_filter(signal: np.ndarray, fs: int, cutoff_hz: float = 45.0, order: int = 4) -> np.ndarray:
    """Optional powerline/EMG noise suppression -- off by default (see
    configs/dataset.yaml), MIT-BIH doesn't need it for beat classification."""
    nyquist = fs / 2
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return filtfilt(b, a, signal).astype(np.float32)


def _to_odd_samples(window_ms: int, fs: int) -> int:
    n = int(round(window_ms * fs / 1000))
    return n if n % 2 == 1 else n + 1
