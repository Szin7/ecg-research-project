import numpy as np
from scipy.signal import welch


def extract_features(beat: np.ndarray, fs: int = 360) -> np.ndarray:
    """Hand-engineered per-beat features for the classical baseline:
    amplitude statistics, a crude QRS-width proxy, and coarse band power.
    No RR-interval context here -- morphology alone is what this baseline
    has to work with, unlike the CNN which sees the raw waveform.
    """
    mean, std = beat.mean(), beat.std()
    peak, trough = beat.max(), beat.min()
    z = (beat - mean) / (std + 1e-8)
    skew = float(np.mean(z**3))
    kurtosis = float(np.mean(z**4))

    # crude QRS-width proxy: how much of the beat sits above half the peak
    qrs_width = float((beat > peak / 2).sum()) / fs

    freqs, psd = welch(beat, fs=fs, nperseg=min(128, len(beat)))
    low = psd[(freqs >= 0.5) & (freqs < 8)].sum()
    mid = psd[(freqs >= 8) & (freqs < 20)].sum()
    high = psd[(freqs >= 20) & (freqs < 40)].sum()

    return np.array(
        [mean, std, peak, trough, skew, kurtosis, qrs_width, low, mid, high],
        dtype=np.float32,
    )


def extract_feature_matrix(beats: np.ndarray, fs: int = 360) -> np.ndarray:
    return np.stack([extract_features(b, fs) for b in beats])


FEATURE_NAMES = [
    "mean", "std", "peak", "trough", "skew", "kurtosis",
    "qrs_width", "band_power_low", "band_power_mid", "band_power_high",
]
