import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def detect_r_peaks(signal: np.ndarray, fs: int, chunk_seconds: float = 10.0) -> np.ndarray:
    """Simplified Pan-Tompkins QRS detector: bandpass -> derivative -> square
    -> moving-window integration -> peak picking with a refractory period.

    The threshold is recomputed per chunk (~10s of signal) rather than once
    from the single global max: on a real 30-minute recording, one unusually
    tall beat or artifact anywhere in the signal would otherwise set a
    threshold too high for every ordinary beat elsewhere to clear -- this
    showed up as a real bug (71 detections vs. 2274 annotated beats on a
    real record) before switching to a locally-adaptive threshold, which is
    the same reason the original Pan-Tompkins algorithm tracks a running
    signal/noise estimate instead of a single fixed cutoff.

    Used for the synthetic demo path (no expert annotations available) and
    as an independent check against expert annotations on real MIT-BIH
    records -- the real pipeline segments beats from the annotations
    themselves, this exists to show the detection step actually works.
    """
    nyquist = fs / 2
    b, a = butter(2, [5 / nyquist, 15 / nyquist], btype="band")
    filtered = filtfilt(b, a, signal)

    derivative = np.diff(filtered, prepend=filtered[0])
    squared = derivative**2

    window = max(int(0.15 * fs), 1)
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")

    min_distance = int(0.25 * fs)  # refractory period, ~240 bpm ceiling
    chunk_size = max(int(chunk_seconds * fs), min_distance * 4)

    peaks = []
    for start in range(0, len(integrated), chunk_size):
        chunk = integrated[start : start + chunk_size]
        if chunk.max() <= 0:
            continue
        threshold = 0.35 * chunk.max()
        local_peaks, _ = find_peaks(chunk, height=threshold, distance=min_distance)
        peaks.append(local_peaks + start)

    all_peaks = np.concatenate(peaks) if peaks else np.array([], dtype=np.int64)
    all_peaks = _drop_close_duplicates(all_peaks, integrated, min_distance)

    return _refine_to_local_extremum(signal, all_peaks, fs)


def _drop_close_duplicates(peaks: np.ndarray, integrated: np.ndarray, min_distance: int) -> np.ndarray:
    """Chunk boundaries are evaluated independently, so two chunks can each
    report a peak from what's really the same beat if it sits near their
    shared edge -- keep whichever of a too-close pair has the larger
    integrated value."""
    if len(peaks) == 0:
        return peaks
    peaks = np.sort(peaks)
    kept = [peaks[0]]
    for p in peaks[1:]:
        if p - kept[-1] >= min_distance:
            kept.append(p)
        elif integrated[p] > integrated[kept[-1]]:
            kept[-1] = p
    return np.array(kept, dtype=np.int64)


def _refine_to_local_extremum(signal: np.ndarray, coarse_peaks: np.ndarray, fs: int, search_ms: int = 50) -> np.ndarray:
    """The integrated signal's peak lags the true R-peak by roughly the
    integration window; search a small neighborhood in the original signal."""
    radius = int(search_ms * fs / 1000)
    refined = []
    for p in coarse_peaks:
        lo, hi = max(0, p - radius), min(len(signal), p + radius)
        if hi <= lo:
            continue
        refined.append(lo + int(np.argmax(np.abs(signal[lo:hi]))))
    return np.array(refined, dtype=np.int64)
