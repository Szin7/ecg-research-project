import numpy as np


def valid_beat_mask(r_peaks: np.ndarray, signal_length: int, pre: int, post: int) -> np.ndarray:
    return (r_peaks - pre >= 0) & (r_peaks + post <= signal_length)


def segment_beats(signal: np.ndarray, r_peaks: np.ndarray, pre: int, post: int) -> np.ndarray:
    """Fixed-length window around each R-peak. Callers are expected to have
    already dropped out-of-range peaks with valid_beat_mask -- this stays
    strict rather than silently zero-padding edge beats, which would bake
    an artificial discontinuity into a handful of samples."""
    return np.stack([signal[p - pre : p + post] for p in r_peaks]).astype(np.float32)


def extract_beats(signal: np.ndarray, r_peaks: np.ndarray, symbols: list, pre: int, post: int):
    """Segment every beat whose window fully fits inside the record. Drops
    the one or two beats at the very start/end of a recording that don't."""
    mask = valid_beat_mask(np.asarray(r_peaks), len(signal), pre, post)
    kept_peaks = np.asarray(r_peaks)[mask]
    kept_symbols = [s for s, keep in zip(symbols, mask) if keep]
    beats = segment_beats(signal, kept_peaks, pre, post)
    return beats, kept_symbols
