import numpy as np


def normalize_beat(beat: np.ndarray, clip_std: float = 6.0) -> np.ndarray:
    mean, std = beat.mean(), beat.std()
    std = std if std > 1e-6 else 1.0
    normalized = (beat - mean) / std
    return np.clip(normalized, -clip_std, clip_std).astype(np.float32)


def normalize_beats(beats: np.ndarray, clip_std: float = 6.0) -> np.ndarray:
    return np.stack([normalize_beat(b, clip_std) for b in beats])
