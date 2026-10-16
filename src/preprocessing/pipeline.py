import numpy as np

from src.data.record import Record
from src.preprocessing.filtering import lowpass_filter, remove_baseline_wander
from src.preprocessing.normalize import normalize_beats
from src.preprocessing.segmentation import extract_beats


class InvalidRecordError(Exception):
    pass


def validate_record(record: Record, expected_fs: int) -> None:
    if record.fs != expected_fs:
        raise InvalidRecordError(f"{record.record_id}: expected fs={expected_fs}, got {record.fs}")
    if len(record.signal) < expected_fs * 10:
        raise InvalidRecordError(f"{record.record_id}: signal too short ({len(record.signal)} samples)")
    if not np.isfinite(record.signal).all():
        raise InvalidRecordError(f"{record.record_id}: signal contains NaN/Inf")


def process_record(record: Record, cfg: dict) -> tuple:
    """Record load -> validation -> baseline removal -> (optional denoise) ->
    segmentation -> normalization. Returns (beats, symbols)."""
    validate_record(record, cfg["sample_rate_hz"])

    signal = remove_baseline_wander(
        record.signal,
        record.fs,
        cfg["baseline_wander"]["window_ms_short"],
        cfg["baseline_wander"]["window_ms_long"],
    )
    if cfg["denoise"]["enabled"]:
        signal = lowpass_filter(signal, record.fs, cfg["denoise"]["lowpass_hz"])

    pre = cfg["beat_window"]["pre_r_samples"]
    post = cfg["beat_window"]["post_r_samples"]
    beats, symbols = extract_beats(signal, record.ann_samples, record.ann_symbols, pre, post)
    beats = normalize_beats(beats, cfg["normalization"]["clip_std"])

    return beats, symbols
