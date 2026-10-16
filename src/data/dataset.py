import numpy as np

from src.data.source import get_data_source
from src.data.splits import map_symbol_to_label
from src.preprocessing.pipeline import InvalidRecordError, process_record


def build_beats(mode: str, record_ids: list, cfg: dict, label_fn=None):
    """Load, preprocess, segment, and label every beat from the given
    records. A record that fails validation is skipped rather than
    crashing the whole build -- that's what validate_record is for.

    label_fn defaults to the binary task; pass map_symbol_to_aami_class
    for the secondary multi-class experiment instead.
    """
    label_fn = label_fn or map_symbol_to_label
    source = get_data_source(mode)
    beat_len = cfg["beat_window"]["pre_r_samples"] + cfg["beat_window"]["post_r_samples"]
    all_beats, all_labels, all_groups = [], [], []

    for record_id in record_ids:
        try:
            record = source.load_record(str(record_id))
            beats, symbols = process_record(record, cfg)
        except InvalidRecordError as e:
            print(f"skipping {record_id}: {e}")
            continue

        for beat, symbol in zip(beats, symbols):
            label = label_fn(symbol, cfg)
            if label is None:
                continue
            all_beats.append(beat)
            all_labels.append(label)
            all_groups.append(record_id)

    X = np.stack(all_beats) if all_beats else np.zeros((0, beat_len), dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    groups = np.array(all_groups)
    return X, y, groups


def class_distribution(y: np.ndarray) -> dict:
    values, counts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def compute_pos_weight(y: np.ndarray) -> float:
    dist = class_distribution(y)
    n_pos, n_neg = dist.get(1, 0), dist.get(0, 0)
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency class weights for CrossEntropyLoss, the
    multi-class analog of compute_pos_weight -- same idea sklearn's
    class_weight='balanced' uses."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return weights.astype(np.float32)
