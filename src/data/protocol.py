"""Subject-disjoint MIT-BIH preparation with past-only RR context."""

import hashlib
import json
from pathlib import Path
import numpy as np
from src.data.mitbih import load_record
from src.data.splits import AAMI_CLASSES, _aami_superclass, assert_no_record_overlap
from src.preprocessing.pipeline import process_record
from src.preprocessing.segmentation import valid_beat_mask
from src.utils.paths import DATA_DIR

PROTOCOL_ID = "mitbih_subject_disjoint_v2"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_subject_disjoint(split):
    assert_no_record_overlap(split)
    groups = {
        name: {"201_202" if str(r) in {"201", "202"} else str(r) for r in records}
        for name, records in split.items()
    }
    names = list(groups)
    for i, name in enumerate(names):
        for other in names[i + 1 :]:
            if groups[name] & groups[other]:
                raise ValueError(f"Subject overlap between {name} and {other}")


def past_rr_features(samples, fs):
    samples = np.asarray(samples)
    if np.any(np.diff(samples) <= 0):
        raise ValueError("Beat timestamps must increase strictly")
    intervals = np.clip(np.diff(samples) / fs, 0.1, 3)
    features = np.zeros((len(samples), 4), dtype=np.float32)
    for i in range(1, len(samples)):
        previous = intervals[i - 1]
        local = intervals[max(0, i - 10) : i].mean()
        older = intervals[i - 2] if i > 1 else previous
        features[i] = np.log(
            np.clip([previous, local, previous / local, previous / older], 0.1, 10)
        )
    return features


def prepare_protocol(cfg):
    split = cfg["records"]
    assert_subject_disjoint(split)
    if any("201" in set(map(str, records)) for records in split.values()):
        raise ValueError("V2 excludes record 201")
    output = DATA_DIR / "processed" / PROTOCOL_ID
    output.mkdir(parents=True, exist_ok=True)
    preprocessing = cfg["preprocessing"]
    beat_symbols = {
        s for symbols in preprocessing["label_map"].values() for s in symbols
    }
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "config": cfg,
        "source_hashes": {},
        "splits": {},
    }
    for name, records in split.items():
        arrays, labels, groups, positions = [], [], [], []
        for record_id in records:
            record = load_record(str(record_id))
            keep = np.array([s in beat_symbols for s in record.ann_symbols])
            record.ann_samples = record.ann_samples[keep]
            record.ann_symbols = [s for s, k in zip(record.ann_symbols, keep) if k]
            rr = past_rr_features(record.ann_samples, record.fs)
            pre = preprocessing["beat_window"]["pre_r_samples"]
            post = preprocessing["beat_window"]["post_r_samples"]
            valid = valid_beat_mask(record.ann_samples, len(record.signal), pre, post)
            waves, symbols = process_record(record, preprocessing)
            rr, peaks = rr[valid], record.ann_samples[valid]
            classes = [_aami_superclass(s, preprocessing) for s in symbols]
            keep = np.array([c in AAMI_CLASSES for c in classes])
            arrays.append(np.concatenate([waves[keep], rr[keep]], axis=1))
            labels.append(
                np.array([AAMI_CLASSES.index(c) for c in classes if c in AAMI_CLASSES])
            )
            groups.append(np.full(keep.sum(), record_id, dtype=np.int64))
            positions.append(peaks[keep])
            for ext in ("dat", "hea", "atr"):
                path = DATA_DIR / "raw" / "mitdb" / f"{record_id}.{ext}"
                manifest["source_hashes"][path.name] = sha256_file(path)
            print(name, record_id, int(keep.sum()), flush=True)
        X, ya, group, peaks = map(np.concatenate, (arrays, labels, groups, positions))
        path = output / f"{name}.npz"
        np.savez_compressed(
            path,
            X=X.astype(np.float32),
            y=(ya > 0).astype(np.int64),
            y_aami=ya,
            groups=group,
            peaks=peaks,
        )
        manifest["splits"][name] = {
            "records": records,
            "n_beats": len(X),
            "class_counts": np.bincount(ya, minlength=4).tolist(),
            "sha256": sha256_file(path),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
