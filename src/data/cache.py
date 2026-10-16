import hashlib
import inspect
import json
import os
import tempfile

import numpy as np

from src.data.dataset import build_beats
from src.data.splits import map_symbol_to_label
from src.utils.paths import DATA_DIR, PROJECT_ROOT

PROCESSED_DIR = DATA_DIR / "processed"


def _fingerprint(mode, record_ids, cfg, label_fn):
    label_fn = label_fn or map_symbol_to_label
    digest = hashlib.sha256(json.dumps({"mode": mode, "records": list(record_ids), "config": cfg},
                                       sort_keys=True).encode())
    digest.update(inspect.getsource(label_fn).encode())
    source_files = sorted((PROJECT_ROOT / "src" / "preprocessing").glob("*.py"))
    source_files += [PROJECT_ROOT / "src" / "data" / name for name in ("dataset.py", "mitbih.py", "splits.py")]
    for record_id in record_ids:
        source_files.extend([DATA_DIR / "demo" / f"{record_id}.npz"] if mode == "demo" else
                            [DATA_DIR / "raw" / "mitdb" / f"{record_id}.{ext}" for ext in ("dat", "hea", "atr")])
    for path in source_files:
        digest.update(path.name.encode())
        if path.exists():
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def load_or_build_split(mode, split_name, record_ids, cfg, label_fn=None, cache_suffix="", force_rebuild=False):
    """Reuse beats only when inputs, preprocessing and label mapping match.

    Legacy caches have no fingerprint and are rebuilt for CLI experiments.
    The dashboard can still inspect their original saved results.
    """
    suffix = f"_{cache_suffix}" if cache_suffix else ""
    cache_path = PROCESSED_DIR / f"{mode}{suffix}_{split_name}.npz"
    fingerprint = _fingerprint(mode, record_ids, cfg, label_fn)
    if cache_path.exists() and not force_rebuild:
        with np.load(cache_path, allow_pickle=False) as data:
            if "fingerprint" in data and str(data["fingerprint"].item()) == fingerprint:
                return data["X"], data["y"], data["groups"]

    X, y, groups = build_beats(mode, record_ids, cfg, label_fn=label_fn)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=PROCESSED_DIR, suffix=".npz", delete=False) as file:
        temporary = file.name
    try:
        np.savez(temporary, X=X, y=y, groups=groups, fingerprint=fingerprint)
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return X, y, groups
