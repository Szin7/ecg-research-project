from pathlib import Path

import numpy as np
import wfdb

from src.data.record import Record
from src.utils.paths import DATA_DIR

RAW_DIR = DATA_DIR / "raw" / "mitdb"


def available_records(raw_dir: Path = RAW_DIR) -> list:
    return sorted({p.stem for p in raw_dir.glob("*.dat")})


def load_record(record_id: str, raw_dir: Path = RAW_DIR) -> Record:
    path = str(raw_dir / record_id)
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    # MLII is the conventional lead for QRS morphology in MIT-BIH beat
    # classification studies; present in all but a handful of records.
    lead = record.sig_name.index("MLII") if "MLII" in record.sig_name else 0
    signal = record.p_signal[:, lead].astype(np.float32)

    return Record(
        record_id=record_id,
        signal=signal,
        fs=record.fs,
        ann_samples=ann.sample,
        ann_symbols=list(ann.symbol),
    )
