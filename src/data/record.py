from dataclasses import dataclass

import numpy as np


@dataclass
class Record:
    record_id: str
    signal: np.ndarray
    fs: int
    ann_samples: np.ndarray
    ann_symbols: list
