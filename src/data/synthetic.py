import json
from pathlib import Path

import numpy as np

from src.data.record import Record
from src.utils.paths import DATA_DIR

FS = 360
DEMO_DIR = DATA_DIR / "demo"

# Parametric sum-of-Gaussians synthetic beat generator. This is a development
# fixture for exercising the pipeline without a download, not a physiological
# simulator -- shapes are loosely realistic stand-ins, not clinical waveforms.


def _gaussian(t, center, amplitude, width):
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width**2))


def _synthesize_beat(rng, label: str, n_samples: int, r_offset: int, fs: int = FS) -> np.ndarray:
    t = np.arange(n_samples) / fs
    center = r_offset / fs

    if label == "N":
        p_amp, qrs_amp, t_amp, qrs_width = 0.15, 1.0, 0.3, 0.012
    else:
        # wider/taller QRS, flattened T-wave -- a stand-in for ventricular
        # ectopic morphology, not a claim about real pathophysiology
        p_amp, qrs_amp, t_amp, qrs_width = 0.02, 1.3, -0.25, 0.028

    beat = _gaussian(t, center - 0.16, p_amp, 0.02)
    beat += _gaussian(t, center, qrs_amp, qrs_width)
    beat += _gaussian(t, center + 0.28, t_amp, 0.05)
    beat += rng.normal(0, 0.02, size=n_samples)
    return beat.astype(np.float32)


def _synthesize_record(rng, beat_labels: list, fs: int = FS):
    mean_rr = 0.8
    chunks, r_peaks = [], []
    cursor = 0
    for label in beat_labels:
        rr = max(mean_rr + rng.normal(0, 0.05), 0.4)
        n_samples = int(rr * fs)
        chunks.append(_synthesize_beat(rng, label, n_samples, r_offset=90, fs=fs))
        r_peaks.append(cursor + 90)
        cursor += n_samples
    return np.concatenate(chunks).astype(np.float32), np.array(r_peaks)


def _add_baseline_wander(signal: np.ndarray, rng, fs: int = FS) -> np.ndarray:
    t = np.arange(len(signal)) / fs
    freq = rng.uniform(0.15, 0.3)  # respiration-like drift
    drift = 0.25 * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return (signal + drift).astype(np.float32)


def generate_demo_dataset(
    out_dir: Path = DEMO_DIR, n_records: int = 6, beats_per_record: int = 80, seed: int = 42
) -> None:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record_ids = []
    for rec_idx in range(n_records):
        record_id = f"demo{rec_idx:02d}"
        arrhythmia_rate = rng.uniform(0.1, 0.4)
        labels = ["V" if rng.random() < arrhythmia_rate else "N" for _ in range(beats_per_record)]

        signal, r_peaks = _synthesize_record(rng, labels)
        signal = _add_baseline_wander(signal, rng)

        np.savez(
            out_dir / f"{record_id}.npz",
            signal=signal,
            r_peaks=r_peaks,
            symbols=np.array(labels),
        )
        record_ids.append(record_id)

    (out_dir / "manifest.json").write_text(json.dumps({"records": record_ids, "fs": FS}, indent=2))


def available_records(demo_dir: Path = DEMO_DIR) -> list:
    manifest = demo_dir / "manifest.json"
    if not manifest.exists():
        return []
    return json.loads(manifest.read_text())["records"]


def load_record(record_id: str, demo_dir: Path = DEMO_DIR) -> Record:
    npz = np.load(demo_dir / f"{record_id}.npz")
    return Record(
        record_id=record_id,
        signal=npz["signal"],
        fs=FS,
        ann_samples=npz["r_peaks"],
        ann_symbols=list(npz["symbols"]),
    )
