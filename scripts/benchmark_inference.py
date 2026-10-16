"""Warm CPU inference timings; run without competing workloads."""

import json
import resource
import time
import numpy as np
import torch
from src.data.protocol import PROTOCOL_ID, sha256_file
from src.models.checkpoint import load_checkpoint
from src.utils.experiment import environment_snapshot
from src.utils.paths import ARTIFACTS_DIR, DATA_DIR

if __name__ == "__main__":
    torch.set_num_threads(2)
    active = json.loads((ARTIFACTS_DIR / "active_study.json").read_text())
    study = ARTIFACTS_DIR / active["study_dir"]
    path = DATA_DIR / "processed" / PROTOCOL_ID / "test.npz"
    data = np.load(
        path if path.exists() else DATA_DIR / "preview" / "v2_test.npz",
        allow_pickle=False,
    )
    rows = []
    for variant in ["morphology", "context", "context_defended"]:
        path = ARTIFACTS_DIR / "models" / f"v2_{variant}.pt"
        loaded = load_checkpoint(path)
        for batch in [1, 256]:
            xb = torch.from_numpy(data["X"][:batch]).unsqueeze(1)
            with torch.inference_mode():
                for _ in range(20):
                    loaded.model(xb)
                durations = []
                for _ in range(100):
                    start = time.perf_counter()
                    loaded.model(xb)
                    durations.append((time.perf_counter() - start) * 1000)
            rows.append(
                {
                    "variant": variant,
                    "batch_size": batch,
                    "iterations": 100,
                    "median_ms_per_batch": float(np.median(durations)),
                    "p95_ms_per_batch": float(np.percentile(durations, 95)),
                    "median_ms_per_beat": float(np.median(durations) / batch),
                    "checkpoint_sha256": sha256_file(path),
                }
            )
    output = {
        "environment": environment_snapshot(),
        "warmup_iterations": 20,
        "results": rows,
        "process_peak_rss_mib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "timing_note": "Excludes preprocessing, transport, cold start, and UI. Memory includes imports and data, not incremental model allocation.",
    }
    (study / "inference_benchmark.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
