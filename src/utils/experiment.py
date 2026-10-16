"""Execution records for reproducible, non-overwriting experiments."""

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import torch
from src.data.protocol import sha256_file
from src.utils.paths import PROJECT_ROOT


def environment_snapshot():
    cpu = platform.processor()
    path = Path("/proc/cpuinfo")
    if cpu.lower() in {"", "x86_64", "amd64", "aarch64"} and path.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in path.read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "packages": {
            name: version(name)
            for name in [
                "torch",
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "wfdb",
                "streamlit",
                "plotly",
            ]
        },
    }


def begin_run(path, config, dataset):
    path.mkdir(parents=True, exist_ok=False)
    hashes = {
        str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
        for folder in ["src", "scripts", "configs"]
        for p in (PROJECT_ROOT / folder).rglob("*")
        if p.suffix in {".py", ".yaml"}
    }
    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "dataset": dataset,
        "environment": environment_snapshot(),
        "source_hashes": hashes,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def finish_run(path, manifest, result):
    manifest.update(
        status="completed",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        result=result,
        checkpoint_sha256=sha256_file(path / "model.pt"),
    )
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
