import yaml

from src.utils.paths import CONFIGS_DIR


def load_config(name: str) -> dict:
    """Load a YAML config by filename, e.g. load_config('training.yaml')."""
    with open(CONFIGS_DIR / name) as f:
        return yaml.safe_load(f)
