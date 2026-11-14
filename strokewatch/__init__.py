"""StrokeWatch: webcam facial-droop change monitor (hackathon demo, not a medical device)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, sets: list[str] | None) -> dict:
    """Apply CLI overrides like ["detector.threshold=3.5", "vision.lip_gap_max=0.1"]."""
    for item in sets or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(raw)
    return cfg
