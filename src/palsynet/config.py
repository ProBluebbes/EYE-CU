"""YAML configs with single inheritance: a file may set `base: other.yaml` (relative to itself)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _merge(load_config(path.parent / base), cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """e.g. ["train.epochs=2", "model.name=mc3_18"]"""
    for o in overrides:
        key, val = o.split("=", 1)
        d = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[leaf] = yaml.safe_load(val)
