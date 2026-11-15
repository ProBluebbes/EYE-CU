"""Optimizer parameter groups: per-part learning-rate multipliers and no weight decay on
biases, norm weights, positional embeddings and tokens."""
from __future__ import annotations

import torch.nn as nn

NO_DECAY_KEYS = ("pos_", "cls_token", "class_token")


def param_groups(model: nn.Module, lr: float, weight_decay: float, lr_mult: dict[str, float] | None = None) -> list[dict]:
    """lr_mult maps parameter-name prefixes to LR multipliers, e.g. {"head": 10, "pos_temporal": 10}.
    The longest matching prefix wins; unmatched parameters use the base lr."""
    lr_mult = lr_mult or {}
    groups: dict[tuple[float, float], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        matches = [k for k in lr_mult if name.startswith(k)]
        mult = lr_mult[max(matches, key=len)] if matches else 1.0
        no_decay = p.ndim <= 1 or any(k in name for k in NO_DECAY_KEYS)
        groups.setdefault((mult, 0.0 if no_decay else weight_decay), []).append(p)
    return [{"params": ps, "lr": lr * m, "weight_decay": wd} for (m, wd), ps in sorted(groups.items())]
