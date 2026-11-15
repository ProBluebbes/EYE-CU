"""Train / eval loops and segment- and video-level metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, scheduler, device, scaler=None, log_every: int = 20, grad_accum: int = 1):
    """One epoch; with grad_accum > 1 the optimizer (and scheduler) step every `grad_accum` batches."""
    model.train()
    total, n = 0.0, 0
    num_steps = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        x, y = batch["video"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=scaler is not None):
            loss = F.cross_entropy(model(x), y)
        scaled = loss / grad_accum
        (scaler.scale(scaled) if scaler is not None else scaled).backward()
        if (step + 1) % grad_accum == 0 or step + 1 == num_steps:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
        if log_every and step % log_every == 0:
            print(f"  step {step}/{len(loader)} loss {loss.item():.4f}")
    return total / max(n, 1)


def _metrics(y_true, prob_pos, threshold=0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob_pos) >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }
    out["auc"] = float(roc_auc_score(y_true, prob_pos)) if len(set(y_true.tolist())) == 2 else float("nan")
    return out


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Returns segment-level metrics, video-level metrics (mean of segment probabilities), and per-video predictions."""
    model.eval()
    probs, labels, vids, loss_sum = [], [], [], 0.0
    for batch in loader:
        x, y = batch["video"].to(device), batch["label"].to(device)
        logits = model(x).float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        probs.append(logits.softmax(-1)[:, 1].cpu())
        labels.append(batch["label"])
        vids.append(batch["video_idx"])
    probs, labels, vids = torch.cat(probs).numpy(), torch.cat(labels).numpy(), torch.cat(vids).numpy()

    per_video = []
    for v in np.unique(vids):
        m = vids == v
        per_video.append({"video_idx": int(v), "label": int(labels[m][0]), "prob_affected": float(probs[m].mean()), "n_segments": int(m.sum())})
    return {
        "loss": loss_sum / max(len(labels), 1),
        "segment": _metrics(labels, probs),
        "video": _metrics([p["label"] for p in per_video], [p["prob_affected"] for p in per_video]),
        "per_video": per_video,
    }
