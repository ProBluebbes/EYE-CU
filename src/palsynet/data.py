"""PyTorch datasets of 10-second video segments for palsynet.

Each sample is a segment of `segment_seconds` of video sampled at `sample_fps`,
returned as a float tensor of shape (T, C, H, W) with T = segment_seconds * sample_fps
(default 5 s x 10 fps = 50 frames). This layout suits frame-sequence models
(per-frame encoder + temporal model); permute in your model if it wants (C, T, H, W).

* Training: `train_segments_per_video` random segments per video per epoch, so long
  and short videos contribute equally (lengths range ~6 s to ~335 s).
* Evaluation: `eval_segments_per_video` evenly spaced, deterministic segments per
  video; `video_idx` is returned so predictions can be aggregated per video.
* Videos shorter than one segment (4 affected videos are 5.7-9.2 s) are sampled
  over their whole length with the same T frames, i.e. slightly faster playback.
  This is deliberate: padding with repeated/blank frames would only ever occur in
  the affected class and would hand the model a label shortcut.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .splits import VideoInfo

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_frames(path: str, indices: np.ndarray) -> np.ndarray:
    """Decode the given (sorted, possibly repeated) frame indices -> uint8 (T, H, W, 3) RGB.

    Seeks once to the first index, then decodes sequentially (grab is cheap; only
    wanted frames are converted).
    """
    cap = cv2.VideoCapture(path)
    first, last = int(indices[0]), int(indices[-1])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    wanted = set(indices.tolist())
    decoded: dict[int, np.ndarray] = {}
    for i in range(first, last + 1):
        if not cap.grab():
            break
        if i in wanted:
            ok, f = cap.retrieve()
            if ok:
                decoded[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    cap.release()
    if not decoded:
        raise RuntimeError(f"No frames decoded from {path} at {first}..{last}")
    # Fall back to the nearest decoded frame if the container under-reported its length.
    keys = np.array(sorted(decoded))
    out = [decoded[i] if i in decoded else decoded[int(keys[np.abs(keys - i).argmin()])] for i in indices.tolist()]
    return np.stack(out)


def build_transforms(train: bool, size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD, hflip: bool = True, crop_scale=(0.7, 1.0)):
    """Transforms on a (T, C, H, W) uint8 tensor; v2 applies the same random params to every frame."""
    if train:
        ops = [v2.RandomResizedCrop(size, scale=tuple(crop_scale), ratio=(0.9, 1.1), antialias=True)]
        if hflip:
            # Palsy can affect either side, so mirroring keeps the label valid.
            ops.append(v2.RandomHorizontalFlip())
        ops.append(v2.ColorJitter(0.2, 0.2, 0.2, 0.02))
    else:
        ops = [v2.Resize(size, antialias=True), v2.CenterCrop(size)]
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    return v2.Compose(ops)


class VideoSegmentDataset(Dataset):
    def __init__(
        self,
        videos: list[VideoInfo],
        raw_root: Path,
        train: bool,
        segment_seconds: float = 5.0,
        sample_fps: float = 10.0,
        segments_per_video: int = 4,
        transform=None,
        seed: int = 0,
    ):
        self.videos = videos
        self.raw_root = Path(raw_root)
        self.train = train
        self.num_frames = max(1, round(segment_seconds * sample_fps))
        self.segment_seconds = segment_seconds
        self.segments_per_video = segments_per_video
        self.transform = transform
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.videos) * self.segments_per_video

    def frame_indices(self, v: VideoInfo, seg_idx: int, index: int) -> tuple[np.ndarray, int]:
        span = min(round(self.segment_seconds * v.fps), v.num_frames)  # whole video if it is short
        max_start = v.num_frames - span
        if self.train:
            start = int(np.random.default_rng((self.seed, self.epoch, index)).integers(0, max_start + 1))
        elif self.segments_per_video == 1:
            start = max_start // 2
        else:
            start = round(seg_idx * max_start / (self.segments_per_video - 1))
        idx = np.linspace(start, start + span - 1, self.num_frames).round().astype(int)
        return idx, start

    def __getitem__(self, index: int):
        vid_idx, seg_idx = divmod(index, self.segments_per_video)
        v = self.videos[vid_idx]
        idx, start = self.frame_indices(v, seg_idx, index)
        frames = read_frames(str(self.raw_root / v.path), idx)
        x = torch.from_numpy(frames).permute(0, 3, 1, 2)  # T, C, H, W (uint8)
        if self.transform is not None:
            x = self.transform(x)
        return {
            "video": x,
            "label": torch.tensor(v.label),
            "video_idx": torch.tensor(vid_idx),
            "start_sec": torch.tensor(start / v.fps, dtype=torch.float32),
        }


def _seed_worker(worker_id: int) -> None:
    np.random.seed(torch.initial_seed() % 2**32)
    cv2.setNumThreads(1)


def build_loaders(cfg: dict, train_v, val_v, test_v, raw_root: Path):
    d = cfg["data"]
    if d.get("input", "video") == "landmarks":
        from .landmarks import build_landmark_loaders
        return build_landmark_loaders(cfg, train_v, val_v, test_v)
    common = dict(segment_seconds=d["segment_seconds"], sample_fps=d["sample_fps"])
    norm = dict(size=d["size"], mean=d.get("mean", IMAGENET_MEAN), std=d.get("std", IMAGENET_STD))
    train_ds = VideoSegmentDataset(
        train_v, raw_root, train=True, segments_per_video=d["train_segments_per_video"],
        transform=build_transforms(True, hflip=d["hflip"], crop_scale=d.get("crop_scale", (0.7, 1.0)), **norm), seed=cfg["seed"], **common,
    )
    eval_tf = build_transforms(False, **norm)
    val_ds = VideoSegmentDataset(val_v, raw_root, train=False, segments_per_video=d["eval_segments_per_video"], transform=eval_tf, **common)
    test_ds = VideoSegmentDataset(test_v, raw_root, train=False, segments_per_video=d["eval_segments_per_video"], transform=eval_tf, **common)

    g = torch.Generator().manual_seed(cfg["seed"])
    kw = dict(batch_size=d["batch_size"], num_workers=d["num_workers"], worker_init_fn=_seed_worker,
              persistent_workers=d["num_workers"] > 0, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, shuffle=True, generator=g, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )
