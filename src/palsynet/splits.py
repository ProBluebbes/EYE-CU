"""Video index + subject-level (one video = one person) stratified K-fold splits."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from sklearn.model_selection import StratifiedKFold

CLASSES = ["unaffected", "affected"]  # 0 = unaffected, 1 = affected (positive class)


@dataclass
class VideoInfo:
    video_id: str  # "<label>_<stem>"; stems repeat across classes
    path: str  # relative to the raw data root
    label: int
    num_frames: int
    fps: float
    fold: int = -1


def index_videos(raw_root: Path) -> list[VideoInfo]:
    videos = []
    for label_idx, cls in enumerate(CLASSES):
        files = sorted((raw_root / cls).glob("*.mp4"), key=lambda p: (len(p.stem), p.stem))
        for f in files:
            cap = cv2.VideoCapture(str(f))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            cap.release()
            if n <= 0:
                raise RuntimeError(f"Could not read {f}")
            videos.append(VideoInfo(f"{cls}_{f.stem}", str(f.relative_to(raw_root)), label_idx, n, fps))
    return videos


def assign_folds(videos: list[VideoInfo], n_folds: int, seed: int) -> None:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = [v.label for v in videos]
    for k, (_, test_idx) in enumerate(skf.split(labels, labels)):
        for i in test_idx:
            videos[i].fold = k


def save_splits(videos: list[VideoInfo], path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "videos": [asdict(v) for v in videos]}, indent=2))


def load_splits(path: Path) -> tuple[dict, list[VideoInfo]]:
    d = json.loads(Path(path).read_text())
    return d["meta"], [VideoInfo(**v) for v in d["videos"]]


def split_for_fold(videos: list[VideoInfo], test_fold: int, n_folds: int):
    """test = fold k, val = fold k+1 (mod n), train = the rest. No subject appears in two splits."""
    val_fold = (test_fold + 1) % n_folds
    train = [v for v in videos if v.fold not in (test_fold, val_fold)]
    val = [v for v in videos if v.fold == val_fold]
    test = [v for v in videos if v.fold == test_fold]
    return train, val, test
