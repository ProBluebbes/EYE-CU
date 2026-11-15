"""Index the raw videos and assign subject-level stratified K folds -> data/splits.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # robust to macOS hiding the editable-install .pth

import argparse
from collections import Counter
from pathlib import Path

from palsynet.splits import CLASSES, assign_folds, index_videos, save_splits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw/data"))
    p.add_argument("--out", type=Path, default=Path("data/splits.json"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    videos = index_videos(args.raw)
    assign_folds(videos, args.folds, args.seed)
    save_splits(videos, args.out, {"n_folds": args.folds, "seed": args.seed, "classes": CLASSES})

    print(f"{len(videos)} videos -> {args.out}")
    for k in range(args.folds):
        c = Counter(CLASSES[v.label] for v in videos if v.fold == k)
        print(f"  fold {k}: {dict(c)}")


if __name__ == "__main__":
    main()
