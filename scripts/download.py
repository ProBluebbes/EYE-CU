"""Download the raw videos of jasir/palsynet-data from the Hugging Face Hub."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "jasir/palsynet-data"
# Pin to a known commit for reproducibility; pass --revision main to get the latest.
DEFAULT_REVISION = "233ae758be37f5af3dba1aea21900bbab75fec58"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--revision", default=DEFAULT_REVISION)
    args = p.parse_args()

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.out,
        allow_patterns=["data/**/*.mp4", "README.md"],
    )
    n = len(list(Path(path).glob("data/*/*.mp4")))
    print(f"Downloaded {n} videos to {path}")


if __name__ == "__main__":
    main()
