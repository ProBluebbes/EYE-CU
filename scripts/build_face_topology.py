"""Build the fixed face topology used by the spatial landmark models -> src/palsynet/face_topology.npz:

  layout  (152, 3)  mean frontalised face of the unaffected palsynet videos, made exactly left/right symmetric
                    (x of mirror partners negated, y and z averaged; midline points at x = 0). Used as the 2D
                    face-surface layout for the geometry-image 3D CNN.
  edges   (E, 2)    graph edges between the 152 points (positions in LANDMARK_IDX): MediaPipe's contour
                    connections (lips, eyes, eyebrows, face outline, nose) plus each point's 4 nearest
                    neighbours in the layout (links the regions). For the ST-GCN.

Run once (needs mediapipe and data/aligned/landmarks); training only reads the saved file.
    uv run python scripts/build_face_topology.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarksConnections as C

from palsynet.landmarks import LANDMARK_IDX, MIRROR, load_video_landmarks_3d
from palsynet.splits import load_splits


def main() -> None:
    _, videos = load_splits(Path("data/splits.json"))
    root = Path("data/aligned/landmarks")
    faces = [np.nanmedian(load_video_landmarks_3d(root / Path(v.path).with_suffix(".npz")), 0)
             for v in videos if v.label == 0]
    mean = np.mean(faces, 0)  # (152, 3)
    mir = np.array(MIRROR)
    flip = mean[mir] * np.array([-1, 1, 1])
    layout = (mean + flip) / 2  # exactly symmetric: layout[mir] == layout * [-1, 1, 1]
    layout[mir == np.arange(len(mir)), 0] = 0.0

    pos = {m: i for i, m in enumerate(LANDMARK_IDX)}
    edges = set()
    for conns in (C.FACE_LANDMARKS_LIPS, C.FACE_LANDMARKS_LEFT_EYE, C.FACE_LANDMARKS_RIGHT_EYE,
                  C.FACE_LANDMARKS_LEFT_EYEBROW, C.FACE_LANDMARKS_RIGHT_EYEBROW, C.FACE_LANDMARKS_FACE_OVAL,
                  C.FACE_LANDMARKS_NOSE):
        for e in conns:
            if e.start in pos and e.end in pos and e.start != e.end:
                edges.add(tuple(sorted((pos[e.start], pos[e.end]))))
    n_contour = len(edges)
    d = np.linalg.norm(layout[:, None, :2] - layout[None, :, :2], axis=-1)
    np.fill_diagonal(d, np.inf)
    for i in range(len(layout)):
        for j in np.argsort(d[i])[:4]:
            edges.add(tuple(sorted((i, int(j)))))
    edges = np.array(sorted(edges), dtype=np.int64)
    out = Path("src/palsynet/face_topology.npz")
    np.savez(out, layout=layout.astype(np.float32), edges=edges)
    print(f"{len(faces)} unaffected faces -> layout x [{layout[:, 0].min():.2f}, {layout[:, 0].max():.2f}] "
          f"y [{layout[:, 1].min():.2f}, {layout[:, 1].max():.2f}]; {len(edges)} edges ({n_contour} contour) -> {out}")


if __name__ == "__main__":
    main()
