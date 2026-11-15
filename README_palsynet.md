# palsynet

Video classification of facial palsy (affected vs. unaffected) on
[`jasir/palsynet-data`](https://huggingface.co/datasets/jasir/palsynet-data) (CC-BY-4.0) with PyTorch.

**Dataset:** 49 MP4s (27 affected, 22 unaffected), one person per video, all 300×300 @ 30 fps, 6 s to 335 s long.

## Setup

```bash
uv sync                                  # install deps into .venv
uv run python scripts/download.py        # videos -> data/raw/data/{affected,unaffected}/*.mp4 (~250 MB, pinned revision)
uv run python scripts/make_splits.py     # stratified 5-fold, subject-level -> data/splits.json
```

Optional face normalisation (used by `configs/r3d_18_aligned.yaml`):

```bash
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
uv run python scripts/preprocess_faces.py --preview runs/face_previews   # -> data/aligned/ (~15 min on 4 processes)
```

## Train / evaluate

```bash
uv run python scripts/train.py --config configs/r3d_18.yaml --fold 0      # test=fold 0, val=fold 1, train=folds 2-4
uv run python scripts/train.py --config configs/vivit_b16.yaml --fold 0 --set train.epochs=30
uv run python scripts/evaluate.py runs/<run>/best.pt                      # writes test_metrics.json + test_predictions.csv
scripts/run_cv.sh configs/r3d_18.yaml                                     # all 5 folds + summary (runs/cv_*/summary.txt)
```

Run folds 0–4 and average the test metrics for a cross-validated estimate. With only ~10 test videos per fold,
a single fold's numbers are very noisy.

## Training on Google Colab

3D CNNs are slow on Apple-Silicon GPUs (r3d_18 with 50-frame segments takes ~10 s per step, so ~27 h per 5-fold run),
so training is set up to run on a Colab GPU:

1. Build the bundle (code, configs, `data/splits.json` and the aligned videos, ~190 MB):
   ```bash
   zip -qr dist/palsynet_colab.zip pyproject.toml README.md src scripts configs notebooks data/splits.json \
       data/aligned/affected data/aligned/unaffected data/aligned/report.csv -x "*/__pycache__/*" "*.pyc" "*.DS_Store"
   ```
2. Upload it to Google Drive as `MyDrive/palsynet/palsynet_colab.zip`.
3. Open `notebooks/palsynet_colab.ipynb` in Colab (*File → Upload notebook*), pick a GPU runtime, and *Run all*.

The notebook unpacks onto the VM's local disk, downloads the raw videos from the Hub, runs a timed smoke test, then
runs 5-fold CV on aligned faces and on raw frames. It writes results to `MyDrive/palsynet/runs/` and compares the two.
It installs nothing: training only needs packages Colab already has. `run_cv.sh` is resumable: rerun with
`CV_DIR=<existing cv_* dir>` and finished folds are skipped. Only `best.pt` is kept per fold.

## Models

| Config | Model | Params | Input per segment (5 s) |
|---|---|---|---|
| `r3d_18.yaml` / `r3d_18_aligned.yaml` | 3D ResNet-18 (raw frames / aligned faces), Kinetics-400 pretrained (`--set model.name=mc3_18` for MC3-18) | 33.2M / 11.5M | 50 × 112², Kinetics norm |
| `r2plus1d_18.yaml` | R(2+1)D-18, Kinetics-400 pretrained; batch 2 × grad-accum 4 | 31.3M | 50 × 112², Kinetics norm |
| `vivit_b16.yaml` | ViViT, joint spatio-temporal attention, init from ImageNet ViT-B/16 | 86.3M | 32 × 160² → 1601 tokens |
| `vivit_b16_aligned.yaml` | ViViT-B/16 on aligned faces: LR 1e-4, 10× LR on tubelet embedding / temporal positions / head (`train.lr_mult`), inflated tubelets | 86.3M | 32 × 160² → 1601 tokens |
| `vivit_small.yaml` | ViViT, joint spatio-temporal attention, from scratch (192-d, 6 layers) | 3.0M | 32 × 112² → 785 tokens |
| `frame_gru.yaml` | 2D ResNet-18 per frame + bidirectional GRU | 12.4M | 50 × 224², ImageNet norm |
| `landmark_tcn.yaml` | **Landmarks only, no pixels:** 152 aligned face landmarks → asymmetry + motion features → dilated temporal CNN. Trains on CPU in minutes; needs `data/aligned/landmarks` | 0.4M | 50 × 152 points × (x, y) |
| `landmark_tcn_2s.yaml` | Landmark TCN on 2 s windows (faster live reaction) | 0.4M | 20 × 152 points × (x, y) |
| `landmark_tcn_3d.yaml` | 2 s landmark TCN on **3D-frontalised, smoothed** landmarks: 5-frame smoothing removes camera-dependent jitter; MediaPipe depth + rigid face points rotate every frame to a frontal view, so head turns no longer look like asymmetry | 0.6M | 20 × 152 points × (x, y, z) |
| `landmark_tcn_3d_vcd.yaml` | **Recommended for webcams.** `landmark_tcn_3d` + ~90 ordinary-people webcam clips from Microsoft VCD2 as extra unaffected data (`scripts/fetch_vcd.py`), classes balanced. Same palsy CV (acc 0.94, AUC 1.00), webcam false alarms 70% → 0% (`scripts/eval_webcam.py`) | 0.6M | 20 × 152 points × (x, y, z) |
| `landmark_tcn_3d_vcd_expr.yaml` | `landmark_tcn_3d_vcd` + 1440 RAVDESS clips (24 actors, 8 emotions, CC BY-NC-SA 4.0, non-commercial) as extra unaffected data (`scripts/fetch_ravdess.py`). Same palsy CV; false alarms on held-out expression clips 3.6% → 0.2%, webcam clips 0% | 0.6M | 20 × 152 points × (x, y, z) |
| `landmark_tcn_3d_vcd_expr_still.yaml` | **Recommended for live use.** `landmark_tcn_3d_vcd_expr` + "still face" augmentation (`data.slow_prob`: 30% of training windows played at 0–50% speed or frozen, both classes). All healthy training clips show people talking, so before this a face held still on a webcam scored ~0.5. Live still-face windows 47% → 0% flagged; palsy CV AUC 1.00, acc 0.90 (two more affected videos just under 0.5) | 0.6M | 20 × 152 points × (x, y, z) |
| `landmark_seq_gru.yaml` | Tried, not better. Memory across windows: the TCN encodes each 2 s window, a causal GRU runs over the last 12 window encodings (1 s apart, ~13 s), history-length matched between classes. Palsy CV acc 0.90 / AUC 1.00 (still-face TCN: the same); webcam false alarms 2.2% (0%); on a saved live session still-face windows went back to 65% flagged (0%). Live script supports it (per-model rolling history, reset when the face is lost) | 0.6M | 12 × 20 × 152 × (x, y, z) |
| `landmark_geom_cnn.yaml` | Tried, not better. Face-surface 3D CNN: per-point features interpolated onto a 32×32 geometry image of a symmetric mean face (`scripts/build_face_topology.py`), residual 3D CNN over time × face height × width. Trained on Colab (`dist/palsynet_landmarks_colab.ipynb`). Palsy CV identical to the still-face TCN (acc 0.90, AUC 1.00, same misses); more healthy false alarms: webcam windows 7.0% (TCN 1.9%), expression windows 1.5% (0.4%), head turned >30° 13% (5%); ~110 ms per prediction (TCN ~12 ms) | 1.0M | 20 × 152 × (x, y, z) → 20 × 32 × 32 |
| `landmark_stgcn.yaml` | Tried, best on palsy CV but worst on healthy webcam faces. Spatio-temporal graph CNN over the face mesh (contour + nearest-neighbour edges, separate left↔right mirror partition). Trained 20 epochs × 64 windows/video on the Mac GPU. Palsy CV acc **0.92 ± 0.04** (23/27 affected, 22/22 unaffected; TCN 0.90 ± 0.07); but healthy webcam windows 5.7% (mean p 0.20 vs 0.08), head turned >30° 31% (5%), and on saved live sessions still-face windows went back to 88–94% flagged (TCN 0%) | 0.6M | 20 × 152 × (x, y, z) |
| `landmark_crnn.yaml` | Landmarks only: same features, 2 temporal conv layers → bidirectional GRU over the 5 s segment (CNN + RNN) | 0.5M | 50 × 152 points × (x, y) |

**ViViT** follows Model 1 of Arnab et al., *ViViT: A Video Vision Transformer* (2021): 2×16×16 tubelet embedding and
one pre-norm transformer in which every space-time token attends to every other (no factorisation), classified
from a CLS token. Positional embeddings are learned spatial + temporal terms so the ViT-B/16 spatial table can be
reused (bicubically resized to the patch grid). The ViT init puts the 2D patch filter on the tubelet's central frame
(`tubelet_init: inflate` averages it over time instead), and it reproduces torchvision's ViT-B/16 features exactly
for a single-tubelet input. Full attention costs grow with tokens², so resolution and frame count are set to keep
tokens ≈ 1.6k; raising `size` to 224 doubles the tokens and roughly quadruples attention cost.

Configs inherit from `configs/default.yaml` via `base:`; override any key with `--set a.b=value`.
`train.grad_accum` steps the optimizer every N batches when memory forces a small batch.

To add a model, register a builder in `src/palsynet/models.py` that maps `(B, T, C, H, W)` to logits `(B, 2)`:

```python
@register("my_model")
def _my_model(num_classes=2, data_cfg=None, **kw):   # data_cfg = the config's data: section
    return MyModel(num_classes=num_classes, **kw)
```

## Face landmarks and normalisation

`scripts/preprocess_faces.py` runs MediaPipe Face Landmarker (478 landmarks) on every frame and writes:

- `data/aligned/<label>/<n>.mp4`: a 224×224 face-normalised video with the same frame count and fps (H.264, CRF 17).
- `data/aligned/landmarks/<label>/<n>.npz`: per-frame landmarks (normalised source coordinates, NaN where no face
  was found), a detection mask, and the 2×3 source→aligned transforms. `palsynet.faces.landmarks_to_aligned` maps
  landmarks into aligned-frame pixels.
- `data/aligned/report.csv`: the share of frames with a detected face, per video.

Normalisation is a **similarity transform only** (rotation, uniform scale, translation). It puts the eye centres
(mean of inner and outer canthus) level, at fixed positions (27% of the width apart, 38% down). Faces are *not* warped
to a template, which would partially remove the left/right asymmetry that is the signal. Eye trajectories are
gap-filled and smoothed over 7 frames, so the aligned video doesn't jitter. Frames without a detection reuse the
interpolated transform. `--preview` saves a raw-vs-aligned contact sheet per video for checking.

Because relative paths match `data/raw/data`, `data/splits.json` applies unchanged; a config only needs
`raw_root: data/aligned`. `r3d_18_aligned.yaml` also narrows the random crop to 85–100% of the area so augmentation
doesn't undo the alignment.

## Live inference

`scripts/live.py` runs a trained checkpoint on a camera, stream URL or video file:

```bash
uv run python scripts/live.py runs/<cv>/<run>/best.pt --source 0                      # webcam
uv run python scripts/live.py runs/<cv>/<run>/best.pt --source clip.mp4 --save annotated.mp4 --log preds.csv
```

```
frame ─► MediaPipe landmarks ─► StreamingFaceAligner ─► sample at sample_fps ─► rolling window
         (~5–10 ms, CPU)        (same smoothing as        (by timestamp)          (segment_seconds)
                                 training, 0.1 s delay)                                 │ every --hop s
                          p(affected) ◄── mean of last --avg windows ◄── model ◄── eval transforms
```

- **Same preprocessing as training.** `StreamingFaceAligner` (in `faces.py`) emits each frame once the 3 frames
  after it have arrived, then applies the same centred 7-frame smoothing and similarity transform as
  `preprocess_faces.py`. On a full video its transforms match the offline ones to ~1e-16. Live aligned frames differ
  from the training videos only by H.264 compression (mean ~2/255 per pixel; model output within 0.01).
- **Missing faces:** a gap of up to `--max-gap` (0.5 s) holds the last eye position. A longer gap resets the window,
  and no prediction is made until 5 s of face has been collected again.
- **Raw vs aligned:** the checkpoint decides. Models trained on `data/aligned` get aligned faces; models trained on
  raw frames get raw frames. Override with `--align` / `--no-align`. A raw-frame model only makes sense on footage
  framed like the training videos (300×300 face close-ups); for webcams use an aligned model.
- **Live sources** are read on a separate thread that drops the oldest frames if processing falls behind; sampling
  works from timestamps, so a variable camera frame rate is fine. Files are processed as fast as possible
  (`--realtime` paces them).
- **Several checkpoints = ensemble:** pass several `best.pt` files, or a whole `runs/cv_*` folder, and the
  probabilities of all its fold models are averaged. Inference runs on a worker thread, so the video never waits for
  the model; a window that comes due while the previous one is still running is skipped.
- **Landmark models** (`landmark_tcn`) use the same aligner, but buffer the aligned landmark coordinates instead of
  aligned images. During a short detection gap the last landmarks are repeated (offline training interpolates).
- **Head turns (3D landmark models):** the overlay shows head yaw / pitch, and every prediction line ends with the
  largest yaw in its window, flagged "head turned - less reliable" above `--warn-yaw` (12°). Predictions are not
  paused unless `--max-yaw N` is given (then frames turned more than N° are skipped like a missing face). Training
  data is ~90% within 10° of yaw, and on healthy webcam clips frontalised asymmetry grows from +0.002 (5–10°) to
  +0.04 (20–25°) and +0.07 (30°+) eye units, because MediaPipe has to guess the far side of the face. Pitch (nodding)
  barely matters. Tried and dropped: oversampling turned-head training windows
  (`landmark_tcn_3d_vcd_expr_still_turn.yaml`, `data.turn_prob`) gave no gain on turned held-out windows
  (`scripts/eval_pose.py`) and lowered palsy accuracy 0.90 → 0.88. The data has too few strongly turned heads.
- **Decision:** the averaged p(affected) (last `--avg` windows) at or above `--threshold` (default 0.4) is shown as
  POSITIVE (red banner at the bottom of the video, and on each printed line), otherwise negative (green). The
  end-of-run summary gives the share of POSITIVE predictions.
- **Several models on one feed:** `--compare` with several run folders (e.g. TCN, 3D CNN and graph network) runs
  each as its own 5-fold ensemble on the same camera frames: one score line and one verdict banner per model.
- **Labelling while testing (`--show`):** press **space** to mark "making a face" (again to stop). Each prediction
  records the labelled share of its window; at the end every model is scored on relaxed windows (false alarms)
  and face windows separately. `--dump` saves the labels (`face_frac`) with the windows.
- Cameras open at 640×480 unless `--width/--height` are given.
- `--show` needs a GUI build of OpenCV: `uv remove opencv-python-headless && uv add opencv-python`. `--save` works
  headless.

Research prototype: p(affected) is a model score, not a diagnosis.

## How the data pipeline works

- **Splits are by video (= by subject).** No person appears in more than one of train/val/test.
- **Samples are 5 s segments:** `segment_seconds`=5 sampled at `sample_fps`=10 → a `(T=50, C, H, W)` float tensor,
  batched as `(B, T, C, H, W)`, normalised with ImageNet stats by default. Frames are decoded straight from the MP4
  (one seek per segment); there is no frame extraction step.
- **Short videos:** the shortest video is 5.7 s, so at 5 s every segment is real-time. If you raise `segment_seconds`
  above a video's length, that video is sampled across its full length with the same T instead of being padded, because
  padding would only ever appear in the affected class (all short videos are affected) and leak the label.
- **Training** draws `train_segments_per_video` random segments per video per epoch, so long and short videos count
  equally. Augmentations (random-resized-crop, horizontal flip, colour jitter) apply the same parameters to every frame.
- **Evaluation** takes `eval_segments_per_video` evenly spaced segments per video and averages their probabilities,
  giving **video-level** metrics (accuracy, balanced accuracy, F1, AUC, confusion matrix) alongside segment-level ones.

## Troubleshooting (Apple Silicon)

- `MPS command buffer execution failed: Impacting Interactivity`: macOS killed a GPU job that took too long.
  Lower `data.batch_size` and raise `train.grad_accum` to keep the effective batch size.
- The scripts add `src/` to `sys.path` themselves, because macOS can mark the editable-install `.pth` file as hidden,
  which Python 3.13 then skips. For ad-hoc Python, use `PYTHONPATH=src uv run python ...`.

## Caveat

**Measured:** on 92 held-out webcam clips of ordinary people (VCD2), landmark models trained on palsynet alone called
47–70% of them "affected". Adding other VCD clips as unaffected training data (`landmark_tcn_3d_vcd.yaml`) brought this
to 0% (1.7% of 2 s windows) with unchanged palsy cross-validation. Adding 1440 RAVDESS clips of actors making
emotional expressions (`landmark_tcn_3d_vcd_expr.yaml`, `scripts/fetch_ravdess.py`) cut false alarms on held-out
actors from 3.6% to 0.2% of clips, again with unchanged palsy CV (acc 0.94, AUC 1.00). Remaining risk: neither set
shows healthy people doing the palsy-exam movements (eyes squeezed shut, maximal grin, pucker), which affected videos
often show, so those movements can still act as a cue.


Affected and unaffected videos were collected from different sources, and it shows: affected videos are mostly home
selfie / webcam recordings, unaffected ones mostly studio talking-head videos (`runs/class_contact_sheet.jpg`). A pixel
model can score well by learning recording style (lighting, framing, sharpness, compression, background) instead of
facial asymmetry, and then calls any webcam face "affected". Cross-validation does not catch this because the test
folds share the bias. `landmark_tcn` sees only landmark geometry, which removes most of that shortcut (camera
distance and head pose still differ between sources). Treat high accuracy with suspicion and check per-video
predictions.

## Layout

```
configs/  default.yaml (base)  r3d_18[_aligned].yaml  r2plus1d_18.yaml  vivit_b16[_aligned].yaml  vivit_small.yaml  frame_gru.yaml  landmark_tcn.yaml  landmark_crnn.yaml
scripts/  download.py  make_splits.py  preprocess_faces.py  train.py  evaluate.py  run_cv.sh  summarize_cv.py  live.py
src/palsynet/  config.py  faces.py  landmarks.py  splits.py  data.py  models.py  engine.py  optim.py
data/  (git-ignored)  raw/  splits.json
runs/  (git-ignored)  <model>_fold<k>_<time>/{best.pt,last.pt,history.json,config.yaml,test_*.{json,csv}}
```
