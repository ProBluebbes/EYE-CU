# StrokeWatch (hackathon MVP)

Webcam monitor that alerts on a NEW one-sided facial droop, relative to the patient's own
automatically built baseline, plus more of BE-FAST: it watches for **eyes held to one side**
and, on a fixed gaze, runs a **voice-guided check** (look left/right, eyes closed
with both arms out) that measures **gaze palsy and arm drift**. A droop alerts on its own;
no check follows it automatically.
Hackathon demo, not a medical device.

## Setup (macOS, Python 3.12, since MediaPipe has no 3.14 wheels)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p models && curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -L -o models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

Without the pose model everything still runs; the BE-FAST check then does only the eye steps.

Grant your terminal camera access (System Settings → Privacy & Security → Camera).
For phone cameras: `brew install cloudflared`.

## Run

```bash
.venv/bin/python -m strokewatch.server --camera 0 --tunnel        # webcam + phones (QR code) → http://127.0.0.1:8000
.venv/bin/python -m strokewatch.server --camera 0 --mock-beds 7   # webcam + 7 simulated patients
.venv/bin/python -m strokewatch.server --camera "ICU 4=0"         # name a bed (NAME=SRC)
.venv/bin/python -m strokewatch.server --camera clip.mp4          # loop a recorded clip in real time
.venv/bin/python -m strokewatch.server --mock                     # one simulated patient (alert at ~55 s)
.venv/bin/python -m strokewatch.mock --scenario predroop
.venv/bin/python -m strokewatch.server                            # webcam 0 only
```

`--camera` is repeatable and takes a webcam index, video file or RTSP URL. Stop the server with
**Ctrl+C** (not Ctrl+Z, which only pauses it and keeps the camera on).

### Combined droop detection (default)

Droop alerts combine StrokeWatch's baseline-change score with a learned facial-palsy model (palsynet 3D-CNN,
see `README_palsynet.md`) measured against the patient's own first 20 s:
`combined = StrokeWatch score / threshold (0 while building its baseline) + 0.25 × (model logit − its first-20-s median)`,
ALERT at ≥ 1.19 for 6 s. The model counts for much less than StrokeWatch's own score (`palsy.model_weight`;
the line scales with it as 1 + 0.74 × weight, from StrokeWatch alone at weight 0 / line 1.0 to the
benchmark-tuned weight 1 / line 1.74). StrokeWatch skips frames while the head moves or tilts and may take long to build its
baseline; the model keeps scoring then, and the personal baseline stops it reacting to someone's normal face.
At weight 1, on six clips with a synthetic one-sided mouth droop (`scripts/droop_benchmark.py`,
`scripts/droop_eval.py`), it caught 18/36 droops with 0/6 false alerts; StrokeWatch alone caught 6/36 (only on
the still clip). The current weight 0.25 has not been benchmarked. The dashboard
and bed page show the combined score against its line; StrokeWatch's own score is kept in the status as
`sw_score`. Settings: `palsy:` in `config.yaml`. `--no-palsy` (or a missing model folder) = StrokeWatch alone;
`--palsy-model RUN_DIR` picks another model. Needs `torch` (this repo's `uv` environment).

### Ward dashboard

`/` shows every bed: live thumbnail, status (building baseline / monitoring / ALERT / camera
offline), change score against threshold, and why frames are being skipped. Click a tile for
the bed page (full video, score chart, per-feature bars, reset).

The alert center lists alerts across all beds. **Acknowledge** silences the alarm (the bed stays
in ALERT). **False alarm** / **Confirmed** close the alert and rebuild that bed's baseline so
monitoring continues. Click **Enable alarm sound** once (browsers need a click before playing
sound); the alarm repeats every 4 s while any alert is unacknowledged.

### Phone cameras (any number of iPhones)

1. Start the server with `--tunnel`. It opens a Cloudflare quick tunnel (public HTTPS link,
   ready in about 5 s; no account needed). This works on networks like eduroam that block
   phone-to-laptop traffic, and gives Safari the HTTPS it requires for the camera.
2. On the dashboard click **Add phone** (or open `http://127.0.0.1:8000/#add-phone`) and scan the
   left QR code with the iPhone camera. The link is also printed, with a QR code, in the terminal.
3. In Safari: optionally type a bed name, pick front/back camera, tap **Start camera**, allow
   camera access. The phone becomes a new bed (PHONE badge). Tap **Rotate 90°** until the
   preview shows the face upright. The preview is exactly what gets analyzed.
4. Keep Safari open on that page with the phone plugged in. A reload or network blip reconnects
   to the same bed and keeps its baseline. The bed page has **Test phone speaker** and **Remove
   this phone**.

The links carry a random access key that changes on every restart (the tunnel URL changes
too), so phones need to rescan after a restart. Requests from this Mac need no key; anything
through the tunnel without the key gets "Access key needed". The right-hand QR code opens the
dashboard on another device. Anyone with these links can see the cameras.

Simulated patients (`--mock-beds N`, labelled SIM) follow scripted timelines from server start:
new left droop + weak left arm at 75 s, stable, old droop worsening at 140 s, talkative, camera
dropouts at 60–90 s, new right droop + eyes to the left + weak right arm at 200 s, eyes held
right + weak left arm at 110 s with a normal face (only the gaze watch catches it), restless
and won't follow the eye steps (keeps staring at one spot, so the eyes part comes out
INCONCLUSIVE) (cycles beyond 8 beds). They follow the BE-FAST check's instructions (except
that last one), so you can watch a full check without a camera: press **Run BE-FAST
check** on any SIM bed page (a droop bed's weak arm only shows up that way, since a droop
alert doesn't start a check).
Everything is in memory: a restart starts fresh.

**Add demo** on the dashboard picks a video from `clips/demo/` (dropdown) and adds it as a new
bed (DEMO badge) that plays in a loop like a camera. The bundled demos (made with
`scripts/demo_droop.py`) show a normal face for 30 s, then a slow left or right droop that
alerts at about 65 s. Drop any other `.mp4`/`.mov` in that folder to offer it too. Remove a
demo bed from its bed page.

Each module runs on its own:

```bash
.venv/bin/python -m strokewatch.vision --source 0                    # overlay + raw features
.venv/bin/python -m strokewatch.vision --source 0 --record clips/talking.mp4
.venv/bin/python -m strokewatch.replay clips/*.mp4                   # → clips/X.png + X.csv
.venv/bin/python -m strokewatch.detector                             # synthetic self-test
python scripts/synth_clip.py face.jpg clips/synth.mp4 [--pre-droop 12] # fake droop from a photo
python scripts/demo_droop.py clips/still.mov clips/demo.mp4 [--side right] [--droop-mm 5] [--label]
                                       # demo video: real clip, normal 30 s, then a slow droop over 60 s
                                       # (clips shorter than ~110 s play forward/backward to fill it)
```

Any config value can be overridden: `--set detector.threshold=3.5 --set vision.lip_gap_max=0.08`.

### BE-FAST check (eyes, arms)

- **Passive gaze watch.** Every frame, the horizontal eye position is measured from the iris
  landmarks the face model already returns (iris offset from the middle of the eye opening,
  per eye width, both eyes averaged), relative to the patient's own centre learned at start.
  Eyes held to one side for 60 s without glancing back past the centre ("gaze held to the
  right") start a check; any glance back starts the count over. In a hemispheric
  stroke the eyes deviate toward the lesion, i.e. away from the weak side.
- **Voice-guided check (~40 s)** starts automatically on a held gaze (max one per 5 min,
  per 30 min after a check found normal eye movement; never on a bed already in facial-droop
  ALERT, since the droop is the stroke sign and its alert stands alone), or with **Run
  BE-FAST check** on the bed page:
  1. "Keep your head still, and look straight at the camera." / "look to your left" /
     "Now look to your right." Can't look fully to one side (gaze palsy), or eyes stay turned
     when asked to look ahead → abnormal. **Only if the patient followed the steps:** eyes
     closed, head turned away or moved (> 8°), eyes roaming instead of holding a side, looking
     the wrong way, eyes staying in one place for every instruction, or barely moving →
     **INCONCLUSIVE (instructions not followed)**, with the reasons. If the first try isn't
     clean (not followed, or a possible sign), the eye steps are asked once more (~+20 s); a
     sign counts only if both tries show it, and a good second try overrides a bad first one.
     (On simulated patients who ignore the prompts, one try read "abnormal" 17% of the time;
     with these checks: 98% inconclusive, ~2% normal, 0.3% abnormal.)
  2. "Close your eyes. Hold both arms straight out in front of you, palms up…" (10 s, MediaPipe
     Pose). One arm not raised, or sinking clearly more than the other (pronator-drift style)
     → abnormal. Shown in shoulder widths; the "eyes stayed open" note checks the instruction.
     Arms already up while the instruction plays count as raised. Neither arm raised, or
     both lowered before the 10 s are up → inconclusive (instructions not followed).
  3. No response to any instruction (eyes and arms both not followed) → "Did not follow the
     check's instructions" (possible confusion/aphasia), which also alerts. One part alone
     not followed (e.g. a face-only camera and an inconclusive eyes part) doesn't alert.
- **Hand test** on the bed page runs only the arm step (~25 s: intro, arms out 10 s, lower).
  It needs the pose model (`models/pose_landmarker_lite.task`, see setup); without it the
  bed page says so instead of running.
- **Where the voice comes out:** phone beds speak through the phone and show the instruction
  in large text (with ←/→ arrows for the patient's own left/right). Webcam beds speak through
  this computer (macOS `say`; set `exam.server_voice: false` if the camera isn't next to the
  patient). SIM beds are silent; the prompt shows on the dashboard and bed page.
- **Results:** findings are added to the bed's open alert ("possible new LEFT facial droop" +
  "Arm drift: left arm drifted down…"), or raise a new alert if there was none. A normal check raises nothing. An inconclusive part shows on the dashboard
  tile in amber ("check 14:02: eyes inconclusive (instructions not followed)", reasons on
  hover) and as a note on the bed's open alert. The bed page shows each part (normal /
  ABNORMAL / INCONCLUSIVE / no response / not assessed) and a live gaze bar. The face detector is paused
  while a check runs, so deliberate eye and arm movements can't cause a droop alert.
- **Where the signs may point** (`strokewatch/localize.py`). Signs are recorded by the
  patient's *weak* side; the alert adds hedged wording about the brain, never a diagnosis.
  Example: eyes can't look left, or left arm drift, gives "signs on the patient's left side:
  this may be a right hemispheric stroke". The rules come from the literature:
  - Hemispheric stroke: eyes turn toward the lesion, and the weakness is on the opposite side.
  - Eye signs plus one-sided weakness: a warning sign of a large-vessel occlusion.
  - Eyes turned toward the weak side ("wrong-way" eyes): may be a brainstem (pontine) stroke,
    a thalamic bleed or a seizure.
  - Face weak on one side and limbs on the other (crossed signs): may be a brainstem stroke.
  - Weakness without eye signs can also come from a small deep (lacunar) stroke.

  Every message ends with "Screening signs only, not a diagnosis".

Thresholds are in `config.yaml` under `gaze:` and `exam:`. They are **provisional** (tuned on
simulated patients and photos only), so see the tuning notes below.

## Demo script

1. Sit still facing the camera; the grey banner fills over about 20 s of valid frames.
2. Turn your head, talk, blink. Frames get skipped ("head turned", "talking") and there's no alert.
3. Hold a one-sided half-smile, or tape a mouth corner down. After about 6 s above threshold
   you get the red banner, "possible new LEFT/RIGHT droop", with the onset time.
4. Reset while already drooping. The droop is absorbed into the baseline; only making it
   worse alerts.
5. The droop alert stands on its own (no check follows). To show arm drift, press **Run
   BE-FAST check** on the bed page, follow it, but let one arm sink during the 10 s hold:
   that arm's drift is added to the alert.
6. Or, after the baseline, keep your eyes on something far to one side (head still) for
   about 60 s without glancing back: the check starts from the gaze watch alone.

## How it works

- **vision.py**: MediaPipe Face Landmarker. Rotates frames by the camera mount angle
  (`camera.rotate`), then removes head rotation (`vision.correction: blend` = average of
  levelling the eye line (`2d`) and full rotation removal from MediaPipe's transformation
  matrix (`3d`)), and normalizes by the distance between the outer eye corners.
  Features (+ = patient's LEFT lower): mouth-corner height difference, log eye-opening
  ratio, brow-height difference; plus horizontal gaze from the iris landmarks (for befast.py,
  measured up to a 30° head turn and also while talking). Also `PoseAnalyzer` (MediaPipe
  Pose, arm heights; only run during a check). Then subtracts the fake asymmetry that
  turning the head causes (`vision.yaw_bias`, a per-feature curve over yaw fitted on
  recorded head-turn clips). Skips frames while the head moves faster than 30°/s (landmarks lag), with the
  head turned > 40°, sideways tilt > 40°, nodded > 20°, talking
  (inner-lip gap at the mouth centre, or jawOpen > 0.35), and no face. Blinks skip only
  the eye feature.
- **befast.py**: `GazeMonitor` (passive held-gaze detection), `ExamRunner` (the voice-guided
  check: steps, timing on frame time, scoring), `ServerVoice` (local speaker). Self-test with
  simulated patients: `python -m strokewatch.befast`.
- **detector.py**: baseline = median and robust spread of `baseline_seconds` of valid
  frames (spread floored by `min_spread`). If the baseline contains head turns (a guided
  calibration, or a patient who looked around), it also learns this patient's and camera's
  leftover turn bias. With the head turned, the landmarks follow a droop less (mostly on
  the far side), so the mouth deviation is scaled back up by the measured loss
  (`yaw_gain`), and the spread widens a little with the turn (`yaw_noise`, halved for
  turns the calibration covered).
- **Calibration (optional)**: "Calibrate head turns" on the phone camera page or the bed
  page rebuilds the baseline with spoken/shown prompts: look straight, turn to one side
  and hold, the other side and hold, straight again (about 15-30 s).
  Optional pose-matched baseline (`pose_match_deg`, off by default): compare only with
  baseline frames at a similar angle.
  Then weighted robust z per feature, EWMA-smoothed, score = max |z|. ALERT when
  score ≥ `threshold` for `alert_seconds` of valid time. Onset = first crossing of
  `pre_threshold`. Held until `/reset`.
- **beds.py**: one worker thread per bed (camera or simulated), analyzing at up to
  `beds.analysis_fps`, plus `BedManager`, which raises/acknowledges/resolves alerts.
- **web/phone.html + phone.js**: phone camera page. It rotates/downscales frames (longest side 640)
  and sends JPEGs at 10 fps over a WebSocket. The server sends back status and text to speak.
- **tunnel.py**: runs `cloudflared` and reads its public URL. **access.py**: access-key check for
  tunnel traffic (cookie after the first `?k=` visit).
- **server.py**: dashboard `/` (one WebSocket carries all bed states plus thumbnails, so any
  number of beds works within the browser's connection limit), bed pages `/bed/<id>`, and the
  JSON API listed in its docstring.

## Tuning notes

See `docs/icu-thresholds.md` for the clinical reasoning, the first measurements and the
provisional changes. Measure a clip with
`.venv/bin/python scripts/mapping_report.py clips/X.mov` (finds the rotation itself, and
reports noise, angle bias and what the detector does).

- Score scale: with `min_spread.mouth_diff: 0.004`, threshold 4 ≈ a 0.016 change in
  mouth_diff ≈ 1.4 mm as tracked. The tracker reports only ~65-75% of a real droop, so
  that's ≈ 2.5 mm of real mouth-corner droop facing the camera; ≈ 5-6 mm with the head
  turned up to 35° without calibration (one person's clips).
- Head-turn bias: record neutral-face clips turning slowly left-right past 40° on each
  device/camera position, then refit and paste the printed block into `config.yaml`:
  `.venv/bin/python scripts/fit_yaw_bias.py clips/*.mov clips/*.MOV`. It also prints how
  well the curve carries over to a clip left out of the fit.
- `vision.lip_gap_max` (0.06) is the talking gate. It was checked only on synthetic clips.
  Watch the `lip_gap` column in replay CSVs on real talking/resting clips and adjust.
  Don't gate on jawOpen alone: a one-sided droop raises it too.
- Record clips of normal / head turns / talking / droop / droop-then-worse with
  `vision --record`, then iterate with `replay`.
- Gaze: calibrate with `.venv/bin/python scripts/calibrate_gaze.py` (spoken prompts, ~1 min,
  head still: straight / full left / full right). It prints the steady readings and suggests
  `gaze.deviation` (half the smaller full-sideways reading) and `exam.gaze_min_excursion`
  (a quarter). First calibration (one person, laptop webcam): straight +0.03, full left
  +0.19, full right -0.20, so deviation 0.08. People's eyes moved ~2 s after a prompt
  finished, so the check waits speech time + 2.5 s before measuring.
- Live end-to-end test: `.venv/bin/python scripts/test_befast_live.py --trials
  normal,weak_left,hold_right` (spoken; normal check, acted-out left weakness, held gaze).
  First run (one person) found: looking by turning the head read as a gaze palsy (now
  "not assessed: turned the head"), excursions were measured from the screen rather than from
  "look straight" (fixed), and a held gaze with the head turned past 20 deg lost its gaze
  samples (gaze is now measured up to 30 deg). The held-gaze trial wasn't flagged because the
  eyes went back to the screen twice; still to be re-run.
- Arms: calibrate with `.venv/bin/python scripts/calibrate_arms.py --trials steady,sink_left`
  (runs the real arm step with spoken trials, records clips/arm_calibration.csv). The check
  compares each wrist with its own resting height, takes the highest steady point early in
  the hold as the start, and tracks the drop over 10 s. First live run (one person, laptop
  webcam): arms raised 5–7 s after the prompt began; arms pointing at the camera read about
  +1.0 (perspective, harmless); eye closure read clearly (eyeBlink ~0.8 closed vs ~0.15
  open); pose tracking sometimes flipped between "up" and "down" within a second. So:
  flipping tracking -> "not assessed", both arms lowered early -> "not assessed", and a
  one-sided drift needs the other arm to stay up. A deliberate slow left-arm sink was flagged
  ("left arm drifted down 0.31 vs -0.02"); the steady arm from the same recording scored
  normal. Raised arms read -0.4..+1.0 and lowered ones -1.3..-1.6 depending on camera angle,
  hence `exam.arm_up_min: -0.9` when the resting height isn't seen. Not yet seen live: a
  steady hold with eyes closed on both arms, or anyone but one person.
  Shoulders and hands in view, camera at chest-to-face height works best.
