# StrokeWatch (hackathon MVP)

Webcam monitor that alerts on a NEW one-sided facial droop, relative to the patient's own
automatically built baseline. Hackathon demo, not a medical device.

## Setup (macOS, Python 3.12, since MediaPipe has no 3.14 wheels)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p models && curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

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
new left droop at 75 s, stable, old droop worsening at 140 s, talkative, camera
dropouts at 60–90 s, new right droop at 200 s, stable, restless (cycles beyond 8 beds).
Everything is in memory: a restart starts fresh.

Each module runs on its own:

```bash
.venv/bin/python -m strokewatch.vision --source 0                    # overlay + raw features
.venv/bin/python -m strokewatch.vision --source 0 --record clips/talking.mp4
.venv/bin/python -m strokewatch.replay clips/*.mp4                   # → clips/X.png + X.csv
.venv/bin/python -m strokewatch.detector                             # synthetic self-test
python scripts/synth_clip.py face.jpg clips/synth.mp4 [--pre-droop 12] # fake droop from a photo
```

Any config value can be overridden: `--set detector.threshold=3.5 --set vision.lip_gap_max=0.08`.

## Demo script

1. Sit still facing the camera; the grey banner fills over about 20 s of valid frames.
2. Turn your head, talk, blink. Frames get skipped ("head turned", "talking") and there's no alert.
3. Hold a one-sided half-smile, or tape a mouth corner down. After about 6 s above threshold
   you get the red banner, "possible new LEFT/RIGHT droop", with the onset time.
4. Reset while already drooping. The droop is absorbed into the baseline; only making it
   worse alerts.

## How it works

- **vision.py**: MediaPipe Face Landmarker. Rotates frames by the camera mount angle
  (`camera.rotate`), then removes head rotation (`vision.correction: blend` = average of
  levelling the eye line (`2d`) and full rotation removal from MediaPipe's transformation
  matrix (`3d`)), and normalizes by the distance between the outer eye corners.
  Features (+ = patient's LEFT lower): mouth-corner height difference, log eye-opening
  ratio, brow-height difference. Then subtracts the fake asymmetry that turning the head
  causes (`vision.yaw_bias`, a per-feature curve over yaw fitted on recorded head-turn
  clips). Skips frames while the head moves faster than 30°/s (landmarks lag), with the
  head turned > 40°, sideways tilt > 40°, nodded > 20°, talking
  (inner-lip gap at the mouth centre, or jawOpen > 0.35), and no face. Blinks skip only
  the eye feature.
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
