# StrokeWatch: threshold notes for an ICU deployment

Written Nov 13, 2026, when the pitch moved from an ambulance to ICU rooms. Numbers
marked *estimate* still need to be measured on real footage. Section 6 records the
first measurements and the config changes made from them, which are **provisional**.

**Working assumptions for the ICU pitch:** the patient lies on their back (head of bed
flat or raised); intubated or masked patients are out of scope; the camera is framed
so only the patient's face is visible; the baseline is rebuilt every few hours; and
the UI reports coverage (% of time measurable).

## 1. Clinical facts the thresholds rest on

**How fast deficits appear**
- Embolic ischemic strokes reach their worst deficit almost immediately (seconds to
  minutes). Thrombotic and small-vessel strokes can fluctuate or worsen in steps over
  minutes to hours. Brain bleeds can worsen over minutes to hours.
- In the ICU, many events are in-hospital or after a procedure (post-thrombectomy,
  post-thrombolysis, cardiac surgery). Early neurological worsening plays out over
  hours, so slow drift matters as much as a sudden step.
- Deficits last minutes or more (even TIAs usually last minutes). Facial expressions
  last 1–3 s. That gap is what lets the detector require a long persistence window.

**How much the face changes**
- A stroke causes a *central* facial palsy: mostly the lower face (mouth corner,
  nasolabial fold). The forehead is largely spared and the eye is mildly affected. A
  brow-only change suggests Bell's palsy, noise or an artifact, not stroke.
- Resting mouth-corner asymmetry in hemispheric stroke patients: about **1–2 mm**
  (3D-video study), versus about 0 in controls. It's larger on expression (smile).
  The mildest NIHSS facial grade shows mostly on smiling.
- People notice about **3 mm** of mouth-corner asymmetry at rest. Lip tilt of 0–1 mm
  is rated normal.
- MediaPipe landmark jitter: **no published figure found.** About 1 mm per frame on a
  webcam is an *estimate*.

**Unit conversion.** Features are divided by the distance between the outer eye
corners, about 90 mm in adults. So **1 mm ≈ 0.011** feature units. With
`min_spread.mouth_diff = 0.012`, mouth z = 1 ≈ 1.1 mm.

**Detection delay.** A sudden change of size Z reaches threshold T after
`-tau * ln(1 - T/Z)`, then has to stay above it for `alert_seconds`. With tau = 1 s,
T = 4: Z = 8 takes 0.7 s, Z = 4.5 takes 2.2 s. Skipped frames add wall-clock time.

## 2. Current values and what the ICU means for them

The original version of this table (before section 6) proposed a threshold of about
3 with a 1.1 mm floor. Measurement showed the floor, not the threshold, was setting
sensitivity, so the floor was lowered instead (section 6).

| Setting | Current | Real-world meaning | ICU recommendation |
|---|---|---|---|
| `detector.threshold` | 4.0 | With the new floor: ≈ **2.2 mm** lasting mouth change (was ≈ 4.3 mm) | Keep at 4 for now; sensitivity is set through `min_spread`. Revisit after more clips. |
| `detector.min_spread.mouth_diff` | **0.006** (was 0.012) | ≈ 0.5 mm, the leftover error after pose matching on one clip | Provisional. Re-measure with `scripts/mapping_report.py` on more people and the real ICU camera. |
| `detector.pose_match_deg` / `min_pose_matches` | **3° / 30** (new) | Compare each frame only with baseline frames at a similar head angle | New, see section 6. 0 = off (original behaviour). |
| `vision.correction` | **2d** (new; was effectively 3d) | Head-tilt correction method | New, see section 6. `3d` restores the original method. |
| `camera.rotate` | 0 (new) | Camera mount angle, 0/90/180/270 clockwise | Set per installation. Phone test clips needed 90 and 180. |
| `detector.alert_seconds` | 6 s | Time above threshold before alerting | **60–300 s.** No race against a ride; long windows cut false alarms over many hours. |
| `detector.pre_threshold` | 2.5 | ≈ 2.7 mm, marks onset | Keep it at roughly 60% of the threshold. For slow drift, onset = when the change became measurable, not when it truly began. |
| `detector.ewma_tau_s` | 1 s | Smoothing | Could be 5–30 s for slow-drift detection. A second, slower score could run alongside. |
| `detector.weights.brow_diff` | 0.7 | — | **≤ 0.2 or off**: the forehead is spared in central palsy. |
| `detector.weights.eye_ratio` | 0.5 | — | Keep it low. Eyes are often closed (sleep, sedation), so expect the feature to drop out for long periods. |
| `detector.baseline_seconds` | 20 s | One frozen baseline | **Not enough for days-long stays.** See section 3. |
| `vision.max_yaw/pitch_deg` | 20° from straight on | Pose gate | Measure it relative to the baseline pose. Camera angle and head-of-bed elevation give a steady pitch offset. |
| `vision.lip_gap_max` / `jaw_open_max` | 0.06 / 0.35 | Talking gate | Will skip most frames in intubated or open-mouthed sleeping patients, which are common in the ICU. See section 4. |
| `detector.min_baseline_frames` | 60 | — | Has no effect (`max_dt` 0.25 s means 20 s already needs ≥ 80 frames). |

## 3. Design changes the ICU needs (not built yet)

- **Baseline that updates slowly.** Over hours or days the face changes for non-stroke
  reasons: sleep, swelling, repositioning, day/night lighting. Use a slow rolling
  baseline (tens of minutes) that **freezes as soon as a change episode starts**, so a
  slow stroke isn't absorbed into it. Alternatively, re-baseline after nurse-confirmed
  neuro checks.
- **Lock onto the patient.** `num_faces=1` can latch onto staff leaning over the bed.
  Track the face nearest the bed/pillow region or match its identity to the baseline
  face.
- **Measure false alarms per patient-day.** ICU staff already suffer alarm fatigue.
  Tune for something like < 1 false alarm per patient-day, not for speed.
- **Coverage reporting.** Show the % of time the face was measurable and the time of
  the last measurable symmetric frame ("last verified symmetric at 03:12"). This
  matters because of section 4.
- **Gravity artifact.** With the head rolled to one side, the cheek sags on the lower
  side. Removing head rotation fixes the geometry, not the soft tissue. Either
  re-baseline per head position or skip frames with large roll relative to the
  baseline.

## 4. Can we assume the face is visible the whole stay? No.

A head-on camera **will not** see usable features the whole time. Lines and monitors
are routed around turns; they don't prevent them. ICU practice actively moves patients:

- **Repositioning about every 2 hours** to prevent pressure injuries, often to a
  30° side tilt. The head usually follows.
- **Head of bed raised 30–45°** for ventilated patients (pneumonia prevention). That's
  a steady pitch offset, fine if the pose gate is relative to the baseline.
- **Prone positioning** (face down, 16+ hours a day) for severe respiratory failure.
  The face is unmeasurable for those hours.
- **Things covering the face:** endotracheal tube and holder, NG tube, BiPAP full-face
  mask, eye pads or lubricant, blankets, staff during care.
- **Sedated patients' heads roll** to the side even when the body is on its back.

**Where it can work:** awake or lightly sedated, non-intubated patients who mostly
lie on their back with the head of the bed raised. A strong fit is **neuro-ICU
patients after thrombolysis or thrombectomy**. Nurses already do neuro checks every
15 min for 2 h, then every 30 min for 6 h, then hourly. StrokeWatch could watch
continuously *between* those checks.

**Design consequence:** treat visibility as intermittent. Score only measurable
frames, report coverage, and expect gaps of minutes to hours. A long gap should show
as "not monitored since HH:MM", not count as "no droop".

## 5. Next steps

1. Record more footage, ideally in the target setup (head of bed raised, camera at the
   foot of the bed): more people, still, sleeping, talking, head-of-bed changes.
   Run `scripts/mapping_report.py` on each. **If the 2d / pose-matching gains don't hold
   up, revert them (section 6).**
2. Sticker test (gain): coloured dots on the mouth and eye corners, compared with
   MediaPipe's values. Not done yet. If gain < 1, the mm figures here are optimistic.
3. Record a talking clip to see whether `lip_gap_max` (0.06) can be tightened.
4. Rebuild the baseline every few hours, only when no change is building, and flag
   (don't silently accept) a new baseline that differs from the old by more than the
   threshold. Add coverage reporting: % measurable, "not monitored since HH:MM",
   including "pose not in baseline" skips.
5. Lower the brow weight. Lengthen `alert_seconds`.
6. Measure false alarms per patient-day on long recordings.

## 6. First measurements and provisional changes (Nov 13, 2026)

> **Caveat:** everything in this section comes from **one person, two 2-minute clips**:
> a phone at 1080p held over the subject lying flat, not an ICU camera at the foot of
> a raised bed. The changes below were made for time's sake and **may be reverted or
> altered** once more clips exist. All of them are config switches.

**Clips:** `clips/still.mov` (lying still, 2 min) and `clips/tilt.mov` (slow head
turns and nods within about ±20°, neutral face, 2 min). The phone recorded them
sideways and upside down, which led to the `camera.rotate` setting. Analysed with
`scripts/mapping_report.py`; mm assume an outer-eye-corner distance of 90 mm.

**Before the changes** (3D correction, single baseline):

| | Still | Head sweep (yaw ±17°, pitch −8..+18°) |
|---|---|---|
| mouth_diff spread | 0.09 mm (0.11 mm smoothed) | 1.43 mm |
| Bias from turning | — | −1.4 to −1.8 mm per 10° of yaw, R² 0.92 |
| Talking-gate inputs at rest | lip_gap p95 0.004, jaw_open p95 0.002 | lip_gap p95 0.012 |

Key findings:
- Measurement noise with a still head is tiny (~0.1 mm). Sensitivity was set by the
  1.1 mm floor in the config, not by noise.
- Head turns cause a systematic fake asymmetry of about ±3 mm at ±17°, larger than the
  1–2 mm stroke signal. The old detector avoided false alarms only because a baseline
  recorded during head movement had a wide spread, which also made it blind below about
  3.7 mm.
- Correction methods compared on the same landmarks (head-sweep clip):

  | Method | Spread | Bias per 10° yaw |
  |---|---|---|
  | 3D, MediaPipe matrix (original) | 1.43 mm | −1.8 mm |
  | 3D, opposite direction (convention check) | 3.61 mm | +3.6 mm |
  | **2D, level the eye line** (chosen) | **0.76 mm** | −0.8 mm |
  | 3D fit to a frontal reference (Procrustes) | 0.95–1.13 mm | −1.1 to −1.4 mm |

  The 3D version was the right way round, but MediaPipe's depth estimates leak into
  the vertical measurement. Some turn-dependent bias remains with every method.

> **Status (Nov 13, 2026, later): reverted for the MVP.** Live phone tests gave too many
> "pose not in baseline" skips (a still 20 s baseline covers < 1° of head angle, and at
> 10 fps the head-sweep clip skipped 74% of frames). `config.yaml` is back to
> `pose_match_deg: 0`, `mouth_diff: 0.012`. `correction` was briefly back to `3d`, then
> set to `2d` again to allow `max_yaw_deg: 35` (was 20). Fake mouth z from turning, at
> floor 0.012: 3d 1.73 mm/10° → z 4.0 at 25°; 2d 0.76 mm/10° → z 2.5 at 35°. Detector
> peak on the head-sweep clip (±24°, no yaw gate): 3d 2.61, 2d 1.06. The detector now also has a nearest-neighbour
> fallback (`pose_max_deg`, default 10°) for when pose matching is turned back on.

**Changes made** (and how to revert):

| Change | Setting | Revert with |
|---|---|---|
| 2D eye-line correction instead of 3D | `vision.correction: 2d` | `3d` |
| Pose-matched baseline: compare with baseline frames within ±3° yaw and pitch; skip unseen poses ("pose not in baseline") | `detector.pose_match_deg: 3`, `min_pose_matches: 30` | `pose_match_deg: 0` |
| Mouth floor lowered: alert level ≈ 2.2 mm instead of ≈ 4.3 mm | `detector.min_spread.mouth_diff: 0.006` | `0.012` |
| Camera mount rotation | `camera.rotate: 0/90/180/270` | `0` (no-op) |

**After the changes** (same clips, baseline = first 20 s of each clip):

| | Still | Head sweep |
|---|---|---|
| mouth_diff spread | 0.12 mm | 0.76 mm (0.44 mm smoothed) |
| Bias from turning | — | −0.4 mm per 10° of yaw (pose matching removes most of the rest) |
| Detector peak score (threshold 4, pre-threshold 2.5) | 0.59 | 1.24 (was 2.04 with the old pipeline, at half the sensitivity) |
| Frames skipped as "pose not in baseline" | 0 | 660 / 3608 (18%) |
| Alerts | none | none |

Other checks after the change: tilting the test photo ±20° still moves mouth_diff by
under 0.5 mm. The synthetic droop clips still alert (about 8 s after onset) and the
normal and stable pre-existing-droop clips still don't (peak 0.5). The detector
self-test now includes turn-dependent bias during the baseline and a 2.7 mm
worsening, and it passes.

**Known weaknesses of these changes:**
- **2D correction depends on the eye corners.** It levels the line between the outer
  eye corners, so anything that moves one corner (swelling, a dressing) tilts every
  feature. The outer eye corners aren't much affected in central facial palsy.
- **Head angles outside the baseline aren't measured.** A 20 s baseline from a still
  patient covers few angles, so movement later means skipped frames. Longer baselines
  (minutes) or rebuilding the baseline every few hours improve coverage.
- **Roll isn't matched.** The head-sweep clip still shows a correlation with roll
  (+1.5 mm per 10°, partly confounded with yaw in that clip). Add roll to pose
  matching if more clips confirm it.
- **One person, a phone camera, lying flat.** Repeat with more people, the real
  camera and the real geometry before trusting the 2.2 mm figure.

## 7. Head-turn correction (Nov 14, 2026)

**Clips:** neutral face, about 45 s each, one person: `macsidetoside.mov` and
`phonesidetoside.MOV` (left-right turns to about ±60°), `mactilt.mov` and `phonetilt.MOV`
(ear-to-shoulder tilt to about ±43°), Mac webcam and iPhone, plus the earlier `tilt.mov`
(iPhone overhead, lying flat).

**Findings:**
- **Sideways tilt (roll) is not the problem.** In the tilt clips every feature stays
  within about 1 mm up to ±43°, with both 2d and 3d.
- **Turning (yaw) is.** With 2d, mouth asymmetry drifts about 1.7 mm per 10° of turn
  (about 6 mm at 35°, a fake score of 6). Eye and brow drift too. The Mac and iPhone
  curves nearly overlap. The overhead clip is flatter.
- Other geometry (facial-midline axis, 3D alignment to the person's own straight-on face,
  2D affine) didn't do clearly better: the bias seems to come from the landmark model.
- In the tilt clips, tilting 40° also reads as about 20° of turn (the person turns slightly
  while tilting), without the matching drift. The correction slightly over-corrects there.

**Changes:**

| Change | Setting | Revert with |
|---|---|---|
| Subtract a fitted per-feature bias curve over yaw (equal weight per clip) | `vision.yaw_bias` | `yaw_bias: null` |
| Widen spread with turn away from the baseline angle: 0.15 x `min_spread` per 10° | `detector.yaw_noise` | 0 |
| Turn gate 35° → 40°; sideways-tilt gate 25° → 40° | `vision.max_yaw_deg`, `max_roll_deg` | 20 / 25 |
| 2d correction (needed for the fitted table) | `vision.correction: 2d` | refit for `3d` |

**Validation:** full pipeline (features + detector at 10 fps); baseline facing the camera,
then the head held at each angle for 10 s; the bias curve refit **without the tested
device's clips**:
- Neutral face, held within the gates: peak score at most 2.8 (Mac, 40° tilt), at most 2.1
  for turns; no alerts. Without the correction: about 6–7 at 35–40°.
- Droop added while the head is held (either side; the ±40° bands are half outside the
  gate, so they're left out): **7 mm** alerts at every angle within ±35°; **6 mm** at all
  but 3 of about 70 angle/clip combinations; **5 mm** (just above the 4.3 mm level
  for a face looking at the camera) mainly within about ±10°, since leftover bias works
  against it on one side. `yaw_noise` 0.25 lost more at large turns, so 0.15 was chosen.
  In short: looking at the camera ≈ 4.3 mm; turned up to 35° ≈ 6–7 mm.
- Leave-one-setup-out, 2d vs 3d with correction (worst held mouth score within ±40°):
  2d 2.0 / 2.3 / 1.8 (overhead / Mac / iPhone), 3d 2.6 / 0.7 / 2.5.

**Follow-up, same day: droops were missed with the head turned.** A heavy droop scored
about 8 facing the camera but stayed under 4 when turned. Measured by warping a 6 mm droop
into every 3rd frame of the turn clips, on each corner: the landmarks report 0.65–0.85 of
it facing the camera, but less when turned, mostly on the far side (relative to facing:
about 0.86 at 30°, 0.73 at 40°, 0.3–0.5 at 45–50°). Together with `yaw_noise` (−10%)
and leftover bias working against the droop on one side, that cost about 60% at 30°.

| Change | Setting | Revert with |
|---|---|---|
| Divide the mouth deviation by the measured signal loss for the side it points to | `detector.yaw_gain` | `yaw_gain: null` |
| Safety margin off | `detector.yaw_noise` 0 | 0.15 x `min_spread` |

Same held-angle test (device left out of the yaw_bias fit, droop scaled by that device's
measured loss for the drooping side), ±35°, about 130 angle/clip combinations per row:

| | Neutral, worst score | 6 mm droop not caught |
|---|---|---|
| Before (`yaw_noise` 0.15, no `yaw_gain`) | 2.4 | 13 |
| `yaw_noise` 0.15 + `yaw_gain` | 2.4 | 7 |
| **`yaw_noise` 0 + `yaw_gain` (chosen)** | **2.6** | **2** |

The 2 misses (score 4.1 and 4.4) are where leftover bias works against the droop.
Note also that facing the camera, only 0.65–0.85 of a droop is reported, so the 4.3 mm
alert level corresponds to roughly 5–6.5 mm of real corner movement (synthetic warp, not
yet checked with stickers).

**Weaknesses:** one person. The curves may differ for other faces and camera heights.
Next step: learn the leftover bias per patient from the baseline when it contains head
turns. Record more people and setups and refit with
`scripts/fit_yaw_bias.py`.

## Sources

From web searches, Nov 2026:
- [Facial Asymmetry in Hemispheric Stroke Patients Assessed by 3D-Video Analysis](https://www.researchgate.net/publication/266772082_Facial_Asymmetry_in_Hemispheric_Stroke_Patients_Assessed_by_3D-Video_Analysis)
- [Quantitative assessment of facial muscle impairment in patients with hemispheric stroke](https://www.researchgate.net/publication/50830412_Quantitative_assessment_of_facial_muscle_impairment_in_patients_with_hemispheric_stroke)
- [Frontiers in Surgery: facial palsy outcome measurement (3 mm detection)](https://www.frontiersin.org/journals/surgery/articles/10.3389/fsurg.2022.735231/full)
- [Acute Ischemic Stroke (IEM, 2024)](https://iem-student.org/2024/12/02/acute-ischemic-stroke-2024/)
- [WikEM: Ischemic stroke](https://wikem.org/wiki/Ischemic_Stroke)
- [MedLink: Ischemic stroke syndromes](https://www.medlink.com/articles/ischemic-stroke-syndromes-and-their-anatomic-localization)
- [Normal facial asymmetry, Korean J Orthod (PMC4807152)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4807152)
- [Thai orthodontic study on perceived lip canting](https://he01.tci-thaijo.org/index.php/THAIORTHO/article/download/276937/189289/1342934)

Section 4 is standard ICU practice from general clinical knowledge, not from those
searches. Confirm the details (turning schedule, post-thrombolysis neuro-check
intervals) with a clinician or the relevant guidelines before relying on them.
