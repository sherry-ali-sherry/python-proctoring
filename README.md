# ExamVision — Phase 1: Webcam-Based Eye-Gaze Proctoring Engine

> **Developers:** start with the [Developer Handbook](docs/DEVELOPER_HANDBOOK.md). It explains
> every module, the core algorithms, and how the pieces connect.

Phase 1 delivers a **local, offline, geometry-based gaze estimation
pipeline** that classifies a webcam subject as `LOOKING_AT_SCREEN`,
`LOOKING_AWAY`, or `UNCERTAIN`, using explicit 3D eye geometry and
gaze rays rather than a 2D iris-position threshold. It is designed as
the reusable computer-vision engine for a future online-exam
platform (object detection, browser/tab monitoring, identity, and the
exam UI itself are explicitly out of scope for this phase).

## Web application (phase 2): upload, review, decide

```bash
pip install -r requirements.txt
python serve.py
```

This opens http://127.0.0.1:8000 in your browser. The app is built for
**reviewers**:

| Screen | What it does |
|---|---|
| **Analyses** | Every uploaded video with its status, automatic verdict and review state. Filters: needs review · reviewed · in progress · failed. Queued and running analyses show live progress and can be cancelled. |
| **New analysis** | Enter the candidate's name, then drag and drop the video. You can upload several; they are analyzed one at a time in the background. |
| **Review** | A video player with a **colored gaze timeline** under it (green: on screen, amber: brief glance, red: incident, grey: no face / not measurable). Click the timeline to seek, or click an incident to play it with 1 s of context. Mark each incident **Confirm** or **False alarm** and add notes. Then record a **final decision**: accept the automatic verdict, or override it with CHEATING / NO CHEATING / INCONCLUSIVE and a reason. |
| **Settings** | Change every detection limit (verdict rules, look-away time, head-turn limits, screen size…). Settings apply to new analyses. The CLI and desktop app read the same `settings.json`. |

Reviewer decisions are saved in `examvision.db`. They are also written
to `results/<Name>/report/review.json`, and `report.html` / `report.txt` /
`incidents.csv` are re-rendered, so each report shows the reviewer's
decision next to the automatic one. The uploaded video is moved into
`results/<Name>/source/`, which keeps each folder self-contained.
Results made earlier with the CLI or desktop app appear in the list
automatically.

All videos (incident clips and the `data/review_video.mp4` used by the
player) are now H.264 MP4, encoded with PyAV, so they play in any
browser.

**Access and security.** `python serve.py` only listens on this computer.
`python serve.py --host 0.0.0.0` lets other computers on your network
use it. There is **no login yet**: reviewers type their name, and it is
recorded with each decision. Anyone who can reach the address can
upload and review, so only open it on a trusted network. Real user
accounts are the next step before any server deployment.

If the server stops during an analysis, that analysis goes back to the
queue on the next start.

## Quick start — analyze an uploaded exam video (command line / desktop)

```bash
pip install -r requirements.txt
python main.py
```

The app no longer watches the webcam in real time. Instead:

1. **Enter the candidate's name.**
2. **Upload the recorded exam video** (mp4, avi, mov, mkv, webm, m4v, wmv, …).
3. The video is analyzed **frame by frame** with the gaze pipeline below,
   and the app gives a verdict of **CHEATING**, **NO CHEATING**, or
   **INCONCLUSIVE**, based on whether the candidate kept looking at the screen.
4. The results are saved to `results/<Candidate_Name>/`:

```
results/Ali_Khan/
├── recordings/   incident_01_at_00h03m12s_to_00h03m21s.mp4 …  one clip per period of
│                 not looking at the screen (+1 s context each side, red banner while away)
├── snapshots/    incident_01_at_00h03m16s.jpg …                a still of each period
├── report/       report.html  (open this: verdict, timestamp table, snapshots, clip links)
│                 report.txt · report.json · incidents.csv
└── data/         frame_log.csv (every frame) · events.json · calibration.json
```

An existing folder for the same name is never overwritten. A new
analysis goes to `results/<Name>_<YYYYmmdd_HHMMSS>/`.

Other ways to run it:

```bash
python main.py --name "Ali Khan" --video "D:\exams\ali.mp4"     # no window (scriptable)
python main.py --console                                        # text prompts
python main.py ... --annotated-video   # also export the whole video with a live banner
python main.py ... --copy-video        # also copy the original video into the folder
python main.py ... --calibration file  # use calibration_data.json (see below)
```

### How the verdict is decided

A candidate counts as **not looking at the screen** when any of these holds:

| Evidence | Condition (all thresholds in `config.py`) |
|---|---|
| Eyes off the screen | the existing hysteresis state machine confirms `LOOKING_AWAY` (gaze off screen ≥ 3.5 s). The incident's start time is when the gaze *left*, not the later confirmation instant. |
| Eyes turned away | the eyes are held away from the candidate's **own screen-reading position** — sideways, up, down or diagonally toward a corner — for ≥ 3.5 s. Glances back shorter than 1.5 s don't reset the count, which is the typical pattern of reading notes. Measured with MediaPipe's eye-direction scores (`gaze/eye_direction.py`); see "Eye limits and diagonals" below. |
| Face not visible | no face in frame for ≥ 1.5 s |
| Head turned away | head yaw > 35° or pitch > 30° away from the candidate's own median pose for ≥ 3.5 s (covers turns too large for the eyes to be measured) |

Periods less than 1.5 s apart are merged into one. **CHEATING** is
returned if there are 3 or more periods, any single period is 8 s or
longer, or the periods add up to 10% or more of the video. If none of
these is met, the verdict is **NO CHEATING**. The exception is when gaze
could be measured on under 40% of the face-visible frames (bad lighting,
low resolution, etc.): then it is **INCONCLUSIVE**, because the video
can't support a reliable "clean" result. When more than one face is
visible, the report notes it, but it doesn't count toward the verdict.

#### Phone, book and other people (v1.3)

A few frames per second (2 by default) are also checked by an object
detector: Google MediaPipe's EfficientDet-Lite2, trained on the COCO dataset
(Apache-2.0, offline; the ~12 MB model downloads once into
`~/.cache/examvision/`, like the face model).

| Evidence | Counts when | Verdict |
|---|---|---|
| Phone visible | a phone is detected (score ≥ 0.40) for ≥ 1 s | CHEATING |
| Book visible | a book is detected (score ≥ 0.45) for ≥ 1 s | CHEATING |
| More than one person | two faces, or two separate people, for ≥ 2 s | CHEATING |

Each becomes its own incident with a clip and snapshot (boxes drawn around
what was found), separate from the not-looking periods, which keep their own
statistics and rules. Books that stay in the same place for most of the
video (a shelf behind the candidate) are reported as a note, not flagged.
Every threshold and switch is on the Settings page ("Phone, book, people").

Measured before release: phones in the hand were found in 5 of 6 test photos
(a phone held flat and edge-on was missed); books in 3 of 4 (a book held
edge-on was missed); a test video with a phone, a book and a second person
inserted flagged all three at the right times; the six real test videos had
no false phone, book or person. Record a few test videos with a real phone
and book before relying on it.

### Eye limits and diagonals (v1.2)

A webcam sits above the screen, so reading the screen means the eyes are
lowered, by an amount that differs per person. ExamVision first finds each
candidate's **screen-reading eye position** from the video (the most common
roughly-ahead eye position; if another common position sits directly
below it, that lower one, because the higher one is looking above the
screen). Every frame is then compared with that position:

| Limit (Settings → Eyes) | Default | Meaning |
|---|---|---|
| Eyes sideways limit | 0.45 | left/right movement from the screen position |
| Eyes up limit | 0.34 | the screen position found is often the lower part of the screen |
| Eyes down limit | 0.35 | |
| Diagonal label ratio | 0.6 | only changes the label (e.g. DOWN-LEFT) |
| Ignore eye flicker shorter than | 0.4 s | single-frame noise near a limit is not a look-away |
| Eyes nearly closed counts after | 1.0 s | looking far down lowers the lids enough to read as a blink |

The limits form an **ellipse**, so a look toward a corner counts even when
it is moderately off on both axes. Directions are reported in eight ways
(UP, DOWN-LEFT, …), and an incident that mixes them lists them in order,
e.g. `LEFT > UP > DOWN`.

These defaults were tuned on four labelled test recordings (2026-09-30). Tune them on your
own labelled videos with `tools/tune_eye_limits.py` (instructions at the
top of that file and in the Developer Handbook §15a).

## Calibration for recorded videos

> **Lesson from real footage.** An earlier version located the screen
> from the candidate's *median* gaze. On a real recording where the
> candidate read notes beside the laptop for the whole clip, that median
> *was* the notes, so the video scored 0% not looking. Two changes fixed
> this. First, the screen is now fitted only from frames where the eyes
> are centred (v1.2: at the candidate's screen-reading position; "centred"
> picked looking *above* the screen on another video). Second, the
> eye-direction rule above flags sustained
> sideways/down gaze without any calibration. The same recording now
> reports 16 s of 21 s (76%) not looking → CHEATING. The existing 3D
> eye-ray model barely moves (~1.5°) when someone looks down, so it is
> no longer the only line of defence.

A recorded video has no calibration dots, so by default (`--calibration
auto`) the app **locates the screen from the video itself**
(`video_analysis/auto_calibration.py`). The screen center is the median
gaze point over all confident frames. The screen size comes from an
angular prior (25° × 15° half-angles, set in `AutoCalibrationConfig`).
This assumes the candidate looks at the screen for most of the
recording. The report warns you when the data contradict that
assumption.

If the video was recorded on **this** computer's webcam, you can
instead create a personal calibration with `python live_webcam.py`
(press `c`, follow the dots) and then analyze with `--calibration file`.
The live mode is kept only for that purpose.

## Architecture

```
VIDEO FILE (video_analysis/video_source.py, media timestamps)
       → [pass 1, every frame] pipeline.GazePipeline.measure:
WEBCAM → FaceTracker (MediaPipe FaceLandmarker, Tasks API, replaceable backend)
       → FaceGeometryEstimator (solvePnP on STABLE points only) → (R, t)
       → HeadPoseEstimator (yaw/pitch/roll, supporting signal)
       → EyeModel (fixed offsets in face frame → 3D eye centers, camera coords)
       → IrisTracker (robust 2D iris centroid → ray/sphere unprojection → 3D iris)
       → gaze_ray.build_gaze_ray (per eye: O=eye_center, D=normalize(iris-eye_center))
       → binocular_gaze.combine_gaze_rays (closest-approach line geometry)
       → CalibrationManager (multipoint fit) + screen_plane (ray/plane intersection)
       → GazeClassifier (single-frame fusion, eye-dominant)
       → TemporalFilter (EMA) → GazeStateManager (hysteresis) → EventEngine (events)
       → OverlayRenderer (debug UI) / SessionLogger (CSV + JSON)      [live_webcam.py only]

Video mode: pass 1 = measure (geometry per frame) → auto-calibration →
pass 2 = GazePipeline.decide (classifier → EMA → hysteresis → events, on
video time) → video_analysis/incidents.py (incidents + verdict) →
pass 3 = exporter.py (clips, snapshots) → report.py.
```

`pipeline.py` holds the per-frame logic that used to live in `main.py`,
split into `measure()` and `decide()`, so live and video modes run the
same code.

## Bugs fixed in the existing pipeline (found while testing with real landmarks)

Before these fixes, a clear frontal face gave **0 of 1380 frames** with
usable gaze, so the system reported `UNCERTAIN` on every frame. That was
true in the original live mode as well. After the fixes, median gaze
confidence on the same video is 0.80. Regression tests are in
`tests/test_gaze_fixes.py` and `tests/test_head_pose.py`.

1. **Iris validity was capped at ~0.49.** The "iris ring" 468–472
   includes the iris *center* landmark (468/473). Its radius is ~0, which
   gives a ring-roundness coefficient of variation of ~0.5 on every
   frame. Validity now uses the 4 contour points only
   (`LEFT/RIGHT_IRIS_CONTOUR`).
2. **The combined gaze ray started at the fixation point.** For
   near-parallel rays, that point slid hundreds of units along the gaze
   line, and it could land beyond the screen plane, making the
   intersection invalid. The ray now starts at the midpoint between the
   eye centers.
3. **The binocular disagreement penalty was on the wrong scale.** It
   divided the ray separation by a fixed 150. In face-model units at real
   viewing distances, normal ~2° landmark noise already reaches ~140,
   which halved every frame's confidence. It is now an angle
   (separation ÷ viewing distance), capped at
   `max_binocular_disagreement_deg`.
4. **Head-pose axes were mislabeled.** `solvePnP` returns a 180° flip
   about X for a frontal face. Decomposing that directly reported a
   left/right *turn* as "pitch", tilt as "yaw", and a nod as "roll ±
   180°". Angles are now taken relative to the frontal pose, so frontal
   = (0, 0, 0), yaw = turn, and pitch = nod.

Modules are independent: `FaceTracker` is the only file that imports
MediaPipe, so the landmark backend can be swapped without touching
geometry, gaze, or proctoring logic.

## Coordinate systems (never mixed)

| Frame | Origin / axes | Used by |
|---|---|---|
| **Image** | pixels `(u, v)`, top-left origin | landmark input, overlay drawing |
| **Camera** | OpenCV convention: X right, Y down, Z forward from the lens optical center | eye centers, iris positions, gaze rays, screen plane — the frame almost everything is computed in |
| **Face model** | rigid frame attached to the head, origin at the nose tip (`FACE_MODEL_3D` in `config.py`) | canonical target for `solvePnP`; eye-center offsets are defined here, then transformed to camera coords each frame |
| **Screen-local (plane u, v)** | 2D coords on the assumed screen plane, before calibration's affine correction | intermediate calibration/classification step |
| **Normalized screen (`sx, sy` ∈ [-1, 1])** | final, calibrated screen-facing coordinate | classification output |

## 1. Stable facial reference strategy

`solvePnP` is fit using **only** points that don't move when the eyes
rotate: nose tip, chin, both eye **outer corners**, both mouth
corners (`FACE_MODEL_LANDMARK_IDS`). Eyelids and iris points are
excluded from pose solving for exactly the reason called out in the
project brief: eye-region landmark positions shift with eye rotation,
which would otherwise bias the estimated head pose. The 3D model used
is the standard generic 6-point face model from classical OpenCV
head-pose tutorials — it is *not* subject-specific, which is a
documented limitation (see below).

## 2. Facial pose estimation

`cv2.solvePnP` with the 6 model↔landmark correspondences and an
approximate pinhole intrinsic matrix (focal length ≈ image width,
principal point at image center — a standard uncalibrated-webcam
approximation) yields a rotation vector/matrix `R` and translation
vector `t` such that `X_camera = R · X_face_model + t`. Reprojecting
the model points and comparing to the observed landmarks gives a
reprojection-error quality metric used to gate downstream trust.

## 3. Head pose

Yaw/pitch/roll are derived from `R` via the standard Y-X-Z Euler
decomposition (`face/head_pose.py`), with the classical gimbal-lock
fallback near ±90° pitch. Head pose only ever **modulates confidence**
in `GazeClassifier`; it cannot flip a confident eye-gaze verdict on
its own (spec requirement: eye gaze dominates).

## 4. Eye-center estimation

Each eyeball center is a **fixed offset inside the rigid face-model
frame**: medially shifted from the outer eye corner toward the nose,
shifted back into the socket, and slightly down (`EyeGeometryConfig`).
Because the offset is defined in face coordinates, **eye rotation
never moves it** — only head pose (`R, t`) does. This directly avoids
the naive `mean(eyelid_landmarks)` approach the brief explicitly
rules out. The offset magnitudes are anthropometric approximations
(documented limitation — no per-subject orbit measurement is
possible from a single webcam).

## 5. Iris-center estimation

The four MediaPipe iris-contour landmarks per eye are centroided; the
**coefficient of variation of their radii around that centroid** is
used as a validity score (a near-circular, tight ring ⇒ trustworthy;
a degenerate ring ⇒ closed/occluded eye, rejected below
`min_iris_validity`). The 2D centroid is then **unprojected to 3D**
by intersecting the camera ray through that pixel with a sphere of
anatomical eyeball radius centered at the 3D eye center
(`iris_tracker.py`); the near intersection is taken as the iris's 3D
position. This constrains the iris to lie on a physically modeled
eyeball surface rather than an assumed constant depth.

## 6. Individual gaze rays

`R(t) = O + tD`, `O = eye_center_3d`, `D = normalize(iris_3d -
eye_center_3d)` — built independently for each eye, entirely in
camera coordinates (`gaze/gaze_ray.py`).

## 7. Binocular ray combination — closest approach

Given skew lines `L1(t) = O1 + tD1`, `L2(s) = O2 + sD2`, let `w0 = O1
- O2`:

```
a = D1·D1,  b = D1·D2,  c = D2·D2,  d = D1·w0,  e = D2·w0
denom = ac - b²
t = (be - cd) / denom
s = (ae - bd) / denom
```

`P1 = O1 + tD1`, `P2 = O2 + sD2`; the combined ray's origin is the
midpoint of the two eye centers `(O1+O2)/2` and its direction is the
confidence-weighted, renormalized average of `D1, D2`. `|P1-P2|` (the
closest-approach separation), as an angle seen from the camera, is
used as a disagreement penalty on combined confidence — large
separation usually reflects noisy iris estimates rather than a real
anatomical effect. Near-parallel rays fall back to the midpoint of the
two origins. One eye's estimate being invalid (blink, occlusion)
degrades gracefully to a monocular ray at reduced confidence rather
than failing outright.

## 8. Virtual screen plane & ray/plane intersection

**Documented simplifying assumption**: a single uncalibrated
monocular webcam cannot observe the true tilt of the physical screen
relative to the camera, so the plane is assumed **parallel to the
camera sensor** (`normal = (0,0,-1)`), at a distance solved during
calibration. Given plane origin `P`, normal `N`:

```
t = ((P - O) · N) / (D · N)
```

`|D·N| < ε` ⇒ ray parallel to the screen ⇒ invalid intersection
(never forced to a number). Otherwise `X = O + tD`, converted to plane
coordinates `(u, v) = ((X-P)·u_axis, (X-P)·v_axis)`.

## 9. Multipoint calibration & personalization

`CalibrationManager` walks 9 normalized targets (`center, left,
right, up, down`, plus 4 corners), collecting `samples_per_point`
ray/plane intersections per target. It then:

1. Searches candidate plane distances (300–1200 model-units) and,
   for each, analytically rescales the stored `(u,v)` (valid because
   for a plane parallel to the image plane, `(u,v)` scales linearly
   with distance for a fixed ray) and fits an **affine transform**
   `(sx, sy) ≈ A·(u,v) + t` to the known target coordinates via least
   squares, picking the distance that minimizes fit RMSE.
2. This affine correction is what makes calibration *personalized*
   (spec §15): it absorbs the individual's face geometry, eye offset
   error, camera/screen placement, and viewing distance — all of
   which appear as systematic bias in the raw `(u,v)` measurements.
3. Computes a **quality score** from fit RMSE (in normalized-target
   units) and sample completeness, and returns `GOOD`,
   `LOW_CONFIDENCE`, or `FAILED` with actionable feedback (move
   closer, improve lighting, keep face visible, follow the dot,
   avoid excess head movement) — never silently accepting a bad
   calibration.
4. Persists to `calibration_data.json` and reloads automatically next
   run.

## 10. Fusion (eye-dominant)

`GazeClassifier` computes the calibrated `(sx, sy)`, checks it
against the screen rectangle **with a configurable margin** (so being
one or two pixels outside the edge isn't instantly "away"), and only
applies a modest confidence penalty from head pose when head yaw/pitch
is extreme — never a hard override. Anything below
`min_gaze_confidence` becomes `UNCERTAIN` rather than a confident
guess (spec §32: no fake accuracy).

## 11. Temporal filtering & state machine

Continuous signals (`screen_x, screen_y, confidence`) are smoothed
with an EMA (O(1) memory/time, real-time friendly) rather than
averaging discrete labels. `GazeStateManager` then applies **duration
hysteresis** on top of the smoothed label:

```
LOOKING_AT_SCREEN --(deviation ≥ seconds_to_confirm_away)--> LOOKING_AWAY
LOOKING_AWAY       --(inside ≥ seconds_to_confirm_screen)--> LOOKING_AT_SCREEN
```
with `POSSIBLE_AWAY` / `POSSIBLE_SCREEN` transitional states so a
200ms deviation self-cancels instead of firing an event, per the
spec's worked example.

## 12. Events (not verdicts)

`EventEngine` emits structured observations only —
`LOOKING_AWAY_STARTED/CONTINUED`, `LOOKING_AT_SCREEN_RESUMED`,
`FACE_LOST`, `MULTIPLE_FACES_DETECTED`, `LOW_GAZE_CONFIDENCE`,
`CALIBRATION_FAILED` — each gated by its own persistence duration
(`TemporalConfig`), so a single noisy frame never fires
`FACE_LOST`/`MULTIPLE_FACES_DETECTED`. It intentionally makes **no
cheating determination**; that policy layer is future work (§36).

## Known limitations (read before treating output as ground truth)

- **The verdict is a screening result, not proof.** It is based on gaze
  only. Open the clips before you act on it. The thresholds are sensible
  defaults, not values validated against labelled exam footage.
- **Video mode assumes a front-facing webcam near the screen**, as on a
  laptop or a monitor-top camera, and a candidate who looks at the
  screen for most of the recording (see "Calibration for recorded
  videos").
- Processing speed is roughly real time to a few times faster, depending
  on the CPU. A 1-hour video takes a while, and the app shows progress
  and an estimated time remaining.

- **Generic (not per-subject) face and eye-socket model.** Eye-center
  offsets are anthropometric approximations, not measured from the
  individual. Calibration's affine correction partially compensates
  but does not fully eliminate this.
- **Assumed screen-parallel plane.** Real screen tilt relative to the
  webcam is not observed; only scale/offset/shear are corrected by
  calibration, not arbitrary 3D tilt.
- **Uncalibrated intrinsics.** Focal length is approximated from image
  width; no chessboard camera calibration is performed.
- **Confidence ≠ validated accuracy.** All confidence values are
  internal quality proxies (ring consistency, reprojection error,
  calibration RMSE) — they are not a measured detection accuracy. See
  `EVALUATION.md` workflow below before quoting any accuracy number.
- Extreme head rotation (beyond `max_reliable_yaw/pitch_deg`), poor
  lighting, glasses glare, or landmark dropout will correctly degrade
  confidence and should trend toward `UNCERTAIN`, not a wrong
  confident label — this is by design (§32/§28), but hasn't been
  formally measured per the spec's evaluation section (§33); the
  `tests/` directory covers geometry, not end-to-end accuracy.
- The local package is named `proctor_logging/`, **not** `logging/`
  as the originally sketched folder layout suggested — a package
  named `logging` on the script's own directory would shadow Python's
  standard library `logging` module (which MediaPipe/absl use
  internally) and crash the app. This is a deliberate, documented
  deviation from the literal folder name for correctness.

## Dependencies & licenses (all free for commercial use, no cloud/API keys)

| Package | License | Source |
|---|---|---|
| opencv-python | BSD-3-Clause | https://github.com/opencv/opencv-python |
| mediapipe | Apache-2.0 | https://github.com/google-ai-edge/mediapipe |
| numpy | BSD-3-Clause | https://numpy.org/ |
| scipy | BSD-3-Clause | https://scipy.org/ (used only in tests) |

No paid APIs and no cloud inference — the entire pipeline runs locally
on-device. One exception: on first run only, `FaceTracker` downloads
Google's free, open `face_landmarker.task` model bundle (a few MB) to
a local cache (`~/.cache/examvision/`); every run after that is fully
offline. To pre-fetch it (or deploy to an offline machine), download
that URL yourself and place it at `~/.cache/examvision/face_landmarker.task`,
or pass an explicit `model_path=` to `FaceTracker`.

## Testing

All tests are pure unit tests: no webcam, video, or MediaPipe needed.
`test_geometry.py` and `test_calibration.py` cover the original
geometry. `test_head_pose.py` and `test_gaze_fixes.py` are regression
tests for the fixed bugs. `test_incidents.py` covers incident
timestamps, merging, and every verdict rule. `test_video_analysis.py`
covers auto-calibration, results-folder naming, and a replay of the
real classifier and state machine on video time:

```bash
pip install pytest
python -m pytest tests/ -q
```

### Manual test-mode checklist (spec §27/§33)

Record a test video (or run `live_webcam.py` and calibrate), then deliberately perform and record
results for: look at screen / left / right / up / down; turn head
left/right with eyes locked on screen; turn head while eyes move away
with head centered; close eyes; move closer/farther; leave frame
briefly; introduce a second person; and try to induce landmark
failure (fast motion, hand over face). For each, the `r` key records a
timestamped CSV + JSON event log to `session_logs/` for later
precision/recall/F1 computation against your own ground-truth
annotation of each segment — no such accuracy numbers are hard-coded
or claimed by the system itself.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `CALIBRATION_FAILED` repeatedly | Poor lighting, face partially out of frame, or too much head movement during calibration — recalibrate slower, centered, well lit. |
| Gaze rays look erratic in the debug overlay | Check `GAZE CONFIDENCE` — low iris-ring validity (glare/glasses/blink) is the usual cause; the classifier should already be discounting these frames. |
| Frequent `UNCERTAIN` | Expected under low confidence by design — improve lighting/distance, or lower `min_gaze_confidence` in `config.py` at the cost of more false confident labels. |
| Low FPS | Lower `camera.processing_width`, or disable overlay toggles (`l/i/e/g/h`) which each add draw cost. |
| `MULTIPLE_FACES_DETECTED` never fires with a visible second person | Increase `max_num_faces` passed to `FaceTracker` (default 2) if testing with more people. |

## Future extensibility (explicitly deferred, not implemented here)

Phone/book/paper/object detection, stronger multi-person identity
handling, browser tab/fullscreen/focus monitoring, the exam
interface, authentication, session management, server-side
monitoring, and a proctoring dashboard are all out of scope for
Phase 1 by design (spec §36) — this module is meant to be imported as
the gaze-detection engine underneath that future system, which is why
every component is isolated behind plain dataclasses rather than
MediaPipe/OpenCV types leaking through the public interfaces.
