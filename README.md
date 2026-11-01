# Badminton Serve Evaluation

A computer-vision system that scores badminton serves for legality and accuracy. It uses a YOLO-based shuttlecock detector across two synchronized cameras (side + back view), automatically aligns the two camera feeds frame-by-frame, triangulates the shuttlecock into 3D court coordinates, and evaluates the resulting trajectory against BWF-style serve rules — landing placement, net clearance, and peak height.

## How It Works

1. **Upload** — The user uploads a side-view and a back-view video of the same serve through the web app.
2. **Frame extraction** — Every frame of both videos is extracted to disk as individual JPGs.
3. **Automatic synchronization** — `matching.py` detects the shuttlecock in each frame of both views and builds a per-camera motion signature (x, y, velocity magnitude, acceleration magnitude). It assumes both cameras were started together (e.g. a manual "1, 2, 3, click" start) and only searches a small window of starting lags — up to 7 frames — right at the beginning of the clips, rather than sliding across the whole video. Whichever camera's clip is shorter becomes the time base: every one of its frames is paired 1-to-1 with a consecutive frame of the longer clip at the best-matching start offset, and any leftover frames at the tail of the longer clip are dropped. The app saves the result as new synchronized videos (`side_sync.mp4` / `back_sync.mp4`) plus a `matched.json` record that also includes the chosen offset and a match-confidence score.
4. **Calibration** — Ground control points (GCPs) on the court are mapped in each camera view to real-world coordinates, producing a projection matrix per camera (`calibrate.py` / `calibrate_core.py`). The calibration frames used are the first synchronized pair, so both cameras are calibrated against the exact same instant. The user first picks the **service area** the serve is aimed into — right (default) or left — with a toggle on the calibration page; this decides which real-world coordinates GCP3–GCP6 are solved against (see "Service area" under Serve Types) and is saved with the calibration. The web app's calibration page is semi-automatic: `calibrate_core.auto_detect_gcps` proposes an initial pixel position for all 7 GCPs on each camera before the user looks at anything, and `calibrate.html` opens with those positions already placed as draggable markers rather than an empty canvas to click through. Points the auto-detector is confident about are marked green; points it isn't sure about (most often the net markers, which are harder to detect than the floor corners) are marked orange and flagged "?" until the user drags them into place. A dashed rectangle connecting GCP3→GCP4→GCP6→GCP5 (the real floor rectangle of the selected service box — note the walk order, GCP4→GCP5 is the box's diagonal, not an edge) is drawn on top of the frame as a visual sanity check: if it doesn't line up with the real court lines, the auto-guess got something wrong and needs correcting. See "Semi-automatic calibration" under Usage below for how the guess itself is produced and what its known limitations are.
5. **Triangulation** — Shuttlecock detections from the synchronized videos are combined with the calibration data to reconstruct the shuttlecock's 3D trajectory (X, Y, Z per frame) (`triangulate.py`). The tracker also reports per-camera motion diagnostics, used to flag likely bad detections (see below). For clips containing several serves, tick **Several serves in this video** on the calibration page: the tracker then keeps every detection, ignores shuttles resting on the court, and `multi_serve.py` splits the result into one trajectory per serve (see "Multiple serves in one video").
6. **Evaluation** — The 3D trajectory's true floor-landing point is estimated (interpolating to Z = 0 past the apex, not just the last tracked frame; see Notes), the serve is auto-classified as short or high/long from landing depth, and it's scored against the classified serve's rules with three component scores, each on a 0–100 scale — landing placement (LPS, distance to the nearest drop-box corner), net clearance (NCS), and peak height (PHS) — whose weighted average (equal weights by default) is the total out of 100 (`evaluator.py`).
7. **Visualization** — Results are shown as an interactive dashboard with a trajectory chart and landing map (`app.py`, Flask web app), or printed as a CLI report.

Both the frame-synchronization step and the evaluation step run as background jobs that the browser polls for progress, so large videos don't block the request.

## Repository Structure

| File / Folder | Purpose |
|---|---|
| `app.py` | Flask web app — upload videos, auto-extract & synchronize frames, calibrate the court in-browser, run evaluation jobs, and view results on a dashboard. Also serves an optional sync-verification debug page. |
| `matching.py` | Automatic frame synchronization — detects the shuttlecock per frame in both camera folders, builds a kinetic motion signature (x, y, velocity, acceleration), searches a small start-lag window assuming both cameras began recording together, and pairs the shorter clip 1-to-1 against the longer one from the best-matching start. Runnable standalone as a CLI tool. |
| `run_pipeline.py` | CLI entry point — runs triangulation + evaluation on a fixed pair of video files and prints the report. |
| `calibrate.py` / `calibrate_core.py` | Court calibration: solves for each camera's projection matrix from 7 ground control points via normalized DLT + nonlinear reprojection refinement. `calibrate.py` is the interactive CLI (click the GCPs on a still image, fully manual — it does not use the auto-detection below). `calibrate_core.py` holds the shared math, GCP definitions (one table per service side, `WORLD_GCPS_BY_SIDE`), and camera metadata used by the web app (the CLI keeps its own right-side table), plus the web app's semi-automatic calibration: `auto_detect_gcps` (best-effort initial GCP guess — landmark feature-matching first, classical court-line detection as a fallback), `match_against_landmarks` / `save_landmark` / `list_landmarks` (the per-camera landmark history used for that feature-matching). |
| `calibration_landmarks/` | Per-camera, per-service-side history (up to 5 most recent, oldest evicted) of confirmed web-app calibrations — each a saved frame plus its 7 confirmed GCP pixel points. Used by `auto_detect_gcps` to seed future calibrations on the same camera rig via feature matching instead of re-running line detection from scratch. Created automatically; safe to delete to force fresh line-detection next time. |
| `triangulate.py` | `BadmintonTracker3D` — runs YOLO shuttlecock detection on both synchronized video feeds, triangulates 3D positions via DLT, and reports per-camera motion diagnostics. |
| `multi_serve.py` | Multi-serve support: flags detections that sit still (resting / held shuttles), picks the right side/back candidate pair per frame by reprojection error, and splits the tracked 3D points into one segment per serve. Pure functions, no video or YOLO needed to test them. |
| `evaluator.py` | `evaluate_serve_performance` — estimates the true landing point, auto-classifies the serve type, judges IN/OUT against the selected service area's drop box, and computes the three 0–100 component scores (LPS, NCS, PHS) and their weighted-average total. Takes `service_side` (`"right"` / `"left"`) and optional `weights`. |
| `train.py` | Trains the YOLOv8 shuttlecock-detection model on `shuttle_dataset/`. |
| `split_dataset.py` | Splits/prepares the raw dataset into train/val sets for training. |
| `visualize_check.py` | Utility for visually sanity-checking detections/calibration on sample frames. |
| `shuttle_dataset/` | Training data for the YOLO shuttlecock detector. |
| `templates/` | HTML templates for the Flask app — `upload.html`, `calibrate.html`, `dashboard.html`, and the optional `sync_verify.html` debug page. |

## Serve Types

The evaluator auto-classifies the serve from where the shuttle actually lands (it does not need to be told which serve was intended):

- **`short_front_corner`** ("Short serve") — landing X between the short service line (1.98 m) and 4.72 m. Rewards a low net clearance (ideal ≤ 0.20 m) and a low peak height (ideal ≤ 1.50 m).
- **`high_back_corner`** ("High serve") — landing X between 4.73 m and the back boundary (6.70 m). Rewards a high net clearance (ideal ≥ 1.00 m) and a high peak height (ideal ≥ 3.50 m).

IN/OUT is judged independently of serve type, against the preferred drop box of the selected service area (X: 1.98–6.70 m; Y: 0.00 to +2.59 m for the right area, −2.59 to 0.00 m for the left area, with a ±0.05 m tolerance for triangulation noise); a landing outside that box is OUT and the final score is forced to 0 (the three component scores are still reported, and `score_before_out_rule` keeps the total it would have had). If a serve lands in the *mirror* box, the OUT reason says so, as a hint that the wrong service area may have been selected. Internally the short/high split is at X = 4.725 m (the midpoint of 4.72 and 4.73), so no landing in the 0.01 m display gap goes unclassified.

### Service area (right / left)

The calibration page has a **Service area** toggle (right by default; switching reloads the page with a fresh auto-guess and asks first if points were already moved). Coordinates: origin at the net center, X = depth from the net, +Y = right half, −Y = left half. GCP0–GCP2 (net) are the same for both areas; GCP3–GCP6 (floor corners) change:

| GCP | Right service area | Left service area |
|---|---|---|
| GCP0 | (0.00, 0.00, 1.524) net center | same |
| GCP1 | (0.00, −2.59, 1.550) left net/post | same |
| GCP2 | (0.00, 2.59, 1.550) right net/post | same |
| GCP3 | (1.98, 0.00, 0) short line / center | (1.98, −2.59, 0) short line / left sideline |
| GCP4 | (1.98, 2.59, 0) short line / right sideline | (1.98, 0.00, 0) short line / center |
| GCP5 | (6.70, 0.00, 0) back boundary / center | (6.70, −2.59, 0) back boundary / left sideline |
| GCP6 | (6.70, 2.59, 0) back boundary / right sideline | (6.70, 0.00, 0) back boundary / center |

The chosen side is stored inside `web_calibration.npz` next to the projection matrices, and the evaluation reads it back, so calibration and IN/OUT always use the same side. A calibration file saved before this option existed has no side and is treated as right.

**Select the side that matches the box you actually click.** Choosing the wrong one does not raise a calibration error — in a synthetic-camera test the reprojection error stayed around 1 px — but it shifts every landing Y by about 2.6 m, so a left-box landing would be reported inside the right box. The legend above the canvas lists what GCP3–GCP6 mean for the selected side, and the dashboard shows the service area next to the landing map.

### Landing target

On the upload page the coach chooses what the landing score aims at. The choice is saved as soon as it changes (in `landing_target.json` next to `app.py`) and is used by the next evaluation; the calibration page shows the current choice.

- **All four corners (GCP3–GCP6)** — default. The nearest corner of the service box is the target.
- **One grid point** — click one intersection of a 7 × 9 grid drawn over the service box. Columns 1–7 run from the **center line** to the sideline in steps of 0.432 m (2.59 m / 6); rows 1–9 run from the short service line (X = 1.98 m) to the back boundary (X = 6.70 m) in steps of 0.59 m (4.72 m / 8). The grid has 63 intersections, including the four corners. Columns are counted from the center line, so the same (row, column) is the mirror-image spot in the left service area.

The same fall-off is used for both options (100 on the target, 0 at 2.69 m, the box-center-to-corner distance), so LPS means the same thing whichever target is chosen. That radius is a tunable constant (`LANDING_ZERO_SCORE_DISTANCE_M` in `evaluator.py`), not a BWF rule. For a single grid point the result page also draws a dashed ring at the target on the landing map, and the deduction line names the target (for example "grid point row 4, col 3 (X 3.75, Y 0.86)"). `evaluate_serve_performance(..., landing_target={"mode": "point", "col": 2, "row": 3})` takes 0-based indices; the default (`None`) is the corner mode.

### Scoring

Each serve gets three component scores, each on its own **0–100** scale. The total (0–100) is their weighted average:

| Component | Abbrev. | How it is scored (0–100) |
|---|---|---|
| Landing placement | LPS | Distance `d` from the landing point to the **landing target**: by default the nearest of the four drop-box corners of the selected area (right: GCP3 = (1.98, 0.00), GCP4 = (1.98, 2.59), GCP5 = (6.70, 0.00), GCP6 = (6.70, 2.59); left: Y becomes −2.59 and 0.00), or one grid point chosen by the coach (see below). LPS = `100 × max(0, 1 − d / 2.69 m)`: 100 on the target, 0 at 2.69 m or more away (with the corner targets, 0 at the box center). The same targets apply to both serve types. |
| Net clearance | NCS | Linear ramp between the thresholds below. Hitting the net (clearance < 0 m) gives 0. |
| Peak height | PHS | Linear ramp between the thresholds below. |

Total = `w_LPS·LPS + w_NCS·NCS + w_PHS·PHS`, with **equal weights (1/3 each) by default**, i.e. the plain average of the three. The weights are set in one place (`DEFAULT_WEIGHTS` in `evaluator.py`, or the `weights=` argument; they are normalised to sum to 1) so the effect of the weighting on the results can be tested rather than assumed — an equal average is itself a weighting choice. The earlier 50/30/20-point split is no longer used.

NCS and PHS thresholds (linear interpolation between the two values, clamped to 0–100):

| Metric | Short serve | High serve |
|---|---|---|
| Net clearance (NCS) | 100 at ≤ 0.20 m, 0 at ≥ 1.00 m | 0 at ≤ 0.20 m, 100 at ≥ 1.00 m |
| Peak height (PHS) | 100 at ≤ 1.50 m, 0 at ≥ 3.50 m | 0 at ≤ 2.00 m, 100 at ≥ 3.50 m |

The full report includes the overall score, serve classification, service area, landing coordinates and status, net clearance, peak height, the three component scores (`component_scores`), and the weights used.

## Requirements

- Python 3.8+
- `opencv-python`
- `numpy`
- `ultralytics`
- `scipy`
- `flask`
- `matplotlib`
- `torch`
- `ffmpeg` on `PATH` (used to re-encode synchronized videos to H.264 for in-browser playback; the app falls back to a plain copy if it's missing, but that copy won't play in the dashboard's `<video>` preview)

Install the Python packages with:

```bash
pip install -r requirements.txt
```

## Usage

### Web App

```bash
python app.py
```

Then open `http://localhost:5000` in a browser to:
1. Upload a side-view and back-view video of the serve. On the same page, optionally choose the **landing target** (see "Landing target" under Serve Types); the default is the four box corners.
2. Wait for automatic frame extraction and synchronization to finish (progress is polled live).
3. Choose the service area (right or left), then review the auto-placed ground control points on each synchronized camera frame, drag any that are wrong into place, and save the calibration. See "Semi-automatic calibration" below.
4. Run the evaluation (tick **Several serves in this video** if the clip has more than one serve) and view the scored report, trajectory chart, and landing map on the dashboard.

If you want to sanity-check the synchronization itself, `/sync-verify/<job_id>` is available as an optional debug page (linked from the calibration page) — it shows the matched frame pairs side by side with a scrub/play control so you can visually confirm both views show the same instant.

#### Semi-automatic calibration

The calibration page (`/calibrate`) no longer starts from an empty canvas. For each camera, `auto_detect_gcps` (in `calibrate_core.py`) tries, in order:

1. **Landmark match** — feature-match (ORB + RANSAC homography) the new frame against up to 5 previously confirmed calibrations for that camera and service area (see `calibration_landmarks/` above). If the camera hasn't moved much since the last confirmed session, this warps the old confirmed points onto the new frame and is the strongest signal available — all 7 points come back marked confident.
2. **Court line detection** — if no landmark match is confident enough (first-ever calibration of a camera, or the rig has clearly moved), classical line detection (white-tape color mask + Hough lines + line intersection) locates the four floor corners (GCP3–GCP6), and the net points (GCP0–GCP2) are approximated by extrapolating the floor lines to the net and looking for the net post.
3. **Generic fallback** — if even that finds nothing, every point still gets *some* on-screen position so there's always a marker to drag, rather than nothing to work with.

Every point is color-coded on the page: **green** means the auto-detector was confident, **orange with a "?"** means it's a low-confidence guess that should be checked. A dashed rectangle over GCP3–GCP4–GCP6–GCP5 is drawn as a visual check — it should sit right on the real floor rectangle of the selected service box; if it looks rotated or mirrored relative to the real court lines, drag the corners until it matches.

Known limitations, worth knowing before trusting the auto-guess blindly:

- **Net points (GCP0–GCP2) are the least reliable.** They're a geometric extrapolation from the floor rectangle, not a direct detection, and are always marked orange for review.
- **Floor-corner role assignment (which detected corner is "near" vs. "far", "center" vs. "right sideline") is a heuristic** and depends on the camera's exact position/orientation. It can occasionally come out swapped even when the four corners themselves were found correctly — this is what the rectangle overlay is for: if it's mirrored or rotated relative to the real court, the roles got mixed up and need dragging into the correct spots.
- **Extreme camera angles can defeat line detection entirely.** If the floor lines are foreshortened into very short segments in frame (a camera positioned nearly edge-on to a court boundary), detection can fail and fall back to the generic layout — expect to place those points manually in that case.
- **Left service area:** the floor-line heuristic assumes the camera sees the left box from the same pose as the right box (a mirror image across the center line), so GCP3/GCP5 are the left sideline and GCP4/GCP6 the center line. If your rig is placed differently, expect to drag those points. The left-area landmark history is kept separately (`side_left/`, `back_left/`), so right-area landmarks are never used for the left area.
- None of this has been validated against real match footage with varied lighting/backgrounds beyond initial testing — expect to need to retune thresholds (e.g. the HSV white-tape range or Hough `minLineLength` in `calibrate_core.py`) for your specific gym/lighting setup if detection is consistently off.

Once a calibration is saved, it's stored as a landmark for that camera, so the *next* session for the same rig should start from a landmark match (green, high-confidence) rather than line detection.

### Multiple serves in one video

Tick **Several serves in this video** before running the evaluation. Left unticked, the tracker behaves exactly as before: one detection per frame and the whole clip is one serve.

When ticked:

1. **All candidates are kept.** YOLO runs on every frame of both videos and every shuttlecock-like detection is stored (the old path kept only the single highest-confidence one, which could jump to a shuttle already lying on the court).
2. **Resting shuttles are removed.** Per camera, a detection that sits on the same spot (within 0.5% of the frame diagonal) in at least 0.6 s worth of frames is treated as a shuttle at rest or held in the hand and ignored. This is the "a trajectory that doesn't change over a number of frames" check. The time threshold is longer than the shuttle lingers near its apex, so a shuttle in flight is not removed.
3. **The right pair is chosen.** If a frame has several moving candidates, every side/back pair is triangulated and the one that reprojects best into both images is used. A pair is rejected if its reprojection error is above 1% of the frame diagonal (several candidates) or 3% (one candidate per camera, which also catches the shuttle in one camera paired with a false positive in the other).
4. **The points are split into serves.** A new serve starts when the shuttle rises above 0.5 m, and ends at floor contact (Z ≤ 0.15 m after the apex), at a pause of more than 0.5 s without detections, or at an impossible jump (> 80 m/s). Resting points on the floor are ignored, fragments with fewer than 6 points or less than 1 m of movement are discarded, and single-frame tracking spikes are dropped before splitting.
5. **Each serve is scored separately.** In the Trajectory summary, click the numbered buttons (**1**, **2**, **3** …; the `<` `>` arrows and the left/right arrow keys also work) to switch between serves: the score, deductions, landing marker and serve type change, and both camera videos jump to just before that serve; the headline score is the average over the serves, the trajectory chart overlays them, and a serve whose track ended before it reached the floor is flagged in the warnings.

All thresholds are starting values at the top of `multi_serve.py`; they are time-based, so 30 and 60 fps clips behave alike. They were checked on synthetic two-camera scenes (two serves, resting shuttles, missed detections, false positives, one frame of sync lag), not yet on real match footage, so expect to tune them on your own clips. Known limits: a serve that is not detected near the floor is scored from its last tracked points; two serves in the air at the same time are not separated; and synchronisation (`matching.py`) uses motion differencing, which already ignores resting shuttles, but a clip that starts with a serve already in the air still needs both cameras started together.

### Training the Detector

To train (or retrain) the shuttlecock detection model on your own dataset:

```bash
python split_dataset.py   # prepare train/val splits
python train.py           # trains YOLOv8n on shuttle_dataset/data.yaml
```

## Notes

- Net height is assumed to be 1.55 m in the evaluation logic. Net clearance is measured at the trajectory point whose X is closest to the net (X = 0), minus 1.55 m.
- Landing point: starting from the apex, the evaluator takes the first point at or below Z = 0.15 m and interpolates X/Y to Z = 0. If the trajectory never gets that low, it falls back to the lowest tracked point after the apex.
- Synchronization assumes the two cameras were started together (manual sync) and only corrects for a small reaction-time lag (up to 7 frames) at the start of the clips — it is **not** designed to find an arbitrary offset anywhere in a long, independently-started recording. It also assumes a constant offset once found (no clock drift during the clip).
- The tracker flags a camera as "suspect static" if its detected shuttlecock barely moves across the clip — usually a sign the detector locked onto a stationary object (a resting shuttle, the racket, a bright spot on the net) instead of the shuttle in flight. These show up as warnings on the results dashboard rather than failing the evaluation outright. When multiple shuttlecock-like objects are detected in a frame, the tracker now keeps the highest-confidence one rather than whichever the model happened to return first.
- The web app stores original uploads under `uploads_video/sync_job_<id>/`, synchronized videos as `uploads_video/side_sync.mp4` / `back_sync.mp4`, calibration in `web_calibration.npz`, and matched-pair metadata in `matched.json`. Confirmed calibrations are additionally stored per-camera under `calibration_landmarks/` for future auto-calibration (see "Semi-automatic calibration" above); this grows unbounded across different physical setups since it's keyed only by camera name and service side (`side`, `back`, `side_left`, `back_left`), not by location, so delete it if you move the rig somewhere new and want a clean slate instead of a stale landmark match.
- `calibrate.py`, the CLI calibration tool, is still fully manual (click all 7 GCPs on a still image) and right-service-area only (it keeps its own copy of the right-side GCP table) — the auto-detection and landmark matching described above are web-app-only, in `app.py`'s `/calibrate` route.
- The scoring thresholds in `evaluator.py` are initial, tunable criteria — not official BWF rules — pending calibration against measured high-level/pro serves.
- This project is part of a course research effort (CSS451/454, SIIT, Thammasat University).

## License

No license specified yet.