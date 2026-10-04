# Badminton Serve Evaluation

A computer-vision system that scores badminton serves for legality and accuracy. It uses a YOLO-based shuttlecock detector across two synchronized cameras (side + back view), automatically aligns the two camera feeds frame-by-frame, triangulates the shuttlecock into 3D court coordinates, and evaluates the resulting trajectory against BWF-style serve rules — landing placement, net clearance, and peak height.

## How It Works

1. **Upload** — The user uploads a side-view and a back-view video of the same serve through the web app.
2. **Frame extraction** — Every frame of both videos is extracted to disk as individual JPGs.
3. **Automatic synchronization** — `matching.py` detects the shuttlecock in each frame of both views and builds a per-camera motion signature (x, y, velocity magnitude, acceleration magnitude). It assumes both cameras were started together (e.g. a manual "1, 2, 3, click" start) and only searches a small window of starting lags — up to 7 frames — right at the beginning of the clips, rather than sliding across the whole video. Whichever camera's clip is shorter becomes the time base: every one of its frames is paired 1-to-1 with a consecutive frame of the longer clip at the best-matching start offset, and any leftover frames at the tail of the longer clip are dropped. The app saves the result as new synchronized videos (`side_sync.mp4` / `back_sync.mp4`) plus a `matched.json` record that also includes the chosen offset and a match-confidence score.
4. **Calibration** — Ground control points (GCPs) on the court are mapped in each camera view to real-world coordinates, producing a projection matrix per camera (`calibrate.py` / `calibrate_core.py`). The calibration frames used are the first synchronized pair, so both cameras are calibrated against the exact same instant. The web app's calibration page is semi-automatic: `calibrate_core.auto_detect_gcps` proposes an initial pixel position for all 7 GCPs on each camera before the user looks at anything, and `calibrate.html` opens with those positions already placed as draggable markers rather than an empty canvas to click through. Points the auto-detector is confident about are marked green; points it isn't sure about (most often the net markers, which are harder to detect than the floor corners) are marked orange and flagged "?" until the user drags them into place. A dashed rectangle connecting GCP3→GCP4→GCP6→GCP5 (the real right-side service-box floor rectangle — note the walk order, GCP4→GCP5 is the box's diagonal, not an edge) is drawn on top of the frame as a visual sanity check: if it doesn't line up with the real court lines, the auto-guess got something wrong and needs correcting. See "Semi-automatic calibration" under Usage below for how the guess itself is produced and what its known limitations are.
5. **Triangulation** — Shuttlecock detections from the synchronized videos are combined with the calibration data to reconstruct the shuttlecock's 3D trajectory (X, Y, Z per frame) (`triangulate.py`). The tracker also reports per-camera motion diagnostics, used to flag likely bad detections (see below).
6. **Evaluation** — The 3D trajectory's true floor-landing point is estimated (interpolating to Z = 0 past the apex, not just the last tracked frame), the serve is auto-classified as short or high/long from landing depth, and it's scored against the classified serve's rules — landing placement, net clearance, and peak height (`evaluator.py`).
7. **Visualization** — Results are shown as an interactive dashboard with a trajectory chart and landing map (`app.py`, Flask web app), or printed as a CLI report.

Both the frame-synchronization step and the evaluation step run as background jobs that the browser polls for progress, so large videos don't block the request.

## Repository Structure

| File / Folder | Purpose |
|---|---|
| `app.py` | Flask web app — upload videos, auto-extract & synchronize frames, calibrate the court in-browser, run evaluation jobs, and view results on a dashboard. Also serves an optional sync-verification debug page. |
| `matching.py` | Automatic frame synchronization — detects the shuttlecock per frame in both camera folders, builds a kinetic motion signature (x, y, velocity, acceleration), searches a small start-lag window assuming both cameras began recording together, and pairs the shorter clip 1-to-1 against the longer one from the best-matching start. Runnable standalone as a CLI tool. |
| `run_pipeline.py` | CLI entry point — runs triangulation + evaluation on a fixed pair of video files and prints the report. |
| `calibrate.py` / `calibrate_core.py` | Court calibration: solves for each camera's projection matrix from 7 ground control points via normalized DLT + nonlinear reprojection refinement. `calibrate.py` is the interactive CLI (click the GCPs on a still image, fully manual — it does not use the auto-detection below). `calibrate_core.py` holds the shared math, GCP definitions, and camera metadata used by both the CLI and the web app, plus the web app's semi-automatic calibration: `auto_detect_gcps` (best-effort initial GCP guess — landmark feature-matching first, classical court-line detection as a fallback), `match_against_landmarks` / `save_landmark` / `list_landmarks` (the per-camera landmark history used for that feature-matching). |
| `calibration_landmarks/` | Per-camera history (up to 5 most recent, oldest evicted) of confirmed web-app calibrations — each a saved frame plus its 7 confirmed GCP pixel points. Used by `auto_detect_gcps` to seed future calibrations on the same camera rig via feature matching instead of re-running line detection from scratch. Created automatically; safe to delete to force fresh line-detection next time. |
| `triangulate.py` | `BadmintonTracker3D` — runs YOLO shuttlecock detection on both synchronized video feeds, triangulates 3D positions via DLT, and reports per-camera motion diagnostics. |
| `evaluator.py` | `evaluate_serve_performance` — estimates the true landing point, auto-classifies the serve type, and scores it (landing placement, net clearance, peak height) against the preferred drop box and target line. |
| `train.py` | Trains the YOLOv8 shuttlecock-detection model on `shuttle_dataset/`. |
| `split_dataset.py` | Splits/prepares the raw dataset into train/val sets for training. |
| `visualize_check.py` | Utility for visually sanity-checking detections/calibration on sample frames. |
| `shuttle_dataset/` | Training data for the YOLO shuttlecock detector. |
| `templates/` | HTML templates for the Flask app — `upload.html`, `calibrate.html`, `dashboard.html`, and the optional `sync_verify.html` debug page. |

## Serve Types

The evaluator auto-classifies the serve from where the shuttle actually lands (it does not need to be told which serve was intended):

- **`short_front_corner`** ("Short serve") — landing X between the short service line (1.98 m) and 4.72 m. Rewards a low net clearance (ideal ≤ 0.20 m) and a low peak height (ideal ≤ 1.50 m).
- **`high_back_corner`** ("High serve") — landing X between 4.73 m and the back boundary (6.70 m). Rewards a high net clearance (ideal ≥ 1.00 m) and a high peak height (ideal ≥ 3.50 m).

IN/OUT is judged independently of serve type, against a single preferred drop box (X: 1.98–6.70 m, Y: 0.00–2.59 m); a landing outside that box is OUT and the final score is forced to 0. Score components:

| Component | Weight |
|---|---|
| Landing placement (distance to the target line) | 50 pts |
| Net clearance | 30 pts |
| Peak height | 20 pts |

The full report includes the overall score, serve classification, landing coordinates and status, net clearance, peak height, and a breakdown of each component's points.

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
1. Upload a side-view and back-view video of the serve.
2. Wait for automatic frame extraction and synchronization to finish (progress is polled live).
3. Review the auto-placed ground control points on each synchronized camera frame, drag any that are wrong into place, and save the calibration. See "Semi-automatic calibration" below.
4. Run the evaluation and view the scored report, trajectory chart, and landing map on the dashboard.

If you want to sanity-check the synchronization itself, `/sync-verify/<job_id>` is available as an optional debug page (linked from the calibration page) — it shows the matched frame pairs side by side with a scrub/play control so you can visually confirm both views show the same instant.

#### Semi-automatic calibration

The calibration page (`/calibrate`) no longer starts from an empty canvas. For each camera, `auto_detect_gcps` (in `calibrate_core.py`) tries, in order:

1. **Landmark match** — feature-match (ORB + RANSAC homography) the new frame against up to 5 previously confirmed calibrations for that camera (see `calibration_landmarks/` above). If the camera hasn't moved much since the last confirmed session, this warps the old confirmed points onto the new frame and is the strongest signal available — all 7 points come back marked confident.
2. **Court line detection** — if no landmark match is confident enough (first-ever calibration of a camera, or the rig has clearly moved), classical line detection (white-tape color mask + Hough lines + line intersection) locates the four floor corners (GCP3–GCP6), and the net points (GCP0–GCP2) are approximated by extrapolating the floor lines to the net and looking for the net post.
3. **Generic fallback** — if even that finds nothing, every point still gets *some* on-screen position so there's always a marker to drag, rather than nothing to work with.

Every point is color-coded on the page: **green** means the auto-detector was confident, **orange with a "?"** means it's a low-confidence guess that should be checked. A dashed rectangle over GCP3–GCP4–GCP6–GCP5 is drawn as a visual check — it should sit right on the real floor rectangle of the right-side service box; if it looks rotated or mirrored relative to the real court lines, drag the corners until it matches.

Known limitations, worth knowing before trusting the auto-guess blindly:

- **Net points (GCP0–GCP2) are the least reliable.** They're a geometric extrapolation from the floor rectangle, not a direct detection, and are always marked orange for review.
- **Floor-corner role assignment (which detected corner is "near" vs. "far", "center" vs. "right sideline") is a heuristic** and depends on the camera's exact position/orientation. It can occasionally come out swapped even when the four corners themselves were found correctly — this is what the rectangle overlay is for: if it's mirrored or rotated relative to the real court, the roles got mixed up and need dragging into the correct spots.
- **Extreme camera angles can defeat line detection entirely.** If the floor lines are foreshortened into very short segments in frame (a camera positioned nearly edge-on to a court boundary), detection can fail and fall back to the generic layout — expect to place those points manually in that case.
- None of this has been validated against real match footage with varied lighting/backgrounds beyond initial testing — expect to need to retune thresholds (e.g. the HSV white-tape range or Hough `minLineLength` in `calibrate_core.py`) for your specific gym/lighting setup if detection is consistently off.

Once a calibration is saved, it's stored as a landmark for that camera, so the *next* session for the same rig should start from a landmark match (green, high-confidence) rather than line detection.

### Training the Detector

To train (or retrain) the shuttlecock detection model on your own dataset:

```bash
python split_dataset.py   # prepare train/val splits
python train.py           # trains YOLOv8n on shuttle_dataset/data.yaml
```

## Notes

- Net height is assumed to be 1.55 m in the evaluation logic.
- Synchronization assumes the two cameras were started together (manual sync) and only corrects for a small reaction-time lag (up to 7 frames) at the start of the clips — it is **not** designed to find an arbitrary offset anywhere in a long, independently-started recording. It also assumes a constant offset once found (no clock drift during the clip).
- The tracker flags a camera as "suspect static" if its detected shuttlecock barely moves across the clip — usually a sign the detector locked onto a stationary object (a resting shuttle, the racket, a bright spot on the net) instead of the shuttle in flight. These show up as warnings on the results dashboard rather than failing the evaluation outright. When multiple shuttlecock-like objects are detected in a frame, the tracker now keeps the highest-confidence one rather than whichever the model happened to return first.
- The web app stores original uploads under `uploads_video/sync_job_<id>/`, synchronized videos as `uploads_video/side_sync.mp4` / `back_sync.mp4`, calibration in `web_calibration.npz`, and matched-pair metadata in `matched.json`. Confirmed calibrations are additionally stored per-camera under `calibration_landmarks/` for future auto-calibration (see "Semi-automatic calibration" above); this grows unbounded across different physical setups since it's keyed only by camera name (`side`/`back`), not by location, so delete it if you move the rig somewhere new and want a clean slate instead of a stale landmark match.
- `calibrate.py`, the CLI calibration tool, is still fully manual (click all 7 GCPs on a still image) — the auto-detection and landmark matching described above are web-app-only, in `app.py`'s `/calibrate` route.
- The scoring thresholds in `evaluator.py` are initial, tunable criteria — not official BWF rules — pending calibration against measured high-level/pro serves.
- This project is part of a course research effort (CSS451/454, SIIT, Thammasat University).

## License

No license specified yet.