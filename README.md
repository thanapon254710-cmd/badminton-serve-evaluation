# Badminton Serve Evaluation

A computer-vision system that scores badminton serves for legality and accuracy. It uses a YOLO-based shuttlecock detector across two synchronized cameras (side + back view), automatically aligns the two camera feeds frame-by-frame, triangulates the shuttlecock into 3D court coordinates, and evaluates the resulting trajectory against BWF-style serve rules — landing placement, net clearance, and peak height.

## How It Works

1. **Upload** — The user uploads a side-view and a back-view video of the same serve through the web app.
2. **Frame extraction** — Every frame of both videos is extracted to disk as individual JPGs.
3. **Automatic synchronization** — `matching.py` detects the shuttlecock in each frame of both views and builds a per-camera motion signature (x, y, velocity magnitude, acceleration magnitude). It assumes both cameras were started together (e.g. a manual "1, 2, 3, click" start) and only searches a small window of starting lags — up to 7 frames — right at the beginning of the clips, rather than sliding across the whole video. Whichever camera's clip is shorter becomes the time base: every one of its frames is paired 1-to-1 with a consecutive frame of the longer clip at the best-matching start offset, and any leftover frames at the tail of the longer clip are dropped. The app saves the result as new synchronized videos (`side_sync.mp4` / `back_sync.mp4`) plus a `matched.json` record that also includes the chosen offset and a match-confidence score.
4. **Calibration** — Ground control points (GCPs) on the court are mapped in each camera view to real-world coordinates, producing a projection matrix per camera (`calibrate.py` / `calibrate_core.py`). The calibration frames used are the first synchronized pair, so both cameras are calibrated against the exact same instant.
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
| `calibrate.py` / `calibrate_core.py` | Court calibration: solves for each camera's projection matrix from 7 ground control points via normalized DLT + nonlinear reprojection refinement. `calibrate.py` is the interactive CLI (click the GCPs on a still image); `calibrate_core.py` holds the shared math, GCP definitions, and camera metadata used by both the CLI and the web app. |
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

### Option 1: Web App

```bash
python app.py
```

Then open `http://localhost:5000` in a browser to:
1. Upload a side-view and back-view video of the serve.
2. Wait for automatic frame extraction and synchronization to finish (progress is polled live).
3. Click the ground control points on each synchronized camera frame to calibrate the court.
4. Run the evaluation and view the scored report, trajectory chart, and landing map on the dashboard.

If you want to sanity-check the synchronization itself, `/sync-verify/<job_id>` is available as an optional debug page (linked from the calibration page) — it shows the matched frame pairs side by side with a scrub/play control so you can visually confirm both views show the same instant.

### Option 2: Command Line

**Full pipeline** — place your model weights at `models/best.pt` and your calibration file at `court_calibration.npz`, then edit the video paths in `run_pipeline.py` as needed:

```bash
python run_pipeline.py
```

This prints the detected 3D trajectory points followed by the serve evaluation report.

**Synchronization only** — to align two folders of extracted frames independently of the web app:

```bash
python matching.py <folder_a> <folder_b> --out matched.json
```

`folder_a` and `folder_b` should each contain the extracted frames of one camera. The output JSON lists the 1-to-1 matched filename pairs and is used to build the synchronized videos consumed by triangulation.

**Calibration only** — to (re)calibrate interactively from still images:

```bash
python calibrate.py
```

Update `SIDE_IMAGE` / `BACK_IMAGE` at the top of the script to point at your calibration frames first; it writes `court_calibration.npz`.

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
- The web app stores original uploads under `uploads_video/sync_job_<id>/`, synchronized videos as `uploads_video/side_sync.mp4` / `back_sync.mp4`, calibration in `web_calibration.npz`, and matched-pair metadata in `matched.json`.
- The scoring thresholds in `evaluator.py` are initial, tunable criteria — not official BWF rules — pending calibration against measured high-level/pro serves.
- This project is part of a course research effort (CSS451/454, SIIT, Thammasat University).

## License

No license specified yet.