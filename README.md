# Badminton Serve Evaluation

A computer-vision system that scores badminton serves for legality and accuracy. It uses a YOLO-based shuttlecock detector across two synchronized cameras (side + back view), triangulates the shuttlecock into 3D court coordinates, and evaluates the resulting trajectory against target-specific serve rules — net clearance, arc height, and landing accuracy.

## How It Works

1. **Detection** — A custom-trained YOLOv8 model (`shuttlecock_v1`) detects the shuttlecock in each frame of a side-view and a back-view video.
2. **Calibration** — Ground control points (GCPs) on the court are mapped in each camera view to real-world coordinates, producing a projection matrix per camera (`calibrate.py` / `calibrate_core.py`).
3. **Triangulation** — Detections from both views are combined with the calibration data to reconstruct the shuttlecock's 3D trajectory (X, Y, Z per frame) (`triangulate.py`).
4. **Evaluation** — The 3D trajectory is scored against the rules for the selected serve type: net clearance, peak height, and landing accuracy relative to the target zone (`evalautor.py`).
5. **Visualization** — Results can be viewed as a printed report (CLI) or as an interactive dashboard with a trajectory chart (`app.py`, Flask web app).

## Repository Structure

| File / Folder | Purpose |
|---|---|
| `app.py` | Flask web app — upload videos, calibrate the court in-browser, run evaluation jobs, and view results on a dashboard. |
| `run_pipeline.py` | CLI entry point — runs triangulation + evaluation on a fixed pair of video files and prints the report. |
| `calibrate.py` / `calibrate_core.py` | Court calibration: solves for each camera's projection matrix from ground control points. |
| `triangulate.py` | `BadmintonTracker3D` — runs shuttlecock detection on both video feeds and triangulates 3D positions. |
| `evalautor.py` | `evaluate_serve_performance` — scores a 3D trajectory against serve-specific rules. |
| `train.py` | Trains the YOLOv8 shuttlecock-detection model on `shuttle_dataset/`. |
| `split_dataset.py` | Splits/prepares the raw dataset into train/val sets for training. |
| `visualize_check.py` | Utility for visually sanity-checking detections/calibration on sample frames. |
| `shuttle_dataset/` | Training data for the YOLO shuttlecock detector. |
| `templates/` | HTML templates for the Flask app (upload, calibration, dashboard pages). |

## Serve Types

The evaluator currently supports two serve targets:

- **`short_front_corner`** — Target: front service line corner (X: 1.98 m, Y: 2.59 m). Penalizes serves that clear the net too high (should graze it) or hit the net, plus landing distance from the target.
- **`high_back_corner`** — Target: deep baseline corner (X: 6.70 m, Y: 2.59 m). Penalizes a flat arc (peak height should exceed 3.5 m), plus landing distance from the target.

Each report includes: overall score (0–100), peak height, net clearance, landing coordinates, and a breakdown of deductions.

## Requirements

- Python 3.8+
- `opencv-python`
- `numpy`
- `ultralytics`
- `scipy`
- `flask`
- `matplotlib`
- `torch`

Install with:

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
2. Click ground control points on each video frame to calibrate the court.
3. Run the evaluation and view the scored report with a trajectory chart.

### Option 2: Command Line

Place your model weights at `models/best.pt` and your calibration file at `court_calibration.npz`, then edit the video paths in `run_pipeline.py` as needed:

```bash
python run_pipeline.py
```

This prints the detected 3D trajectory points followed by the serve evaluation report.

### Training the Detector

To train (or retrain) the shuttlecock detection model on your own dataset:

```bash
python split_dataset.py   # prepare train/val splits
python train.py           # trains YOLOv8n on shuttle_dataset/data.yaml
```

## Notes

- Net height is assumed to be 1.55 m in the evaluation logic.
- The web app stores calibration in `web_calibration.npz` and uploaded videos in `uploads_video/`.
- This project is part of a course research effort (CSS451/454, SIIT, Thammasat University).

## License

No license specified yet.
