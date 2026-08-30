import os
import cv2
import numpy as np
import threading
import time
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify

from calibrate_core import solve_camera, WORLD_GCPS, GCP_LABELS, CAMERA_META
from triangulate import BadmintonTracker3D
from evalautor import evaluate_serve_performance

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_VIDEO_DIR = os.path.join(BASE, "uploads_video")
STATIC_DIR = os.path.join(BASE, "static", "uploads")
MODEL_PATH = os.path.join(BASE, "models", "best.pt")

os.makedirs(UPLOAD_VIDEO_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# In-memory evaluation jobs. This is enough for the local Flask application
# and lets the browser poll progress while triangulate.py runs in a worker.
evaluation_jobs = {}
evaluation_jobs_lock = threading.Lock()


def update_evaluation_job(job_id, **updates):
    with evaluation_jobs_lock:
        job = evaluation_jobs.get(job_id)
        if job is not None:
            job.update(updates)


def get_evaluation_job(job_id):
    with evaluation_jobs_lock:
        job = evaluation_jobs.get(job_id)
        return dict(job) if job is not None else None


def run_evaluation_job(job_id, serve_type):
    try:
        update_evaluation_job(job_id, status="loading", message="Loading YOLO model…")

        tracker = BadmintonTracker3D(
            model_path=MODEL_PATH,
            calib_file=os.path.join(BASE, "web_calibration.npz")
        )

        def progress_callback(current, total, fps, detected):
            update_evaluation_job(
                job_id,
                status="processing",
                current=int(current),
                total=int(total),
                fps=float(fps),
                detected=int(detected),
                message=f"Processing frame {current} / {total}",
            )

        trajectory = tracker.process_videos(
            os.path.join(UPLOAD_VIDEO_DIR, "side.mp4"),
            os.path.join(UPLOAD_VIDEO_DIR, "back.mp4"),
            progress_callback=progress_callback,
        )

        if len(trajectory) == 0:
            update_evaluation_job(
                job_id,
                status="error",
                error="No trajectory points detected.",
            )
            return

        update_evaluation_job(
            job_id,
            status="evaluating",
            message="Calculating serve performance…",
            current=get_evaluation_job(job_id).get("total", 0),
        )

        report = evaluate_serve_performance(trajectory, serve_type=serve_type)
        save_trajectory_chart(
            trajectory,
            os.path.join(STATIC_DIR, "trajectory.png")
        )

        update_evaluation_job(
            job_id,
            status="done",
            done=True,
            message="Evaluation complete.",
            result_ready=True,
            trajectory_points=len(trajectory),
            report=report,
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        update_evaluation_job(
            job_id,
            status="error",
            error=str(exc),
        )

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    side_video = request.files["side_video"]
    back_video = request.files["back_video"]

    side_path = os.path.join(UPLOAD_VIDEO_DIR, "side.mp4")
    back_path = os.path.join(UPLOAD_VIDEO_DIR, "back.mp4")
    side_video.save(side_path)
    back_video.save(back_path)

    for video_path, out_name in [(side_path, "side_frame.jpg"), (back_path, "back_frame.jpg")]:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return f"Could not read {video_path}", 400
        cv2.imwrite(os.path.join(STATIC_DIR, out_name), frame)

    return redirect(url_for("calibrate_page"))

@app.route("/calibrate")
def calibrate_page():
    side_img = cv2.imread(os.path.join(STATIC_DIR, "side_frame.jpg"))
    back_img = cv2.imread(os.path.join(STATIC_DIR, "back_frame.jpg"))
    side_h, side_w = side_img.shape[:2]
    back_h, back_w = back_img.shape[:2]
    return render_template("calibrate.html",
                            side_w=side_w, side_h=side_h,
                            back_w=back_w, back_h=back_h,
                            num_points=len(WORLD_GCPS),
                            gcp_labels=GCP_LABELS,
                            side_meta=CAMERA_META["side"],
                            back_meta=CAMERA_META["back"])

@app.route("/save_calibration", methods=["POST"])
def save_calibration():
    data = request.get_json()
    try:
        P_side = solve_camera(data["side_points"], data["side_w"], data["side_h"])
        P_back = solve_camera(data["back_points"], data["back_w"], data["back_h"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    np.savez(os.path.join(BASE, "web_calibration.npz"), P_side=P_side, P_back=P_back)
    return jsonify({"status": "ok"})

@app.route("/run_evaluation", methods=["POST"])
def run_evaluation():
    serve_type = request.form.get("serve_type", "short_front_corner")

    if not os.path.exists(os.path.join(BASE, "web_calibration.npz")):
        return jsonify({"error": "Calibration has not been saved yet."}), 400

    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "status": "starting",
        "current": 0,
        "total": 0,
        "fps": 0.0,
        "detected": 0,
        "done": False,
        "message": "Starting evaluation…",
        "created_at": time.time(),
    }

    with evaluation_jobs_lock:
        evaluation_jobs[job_id] = job

    worker = threading.Thread(
        target=run_evaluation_job,
        args=(job_id, serve_type),
        daemon=True,
    )
    worker.start()

    return jsonify({
        "job_id": job_id,
        "status_url": url_for("evaluation_status", job_id=job_id),
        "result_url": url_for("evaluation_result", job_id=job_id),
    })


@app.route("/evaluation_status/<job_id>")
def evaluation_status(job_id):
    job = get_evaluation_job(job_id)
    if job is None:
        return jsonify({"error": "Evaluation job not found."}), 404

    # Do not send the full report on every polling request.
    public_job = {
        "job_id": job["job_id"],
        "status": job["status"],
        "current": job.get("current", 0),
        "total": job.get("total", 0),
        "fps": job.get("fps", 0.0),
        "detected": job.get("detected", 0),
        "done": job.get("done", False),
        "message": job.get("message", ""),
        "error": job.get("error"),
        "result_url": url_for("evaluation_result", job_id=job_id),
    }
    return jsonify(public_job)


@app.route("/evaluation_result/<job_id>")
def evaluation_result(job_id):
    job = get_evaluation_job(job_id)
    if job is None:
        return "Evaluation job not found.", 404

    if job.get("status") == "error":
        return render_template("dashboard.html", error=job.get("error"))

    if not job.get("done"):
        return redirect(url_for("calibrate_page"))

    return render_template(
        "dashboard.html",
        report=job["report"],
        num_points=job.get("trajectory_points", 0),
    )

def save_trajectory_chart(trajectory, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = [p["X"] for p in trajectory]
    Z = [p["Z"] for p in trajectory]

    plt.figure(figsize=(8, 4))
    plt.plot(X, Z, marker="o")
    plt.xlabel("X - Depth from net (m)")
    plt.ylabel("Z - Height (m)")
    plt.title("Serve Trajectory (Side Profile)")
    plt.axhline(1.55, color="red", linestyle="--", label="Net height")
    plt.ylim(bottom=-0.1, top=4.0)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    app.run(debug=True)