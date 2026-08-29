import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify

from calibrate_core import solve_camera, WORLD_GCPs
from triangulate import BadmintonTracker3D
from evalautor import evaluate_serve_performance

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_VIDEO_DIR = os.path.join(BASE, "uploads_video")
STATIC_DIR = os.path.join(BASE, "static", "uploads")
MODEL_PATH = os.path.join(BASE, "models", "best.pt")

os.makedirs(UPLOAD_VIDEO_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

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
                            num_points=len(WORLD_GCPs))

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

    tracker = BadmintonTracker3D(
        model_path=MODEL_PATH,
        calib_file=os.path.join(BASE, "web_calibration.npz")
    )
    trajectory = tracker.process_videos(
        os.path.join(UPLOAD_VIDEO_DIR, "side.mp4"),
        os.path.join(UPLOAD_VIDEO_DIR, "back.mp4")
    )

    if len(trajectory) == 0:
        return render_template("dashboard.html", error="No trajectory points detected.")

    report = evaluate_serve_performance(trajectory, serve_type=serve_type)
    save_trajectory_chart(trajectory, os.path.join(STATIC_DIR, "trajectory.png"))

    return render_template("dashboard.html", report=report, num_points=len(trajectory))

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
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    app.run(debug=True)