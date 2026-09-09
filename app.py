import os
import cv2
import numpy as np
import threading
import time
import uuid
import json
import subprocess
import shutil
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file

from calibrate_core import solve_camera, WORLD_GCPS, GCP_LABELS, CAMERA_META
from triangulate import BadmintonTracker3D
from evaluator import evaluate_serve_performance

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


def run_evaluation_job(job_id):
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
            os.path.join(UPLOAD_VIDEO_DIR, "side_sync.mp4"),
            os.path.join(UPLOAD_VIDEO_DIR, "back_sync.mp4"),
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

        # serve_type defaults to "auto": the evaluator classifies short vs.
        # high/long serve itself from where the shuttle actually landed.
        report = evaluate_serve_performance(trajectory)
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


# ---------------------------------------------------------------------------
# Upload -> frame extraction -> automatic synchronization
# ---------------------------------------------------------------------------

upload_jobs = {}
upload_jobs_lock = threading.Lock()


def update_upload_job(job_id, **updates):
    with upload_jobs_lock:
        job = upload_jobs.get(job_id)
        if job is not None:
            job.update(updates)


def get_upload_job(job_id):
    with upload_jobs_lock:
        job = upload_jobs.get(job_id)
        return dict(job) if job is not None else None


def clean_directory(path):
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        target = os.path.join(path, name)
        if os.path.isfile(target):
            try:
                os.remove(target)
            except OSError:
                pass


def extract_video_frames(video_path, output_dir, job_id, label):
    """Extract every video frame as zero-padded JPG files."""
    clean_directory(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {label} video.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
        if not cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            cap.release()
            raise RuntimeError(f"Could not save extracted frame {frame_idx} for {label}.")

        frame_idx += 1

        if frame_idx % 5 == 0 or frame_idx == total:
            update_upload_job(
                job_id,
                status="extracting",
                current=frame_idx,
                total=total,
                message=f"Extracting {label} frames… {frame_idx} / {total}",
            )

    cap.release()

    if frame_idx == 0:
        raise RuntimeError(f"No frames could be extracted from the {label} video.")

    return frame_idx, fps


def create_sync_video(frame_paths, output_path, fps, frame_size):
    """Create a video whose frame N is the synchronized pair's frame N."""
    if not frame_paths:
        raise RuntimeError("No synchronized frames available.")

    width, height = frame_size
    fourcc_candidates = ["mp4v", "avc1"]

    writer = None
    for codec in fourcc_candidates:
        candidate = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None:
        raise RuntimeError(
            "Could not create synchronized MP4 video. "
            "Please check that OpenCV has a usable MP4 codec."
        )

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)
    finally:
        writer.release()


def save_synchronized_outputs(match_result, output_root, job_id, side_fps, back_fps):
    """
    Save the 1-to-1 pairs as:
      matched/back2/pair_XXXX_back.jpg
      matched/side2/pair_XXXX_side.jpg
    and also create:
      uploads_video/back_sync.mp4
      uploads_video/side_sync.mp4
    """
    matches = match_result["matches"]
    if not matches:
        raise RuntimeError(
            "No synchronized frame pairs were found. "
            "The videos may not contain enough usable shuttlecock motion."
        )

    back_dir = os.path.join(output_root, "back2")
    side_dir = os.path.join(output_root, "side2")
    clean_directory(back_dir)
    clean_directory(side_dir)

    back_paths = []
    side_paths = []

    for idx, match in enumerate(matches, start=1):
        src_back = os.path.join(match_result["folder_a"], match["file_a"])
        src_side = os.path.join(match_result["folder_b"], match["file_b"])

        back_frame = cv2.imread(src_back)
        side_frame = cv2.imread(src_side)

        if back_frame is None or side_frame is None:
            continue

        back_ext = ".jpg"
        side_ext = ".jpg"

        back_dst = os.path.join(back_dir, f"pair_{idx:04d}_back{back_ext}")
        side_dst = os.path.join(side_dir, f"pair_{idx:04d}_side{side_ext}")

        cv2.imwrite(back_dst, back_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(side_dst, side_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        back_paths.append(back_dst)
        side_paths.append(side_dst)

    if not back_paths or not side_paths:
        raise RuntimeError("Matched pairs were found, but no frames could be saved.")

    # Use the first synchronized frames for calibration.
    cv2.imwrite(
        os.path.join(STATIC_DIR, "back_frame.jpg"),
        cv2.imread(back_paths[0]),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    cv2.imwrite(
        os.path.join(STATIC_DIR, "side_frame.jpg"),
        cv2.imread(side_paths[0]),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    # Create synchronized videos for the later triangulation stage.
    back0 = cv2.imread(back_paths[0])
    side0 = cv2.imread(side_paths[0])

    # Every pair index represents the same instant, so both output videos
    # must use the same playback FPS.
    sync_fps = min(side_fps, back_fps)
    if sync_fps <= 0:
        sync_fps = 30.0

    create_sync_video(
        back_paths,
        os.path.join(UPLOAD_VIDEO_DIR, "back_sync.mp4"),
        sync_fps,
        (back0.shape[1], back0.shape[0]),
    )
    create_sync_video(
        side_paths,
        os.path.join(UPLOAD_VIDEO_DIR, "side_sync.mp4"),
        sync_fps,
        (side0.shape[1], side0.shape[0]),
    )

    return len(back_paths), sync_fps


def get_ffmpeg_path():
    """Return the FFmpeg executable path, or raise a clear error."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg was not found in PATH. Install FFmpeg and restart Flask. "
            "You can verify it with: ffmpeg -version"
        )
    return ffmpeg


def needs_browser_transcode(input_path):
    """
    Return False only when we can positively confirm the source is already
    an MP4 container with H.264 video + yuv420p pixel format — the exact
    combination Chrome/Safari play natively, and the same combination
    convert_to_browser_mp4() would produce anyway. Everything else
    (HEVC/H.265 from iPhone, MOV containers, unusual pixel formats, or any
    ffprobe failure) defaults to True so we never risk serving an
    unplayable "preview".
    """
    if os.path.splitext(input_path)[1].lower() != ".mp4":
        # A .mov extension can still contain H.264, but many iPhone .mov
        # files are HEVC — and even H.264-in-.mov sometimes has moov-atom
        # placement Chrome dislikes. Only trust the container when it's
        # already .mp4.
        return True

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return True

    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,pix_fmt",
                "-of", "csv=p=0",
                input_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )

        if result.returncode != 0:
            return True

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 2:
            return True

        codec_name, pix_fmt = parts[0], parts[1]
        return not (codec_name == "h264" and pix_fmt == "yuv420p")

    except Exception:
        return True


def convert_to_browser_mp4(input_path, output_path, job_id, label):
    """
    Produce a browser-compatible H.264/yuv420p/AAC MP4 preview copy.

    The original upload is never modified. If the source already matches
    that format (already H.264 yuv420p inside an .mp4 container), we just
    copy the file instead of paying for a pointless FFmpeg re-encode.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not needs_browser_transcode(input_path):
        update_upload_job(
            job_id,
            status="converting",
            current=0,
            total=0,
            message=f"{label.capitalize()} video is already browser-compatible — copying…",
        )
        shutil.copy2(input_path, output_path)
        return output_path

    ffmpeg = get_ffmpeg_path()

    update_upload_job(
        job_id,
        status="converting",
        current=0,
        total=0,
        message=f"Converting {label} video to browser-compatible MP4…",
    )

    cmd = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        # Fast browser-preview encode only.
        # The original uploaded video is still used for CV processing.
        "-preset", "ultrafast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    # Capture FFmpeg output so errors can be surfaced in Flask instead of
    # printing a large log to the browser.
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        error_tail = result.stderr[-2500:].strip()
        raise RuntimeError(
            f"Could not convert {label} video to browser-compatible MP4.\n"
            f"FFmpeg error:\n{error_tail}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(
            f"FFmpeg finished but did not create the {label} preview MP4."
        )

    return output_path



def get_cached_preview_path(preview_job_id, camera):
    """
    Return the path to a browser-preview MP4 already produced by an earlier
    /preview-upload call for this exact file, if one finished successfully.
    """
    if not preview_job_id:
        return None

    cached_path = os.path.join(
        UPLOAD_VIDEO_DIR, f"preview_{preview_job_id}", f"{camera}_preview.mp4"
    )

    if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
        return cached_path

    return None


def ensure_browser_preview(input_path, output_path, job_id, label, preview_job_id):
    """
    Produce the browser-compatible preview MP4 for "side"/"back".

    The moment a video is attached, the page already sent it to
    /preview-upload and converted it once. Re-running FFmpeg on the same
    source file here would just duplicate that work and slow down
    "Upload & Extract Frames" for no benefit, so we reuse that output when
    it's available and only fall back to converting from scratch if it
    isn't (e.g. the live conversion failed or was skipped).
    """
    cached = get_cached_preview_path(preview_job_id, label)

    if cached:
        update_upload_job(
            job_id,
            status="converting",
            message=f"Reusing already-converted {label} preview…",
        )
        shutil.copy2(cached, output_path)
        return output_path

    return convert_to_browser_mp4(input_path, output_path, job_id, label)


def run_upload_job(job_id):
    try:
        job = get_upload_job(job_id)
        if not job:
            return

        side_path = job["side_path"]
        back_path = job["back_path"]
        work_dir = job["work_dir"]
        side_preview_path = job["side_preview_path"]
        back_preview_path = job["back_preview_path"]
        side_preview_job_id = job.get("side_preview_job_id")
        back_preview_job_id = job.get("back_preview_job_id")

        side_frames = os.path.join(work_dir, "side_frames")
        back_frames = os.path.join(work_dir, "back_frames")
        matched_root = os.path.join(work_dir, "matched")

        update_upload_job(
            job_id,
            status="converting",
            current=0,
            total=0,
            message="Preparing browser-compatible video previews…",
        )

        # Convert only the copies used by the browser. Frame extraction and
        # synchronization continue to use the original uploaded files.
        # If the page already converted these clips live (as soon as they
        # were attached), reuse that output instead of re-encoding.
        ensure_browser_preview(
            side_path, side_preview_path, job_id, "side", side_preview_job_id
        )
        ensure_browser_preview(
            back_path, back_preview_path, job_id, "back", back_preview_job_id
        )

        side_count, side_fps = extract_video_frames(
            side_path, side_frames, job_id, "side"
        )
        back_count, back_fps = extract_video_frames(
            back_path, back_frames, job_id, "back"
        )

        update_upload_job(
            job_id,
            status="matching",
            current=0,
            total=max(side_count, back_count),
            message="Synchronizing side and back camera frames…",
        )

        # matching.py remains the single source of truth for synchronization.
        from matching import match_badminton

        # Folder A = back, Folder B = side, matching.py returns
        # file_a/file_b pairs with a strict 1-to-1 frame offset.
        result = match_badminton(back_frames, side_frames)

        os.makedirs(matched_root, exist_ok=True)
        with open(os.path.join(matched_root, "matched.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        pair_count, sync_fps = save_synchronized_outputs(
            result,
            matched_root,
            job_id,
            side_fps,
            back_fps,
        )

        # Keep the matched metadata at the project-level path too, for easy debugging.
        with open(os.path.join(BASE, "matched.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # matching.py uses a constant frame offset. Because every synchronized
        # pair is kept in its original frame order, the first pair tells us
        # the same offset without running the detector a second time.
        try:
            first_match = result["matches"][0]
            back_idx = int(os.path.splitext(first_match["file_a"])[0].split("_")[-1])
            side_idx = int(os.path.splitext(first_match["file_b"])[0].split("_")[-1])
            offset = back_idx - side_idx
        except Exception:
            offset = None

        # Triangulation should use the synchronized videos, not the raw uploads.
        update_upload_job(
            job_id,
            status="done",
            done=True,
            current=pair_count,
            total=pair_count,
            message=f"Synchronization complete — {pair_count} frame pairs ready.",
            offset=offset,
            pair_count=pair_count,
            sync_fps=sync_fps,
            side_frames=side_count,
            back_frames=back_count,
            offset_method=result.get("offset_method"),
            offset_confidence=result.get("offset_confidence"),
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        update_upload_job(
            job_id,
            status="error",
            done=False,
            error=str(exc),
            message="Upload processing failed.",
        )



@app.route("/preview-upload", methods=["POST"])
def preview_upload():
    """
    Upload one selected video and immediately convert it to a
    browser-compatible H.264 MP4 preview.

    This endpoint is intentionally separate from /upload:
    /upload still performs the full extraction + synchronization workflow.
    """
    camera = request.form.get("camera", "").strip().lower()
    video = request.files.get("video")

    if camera not in {"side", "back"}:
        return jsonify({"ok": False, "error": "Camera must be 'side' or 'back'."}), 400

    if video is None or not video.filename:
        return jsonify({"ok": False, "error": "No video was attached."}), 400

    preview_job_id = uuid.uuid4().hex
    work_dir = os.path.join(UPLOAD_VIDEO_DIR, f"preview_{preview_job_id}")
    os.makedirs(work_dir, exist_ok=True)

    ext = os.path.splitext(video.filename)[1].lower() or ".mov"
    original_path = os.path.join(work_dir, f"{camera}_original{ext}")
    preview_path = os.path.join(work_dir, f"{camera}_preview.mp4")

    video.save(original_path)

    # The job must exist in upload_jobs BEFORE update_upload_job is called on
    # it — update_upload_job only updates an existing entry, it never creates
    # one. Without this insert, every subsequent update_upload_job() call for
    # this preview_job_id silently does nothing, and
    # /preview-upload-status/<preview_job_id> would 404 forever.
    job = {
        "job_id": preview_job_id,
        "status": "converting",
        "current": 0,
        "total": 0,
        "message": f"Converting {camera} video to browser-compatible MP4…",
        "camera": camera,
        "filename": video.filename,
        "original_path": original_path,
        "preview_path": preview_path,
    }
    with upload_jobs_lock:
        upload_jobs[preview_job_id] = job

    def convert_preview_job():
        try:
            convert_to_browser_mp4(
                original_path,
                preview_path,
                preview_job_id,
                camera,
            )

            update_upload_job(
                preview_job_id,
                status="done",
                current=1,
                total=1,
                message="Browser preview is ready.",
                preview_url=url_for(
                    "browser_preview_file",
                    preview_job_id=preview_job_id,
                    camera=camera,
                ),
            )
        except Exception as exc:
            update_upload_job(
                preview_job_id,
                status="error",
                message=str(exc),
            )

    threading.Thread(
        target=convert_preview_job,
        daemon=True,
    ).start()

    return jsonify({
        "ok": True,
        "preview_job_id": preview_job_id,
        "camera": camera,
        "filename": video.filename,
    })


@app.route("/preview-upload-status/<preview_job_id>")
def preview_upload_status(preview_job_id):
    job = get_upload_job(preview_job_id)
    if job is None:
        return jsonify({
            "ok": False,
            "error": "Preview job not found.",
        }), 404

    return jsonify({
        "ok": True,
        "status": job.get("status"),
        "message": job.get("message", ""),
        "preview_url": job.get("preview_url"),
        "camera": job.get("camera"),
        "filename": job.get("filename"),
    })


@app.route("/browser-preview-file/<preview_job_id>/<camera>")
def browser_preview_file(preview_job_id, camera):
    job = get_upload_job(preview_job_id)

    if job is None:
        return "Preview job not found.", 404

    if camera not in {"side", "back"}:
        return "Invalid camera.", 400

    if job.get("camera") != camera:
        return "Camera mismatch.", 400

    path = job.get("preview_path")

    if not path or not os.path.exists(path):
        return "Browser preview is not ready.", 404

    response = send_file(
        path,
        mimetype="video/mp4",
        conditional=True,
        max_age=0,
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


@app.route("/upload", methods=["POST"])
def upload():
    side_video = request.files.get("side_video")
    back_video = request.files.get("back_video")

    # These reference the /preview-upload conversions already done while
    # the videos were being attached (see setVideoPreview in upload.html),
    # so run_upload_job can reuse that output instead of re-encoding.
    side_preview_job_id = request.form.get("side_preview_job_id") or None
    back_preview_job_id = request.form.get("back_preview_job_id") or None

    if not side_video or not back_video:
        return jsonify({"error": "Please upload both side and back camera videos."}), 400

    if not side_video.filename or not back_video.filename:
        return jsonify({"error": "Please upload both side and back camera videos."}), 400

    job_id = uuid.uuid4().hex
    work_dir = os.path.join(UPLOAD_VIDEO_DIR, f"sync_job_{job_id}")
    os.makedirs(work_dir, exist_ok=True)

    # Preserve the uploaded files in their original container/extension.
    # The browser preview is a separate H.264 MP4 generated by FFmpeg.
    side_ext = os.path.splitext(side_video.filename)[1].lower() or ".mov"
    back_ext = os.path.splitext(back_video.filename)[1].lower() or ".mov"

    side_path = os.path.join(work_dir, f"side_original{side_ext}")
    back_path = os.path.join(work_dir, f"back_original{back_ext}")

    side_preview_path = os.path.join(work_dir, "side_preview.mp4")
    back_preview_path = os.path.join(work_dir, "back_preview.mp4")

    side_video.save(side_path)
    back_video.save(back_path)

    job = {
        "job_id": job_id,
        "status": "starting",
        "current": 0,
        "total": 0,
        "done": False,
        "message": "Starting automatic extraction and synchronization…",
        "error": None,
        "side_path": side_path,
        "back_path": back_path,
        "side_preview_path": side_preview_path,
        "back_preview_path": back_preview_path,
        "side_preview_job_id": side_preview_job_id,
        "back_preview_job_id": back_preview_job_id,
        "work_dir": work_dir,
        "side_filename": side_video.filename,
        "back_filename": back_video.filename,
        "created_at": time.time(),
    }

    with upload_jobs_lock:
        upload_jobs[job_id] = job

    worker = threading.Thread(
        target=run_upload_job,
        args=(job_id,),
        daemon=True,
    )
    worker.start()

    return jsonify({
        "job_id": job_id,
        "status_url": url_for("upload_status", job_id=job_id),
    })


@app.route("/upload_status/<job_id>")
def upload_status(job_id):
    job = get_upload_job(job_id)
    if job is None:
        return jsonify({"error": "Upload job not found."}), 404

    return jsonify({
        "job_id": job["job_id"],
        "status": job["status"],
        "current": job.get("current", 0),
        "total": job.get("total", 0),
        "done": job.get("done", False),
        "message": job.get("message", ""),
        "error": job.get("error"),
        "pair_count": job.get("pair_count", 0),
        "offset": job.get("offset"),
        "side_frames": job.get("side_frames", 0),
        "back_frames": job.get("back_frames", 0),
        "sync_fps": job.get("sync_fps", 30.0),
        "offset_method": job.get("offset_method"),
        "offset_confidence": job.get("offset_confidence"),
        "side_preview_url": (
            url_for("browser_preview", job_id=job_id, camera="side")
            if os.path.exists(job.get("side_preview_path", "")) else None
        ),
        "back_preview_url": (
            url_for("browser_preview", job_id=job_id, camera="back")
            if os.path.exists(job.get("back_preview_path", "")) else None
        ),
        "verify_url": (
            url_for("sync_verify_page", job_id=job_id)
            if job.get("done") else None
        ),
    })


@app.route("/browser-preview/<job_id>/<camera>")
def browser_preview(job_id, camera):
    """Serve the FFmpeg-generated H.264 MP4 preview for the upload page."""
    job = get_upload_job(job_id)
    if job is None:
        return "Upload job not found.", 404

    if camera == "side":
        path = job.get("side_preview_path")
    elif camera == "back":
        path = job.get("back_preview_path")
    else:
        return "Invalid camera.", 400

    if not path or not os.path.exists(path):
        return "Browser preview is not ready.", 404

    response = send_file(
        path,
        mimetype="video/mp4",
        conditional=True,
        max_age=0,
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


@app.route("/sync-verify/<job_id>")
def sync_verify_page(job_id):
    job = get_upload_job(job_id)
    if job is None:
        return "Synchronization job not found.", 404
    if not job.get("done"):
        return redirect(url_for("index"))

    pair_count = int(job.get("pair_count", 0))
    if pair_count <= 0:
        return "No synchronized frame pairs are available.", 400

    return render_template(
        "sync_verify.html",
        job_id=job_id,
        pair_count=pair_count,
        offset=job.get("offset"),
        fps=job.get("sync_fps", 30.0),
    )


@app.route("/sync-frame/<job_id>/<int:pair_idx>/<camera>")
def sync_frame(job_id, pair_idx, camera):
    job = get_upload_job(job_id)
    if job is None or not job.get("done"):
        return "Synchronization job not found.", 404

    if camera not in ("side", "back"):
        return "Invalid camera.", 400

    pair_count = int(job.get("pair_count", 0))
    if pair_idx < 0 or pair_idx >= pair_count:
        return "Frame out of range.", 404

    matched_root = os.path.join(job["work_dir"], "matched")
    if camera == "side":
        frame_path = os.path.join(
            matched_root, "side2", f"pair_{pair_idx + 1:04d}_side.jpg"
        )
    else:
        frame_path = os.path.join(
            matched_root, "back2", f"pair_{pair_idx + 1:04d}_back.jpg"
        )

    if not os.path.exists(frame_path):
        return "Synchronized frame not found.", 404

    from flask import send_file
    response = send_file(frame_path, mimetype="image/jpeg", max_age=0)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/sync-confirm/<job_id>", methods=["POST"])
def sync_confirm(job_id):
    job = get_upload_job(job_id)
    if job is None:
        return jsonify({"error": "Synchronization job not found."}), 404

    if not job.get("done") or int(job.get("pair_count", 0)) <= 0:
        return jsonify({"error": "Synchronization is not ready."}), 400

    return jsonify({"ok": True, "redirect_url": url_for("calibrate_page")})


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
        args=(job_id,),
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