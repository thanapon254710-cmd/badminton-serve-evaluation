import os
import shutil
import subprocess
import cv2
import numpy as np
import threading
import time
import uuid
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file

from calibrate_core import (
    solve_camera, WORLD_GCPS, GCP_LABELS, CAMERA_META, auto_detect_gcps, save_landmark,
    SERVICE_SIDES, DEFAULT_SERVICE_SIDE, normalize_service_side, gcp_labels_for, landmark_key,
)
from triangulate import BadmintonTracker3D
import evaluator
from evaluator import evaluate_serve_performance
import multi_serve

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


def build_trajectory_warnings(diagnostics):
    """
    Turns the tracker's per-camera motion diagnostics into user-facing
    warnings. A camera whose detected "shuttlecock" barely moves across
    the whole clip is a strong sign the detector locked onto a
    stationary object (a resting shuttle left on the court, the racket,
    a bright spot on the net tape, etc.) instead of the shuttle actually
    in flight -- so the resulting 3D trajectory and score shouldn't be
    trusted at face value even though the pipeline still produced a
    number.
    """
    warnings = []
    labels = {"side": "Side camera", "back": "Back camera"}
    for cam, label in labels.items():
        if diagnostics.get(f"{cam}_suspect_static"):
            frac = diagnostics.get(f"{cam}_motion_fraction", 0.0)
            warnings.append(
                f"{label}: the tracked shuttlecock barely moved across the clip "
                f"(only {frac:.1%} of the frame). This usually means the detector "
                f"locked onto a stationary object rather than the shuttle actually "
                f"in flight. Re-check the source video for this camera -- if the "
                f"shuttle isn't visibly moving in it, re-record that take before "
                f"trusting this result."
            )
    return warnings


CALIBRATION_FILE = os.path.join(BASE, "web_calibration.npz")

SERVICE_SIDE_LABELS = {"right": "Right service area", "left": "Left service area"}


def read_saved_service_side():
    """
    Service side the saved calibration was solved for ("right" / "left").

    The side is stored inside web_calibration.npz next to the projection
    matrices, because the matrices are only meaningful together with the
    world coordinates they were solved against. A calibration saved before
    the left side existed has no such key and is a right-side calibration.
    """
    try:
        with np.load(CALIBRATION_FILE) as calib:
            if "service_side" in calib.files:
                return normalize_service_side(str(calib["service_side"]))
    except (OSError, ValueError):
        pass
    return DEFAULT_SERVICE_SIDE


# A serve whose nearest tracked point is further than this from the net gets a warning.
NET_GAP_WARNING_M = 0.75


def build_multi_serve_report(trajectory, diagnostics, service_side, warnings, landing_target=None):
    """
    Split the tracked points into serves, evaluate each one, and return
    (report, serves). `report` is the first serve's report plus:
        shots          list of per-serve reports (the dashboard pager uses it)
        serve_count    number of serves found
        average_score  mean of the serves' final scores
    Returns (None, serves, discarded) when no serve could be evaluated. Notes
    about unreliable serves are appended to `warnings`.
    """
    fps = diagnostics.get("fps") or 30.0
    serves, discarded = multi_serve.split_serves(trajectory, fps)

    shots, kept = [], []
    for serve in serves:
        rep = evaluate_serve_performance(
            serve["points"], service_side=service_side, landing_target=landing_target
        )
        if "error" in rep:
            continue
        rep["serve_index"] = len(shots) + 1
        rep["start_s"] = serve["start_s"]
        rep["end_s"] = serve["end_s"]
        rep["track_complete"] = serve["complete"]
        # How close to the net did tracking get? Peak height and net clearance are
        # measured from the tracked points, so a track that only starts well past
        # the net (the shuttle was not detected right after the hit) is less reliable.
        net_gap = min(abs(float(p["X"])) for p in serve["points"])
        rep["net_gap_m"] = net_gap
        if net_gap > NET_GAP_WARNING_M:
            warnings.append(
                f"Serve {rep['serve_index']}: tracking only began {net_gap:.1f} m past the net "
                f"(the shuttle was not detected earlier), so its peak height and net clearance "
                f"are measured from the first tracked point rather than at the net."
            )
        if not serve["complete"]:
            warnings.append(
                f"Serve {rep['serve_index']}: the shuttle was lost before it touched the "
                f"floor, so its landing point is estimated from the last tracked points."
            )
        shots.append(rep)
        kept.append(dict(serve, serve_index=rep["serve_index"]))

    if not shots:
        return None, kept, discarded

    report = dict(shots[0])
    report["shots"] = shots
    report["serve_count"] = len(shots)
    report["average_score"] = float(np.mean([s["final_score"] for s in shots]))
    report["discarded_fragments"] = len(discarded)
    return report, kept, discarded


MULTI_SERVE_DEBUG_FILE = os.path.join(BASE, "multi_serve_debug.json")


def write_multi_serve_debug(diagnostics, serves, discarded):
    """
    Save everything needed to replay the multi-serve tracking offline
    (detections per frame, camera matrices, chosen pairs, 3D points, how the
    points were split). Returns the file path, or None if it could not be written.
    """
    debug = diagnostics.pop("debug", None)
    if not debug:
        return None
    debug["diagnostics"] = dict(diagnostics)
    debug["serves"] = [
        {"serve_index": s["serve_index"], "start_frame": s["start_frame"],
         "end_frame": s["end_frame"], "points": len(s["points"]), "complete": s["complete"]}
        for s in serves
    ]
    debug["discarded"] = discarded
    try:
        with open(MULTI_SERVE_DEBUG_FILE, "w", encoding="utf-8") as f:
            json.dump(debug, f)
        return MULTI_SERVE_DEBUG_FILE
    except (OSError, TypeError, ValueError):
        return None


def explain_multi_serve_failure(diag, trajectory, discarded, debug_path):
    """A message that says WHERE the points were lost, not just that none were found."""
    parts = [
        "No complete serve was found.",
        f"Frames processed: {diag.get('frames_processed', '?')} at {diag.get('fps', 0):.0f} fps.",
        f"YOLO detections: side {diag.get('candidates_side', '?')} "
        f"(in {diag.get('frames_with_candidates_side', '?')} frames), "
        f"back {diag.get('candidates_back', '?')} (in {diag.get('frames_with_candidates_back', '?')} frames).",
        f"Ignored as resting shuttles: side {diag.get('static_removed_side', '?')}, "
        f"back {diag.get('static_removed_back', '?')}.",
        f"Frames where both cameras had a moving shuttle: {diag.get('frames_both_moving', '?')}; "
        f"rejected by the camera-pair check: {diag.get('rejected_pair_frames', '?')} "
        f"(typical pair error {diag.get('median_pair_error_pct', 0):.1f}% of the frame diagonal, "
        f"accepted up to {diag.get('pair_error_limit_pct', 0):.1f}%).",
    ]
    if trajectory:
        frames = [int(p["frame"]) for p in trajectory]
        zs = [float(p["Z"]) for p in trajectory]
        parts.append(
            f"Tracked 3D points: {len(trajectory)} (frames {min(frames)}-{max(frames)}), "
            f"highest point Z = {max(zs):.2f} m, lowest Z = {min(zs):.2f} m."
        )
    else:
        parts.append("Tracked 3D points: 0.")
    if discarded:
        shown = "; ".join(
            f"frames {d['start_frame']}-{d['end_frame']}: {d['points']} pts, {d['reason']}"
            for d in discarded[:5]
        )
        more = f" (+{len(discarded) - 5} more)" if len(discarded) > 5 else ""
        parts.append(f"Fragments discarded: {shown}{more}.")
    elif trajectory:
        parts.append(
            "No fragments were discarded, so no track ever rose above the launch height "
            "(0.5 m) - check the Z values above."
        )
    else:
        parts.append(
            "Nothing reached the 3D stage: check that the shuttle is detected as MOVING in both "
            "cameras at the same time (see the detection counts above)."
        )
    if debug_path:
        parts.append(f"Details saved to {os.path.basename(debug_path)}.")
    return " ".join(parts)


def run_evaluation_job(job_id):
    try:
        update_evaluation_job(job_id, status="loading", message="Loading YOLO model…")

        tracker = BadmintonTracker3D(
            model_path=MODEL_PATH,
            calib_file=CALIBRATION_FILE
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

        # The number of serves is decided automatically -- the user is not asked.
        # Pass 1 always runs the multi-serve tracker (it ignores shuttles resting
        # on the court and picks the best side/back pair per frame), then
        # multi_serve.split_serves() counts the serves from the trajectory itself
        # (launch above LAUNCH_Z_M, floor contact after an apex, detection gaps).
        #   >= 2 serves -> multi-serve report (dashboard pager)
        #   exactly 1   -> normal single-serve report
        #   0 found     -> fall back to the original single-serve tracker
        trajectory, motion_diagnostics = tracker.process_videos(
            os.path.join(UPLOAD_VIDEO_DIR, "side_sync.mp4"),
            os.path.join(UPLOAD_VIDEO_DIR, "back_sync.mp4"),
            progress_callback=progress_callback,
            multi_serve=True,
        )

        warnings = build_trajectory_warnings(motion_diagnostics)

        # service_side comes from the calibration the trajectory was
        # triangulated with: right = Y 0..+2.59 m is IN, left = Y -2.59..0 m.
        # serve_type defaults to "auto": the evaluator classifies short vs.
        # high/long serve itself from where the shuttle actually landed.
        service_side = read_saved_service_side()
        landing_target = read_landing_target()

        report, serves, discarded = None, [], []
        if len(trajectory) > 0:
            update_evaluation_job(
                job_id,
                status="evaluating",
                message="Finding and scoring serves…",
                current=get_evaluation_job(job_id).get("total", 0),
            )
            report, serves, discarded = build_multi_serve_report(
                trajectory, motion_diagnostics, service_side, warnings, landing_target
            )
        debug_path = write_multi_serve_debug(motion_diagnostics, serves, discarded)

        if report is not None:
            if report["serve_count"] > 1:
                report["detection_mode"] = "multi"
            else:
                # One serve: present it exactly like a normal single-serve run.
                for key in ("shots", "serve_count", "average_score", "discarded_fragments"):
                    report.pop(key, None)
                report["detection_mode"] = "single"
            chart_serves = serves
        else:
            # The multi-serve tracker found no serve. Keep the original behaviour
            # as a safety net: treat the whole clip as one serve (this re-runs
            # YOLO, so it only happens when the first pass found nothing).
            multi_message = explain_multi_serve_failure(
                motion_diagnostics, trajectory, discarded, debug_path
            )
            print(f"[evaluate] no serve found automatically, falling back to single-serve mode. {multi_message}")
            update_evaluation_job(
                job_id,
                status="processing",
                current=0,
                message="No separate serve found - retrying as a single serve…",
            )
            trajectory, motion_diagnostics = tracker.process_videos(
                os.path.join(UPLOAD_VIDEO_DIR, "side_sync.mp4"),
                os.path.join(UPLOAD_VIDEO_DIR, "back_sync.mp4"),
                progress_callback=progress_callback,
                multi_serve=False,
            )
            warnings = build_trajectory_warnings(motion_diagnostics)

            if len(trajectory) == 0:
                update_evaluation_job(
                    job_id,
                    status="error",
                    error="No trajectory points detected. " + multi_message,
                    warnings=warnings,
                )
                return

            update_evaluation_job(
                job_id,
                status="evaluating",
                message="Calculating serve performance…",
                current=get_evaluation_job(job_id).get("total", 0),
            )
            report = evaluate_serve_performance(
                trajectory, service_side=service_side, landing_target=landing_target
            )
            report["detection_mode"] = "single_fallback"
            chart_serves = None

        save_trajectory_chart(
            trajectory,
            os.path.join(STATIC_DIR, "trajectory.png"),
            serves=chart_serves,
        )

        update_evaluation_job(
            job_id,
            status="done",
            done=True,
            message="Evaluation complete.",
            result_ready=True,
            trajectory_points=len(trajectory),
            report=report,
            warnings=warnings,
            motion_diagnostics=motion_diagnostics,
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

# Track the most recently *completed* synchronization job. This app is
# single-session (one calibration in progress at a time), so the calibrate
# page doesn't carry a job_id of its own — but the debug "Check
# Synchronization" link still needs to know which job's frame pairs to show.
latest_synced_job_id = None


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


PREVIEW_MAX_WIDTH = 1280


def make_browser_playable_copy(src_path, dst_path):
    """Make a small H.264 preview of a synchronized video for the browser.

    create_sync_video() below writes with OpenCV's "mp4v" fourcc (MPEG-4
    Part 2) because that's the codec most reliably available for
    VideoWriter across machines, and OpenCV can read its own mp4v output
    fine for triangulation. Browsers, however, essentially never support
    mp4v decoding in a <video> tag -- they need proper H.264 -- so a plain
    copy of that file will silently fail to play (Chrome/Firefox just fire
    a media error event with no dialog, which looks like a missing file).

    This writes an H.264/yuv420p copy with ffmpeg, scaled down to at most
    PREVIEW_MAX_WIDTH pixels wide and encoded with a fast preset. The preview
    is only for the results page -- triangulation always reads the original
    full-resolution file -- and frame rate and timing are unchanged, so the
    serve start times shown on the results page match the preview. A 4K,
    60 fps, ~900 frame clip takes tens of seconds this way; re-encoding it at
    full resolution with the default preset took several minutes and hit the
    old 120 s timeout, which left an unplayable copy (the "Back view"
    placeholder).

    If ffmpeg isn't installed or the encode fails, we fall back to a plain
    copy rather than crashing the request -- that copy just won't play
    in-browser.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-vf", f"scale='min({PREVIEW_MAX_WIDTH},iw)':-2",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "26",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                dst_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=900,
        )
        if result.returncode == 0 and os.path.exists(dst_path):
            return
    except (OSError, subprocess.SubprocessError):
        pass

    shutil.copyfile(src_path, dst_path)


def _video_codec(path):
    """Codec name of the first video stream (via ffprobe), or None if it can't be read."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        name = out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 and out.stdout.strip() else ""
        return name or None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


_preview_repair_attempted = set()


def ensure_browser_previews():
    """
    Repair the browser previews of an already-synchronized clip.

    Clips synchronized before the preview fix may have an unplayable (mp4v)
    copy in static/uploads. If ffprobe says a preview is not H.264, rebuild it
    from the full-resolution file once per process (a few tens of seconds for
    a 4K clip, only the first time the results page is opened).
    """
    for name in ("side_sync.mp4", "back_sync.mp4"):
        src = os.path.join(UPLOAD_VIDEO_DIR, name)
        dst = os.path.join(STATIC_DIR, name)
        if not os.path.exists(src):
            continue
        key = (name, os.path.getmtime(src))
        if key in _preview_repair_attempted:
            continue
        codec = _video_codec(dst) if os.path.exists(dst) else ""
        if codec is None or codec == "h264":      # None: ffprobe unavailable -> leave as is
            continue
        _preview_repair_attempted.add(key)
        make_browser_playable_copy(src, dst)


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

    # The dashboard plays these directly in the browser, so make them
    # reachable the same way the calibration frame JPGs already are
    # (Flask's default /static route serves anything under STATIC_DIR),
    # re-encoded to a codec browsers can actually decode.
    make_browser_playable_copy(
        os.path.join(UPLOAD_VIDEO_DIR, "back_sync.mp4"),
        os.path.join(STATIC_DIR, "back_sync.mp4"),
    )
    make_browser_playable_copy(
        os.path.join(UPLOAD_VIDEO_DIR, "side_sync.mp4"),
        os.path.join(STATIC_DIR, "side_sync.mp4"),
    )

    return len(back_paths), sync_fps


def run_upload_job(job_id):
    try:
        job = get_upload_job(job_id)
        if not job:
            return

        side_path = job["side_path"]
        back_path = job["back_path"]
        work_dir = job["work_dir"]
        side_frames = os.path.join(work_dir, "side_frames")
        back_frames = os.path.join(work_dir, "back_frames")
        matched_root = os.path.join(work_dir, "matched")

        update_upload_job(
            job_id,
            status="extracting",
            current=0,
            total=0,
            message="Extracting video frames…",
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

        global latest_synced_job_id
        latest_synced_job_id = job_id

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



@app.route("/upload", methods=["POST"])
def upload():
    side_video = request.files.get("side_video")
    back_video = request.files.get("back_video")

    if not side_video or not back_video:
        return jsonify({"error": "Please upload both side and back camera videos."}), 400

    if not side_video.filename or not back_video.filename:
        return jsonify({"error": "Please upload both side and back camera videos."}), 400

    job_id = uuid.uuid4().hex
    work_dir = os.path.join(UPLOAD_VIDEO_DIR, f"sync_job_{job_id}")
    os.makedirs(work_dir, exist_ok=True)

    # Preserve the uploaded files in their original container/extension.
    # Videos are expected to already be browser-compatible MP4s, so this
    # same file doubles as both the CV-processing source and the preview.
    side_ext = os.path.splitext(side_video.filename)[1].lower() or ".mp4"
    back_ext = os.path.splitext(back_video.filename)[1].lower() or ".mp4"

    side_path = os.path.join(work_dir, f"side_original{side_ext}")
    back_path = os.path.join(work_dir, f"back_original{back_ext}")

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
            if os.path.exists(job.get("side_path", "")) else None
        ),
        "back_preview_url": (
            url_for("browser_preview", job_id=job_id, camera="back")
            if os.path.exists(job.get("back_path", "")) else None
        ),
        "verify_url": (
            url_for("sync_verify_page", job_id=job_id)
            if job.get("done") else None
        ),
    })


@app.route("/browser-preview/<job_id>/<camera>")
def browser_preview(job_id, camera):
    """Serve the originally uploaded video for the upload page preview.

    Videos are expected to already be browser-compatible MP4 (H.264/AAC)
    before upload, so no server-side transcoding happens here — this just
    streams the same file that CV processing uses.
    """
    job = get_upload_job(job_id)
    if job is None:
        return "Upload job not found.", 404

    if camera == "side":
        path = job.get("side_path")
    elif camera == "back":
        path = job.get("back_path")
    else:
        return "Invalid camera.", 400

    if not path or not os.path.exists(path):
        return "Video not found.", 404

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
    if side_img is None or back_img is None:
        # No calibration frames yet — e.g. straight after a server restart,
        # before any upload has finished. Send the user back to upload
        # instead of crashing with a 500 (cv2 returns None, not an
        # exception, for a missing/unreadable file).
        return redirect(url_for("index"))

    side_h, side_w = side_img.shape[:2]
    back_h, back_w = back_img.shape[:2]

    # Service area being served into. The page reloads with ?side=left /
    # ?side=right when the toggle is clicked; the default is right. An
    # unknown value falls back to the default instead of erroring.
    try:
        service_side = normalize_service_side(request.args.get("side"))
    except ValueError:
        service_side = DEFAULT_SERVICE_SIDE

    side_guess, side_conf, side_source = auto_detect_gcps(side_img, "side", service_side)
    back_guess, back_conf, back_source = auto_detect_gcps(back_img, "back", service_side)

    def as_list(points_dict):
        return [list(points_dict[i]) for i in range(len(WORLD_GCPS))]

    def as_list_conf(conf_dict):
        return [conf_dict[i] for i in range(len(WORLD_GCPS))]

    return render_template("calibrate.html",
                            side_w=side_w, side_h=side_h,
                            back_w=back_w, back_h=back_h,
                            num_points=len(WORLD_GCPS),
                            gcp_labels=gcp_labels_for(service_side),
                            service_side=service_side,
                            service_side_label=SERVICE_SIDE_LABELS[service_side],
                            landing_target_text=evaluator.describe_landing_target(
                                read_landing_target(), service_side
                            )["text"],
                            service_sides=[
                                {"key": k, "label": SERVICE_SIDE_LABELS[k]}
                                for k in SERVICE_SIDES
                            ],
                            side_meta=CAMERA_META["side"],
                            back_meta=CAMERA_META["back"],
                            side_guess=as_list(side_guess),
                            back_guess=as_list(back_guess),
                            side_confidence=as_list_conf(side_conf),
                            back_confidence=as_list_conf(back_conf),
                            side_source=side_source,
                            back_source=back_source,
                            sync_job_id=latest_synced_job_id)

@app.route("/save_calibration", methods=["POST"])
def save_calibration():
    data = request.get_json()
    try:
        # A missing service_side means an older page: treat it as right.
        service_side = normalize_service_side(data.get("service_side"))
        P_side = solve_camera(data["side_points"], data["side_w"], data["side_h"],
                              service_side=service_side)
        P_back = solve_camera(data["back_points"], data["back_w"], data["back_h"],
                              service_side=service_side)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Store the side with the matrices: the evaluation step reads it back
    # so IN/OUT is judged against the box that was actually calibrated.
    np.savez(CALIBRATION_FILE, P_side=P_side, P_back=P_back,
             service_side=np.array(service_side))

    # Remember this confirmed calibration for future auto-matching.
    # Best-effort: a failure here must not block saving the calibration.
    try:
        side_img = cv2.imread(os.path.join(STATIC_DIR, "side_frame.jpg"))
        back_img = cv2.imread(os.path.join(STATIC_DIR, "back_frame.jpg"))
        if side_img is not None:
            save_landmark(side_img, landmark_key("side", service_side), data["side_points"])
        if back_img is not None:
            save_landmark(back_img, landmark_key("back", service_side), data["back_points"])
    except Exception:
        import traceback
        traceback.print_exc()

    return jsonify({"status": "ok", "service_side": service_side})

@app.route("/run_evaluation", methods=["POST"])
def run_evaluation():
    if not os.path.exists(CALIBRATION_FILE):
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
        return render_template(
            "dashboard.html",
            error=job.get("error"),
            warnings=job.get("warnings", []),
        )

    if not job.get("done"):
        return redirect(url_for("calibrate_page"))

    ensure_browser_previews()
    return render_template(
        "dashboard.html",
        report=job["report"],
        num_points=job.get("trajectory_points", 0),
        warnings=job.get("warnings", []),
    )

def save_trajectory_chart(trajectory, out_path, serves=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    if serves:
        # multi-serve: one line per serve
        for serve in serves:
            plt.plot(
                [p["X"] for p in serve["points"]],
                [p["Z"] for p in serve["points"]],
                marker="o", markersize=3,
                label=f"Serve {serve['serve_index']}",
            )
    else:
        plt.plot([p["X"] for p in trajectory], [p["Z"] for p in trajectory], marker="o")
    plt.xlabel("X - Depth from net (m)")
    plt.ylabel("Z - Height (m)")
    plt.title("Serve Trajectory (Side Profile)")
    plt.axhline(1.55, color="red", linestyle="--", label="Net height")
    plt.ylim(bottom=-0.1, top=4.0)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------------
# Landing target (what the landing score aims at), chosen on the upload page
# ---------------------------------------------------------------------------
LANDING_TARGET_FILE = os.path.join(BASE, "landing_target.json")


def read_landing_target():
    """
    The coach's saved landing target, validated. Falls back to the default
    (all four corners, GCP3-GCP6) if nothing was saved or the file is unusable.
    """
    try:
        with open(LANDING_TARGET_FILE, "r", encoding="utf-8") as f:
            return evaluator.normalize_landing_target(json.load(f))
    except (OSError, ValueError, TypeError):
        return evaluator.normalize_landing_target(None)


def write_landing_target(target):
    """Save atomically so a crash can never leave a half-written file."""
    tmp = LANDING_TARGET_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(target, f)
    os.replace(tmp, LANDING_TARGET_FILE)


def _landing_target_payload(target):
    """Everything the upload page needs to draw the grid and describe the choice."""
    return {
        "target": target,
        "grid": {
            "cols": evaluator.GRID_COLS,
            "rows": evaluator.GRID_ROWS,
            "x_min_m": evaluator.PREFERRED_DROP_X_MIN,
            "x_max_m": evaluator.PREFERRED_DROP_X_MAX,
            "width_m": evaluator.SERVICE_AREAS["right"]["y_max"] - evaluator.SERVICE_AREAS["right"]["y_min"],
        },
        "zero_score_distance_m": evaluator.LANDING_ZERO_SCORE_DISTANCE_M,
    }


@app.route("/api/landing_target", methods=["GET", "POST"])
def api_landing_target():
    if request.method == "GET":
        return jsonify(_landing_target_payload(read_landing_target()))

    try:
        target = evaluator.normalize_landing_target(request.get_json(silent=True) or {})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        write_landing_target(target)
    except OSError as exc:
        return jsonify({"error": f"Could not save the landing target: {exc}"}), 500
    return jsonify(_landing_target_payload(target))


if __name__ == "__main__":
    app.run(debug=True)