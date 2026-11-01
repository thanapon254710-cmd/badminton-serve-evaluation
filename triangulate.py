import cv2
import numpy as np
import time
from ultralytics import YOLO

import multi_serve

class BadmintonTracker3D:
    def __init__(self, model_path, calib_file, infer_width=960):
        """
        infer_width: resize frames to this width before YOLO inference
        (keeps aspect ratio). Smaller = faster, at some cost to small-object
        detection accuracy. Set to None to disable resizing and run at
        native resolution.
        """
        self.model = YOLO(model_path)
        self.infer_width = infer_width

        device = getattr(self.model, "device", None)
        print(f"[triangulate] YOLO model loaded. Device: {device}")
        try:
            import torch
            print(f"[triangulate] CUDA available: {torch.cuda.is_available()}")
        except ImportError:
            pass

        # Load precomputed camera projection matrices
        calib = np.load(calib_file)
        self.P_side = calib['P_side']
        self.P_back = calib['P_back']

    def detect_shuttlecock_candidates_2d(self, frame):
        """
        Run YOLOv8 and return EVERY shuttlecock-like detection in the frame as
        a list of (cx, cy, conf) in full-resolution pixels (possibly empty).
        Used by the multi-serve path, where resting shuttles on the court are
        detected alongside the one in flight and must be told apart later.
        """
        infer_frame = frame
        scale = 1.0

        if self.infer_width is not None and frame.shape[1] > self.infer_width:
            scale = self.infer_width / frame.shape[1]
            new_h = int(frame.shape[0] * scale)
            infer_frame = cv2.resize(frame, (self.infer_width, new_h))

        results = self.model(infer_frame, conf=0.20, verbose=False)
        candidates = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                candidates.append((
                    float((x1 + x2) / 2.0) / scale,
                    float((y1 + y2) / 2.0) / scale,
                    conf,
                ))
        return candidates

    def detect_shuttlecock_2d(self, frame):
        """Runs YOLOv8 inference to detect shuttlecock centroid (x, y).

        When more than one candidate box is found in a frame (e.g. a
        stray/resting shuttle sitting on the court in addition to the
        one actually in play), the previous implementation just
        returned whichever box happened to come back first -- with no
        regard for confidence. That made it easy to silently lock onto
        the wrong object for some frames. We now keep the
        highest-confidence detection instead. (The multi-serve path
        uses detect_shuttlecock_candidates_2d() and handles resting
        shuttles explicitly.)
        """
        candidates = self.detect_shuttlecock_candidates_2d(frame)
        if not candidates:
            return None
        best = candidates[0]
        for c in candidates[1:]:
            if c[2] > best[2]:
                best = c
        return np.array([best[0], best[1]], dtype=np.float32)

    # A camera whose detected shuttlecock position moves less than this
    # fraction of the frame diagonal across the whole clip is treated as
    # "suspiciously static" -- a strong sign the detector locked onto a
    # stationary object (a resting shuttle, the racket, a bright spot on
    # the net tape, etc.) rather than the shuttle actually in flight.
    MOTION_SPAN_MIN_FRACTION = 0.05

    @staticmethod
    def _motion_diagnostics_for_camera(points_2d, frame_diagonal_px):
        """
        Summarizes how much a single camera's detected 2D points moved
        over the course of the clip, as a fraction of that camera's
        frame diagonal (so it's comparable across different resolutions
        / aspect ratios).
        """
        if len(points_2d) < 2 or frame_diagonal_px <= 0:
            return {
                "points_detected": len(points_2d),
                "motion_px": 0.0,
                "motion_fraction": 0.0,
                "suspect_static": len(points_2d) > 0,
            }

        pts = np.array(points_2d, dtype=np.float32)
        span_x = float(pts[:, 0].max() - pts[:, 0].min())
        span_y = float(pts[:, 1].max() - pts[:, 1].min())
        motion_px = float(np.hypot(span_x, span_y))
        motion_fraction = motion_px / frame_diagonal_px

        return {
            "points_detected": len(points_2d),
            "motion_px": motion_px,
            "motion_fraction": motion_fraction,
            "suspect_static": motion_fraction < BadmintonTracker3D.MOTION_SPAN_MIN_FRACTION,
        }

    def triangulate_dlt(self, pt_side, pt_back):
        """
        Reconstructs 3D coordinate (X, Y, Z) in meters from 2D pixel coordinates
        using Direct Linear Transformation (DLT).
        """
        u1, v1 = pt_side[0], pt_side[1]
        u2, v2 = pt_back[0], pt_back[1]

        A = np.zeros((4, 4), dtype=np.float32)
        A[0] = u1 * self.P_side[2] - self.P_side[0]
        A[1] = v1 * self.P_side[2] - self.P_side[1]
        A[2] = u2 * self.P_back[2] - self.P_back[0]
        A[3] = v2 * self.P_back[2] - self.P_back[1]

        # Singular Value Decomposition
        _, _, Vh = np.linalg.svd(A)
        X_homo = Vh[-1]
        X_3d = X_homo[:3] / X_homo[3]  # Normalize by w
        return X_3d

    def process_videos(self, side_video_path, back_video_path, log_every=10,
                       progress_callback=None, multi_serve=False):
        """
        multi_serve=False (default): original behaviour -- one detection per
        frame, the whole clip is one serve.

        multi_serve=True: keep every candidate, drop shuttles resting on the
        court, pick the right side/back pair per frame and return all moving
        points; the caller then separates the serves with
        multi_serve.split_serves(trajectory, diagnostics["fps"]).
        """
        if multi_serve:
            return self._process_videos_multi(
                side_video_path, back_video_path, log_every, progress_callback
            )
        cap_side = cv2.VideoCapture(side_video_path)
        cap_back = cv2.VideoCapture(back_video_path)

        total_side = int(cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        total_back = int(cap_back.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = min(total_side, total_back)
        print(
            f"[triangulate] side frames: {total_side}, "
            f"back frames: {total_back}, processing: {total_frames}"
        )

        side_diag_px = float(np.hypot(
            cap_side.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap_side.get(cv2.CAP_PROP_FRAME_HEIGHT),
        ))
        back_diag_px = float(np.hypot(
            cap_back.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap_back.get(cv2.CAP_PROP_FRAME_HEIGHT),
        ))

        trajectory_3d = []
        side_points_2d = []
        back_points_2d = []
        frame_idx = 0
        start_time = time.time()

        while cap_side.isOpened() and cap_back.isOpened():
            ret_s, frame_s = cap_side.read()
            ret_b, frame_b = cap_back.read()

            if not ret_s or not ret_b:
                break

            pt_side = self.detect_shuttlecock_2d(frame_s)
            pt_back = self.detect_shuttlecock_2d(frame_b)

            if pt_side is not None:
                side_points_2d.append(pt_side)
            if pt_back is not None:
                back_points_2d.append(pt_back)

            if pt_side is not None and pt_back is not None:
                pt_3d = self.triangulate_dlt(pt_side, pt_back)
                trajectory_3d.append({
                    'frame': frame_idx,
                    'X': pt_3d[0],  # Length (m)
                    'Y': pt_3d[1],  # Width (m)
                    'Z': pt_3d[2]   # Height (m)
                })

            frame_idx += 1

            elapsed = time.time() - start_time
            fps = frame_idx / elapsed if elapsed > 0 else 0

            # Web UI progress: report every processed frame. The browser
            # polls the server, so these updates are cheap and give an
            # accurate "current frame / total frames" display.
            if progress_callback is not None:
                progress_callback(
                    current=frame_idx,
                    total=total_frames,
                    fps=fps,
                    detected=len(trajectory_3d),
                )

            if frame_idx % log_every == 0:
                print(
                    f"[triangulate] frame {frame_idx} "
                    f"({elapsed:.1f}s elapsed, {fps:.2f} fps, "
                    f"{len(trajectory_3d)} points detected so far)"
                )

        elapsed = time.time() - start_time
        print(
            f"[triangulate] done: {frame_idx} frames processed in "
            f"{elapsed:.1f}s, {len(trajectory_3d)} 3D points detected"
        )

        cap_side.release()
        cap_back.release()

        side_diag_result = self._motion_diagnostics_for_camera(side_points_2d, side_diag_px)
        back_diag_result = self._motion_diagnostics_for_camera(back_points_2d, back_diag_px)
        diagnostics = {
            "side_points_detected": side_diag_result["points_detected"],
            "side_motion_px": side_diag_result["motion_px"],
            "side_motion_fraction": side_diag_result["motion_fraction"],
            "side_suspect_static": side_diag_result["suspect_static"],
            "back_points_detected": back_diag_result["points_detected"],
            "back_motion_px": back_diag_result["motion_px"],
            "back_motion_fraction": back_diag_result["motion_fraction"],
            "back_suspect_static": back_diag_result["suspect_static"],
        }
        if diagnostics["side_suspect_static"] or diagnostics["back_suspect_static"]:
            print(f"[triangulate] WARNING -- suspiciously static camera(s): {diagnostics}")

        return trajectory_3d, diagnostics

    def _process_videos_multi(self, side_video_path, back_video_path, log_every, progress_callback):
        """
        Multi-serve tracking. Two passes:

          1. run YOLO on every frame of both videos and keep ALL candidates;
          2. per camera, drop candidates sitting on the same spot for a long
             time (shuttles resting on the court / held in the hand), then in
             every frame pick the side/back pair that triangulates and
             reprojects best, and triangulate it.

        Returns (trajectory_3d, diagnostics) like the single-serve path, with
        extra diagnostics (fps, how many static detections were dropped, how
        many frames needed pair selection).
        """
        cap_side = cv2.VideoCapture(side_video_path)
        cap_back = cv2.VideoCapture(back_video_path)

        total_side = int(cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        total_back = int(cap_back.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = min(total_side, total_back)
        fps_video = float(cap_side.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps_video) or fps_video <= 0:
            fps_video = 30.0
        print(
            f"[triangulate] multi-serve mode. side frames: {total_side}, "
            f"back frames: {total_back}, processing: {total_frames}, video fps: {fps_video:.2f}"
        )

        side_diag_px = float(np.hypot(
            cap_side.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap_side.get(cv2.CAP_PROP_FRAME_HEIGHT),
        ))
        back_diag_px = float(np.hypot(
            cap_back.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap_back.get(cv2.CAP_PROP_FRAME_HEIGHT),
        ))

        # ---- pass 1: all candidates, every frame --------------------------
        side_cands, back_cands = [], []
        start_time = time.time()
        frames_with_both = 0
        while cap_side.isOpened() and cap_back.isOpened():
            ret_s, frame_s = cap_side.read()
            ret_b, frame_b = cap_back.read()
            if not ret_s or not ret_b:
                break

            cs = self.detect_shuttlecock_candidates_2d(frame_s)
            cb = self.detect_shuttlecock_candidates_2d(frame_b)
            side_cands.append(cs)
            back_cands.append(cb)
            if cs and cb:
                frames_with_both += 1

            n_done = len(side_cands)
            elapsed = time.time() - start_time
            if progress_callback is not None:
                progress_callback(
                    current=n_done,
                    total=total_frames,
                    fps=(n_done / elapsed if elapsed > 0 else 0.0),
                    detected=frames_with_both,
                )
            if n_done % log_every == 0:
                print(f"[triangulate] detected frame {n_done} / {total_frames}")

        cap_side.release()
        cap_back.release()

        # ---- static (resting / held) shuttles ------------------------------
        side_static = multi_serve.flag_static_detections(side_cands, side_diag_px, fps_video)
        back_static = multi_serve.flag_static_detections(back_cands, back_diag_px, fps_video)
        static_removed_side = sum(f for fl in side_static for f in fl)
        static_removed_back = sum(f for fl in back_static for f in fl)

        # ---- pass 2: choose a pair per frame and triangulate ---------------
        pairs, pair_info = multi_serve.select_pairs(
            side_cands, back_cands, side_static, back_static,
            self.P_side, self.P_back, side_diag_px, back_diag_px,
        )
        trajectory_3d = []
        side_points_2d, back_points_2d = [], []
        errors = []
        debug_frames = []   # replayable record of what happened in every frame
        for frame_idx, pair in enumerate(pairs):
            debug_frames.append({
                "side": [[round(c[0], 1), round(c[1], 1), round(c[2], 3), int(side_static[frame_idx][j])]
                         for j, c in enumerate(side_cands[frame_idx])],
                "back": [[round(c[0], 1), round(c[1], 1), round(c[2], 3), int(back_static[frame_idx][j])]
                         for j, c in enumerate(back_cands[frame_idx])],
                # [side index, back index, error as % of each camera's own diagonal]
                "pair": None if pair is None else [pair["i_side"], pair["i_back"], round(pair["err"] * 100.0, 3)],
            })
            if pair is None:
                continue
            errors.append(pair["err"])
            side_points_2d.append(np.array(side_cands[frame_idx][pair["i_side"]][:2], dtype=np.float32))
            back_points_2d.append(np.array(back_cands[frame_idx][pair["i_back"]][:2], dtype=np.float32))
            trajectory_3d.append({
                'frame': frame_idx,
                'X': pair["X3"][0],
                'Y': pair["X3"][1],
                'Z': pair["X3"][2],
            })
        frames_both_moving = pair_info["frames_both_moving"]
        ambiguous_frames = pair_info["ambiguous"]
        rejected_frames = pair_info["rejected"]

        print(
            f"[triangulate] multi-serve done: {len(side_cands)} frames, "
            f"{len(trajectory_3d)} 3D points, dropped static detections "
            f"side={static_removed_side} back={static_removed_back}, "
            f"pair-selection frames={ambiguous_frames}, rejected={rejected_frames}"
        )

        side_diag_result = self._motion_diagnostics_for_camera(side_points_2d, side_diag_px)
        back_diag_result = self._motion_diagnostics_for_camera(back_points_2d, back_diag_px)
        diagnostics = {
            "side_points_detected": side_diag_result["points_detected"],
            "side_motion_px": side_diag_result["motion_px"],
            "side_motion_fraction": side_diag_result["motion_fraction"],
            "side_suspect_static": side_diag_result["suspect_static"],
            "back_points_detected": back_diag_result["points_detected"],
            "back_motion_px": back_diag_result["motion_px"],
            "back_motion_fraction": back_diag_result["motion_fraction"],
            "back_suspect_static": back_diag_result["suspect_static"],
            "multi_serve": True,
            "fps": fps_video,
            "frames_processed": len(side_cands),
            "static_removed_side": int(static_removed_side),
            "static_removed_back": int(static_removed_back),
            "candidates_side": int(sum(len(c) for c in side_cands)),
            "candidates_back": int(sum(len(c) for c in back_cands)),
            "frames_with_candidates_side": int(sum(1 for c in side_cands if c)),
            "frames_with_candidates_back": int(sum(1 for c in back_cands if c)),
            "frames_both_moving": int(frames_both_moving),
            "pair_selection_frames": int(ambiguous_frames),
            "rejected_pair_frames": int(rejected_frames),
            "mean_reprojection_pct": float(np.mean(errors)) * 100.0 if errors else 0.0,   # % of each camera's own diagonal
            "median_pair_error_pct": pair_info["median_error"] * 100.0,
            "pair_error_limit_pct": pair_info["limit"] * 100.0,
        }
        # Everything needed to replay the tracking offline (written to a JSON file
        # by the web app; ~50 KB for a few seconds of video).
        diagnostics["debug"] = {
            "fps": fps_video,
            "side_diag_px": side_diag_px,
            "back_diag_px": back_diag_px,
            "P_side": np.asarray(self.P_side, dtype=float).tolist(),
            "P_back": np.asarray(self.P_back, dtype=float).tolist(),
            "frames": debug_frames,
            "trajectory": [
                {"frame": int(p["frame"]), "X": round(float(p["X"]), 3),
                 "Y": round(float(p["Y"]), 3), "Z": round(float(p["Z"]), 3)}
                for p in trajectory_3d
            ],
        }
        if diagnostics["side_suspect_static"] or diagnostics["back_suspect_static"]:
            print(f"[triangulate] WARNING -- suspiciously static camera(s): "
                  f"{ {k: v for k, v in diagnostics.items() if k != 'debug'} }")

        return trajectory_3d, diagnostics