"""
multi_serve.py -- find and separate several serves in one video pair.

Why this is needed
------------------
The original tracker keeps ONE detection per frame (the highest-confidence
one) and treats the whole clip as ONE serve. With several serves in a clip
two things break:

  1. Shuttles that already landed are still lying on the court. YOLO keeps
     detecting them, sometimes with a higher confidence than the shuttle in
     flight, so the "best detection" jumps to a resting shuttle.
  2. Even with perfect detections, all serves end up in one long trajectory.

Pipeline (each step is a pure function, testable without video or YOLO)
-----------------------------------------------------------------------
  flag_static_detections()  mark detections that sit on the same spot for a
                            long time (resting / held shuttles) in ONE camera
  choose_best_pair()        when a frame has several moving candidates in
                            each camera, pick the side/back pair whose 3D
                            triangulation reprojects best into both views
                            (the 3D point minimises the pixel error in both
                            cameras, each measured against its own frame size)
  split_serves()            cut the resulting 3D trajectory into one segment
                            per serve (gap in detections, floor contact, or an
                            impossible jump), trim each segment at landing,
                            and discard fragments that are not serves

Every threshold below is a STARTING VALUE chosen from geometry (see comments)
and is meant to be tuned on your own clips. Time-based values are in seconds
so 30 fps and 60 fps footage behave alike.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Static (resting / held shuttle) detection
# ---------------------------------------------------------------------------
# Detections within this fraction of the frame diagonal count as "the same
# spot" (about 11 px on a 1080p frame): larger than detector jitter on a
# resting shuttle, smaller than the shuttle's motion between frames.
STATIC_RADIUS_FRACTION = 0.005
# A spot detected in at least this many seconds' worth of frames is static.
# 0.6 s is longer than the shuttle lingers near its apex in the camera that
# looks along the court (about 0.25 s inside the radius), so a shuttle in
# flight is not mistaken for a resting one.
STATIC_MIN_SECONDS = 0.6
STATIC_MIN_FRAMES_FLOOR = 4

# ---------------------------------------------------------------------------
# Camera-pair selection
# ---------------------------------------------------------------------------
# Reprojection error of a side/back pair, as a fraction of each camera's OWN
# frame diagonal (error_side / diag_side and error_back / diag_back, averaged).
# Measuring each camera against its own frame size matters when the cameras
# record at different resolutions (for example a 4K back camera and a 1080p
# side camera): adding raw pixel errors would let the higher-resolution camera
# dominate.
#
# The accepted limit adapts to the clip: PAIR_ADAPTIVE_FACTOR x the median error
# of all best pairs in the clip, kept between PAIR_MIN_ERROR_FRACTION and
# PAIR_MAX_ERROR_FRACTION. Two cameras that agree well (small median error) get
# a strict limit that rejects wrong pairings; cameras whose calibrations are
# inconsistent get a wider one. On a real two-iPhone clip with hand-clicked
# GCPs the correct pairs had errors of 3-5% (up to 7.7%) of the diagonal, which
# a fixed 1-3% limit rejected almost entirely. Wrong pairs are usually larger
# still. Tune from the multi_serve_debug.json of your own clips.
PAIR_MIN_ERROR_FRACTION = 0.03
PAIR_MAX_ERROR_FRACTION = 0.10
PAIR_ADAPTIVE_FACTOR = 3.0
# Plausible world volume (m); used only to reject wrong pairings when a frame
# has more than one candidate pair.
PLAUSIBLE_X = (-12.0, 15.0)
PLAUSIBLE_Y = (-8.0, 8.0)
PLAUSIBLE_Z = (-0.5, 8.0)

# ---------------------------------------------------------------------------
# Serve segmentation
# ---------------------------------------------------------------------------
# A pause in detections this long ends a serve (the shuttle landed and is no
# longer tracked, or the next serve has not started).
MAX_GAP_SECONDS = 0.5
# Consecutive 3D points further apart than this speed are a tracking jump,
# not a shuttle (serves are well below this; smashes are not serves).
MAX_SPEED_MPS = 80.0
# Floor contact (same value as evaluator.py's landing rule) and the height the
# shuttle must have reached before a floor contact counts as landing.
FLOOR_EPS_M = 0.15
APEX_MIN_M = 0.50
# After a landing, a new serve begins when the shuttle rises above this.
LAUNCH_Z_M = 0.50
# A segment is only a serve if it has enough points and actually travelled.
MIN_SEGMENT_POINTS = 6
MIN_SEGMENT_SPAN_M = 1.0


# ===========================================================================
# 1. Static detections
# ===========================================================================
def flag_static_detections(per_frame, frame_diag_px, fps,
                           radius_frac=STATIC_RADIUS_FRACTION,
                           min_seconds=STATIC_MIN_SECONDS):
    """
    per_frame : list with one entry per frame; each entry is a list of
                (x, y) or (x, y, conf) candidates in pixels.
    Returns a list of the same shape holding True for a candidate that sits
    on a spot detected in at least `min_seconds` of frames, else False.
    """
    flat_xy, flat_frame = [], []
    for fi, cands in enumerate(per_frame):
        for c in cands:
            flat_xy.append((float(c[0]), float(c[1])))
            flat_frame.append(fi)

    flags = [[False] * len(c) for c in per_frame]
    if not flat_xy:
        return flags

    xy = np.asarray(flat_xy, dtype=np.float64)
    frames = np.asarray(flat_frame, dtype=np.int64)
    radius = max(1.0, radius_frac * float(frame_diag_px))
    min_frames = max(STATIC_MIN_FRAMES_FLOOR, int(round(min_seconds * float(fps))))

    from scipy.spatial import cKDTree
    neighbours = cKDTree(xy).query_ball_point(xy, r=radius)

    static_flat = np.zeros(len(xy), dtype=bool)
    for i, idx in enumerate(neighbours):
        # distinct FRAMES, so two overlapping boxes in one frame count once
        if len(np.unique(frames[idx])) >= min_frames:
            static_flat[i] = True

    k = 0
    for fi, cands in enumerate(per_frame):
        for j in range(len(cands)):
            flags[fi][j] = bool(static_flat[k])
            k += 1
    return flags


# ===========================================================================
# 2. Triangulation helpers + choosing the camera pair
# ===========================================================================
def triangulate_dlt(P_side, P_back, pt_side, pt_back):
    """DLT triangulation of one side/back pixel pair -> (X, Y, Z)."""
    u1, v1 = pt_side[0], pt_side[1]
    u2, v2 = pt_back[0], pt_back[1]
    A = np.zeros((4, 4), dtype=np.float64)
    A[0] = u1 * P_side[2] - P_side[0]
    A[1] = v1 * P_side[2] - P_side[1]
    A[2] = u2 * P_back[2] - P_back[0]
    A[3] = v2 * P_back[2] - P_back[1]
    _, _, Vh = np.linalg.svd(A)
    X = Vh[-1]
    return X[:3] / X[3]


def project(P, X3):
    """Project a 3D point with a 3x4 matrix -> (u, v)."""
    h = P @ np.append(np.asarray(X3, dtype=np.float64), 1.0)
    return h[:2] / h[2]


def reprojection_error(P_side, P_back, pt_side, pt_back, X3):
    """Mean pixel distance between the detections and the reprojected point
    (raw pixels; see error_fraction() for the resolution-independent version)."""
    e1 = np.linalg.norm(project(P_side, X3) - np.asarray(pt_side[:2], dtype=np.float64))
    e2 = np.linalg.norm(project(P_back, X3) - np.asarray(pt_back[:2], dtype=np.float64))
    return float((e1 + e2) / 2.0)


def error_fraction(P_side, P_back, pt_side, pt_back, X3, side_diag_px, back_diag_px):
    """Reprojection error as a fraction of each camera's own frame diagonal, averaged."""
    e1 = np.linalg.norm(project(P_side, X3) - np.asarray(pt_side[:2], dtype=np.float64))
    e2 = np.linalg.norm(project(P_back, X3) - np.asarray(pt_back[:2], dtype=np.float64))
    return float((e1 / side_diag_px + e2 / back_diag_px) / 2.0)


def triangulate_refined(P_side, P_back, pt_side, pt_back, side_diag_px, back_diag_px):
    """
    Triangulate one side/back pixel pair by minimising the pixel reprojection
    error in both images, each normalised by its own frame diagonal
    (a few Levenberg-Marquardt steps starting from the DLT solution).

    The plain DLT minimises an *algebraic* error in which each camera is
    weighted by the scale of its projection matrix and the point's depth, so
    one camera can dominate the other. On a real clip it left a median 158 px
    error in the side view and 2 px in the back view for the same point.
    Returns (X3, error_fraction).
    """
    from scipy.optimize import least_squares
    p1 = np.asarray(pt_side[:2], dtype=np.float64)
    p2 = np.asarray(pt_back[:2], dtype=np.float64)
    X0 = triangulate_dlt(P_side, P_back, pt_side, pt_back)
    if not np.all(np.isfinite(X0)):
        return X0, float("inf")

    def residual(X):
        return np.concatenate([
            (project(P_side, X) - p1) / side_diag_px,
            (project(P_back, X) - p2) / back_diag_px,
        ])

    try:
        X = least_squares(residual, X0, method="lm").x
    except Exception:                      # numerical trouble: fall back to the DLT point
        X = X0
    if not np.all(np.isfinite(X)):
        X = X0
    return X, error_fraction(P_side, P_back, pt_side, pt_back, X, side_diag_px, back_diag_px)


def _plausible(X3):
    return (PLAUSIBLE_X[0] <= X3[0] <= PLAUSIBLE_X[1]
            and PLAUSIBLE_Y[0] <= X3[1] <= PLAUSIBLE_Y[1]
            and PLAUSIBLE_Z[0] <= X3[2] <= PLAUSIBLE_Z[1])


def choose_best_pair(cands_side, cands_back, P_side, P_back,
                     side_diag_px, back_diag_px, max_error=PAIR_MAX_ERROR_FRACTION):
    """
    Pick the (side, back) candidate pair that is the same physical shuttle.

    Returns (X3, error_fraction, ambiguous, i_side, i_back) or None, where
    error_fraction is the per-camera-normalised reprojection error (see
    error_fraction()) and the last two items are the indices of the chosen
    candidates.
      * exactly one candidate in each view -> that pair (ambiguous=False);
      * several candidates -> every pair is triangulated and the one with the
        smallest error (and a plausible position) is chosen (ambiguous=True).
    In both cases the pair is rejected if its error exceeds `max_error`
    (pass float("inf") to get the best pair whatever its error).
    """
    if not cands_side or not cands_back:
        return None

    if len(cands_side) == 1 and len(cands_back) == 1:
        X3, err = triangulate_refined(
            P_side, P_back, cands_side[0], cands_back[0], side_diag_px, back_diag_px
        )
        if err > max_error:
            return None
        return X3, err, False, 0, 0

    best = None
    for i, cs in enumerate(cands_side):
        for j, cb in enumerate(cands_back):
            X3, err = triangulate_refined(P_side, P_back, cs, cb, side_diag_px, back_diag_px)
            if not _plausible(X3):
                continue
            if best is None or err < best[1]:
                best = (X3, err, i, j)
    if best is None or best[1] > max_error:
        return None
    return best[0], best[1], True, best[2], best[3]


def adaptive_pair_limit(errors):
    """Accepted pair error for a clip, from the errors of its best pairs (see the constants)."""
    if len(errors) == 0:
        return PAIR_MAX_ERROR_FRACTION
    return float(np.clip(PAIR_ADAPTIVE_FACTOR * float(np.median(errors)),
                         PAIR_MIN_ERROR_FRACTION, PAIR_MAX_ERROR_FRACTION))


def select_pairs(side_cands, back_cands, side_static, back_static,
                 P_side, P_back, side_diag_px, back_diag_px):
    """
    Choose the shuttle pair in every frame of a clip.

    side_cands / back_cands : per frame, a list of (x, y, ...) candidates
    side_static / back_static : per frame, a list of bools (True = resting)
    Returns (pairs, info):
      pairs -- one entry per frame: None, or a dict with X3, err (fraction),
               ambiguous, i_side, i_back (indices into the FULL candidate lists)
      info  -- frames_both_moving, rejected, ambiguous, median_error, limit
    Pass 1 takes the best pair in every frame without any limit; the limit is then
    derived from those errors (adaptive_pair_limit) and applied in pass 2.
    """
    n = len(side_cands)
    best = [None] * n
    for f in range(n):
        idx_s = [j for j, st in enumerate(side_static[f]) if not st]
        idx_b = [j for j, st in enumerate(back_static[f]) if not st]
        if not idx_s or not idx_b:
            continue
        ch = choose_best_pair([side_cands[f][j] for j in idx_s], [back_cands[f][j] for j in idx_b],
                              P_side, P_back, side_diag_px, back_diag_px, max_error=float("inf"))
        if ch is not None:
            best[f] = {"X3": ch[0], "err": ch[1], "ambiguous": ch[2],
                       "i_side": idx_s[ch[3]], "i_back": idx_b[ch[4]]}

    errors = [b["err"] for b in best if b is not None]
    limit = adaptive_pair_limit(errors)
    pairs, rejected, ambiguous = [], 0, 0
    for b in best:
        if b is None or b["err"] > limit:
            rejected += int(b is not None)
            pairs.append(None)
        else:
            ambiguous += int(b["ambiguous"])
            pairs.append(b)
    both = sum(1 for f in range(n) if any(not st for st in side_static[f]) and any(not st for st in back_static[f]))
    return pairs, {
        "frames_both_moving": both, "rejected": rejected, "ambiguous": ambiguous,
        "median_error": float(np.median(errors)) if errors else 0.0, "limit": limit,
    }


# ===========================================================================
# 3. Splitting a trajectory into serves
# ===========================================================================
def _xyz(p):
    return np.array([float(p["X"]), float(p["Y"]), float(p["Z"])])


def _drop_spikes(points, fps, max_speed):
    """
    Remove single-frame tracking spikes: a point far from BOTH neighbours
    while the neighbours are close to each other. Without this a lone bad
    triangulation would cut a real serve in two.
    """
    if len(points) < 3:
        return points
    kept = [points[0]]
    for i in range(1, len(points) - 1):
        prev, cur, nxt = kept[-1], points[i], points[i + 1]

        def speed(a, b):
            dt = max(1, int(b["frame"]) - int(a["frame"])) / fps
            return float(np.linalg.norm(_xyz(b) - _xyz(a))) / dt

        if (speed(prev, cur) > max_speed and speed(cur, nxt) > max_speed
                and speed(prev, nxt) <= max_speed):
            continue
        kept.append(cur)
    kept.append(points[-1])
    return kept


def _span_m(points):
    xyz = np.array([_xyz(p) for p in points])
    return float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))


def split_serves(trajectory, fps,
                 max_gap_seconds=MAX_GAP_SECONDS, max_speed_mps=MAX_SPEED_MPS,
                 floor_eps=FLOOR_EPS_M, apex_min=APEX_MIN_M, launch_z=LAUNCH_Z_M,
                 min_points=MIN_SEGMENT_POINTS, min_span_m=MIN_SEGMENT_SPAN_M):
    """
    Split a triangulated trajectory (list of {'frame','X','Y','Z'}) into serves.

    Returns (serves, discarded):
      serves    -- list of dicts: serve_index (1-based), points, start_frame,
                   end_frame, start_s, end_s, complete (True when a floor
                   contact after the apex was seen, False when the track just
                   ended, e.g. the shuttle was lost near the floor)
      discarded -- list of {start_frame, end_frame, points, reason} for
                   fragments that were not serves
    """
    fps = float(fps) if fps and fps > 0 else 30.0
    pts = sorted(trajectory, key=lambda p: int(p["frame"]))
    if not pts:
        return [], []

    # --- 1. runs of continuous tracking (gap in frames or an impossible jump)
    max_gap = max(2, int(round(max_gap_seconds * fps)))
    pts = _drop_spikes(pts, fps, max_speed_mps)
    runs, cur = [], [pts[0]]
    for prev, p in zip(pts, pts[1:]):
        gap = int(p["frame"]) - int(prev["frame"])
        dt = max(1, gap) / fps
        jump = float(np.linalg.norm(_xyz(p) - _xyz(prev))) / dt > max_speed_mps
        if gap > max_gap or jump:
            runs.append(cur)
            cur = []
        cur.append(p)
    runs.append(cur)

    # --- 2. inside a run: a serve starts when the shuttle is above launch_z
    #        (anything lower is a shuttle resting or rolling on the floor),
    #        and ends at floor contact after the apex; the next serve starts
    #        when the shuttle rises again
    raw_segments = []   # (points, complete)
    for run in runs:
        seg, max_z, landed = [], -np.inf, True      # True = waiting for a launch
        for p in run:
            z = float(p["Z"])
            if landed:
                if z >= launch_z:               # a serve is under way
                    seg, max_z, landed = [p], z, False
                continue                         # still resting / rolling on the floor
            seg.append(p)
            max_z = max(max_z, z)
            if max_z >= apex_min and z <= floor_eps:
                raw_segments.append((seg, True))
                seg, max_z, landed = [], -np.inf, True
        if seg:
            raw_segments.append((seg, False))

    # --- 3. keep only real serves
    serves, discarded = [], []
    for seg, complete in raw_segments:
        first, last = int(seg[0]["frame"]), int(seg[-1]["frame"])
        if len(seg) < min_points:
            discarded.append({"start_frame": first, "end_frame": last,
                              "points": len(seg), "reason": "too few points"})
            continue
        if _span_m(seg) < min_span_m:
            discarded.append({"start_frame": first, "end_frame": last,
                              "points": len(seg), "reason": "did not move (resting shuttle?)"})
            continue
        serves.append({
            "serve_index": len(serves) + 1,
            "points": seg,
            "start_frame": first,
            "end_frame": last,
            "start_s": first / fps,
            "end_s": last / fps,
            "complete": complete,
        })
    return serves, discarded