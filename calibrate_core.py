import glob
import json
import os
import time

import cv2
import numpy as np

# -----------------------------------------------------------------------
# 3D GCP coordinates (right-side court, origin at net center, floor level)
# -----------------------------------------------------------------------
WORLD_GCPS = np.array([
    [0.00,  0.00, 1.524],  # GCP0: center of net
    [0.00, -2.59, 1.550],  # GCP1: left net/post side
    [0.00,  2.59, 1.550],  # GCP2: right net/post side
    [1.98,  0.00, 0.000],  # GCP3: short service line / center
    [1.98,  2.59, 0.000],  # GCP4: short service line / right sideline
    [6.70,  0.00, 0.000],  # GCP5: back boundary / center
    [6.70,  2.59, 0.000],  # GCP6: back boundary / right sideline
], dtype=np.float64)

NUM_GCPS = len(WORLD_GCPS)

# Human-readable role of each GCP, in click order. Shared by the CLI
# (calibrate.py) and the web UI (app.py / calibrate.html) so the
# instructions shown to the user always match WORLD_GCPS above.
GCP_LABELS = [
    "Net -- center tape (1.524 m)",
    "Net -- left post / left sideline (1.550 m)",
    "Net -- right post / right sideline (1.550 m)",
    "Short service line -- center",
    "Short service line -- right sideline",
    "Back boundary -- center",
    "Back boundary -- right sideline",
]

# Camera rig metadata, shared by calibrate.py (CLI) and app.py (web UI)
# so both interfaces describe the same physical setup.
CAMERA_META = {
    "side": {
        "name": "Side Camera",
        "lens": "iPhone 15 Pro Max, Main 24 mm",
        "height_m": 1.56,
    },
    "back": {
        "name": "Back Camera",
        "lens": "iPhone 15 Pro, Ultra Wide 13 mm",
        "height_m": 2.10,
    },
}


# ============================================================
# Normalized DLT
# ============================================================
def normalize_points_2d(points):
    points = np.asarray(points, dtype=np.float64)
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))

    if mean_dist < 1e-12:
        raise ValueError("2D points are degenerate.")

    scale = np.sqrt(2.0) / mean_dist
    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ])

    pts_h = np.hstack([points, np.ones((len(points), 1))])
    norm_h = (T @ pts_h.T).T
    return norm_h[:, :2] / norm_h[:, 2:3], T


def normalize_points_3d(points):
    points = np.asarray(points, dtype=np.float64)
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))

    if mean_dist < 1e-12:
        raise ValueError("3D points are degenerate.")

    scale = np.sqrt(3.0) / mean_dist
    U = np.array([
        [scale, 0.0, 0.0, -scale * centroid[0]],
        [0.0, scale, 0.0, -scale * centroid[1]],
        [0.0, 0.0, scale, -scale * centroid[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])

    pts_h = np.hstack([points, np.ones((len(points), 1))])
    norm_h = (U @ pts_h.T).T
    return norm_h[:, :3] / norm_h[:, 3:4], U


def normalized_dlt(world_points, image_points):
    """Estimate 3x4 projection matrix P from 3D-2D correspondences."""
    x_norm, T = normalize_points_2d(image_points)
    X_norm, U = normalize_points_3d(world_points)

    A = []
    for X, (u, v) in zip(X_norm, x_norm):
        Xh = np.r_[X, 1.0]
        A.append(np.r_[Xh, np.zeros(4), -u * Xh])
        A.append(np.r_[np.zeros(4), Xh, -v * Xh])

    A = np.asarray(A, dtype=np.float64)
    _, singular_values, Vt = np.linalg.svd(A)

    Pn = Vt[-1].reshape(3, 4)

    # Denormalize: x = T^-1 * Pn * U * X
    P = np.linalg.inv(T) @ Pn @ U

    # Normalize for reproducibility.
    P /= np.linalg.norm(P[:3, :3])
    if P[2, 3] < 0:
        P *= -1.0

    return P, singular_values


# ============================================================
# Nonlinear projection refinement
# ============================================================
def project_with_P(P, world_points):
    Xh = np.hstack([
        world_points,
        np.ones((len(world_points), 1), dtype=np.float64)
    ])
    q = (P @ Xh.T).T
    if np.any(np.abs(q[:, 2]) < 1e-12):
        raise ValueError("Projection produced a point at/near infinity.")
    return q[:, :2] / q[:, 2:3]


def refine_projection_matrix(P0, world_points, image_points):
    """Refine P by minimizing pixel reprojection error."""
    try:
        from scipy.optimize import least_squares
    except ImportError:
        print("WARNING: scipy not installed; skipping nonlinear P refinement.")
        return P0

    # Fix P[2,3] = 1 when possible; otherwise use P[2,2] as the fixed scale.
    # We optimize the remaining 11 parameters.
    P0 = P0.astype(np.float64).copy()
    scale_index = (2, 3)
    if abs(P0[2, 3]) < 1e-10:
        scale_index = (2, 2)

    scale = P0[scale_index]
    if abs(scale) < 1e-10:
        return P0

    P0 /= scale
    mask = np.ones((3, 4), dtype=bool)
    mask[scale_index] = False
    x0 = P0[mask]

    def unpack(x):
        P = np.zeros((3, 4), dtype=np.float64)
        P[mask] = x
        P[scale_index] = 1.0
        return P

    def residual(x):
        P = unpack(x)
        try:
            proj = project_with_P(P, world_points)
        except ValueError:
            return np.full(image_points.size, 1e6)
        return (proj - image_points).ravel()

    result = least_squares(
        residual,
        x0,
        method="trf",
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=5000,
    )

    P = unpack(result.x)
    P /= np.linalg.norm(P[:3, :3])
    if P[2, 3] < 0:
        P *= -1.0
    return P


# ============================================================
# Camera center and error reporting
# ============================================================
def camera_center(P):
    _, _, Vt = np.linalg.svd(P)
    C = Vt[-1]
    C /= C[3]
    return C[:3]


def reprojection_errors(P, world_points, image_points):
    projected = project_with_P(P, world_points)
    delta = projected - image_points
    errors = np.linalg.norm(delta, axis=1)
    return projected, delta, errors


# ============================================================
# Public entry point (drop-in replacement)
# ============================================================
def solve_camera(img_pts, img_w=None, img_h=None):
    """
    Solve for the 3x4 projection matrix P from the 7 clicked GCP points.

    Same call signature as the old solvePnP-based solve_camera(), so
    existing callers (Flask app, run_pipeline.py) don't need to change.
    img_w / img_h are accepted for compatibility but are not used --
    the DLT solve estimates the effective intrinsics directly from the
    correspondences instead of assuming K.
    """
    img_pts = np.asarray(img_pts, dtype=np.float64)
    if len(img_pts) != NUM_GCPS:
        raise ValueError(
            f"Expected {NUM_GCPS} GCP points, got {len(img_pts)}."
        )

    P_dlt, _ = normalized_dlt(WORLD_GCPS, img_pts)
    P = refine_projection_matrix(P_dlt, WORLD_GCPS, img_pts)
    return P

# ============================================================
# Landmark library -- a small per-camera history of confirmed
# calibrations, used to auto-calibrate future sessions on the same
# (fixed) rig via feature matching instead of re-deriving from
# scratch every time.
# ============================================================
LANDMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_landmarks")
MAX_LANDMARKS_PER_CAMERA = 5

# Minimum RANSAC-inlier evidence before we trust a landmark match over
# falling back to line-detection.
LANDMARK_MIN_INLIERS = 15
LANDMARK_MIN_INLIER_RATIO = 0.5


def _landmark_dir(camera_key):
    d = os.path.join(LANDMARK_DIR, camera_key)
    os.makedirs(d, exist_ok=True)
    return d


def list_landmarks(camera_key):
    """Confirmed calibration landmarks for a camera, oldest first."""
    d = _landmark_dir(camera_key)
    entries = []
    for json_path in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
            jpg_path = json_path[:-5] + ".jpg"
            if os.path.exists(jpg_path):
                entries.append((meta["timestamp"], jpg_path, meta["points"]))
        except (OSError, KeyError, ValueError):
            continue
    entries.sort(key=lambda e: e[0])
    return entries


def save_landmark(image, camera_key, points):
    """
    Store a confirmed calibration frame + its 7 GCP pixel points as a
    landmark for future auto-calibration. Keeps only the most recent
    MAX_LANDMARKS_PER_CAMERA per camera (oldest evicted first).
    """
    d = _landmark_dir(camera_key)
    ts = time.time()
    stamp = f"{int(ts * 1000)}"
    jpg_path = os.path.join(d, f"{stamp}.jpg")
    json_path = os.path.join(d, f"{stamp}.json")

    cv2.imwrite(jpg_path, image)
    with open(json_path, "w") as f:
        json.dump({"timestamp": ts, "points": [list(map(float, p)) for p in points]}, f)

    entries = list_landmarks(camera_key)
    if len(entries) > MAX_LANDMARKS_PER_CAMERA:
        for _, old_jpg, _ in entries[: len(entries) - MAX_LANDMARKS_PER_CAMERA]:
            for path in (old_jpg, old_jpg[:-4] + ".json"):
                try:
                    os.remove(path)
                except OSError:
                    pass


def match_against_landmarks(image, camera_key):
    """
    Try to warp a previous confirmed calibration onto the current frame
    via ORB feature matching + RANSAC homography. This is the strongest
    auto-calibration signal when it works, since it reuses real
    confirmed points (net posts included) rather than re-deriving them
    from scratch. Matches against every stored landmark for this camera
    and keeps the best-scoring one.

    Returns {'points': [(x,y)*7], 'inliers': int, 'inlier_ratio': float}
    or None if nothing matched confidently enough.
    """
    landmarks = list_landmarks(camera_key)
    if not landmarks:
        return None

    orb = cv2.ORB_create(2000)
    kp2, des2 = orb.detectAndCompute(image, None)
    if des2 is None or len(kp2) < 8:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    best = None

    for _, jpg_path, points in landmarks:
        landmark_img = cv2.imread(jpg_path)
        if landmark_img is None:
            continue
        kp1, des1 = orb.detectAndCompute(landmark_img, None)
        if des1 is None or len(kp1) < 8:
            continue

        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            continue

        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            continue

        inliers = int(mask.sum())
        inlier_ratio = inliers / len(good)
        if inliers < LANDMARK_MIN_INLIERS or inlier_ratio < LANDMARK_MIN_INLIER_RATIO:
            continue

        score = inliers * inlier_ratio
        if best is None or score > best["score"]:
            warped = cv2.perspectiveTransform(
                np.array(points, dtype=np.float32).reshape(-1, 1, 2), H
            ).reshape(-1, 2)
            best = {
                "points": [tuple(p) for p in warped],
                "inliers": inliers,
                "inlier_ratio": inlier_ratio,
                "score": score,
            }

    return best


# ============================================================
# Fallback auto-detection: classical line detection for the floor
# GCPs, plus geometric extrapolation for the net GCPs. Used when no
# landmark match is available (first-ever calibration of a camera, or
# the rig moved enough that feature matching can't find it).
# ============================================================
def _white_line_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    # Smaller kernel for the noise-removal OPEN: a 5x5 open erodes a
    # thin (~4px) court-tape line to almost nothing once it's at a
    # steep angle. 3x3 still clears speckle noise without eating lines.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return mask


def _detect_segments(mask):
    min_len = max(30, mask.shape[1] // 12)
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, 60, minLineLength=min_len, maxLineGap=20)
    if segs is None:
        return []
    # cv2.HoughLinesP is documented as returning shape (N, 1, 4), but some
    # opencv-python builds return (N, 4) directly -- reshape(-1)[:4]
    # handles either without caring which one this install uses.
    return [tuple(int(v) for v in np.asarray(s).reshape(-1)[:4]) for s in segs]


def _seg_angle_length(seg):
    x1, y1, x2, y2 = seg
    angle = np.arctan2(y2 - y1, x2 - x1) % np.pi
    length = float(np.hypot(x2 - x1, y2 - y1))
    return angle, length


def _dominant_angle(segs, exclude=None, min_sep=np.deg2rad(35), peak_tol=np.deg2rad(6)):
    """
    Strongest coherent line orientation, mod pi (circular distances, so
    179 deg and 1 deg are 2 deg apart). A mode (peak), not a circular
    mean: with several distinct line families present, an average would
    blend them into a direction matching neither.
    """
    candidates = []
    for seg in segs:
        angle, length = _seg_angle_length(seg)
        if exclude is not None:
            sep = min(abs(angle - exclude), np.pi - abs(angle - exclude))
            if sep < min_sep:
                continue
        candidates.append((angle, length))
    if not candidates:
        return None

    best_angle, best_score = None, -1.0
    for deg in range(180):
        center = np.deg2rad(deg)
        score = sum(
            length for angle, length in candidates
            if min(abs(angle - center), np.pi - abs(angle - center)) < peak_tol
        )
        if score > best_score:
            best_angle, best_score = center, score
    return best_angle


def _fit_line(segs):
    pts = np.array(
        [[x, y] for (x1, y1, x2, y2) in segs for (x, y) in [(x1, y1), (x2, y2)]],
        dtype=np.float32,
    )
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _two_strongest_lines(segs, angle, tol=np.deg2rad(40)):
    """
    Split segments near `angle` into two parallel-ish groups, fit one
    line each. tol is generous (40 deg): two lines that are parallel in
    world space can differ a lot in image angle under perspective.
    """
    family = []
    for seg in segs:
        a, length = _seg_angle_length(seg)
        sep = min(abs(a - angle), np.pi - abs(a - angle))
        if sep < tol:
            family.append((seg, length))
    if not family:
        return []

    segs_only = [s for s, _ in family]

    # Project midpoints onto the normal of the *known* family angle
    # (not a fresh blended fit, which can represent neither line).
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-direction[1], direction[0]])
    mids = np.array([[(s[0] + s[2]) / 2.0, (s[1] + s[3]) / 2.0] for s, _ in family])
    offsets = (mids - mids.mean(axis=0)) @ normal

    if offsets.max() - offsets.min() < 8:  # only one physical line present
        p0f, df = _fit_line(segs_only)
        return [(p0f, df, sum(l for _, l in family))]

    # Split at the largest gap between consecutive offsets.
    order = np.argsort(offsets)
    split = int(np.argmax(np.diff(offsets[order])))
    group_a = [family[order[i]][0] for i in range(split + 1)]
    group_b = [family[order[i]][0] for i in range(split + 1, len(order))]

    lines = []
    for group in (group_a, group_b):
        if not group:
            continue
        pt, dirn = _fit_line(group)
        strength = sum(_seg_angle_length(s)[1] for s in group)
        lines.append((pt, dirn, strength))
    lines.sort(key=lambda t: -t[2])
    return lines[:2]


def _line_intersection(line1, line2):
    (p1, d1, _), (p2, d2, _) = line1, line2
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def _court_floor_region(image):
    """
    Rough mask of the court surface's own color (sampled from the
    lower-middle band of the frame) so floor-line detection ignores the
    net tape / background lines. Follows the real court edge instead of
    an arbitrary top-of-frame crop.
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sample = hsv[int(h * 0.6): int(h * 0.95), int(w * 0.25): int(w * 0.75)].reshape(-1, 3)
    if sample.size == 0:
        return np.full((h, w), 255, dtype=np.uint8)
    med = np.median(sample, axis=0)
    lower = np.clip(med - np.array([15, 60, 60]), 0, 255)
    upper = np.clip(med + np.array([15, 60, 60]), 0, 255)
    region = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel, iterations=2)
    region = cv2.dilate(region, kernel, iterations=1)  # keep tape at the court edge
    return region


def _detect_floor_rectangle(image):
    """Best-effort line-detection guess for GCP3/GCP4/GCP5/GCP6."""
    mask = _white_line_mask(image)
    mask = cv2.bitwise_and(mask, _court_floor_region(image))
    h, w = mask.shape

    result = {3: None, 4: None, 5: None, 6: None}
    segs = _detect_segments(mask)
    if len(segs) < 4:
        return result, mask

    angle_a = _dominant_angle(segs)
    if angle_a is None:
        return result, mask
    angle_b = _dominant_angle(segs, exclude=angle_a)
    if angle_b is None:
        return result, mask

    lines_a = _two_strongest_lines(segs, angle_a)  # "cross" lines: near/far
    lines_b = _two_strongest_lines(segs, angle_b)  # "long" lines: center/sideline
    if len(lines_a) < 2 or len(lines_b) < 2:
        return result, mask

    # Role assignment is a HEURISTIC and depends on camera orientation:
    # larger y = near, smaller x = center. Can come out swapped.
    lines_a.sort(key=lambda l: l[0][1], reverse=True)
    lines_b.sort(key=lambda l: l[0][0])
    near_line, far_line = lines_a[0], lines_a[1]
    center_line, right_line = lines_b[0], lines_b[1]

    candidates = {
        3: _line_intersection(near_line, center_line),
        4: _line_intersection(near_line, right_line),
        5: _line_intersection(far_line, center_line),
        6: _line_intersection(far_line, right_line),
    }
    margin = -0.1 * max(w, h)  # allow a little outside-frame slack
    for idx, pt in candidates.items():
        if pt is None:
            continue
        x, y = pt
        if margin <= x <= w - margin and margin <= y <= h - margin:
            result[idx] = (float(x), float(y))
    return result, mask


def _lift_net_points(image, mask, floor_points):
    """
    Approximate GCP0/GCP1/GCP2 from the floor rectangle by extrapolating
    the center-line/sideline to the net (X=0) and looking for a
    near-vertical post segment at the right-sideline floor point. A
    geometric seed, not a measurement -- expect manual correction,
    especially GCP1 (only mirrored from the right post).
    """
    result = {0: None, 1: None, 2: None}
    p3, p4, p5, p6 = (floor_points.get(i) for i in (3, 4, 5, 6))
    if None in (p3, p4, p5, p6):
        return result

    p3, p4, p5, p6 = (np.array(p, dtype=np.float64) for p in (p3, p4, p5, p6))
    # p3/p4 at world X=1.98, p5/p6 at X=6.70 -> parametric t where X=0.
    t0 = (0.0 - 1.98) / (6.70 - 1.98)
    center_floor = p3 + t0 * (p5 - p3)
    right_floor = p4 + t0 * (p6 - p4)

    h, w = mask.shape
    x, y = right_floor
    pad = int(0.12 * w)
    x0, x1 = int(max(0, x - pad)), int(min(w, x + pad))
    y0, y1 = int(max(0, y - int(0.35 * h))), int(min(h, y + int(0.1 * h)))

    region_segs = []
    if x1 > x0 and y1 > y0:
        region_mask = np.zeros_like(mask)
        region_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        region_segs = [
            s for s in _detect_segments(region_mask)
            if abs(_seg_angle_length(s)[0] - np.pi / 2) < np.deg2rad(25)
        ]

    if region_segs:
        best = max(region_segs, key=lambda s: _seg_angle_length(s)[1])
        sx1, sy1, sx2, sy2 = best
        top = (sx1, sy1) if sy1 < sy2 else (sx2, sy2)
        right_net = np.array(top, dtype=np.float64)
    else:
        # No post found: rough vertical scale from the floor rectangle.
        floor_span_px = np.linalg.norm(p6 - p3)
        floor_span_m = np.hypot(6.70 - 1.98, 2.59)
        px_per_m = floor_span_px / floor_span_m if floor_span_m > 0 else 0.0
        right_net = right_floor - np.array([0.0, px_per_m * 1.550])

    vertical_offset = right_net - right_floor
    left_floor = 2 * center_floor - right_floor  # mirror across center line
    left_net = left_floor + vertical_offset
    center_net = (left_net + right_net) / 2.0

    result[0] = tuple(center_net)
    result[1] = tuple(left_net)
    result[2] = tuple(right_net)
    return result


def auto_detect_gcps(image, camera_key):
    """
    Best-effort initial guess for all 7 GCPs, to seed a draggable-marker
    UI. Order: (1) match a previously confirmed landmark calibration,
    (2) line detection for GCP3-6 + geometric lift for GCP0-2,
    (3) generic fallback layout so every point has a draggable position.

    Returns (points, confidence, source):
      points     -- {idx: (x, y)}, all 7 keys, plain Python floats
      confidence -- {idx: "high" | "low"} ("low" = UI flags for review)
      source     -- short string for the status bar
    """
    h, w = image.shape[:2]

    match = match_against_landmarks(image, camera_key)
    if match is not None:
        points = {i: (float(match["points"][i][0]), float(match["points"][i][1]))
                  for i in range(NUM_GCPS)}
        confidence = {i: "high" for i in range(NUM_GCPS)}
        source = (
            f"matched a previous confirmed calibration "
            f"({match['inliers']} inliers, {match['inlier_ratio']:.0%} inlier ratio)"
        )
        return points, confidence, source

    floor_points, mask = _detect_floor_rectangle(image)
    net_points = _lift_net_points(image, mask, floor_points)

    fallback = {
        0: (0.50 * w, 0.35 * h), 1: (0.30 * w, 0.38 * h), 2: (0.70 * w, 0.38 * h),
        3: (0.45 * w, 0.55 * h), 4: (0.65 * w, 0.55 * h),
        5: (0.40 * w, 0.85 * h), 6: (0.75 * w, 0.85 * h),
    }

    points, confidence = {}, {}
    for i in range(NUM_GCPS):
        if i in (3, 4, 5, 6):
            guess, is_high = floor_points.get(i), floor_points.get(i) is not None
        else:
            guess, is_high = net_points.get(i), False  # net lift is always "low"
        if guess is not None:
            points[i] = (float(guess[0]), float(guess[1]))
            confidence[i] = "high" if is_high else "low"
        else:
            points[i] = (float(fallback[i][0]), float(fallback[i][1]))
            confidence[i] = "low"

    found_floor = sum(1 for i in (3, 4, 5, 6) if confidence[i] == "high")
    source = (
        f"line detection: {found_floor}/4 floor corners located; "
        f"net points and any missing floor corners are approximate -- please check"
    )
    return points, confidence, source