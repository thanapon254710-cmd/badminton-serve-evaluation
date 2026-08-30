"""
calibrate_core.py

Camera calibration core, revised to use normalized DLT + nonlinear
reprojection refinement instead of an assumed focal length + solvePnP.

The old solve_camera() assumed K with focal_length = image_width and zero
distortion, then solved only for R, t via solvePnP. That guessed K was the
source of the systematic reprojection error (worse at distance / frame
edges). This version solves the full 3x4 projection matrix P directly
from the 7 GCP correspondences via normalized DLT, then refines it with
least-squares to minimize pixel reprojection error -- no K guess needed.

solve_camera(img_pts, img_w, img_h) keeps the same call signature used
elsewhere in the pipeline (Flask app, run_pipeline.py) so this is a
drop-in replacement. img_w / img_h are accepted but unused.
"""

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