import cv2
import numpy as np
import os

# ============================================================
# BADMINTON 2-CAMERA COURT CALIBRATION
# Revised for:
#   SIDE  : iPhone 15 Pro Max, Main 24 mm, height 1.56 m
#   BACK  : iPhone 15 Pro, Ultra Wide 13 mm, height 2.10 m
#   Both  : 1920 x 1080, fixed cameras, slow motion
#
# Main change from the old version:
#   Do NOT assume one fake focal length for both cameras and then
#   use solvePnP.  Instead, estimate each camera's 3x4 projection
#   matrix directly from the 7 measured 3D<->2D GCP correspondences
#   using normalized DLT, then refine it by minimizing reprojection
#   error.  This is much better suited to the current 7-point setup.
#
# GCP roles:
#   GCP0-GCP2 = net-height references
#   GCP3-GCP6 = floor points and serve-target references
#
# Coordinate system:
#   X = court length, from net toward the opponent back boundary
#   Y = court width, center line = 0, right singles sideline = +2.59
#   Z = height above floor
#
# IMPORTANT:
#   The projection matrices produced here are intended for the
#   current fixed-camera setup. For highest accuracy, especially
#   with the 13 mm ultra-wide camera, a separate checkerboard/Charuco
#   intrinsic calibration is still recommended later.
# ============================================================

# -----------------------------
# Input images
# -----------------------------
SIDE_IMAGE = "preprocessing/frames/matched_120fps/side_take1/pair_0001_side.png"
BACK_IMAGE = "preprocessing/frames/matched_120fps/back_take1/pair_0001_back.png"

OUTPUT_FILE = "court_calibration.npz"
OUTPUT_DIR = "outputs/calibration_revised"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Camera metadata
# -----------------------------
CAMERAS = {
    "side": {
        "image": SIDE_IMAGE,
        "name": "Side Camera",
        "height_m": 1.56,
        "lens": "iPhone 15 Pro Max Main 24 mm",
    },
    "back": {
        "image": BACK_IMAGE,
        "name": "Back Camera",
        "height_m": 2.10,
        "lens": "iPhone 15 Pro Ultra Wide 13 mm",
    },
}

# -----------------------------
# 3D GCP coordinates
# -----------------------------
# Origin = center of net at floor level.
#
# Net:
#   GCP0 = center of net tape: 1.524 m
#   GCP1/GCP2 = near the net posts: 1.550 m
#
# Floor:
#   GCP3 = short service line / center line
#   GCP4 = short service line / right singles sideline
#   GCP5 = back boundary / center line
#   GCP6 = back boundary / right singles sideline
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
# Interactive clicking
# ============================================================
def collect_clicks(image, window_name):
    clicked = []
    base = image.copy()
    display = base.copy()

    instructions = (
        "Click GCP0 -> GCP6 | R=reset | ENTER=finish | ESC=cancel"
    )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    def redraw():
        nonlocal display
        display = base.copy()

        cv2.rectangle(
            display, (0, 0), (display.shape[1], 48),
            (0, 0, 0), -1
        )
        cv2.putText(
            display, instructions, (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        for i, (x, y) in enumerate(clicked):
            cv2.circle(display, (int(x), int(y)), 8, (0, 255, 0), -1)
            cv2.putText(
                display, f"GCP{i}",
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0), 2, cv2.LINE_AA
            )

        cv2.imshow(window_name, display)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < NUM_GCPS:
            clicked.append((float(x), float(y)))
            print(
                f"{window_name}: GCP{len(clicked)-1} "
                f"= ({x}, {y})"
            )
            redraw()

    cv2.setMouseCallback(window_name, mouse_callback)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            clicked.clear()
            print(f"{window_name}: reset")
            redraw()

        elif key in (10, 13):
            if len(clicked) == NUM_GCPS:
                break
            print(
                f"Need all {NUM_GCPS} points. "
                f"Currently {len(clicked)}/{NUM_GCPS}."
            )

        elif key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Calibration cancelled.")

    cv2.destroyWindow(window_name)
    return np.asarray(clicked, dtype=np.float64)


# ============================================================
# Visualization
# ============================================================
def save_comparison(
    image,
    clicked,
    projected,
    errors,
    camera_name,
    output_path,
):
    result = image.copy()

    cv2.rectangle(
        result, (0, 0), (result.shape[1], 55),
        (0, 0, 0), -1
    )
    cv2.putText(
        result,
        f"{camera_name.upper()} - REVISED DLT CALIBRATION",
        (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for i in range(NUM_GCPS):
        cx, cy = np.round(clicked[i]).astype(int)
        px, py = np.round(projected[i]).astype(int)

        # Error vector
        cv2.line(
            result, (cx, cy), (px, py),
            (0, 255, 255), 2, cv2.LINE_AA
        )

        # GREEN = clicked
        cv2.circle(result, (cx, cy), 9, (0, 255, 0), -1)

        # RED = projected
        cv2.circle(result, (px, py), 8, (0, 0, 255), -1)

        cv2.putText(
            result,
            f"GCP{i} click",
            (cx + 10, cy + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            result,
            f"GCP{i} proj {errors[i]:.1f}px",
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255), 2, cv2.LINE_AA
        )

    # Legend
    y = 82
    cv2.circle(result, (20, y), 7, (0, 255, 0), -1)
    cv2.putText(result, "GREEN = clicked", (35, y + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(result, (220, y), 7, (0, 0, 255), -1)
    cv2.putText(result, "RED = projected", (235, y + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(result, (430, y), (470, y), (0, 255, 255), 2)
    cv2.putText(result, "YELLOW = error", (480, y + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, result)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    results = {}

    for camera_key in ("side", "back"):
        cfg = CAMERAS[camera_key]
        image = cv2.imread(cfg["image"])

        if image is None:
            raise FileNotFoundError(
                f"Could not read {cfg['image']}. "
                "Update SIDE_IMAGE/BACK_IMAGE paths if needed."
            )

        h, w = image.shape[:2]
        if (w, h) != (1920, 1080):
            print(
                f"WARNING: {cfg['name']} image is {w}x{h}, "
                "not the expected 1920x1080."
            )

        print("\n" + "=" * 70)
        print(f"{cfg['name']}")
        print(f"Lens: {cfg['lens']}")
        print(f"Known camera height: {cfg['height_m']:.2f} m")
        print("Click GCP0 -> GCP6 in exactly the same physical locations")
        print("as shown in the calibration image.")
        print("=" * 70)

        clicked = collect_clicks(
            image,
            f"{cfg['name']} - Revised Calibration"
        )

        # Initial normalized DLT
        P_dlt, sv = normalized_dlt(WORLD_GCPS, clicked)

        # Nonlinear refinement
        P = refine_projection_matrix(P_dlt, WORLD_GCPS, clicked)

        projected, delta, errors = reprojection_errors(
            P, WORLD_GCPS, clicked
        )

        C = camera_center(P)
        mean_error = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        max_idx = int(np.argmax(errors))

        print(f"\n{cfg['name']} results")
        print("-" * 70)
        print(f"Camera center (X,Y,Z): {C[0]:.4f}, {C[1]:.4f}, {C[2]:.4f} m")
        print(f"Expected camera height : {cfg['height_m']:.4f} m")
        print(f"Height difference      : {C[2] - cfg['height_m']:+.4f} m")
        print()
        print(
            f"{'GCP':<6}{'Clicked':<22}{'Projected':<22}"
            f"{'dx':>9}{'dy':>9}{'Error(px)':>12}"
        )
        print("-" * 70)

        for i in range(NUM_GCPS):
            print(
                f"GCP{i:<2} "
                f"({clicked[i,0]:7.2f},{clicked[i,1]:7.2f})   "
                f"({projected[i,0]:7.2f},{projected[i,1]:7.2f})   "
                f"{delta[i,0]:9.2f}"
                f"{delta[i,1]:9.2f}"
                f"{errors[i]:12.3f}"
            )

        print("-" * 70)
        print(f"Mean pixel error : {mean_error:.4f} px")
        print(f"RMSE             : {rmse:.4f} px")
        print(
            f"Max error        : {errors[max_idx]:.4f} px "
            f"(GCP{max_idx})"
        )

        output_image = os.path.join(
            OUTPUT_DIR,
            f"{camera_key}_revised_calibration_check.jpg"
        )
        save_comparison(
            image,
            clicked,
            projected,
            errors,
            cfg["name"],
            output_image,
        )
        print(f"Saved: {output_image}")

        results[camera_key] = {
            "P": P,
            "clicked": clicked,
            "projected": projected,
            "errors": errors,
            "camera_center": C,
            "mean_error": mean_error,
            "rmse": rmse,
            "max_error": float(errors[max_idx]),
            "max_gcp": max_idx,
            "image_size": (w, h),
            "camera_height_m": cfg["height_m"],
            "lens": cfg["lens"],
        }

    # --------------------------------------------------------
    # Save calibration
    # --------------------------------------------------------
    np.savez(
        OUTPUT_FILE,
        P_side=results["side"]["P"],
        P_back=results["back"]["P"],
        WORLD_GCPs=WORLD_GCPS.astype(np.float32),
        clicked_side=results["side"]["clicked"].astype(np.float32),
        clicked_back=results["back"]["clicked"].astype(np.float32),
        projected_side=results["side"]["projected"].astype(np.float32),
        projected_back=results["back"]["projected"].astype(np.float32),
        side_camera_center=results["side"]["camera_center"].astype(np.float32),
        back_camera_center=results["back"]["camera_center"].astype(np.float32),
        side_camera_height_m=np.float32(1.56),
        back_camera_height_m=np.float32(2.10),
        image_width=np.int32(1920),
        image_height=np.int32(1080),
    )

    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"Saved calibration: {OUTPUT_FILE}")
    print()
    print(
        f"SIDE: mean={results['side']['mean_error']:.3f}px, "
        f"RMSE={results['side']['rmse']:.3f}px, "
        f"max={results['side']['max_error']:.3f}px"
    )
    print(
        f"BACK: mean={results['back']['mean_error']:.3f}px, "
        f"RMSE={results['back']['rmse']:.3f}px, "
        f"max={results['back']['max_error']:.3f}px"
    )
    print()
    print("The saved P_side/P_back matrices can be used directly for")
    print("3D triangulation with the same fixed camera positions.")
    print("For production-level accuracy, calibrate lens intrinsics")
    print("separately with a checkerboard/Charuco target, especially")
    print("for the 13 mm ultra-wide back camera.")
