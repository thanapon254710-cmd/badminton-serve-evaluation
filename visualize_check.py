import cv2
import numpy as np
import csv
import os

# ============================================================
# GCP REPROJECTION CHECK
#
# Purpose:
#   1. Click the 7 GCPs in each camera image.
#   2. Project the known 3D GCP coordinates into each image.
#   3. Compare CLICKED pixels vs PROJECTED pixels.
#   4. Calculate exact pixel reprojection error for every GCP.
#   5. Save annotated images + CSV error report.
#
# Coordinate convention:
#   X = court width direction
#   Y = court length direction
#   Z = vertical height
#
# Your existing world GCP coordinates are kept unchanged.
# ============================================================

# -----------------------------
# Input files
# -----------------------------
SIDE_IMAGE = "preprocessing/frames/matched_120fps/side_take1/pair_0001_side.png"
BACK_IMAGE = "preprocessing/frames/matched_120fps/back_take1/pair_0001_back.png"
CALIBRATION_FILE = "court_calibration.npz"

# Output folder
OUTPUT_DIR = "outputs/gcp_reprojection_check"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Known 3D world coordinates
# -----------------------------
WORLD_GCPs = np.array([
    [0.00,  0.00, 1.55],  # GCP0
    [0.00, -2.59, 1.55],  # GCP1
    [0.00,  2.59, 1.55],  # GCP2
    [1.98,  0.00, 0.00],  # GCP3
    [1.98,  2.59, 0.00],  # GCP4
    [6.70,  0.00, 0.00],  # GCP5
    [6.70,  2.59, 0.00],  # GCP6
], dtype=np.float32)

NUM_GCPS = len(WORLD_GCPs)

# -----------------------------
# Load images and calibration
# -----------------------------
img_side = cv2.imread(SIDE_IMAGE)
img_back = cv2.imread(BACK_IMAGE)

if img_side is None:
    raise FileNotFoundError(f"Could not read side image: {SIDE_IMAGE}")

if img_back is None:
    raise FileNotFoundError(f"Could not read back image: {BACK_IMAGE}")

data = np.load(CALIBRATION_FILE)

if "P_side" not in data or "P_back" not in data:
    raise KeyError(
        f"{CALIBRATION_FILE} must contain 'P_side' and 'P_back'."
    )

P_side = data["P_side"]
P_back = data["P_back"]

# -----------------------------
# Project 3D world points -> pixels
# -----------------------------
def project_points(P, world_points):
    """
    Project Nx3 world points into image pixels using a 3x4
    camera projection matrix P.
    """
    world_h = np.hstack([
        world_points,
        np.ones((len(world_points), 1), dtype=np.float32)
    ])

    projected_h = (P @ world_h.T).T

    # Avoid division by zero
    z = projected_h[:, 2:3]
    projected_2d = projected_h[:, :2] / z

    return projected_2d


projected_side = project_points(P_side, WORLD_GCPs)
projected_back = project_points(P_back, WORLD_GCPs)


# ============================================================
# Interactive GCP clicking
# ============================================================

def collect_clicks(image, window_name):
    """
    User clicks GCP0 ... GCP6 in order.
    Returns Nx2 array of pixel coordinates.
    """
    display = image.copy()
    clicked = []

    instructions = (
        "Click GCP0 -> GCP6 in order | "
        "R = reset | ENTER = finish"
    )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    def redraw():
        nonlocal display

        display = image.copy()

        # Instruction text
        cv2.rectangle(display, (0, 0), (display.shape[1], 45),
                      (0, 0, 0), -1)
        cv2.putText(
            display,
            instructions,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Draw already-clicked points
        for i, (x, y) in enumerate(clicked):
            cv2.circle(display, (int(x), int(y)), 7, (0, 255, 0), -1)
            cv2.putText(
                display,
                f"GCP{i}",
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow(window_name, display)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked) < NUM_GCPS:
                clicked.append((float(x), float(y)))
                print(
                    f"{window_name}: GCP{len(clicked)-1} "
                    f"clicked at ({x}, {y})"
                )
                redraw()

    cv2.setMouseCallback(window_name, mouse_callback)

    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        # Reset clicks
        if key == ord("r"):
            clicked.clear()
            print(f"{window_name}: reset")
            redraw()

        # Finish only after all 7 points are selected
        elif key in (13, 10):  # ENTER
            if len(clicked) == NUM_GCPS:
                break
            print(
                f"{window_name}: please click all "
                f"{NUM_GCPS} GCPs first "
                f"({len(clicked)}/{NUM_GCPS})."
            )

        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled by user.")

    cv2.destroyWindow(window_name)

    return np.array(clicked, dtype=np.float32)


print("\n==============================================")
print("SIDE CAMERA: click GCP0 -> GCP6")
print("==============================================")
clicked_side = collect_clicks(img_side, "SIDE CAMERA - Click GCPs")

print("\n==============================================")
print("BACK CAMERA: click GCP0 -> GCP6")
print("==============================================")
clicked_back = collect_clicks(img_back, "BACK CAMERA - Click GCPs")


# ============================================================
# Error calculation
# ============================================================

def calculate_errors(clicked, projected):
    """
    Calculate dx, dy and Euclidean pixel error.
    """
    delta = projected - clicked

    dx = delta[:, 0]
    dy = delta[:, 1]

    error = np.sqrt(dx**2 + dy**2)

    return dx, dy, error


side_dx, side_dy, side_error = calculate_errors(
    clicked_side, projected_side
)

back_dx, back_dy, back_error = calculate_errors(
    clicked_back, projected_back
)


# ============================================================
# Print numerical report
# ============================================================

def print_report(label, clicked, projected, dx, dy, error):
    print("\n")
    print("=" * 78)
    print(f"{label} REPROJECTION ERROR")
    print("=" * 78)

    print(
        f"{'GCP':<6}"
        f"{'Clicked (u,v)':<22}"
        f"{'Projected (u,v)':<24}"
        f"{'dx':>8}"
        f"{'dy':>8}"
        f"{'Error(px)':>12}"
    )

    print("-" * 78)

    for i in range(NUM_GCPS):
        print(
            f"GCP{i:<2} "
            f"({clicked[i,0]:7.2f}, {clicked[i,1]:7.2f})   "
            f"({projected[i,0]:7.2f}, {projected[i,1]:7.2f})   "
            f"{dx[i]:8.2f}"
            f"{dy[i]:8.2f}"
            f"{error[i]:12.2f}"
        )

    rmse = np.sqrt(np.mean(error ** 2))
    mean_error = np.mean(error)
    max_error = np.max(error)
    max_index = int(np.argmax(error))

    print("-" * 78)
    print(f"Mean error : {mean_error:.3f} px")
    print(f"RMSE       : {rmse:.3f} px")
    print(
        f"Max error  : {max_error:.3f} px "
        f"(GCP{max_index})"
    )
    print("=" * 78)

    return rmse, mean_error, max_error, max_index


side_stats = print_report(
    "SIDE CAMERA",
    clicked_side,
    projected_side,
    side_dx,
    side_dy,
    side_error
)

back_stats = print_report(
    "BACK CAMERA",
    clicked_back,
    projected_back,
    back_dx,
    back_dy,
    back_error
)


# ============================================================
# Draw comparison images
#
# GREEN = manually clicked pixel
# RED   = 3D -> pixel projected location
# YELLOW LINE = difference between them
# ============================================================

def draw_comparison(
    image,
    clicked,
    projected,
    errors,
    output_path,
    title
):
    result = image.copy()

    # Title
    cv2.rectangle(
        result,
        (0, 0),
        (result.shape[1], 55),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        result,
        title,
        (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    for i in range(NUM_GCPS):
        cx, cy = clicked[i]
        px, py = projected[i]

        cx, cy = int(round(cx)), int(round(cy))
        px, py = int(round(px)), int(round(py))

        # Difference line
        cv2.line(
            result,
            (cx, cy),
            (px, py),
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # CLICKED = GREEN
        cv2.circle(
            result,
            (cx, cy),
            9,
            (0, 255, 0),
            -1
        )

        # PROJECTED = RED
        cv2.circle(
            result,
            (px, py),
            8,
            (0, 0, 255),
            -1
        )

        # Labels
        cv2.putText(
            result,
            f"GCP{i} click",
            (cx + 10, cy + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            result,
            f"GCP{i} proj {errors[i]:.1f}px",
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    # Legend
    legend_y = 80

    cv2.circle(result, (20, legend_y), 7, (0, 255, 0), -1)
    cv2.putText(
        result,
        "GREEN = clicked",
        (35, legend_y + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.circle(result, (220, legend_y), 7, (0, 0, 255), -1)
    cv2.putText(
        result,
        "RED = projected",
        (235, legend_y + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.line(
        result,
        (430, legend_y),
        (470, legend_y),
        (0, 255, 255),
        2
    )
    cv2.putText(
        result,
        "YELLOW = error",
        (480, legend_y + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")


side_output = os.path.join(
    OUTPUT_DIR,
    "side_gcp_comparison.jpg"
)

back_output = os.path.join(
    OUTPUT_DIR,
    "back_gcp_comparison.jpg"
)

draw_comparison(
    img_side,
    clicked_side,
    projected_side,
    side_error,
    side_output,
    "SIDE CAMERA - GCP REPROJECTION CHECK"
)

draw_comparison(
    img_back,
    clicked_back,
    projected_back,
    back_error,
    back_output,
    "BACK CAMERA - GCP REPROJECTION CHECK"
)


# ============================================================
# Save CSV report
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "gcp_reprojection_errors.csv"
)

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "camera",
        "gcp",
        "world_X_m",
        "world_Y_m",
        "world_Z_m",
        "clicked_u_px",
        "clicked_v_px",
        "projected_u_px",
        "projected_v_px",
        "dx_px",
        "dy_px",
        "error_px"
    ])

    for camera_name, clicked, projected, dx, dy, error in [
        ("side", clicked_side, projected_side,
         side_dx, side_dy, side_error),
        ("back", clicked_back, projected_back,
         back_dx, back_dy, back_error)
    ]:
        for i in range(NUM_GCPS):
            writer.writerow([
                camera_name,
                f"GCP{i}",
                WORLD_GCPs[i, 0],
                WORLD_GCPs[i, 1],
                WORLD_GCPs[i, 2],
                clicked[i, 0],
                clicked[i, 1],
                projected[i, 0],
                projected[i, 1],
                dx[i],
                dy[i],
                error[i]
            ])

print(f"Saved: {csv_path}")


# ============================================================
# Final summary
# ============================================================

print("\n")
print("============================================================")
print("FINAL SUMMARY")
print("============================================================")
print(
    f"Side camera: "
    f"RMSE={side_stats[0]:.3f}px, "
    f"Mean={side_stats[1]:.3f}px, "
    f"Max={side_stats[2]:.3f}px (GCP{side_stats[3]})"
)
print(
    f"Back camera: "
    f"RMSE={back_stats[0]:.3f}px, "
    f"Mean={back_stats[1]:.3f}px, "
    f"Max={back_stats[2]:.3f}px (GCP{back_stats[3]})"
)
print("============================================================")
print("\nOutput files:")
print(f"  {side_output}")
print(f"  {back_output}")
print(f"  {csv_path}")
