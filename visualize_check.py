import cv2
import numpy as np

img_side = cv2.imread("frames/side1_0001.jpg")
img_back = cv2.imread("frames/back1_0001.jpg")

WORLD_GCPs = np.array([
    [0.00,  0.00, 1.55],
    [0.00, -2.59, 1.55],
    [0.00,  2.59, 1.55],
    [1.98,  0.00, 0.00],
    [1.98,  2.59, 0.00],
    [6.70,  0.00, 0.00],
    [6.70,  2.59, 0.00],
], dtype=np.float32)

data = np.load("court_calibration.npz")
pts_h = np.hstack([WORLD_GCPs, np.ones((7,1))])  # homogeneous, shared by both

def reproject(P, label):
    proj = (P @ pts_h.T).T
    proj_2d = proj[:, :2] / proj[:, 2:3]
    print(f"\n--- {label} ---")
    for i, p in enumerate(proj_2d):
        print(f"GCP{i} reprojects to pixel: ({p[0]:.1f}, {p[1]:.1f})")
    return proj_2d

p_side = reproject(data['P_side'], "P_side")
p_back = reproject(data['P_back'], "P_back")

def draw_gcps(img, points, out_path):
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(img, f"GCP{i}", (int(x)+10, int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(out_path, img)
    print(f"\nSaved {out_path}")

draw_gcps(img_side, p_side, "outputs/side_calib_check.jpg")
draw_gcps(img_back, p_back, "outputs/back_calib_check.jpg")