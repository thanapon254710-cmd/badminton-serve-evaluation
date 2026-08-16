import numpy as np
import cv2

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

side_pts = reproject(data['P_side'], "P_side")
back_pts = reproject(data['P_back'], "P_back")