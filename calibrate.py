import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Official BWF Badminton Court Reference Coordinates (Origin at Net Center Base)
# X = Length along court (m), Y = Width across court (m), Z = Height (m)
# ---------------------------------------------------------------------------

''' left side court coordinates
WORLD_GCPs = np.array([
    [0.00,  0.00, 1.55],  # GCP0: Net Line / Center Line
    [0.00, -2.59, 1.55],  # GCP1: Left Net Line
    [0.00,  2.59, 1.55],  # GCP2: Right Net Line
    [1.98, -2.59, 0.00],  # GCP3: Short Service Line / Left Singles Sideline
    [1.98,  0.00, 0.00],  # GCP4: Short Service Line / Center Line
    [6.70, -2.59, 0.00],  # GCP5: Back Boundary Line / Left Singles Sideline
    [6.70,  0.00, 0.00],  # GCP6: Back Boundary Line / Center Line
], dtype=np.float32)
'''

# right side court coordinates
WORLD_GCPs = np.array([
    [0.00,  0.00, 1.55],  # GCP0: Net Line / Center Line
    [0.00, -2.59, 1.55],  # GCP1: Left Net Line
    [0.00,  2.59, 1.55],  # GCP2: Right Net Line
    [1.98,  0.00, 0.00],  # GCP3: Short Service Line / Center Line
    [1.98,  2.59, 0.00],  # GCP4: Short Service Line / Right Singles Sideline
    [6.70,  0.00, 0.00],  # GCP5: Back Boundary Line / Center Line
    [6.70,  2.59, 0.00],  # GCP6: Back Boundary Line / Right Singles Sideline
], dtype=np.float32)

def calibrate_camera(image_path, camera_name):
    """
    Interactive GUI to click the 5 reference GCP points and calculate
    the 3x4 Projection Matrix P = K * [R | T].
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    clicked_pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pts) < len(WORLD_GCPs):
            clicked_pts.append([x, y])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, f"GCP {len(clicked_pts)-1}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(f"Calibrate {camera_name}", img)

    cv2.imshow(f"Calibrate {camera_name}", img)
    cv2.setMouseCallback(f"Calibrate {camera_name}", on_mouse)
    print(f"[{camera_name}] Click GCP points 0 to {len(WORLD_GCPs)-1} in order. Press any key when done.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_pts = np.array(clicked_pts, dtype=np.float32)

    # Approximate Intrinsic Camera Matrix K for 1080p stream
    focal_length = w  # Heuristic focal length estimation
    K = np.array([
        [focal_length, 0, w / 2.0],
        [0, focal_length, h / 2.0],
        [0, 0, 1.0]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # Solve Perspective-n-Point to find Rotation (R) and Translation (T)
    success, rvec, tvec = cv2.solvePnP(WORLD_GCPs, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        raise ValueError("PnP Calibration failed. Check selected points.")

    R, _ = cv2.Rodrigues(rvec)
    Rt = np.hstack((R, tvec))
    P = np.dot(K, Rt)

    return P, K, dist_coeffs

if __name__ == "__main__":
    # Example Calibration Execution (Save matrices to disk)
    P_side, _, _ = calibrate_camera("frames/side1_0001.jpg", "Side_Camera")
    P_back, _, _ = calibrate_camera("frames/back1_0001.jpg", "Back_Camera")
    np.savez("court_calibration.npz", P_side=P_side, P_back=P_back)
    print("Calibration module ready.")