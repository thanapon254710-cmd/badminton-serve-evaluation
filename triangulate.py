import cv2
import numpy as np
from ultralytics import YOLO

class BadmintonTracker3D:
    def __init__(self, model_path, calib_file):
        # Load YOLO model fine-tuned for badminton shuttlecocks
        self.model = YOLO(model_path)

        # Load precomputed camera projection matrices
        calib = np.load(calib_file)
        self.P_side = calib['P_side']
        self.P_back = calib['P_back']

    def detect_shuttlecock_2d(self, frame):
        """Runs YOLOv8 inference to detect shuttlecock centroid (x, y)."""
        results = self.model(frame, conf=0.20, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                return np.array([cx, cy], dtype=np.float32)
        return None

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

    def process_videos(self, side_video_path, back_video_path):
        cap_side = cv2.VideoCapture(side_video_path)
        cap_back = cv2.VideoCapture(back_video_path)

        trajectory_3d = []
        frame_idx = 0

        while cap_side.isOpened() and cap_back.isOpened():
            ret_s, frame_s = cap_side.read()
            ret_b, frame_b = cap_back.read()

            if not ret_s or not ret_b:
                break

            pt_side = self.detect_shuttlecock_2d(frame_s)
            pt_back = self.detect_shuttlecock_2d(frame_b)

            if pt_side is not None and pt_back is not None:
                pt_3d = self.triangulate_dlt(pt_side, pt_back)
                trajectory_3d.append({
                    'frame': frame_idx,
                    'X': pt_3d[0],  # Length (m)
                    'Y': pt_3d[1],  # Width (m)
                    'Z': pt_3d[2]   # Height (m)
                })

            frame_idx += 1

        cap_side.release()
        cap_back.release()
        return trajectory_3d