import os
import glob
import json
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ----------------------------------------------------------------------------
# 1. Advanced Shuttlecock Contour & Motion Extraction (keep everything unchanged)
# ----------------------------------------------------------------------------

@dataclass
class FrameSignal:
    frame_idx: int
    shuttle_x: Optional[float] = None
    shuttle_y: Optional[float] = None
    score: float = 0.0

def detect_shuttle_in_frame(
    frame: np.ndarray, 
    prev_frame: Optional[np.ndarray],
    last_known_pos: Optional[Tuple[float, float]] = None
) -> Tuple[Optional[float], Optional[float], float]:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Top-Hat Filter
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
    _, bright_mask = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)

    # Motion Mask
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(
            motion_mask,
            np.ones((3, 3), np.uint8),
            iterations=1
        )
        candidate_mask = cv2.bitwise_and(bright_mask, motion_mask)
    else:
        candidate_mask = bright_mask

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        kernel_clean
    )

    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_candidate = None
    max_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if 4 <= area <= 350:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0

            if 0.25 <= aspect_ratio <= 2.5:
                M = cv2.moments(cnt)

                if M["m00"] == 0:
                    continue

                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

                score = area * (
                    1.0 / (abs(1.0 - aspect_ratio) + 0.5)
                )

                if last_known_pos is not None:
                    dist = np.sqrt(
                        (cx - last_known_pos[0])**2 +
                        (cy - last_known_pos[1])**2
                    )

                    if dist > 80:
                        continue

                    score += max(0, (80 - dist))

                if score > max_score:
                    max_score = score
                    best_candidate = (cx, cy, score)

    if best_candidate:
        return best_candidate

    return None, None, 0.0


def extract_trajectory(folder_path: str):
    image_paths = sorted([
        p for p in glob.glob(os.path.join(folder_path, "*"))
        if p.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))
    ])

    if not image_paths:
        raise FileNotFoundError(
            f"No image files found in: {folder_path}"
        )

    filenames = [os.path.basename(p) for p in image_paths]

    signals = []
    prev_frame = None
    last_pos = None

    for idx, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        x, y, score = detect_shuttle_in_frame(
            frame,
            prev_frame,
            last_known_pos=last_pos
        )

        if x is not None and y is not None:
            last_pos = (x, y)

        signals.append(
            FrameSignal(
                frame_idx=idx,
                shuttle_x=x,
                shuttle_y=y,
                score=score
            )
        )

        prev_frame = frame

    return signals, filenames

# ----------------------------------------------------------------------------
# 2. Kinetics & Strict 1-to-1 Matching
#    (Only change the matching algorithm to prevent duplicate images)
# ----------------------------------------------------------------------------

def extract_kinetics(signals: List[FrameSignal]) -> np.ndarray:
    n = len(signals)

    y_vals = np.array([
        s.shuttle_y if s.shuttle_y is not None else np.nan
        for s in signals
    ], dtype=float)

    nans = np.isnan(y_vals)

    if nans.all():
        y_vals = np.zeros(n)
    else:
        idxs = np.arange(n)
        y_vals[nans] = np.interp(
            idxs[nans],
            idxs[~nans],
            y_vals[~nans]
        )

    y_norm = (
        y_vals - np.mean(y_vals)
    ) / (np.std(y_vals) + 1e-6)

    velocity = np.gradient(y_vals)

    v_norm = (
        velocity - np.mean(velocity)
    ) / (np.std(velocity) + 1e-6)

    acceleration = np.gradient(velocity)

    a_norm = (
        acceleration - np.mean(acceleration)
    ) / (np.std(acceleration) + 1e-6)

    return np.column_stack((
        y_norm,
        v_norm,
        a_norm
    ))


def align_strict_1to1(
    kin_a: np.ndarray,
    kin_b: np.ndarray
) -> List[Tuple[int, int]]:

    # Calculate the overall correlation (Energy Signature)
    # from your original precise feature set
    energy_a = np.linalg.norm(kin_a, axis=1)
    energy_b = np.linalg.norm(kin_b, axis=1)

    # Find the best Time Offset that matches both clips
    corr = np.correlate(
        energy_a,
        energy_b,
        mode='full'
    )

    best_offset = int(
        np.argmax(corr) - (len(energy_b) - 1)
    )

    print(
        f"[OK] Locked the highest-accuracy Frame Offset at: "
        f"{best_offset} frames (1-to-1 matching, no duplicates)"
    )

    matches = []

    len_a, len_b = len(kin_a), len(kin_b)

    for i in range(len_a):
        j = i - best_offset

        if 0 <= j < len_b:
            matches.append((i, j))

    return matches


# ----------------------------------------------------------------------------
# 3. Main Matching Process
# ----------------------------------------------------------------------------

def match_badminton(folder_a: str, folder_b: str) -> dict:

    print(
        f"[1/3] Extracting shuttlecock trajectory "
        f"from Folder A: {folder_a}"
    )

    sig_a, files_a = extract_trajectory(folder_a)

    print(
        f"[2/3] Extracting shuttlecock trajectory "
        f"from Folder B: {folder_b}"
    )

    sig_b, files_b = extract_trajectory(folder_b)

    print(
        "[3/3] Matching using Unique Kinetic Alignment "
        "(Strict 1-to-1)..."
    )

    kin_a = extract_kinetics(sig_a)
    kin_b = extract_kinetics(sig_b)

    path = align_strict_1to1(kin_a, kin_b)

    matches = [
        {
            "file_a": files_a[i],
            "file_b": files_b[j]
        }
        for i, j in path
    ]

    return {
        "folder_a": folder_a,
        "folder_b": folder_b,
        "matches": matches
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Badminton Precision 1-to-1 Matcher"
    )

    parser.add_argument(
        "folder_a",
        help="Path to Folder A"
    )

    parser.add_argument(
        "folder_b",
        help="Path to Folder B"
    )

    parser.add_argument(
        "--out",
        default=r"C:\Users\Tawan\matchPhoto\matched.json"
    )

    args = parser.parse_args()

    result = match_badminton(
        args.folder_a,
        args.folder_b
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Matching results saved successfully to "
        f"{args.out} "
        f"(total {len(result['matches'])} unique 1-to-1 pairs)"
    )