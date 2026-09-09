import os
import glob
import json
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ----------------------------------------------------------------------------
# 1. Advanced Shuttlecock Contour & Motion Extraction (unchanged)
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
# 2. Kinetics & Strict 1-to-1 Matching (uses x AND y now; alignment rewritten)
# ----------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-6)


def extract_kinetics(signals: List[FrameSignal]) -> np.ndarray:
    n = len(signals)

    def fill(values: np.ndarray) -> np.ndarray:
        nans = np.isnan(values)
        if nans.all():
            return np.zeros(n)
        idxs = np.arange(n)
        values = values.copy()
        values[nans] = np.interp(idxs[nans], idxs[~nans], values[~nans])
        return values

    y_vals = fill(np.array(
        [s.shuttle_y if s.shuttle_y is not None else np.nan for s in signals],
        dtype=float
    ))
    x_vals = fill(np.array(
        [s.shuttle_x if s.shuttle_x is not None else np.nan for s in signals],
        dtype=float
    ))

    y_norm = _normalize(y_vals)
    x_norm = _normalize(x_vals)

    y_vel = np.gradient(y_vals)
    x_vel = np.gradient(x_vals)
    v_norm = _normalize(np.sqrt(y_vel**2 + x_vel**2))

    y_acc = np.gradient(y_vel)
    x_acc = np.gradient(x_vel)
    a_norm = _normalize(np.sqrt(y_acc**2 + x_acc**2))

    # NOTE: x is included because two different camera angles usually see the
    # shuttlecock's *horizontal* motion very differently (this is what makes
    # them distinct angles), but the vertical rise/fall of a rally shot and
    # its speed/acceleration profile are largely angle-invariant. Keeping
    # both x and y (rather than y alone) gives the aligner more real signal
    # per frame, mostly through the velocity/acceleration magnitudes.
    return np.column_stack((y_norm, x_norm, v_norm, a_norm))


def align_strict_1to1(
    kin_a: np.ndarray,
    kin_b: np.ndarray,
    gap_penalty: Optional[float] = None
) -> List[Tuple[int, int]]:
    """
    Finds a strict one-to-one, time-monotonic correspondence between the two
    camera angles' shuttlecock kinetics.

    The previous version found ONE global cross-correlation offset between
    the two clips and then paired frames with a fixed `i -> i - offset`
    shift. That only works if both cameras run at the exact same frame rate
    and never drop/duplicate a frame anywhere in the whole clip -- one
    dropped frame partway through silently desyncs every pair after it,
    which is what caused frames from the two angles to stop actually
    matching the same real-world moment.

    This version instead compares every frame in A against every frame in B
    (via their kinetic feature vectors) and finds the lowest-cost monotonic
    path through that similarity space -- a global sequence alignment
    (Needleman-Wunsch style), the same idea used to align two time series
    that may be offset AND locally stretched/shrunk relative to each other.
    A frame is only left unmatched ("gapped") when pairing it would be worse
    than skipping it, and the DP guarantees every index is used at most once
    on each side, so there are never duplicate pairs.
    """
    len_a, len_b = len(kin_a), len(kin_b)

    if len_a == 0 or len_b == 0:
        return []

    # Pairwise distance between every (frame_a, frame_b) kinetic vector.
    diff = kin_a[:, None, :] - kin_b[None, :, :]
    cost = np.sqrt(np.sum(diff ** 2, axis=2))  # shape (len_a, len_b)

    if gap_penalty is None:
        # How bad a "skip" has to be relative to each frame's best possible
        # match before the aligner prefers to pair it up anyway. Tune this
        # up if you see too many frames being skipped, or down if you see
        # frames being force-matched to a clearly wrong partner.
        best_per_row = np.min(cost, axis=1)
        gap_penalty = float(np.median(best_per_row) * 2.0 + 1e-6)

    dp = np.full((len_a + 1, len_b + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, len_a + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_penalty
    for j in range(1, len_b + 1):
        dp[0, j] = dp[0, j - 1] + gap_penalty

    # 0 = diagonal (match), 1 = skip a-frame, 2 = skip b-frame
    back = np.zeros((len_a + 1, len_b + 1), dtype=np.uint8)

    for i in range(1, len_a + 1):
        row_cost = cost[i - 1]
        dp_prev_row = dp[i - 1]
        dp_row = dp[i]
        for j in range(1, len_b + 1):
            match = dp_prev_row[j - 1] + row_cost[j - 1]
            skip_a = dp_prev_row[j] + gap_penalty
            skip_b = dp_row[j - 1] + gap_penalty

            best = match
            move = 0
            if skip_a < best:
                best = skip_a
                move = 1
            if skip_b < best:
                best = skip_b
                move = 2

            dp_row[j] = best
            back[i, j] = move

    matches: List[Tuple[int, int]] = []
    i, j = len_a, len_b
    while i > 0 or j > 0:
        if i > 0 and j > 0 and back[i, j] == 0:
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or back[i, j] == 1):
            i -= 1
        else:
            j -= 1
    matches.reverse()

    if matches:
        offsets = [i - j for i, j in matches]
        print(
            f"[OK] Aligned {len(matches)} unique 1-to-1 pairs via monotonic "
            f"kinetic alignment (median frame lag: {int(np.median(offsets))}, "
            f"gap_penalty={gap_penalty:.4f})"
        )
    else:
        print("[WARN] No confident matches found between the two clips.")

    return matches


# ----------------------------------------------------------------------------
# 3. Main Matching Process
# ----------------------------------------------------------------------------

def match_badminton(folder_a: str, folder_b: str) -> dict:

    print(f"[1/3] Extracting shuttlecock trajectory from Folder A: {folder_a}")
    sig_a, files_a = extract_trajectory(folder_a)

    print(f"[2/3] Extracting shuttlecock trajectory from Folder B: {folder_b}")
    sig_b, files_b = extract_trajectory(folder_b)

    print("[3/3] Matching using monotonic kinetic alignment (strict 1-to-1)...")

    kin_a = extract_kinetics(sig_a)
    kin_b = extract_kinetics(sig_b)

    path = align_strict_1to1(kin_a, kin_b)

    matches = [
        {"file_a": files_a[i], "file_b": files_b[j]}
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
    parser.add_argument("folder_a", help="Path to Folder A")
    parser.add_argument("folder_b", help="Path to Folder B")
    parser.add_argument(
        "--out",
        default=r"C:\Users\Tawan\matchPhoto\matched.json"
    )

    args = parser.parse_args()

    result = match_badminton(args.folder_a, args.folder_b)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"Matching results saved successfully to {args.out} "
        f"(total {len(result['matches'])} unique 1-to-1 pairs)"
    )