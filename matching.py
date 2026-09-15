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


# Manual-click sync can be off by at most a couple of reaction-time frames.
# We only ever search this many candidate starting offsets, right at the
# beginning of the clips -- never across the whole video.
MAX_START_LAG = 7


def align_shorter_base(
    kin_a: np.ndarray,
    kin_b: np.ndarray
) -> Tuple[List[Tuple[int, int]], int, str, float]:
    """
    Synchronize both views assuming they started together (same "1, 2,
    click" count), give or take a few frames of manual-trigger reaction lag.

    Rather than sliding the shorter clip across the *entire* longer clip
    (which risks locking onto a coincidentally-similar moment mid-rally),
    we only test a small number of candidate starting offsets right at the
    beginning -- MAX_START_LAG frames -- and pick whichever gives the
    closest kinetic match. Once that start is chosen, every frame of the
    shorter video is paired 1-to-1 with one consecutive frame of the longer
    video, and any leftover frames at the tail of the longer video are
    ignored.
    """
    len_a, len_b = len(kin_a), len(kin_b)

    if len_a == 0 or len_b == 0:
        return [], 0, "none", 0.0

    if len_a <= len_b:
        short, long = kin_a, kin_b
        short_is_a = True
        short_name = "folder_a"
    else:
        short, long = kin_b, kin_a
        short_is_a = False
        short_name = "folder_b"

    n_short = len(short)
    n_long = len(long)
    max_start = n_long - n_short

    # Only look at the first MAX_START_LAG possible starting positions
    # (or fewer, if the clips are close enough in length that fewer exist).
    search_limit = min(max_start, MAX_START_LAG)

    weights = np.array([0.15, 0.05, 0.50, 0.30], dtype=float)
    scores = np.full(search_limit + 1, np.inf, dtype=float)

    for start_idx in range(search_limit + 1):
        window = long[start_idx:start_idx + n_short]
        diff = short - window

        weighted_sq = np.sum((diff ** 2) * weights, axis=1)
        frame_cost = np.sqrt(weighted_sq)

        frame_cost = np.sort(frame_cost)
        trim = int(len(frame_cost) * 0.10)

        if len(frame_cost) > 2 * trim and trim > 0:
            trimmed = frame_cost[trim:-trim]
        else:
            trimmed = frame_cost

        scores[start_idx] = (
            0.65 * float(np.median(trimmed))
            + 0.35 * float(np.mean(trimmed))
        )

    best_start = int(np.argmin(scores))
    best_score = float(scores[best_start])

    if len(scores) > 1:
        sorted_scores = np.sort(scores)
        second_score = float(sorted_scores[1])
        separation = max(0.0, (second_score - best_score) / (second_score + 1e-6))
        confidence = float(np.clip(separation * 5.0, 0.0, 1.0))
    else:
        confidence = 1.0

    pairs: List[Tuple[int, int]] = []
    for k in range(n_short):
        if short_is_a:
            pairs.append((k, best_start + k))
        else:
            pairs.append((best_start + k, k))

    # Offset is always expressed as: frame_a - frame_b, matching app.py.
    offset = pairs[0][0] - pairs[0][1] if pairs else 0

    print(
        f"[OK] Start-aligned synchronization: "
        f"{short_name} is shorter ({n_short} frames); "
        f"longer video has {n_long} frames"
    )
    print(
        f"[OK] Searched start lag 0..{search_limit} frames; "
        f"best start={best_start} (offset A-B={offset:+d} frames), "
        f"confidence={confidence:.3f}"
    )
    print(
        f"[OK] Paired all {n_short} shorter-video frames; "
        f"ignored {n_long - n_short - best_start} leftover frames from the "
        f"longer video"
    )

    return pairs, offset, "start_aligned_small_lag_search", confidence
# ----------------------------------------------------------------------------
# 3. Main Matching Process
# ----------------------------------------------------------------------------

def match_badminton(folder_a: str, folder_b: str) -> dict:

    print(f"[1/3] Extracting shuttlecock trajectory from Folder A: {folder_a}")
    sig_a, files_a = extract_trajectory(folder_a)

    print(f"[2/3] Extracting shuttlecock trajectory from Folder B: {folder_b}")
    sig_b, files_b = extract_trajectory(folder_b)

    print(
        "[3/3] Synchronizing with the shorter video as the complete "
        "time base..."
    )

    kin_a = extract_kinetics(sig_a)
    kin_b = extract_kinetics(sig_b)

    path, offset, offset_method, offset_confidence = align_shorter_base(
        kin_a,
        kin_b,
    )

    matches = [
        {"file_a": files_a[i], "file_b": files_b[j]}
        for i, j in path
    ]

    return {
        "folder_a": folder_a,
        "folder_b": folder_b,
        "matches": matches,
        "offset": offset,
        "offset_method": offset_method,
        "offset_confidence": offset_confidence,
        "shorter_video": "folder_a" if len(files_a) <= len(files_b) else "folder_b",
        "longer_video": "folder_b" if len(files_a) <= len(files_b) else "folder_a",
        "shorter_frame_count": min(len(files_a), len(files_b)),
        "ignored_longer_frames": abs(len(files_a) - len(files_b)),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Badminton shorter-video-base synchronization"
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