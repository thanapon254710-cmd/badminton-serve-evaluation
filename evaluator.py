import numpy as np

# Depth (X) of the short service line and the back boundary, in metres
# from the net -- same values used for the two serve targets below.
SHORT_SERVICE_LINE_X = 1.98
BACK_BOUNDARY_X = 6.70

# Serves landing in front of this depth are treated as short serves and
# scored against the front corner target; everything at or beyond it is
# treated as a high/long serve and scored against the back corner target.
FRONT_BACK_SPLIT_X = (SHORT_SERVICE_LINE_X + BACK_BOUNDARY_X) / 2.0  # 4.34 m

# Target lines used both for classification and for scoring. Each target
# is the full width of the line at that depth -- from the center line
# (Y=0.00) out to the right singles sideline (Y=2.59) -- not a single
# corner point. A serve landing anywhere along that line is equally
# "on target" depth-wise; only strays outside the [0, 2.59] width or off
# the correct depth should cost points.
TARGET_SHORT_FRONT = (
    np.array([SHORT_SERVICE_LINE_X, 0.00]),
    np.array([SHORT_SERVICE_LINE_X, 2.59]),
)  # Short service line, center to sideline
TARGET_HIGH_BACK = (
    np.array([BACK_BOUNDARY_X, 0.00]),
    np.array([BACK_BOUNDARY_X, 2.59]),
)  # Deep baseline, center to sideline

SERVE_TYPE_LABELS = {
    "short_front_corner": "Short serve",
    "high_back_corner": "High / long serve",
}


def _point_segment_distance(point, seg_start, seg_end):
    """
    Shortest distance from `point` to the line segment seg_start->seg_end
    -- i.e. distance to the closest point actually ON the segment, not the
    infinite line through it. For a landing point whose Y falls between
    the segment's endpoints this reduces to pure depth error (perpendicular
    distance to the line); for a landing point beyond either end it's the
    ordinary distance to that nearest corner, so landing wide of the
    sideline (or short of the center line) still costs accuracy points.
    """
    point = np.asarray(point, dtype=float)
    seg_start = np.asarray(seg_start, dtype=float)
    seg_end = np.asarray(seg_end, dtype=float)

    seg_vec = seg_end - seg_start
    seg_len_sq = float(np.dot(seg_vec, seg_vec))

    if seg_len_sq < 1e-12:
        # Degenerate (zero-length) segment -- fall back to point distance.
        return float(np.linalg.norm(point - seg_start))

    t = float(np.dot(point - seg_start, seg_vec) / seg_len_sq)
    t = float(np.clip(t, 0.0, 1.0))
    closest = seg_start + t * seg_vec
    return float(np.linalg.norm(point - closest))


def _find_landing_point(pts):
    """
    Estimate the shuttle's true floor-contact point.

    The previous implementation just took the last tracked frame
    (pts[-1]), but that is wherever the YOLO detector last happened to
    see the shuttle -- not necessarily the floor. If detection drops
    out early (motion blur, the shuttle passing behind a player, an
    edge-of-frame crop, etc.) that last tracked point can be anywhere
    still in the air, well short of where the shuttle actually landed.

    Instead, walk the trajectory forward from its apex and look for the
    first frame where height (Z) has come down to (near) floor level,
    then linearly interpolate the X/Y position at the exact Z == 0
    crossing between that frame and the one before it. If the shuttle
    is never tracked that close to the floor, fall back to the lowest
    point that was actually tracked after the apex.
    """
    apex_idx = int(np.argmax(pts[:, 2]))
    floor_eps = 0.15  # metres -- "close enough to the floor" tolerance

    for i in range(apex_idx + 1, len(pts)):
        if pts[i, 2] <= floor_eps:
            p1 = pts[i]
            if i == 0:
                return p1[:2]
            p0 = pts[i - 1]
            dz = p0[2] - p1[2]
            if dz <= 1e-9:
                return p1[:2]
            t = float(np.clip(p0[2] / dz, 0.0, 1.0))
            return p0[:2] + t * (p1[:2] - p0[:2])

    # Never tracked close to the floor -- best guess is the lowest point
    # actually recorded after the apex.
    lowest_idx = int(np.argmin(pts[apex_idx:, 2])) + apex_idx
    return pts[lowest_idx, :2]


def classify_serve_type(landing_pt):
    """
    Decide whether a serve was aiming short (front line) or long/high
    (back line) purely from where the shuttlecock actually landed --
    whichever target line it landed closer to wins.
    """
    dist_short = _point_segment_distance(landing_pt, *TARGET_SHORT_FRONT)
    dist_long = _point_segment_distance(landing_pt, *TARGET_HIGH_BACK)
    return "short_front_corner" if dist_short <= dist_long else "high_back_corner"


def evaluate_serve_performance(trajectory_data, serve_type="auto"):
    """
    Evaluates serve quality based on trajectory feature extraction.
    Target 1: Short serve -> the short service line, Y in [0.00m, 2.59m]
    Target 2: Long serve  -> the back boundary line, Y in [0.00m, 2.59m]

    serve_type: "short_front_corner", "high_back_corner", or "auto"
    (default). "auto" classifies the serve itself from where the
    shuttle actually landed -- front half of the court -> short serve
    criteria, back half -> high/long serve criteria -- instead of
    requiring the user to pick a target up front.
    """
    if len(trajectory_data) == 0:
        return {"error": "No 3D trajectory points recorded."}

    # Convert trajectory to array
    pts = np.array([[p['X'], p['Y'], p['Z']] for p in trajectory_data])

    # Feature 1: Max Height (Apex)
    max_height = np.max(pts[:, 2])

    # Feature 2: Net Clearance Height (Point where X is closest to 0.0)
    net_idx = np.argmin(np.abs(pts[:, 0]))
    net_clearance = pts[net_idx, 2] - 1.55  # Net top height is 1.55m

    # Feature 3: Ground Landing Point -- the actual floor-contact point,
    # not just the last frame the shuttle happened to be detected in.
    landing_pt = _find_landing_point(pts)  # (X_land, Y_land)

    if serve_type == "auto":
        serve_type = (
            "short_front_corner"
            if landing_pt[0] < FRONT_BACK_SPLIT_X
            else "high_back_corner"
        )

    # Initialize Score Parameters
    base_score = 100.0
    deductions = []

    if serve_type == "short_front_corner":
        target = TARGET_SHORT_FRONT

        # Rule A: Net Clearance Penalty (Short serve should graze the net, ~0.05m to 0.20m clearance)
        if net_clearance > 0.25:
            pen = (net_clearance - 0.25) * 40.0
            base_score -= pen
            deductions.append(f"Net clearance too high ({net_clearance:.2f}m above net): -{pen:.1f} pts")
        elif net_clearance < 0.0:
            base_score -= 50.0
            deductions.append("Shuttlecock hit the net: -50.0 pts")

        # Rule B: Landing Accuracy Penalty -- distance to the short service
        # line itself (clamped to the center-to-sideline width), not to a
        # single corner point.
        dist_err = _point_segment_distance(landing_pt, *target)
        pen_dist = dist_err * 25.0
        base_score -= pen_dist
        deductions.append(f"Landing error ({dist_err:.2f}m from short service line): -{pen_dist:.1f} pts")

    elif serve_type == "high_back_corner":
        target = TARGET_HIGH_BACK

        # Rule A: High Serve Arc Requirement (Apex should be > 3.5 meters)
        if max_height < 3.5:
            pen = (3.5 - max_height) * 20.0
            base_score -= pen
            deductions.append(f"Serve arc too flat (Peak height {max_height:.2f}m): -{pen:.1f} pts")

        # Rule B: Landing Accuracy Penalty -- distance to the back boundary
        # line itself (clamped to the center-to-sideline width), not to a
        # single corner point.
        dist_err = _point_segment_distance(landing_pt, *target)
        pen_dist = dist_err * 20.0
        base_score -= pen_dist
        deductions.append(f"Landing error ({dist_err:.2f}m from back boundary line): -{pen_dist:.1f} pts")

    final_score = float(np.clip(base_score, 0.0, 100.0))

    return {
        "final_score": final_score,
        "max_height_m": float(max_height),
        "net_clearance_m": float(net_clearance),
        "landing_coordinate_m": (float(landing_pt[0]), float(landing_pt[1])),
        "serve_type": serve_type,
        "serve_type_label": SERVE_TYPE_LABELS.get(serve_type, serve_type),
        "deductions": deductions
    }

# --- Example Execution ---
if __name__ == "__main__":
    # Simulated trajectory output
    simulated_traj = [
        {'X': -0.5, 'Y': 1.0, 'Z': 1.2},
        {'X': 0.0,  'Y': 1.8, 'Z': 1.68}, # Net clearance point
        {'X': 1.2,  'Y': 2.3, 'Z': 1.4},
        {'X': 1.90, 'Y': 2.50, 'Z': 0.05} # Landing point
    ]
    report = evaluate_serve_performance(simulated_traj, serve_type="auto")
    print("=== Serve Evaluation Report ===")
    print(f"Detected serve type: {report['serve_type_label']}")
    print(f"Overall Score: {report['final_score']:.1f} / 100")
    print(f"Peak Height: {report['max_height_m']:.2f} m")
    print(f"Net Clearance: {report['net_clearance_m']:.2f} m")
    for d in report['deductions']:
        print(f" - {d}")