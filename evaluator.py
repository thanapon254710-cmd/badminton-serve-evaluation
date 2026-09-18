import numpy as np

# -----------------------------------------------------------------------
# Court geometry -- same coordinate system used by calibrate_core.py
# Origin: net center, floor level.
# -----------------------------------------------------------------------
SHORT_SERVICE_LINE_X = 1.98
BACK_BOUNDARY_X = 6.70

# Preferred drop box: the ONLY region that is considered IN.
# (0, 0) is the centre of the net; X is court depth and positive Y is
# the selected/right side of the singles court.
PREFERRED_DROP_X_MIN = 1.98
PREFERRED_DROP_X_MAX = 6.70
PREFERRED_DROP_Y_MIN = 0.00
PREFERRED_DROP_Y_MAX = 2.59

# Automatic serve classification is based on landing depth inside the
# preferred drop box:
#   Short serve: X = 1.98 .. 4.72 m
#   High/long serve: X = 4.73 .. 6.70 m
#
# 4.725 m is the internal split so the 0.01 m display gap does not create
# an unclassifiable landing.
SHORT_SERVE_MAX_X = 4.72
HIGH_SERVE_MIN_X = 4.73
FRONT_BACK_SPLIT_X = (SHORT_SERVE_MAX_X + HIGH_SERVE_MIN_X) / 2.0

# Compatibility aliases used by older code.
COURT_Y_MIN = PREFERRED_DROP_Y_MIN
COURT_Y_MAX = PREFERRED_DROP_Y_MAX

# A shuttle landing on/before the short service line is inside the
# short-service depth region.  A landing beyond it is OUT for a short serve.
# Small tolerance accounts for triangulation noise.
LANDING_TOLERANCE_M = 0.05

# Target lines used for the landing-accuracy score. Each target is the
# full width of the line at that depth -- from the center line (Y=0.00)
# out to the right singles sideline (Y=2.59) -- not a single corner
# point. A serve landing anywhere along that line is equally "on
# target" depth-wise; only strays outside the [0, 2.59] width or off
# the correct depth should cost accuracy points.
TARGET_SHORT_FRONT = (
    np.array([SHORT_SERVICE_LINE_X, 0.00]),
    np.array([SHORT_SERVICE_LINE_X, COURT_Y_MAX]),
)  # Short service line, center to sideline
TARGET_HIGH_BACK = (
    np.array([BACK_BOUNDARY_X, 0.00]),
    np.array([BACK_BOUNDARY_X, COURT_Y_MAX]),
)  # Deep baseline, center to sideline

# -----------------------------------------------------------------------
# Scoring model
#
# Total = 100 points:
#   Landing placement : 50
#   Net clearance     : 30
#   Peak height       : 20
#
# These thresholds are INITIAL, TUNABLE criteria. They are not BWF rules
# and should later be calibrated against measured high-level/pro serves.
# -----------------------------------------------------------------------

LANDING_WEIGHT = 50.0
NET_CLEARANCE_WEIGHT = 30.0
PEAK_HEIGHT_WEIGHT = 20.0

# Landing placement uses a smooth distance curve. A larger sigma makes
# the score more forgiving. sigma=1.00 m means a landing 1.48 m from
# the nearest target still receives meaningful credit rather than an
# abrupt zero.
LANDING_SCORE_SIGMA_M = 1.00

# Short serve: lower clearance is preferred.
# <= 0.20 m -> full 30 points.
# > 1.00 m -> 0 points.
# Between them -> linear interpolation.
SHORT_NET_IDEAL_CLEARANCE_M = 0.20
SHORT_NET_ZERO_SCORE_CLEARANCE_M = 1.00

# Short serve: lower peak is preferred.
# <= 1.50 m -> full 20 points.
# > 3.50 m -> 0 points.
# Between them -> linear interpolation.
SHORT_PEAK_IDEAL_M = 1.50
SHORT_PEAK_ZERO_SCORE_M = 3.50

# High/long serve: higher clearance is preferred.
# < 0.20 m -> 0 points.
# >= 1.00 m -> full 30 points.
# Between them -> linear interpolation.
HIGH_NET_ZERO_SCORE_CLEARANCE_M = 0.20
HIGH_NET_IDEAL_CLEARANCE_M = 1.00

# High/long serve: higher peak is preferred.
# < 2.00 m -> 0 points.
# >= 3.50 m -> full 20 points.
# Between them -> linear interpolation.
HIGH_PEAK_ZERO_SCORE_M = 2.00
HIGH_PEAK_IDEAL_M = 3.50

# Hitting the net is always zero for the net-clearance component.
NET_HIT_SCORE = 0.0

SERVE_TYPE_LABELS = {
    "short_front_corner": "Short serve",
    "high_back_corner": "High serve",
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
    point = np.asarray(point, dtype=np.float64)
    seg_start = np.asarray(seg_start, dtype=np.float64)
    seg_end = np.asarray(seg_end, dtype=np.float64)

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
    out early, the last tracked point can still be in the air.

    Instead, walk forward from the apex and find the first point close
    to floor level. Linearly interpolate the X/Y position at Z == 0.
    If the trajectory never reaches the floor tolerance, fall back to
    the lowest tracked point after the apex.
    """
    pts = np.asarray(pts, dtype=np.float64)

    apex_idx = int(np.argmax(pts[:, 2]))
    floor_eps = 0.15  # metres -- close enough to the floor

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

    # Never tracked close to floor -- use the lowest post-apex point.
    lowest_idx = int(np.argmin(pts[apex_idx:, 2])) + apex_idx
    return pts[lowest_idx, :2]


def _classify_serve_type(landing_x):
    """
    Automatically classify the serve from landing depth.

    This does NOT determine IN/OUT. IN/OUT is determined separately by
    the preferred drop box.
    """
    x = float(landing_x)
    if x <= FRONT_BACK_SPLIT_X:
        return "short_front_corner"
    return "high_back_corner"


def _evaluate_landing_status(landing_pt, serve_type=None):
    """
    Determine IN/OUT from the preferred drop box shared by BOTH serves.

    Preferred drop box:
        X = 1.98 .. 6.70 m
        Y = 0.00 .. 2.59 m

    Serve type does not change this IN/OUT boundary.
    """
    x = float(landing_pt[0])
    y = float(landing_pt[1])
    tol = LANDING_TOLERANCE_M

    inside_x = (
        PREFERRED_DROP_X_MIN - tol <= x <= PREFERRED_DROP_X_MAX + tol
    )
    inside_y = (
        PREFERRED_DROP_Y_MIN - tol <= y <= PREFERRED_DROP_Y_MAX + tol
    )

    if inside_x and inside_y:
        return {
            "landing_status": "IN",
            "landing_status_reason": (
                "Landing is inside the preferred drop box "
                f"(X={PREFERRED_DROP_X_MIN:.2f}-{PREFERRED_DROP_X_MAX:.2f} m, "
                f"Y={PREFERRED_DROP_Y_MIN:.2f}-{PREFERRED_DROP_Y_MAX:.2f} m)."
            ),
        }

    reasons = []
    if not inside_x:
        reasons.append(
            f"X={x:.2f} m is outside "
            f"{PREFERRED_DROP_X_MIN:.2f}-{PREFERRED_DROP_X_MAX:.2f} m"
        )
    if not inside_y:
        reasons.append(
            f"Y={y:.2f} m is outside "
            f"{PREFERRED_DROP_Y_MIN:.2f}-{PREFERRED_DROP_Y_MAX:.2f} m"
        )

    return {
        "landing_status": "OUT",
        "landing_status_reason": (
            "Landing is outside the preferred drop box: "
            + "; ".join(reasons)
            + "."
        ),
    }


def _landing_target_score(landing_pt, serve_type):
    """
    Return landing-placement score on a 0..100 scale.

    The score is based on Euclidean distance to the nearer serve-specific
    GCP. It is deliberately smooth rather than having a hard cutoff.

    Short:
        GCP3 = (1.98, 0.00)
        GCP4 = (1.98, 2.59)

    High:
        GCP5 = (6.70, 0.00)
        GCP6 = (6.70, 2.59)
    """
    point = np.asarray(landing_pt, dtype=np.float64)

    if serve_type == "short_front_corner":
        gcp_a = np.array([1.98, 0.00], dtype=np.float64)
        gcp_b = np.array([1.98, 2.59], dtype=np.float64)
        name_a, name_b = "GCP3", "GCP4"
    elif serve_type == "high_back_corner":
        gcp_a = np.array([6.70, 0.00], dtype=np.float64)
        gcp_b = np.array([6.70, 2.59], dtype=np.float64)
        name_a, name_b = "GCP5", "GCP6"
    else:
        return 0.0, float("nan"), "unknown target"

    dist_a = float(np.linalg.norm(point - gcp_a))
    dist_b = float(np.linalg.norm(point - gcp_b))
    nearest_dist = min(dist_a, dist_b)
    nearest_name = name_a if dist_a <= dist_b else name_b

    # Gaussian decay:
    #   d=0      -> 100
    #   d=sigma  -> 60.7
    #   d=1.48m  -> about 33.5 when sigma=1.00m
    score = 100.0 * np.exp(
        -(nearest_dist ** 2) / (2.0 * LANDING_SCORE_SIGMA_M ** 2)
    )
    score = float(np.clip(score, 0.0, 100.0))

    return score, nearest_dist, nearest_name


def _bounded_linear_score(value, low, high, increasing=True):
    """
    Map a value to 0..1 with a linear, clamped relationship.

    increasing=True:
        low -> 0, high -> 1
    increasing=False:
        low -> 1, high -> 0
    """
    if high <= low:
        return 0.0

    t = float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))
    return t if increasing else 1.0 - t


def _net_clearance_score(net_clearance, serve_type):
    """Return the net-clearance component on a 0..30 scale."""
    if net_clearance < 0.0:
        return NET_HIT_SCORE

    if serve_type == "short_front_corner":
        # Lower is better.
        normalized = _bounded_linear_score(
            net_clearance,
            SHORT_NET_IDEAL_CLEARANCE_M,
            SHORT_NET_ZERO_SCORE_CLEARANCE_M,
            increasing=False,
        )
    else:
        # Higher is better.
        normalized = _bounded_linear_score(
            net_clearance,
            HIGH_NET_ZERO_SCORE_CLEARANCE_M,
            HIGH_NET_IDEAL_CLEARANCE_M,
            increasing=True,
        )

    return float(normalized * NET_CLEARANCE_WEIGHT)


def _peak_height_score(max_height, serve_type):
    """Return the peak-height component on a 0..20 scale."""
    if serve_type == "short_front_corner":
        # Lower is better.
        normalized = _bounded_linear_score(
            max_height,
            SHORT_PEAK_IDEAL_M,
            SHORT_PEAK_ZERO_SCORE_M,
            increasing=False,
        )
    else:
        # Higher is better.
        normalized = _bounded_linear_score(
            max_height,
            HIGH_PEAK_ZERO_SCORE_M,
            HIGH_PEAK_IDEAL_M,
            increasing=True,
        )

    return float(normalized * PEAK_HEIGHT_WEIGHT)


def evaluate_serve_performance(trajectory_data, serve_type="auto"):
    """
    Evaluate serve quality from a triangulated 3D trajectory.

    Score:
        Landing placement : 50 pts
        Net clearance     : 30 pts
        Peak height       : 20 pts
        Total             : 100 pts

    Short serve:
        lower net clearance -> higher score
        lower peak height   -> higher score

    High/long serve:
        higher net clearance -> higher score
        higher peak height   -> higher score

    IN/OUT remains separate from scoring. A landing outside the preferred
    drop box is OUT and the final serve score is forced to 0.
    """
    if len(trajectory_data) == 0:
        return {"error": "No 3D trajectory points recorded."}

    pts = np.array(
        [[p["X"], p["Y"], p["Z"]] for p in trajectory_data],
        dtype=np.float64,
    )

    if pts.ndim != 2 or pts.shape[1] != 3:
        return {"error": "Invalid 3D trajectory format."}

    # Feature 1: maximum height.
    max_height = float(np.max(pts[:, 2]))

    # Feature 2: net clearance height.
    net_idx = int(np.argmin(np.abs(pts[:, 0])))
    net_clearance = float(pts[net_idx, 2] - 1.55)

    # Feature 3: estimated floor landing point.
    landing_pt = _find_landing_point(pts)
    landing_x = float(landing_pt[0])
    landing_y = float(landing_pt[1])

    # IN/OUT is ALWAYS determined by the preferred drop box.
    landing_result = _evaluate_landing_status(landing_pt)

    # Automatic serve classification is based only on landing X.
    if serve_type == "auto":
        serve_type = _classify_serve_type(landing_x)

    if serve_type not in SERVE_TYPE_LABELS:
        return {"error": f"Unknown serve type: {serve_type}"}

    # ---------------------------------------------------------------
    # Three independent score components.
    # ---------------------------------------------------------------
    landing_score_100, dist_err, nearest_gcp = _landing_target_score(
        landing_pt, serve_type
    )
    landing_points = LANDING_WEIGHT * (landing_score_100 / 100.0)

    net_points = _net_clearance_score(net_clearance, serve_type)
    peak_points = _peak_height_score(max_height, serve_type)

    total_score = landing_points + net_points + peak_points

    deductions = [
        (
            f"Landing placement: {landing_points:.1f}/{LANDING_WEIGHT:.0f} pts "
            f"({nearest_gcp}, {dist_err:.2f}m away)"
        ),
        (
            f"Net clearance: {net_points:.1f}/{NET_CLEARANCE_WEIGHT:.0f} pts "
            f"({net_clearance:.2f}m above net)"
        ),
        (
            f"Peak height: {peak_points:.1f}/{PEAK_HEIGHT_WEIGHT:.0f} pts "
            f"({max_height:.2f}m)"
        ),
    ]

    if net_clearance < 0.0:
        deductions[1] += " — shuttlecock hit the net"

    # A serve outside the preferred drop box is a fault regardless of
    # trajectory quality.
    if landing_result["landing_status"] == "OUT":
        total_score = 0.0
        deductions.append(
            f"Serve is OUT ({landing_result['landing_status_reason']}): "
            "final score set to 0.0 pts"
        )

    final_score = float(np.clip(total_score, 0.0, 100.0))

    return {
        "final_score": final_score,
        "max_height_m": max_height,
        "net_clearance_m": net_clearance,
        "landing_coordinate_m": (landing_x, landing_y),
        "landing_status": landing_result["landing_status"],
        "landing_status_reason": landing_result["landing_status_reason"],
        "serve_type": serve_type,
        "serve_type_label": SERVE_TYPE_LABELS.get(serve_type, serve_type),
        "front_back_split_x_m": FRONT_BACK_SPLIT_X,
        "preferred_drop_box": {
            "x_min_m": PREFERRED_DROP_X_MIN,
            "x_max_m": PREFERRED_DROP_X_MAX,
            "y_min_m": PREFERRED_DROP_Y_MIN,
            "y_max_m": PREFERRED_DROP_Y_MAX,
        },
        "serve_classification_ranges": {
            "short": {
                "x_min_m": SHORT_SERVICE_LINE_X,
                "x_max_m": SHORT_SERVE_MAX_X,
                "y_min_m": PREFERRED_DROP_Y_MIN,
                "y_max_m": PREFERRED_DROP_Y_MAX,
            },
            "high": {
                "x_min_m": HIGH_SERVE_MIN_X,
                "x_max_m": BACK_BOUNDARY_X,
                "y_min_m": PREFERRED_DROP_Y_MIN,
                "y_max_m": PREFERRED_DROP_Y_MAX,
            },
        },
        "score_breakdown": {
            "landing_placement": landing_points,
            "net_clearance": net_points,
            "peak_height": peak_points,
            "total": final_score,
        },
        "landing_target": {
            "gcp": nearest_gcp,
            "distance_m": dist_err,
            "score_percent": landing_score_100,
        },
        "scoring_weights": {
            "landing_placement": LANDING_WEIGHT,
            "net_clearance": NET_CLEARANCE_WEIGHT,
            "peak_height": PEAK_HEIGHT_WEIGHT,
        },
        "deductions": deductions,
    }


# --- Example Execution ---
if __name__ == "__main__":
    simulated_traj = [
        {"X": -0.5, "Y": 1.0, "Z": 1.2},
        {"X": 0.0, "Y": 1.8, "Z": 1.68},
        {"X": 1.2, "Y": 2.3, "Z": 1.4},
        {"X": 1.90, "Y": 2.50, "Z": 0.05},
    ]

    report = evaluate_serve_performance(
        simulated_traj,
        serve_type="auto",
    )

    print("=== Serve Evaluation Report ===")
    print(f"Detected serve type: {report['serve_type_label']}")
    print(f"Overall Score: {report['final_score']:.1f} / 100")
    print(f"Peak Height: {report['max_height_m']:.2f} m")
    print(f"Net Clearance: {report['net_clearance_m']:.2f} m")
    print(
        f"Landing: "
        f"X={report['landing_coordinate_m'][0]:.2f} m, "
        f"Y={report['landing_coordinate_m'][1]:.2f} m"
    )
    print(f"Landing Status: {report['landing_status']}")
    print(f"Reason: {report['landing_status_reason']}")

    for d in report["deductions"]:
        print(f" - {d}")
