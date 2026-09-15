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
# Scoring weights.
#
# IMPORTANT: unlike the geometry constants above (which come straight
# from actual BWF court dimensions), everything below is an engineering
# estimate based on badminton intuition -- NOT a value derived from BWF
# rules or measured real serve data. They used to be unnamed literals
# scattered through the scoring logic (e.g. "* 40.0", "* 25.0"); they're
# named and centralized here instead so they're visible, easy to tune,
# and honestly labeled as guesses rather than looking authoritative.
# Treat these as placeholders to be recalibrated once real serve data
# is available, not as settled constants.
# -----------------------------------------------------------------------

# Short serve: a low, flat trajectory that just grazes the net is the
# optimal technique (harder for the opponent to attack), so clearance
# within this window costs nothing. This window itself isn't a BWF rule,
# just server-technique convention.
SHORT_SERVE_IDEAL_CLEARANCE_M = 0.25

# Points lost per metre of net clearance above that ideal window.
SHORT_SERVE_CLEARANCE_PENALTY_PER_M = 40.0

# Flat penalty for clipping/hitting the net (net_clearance < 0).
NET_HIT_PENALTY = 50.0

# Points lost per metre of landing error from the target line (distance
# to TARGET_SHORT_FRONT / TARGET_HIGH_BACK). These two are intentionally
# different: the legal short-serve depth range (net to 1.98m) is much
# smaller than the long-serve range (1.98m to 6.70m), so the same
# per-metre penalty would make a short-serve miss look proportionally
# far worse than an equivalent long-serve miss. Scaling the short-serve
# weight up keeps the two roughly comparable in severity -- still a
# rough estimate, not a derived value.
SHORT_SERVE_LANDING_PENALTY_PER_M = 25.0
LONG_SERVE_LANDING_PENALTY_PER_M = 20.0

# Long/high serve: minimum apex height, in metres, for the arc to be
# considered a "proper" high serve rather than a flat, attackable one.
LONG_SERVE_MIN_APEX_M = 3.5

# Points lost per metre the apex falls short of LONG_SERVE_MIN_APEX_M.
LONG_SERVE_ARC_PENALTY_PER_M = 20.0

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
    Score landing accuracy by proximity to the nearest target GCP.

    Short:
        GCP3 = (1.98, 0.00)
        GCP4 = (1.98, 2.59)

    High:
        GCP5 = (6.70, 0.00)
        GCP6 = (6.70, 2.59)

    Exact GCP = 100. The midpoint Y=1.295 is farther from either GCP
    and therefore scores lower. This implements the requested preference
    for the two 0.50 m endpoint zones over the centre of the line.
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

    # The midpoint between the two GCPs is 1.295 m from either endpoint.
    max_target_dist = 2.59 / 2.0
    score = float(np.clip(
        100.0 * (1.0 - nearest_dist / max_target_dist),
        0.0,
        100.0,
    ))

    nearest_name = name_a if dist_a <= dist_b else name_b
    return score, nearest_dist, nearest_name



def evaluate_serve_performance(trajectory_data, serve_type="auto"):
    """
    Evaluate serve quality from a triangulated 3D trajectory.

    serve_type:
        "short_front_corner"
        "high_back_corner"
        "auto"

    In "auto" mode, FRONT_BACK_SPLIT_X = 4.34 m is retained to classify
    the serve based on its estimated landing depth.

    IN/OUT is then determined independently by testing the landing
    coordinate against the calibrated court boundaries.

    The landing-accuracy score is measured against the full target
    LINE (center line to sideline) at the correct depth, not a single
    corner point -- see TARGET_SHORT_FRONT / TARGET_HIGH_BACK.
    """
    if len(trajectory_data) == 0:
        return {"error": "No 3D trajectory points recorded."}

    # Convert trajectory to array.
    pts = np.array(
        [[p["X"], p["Y"], p["Z"]] for p in trajectory_data],
        dtype=np.float64,
    )

    if pts.ndim != 2 or pts.shape[1] != 3:
        return {"error": "Invalid 3D trajectory format."}

    # Feature 1: Max Height (Apex).
    max_height = float(np.max(pts[:, 2]))

    # Feature 2: Net Clearance Height.
    net_idx = int(np.argmin(np.abs(pts[:, 0])))
    net_clearance = float(pts[net_idx, 2] - 1.55)

    # Feature 3: Estimated ground landing point.
    landing_pt = _find_landing_point(pts)
    landing_x = float(landing_pt[0])
    landing_y = float(landing_pt[1])

    # IN/OUT is ALWAYS determined by the preferred drop box.
    landing_result = _evaluate_landing_status(landing_pt)

    # Serve type is used only for selecting the appropriate scoring target.
    if serve_type == "auto":
        serve_type = _classify_serve_type(landing_x)

    if serve_type not in SERVE_TYPE_LABELS:
        return {"error": f"Unknown serve type: {serve_type}"}

    # Initialize score parameters.
    base_score = 100.0
    deductions = []

    if serve_type == "short_front_corner":
        target = TARGET_SHORT_FRONT

        # Rule A: Net clearance penalty.
        if net_clearance > SHORT_SERVE_IDEAL_CLEARANCE_M:
            pen = (
                net_clearance - SHORT_SERVE_IDEAL_CLEARANCE_M
            ) * SHORT_SERVE_CLEARANCE_PENALTY_PER_M
            base_score -= pen
            deductions.append(
                f"Net clearance too high ({net_clearance:.2f}m above net): "
                f"-{pen:.1f} pts"
            )
        elif net_clearance < 0.0:
            base_score -= NET_HIT_PENALTY
            deductions.append(f"Shuttlecock hit the net: -{NET_HIT_PENALTY:.1f} pts")

        # Rule B: Prefer landing near GCP3/GCP4 rather than the centre
        # of the short target line.
        landing_score, dist_err, nearest_gcp = _landing_target_score(
            landing_pt, serve_type
        )
        pen_dist = 100.0 - landing_score
        base_score -= pen_dist
        deductions.append(
            f"Landing target score {landing_score:.1f}/100 "
            f"({nearest_gcp}, {dist_err:.2f}m away): "
            f"-{pen_dist:.1f} pts"
        )

    elif serve_type == "high_back_corner":
        target = TARGET_HIGH_BACK

        # Rule A: High serve arc requirement.
        if max_height < LONG_SERVE_MIN_APEX_M:
            pen = (
                LONG_SERVE_MIN_APEX_M - max_height
            ) * LONG_SERVE_ARC_PENALTY_PER_M
            base_score -= pen
            deductions.append(
                f"Serve arc too flat (Peak height {max_height:.2f}m): "
                f"-{pen:.1f} pts"
            )

        # Rule B: Prefer landing near GCP5/GCP6 rather than the centre
        # of the high-serve target line.
        landing_score, dist_err, nearest_gcp = _landing_target_score(
            landing_pt, serve_type
        )
        pen_dist = 100.0 - landing_score
        base_score -= pen_dist
        deductions.append(
            f"Landing target score {landing_score:.1f}/100 "
            f"({nearest_gcp}, {dist_err:.2f}m away): "
            f"-{pen_dist:.1f} pts"
        )

    final_score = float(np.clip(base_score, 0.0, 100.0))

    # A serve that lands OUT is a fault -- the point goes to the opponent
    # outright, regardless of how good the net clearance/arc/depth
    # otherwise looked. The distance-based deductions above are still
    # computed and reported (useful context for "how far out"), but they
    # no longer determine the score once the serve is a fault.
    if landing_result["landing_status"] == "OUT":
        final_score = 0.0
        deductions.append(
            f"Serve is OUT ({landing_result['landing_status_reason']}): "
            f"score set to 0.0 pts"
        )

    return {
        "final_score": final_score,
        "max_height_m": max_height,
        "net_clearance_m": net_clearance,
        "landing_coordinate_m": (landing_x, landing_y),
        "landing_status": landing_result["landing_status"],
        "landing_status_reason": landing_result["landing_status_reason"],
        "serve_type": serve_type,
        "serve_type_label": SERVE_TYPE_LABELS.get(
            serve_type,
            serve_type,
        ),
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