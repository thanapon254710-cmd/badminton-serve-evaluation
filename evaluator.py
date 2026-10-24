import numpy as np

# -----------------------------------------------------------------------
# Court geometry -- same coordinate system used by calibrate_core.py
# Origin: net center, floor level.
# -----------------------------------------------------------------------
SHORT_SERVICE_LINE_X = 1.98
BACK_BOUNDARY_X = 6.70

# Service areas. The origin is the net centre, X is court depth from the net
# and Y is lateral: positive Y = right half, negative Y = left half.
# Each service area is a drop box with the same X range and a mirrored Y range:
#   right: Y =  0.00 .. +2.59 m   (the original, default setting)
#   left : Y = -2.59 ..  0.00 m
# The four box corners are GCP3..GCP6 for both sides, in the order
#   GCP3 = (x_min, y_min)  GCP4 = (x_min, y_max)
#   GCP5 = (x_max, y_min)  GCP6 = (x_max, y_max)
# which reproduces calibrate_core.WORLD_GCPS_BY_SIDE for both sides.
SERVICE_AREAS = {
    "right": {"label": "Right service area", "y_min": 0.00, "y_max": 2.59},
    "left": {"label": "Left service area", "y_min": -2.59, "y_max": 0.00},
}
DEFAULT_SERVICE_SIDE = "right"

# Preferred drop box X range (shared by both sides). Only the box of the
# selected service side is considered IN.
PREFERRED_DROP_X_MIN = SHORT_SERVICE_LINE_X
PREFERRED_DROP_X_MAX = BACK_BOUNDARY_X
# Right-side Y range kept as module constants for older callers.
PREFERRED_DROP_Y_MIN = SERVICE_AREAS["right"]["y_min"]
PREFERRED_DROP_Y_MAX = SERVICE_AREAS["right"]["y_max"]

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

# Small tolerance (m) on the IN/OUT box edges to absorb triangulation noise.
LANDING_TOLERANCE_M = 0.05

# -----------------------------------------------------------------------
# Scoring model
#
# Each component is scored on its own 0..100 scale:
#   LPS = Landing Placement Score
#   NCS = Net Clearance Score
#   PHS = Peak Height Score
# and the serve total (0..100) is the weighted average of the three.
#
# The weights are the ONE place the weighting decision lives. They default
# to an equal average (1/3 each); change them here, or pass `weights=` to
# evaluate_serve_performance(), to test how sensitive the ranking is to the
# weighting. They are normalised to sum to 1, so any positive numbers work.
#
# These thresholds are INITIAL, TUNABLE criteria. They are not BWF rules
# and should later be calibrated against measured high-level/pro serves.
# -----------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "landing_placement": 1.0,
    "net_clearance": 1.0,
    "peak_height": 1.0,
}

# Landing placement: linear distance ramp. All four corners of the
# scoring box (GCP3-GCP6) are targets, for both serve types.
#   d   = distance from the landing point to the NEAREST of the four corners
#   LPS = 100 * max(0, 1 - d / LANDING_ZERO_SCORE_DISTANCE_M)
# The zero-score distance is the distance from the box centre to a corner
# (2.69 m), so LPS is 100 on any corner and 0 at the centre of the box.
# The box has the same size on both sides, so this is side-independent.
LANDING_ZERO_SCORE_DISTANCE_M = float(np.hypot(
    (PREFERRED_DROP_X_MAX - PREFERRED_DROP_X_MIN) / 2.0,
    (PREFERRED_DROP_Y_MAX - PREFERRED_DROP_Y_MIN) / 2.0,
))

# Short serve: lower clearance is preferred.
# <= 0.20 m -> NCS 100.
# >= 1.00 m -> NCS 0.
# Between them -> linear interpolation.
SHORT_NET_IDEAL_CLEARANCE_M = 0.20
SHORT_NET_ZERO_SCORE_CLEARANCE_M = 1.00

# Short serve: lower peak is preferred.
# <= 1.50 m -> PHS 100.
# >= 3.50 m -> PHS 0.
# Between them -> linear interpolation.
SHORT_PEAK_IDEAL_M = 1.50
SHORT_PEAK_ZERO_SCORE_M = 3.50

# High/long serve: higher clearance is preferred.
# <= 0.20 m -> NCS 0.
# >= 1.00 m -> NCS 100.
# Between them -> linear interpolation.
HIGH_NET_ZERO_SCORE_CLEARANCE_M = 0.20
HIGH_NET_IDEAL_CLEARANCE_M = 1.00

# High/long serve: higher peak is preferred.
# <= 2.00 m -> PHS 0.
# >= 3.50 m -> PHS 100.
# Between them -> linear interpolation.
HIGH_PEAK_ZERO_SCORE_M = 2.00
HIGH_PEAK_IDEAL_M = 3.50

# Hitting the net (clearance < 0) is always zero for the NCS.
NET_HIT_SCORE = 0.0

SERVE_TYPE_LABELS = {
    "short_front_corner": "Short serve",
    "high_back_corner": "High serve",
}


def _service_area(service_side):
    """Return (side, area dict) for a side name; raises ValueError if unknown."""
    side = DEFAULT_SERVICE_SIDE if service_side is None else service_side
    side = str(side).strip().lower()
    if side not in SERVICE_AREAS:
        raise ValueError(
            f"Unknown service side: {service_side!r} "
            f"(expected one of {sorted(SERVICE_AREAS)})."
        )
    return side, SERVICE_AREAS[side]


def _landing_gcps(service_side):
    """The four landing-score corner targets (GCP3..GCP6) for a side."""
    _, area = _service_area(service_side)
    x0, x1 = PREFERRED_DROP_X_MIN, PREFERRED_DROP_X_MAX
    y0, y1 = area["y_min"], area["y_max"]
    return (
        ("GCP3", np.array([x0, y0])),
        ("GCP4", np.array([x0, y1])),
        ("GCP5", np.array([x1, y0])),
        ("GCP6", np.array([x1, y1])),
    )


def _normalise_weights(weights):
    """Validate weights and scale them so they sum to 1."""
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    keys = ("landing_placement", "net_clearance", "peak_height")
    if set(w) != set(keys):
        raise ValueError(f"weights must have exactly the keys {keys}.")
    vals = np.array([float(w[k]) for k in keys], dtype=np.float64)
    if not np.isfinite(vals).all() or np.any(vals < 0) or vals.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum.")
    vals = vals / vals.sum()
    return {k: float(v) for k, v in zip(keys, vals)}


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


def _evaluate_landing_status(landing_pt, service_side=DEFAULT_SERVICE_SIDE):
    """
    Determine IN/OUT from the preferred drop box of the selected service
    side. The box is shared by BOTH serve types.

        right: X = 1.98 .. 6.70 m, Y =  0.00 .. +2.59 m
        left : X = 1.98 .. 6.70 m, Y = -2.59 ..  0.00 m

    Serve type does not change this IN/OUT boundary.
    """
    _, area = _service_area(service_side)
    y_min, y_max = area["y_min"], area["y_max"]

    x = float(landing_pt[0])
    y = float(landing_pt[1])
    tol = LANDING_TOLERANCE_M

    inside_x = (
        PREFERRED_DROP_X_MIN - tol <= x <= PREFERRED_DROP_X_MAX + tol
    )
    inside_y = y_min - tol <= y <= y_max + tol

    if inside_x and inside_y:
        return {
            "landing_status": "IN",
            "landing_status_reason": (
                f"Landing is inside the preferred drop box ({area['label'].lower()}: "
                f"X={PREFERRED_DROP_X_MIN:.2f}-{PREFERRED_DROP_X_MAX:.2f} m, "
                f"Y={y_min:.2f}-{y_max:.2f} m)."
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
            f"Y={y:.2f} m is outside {y_min:.2f}-{y_max:.2f} m"
        )
        # Helpful hint: landed in the mirrored box -> wrong side selected?
        mirrored = (-y_max - tol <= y <= -y_min + tol)
        if mirrored and inside_x:
            reasons.append(
                "it landed in the opposite service area - check that the "
                "correct service side was selected"
            )

    return {
        "landing_status": "OUT",
        "landing_status_reason": (
            f"Landing is outside the preferred drop box ({area['label'].lower()}): "
            + "; ".join(reasons)
            + "."
        ),
    }


def _landing_target_score(landing_pt, service_side=DEFAULT_SERVICE_SIDE):
    """
    Return the Landing Placement Score (LPS) on a 0..100 scale.

    Linear ramp on the Euclidean distance to the nearest of the four box
    corners (GCP3-GCP6) of the selected service side. For the right side:
        GCP3 = (1.98, 0.00)   GCP4 = (1.98, 2.59)
        GCP5 = (6.70, 0.00)   GCP6 = (6.70, 2.59)
    and for the left side the Y values become -2.59 and 0.00.
    100 on any corner, 0 at the centre of the box.

    All four corners are targets for both serve types.
    """
    point = np.asarray(landing_pt, dtype=np.float64)

    nearest_dist, nearest_name = min(
        (float(np.linalg.norm(point - xy)), name)
        for name, xy in _landing_gcps(service_side)
    )

    score = 100.0 * max(
        0.0, 1.0 - nearest_dist / LANDING_ZERO_SCORE_DISTANCE_M
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
    """Return the Net Clearance Score (NCS) on a 0..100 scale."""
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

    return float(normalized * 100.0)


def _peak_height_score(max_height, serve_type):
    """Return the Peak Height Score (PHS) on a 0..100 scale."""
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

    return float(normalized * 100.0)


def evaluate_serve_performance(
    trajectory_data,
    serve_type="auto",
    service_side=DEFAULT_SERVICE_SIDE,
    weights=None,
):
    """
    Evaluate serve quality from a triangulated 3D trajectory.

    Three component scores, each on its own 0..100 scale:
        LPS  Landing Placement Score  (distance to nearest drop-box corner)
        NCS  Net Clearance Score
        PHS  Peak Height Score
    Total (0..100) = weighted average of LPS, NCS and PHS. The default is an
    equal average; see DEFAULT_WEIGHTS / the `weights` argument.

    Short serve:
        lower net clearance -> higher NCS
        lower peak height   -> higher PHS

    High/long serve:
        higher net clearance -> higher NCS
        higher peak height   -> higher PHS

    `service_side` ("right" or "left") selects which service area is the
    IN box: right = Y 0..+2.59 m, left = Y -2.59..0 m.

    IN/OUT remains separate from scoring. A landing outside the preferred
    drop box is OUT and the final serve score is forced to 0 (the three
    component scores are still reported).
    """
    if len(trajectory_data) == 0:
        return {"error": "No 3D trajectory points recorded."}

    try:
        service_side, area = _service_area(service_side)
        norm_weights = _normalise_weights(weights)
    except ValueError as exc:
        return {"error": str(exc)}

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

    # IN/OUT is ALWAYS determined by the preferred drop box of the side.
    landing_result = _evaluate_landing_status(landing_pt, service_side)

    # Automatic serve classification is based only on landing X.
    if serve_type == "auto":
        serve_type = _classify_serve_type(landing_x)

    if serve_type not in SERVE_TYPE_LABELS:
        return {"error": f"Unknown serve type: {serve_type}"}

    # ---------------------------------------------------------------
    # Three independent component scores, each 0..100.
    # ---------------------------------------------------------------
    lps, dist_err, nearest_gcp = _landing_target_score(landing_pt, service_side)
    ncs = _net_clearance_score(net_clearance, serve_type)
    phs = _peak_height_score(max_height, serve_type)

    unpenalised_total = (
        norm_weights["landing_placement"] * lps
        + norm_weights["net_clearance"] * ncs
        + norm_weights["peak_height"] * phs
    )
    total_score = unpenalised_total

    deductions = [
        (
            f"Landing placement (LPS): {lps:.1f}/100 "
            f"({nearest_gcp}, {dist_err:.2f}m away)"
        ),
        (
            f"Net clearance (NCS): {ncs:.1f}/100 "
            f"({net_clearance:.2f}m above net)"
        ),
        (
            f"Peak height (PHS): {phs:.1f}/100 "
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
            "final score set to 0.0"
        )

    final_score = float(np.clip(total_score, 0.0, 100.0))

    return {
        "final_score": final_score,
        "score_before_out_rule": float(np.clip(unpenalised_total, 0.0, 100.0)),
        "max_height_m": max_height,
        "net_clearance_m": net_clearance,
        "landing_coordinate_m": (landing_x, landing_y),
        "landing_status": landing_result["landing_status"],
        "landing_status_reason": landing_result["landing_status_reason"],
        "service_side": service_side,
        "service_side_label": area["label"],
        "serve_type": serve_type,
        "serve_type_label": SERVE_TYPE_LABELS.get(serve_type, serve_type),
        "front_back_split_x_m": FRONT_BACK_SPLIT_X,
        "preferred_drop_box": {
            "x_min_m": PREFERRED_DROP_X_MIN,
            "x_max_m": PREFERRED_DROP_X_MAX,
            "y_min_m": area["y_min"],
            "y_max_m": area["y_max"],
        },
        "serve_classification_ranges": {
            "short": {
                "x_min_m": PREFERRED_DROP_X_MIN,
                "x_max_m": SHORT_SERVE_MAX_X,
                "y_min_m": area["y_min"],
                "y_max_m": area["y_max"],
            },
            "high": {
                "x_min_m": HIGH_SERVE_MIN_X,
                "x_max_m": PREFERRED_DROP_X_MAX,
                "y_min_m": area["y_min"],
                "y_max_m": area["y_max"],
            },
        },
        # Each component is 0..100; "total" is their weighted average
        # (0 if the serve is OUT).
        "component_scores": {"LPS": lps, "NCS": ncs, "PHS": phs},
        "score_breakdown": {
            "landing_placement": lps,
            "net_clearance": ncs,
            "peak_height": phs,
            "total": final_score,
        },
        "landing_target": {
            "gcp": nearest_gcp,
            "distance_m": dist_err,
            "score_percent": lps,
        },
        # Normalised weights (sum to 1) used for the total.
        "scoring_weights": norm_weights,
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
    print(f"Service area: {report['service_side_label']}")
    print(f"Overall Score: {report['final_score']:.1f} / 100")
    cs = report["component_scores"]
    print(
        f"  LPS={cs['LPS']:.1f}  NCS={cs['NCS']:.1f}  PHS={cs['PHS']:.1f}"
    )
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