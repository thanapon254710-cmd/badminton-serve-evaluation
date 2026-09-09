import numpy as np

def evaluate_serve_performance(trajectory_data, serve_type="short_front_corner"):
    """
    Evaluates serve quality based on trajectory feature extraction.
    Target 1: Short Serve Front Corner -> Target = (X: 1.98m, Y: 2.59m)
    Target 2: Long Serve Back Corner  -> Target = (X: 6.70m, Y: 2.59m)
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

    # Feature 3: Ground Landing Point (Final point before Z approaches floor)
    landing_pt = pts[-1, :2]  # (X_land, Y_land)

    # Initialize Score Parameters
    base_score = 100.0
    deductions = []

    if serve_type == "short_front_corner":
        target = np.array([1.98, 2.59])  # Front service line corner

        # Rule A: Net Clearance Penalty (Short serve should graze the net, ~0.05m to 0.20m clearance)
        if net_clearance > 0.25:
            pen = (net_clearance - 0.25) * 40.0
            base_score -= pen
            deductions.append(f"Net clearance too high ({net_clearance:.2f}m above net): -{pen:.1f} pts")
        elif net_clearance < 0.0:
            base_score -= 50.0
            deductions.append("Shuttlecock hit the net: -50.0 pts")

        # Rule B: Landing Accuracy Penalty
        dist_err = np.linalg.norm(landing_pt - target)
        pen_dist = dist_err * 25.0
        base_score -= pen_dist
        deductions.append(f"Landing error ({dist_err:.2f}m from corner target): -{pen_dist:.1f} pts")

    elif serve_type == "high_back_corner":
        target = np.array([6.70, 2.59])  # Deep baseline corner

        # Rule A: High Serve Arc Requirement (Apex should be > 3.5 meters)
        if max_height < 3.5:
            pen = (3.5 - max_height) * 20.0
            base_score -= pen
            deductions.append(f"Serve arc too flat (Peak height {max_height:.2f}m): -{pen:.1f} pts")

        # Rule B: Landing Accuracy Penalty
        dist_err = np.linalg.norm(landing_pt - target)
        pen_dist = dist_err * 20.0
        base_score -= pen_dist
        deductions.append(f"Landing error ({dist_err:.2f}m from back corner): -{pen_dist:.1f} pts")

    final_score = float(np.clip(base_score, 0.0, 100.0))

    return {
        "final_score": final_score,
        "max_height_m": float(max_height),
        "net_clearance_m": float(net_clearance),
        "landing_coordinate_m": (float(landing_pt[0]), float(landing_pt[1])),
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
    report = evaluate_serve_performance(simulated_traj, serve_type="short_front_corner")
    print("=== Serve Evaluation Report ===")
    print(f"Overall Score: {report['final_score']:.1f} / 100")
    print(f"Peak Height: {report['max_height_m']:.2f} m")
    print(f"Net Clearance: {report['net_clearance_m']:.2f} m")
    for d in report['deductions']:
        print(f" - {d}")