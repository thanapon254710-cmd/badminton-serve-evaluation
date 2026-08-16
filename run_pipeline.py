from triangulate import BadmintonTracker3D
from evalautor import evaluate_serve_performance

# Step 1: Load model + calibration, run triangulation
tracker = BadmintonTracker3D(
    model_path="models/best.pt",
    calib_file="court_calibration.npz"
)

trajectory = tracker.process_videos(
    "raw_videos/side_take1.mov",
    "raw_videos/back_take1.MOV"
)

print(f"Got {len(trajectory)} 3D trajectory points")
for p in trajectory:
    print(f"frame {p['frame']}: X={p['X']:.2f}, Y={p['Y']:.2f}, Z={p['Z']:.2f}")

# Step 2: Score the trajectory
if len(trajectory) == 0:
    print("No trajectory points detected — check model, calibration, or video paths.")
else:
    report = evaluate_serve_performance(trajectory, serve_type="short_front_corner")
    print("\n=== Serve Evaluation Report ===")
    print(f"Overall Score: {report['final_score']:.1f} / 100")
    print(f"Peak Height: {report['max_height_m']:.2f} m")
    print(f"Net Clearance: {report['net_clearance_m']:.2f} m")
    for d in report['deductions']:
        print(f" - {d}")