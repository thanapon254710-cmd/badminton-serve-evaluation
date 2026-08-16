import cv2

img_side = cv2.imread("frames/side1_0001.jpg")
img_back = cv2.imread("frames/back1_0001.jpg")

# paste in the two coordinate lists printed by projection_check.py
p_side = [
    (1561.4, 472.2),
    (1422.7, 465.4),
    (1793.4, 483.5),
    (1194.3, 765.8),
    (1302.9, 876.0),
    (298.1, 780.5),
    (99.6, 899.0),
]

p_back = [
    (937.5, 389.6),
    (575.4, 391.2),
    (1291.1, 388.0),
    (953.6, 669.0),
    (1353.5, 663.5),
    (1026.1, 981.4),
    (1661.2, 966.1),
]

def draw_gcps(img, points, out_path):
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(img, f"GCP{i}", (int(x)+10, int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(out_path, img)
    print(f"Saved {out_path}")

draw_gcps(img_side, p_side, "side_calib_check.jpg")
draw_gcps(img_back, p_back, "back_calib_check.jpg")