"""D415 depth against the colour image's own ChArUco distance, per board corner, over a survey's frames.

No robot, no hand-eye: both numbers are in the camera frame. For each frame with a saved aligned depth image, the ChArUco
board (8x6, 30/22 mm, DICT_7X7_50; print scale checked with a ruler 2026-10-14) is solved with solvePnP using the localizer's
K + distortion, every detected corner's camera-frame z is compared with the D415 depth at that pixel (median of a 7x7 patch),
and depth = a * z_pnp + b is fitted: a != 1 is a scale error, b an offset.

usage: depth_vs_pnp.py survey.json [robot_config.yaml]
"""
import json
import sys

import cv2
import numpy as np
import yaml

survey = json.load(open(sys.argv[1]))
cfg = yaml.safe_load(open(sys.argv[2] if len(sys.argv) > 2 else
                          '/home/aldrin/Desktop/ros2_ws/src/aruco_camera_localizer/config/robot_config.yaml'))
K = np.array([[cfg['camera_matrix'][0], 0, cfg['camera_matrix'][2]], [0, cfg['camera_matrix'][1], cfg['camera_matrix'][3]], [0, 0, 1]])
D = np.array(cfg['distortion_coeffs'])
board = cv2.aruco.CharucoBoard((8, 6), 0.030, 0.022, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50))
det = cv2.aruco.CharucoDetector(board)

rows, per_frame = [], []
for p in survey['poses']:
    for f in p.get('frames', []):
        if 'depth_file' not in f:
            continue
        img = cv2.imread(f['file'], cv2.IMREAD_GRAYSCALE)
        dep = cv2.imread(f['depth_file'], cv2.IMREAD_UNCHANGED).astype(float) / 1000.0
        cc, ci, _, _ = det.detectBoard(img)
        if ci is None or len(ci) < 8:
            continue
        op, ip = board.matchImagePoints(cc, ci)
        ok, rv, tv = cv2.solvePnP(op, ip, K, D)
        Pc = op.reshape(-1, 3) @ cv2.Rodrigues(rv)[0].T + tv.ravel()
        zs = []
        for (u, v), z in zip(ip.reshape(-1, 2).round().astype(int), Pc[:, 2]):
            patch = dep[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4]
            patch = patch[patch > 0]
            if len(patch) >= 10:
                rows.append((z, float(np.median(patch))))
                zs.append(float(np.median(patch)) - z)
        if zs:
            per_frame.append((p['label'], float(np.mean(Pc[:, 2])), float(np.mean(zs)), len(zs)))

A = np.array(rows)
n_all = len(A)
A = A[np.abs(A[:, 1] - A[:, 0]) < 0.03]       # depth holes / edge pixels (black-white transitions) are 50-300 mm off: drop
a, b = np.polyfit(A[:, 0], A[:, 1], 1)
r = A[:, 1] - (a * A[:, 0] + b)
keep = np.abs(r) < 3 * r.std()
a, b = np.polyfit(A[keep, 0], A[keep, 1], 1)
r = A[keep, 1] - (a * A[keep, 0] + b)
print(f"{n_all - len(A)} of {n_all} corner reads > 30 mm off dropped as depth holes/edges")
print(f"{len(per_frame)} frames, {len(A)} corners, colour-PnP distance {1000 * A[:, 0].min():.0f}-{1000 * A[:, 0].max():.0f} mm")
print(f"depth = {a:.4f} * z_pnp {1000 * b:+.1f} mm   (fit residual {1000 * r.std():.1f} mm)")
print(f"if pure scale: depth / z_pnp = {np.median(A[:, 1] / A[:, 0]):.4f};  if pure offset: depth - z_pnp = "
      f"{1000 * np.median(A[:, 1] - A[:, 0]):+.1f} mm (spread {1000 * np.std(A[:, 1] - A[:, 0]):.1f})")
for z0 in (0.35, 0.45, 0.55, 0.65):
    sel = np.abs(A[:, 0] - z0) < 0.05
    if sel.sum() > 20:
        print(f"  near {1000 * z0:.0f} mm: depth - PnP = {1000 * np.median(A[sel, 1] - A[sel, 0]):+.1f} mm "
              f"({100 * (np.median(A[sel, 1] / A[sel, 0]) - 1):+.2f} %), n={sel.sum()}")
