"""Offline check of a camera-aimed calibration survey: no motion, no ROS publishing.

For each viewpoint of `scan_workspace.survey_viewpoints(..., cam_X=...)`, in the order the survey flies them and starting
from the given joints, runs the survey's own Cartesian IK (compute_cartesian_waypoints_ik, the call move_to makes) and
reports: reachable, joint step (survey refuses > 1.5 rad), wrist_3 (cable: |w3| <= 4 rad), table / self collision along
the waypoints, and how much of the ChArUco board (35 inner corners) and of base1's markers project inside the 1280x720
image, with a 30 px margin. The board pose comes from a still frame (board_pose.json: T_world_board), base1's from the
localizer.

usage: check_views.py board_pose.json --joints q0..q5 --base1 x y z yaw_deg [--wide]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from primitives.scan_workspace import (survey_viewpoints, localizer_cam_X, SURVEY_MAX_JOINT_STEP,  # noqa: E402
                                       SURVEY_MAX_TILT_DEG, SURVEY_MIN_RADIUS)
from primitives.shared.ik import compute_cartesian_waypoints_ik, forward_kinematics, dh_params  # noqa: E402
from primitives.shared.collision import check_collision_with_table, check_self_collision  # noqa: E402
from primitives.shared.config import CAD_DATA_DIR, LOCALIZER_CONFIG  # noqa: E402

W_IMG, H_IMG, MARGIN = 1280, 720, 30
MOUNT_DIR = REPO / 'assets' / 'cell' / 'camera_mount_v31'   # v3.1 saddles + D415 body, rg2_base_link frame, mm (robo-chain rg2-camera-mount)
TABLE_MARGIN_M = 0.015      # mount, camera and fingers stay this far above the table top
ARM_MARGIN_M = 0.020        # and this far outside the arm-link capsules (collision.py radii)
LINK_RADII = [0.075, 0.065, 0.055, 0.045, 0.045]   # base .. wrist2 (primitives/shared/collision.py); wrist3 carries the mount


def _stl_vertices(path):
    import struct
    b = open(path, 'rb').read()
    n = struct.unpack('<I', b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]))
    return np.unique(a['v'].reshape(-1, 3).astype(float).round(1), axis=0) / 1000.0


def tool_attachments(X):
    """Points (tool0 frame, m) on what the arm carries beyond its links: the v3.1 saddle + camera tower (STL, through
    the composite chain tool0 -> quick_changer_link rpy (0,0,-pi/2) -> rg2_base_link xyz (0,0,0.0181) rpy (0,-pi/2,0)),
    the D415 body (D415_body_rg2_base_link.stl: the mount job's final collision BREPs, fitted onto the saddle STLs at
    0.10 mm rms), and a box around the open RG2 (fingertips 231 mm out)."""
    T_qc = np.eye(4); T_qc[:3, :3] = R.from_euler('xyz', [0, 0, -np.pi / 2]).as_matrix()
    T_rg = np.eye(4); T_rg[:3, :3] = R.from_euler('xyz', [0, -np.pi / 2, 0]).as_matrix(); T_rg[:3, 3] = [0, 0, 0.0181]
    T_tool_rg2 = T_qc @ T_rg
    pts = [np.vstack([_stl_vertices(p) for p in sorted(MOUNT_DIR.glob('*.stl'))])]
    pts[0] = pts[0][::max(1, len(pts[0]) // 1500)] @ T_tool_rg2[:3, :3].T + T_tool_rg2[:3, 3]
    f = np.stack(np.meshgrid(np.linspace(-0.06, 0.06, 7), np.linspace(-0.03, 0.03, 3), np.linspace(0.10, 0.235, 6)), -1).reshape(-1, 3)
    pts.append(f)
    return np.vstack(pts)


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def attachment_clearance(q, P_tool):
    """(min height of the carried points above the table, min clearance to the arm-link capsules) at joints q."""
    from primitives.shared.collision import compute_all_joint_positions
    from primitives.shared.config import TABLE_HEIGHT
    Pw = P_tool @ forward_kinematics(dh_params, q)[:3, :3].T + forward_kinematics(dh_params, q)[:3, 3]
    J = [np.array(p) for p in compute_all_joint_positions(q)]
    arm = min(float(np.min(_seg_dist(Pw, J[i], J[i + 1]))) - LINK_RADII[i] for i in range(5))
    return float(Pw[:, 2].min() - TABLE_HEIGHT), arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('board_pose')
    ap.add_argument('--joints', type=float, nargs=6, required=True, help='start joints, UR order')
    ap.add_argument('--base1', type=float, nargs=4, required=True, metavar=('X', 'Y', 'Z', 'YAW_DEG'))
    ap.add_argument('--wide', action='store_true')
    a = ap.parse_args()
    import yaml
    c = yaml.safe_load(open(LOCALIZER_CONFIG))
    K = np.array([[c['camera_matrix'][0], 0, c['camera_matrix'][2]], [0, c['camera_matrix'][1], c['camera_matrix'][3]], [0, 0, 1]])
    D = np.array(c['distortion_coeffs'])
    X = localizer_cam_X()
    bp = json.load(open(a.board_pose))
    Wb = np.array(bp['T_world_board'])
    board = cv2.aruco.CharucoBoard((8, 6), 0.030, 0.022, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50))
    bc = np.asarray(board.getChessboardCorners()).reshape(-1, 3)
    board_w = bc @ Wb[:3, :3].T + Wb[:3, 3]
    target = np.array(bp['centre_m'])
    ann = json.load(open(CAD_DATA_DIR / 'aruco' / 'base1_aruco.json'))
    Rb = R.from_euler('z', a.base1[3], degrees=True).as_matrix()
    b1 = np.array([[m['T_object_to_marker']['position'][k] for k in 'xyz'] for m in ann['markers']])
    b1_w = b1 @ Rb.T + np.array(a.base1[:3])

    def in_frame(Pw, Tcam):
        Tinv = np.linalg.inv(Tcam)
        Pc = Pw @ Tinv[:3, :3].T + Tinv[:3, 3]
        front = Pc[:, 2] > 0.05
        uv, _ = cv2.projectPoints(Pc[front], np.zeros(3), np.zeros(3), K, D)
        uv = uv.reshape(-1, 2)
        ok = (uv[:, 0] > MARGIN) & (uv[:, 0] < W_IMG - MARGIN) & (uv[:, 1] > MARGIN) & (uv[:, 1] < H_IMG - MARGIN)
        return int(ok.sum()), float(np.median(Pc[front, 2])) if front.any() else float('nan')

    P_tool = tool_attachments(X)
    q = np.array(a.joints)
    T0 = forward_kinematics(dh_params, q)
    views = survey_viewpoints(target, T0[:3, :3], wide=a.wide, cam_X=X)
    print(f"target (board centre) {np.round(1000 * target, 1).tolist()} mm, {len(views)} views, start flange "
          f"{np.round(1000 * T0[:3, 3], 1).tolist()} mm")
    rows, flown = [], 0
    for i, (pos, Rf, label) in enumerate(views):
        tilt = float(np.degrees(np.arccos(np.clip(-Rf[2, 2], -1, 1))))
        why = []
        if np.hypot(pos[0], pos[1]) < SURVEY_MIN_RADIUS:
            why.append(f"radius {np.hypot(pos[0], pos[1]):.3f}")
        if tilt > SURVEY_MAX_TILT_DEG + 0.5:
            why.append(f"tilt {tilt:.1f}")
        if why:
            print(f"[{i + 1:2d}] {label:34s} SKIPPED by the survey ({', '.join(why)})")
            continue
        wps = compute_cartesian_waypoints_ik(q, pos[2], num_waypoints=60, target_pos=pos, target_orientation=Rf)
        if wps is None:
            print(f"[{i + 1:2d}] {label:34s} NO IK PATH (the survey skips it)")
            continue
        qe = np.array(wps[-1])
        err = 1000 * np.linalg.norm(forward_kinematics(dh_params, qe)[:3, 3] - pos)
        step = float(np.max(np.abs(qe - q)))
        coll = next((k for k, w in enumerate(wps) if check_collision_with_table(np.array(w)) or check_self_collision(np.array(w))), None)
        cl = [attachment_clearance(np.array(w), P_tool) for w in wps[::3] + [wps[-1]]]
        tab, arm = min(c[0] for c in cl), min(c[1] for c in cl)
        flags = []
        if err > 2.0:
            flags.append(f"IK end {err:.1f} mm (skipped)")
        if step > SURVEY_MAX_JOINT_STEP:
            flags.append(f"step {step:.2f} rad (skipped)")
        if abs(qe[5]) > 4.0:
            flags.append(f"wrist_3 {qe[5]:.2f} > 4")
        if coll is not None:
            flags.append(f"COLLISION at waypoint {coll}")
        if tab < TABLE_MARGIN_M:
            flags.append(f"MOUNT/CAMERA/FINGERS {1000 * tab:.0f} mm above the table")
        if arm < ARM_MARGIN_M:
            flags.append(f"MOUNT/CAMERA {1000 * arm:.0f} mm from the arm links")
        Tcam = forward_kinematics(dh_params, qe) @ X
        nb, zb = in_frame(board_w, Tcam)
        nm, _ = in_frame(b1_w, Tcam)
        moves = not any('skipped' in f for f in flags)
        print(f"[{i + 1:2d}] {label:34s} flange {np.round(1000 * pos, 0).astype(int).tolist()} tilt {tilt:4.1f} "
              f"step {step:4.2f} w3 {qe[5]:+5.2f} | board {nb:2d}/35 at {1000 * zb:4.0f} mm, base1 markers {nm}/4"
              + f" | carried: table {1000 * tab:3.0f} mm, arm {1000 * arm:3.0f} mm"
              + (f" | {'; '.join(flags)}" if flags else ""))
        rows.append(dict(label=label, moves=moves, board=nb, base1=nm, w3=float(qe[5]), table=tab, arm=arm,
                         collision=coll is not None or tab < TABLE_MARGIN_M or arm < ARM_MARGIN_M))
        if moves:
            q, flown = qe, flown + 1
    good = [r for r in rows if r['moves'] and not r['collision'] and r['board'] >= 30]
    print(f"\n{flown} views fly; {len(good)} with >= 30/35 board corners in frame and no collision; "
          f"{sum(r['base1'] >= 2 for r in good)} of those also see >= 2 base1 markers; "
          f"collisions (arm or carried): {sum(r['collision'] for r in rows)}; max |wrist_3| {max(abs(r['w3']) for r in rows):.2f}; "
          f"carried min clearance: table {1000 * min(r['table'] for r in rows):.0f} mm, arm {1000 * min(r['arm'] for r in rows):.0f} mm")


if __name__ == '__main__':
    main()
