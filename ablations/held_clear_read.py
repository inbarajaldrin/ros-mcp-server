#!/usr/bin/env python3
"""Base read before an insert, with the part in hand, from a view the held part does not block (2026-10-20).

Operator: "reading a pose before insertion to correct can be done even with a part in hand if you rotate
the wrist of the arm". The read from the insert hover was 4-8 mm off (the held part hid markers, the base
filled the image) and a read from the standard view while holding a bar was 8 mm off in y. Here:

  1. the held part's pose in the flange frame from the chain (held orientation in the world, the grasp
     point at the gripper centre at the recorded grasp depth) and its CAD mesh
  2. for wrist turns of 0 / +90 / -90 deg about the tool axis (the D415 cable allows 90 deg), at the
     standard view (flange above the base at VIEW_FLANGE_Z, top-down), project the held part's hull and the
     base's top square into the image (camera intrinsics + the hand-eye mount) and score the base area the
     part covers (and any of the base outside the image)
  3. move to the best view, read the base (markers + depth outline), and come back to the starting TCP pose
     exactly, so the held-orientation chain the insert uses is untouched

    python3 -u ablations/held_clear_read.py --object fork_orange --base base3 --held-quat QX QY QZ QW [--move]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))

TURNS_DEG = (0.0, 90.0, -90.0)
IMG_W, IMG_H = 1280, 720


def camera_K(timeout=5.0):
    import rclpy
    from sensor_msgs.msg import CameraInfo
    rclpy.init(); n = rclpy.create_node('held_clear_K'); st = {}
    n.create_subscription(CameraInfo, '/camera/camera/color/camera_info', lambda m: st.__setitem__('i', m), 2)
    end = time.time() + timeout
    while 'i' not in st and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
    n.destroy_node(); rclpy.shutdown()
    return np.array(st['i'].k).reshape(3, 3) if 'i' in st else None


def T(p, q):
    M = np.eye(4); M[:3, :3] = R.from_quat(q).as_matrix(); M[:3, 3] = p
    return M


def held_in_flange(obj, grasp_id, held_quat, tcp_p, tcp_q, tool_offset_m):
    """Held part pose in the flange frame: grasp point at the gripper centre (flange + R_f [0, 0, L])."""
    from compliant_insertion_studio.wrapper.cad_lookup import load_grasp_offset_in_object
    gp = np.asarray(load_grasp_offset_in_object(obj, int(grasp_id)), dtype=float)
    Rf = R.from_quat(tcp_q); Ro = R.from_quat(held_quat)
    gc = np.asarray(tcp_p) + Rf.apply([0.0, 0.0, tool_offset_m])
    po = gc - Ro.apply(gp)
    return np.linalg.inv(T(tcp_p, tcp_q)) @ T(po, held_quat)


OFFSET_MAX_M = 0.30                     # flange this far at most from straight above the base
OFFSET_STEP_M = 0.02
OFFSET_PENALTY = 0.5                    # per metre of offset: prefer the least oblique clear view
MIN_PART_CLEAR_M = 0.10                 # held part's lowest point at least this above the base top
COVER_OK = 0.01                         # a view is clear below this covered fraction
STEREO_BASELINE_M = 0.055              # D415 IR baseline
DEPTH_MARGIN = 0.15                     # aligned depth: no data in a band at the image edges (D415 FOV, IR shadow)


def score_views(obj, base_name, ob, T_f_o, tcp_q, K, view_z):
    """The held part is rigid with the camera, so it always occupies the same image region: what can be
    chosen is where the BASE lands in the image. Top-down views at view_z, flange yaw turned 0/+-90 deg,
    flange shifted on a grid around straight-above-the-base; score = base area the part's hull covers + base
    area outside the image + OFFSET_PENALTY * shift. Returns dicts, best first."""
    import cv2
    from grasp_depth import load_mesh
    from seat_check import X_CAM
    from spacing_check import BASE_SIDE_M
    from compliant_insertion_studio.wrapper.cad_geometry import _load_base_top_z_local
    V, _ = load_mesh(obj)
    (bx, by, bz), bq = ob
    h = BASE_SIDE_M[base_name] / 2
    top_z = bz + _load_base_top_z_local(base_name)
    sq = R.from_quat(bq).apply(np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]])) + [bx, by, top_z]
    # current wrist yaw relative to top-down (0,1,0,0), from the flange x axis: euler 'xyz' of a top-down
    # flange flips by 180 deg (2026-10-20: asked for a 3.6 rad wrist_3 turn, refused)
    xf = R.from_quat(tcp_q).apply([1.0, 0.0, 0.0]); xd = R.from_quat([0, 1, 0, 0]).apply([1.0, 0.0, 0.0])
    yaw0 = float(np.arctan2(xf[1], xf[0]) - np.arctan2(xd[1], xd[0]))
    s = 4
    W, H = IMG_W // s, IMG_H // s
    grid = np.arange(-OFFSET_MAX_M, OFFSET_MAX_M + 1e-9, OFFSET_STEP_M)
    out = []
    for d in TURNS_DEG:
        qf = (R.from_euler('z', yaw0 + np.radians(d)) * R.from_quat([0, 1, 0, 0])).as_quat()
        Rf = R.from_quat(qf).as_matrix()
        Rc = Rf @ X_CAM[:3, :3]
        # the part's image region is the same for every shift: compute it once
        Tf0 = np.eye(4); Tf0[:3, :3] = Rf
        Pf = (Tf0 @ T_f_o @ np.c_[V, np.ones(len(V))].T).T[:, :3]          # part, flange at the origin
        pc0 = Rf @ X_CAM[:3, 3]
        Pc = (Pf - pc0) @ Rc
        if (Pc[:, 2] <= 0.02).any():
            continue
        part_uv = (K @ (Pc / Pc[:, 2:3]).T).T[:, :2]
        big = np.zeros((3 * H, 3 * W), np.uint8); off = np.array([W, H])
        mp = big.copy(); cv2.fillConvexPoly(mp, cv2.convexHull((part_uv / s + off).astype(np.int32)), 1)
        # the D415's stereo shadow: a part at z_p in front of a surface at z_s blanks depth over ~ f * B * (1/z_p - 1/z_s)
        # px along the image x (IR baseline B 55 mm) - colour-clear views of a held line gave ~1200 depth points
        z_p = float(np.median(Pc[:, 2])); z_s = view_z + 0.07
        sh = int(np.ceil(K[0, 0] * STEREO_BASELINE_M * max(0.0, 1 / z_p - 1 / z_s) / s))
        if sh > 0:
            mp = cv2.dilate(mp, np.ones((1, 2 * sh + 1), np.uint8))
        for dx in grid:
            for dy in grid:
                pf = np.array([bx + dx, by + dy, view_z])
                if (Pf[:, 2] + view_z).min() < top_z + MIN_PART_CLEAR_M:
                    continue
                pc = pf + pc0
                Bc = (sq - pc) @ Rc
                base_uv = (K @ (Bc / Bc[:, 2:3]).T).T[:, :2]
                mb = big.copy(); cv2.fillConvexPoly(mb, (base_uv / s + off).astype(np.int32), 1)
                area = max(int(mb.sum()), 1)
                covered = float((mb & mp).sum()) / area
                # the base must sit where the aligned depth is valid: not the outer DEPTH_MARGIN of the image
                # (2026-10-20: a view that put it at the edge got 1656 depth points and no fit)
                mx, my = int(W * DEPTH_MARGIN), int(H * DEPTH_MARGIN)
                outside = 1.0 - float(mb[H + my:2 * H - my, W + mx:2 * W - mx].sum()) / area
                shift = float(np.hypot(dx, dy))
                out.append(dict(turn=d, shift_mm=[round(1000 * dx), round(1000 * dy)], covered=round(covered, 3),
                                outside=round(outside, 3), score=round(covered + outside + OFFSET_PENALTY * shift, 4),
                                flange_xyz=[round(float(v), 4) for v in pf], flange_quat=qf.tolist()))
    out.sort(key=lambda r: r['score'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--object', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--grasp-id', type=int, default=None)
    ap.add_argument('--held-quat', type=float, nargs=4, required=True)
    ap.add_argument('--tool-offset-m', type=float, default=None)
    a = ap.parse_args()
    from replay_real_assembly import observe_base_pose, snapshot_robot_state
    from primitives.shared.config import get_grasp_id_for_assembly, held_tool_offset
    from base_depth import VIEW_FLANGE_Z
    gid = a.grasp_id or get_grasp_id_for_assembly(a.object)
    L = a.tool_offset_m or float(np.asarray(held_tool_offset())[2])
    snap = snapshot_robot_state(); tcp = snap['tcp']
    ob = observe_base_pose(a.base)
    K = camera_K()
    Tfo = held_in_flange(a.object, gid, a.held_quat, tcp['xyz'], tcp['quat_xyzw'], L)
    for r in score_views(a.object, a.base, ob, Tfo, tcp['quat_xyzw'], K, VIEW_FLANGE_Z)[:6]:
        print(json.dumps({k: v for k, v in r.items() if k != 'flange_quat'}))


if __name__ == '__main__':
    main()
