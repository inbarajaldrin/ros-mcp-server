#!/usr/bin/env python3
"""Hand-eye (wrist D415 mount) refit from CAD-to-depth registration over many views (2026-10-20).

base_icp.py fits base3 + its seated parts to depth at 1.3-1.5 mm rms in every view, but views from +x and -x
put the base 11 mm apart in y (markers and the outline too): the mount X is off at tilted wrist poses. This:

  survey  move to N views of the base (top-down at two heights and wrist yaws, tilted 20/30 deg from 4 sides),
          save each view's median aligned depth, intrinsics and TCP pose  -> ablations/logs/handeye_icp/<ts>/
  solve   one shared base pose (x, y, yaw, z) and one mount correction D (6 dof, X = X0 * D), least squares over
          every view's points against the CAD model (nearest-point correspondences re-found each round,
          trimmed), then report the per-view base spread before/after and the corrected cam_offset_* values.

    python3 -u ablations/handeye_icp_survey.py survey --base base3 --seated line_brown line_red fork_orange fork_yellow
    python3 -u ablations/handeye_icp_survey.py solve ablations/logs/handeye_icp/<ts> --base base3 --seated ...
Does not write the localizer config.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))
OUT = REPO / 'ablations/logs/handeye_icp'


def T(p, q):
    M = np.eye(4); M[:3, :3] = R.from_quat(q).as_matrix(); M[:3, 3] = p
    return M


def views(ob):
    """(label, flange xyz, flange quat) around the base."""
    from seat_check import X_CAM
    from tilted_seat_view import aimed_flange
    (bx, by, bz), _ = ob
    tgt = np.array([bx, by, bz + 0.0175])
    out = []
    for z in (0.52, 0.60):
        for yaw in (0.0, 60.0, -60.0):
            q = (R.from_euler('z', yaw, degrees=True) * R.from_quat([0, 1, 0, 0])).as_quat()
            out.append((f'top z{z:.2f} yaw{yaw:+.0f}', [bx, by, z], q))
    for tilt in (20.0, 30.0):
        for side in ('+x', '-x', '+y', '-y'):
            d = np.zeros(3); d['xy'.index(side[1])] = 1.0 if side[0] == '+' else -1.0
            pf, qf, _ = aimed_flange(tgt, d, tilt, 0.52, X_CAM, 0.0)
            out.append((f'tilt{tilt:.0f} from {side}', list(pf), qf))
    return out


def capture(n=12, timeout=6.0):
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image
    rclpy.init(); node = rclpy.create_node('he_icp'); st = {'d': [], 't': []}
    node.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', lambda m: st['d'].append(m), 20)
    node.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', lambda m: st.__setitem__('i', m), 2)
    node.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: st['t'].append(m), 50)
    t0 = time.time() + 0.3; end = time.time() + timeout
    while time.time() < end and (len([m for m in st['d'] if m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec > t0]) < n or 'i' not in st):
        rclpy.spin_once(node, timeout_sec=0.02)
    node.destroy_node(); rclpy.shutdown()
    br = CvBridge()
    fr = [m for m in st['d'] if m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec > t0][-n:]
    if len(fr) < 3 or 'i' not in st or not st['t']:
        return None
    d = np.median(np.stack([br.imgmsg_to_cv2(m, 'passthrough').astype(np.float32) for m in fr]), 0)
    P = np.array([[m.pose.position.x, m.pose.position.y, m.pose.position.z, m.pose.orientation.x, m.pose.orientation.y,
                   m.pose.orientation.z, m.pose.orientation.w] for m in st['t'][-50:]])
    if np.ptp(P[:, :3], axis=0).max() > 0.0005:
        return None                                   # arm not still
    return d, np.array(st['i'].k).reshape(3, 3), np.median(P, 0)


def survey(a):
    from replay_real_assembly import observe_base_pose
    ob = observe_base_pose(a.base)
    if ob is None:
        sys.exit(f'{a.base} not observed')
    out = OUT / time.strftime('%Y%m%d_%H%M%S'); out.mkdir(parents=True, exist_ok=True)
    meta = {'base': a.base, 'seated': a.seated, 'base_marker_start': [list(ob[0]), list(ob[1])], 'views': []}
    for i, (lab, pf, qf) in enumerate(views(ob)):
        r = subprocess.run(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                            *[f"{x:.4f}" for x in pf], '--quat', *[f"{x:.6f}" for x in qf], '--min-z', '0.15'],
                           cwd=str(REPO), capture_output=True, text=True, timeout=90)
        if r.returncode != 0:
            print(f"  {lab}: move refused", flush=True); continue
        time.sleep(2.0)
        c = capture()
        if c is None:
            print(f"  {lab}: no still capture", flush=True); continue
        d, K, tcp = c
        np.save(out / f'depth_{i:02d}.npy', d.astype(np.uint16))
        meta['views'].append({'i': i, 'label': lab, 'K': K.tolist(), 'tcp': tcp.tolist()})
        print(f"  {lab}: saved", flush=True)
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
    subprocess.run(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                    f"{ob[0][0]:.4f}", f"{ob[0][1]:.4f}", '0.56', '--quat', '0', '1', '0', '0'], cwd=str(REPO),
                   capture_output=True, timeout=90)
    print(f"survey: {len(meta['views'])} views -> {out}")
    return out


def load_views(d):
    meta = json.load(open(d / 'meta.json'))
    V = []
    for v in meta['views']:
        dep = np.load(d / f"depth_{v['i']:02d}.npy").astype(np.float32)[::2, ::2] / 1000.0
        K = np.array(v['K']); K[:2] /= 2.0
        vv, uu = np.nonzero(dep > 0.1); z = dep[vv, uu]
        Pc = np.c_[(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z]
        V.append(dict(label=v['label'], Pc=Pc, Ttcp=T(v['tcp'][:3], v['tcp'][3:])))
    return meta, V


def solve(a):
    from seat_check import X_CAM as X0
    from base_icp import model_points
    d = Path(a.dir); meta, V = load_views(d)
    base = meta['base']; seated = meta['seated']
    M = model_points(base, seated); tree = cKDTree(M)
    (bx, by, bz), bq = meta['base_marker_start']
    p0 = np.array([0, 0, 0, 0, 0, 0, bx, by, float(R.from_quat(bq).as_euler('xyz')[2]), 0.0])   # D(rotvec, t), base x, y, yaw, dz

    def world(p, v):
        D = T(p[3:6], R.from_rotvec(p[:3]).as_quat())
        Tc = v['Ttcp'] @ X0 @ D
        return v['Pc'] @ Tc[:3, :3].T + Tc[:3, 3]

    def to_base(p, Pw):
        c, s = np.cos(p[8]), np.sin(p[8])
        xy = (Pw[:, :2] - p[6:8]) @ np.array([[c, -s], [s, c]])
        return np.c_[xy, Pw[:, 2] - bz - p[9]]

    rng = np.random.default_rng(0)
    # data points near the base, above the table, per view (subsampled)
    for v in V:
        Pw = world(p0, v)
        table = np.percentile(Pw[:, 2], 30)
        m = (np.hypot(Pw[:, 0] - bx, Pw[:, 1] - by) < 0.13) & (Pw[:, 2] > table + 0.004) & (Pw[:, 2] < table + 0.12)
        idx = np.nonzero(m)[0]
        v['Pc'] = v['Pc'][rng.choice(idx, min(len(idx), 2500), replace=False)]

    def per_view_fit(p):
        """base pose per view with the mount p (x, y, yaw), by the same ICP as base_icp, for the spread report."""
        from base_icp import fit
        rows = []
        for v in V:
            Pw = world(p, v)
            pose, info = fit(base, seated, Pw, ((p[6], p[7], bz), R.from_euler('z', p[8]).as_quat()))
            if pose:
                rows.append([1000 * pose[0][0], 1000 * pose[0][1], float(np.degrees(R.from_quat(pose[1]).as_euler('xyz')[2])),
                             info['dz_mm']])
        return np.array(rows)

    p = p0.copy()
    for gate in (0.015, 0.008, 0.004, 0.004, 0.003):
        corr = []
        for v in V:
            Pb = to_base(p, world(p, v))
            dd, ii = tree.query(Pb, distance_upper_bound=gate)
            ok = np.isfinite(dd)
            corr.append((ok, M[ii[ok]]))

        def res(q):
            r = []
            for v, (ok, Mi) in zip(V, corr):
                r.append((to_base(q, world(q, v)[ok]) - Mi).ravel())
            r.append(np.r_[q[:3] / np.radians(5.0), q[3:6] / 0.02] * 1e-3)       # weak prior: D small
            return np.concatenate(r)
        sol = least_squares(res, p, loss='soft_l1', f_scale=0.002, x_scale=[0.01] * 3 + [0.005] * 3 + [0.005, 0.005, 0.01, 0.002])
        p = sol.x
        print(f"  gate {1000*gate:.0f} mm: rms {1000*np.sqrt(np.mean(sol.fun[:-6]**2)):.2f} mm, "
              f"D rot {np.round(np.degrees(p[:3]), 3).tolist()} deg, trans {np.round(1000*p[3:6], 2).tolist()} mm", flush=True)
    before = per_view_fit(np.r_[np.zeros(6), p[6:]]); after = per_view_fit(p)
    for lab, B in (('before', before), ('after', after)):
        dxy = np.hypot(*(B[:, :2] - np.median(B[:, :2], 0)).T)
        print(f"{lab:6s}: base xy spread median {np.median(dxy):.1f} max {dxy.max():.1f} mm | yaw range {np.ptp(B[:, 2]):.2f} deg | "
              f"dz range {np.ptp(B[:, 3]):.1f} mm  ({len(B)} views)")
    X = X0 @ T(p[3:6], R.from_rotvec(p[:3]).as_quat())
    res = {'cam_offset_position': np.round(X[:3, 3], 5).tolist(), 'cam_offset_quat': np.round(R.from_matrix(X[:3, :3]).as_quat(), 5).tolist(),
           'D_rot_deg': np.round(np.degrees(p[:3]), 3).tolist(), 'D_trans_mm': np.round(1000 * p[3:6], 2).tolist(),
           'base': [round(1000 * p[6], 1), round(1000 * p[7], 1), round(float(np.degrees(p[8])), 2)],
           'before': before.tolist(), 'after': after.tolist()}
    json.dump(res, open(d / 'solve.json', 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('cam_offset_position', 'cam_offset_quat', 'D_rot_deg', 'D_trans_mm', 'base')}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    s1 = sp.add_parser('survey'); s1.add_argument('--base', required=True); s1.add_argument('--seated', nargs='*', default=[])
    s2 = sp.add_parser('solve'); s2.add_argument('dir')
    a = ap.parse_args()
    if a.cmd == 'survey':
        survey(a)
    else:
        solve(a)


if __name__ == '__main__':
    main()
