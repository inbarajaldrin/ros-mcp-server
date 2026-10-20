#!/usr/bin/env python3
"""How long after the arm stops is the wrist D415's depth reliable, and is its timestamp right? (2026-10-20)

Operator: "do some deterministic analysis on a moving target and how long it takes to calibrate or get a
reliable reading - it takes time for depth to update; are you working around it instead of tackling it".
Every depth read so far waited a guessed 1-2 s after a move.

The base is static, so a correctly stamped, settled depth frame put through the TCP pose AT ITS STAMP must
give the same base pose whatever the arm does. This:
  1. records every aligned depth frame (stamp, arrival time) and the 500 Hz TCP pose while the arm moves
     between views over the base (and N s after each stop)
  2. fits the base CAD to each frame (6 dof, camera frame) and maps it to the world with the TCP pose at
     stamp + delta for a sweep of delta: the delta that makes the moving frames agree is the stamp's latency
  3. reports, per stop: when the TCP stopped, the first frame stamped after it, and when the per-frame base
     pose comes and stays within SETTLE_MM / SETTLE_DEG of its final value (the settle time)

    python3 -u ablations/depth_timing.py --base base3 [--moves 4] [--hold 6]
Writes ablations/logs/depth_timing/<ts>/result.json.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R, Slerp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))
OUT = REPO / 'ablations/logs/depth_timing'
SETTLE_MM, SETTLE_DEG = 0.5, 0.1


def record(stop_evt, buf):
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image
    rclpy.init(); n = rclpy.create_node('depth_timing'); br = CvBridge()
    def on_d(m):
        d = br.imgmsg_to_cv2(m, 'passthrough')[::2, ::2].copy()
        buf['d'].append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, time.time(), d))
    n.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', on_d, 50)
    n.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', lambda m: buf.__setitem__('K', np.array(m.k).reshape(3, 3)), 2)
    n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: buf['t'].append(
        (m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, time.time(), m.pose.position.x, m.pose.position.y, m.pose.position.z,
         m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w)), 1000)
    while not stop_evt.is_set():
        rclpy.spin_once(n, timeout_sec=0.01)
    n.destroy_node(); rclpy.shutdown()


def tcp_at(T, t):
    """TCP pose (4x4) at time t by interpolation of the stamped samples T (N x 9: stamp, arrival, xyz, quat)."""
    i = int(np.clip(np.searchsorted(T[:, 0], t), 1, len(T) - 1))
    t0, t1 = T[i - 1, 0], T[i, 0]
    a = 0.0 if t1 <= t0 else float(np.clip((t - t0) / (t1 - t0), 0, 1))
    p = (1 - a) * T[i - 1, 2:5] + a * T[i, 2:5]
    q = Slerp([0, 1], R.from_quat([T[i - 1, 5:9], T[i, 5:9]]))([a]).as_quat()[0]
    M = np.eye(4); M[:3, :3] = R.from_quat(q).as_matrix(); M[:3, 3] = p
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True)
    ap.add_argument('--moves', type=int, default=4)
    ap.add_argument('--hold', type=float, default=6.0)
    a = ap.parse_args()
    from replay_real_assembly import observe_base_pose
    from seat_check import X_CAM as X0
    from base_icp import model_points
    from handeye_icp_survey import fit6
    ob = observe_base_pose(a.base)
    if ob is None:
        sys.exit(f"{a.base} not observed")
    (bx, by, bz), bq = ob
    views = [(bx, by, 0.56), (bx + 0.08, by, 0.56), (bx, by + 0.06, 0.52), (bx - 0.06, by - 0.04, 0.60), (bx, by, 0.56)]
    go = lambda v: subprocess.run(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                                   *[f"{x:.4f}" for x in v], '--quat', '0', '1', '0', '0'], cwd=str(REPO),
                                  capture_output=True, text=True, timeout=90)
    go(views[0]); time.sleep(2.0)
    buf = {'d': [], 't': []}; ev = threading.Event()
    th = threading.Thread(target=record, args=(ev, buf), daemon=True); th.start()
    time.sleep(a.hold)
    for v in views[1:a.moves + 1]:
        go(v); time.sleep(a.hold)
    ev.set(); th.join(timeout=5)
    T = np.array(buf['t']); K = buf['K'].copy(); K[:2] /= 2.0
    print(f"recorded {len(buf['d'])} depth frames, {len(T)} TCP samples over {T[-1,0]-T[0,0]:.1f} s", flush=True)
    # stamps: arrival - stamp
    lat_arr = np.array([arr - st for st, arr, _ in buf['d']])
    print(f"depth arrival - stamp: median {1000*np.median(lat_arr):.0f} ms (p90 {1000*np.percentile(lat_arr, 90):.0f})", flush=True)
    # motion segments from the TCP speed
    v = np.linalg.norm(np.diff(T[:, 2:5], axis=0), axis=1) / np.maximum(np.diff(T[:, 0]), 1e-4)
    moving = np.r_[False, v > 0.001]
    stops = [T[i, 0] for i in range(1, len(T)) if moving[i - 1] and not moving[i] and not moving[min(i + 250, len(T) - 1)]]
    # per-frame camera-frame fits (independent of the TCP), then world poses for a sweep of delta
    M = model_points(a.base, []); tree = cKDTree(M)
    rng = np.random.default_rng(0)
    fits = []
    Tbw0 = np.eye(4); Tbw0[:3, :3] = R.from_quat(bq).as_matrix(); Tbw0[:3, 3] = [bx, by, bz]
    for st, arr, d in buf['d']:
        z = d.astype(np.float32) / 1000.0; vv, uu = np.nonzero(z > 0.1); zz = z[vv, uu]
        Pc = np.c_[(uu - K[0, 2]) / K[0, 0] * zz, (vv - K[1, 2]) / K[1, 1] * zz, zz]
        Tw = tcp_at(T, st) @ X0
        Pw = Pc @ Tw[:3, :3].T + Tw[:3, 3]
        table = np.percentile(Pw[:, 2], 30)
        m = (np.hypot(Pw[:, 0] - bx, Pw[:, 1] - by) < 0.13) & (Pw[:, 2] > table + 0.004) & (Pw[:, 2] < table + 0.12)
        idx = np.nonzero(m)[0]
        if len(idx) < 1500:
            fits.append(None); continue
        Tcb, rms, n = fit6(Pc[rng.choice(idx, min(len(idx), 4000), replace=False)], M, tree, np.linalg.inv(Tw) @ Tbw0)
        fits.append(Tcb)
    ok = [i for i, f in enumerate(fits) if f is not None]
    print(f"fitted {len(ok)} of {len(fits)} frames", flush=True)

    def world_poses(delta):
        P = []
        for i in ok:
            Bw = tcp_at(T, buf['d'][i][0] + delta) @ X0 @ fits[i]
            P.append([1000 * Bw[0, 3], 1000 * Bw[1, 3], np.degrees(np.arctan2(Bw[1, 0], Bw[0, 0]))])
        return np.array(P)
    stamps = np.array([buf['d'][i][0] for i in ok])
    mov_f = np.array([np.interp(s, T[1:, 0], v) > 0.005 for s in stamps])
    sweep = []
    for delta in np.arange(-0.20, 0.201, 0.01):
        P = world_poses(delta)
        if mov_f.sum() >= 3:
            spread = float(np.median(np.hypot(*(P[mov_f, :2] - np.median(P[~mov_f, :2], 0)).T)))
            sweep.append((round(float(delta), 3), round(spread, 2)))
    best = min(sweep, key=lambda r: r[1]) if sweep else (0.0, None)
    print(f"stamp latency: moving frames agree best with the TCP at stamp {best[0]*1000:+.0f} ms "
          f"(median xy {best[1]} mm; at 0 ms {dict(sweep).get(0.0)} mm)", flush=True)
    P = world_poses(best[0])
    res = {'n_frames': len(fits), 'arrival_minus_stamp_ms': round(1000 * float(np.median(lat_arr)), 1),
           'stamp_latency_ms': round(1000 * best[0]), 'sweep': sweep, 'stops': []}
    for ts in stops:
        after = [k for k, s in enumerate(stamps) if s > ts]
        if len(after) < 5:
            continue
        fin = np.median(P[after[-5:]], 0)
        err = np.hypot(*(P[after, :2] - fin[:2]).T); eyaw = np.abs(P[after, 2] - fin[2])
        good = (err <= SETTLE_MM) & (eyaw <= SETTLE_DEG)
        k_settle = next((j for j in range(len(after)) if good[j:].all()), None)
        first = stamps[after[0]] - ts
        st = (stamps[after[k_settle]] - ts) if k_settle is not None else None
        res['stops'].append({'first_frame_after_stop_s': round(float(first), 3),
                             'settled_after_s': None if st is None else round(float(st), 3),
                             'err_first_frames_mm': np.round(err[:6], 2).tolist()})
        print(f"stop at {ts - T[0,0]:.1f} s: first frame +{1000*first:.0f} ms, settled (<= {SETTLE_MM} mm, {SETTLE_DEG} deg) "
              f"after {'never' if st is None else f'{st:.2f} s'}; first frames off {np.round(err[:6], 1).tolist()} mm", flush=True)
    out = OUT / time.strftime('%Y%m%d_%H%M%S'); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / 'result.json', 'w'), indent=1)
    print(f"-> {out}/result.json")


if __name__ == '__main__':
    main()
