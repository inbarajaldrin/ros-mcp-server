#!/usr/bin/env python3
"""Depth check that a seated part is flat and at its seat height (2026-10-14).

    python3 -u ablations/seat_check.py --object line_green [--no-move] [--out-dir DIR]

Views the seat from straight above at flange 0.56 m (the D415 returns no depth closer than
~0.45 m at 1280x720), takes the median of 10 aligned depth frames, and fits a plane to the
part's top face inside its CAD footprint in the camera-observed base frame. Reports pitch
(along the part's long axis — what a bar that pivoted in the jaws shows), roll, and the top-face
height error. Used by replay_real_assembly after line_green's insert; the result decides whether
to re-grasp (with the insert's own grasp) and reinsert.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'ablations'))

def _load_x_cam():
    """The wrist camera mount from the localizer's config - the single source (the depth path used to carry its
    own copy, and after the 2026-10-20 recalibration markers and depth disagreed by 10 mm)."""
    import yaml
    c = yaml.safe_load(open('/home/aldrin/Desktop/ros2_ws/src/aruco_camera_localizer/config/robot_config.yaml'))
    X = np.eye(4)
    X[:3, :3] = R.from_quat(c['cam_offset_quat']).as_matrix()
    X[:3, 3] = c['cam_offset_position']
    return X


X_CAM = _load_x_cam()
VIEW_FLANGE_Z = 0.56
# Top face in the base frame and the half-extents to sample (CAD, trimmed off the edges).
# line_green: 25.9 x 140 x 25.9 mm bar along base y, seat centre z 24.5 mm -> top 37.45 mm.
# 2026-10-20: from CAD for any part in any base (seat pose * mesh): the face at the part's highest z,
# its centre (cx, cy) and half-extents trimmed 5 mm across / 8 mm along. line_green gives the old values.
TRIM_CROSS_M, TRIM_LONG_M = 0.005, 0.008


def top_face(obj, base='base1'):
    from grasp_depth import load_mesh
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    V, _ = load_mesh(obj)
    sp, sq = load_seat_pose_in_base(base, obj)
    W = V @ R.from_quat(sq).as_matrix().T + sp
    top = float(W[:, 2].max()); T = W[W[:, 2] > top - 1e-3]
    lo, hi = T[:, :2].min(0), T[:, :2].max(0)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    long_axis = 'y' if h[1] >= h[0] else 'x'
    trim = (TRIM_CROSS_M, TRIM_LONG_M) if long_axis == 'y' else (TRIM_LONG_M, TRIM_CROSS_M)
    return dict(z=round(top, 5), cx=float(c[0]), cy=float(c[1]), hx=float(h[0] - trim[0]), hy=float(h[1] - trim[1]),
                long_axis=long_axis, base=base, half_cross_m=float(min(h)), half_long_m=float(max(h)))


class _TopFaces(dict):
    def __missing__(self, k):                     # k = obj or (obj, base)
        obj, base = (k if isinstance(k, tuple) else (k, 'base1'))
        v = top_face(obj, base); self[k] = v; return v


TOP_FACE = _TopFaces()
STILL_MARGIN_S = 0.25        # use depth frames stamped this long after the TCP stopped (depth_timing 2026-10-20)
ARRIVAL_WINDOW_S = 1.2       # frames ARRIVING this long after the TCP stopped (camera stamps drift up to ~1 s)
PITCH_MAX_DEG = 1.5
HEIGHT_MAX_MM = 2.5


def capture_world(n=10, timeout=6.0):
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image
    rclpy.init(); node = rclpy.create_node('seat_check'); st = {'d': []}
    node.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', lambda m: st['d'].append((time.time(), m)), 20)
    node.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', lambda m: st.__setitem__('i', m), 2)
    node.create_subscription(Image, '/camera/camera/color/image_raw', lambda m: st.__setitem__('c', m), 2)
    node.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: st.__setitem__('t', m), 10)
    # Frames stamped >= STILL_MARGIN_S after the TCP last moved (depth_timing 2026-10-20: frames taken while the
    # arm moves are ~10 mm off and no time shift fixes them; after the stop they are right at once, but arrive
    # 150 ms (median) to 650 ms (p90) after their stamp - the old fixed sleeps were waiting for those).
    # 2026-10-20: the RealSense stamps drift -0.5..+1.1 s against this PC's clock (global time under load average
    # ~19), so stillness is judged by ARRIVAL time: a frame counts if the TCP was still for ARRIVAL_WINDOW_S before
    # it arrived (covers the worst measured delay)
    st['mv'] = {'ref': None, 't': None}         # last time the TCP moved > 0.5 mm (arrival clock, from the first sample)
    def on_tcp(m):
        now = time.time(); p = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])
        if st['mv']['ref'] is None or np.linalg.norm(p - st['mv']['ref']) > 0.0005:
            st['mv']['ref'] = p; st['mv']['t'] = now
    node.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', on_tcp, 100)
    end = time.time() + timeout + 3.0
    def still_since():
        return st['mv']['t']
    t_call = time.time() + 0.05
    t_start = t_call
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.02)
        ss = still_since()
        t_start = max(t_call, ss + ARRIVAL_WINDOW_S) if ss is not None else t_call + ARRIVAL_WINDOW_S
        if len([m for ta, m in st['d'] if ta > t_start]) >= n and 'i' in st and 't' in st:
            break
    node.destroy_node(); rclpy.shutdown()
    frames = [m for ta, m in st['d'] if ta > t_start][-n:]
    if len(frames) < 3 or 'i' not in st or 't' not in st:
        return None
    br = CvBridge()
    d = np.median(np.stack([br.imgmsg_to_cv2(m, 'passthrough').astype(float) for m in frames]), 0) / 1000.0
    K = np.array(st['i'].k).reshape(3, 3)
    p, o = st['t'].pose.position, st['t'].pose.orientation
    Tw = np.eye(4); Tw[:3, :3] = R.from_quat([o.x, o.y, o.z, o.w]).as_matrix(); Tw[:3, 3] = [p.x, p.y, p.z]
    Tc = Tw @ X_CAM
    v, u = np.nonzero(d > 0.1); z = d[v, u]
    Pc = np.c_[(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z]
    color = br.imgmsg_to_cv2(st['c'], 'bgr8') if 'c' in st else None
    return Pc @ Tc[:3, :3].T + Tc[:3, 3], d, color, len(frames)


def _stamp(m):
    return m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec if m.header.stamp.sec else time.time()


def measure(obj, observed_base, Pw, base='base1'):
    """Plane fit to the part's top face inside its footprint (base frame, centred on the CAD top face)."""
    f = TOP_FACE[(obj, base)]
    (bx, by, bz), bq = observed_base
    Tb = np.eye(4); Tb[:3, :3] = R.from_quat(bq).as_matrix(); Tb[:3, 3] = [bx, by, bz]
    Pb = (Pw - Tb[:3, 3]) @ Tb[:3, :3]                         # world -> base frame
    # heights against the base top's own plane (fitted in the same cloud), not the base read's z: the base
    # read snaps z to the table, and a view-dependent depth tilt cancels when both come from one frame
    from compliant_insertion_studio.wrapper.cad_geometry import _load_base_top_z_local
    bt = _load_base_top_z_local(base)
    mb = (np.abs(Pb[:, 0]) < 0.078) & (np.abs(Pb[:, 1]) < 0.078) & (np.abs(Pb[:, 2] - bt) < 0.004)
    if mb.sum() > 2000:
        Bp = Pb[mb]
        cb, *_ = np.linalg.lstsq(np.c_[Bp[:, 0], Bp[:, 1], np.ones(len(Bp))], Bp[:, 2], rcond=None)
        Pb = Pb.copy(); Pb[:, 2] -= (cb[0] * Pb[:, 0] + cb[1] * Pb[:, 1] + cb[2] - bt)
    Pb = Pb - np.array([f['cx'], f['cy'], 0.0])                # ... about the part's own top-face centre
    m = (np.abs(Pb[:, 0]) < f['hx']) & (np.abs(Pb[:, 1]) < f['hy']) & (np.abs(Pb[:, 2] - f['z']) < 0.012)
    P = Pb[m]
    if len(P) < 200:
        # small top face (forks: 21 x 21 mm): no plane, the median height of the points over the CAD top
        # footprint vs CAD (2026-10-20: both forks' telemetry ended before the retry that seated them; this
        # read them at +0.2 / -0.3 mm)
        mm = (np.abs(Pb[:, 0]) < f['hx'] + 0.005) & (np.abs(Pb[:, 1]) < f['hy'] + 0.005) & (Pb[:, 2] > f['z'] - 0.025)
        if mm.sum() < 50:
            return {'ok': False, 'error': f'only {int(mm.sum())} depth points over the {obj} top'}
        h = float(1000 * (np.median(Pb[mm, 2]) - f['z']))
        return {'ok': bool(abs(h) <= HEIGHT_MAX_MM), 'height_err_mm': round(h, 2), 'n_points': int(mm.sum()),
                'mode': 'median height (small top face)', 'face': {k: f[k] for k in ('cx', 'cy', 'z', 'long_axis', 'half_long_m', 'half_cross_m')},
                'limits': {'height_mm': HEIGHT_MAX_MM}}
    for _ in range(3):                                         # trimmed least squares: z = a x + b y + c
        A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
        coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
        r = P[:, 2] - A @ coef
        keep = np.abs(r) < max(0.0015, 2.5 * np.median(np.abs(r)))
        P = P[keep]
    a, b, c = coef
    # Height profile along the long axis, past both ends (2 mm bins, 90th-percentile height in a
    # strip on the bar's centreline): shows WHERE the bar lies and which end rides on what — a bar
    # shifted along its length onto the channel end vs one end raised in place (2026-10-14).
    ax, cx = (1, 0) if f['long_axis'] == 'y' else (0, 1)
    strip = Pb[(np.abs(Pb[:, cx]) < 0.006) & (np.abs(Pb[:, ax]) < 0.095)]
    bins = np.arange(-90, 91, 2)
    prof = [float(np.percentile(1000 * s[:, 2], 90)) if len(s := strip[np.abs(1000 * strip[:, ax] - b0) < 1.0]) > 5
            else None for b0 in bins]
    on = [b0 for b0, z in zip(bins, prof) if z is not None and z > 1000 * f['z'] - 6.0]
    axial = dict(extent_mm=[int(min(on)), int(max(on))], shift_mm=round((min(on) + max(on)) / 2, 1)) if on else {}
    long_slope, cross_slope = (b, a) if f['long_axis'] == 'y' else (a, b)
    pitch = float(np.degrees(np.arctan(long_slope)))
    roll = float(np.degrees(np.arctan(cross_slope)))
    h_err = float(1000 * (c - f['z']))                         # top-face height at the seat centre
    ends = [float(1000 * (c + long_slope * s * f['half_long_m'] - f['z'])) for s in (-1, 1)]
    # across the part: where its top face sits (centroid of the points at its height, top-face frame) and
    # the cross profile - which side is high / which way it is shifted (2026-10-20, FMB3 lines)
    band = Pb[(np.abs(Pb[:, ax]) < f['half_long_m'] - 0.01) & (np.abs(Pb[:, cx]) < f['half_cross_m'] + 0.012)
              & (Pb[:, 2] > f['z'] - 0.006) & (Pb[:, 2] < f['z'] + 0.012)]
    cross = {}
    if len(band) > 100:
        xs = 1000 * band[:, cx]
        cross = dict(extent_mm=[round(float(np.percentile(xs, 2)), 1), round(float(np.percentile(xs, 98)), 1)],
                     centre_mm=round(float((np.percentile(xs, 2) + np.percentile(xs, 98)) / 2), 1),
                     cad_half_mm=round(1000 * f['half_cross_m'], 1),
                     edge_heights_mm=[round(float(np.median(1000 * band[xs < np.percentile(xs, 15), 2]) - 1000 * f['z']), 1),
                                      round(float(np.median(1000 * band[xs > np.percentile(xs, 85), 2]) - 1000 * f['z']), 1)])
        # per end: the two long edges' heights (what a bar riding a channel lip at one end shows: that end
        # high AND its edge on the lip side high - 2026-10-20 line_brown, -y end, -x edge +2 mm)
        per_end = []
        for sgn_ in (-1, 1):
            e = band[np.abs(band[:, ax] - sgn_ * (f['half_long_m'] - 0.022)) < 0.010]
            if len(e) < 40:
                per_end.append(None); continue
            xe = 1000 * e[:, cx]; ce = (np.percentile(xe, 2) + np.percentile(xe, 98)) / 2
            lo_, hi_ = e[xe < ce - 4.0], e[xe > ce + 4.0]
            per_end.append([round(float(np.percentile(1000 * lo_[:, 2], 80) - 1000 * f['z']), 1) if len(lo_) > 10 else None,
                            round(float(np.percentile(1000 * hi_[:, 2], 80) - 1000 * f['z']), 1) if len(hi_) > 10 else None])
        cross['end_edge_heights_mm'] = per_end        # [[-cross edge, +cross edge] at the -long end, ... at the +long end]
    ok = abs(pitch) <= PITCH_MAX_DEG and abs(h_err) <= HEIGHT_MAX_MM and max(map(abs, ends)) <= HEIGHT_MAX_MM + 1.5
    return {'ok': bool(ok), 'pitch_deg': round(pitch, 2), 'roll_deg': round(roll, 2),
            'height_err_mm': round(h_err, 2), 'end_heights_mm': [round(e, 2) for e in ends],
            'residual_mm': round(float(1000 * np.std(r[keep])), 2), 'n_points': int(len(P)),
            'axial': axial, 'cross': cross, 'face': {k: f[k] for k in ('cx', 'cy', 'z', 'long_axis', 'half_long_m', 'half_cross_m')},
            'profile_mm': {'bins': bins.tolist(), 'top_z': [None if z is None else round(z, 1) for z in prof]},
            'limits': {'pitch_deg': PITCH_MAX_DEG, 'height_mm': HEIGHT_MAX_MM}}


VIEW_TILT_DEG = 25.0         # 2026-10-20: straight down, the overhead lights reflect off glossy tops (line_red read
VIEW_DIST_M = 0.47           # 3 mm high where it was seated); tilted 25 deg and close (D415 min ~0.45 m) it read flat


def view_pose(obj, observed_base, base='base1'):
    """Close, tilted view of the part: camera aimed at its CAD top-face centre from VIEW_DIST_M, tilted
    VIEW_TILT_DEG, from the part's outer side across its long axis (out of the lights' mirror direction)."""
    from tilted_seat_view import aimed_flange
    f = TOP_FACE[(obj, base)]
    (bx, by, bz), bq = observed_base
    Rb = R.from_quat(bq)
    tgt = np.array([bx, by, bz]) + Rb.apply([f['cx'], f['cy'], f['z']])
    cross = 0 if f['long_axis'] == 'y' else 1
    c = (f['cx'], f['cy'])[cross]
    d = np.zeros(3); d[cross] = 1.0 if c > 0.005 else -1.0
    pf, qf, _ = aimed_flange(tgt, Rb.apply(d), VIEW_TILT_DEG, VIEW_DIST_M, X_CAM, float(Rb.as_euler('xyz')[2]))
    return ['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
            *[f"{v:.4f}" for v in pf], '--quat', *[f"{v:.6f}" for v in qf], '--min-z', '0.15']


def check(obj, observed_base, move=True, out_dir=None, base='base1'):
    if move:
        env = os.environ.copy(); env['PYTHONPATH'] = f"{REPO}:{env.get('PYTHONPATH', '')}"
        p = subprocess.run(view_pose(obj, observed_base, base), capture_output=True, text=True, timeout=60,
                           env=env, cwd=str(REPO))
        if p.returncode != 0:
            return {'ok': False, 'error': 'view move failed: ' + (p.stdout + p.stderr).strip().splitlines()[-1][:200]}
        time.sleep(1.0)                                        # settle before the depth frames
        # Same view as the base read: re-fit the base outline so the part is measured against
        # where the base is NOW (the arm's 7th-axis slider creeps under the insert's push).
        from base_depth import measure as base_from_depth
        ob_d, info = base_from_depth(observed_base, base)
        if ob_d is not None:
            observed_base = ob_d
    cap = capture_world()
    if cap is None:
        return {'ok': False, 'error': 'no depth frames'}
    Pw, d, color, nf = cap
    res = measure(obj, observed_base, Pw, base)
    res['n_frames'] = nf
    res['observed_base'] = observed_base
    if out_dir:
        import cv2
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        np.save(out_dir / f'seatcheck_{obj}_{ts}_depth.npy', d.astype(np.float32))
        if color is not None:
            cv2.imwrite(str(out_dir / f'seatcheck_{obj}_{ts}_color.png'), color)
        json.dump(res, open(out_dir / f'seatcheck_{obj}_{ts}.json', 'w'), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--object', default='line_green')
    ap.add_argument('--base', default='base1')
    ap.add_argument('--no-move', action='store_true', help='measure from where the arm is')
    ap.add_argument('--out-dir', default=None)
    a = ap.parse_args()
    from replay_real_assembly import observe_base_pose
    from disassemble import observe_base
    ob = observe_base_pose(a.base) if a.no_move else observe_base(a.base)
    if ob is None:
        print(json.dumps({'ok': False, 'error': f'{a.base} not observed'})); return 1
    res = check(a.object, ob, move=not a.no_move, out_dir=a.out_dir, base=a.base)
    print(json.dumps(res))
    return 0 if res.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
