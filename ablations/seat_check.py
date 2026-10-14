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

X_CAM = np.eye(4)   # localizer's calibrated camera mount (config/robot_config.yaml, 2026-10-13)
X_CAM[:3, :3] = R.from_quat([-0.0238, -0.00386, -0.01645, 0.99957]).as_matrix()
X_CAM[:3, 3] = [-0.01446, -0.04424, 0.07317]
VIEW_FLANGE_Z = 0.56
# Top face in the base frame and the half-extents to sample (CAD, trimmed off the edges).
# line_green: 25.9 x 140 x 25.9 mm bar along base y, seat centre z 24.5 mm -> top 37.45 mm.
TOP_FACE = {'line_green': dict(z=0.03745, hx=0.008, hy=0.062, long_axis='y')}
PITCH_MAX_DEG = 1.5
HEIGHT_MAX_MM = 2.5


def capture_world(n=10, timeout=6.0):
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image
    rclpy.init(); node = rclpy.create_node('seat_check'); st = {'d': []}
    node.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', lambda m: st['d'].append(m), 20)
    node.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', lambda m: st.__setitem__('i', m), 2)
    node.create_subscription(Image, '/camera/camera/color/image_raw', lambda m: st.__setitem__('c', m), 2)
    node.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: st.__setitem__('t', m), 10)
    t_start = time.time() + 0.3            # frames from after the call, not buffered ones
    end = time.time() + timeout
    while time.time() < end and (len([m for m in st['d'] if _stamp(m) > t_start]) < n or 'i' not in st or 't' not in st):
        rclpy.spin_once(node, timeout_sec=0.02)
    node.destroy_node(); rclpy.shutdown()
    frames = [m for m in st['d'] if _stamp(m) > t_start][-n:]
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


def measure(obj, observed_base, Pw):
    """Plane fit to the part's top face inside its footprint (base frame)."""
    f = TOP_FACE[obj]
    (bx, by, bz), bq = observed_base
    Tb = np.eye(4); Tb[:3, :3] = R.from_quat(bq).as_matrix(); Tb[:3, 3] = [bx, by, bz]
    Pb = (Pw - Tb[:3, 3]) @ Tb[:3, :3]                         # world -> base frame
    m = (np.abs(Pb[:, 0]) < f['hx']) & (np.abs(Pb[:, 1]) < f['hy']) & (np.abs(Pb[:, 2] - f['z']) < 0.012)
    P = Pb[m]
    if len(P) < 200:
        return {'ok': False, 'error': f'only {len(P)} depth points on the {obj} top face'}
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
    ends = [float(1000 * (c + long_slope * s * (f['hy'] + 0.008) - f['z'])) for s in (-1, 1)]
    ok = abs(pitch) <= PITCH_MAX_DEG and abs(h_err) <= HEIGHT_MAX_MM and max(map(abs, ends)) <= HEIGHT_MAX_MM + 1.5
    return {'ok': bool(ok), 'pitch_deg': round(pitch, 2), 'roll_deg': round(roll, 2),
            'height_err_mm': round(h_err, 2), 'end_heights_mm': [round(e, 2) for e in ends],
            'residual_mm': round(float(1000 * np.std(r[keep])), 2), 'n_points': int(len(P)),
            'axial': axial, 'profile_mm': {'bins': bins.tolist(), 'top_z': [None if z is None else round(z, 1) for z in prof]},
            'limits': {'pitch_deg': PITCH_MAX_DEG, 'height_mm': HEIGHT_MAX_MM}}


def view_pose(obj, observed_base):
    from disassemble import SEAT_EE_QUAT
    (bx, by, _), _ = observed_base
    return ['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
            f"{bx:.4f}", f"{by:.4f}", f"{VIEW_FLANGE_Z:.4f}", '--quat', *map(str, SEAT_EE_QUAT)]


def check(obj, observed_base, move=True, out_dir=None):
    if move:
        env = os.environ.copy(); env['PYTHONPATH'] = f"{REPO}:{env.get('PYTHONPATH', '')}"
        p = subprocess.run(view_pose(obj, observed_base), capture_output=True, text=True, timeout=60,
                           env=env, cwd=str(REPO))
        if p.returncode != 0:
            return {'ok': False, 'error': 'view move failed: ' + (p.stdout + p.stderr).strip().splitlines()[-1][:200]}
        time.sleep(1.0)                                        # settle before the depth frames
    cap = capture_world()
    if cap is None:
        return {'ok': False, 'error': 'no depth frames'}
    Pw, d, color, nf = cap
    res = measure(obj, observed_base, Pw)
    res['n_frames'] = nf
    if out_dir:
        import cv2
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        np.save(out_dir / f'seatcheck_{obj}_{ts}_depth.npy', d.astype(np.float32))
        if color is not None:
            cv2.imwrite(str(out_dir / f'seatcheck_{obj}_{ts}_color.png'), color)
        json.dump(dict(res, observed_base=observed_base), open(out_dir / f'seatcheck_{obj}_{ts}.json', 'w'), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--object', default='line_green', choices=sorted(TOP_FACE))
    ap.add_argument('--no-move', action='store_true', help='measure from where the arm is')
    ap.add_argument('--out-dir', default=None)
    a = ap.parse_args()
    from replay_real_assembly import observe_base_pose
    from disassemble import observe_base
    ob = observe_base_pose('base1') if a.no_move else observe_base('base1')
    if ob is None:
        print(json.dumps({'ok': False, 'error': 'base1 not observed'})); return 1
    res = check(a.object, ob, move=not a.no_move, out_dir=a.out_dir)
    print(json.dumps(res))
    return 0 if res.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
