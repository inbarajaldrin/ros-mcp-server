#!/usr/bin/env python3
"""base1's pose from depth: fit its top-face outline (2026-10-15).

    python3 -u ablations/base_depth.py          # arm already at the base view (flange 0.56 m)

The marker read of base1 moved 55-61 mm in x between viewpoints and runs (2026-10-15) while
the parts really seated at ~61 (the U inserts' own search found +3..+6 mm, and line_green, aimed
6.4 mm short, could not reach its slot within its 5 mm search). The depth outline of the base's
top face (CAD: a 161 x 161 mm square centred on base1's origin, top 35 mm above the table)
repeated to ~1 mm from two viewpoints. Inserts aim at this instead.

Needs the camera >= ~0.45 m from the table (D415 min depth range at 1280x720): flange 0.56 m.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))

BASE_SIDE_MM = {'base1': 161.0}
TOP_ABOVE_TABLE_MM = {'base1': 35.0}
VIEW_FLANGE_Z = 0.56
SIZE_TOL_MM = 9.0              # fitted side vs CAD (edge blur makes it read ~+2..5 mm)
MAX_SHIFT_MM = 15.0            # depth vs marker centre; more means the fit caught something else


def measure(marker_ob, base='base1'):
    """((x, y, z), quat) with x, y and yaw from the depth outline; z from the marker read.
    Returns (pose, info) — pose is None when the fit fails its checks."""
    import cv2
    from seat_check import capture_world
    (mx, my, mz), mq = marker_ob
    cap = capture_world(n=10)
    if cap is None:
        return None, {'error': 'no depth frames'}
    Pw = cap[0]
    try:                                    # keep the cloud: seated-part geometry is read from it later
        out = REPO / 'ablations/logs/base_depth'; out.mkdir(parents=True, exist_ok=True)
        near = (np.abs(Pw[:, 0] - mx) < 0.15) & (np.abs(Pw[:, 1] - my) < 0.15)
        np.save(out / f"base_{__import__('time').strftime('%Y%m%d_%H%M%S')}.npy", Pw[near].astype(np.float32))
    except Exception:
        pass
    table = np.percentile(Pw[:, 2], 30)
    h = 1000 * (Pw[:, 2] - table)
    top = TOP_ABOVE_TABLE_MM[base]; half = (BASE_SIDE_MM[base] / 2 + 15) / 1000
    m = (np.abs(Pw[:, 0] - mx) < half) & (np.abs(Pw[:, 1] - my) < half) & (np.abs(h - top) < 4)
    xy = (1000 * Pw[m, :2]).astype(np.float32)
    if len(xy) < 5000:
        return None, {'error': f'only {len(xy)} top-face points'}
    (cx, cy), (w, l), ang = cv2.minAreaRect(xy)
    info = {'n': int(len(xy)), 'size_mm': [round(w, 1), round(l, 1)], 'centre_mm': [round(cx, 1), round(cy, 1)],
            'marker_mm': [round(1000 * mx, 1), round(1000 * my, 1)]}
    side = BASE_SIDE_MM[base]
    if max(abs(w - side), abs(l - side)) > SIZE_TOL_MM:
        return None, dict(info, error='outline is not the base square')
    if np.hypot(cx - 1000 * mx, cy - 1000 * my) > MAX_SHIFT_MM:
        return None, dict(info, error='outline too far from the marker read')
    # A square's rectangle angle is defined mod 90: take the one nearest the marker yaw.
    m_yaw = R.from_quat(mq).as_euler('xyz', degrees=True)[2]
    yaw = ang + 90 * round((m_yaw - ang) / 90)
    info['yaw_deg'] = round(float(yaw), 2); info['marker_yaw_deg'] = round(float(m_yaw), 2)
    q = R.from_euler('z', yaw, degrees=True).as_quat()
    return ((cx / 1000, cy / 1000, mz), tuple(float(v) for v in q)), info


def main():
    from replay_real_assembly import observe_base_pose
    ob = observe_base_pose('base1')
    if ob is None:
        print(json.dumps({'error': 'base1 not observed'})); return 1
    pose, info = measure(ob)
    print(json.dumps(info))
    return 0 if pose else 1


if __name__ == '__main__':
    sys.exit(main())
