#!/usr/bin/env python3
"""Insert by guarded touches, no force mode: descend, stop on first contact, retract, correct, try again (2026-10-21).

    python3 -u -m compliant_insertion_studio.wrapper.touch_insert --object-name u_brown --base-name base1 --grasp-id 1 \
        --current-object-orientation QX QY QZ QW --final-base-pos X Y Z --final-base-orientation QX QY QZ QW

Operator 2026-10-21: "setup a system where you don't need compliance to perform the insert because you can retract on
first contact if it didn't go through and fix the reading, make the fix and try again, be it offset or yaw". On an
untaped base the compliant search's lateral push dragged and turned base1 (0.19 -> ~8 deg over two u_brown searches);
a vertical guarded touch puts no lateral load on it.

Each probe: from 3 mm above the rim, a straight descent toward the seat at <= 5 mm/s that stops when |dF| > 4 N
(primitives.move_cartesian --max-force). Reaching the seat height (within SEATED_DZ) = in; stopped on the rim =
retract and move to the next point; stopped well below the rim (> ENTER_DZ) = part of the part is in the pocket, keep
going from there. Points: the aim, then rings of 2/4/6 mm; the ring after the first is centred on the lowest rim
contact so far (the rim reads 0.1-0.25 mm lower near the pocket: 81 search-then-seat inserts). If no xy point goes
in, yaw +-1.5 / +-3 deg at the best point. The seat target is predict_tcp_at_seat (the compliant insert's CAD chain).

Writes insert_<obj>_<ts>.meta.json (predicted_seat_xyz) + an empty .csv so drive-mate judges the result where the arm
rests. Prints __RESULT_JSON__.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'ablations')]
LOGS = REPO / 'compliant_insertion_studio' / 'logs'
MAX_FORCE_N = 4.0
SPEED_M_S = 0.005
ABOVE_RIM_M = 0.003
SEATED_DZ_M = 0.003          # within this of the predicted seat = in
ENTER_DZ_M = 0.004           # stopped this far below the rim contact height = partly in the pocket
RINGS = ((0.002, 6), (0.004, 8), (0.006, 10))
YAWS_DEG = (1.5, -1.5, 3.0, -3.0)


def _env():
    return dict(os.environ, PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}")


def _result(out):
    lines = out.splitlines()
    try:
        i = lines.index('__RESULT_JSON__'); return json.loads(lines[i + 1])
    except Exception:
        return {'result': 'failure', 'error': (lines[-1] if lines else 'no output')}


def move(xyz, quat, max_force=None, min_duration=None, min_z=0.15):
    argv = [sys.executable, '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
            *[f"{v:.5f}" for v in xyz], '--quat', *[f"{v:.6f}" for v in quat], '--min-z', f"{min_z:.3f}"]
    if max_force is not None:
        argv += ['--max-force', f"{max_force:g}"]
    if min_duration is not None:
        argv += ['--min-duration', f"{min_duration:.2f}"]
    p = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True, timeout=120, env=_env())
    return _result(p.stdout)


def tcp_now():
    import rclpy
    from geometry_msgs.msg import PoseStamped
    rclpy.init(); n = rclpy.create_node('touch_insert_tcp'); s = {}
    n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: s.__setitem__('p', m.pose), 5)
    t = time.time()
    while 'p' not in s and time.time() - t < 3:
        rclpy.spin_once(n, timeout_sec=0.05)
    n.destroy_node(); rclpy.shutdown()
    p = s['p']
    return (np.array([p.position.x, p.position.y, p.position.z]),
            np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--object-name', required=True); ap.add_argument('--base-name', required=True)
    ap.add_argument('--grasp-id', type=int, default=1)
    ap.add_argument('--current-object-orientation', type=float, nargs=4, required=True)
    ap.add_argument('--final-base-pos', type=float, nargs=3, required=True)
    ap.add_argument('--final-base-orientation', type=float, nargs=4, required=True)
    a = ap.parse_args()
    from compliant_insertion_studio.wrapper.cad_lookup import predict_tcp_at_seat
    from compliant_insertion_studio.wrapper.cad_geometry import _load_base_top_z_local
    from primitives.shared.config import held_tool_offset

    p0, q0 = tcp_now()
    pred = predict_tcp_at_seat(base_name=a.base_name, object_name=a.object_name, grasp_id=a.grasp_id,
                               base_world_xyz=a.final_base_pos, base_world_quat_xyzw=a.final_base_orientation,
                               held_quat_xyzw=a.current_object_orientation, ee_orientation_xyzw=list(q0),
                               flange_offset_m=float(held_tool_offset()[2]))
    seat = np.array(pred['predicted_tcp_at_seat']['xyz_m'], float)
    q_seat = np.array(pred['predicted_tcp_at_seat']['quat_xyzw'], float)
    # rim contact height: the part's lowest point at the seat sits below the base top by (base top - seat bottom)
    from compliant_insertion_studio.wrapper.cad_geometry import expected_contact_tcp_z
    base_top_w = a.final_base_pos[2] + _load_base_top_z_local(a.base_name)
    try:
        rim_z = expected_contact_tcp_z(surface_world_z=base_top_w, object_name=a.object_name, grasp_id=a.grasp_id,
                                       R_grasp_quat_xyzw=a.current_object_orientation,
                                       predicted_tcp_z_at_seat=float(seat[2]),
                                       flange_offset_world_z=-float(held_tool_offset()[2]))
    except Exception:
        rim_z = float(seat[2]) + 0.031
    ts = time.strftime('%Y%m%d_%H%M%S'); stem = LOGS / f'insert_{a.object_name}_{ts}'
    meta = {'object': a.object_name, 'base': a.base_name, 'grasp_id': a.grasp_id, 'router': 'touch_insert',
            'predicted_seat_xyz': [float(v) for v in seat], 'rim_tcp_z': float(rim_z), 'probes': []}
    open(str(stem) + '.csv', 'w').write('t_s,tcp_x,tcp_y,tcp_z\n')
    print(f"[touch] seat TCP ({1000*seat[0]:.1f}, {1000*seat[1]:.1f}, {1000*seat[2]:.1f}) mm, rim contact "
          f"{1000*(rim_z - seat[2]):+.1f} mm above it", flush=True)

    start_z = rim_z + ABOVE_RIM_M
    # over the aim at the current height first (never move sideways near the base), then down to the probe start
    r = move([seat[0], seat[1], max(p0[2], start_z + 0.03)], q_seat)
    if r.get('result') == 'failure':
        print(f"[touch] could not reach the hover: {r.get('error')}", flush=True); return finish(stem, meta, 'failure')

    def probe(dx, dy, dyaw_deg=0.0):
        q = (R.from_euler('z', dyaw_deg, degrees=True) * R.from_quat(q_seat)).as_quat()
        x, y = seat[0] + dx, seat[1] + dy
        r1 = move([x, y, start_z], q)
        if r1.get('result') == 'failure':
            return None
        dur = (start_z - (seat[2] - 0.002)) / SPEED_M_S
        r2 = move([x, y, seat[2] - 0.002], q, max_force=MAX_FORCE_N, min_duration=dur)
        pz, _ = tcp_now()
        z = float(pz[2])
        rec = {'dx_mm': round(1000 * dx, 2), 'dy_mm': round(1000 * dy, 2), 'dyaw_deg': dyaw_deg,
               'z_above_seat_mm': round(1000 * (z - seat[2]), 2), 'contact': bool(r2.get('contact')),
               'error': r2.get('error')}
        meta['probes'].append(rec)
        print(f"[touch] probe ({rec['dx_mm']:+.1f}, {rec['dy_mm']:+.1f}) mm yaw {dyaw_deg:+.1f}: stopped "
              f"{rec['z_above_seat_mm']:+.1f} mm above the seat" + (" (contact)" if rec['contact'] else ""), flush=True)
        return z

    def outcome(z):
        if z is None:
            return 'error'
        if z - seat[2] <= SEATED_DZ_M:
            return 'in'
        if z < rim_z - ENTER_DZ_M:
            return 'entering'
        return 'rim'

    pts = [(0.0, 0.0)]
    best = None                                         # (z, dx, dy) lowest rim contact
    centre = (0.0, 0.0)
    tried = 0
    for ring_i in range(len(RINGS) + 1):
        if ring_i > 0:
            rad, n = RINGS[ring_i - 1]
            c = (best[1], best[2]) if (best is not None and ring_i > 1) else centre
            pts = [(c[0] + rad * math.cos(2 * math.pi * k / n), c[1] + rad * math.sin(2 * math.pi * k / n)) for k in range(n)]
        for dx, dy in pts:
            z = probe(dx, dy); tried += 1
            st = outcome(z)
            if st == 'in':
                return finish(stem, meta, 'success')
            if st == 'entering':
                # partly in the pocket: it is here - a lighter second try from just above, then hand on to the judge
                z2 = probe(dx, dy)
                return finish(stem, meta, 'success' if outcome(z2) == 'in' else 'entering')
            if st == 'rim' and (best is None or z < best[0]):
                best = (z, dx, dy)
            if z is None:
                return finish(stem, meta, 'failure')
            move([seat[0] + dx, seat[1] + dy, start_z], q_seat)        # retract before any sideways step
    bx, by = (best[1], best[2]) if best else (0.0, 0.0)
    for dyaw in YAWS_DEG:
        z = probe(bx, by, dyaw)
        if outcome(z) == 'in':
            return finish(stem, meta, 'success')
        move([seat[0] + bx, seat[1] + by, start_z], q_seat)
    return finish(stem, meta, 'not_found')


def finish(stem, meta, result):
    meta['outcome'] = result
    json.dump(meta, open(str(stem) + '.meta.json', 'w'), indent=1)
    print('__RESULT_JSON__'); print(json.dumps({'result': 'success' if result == 'success' else 'failure',
                                                'touch_outcome': result, 'n_probes': len(meta['probes']),
                                                'meta': str(stem) + '.meta.json'}))
    print('__END_RESULT_JSON__', flush=True)
    return 0 if result == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
