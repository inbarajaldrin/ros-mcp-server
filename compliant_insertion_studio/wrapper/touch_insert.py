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
PUSH_SPEED_M_S = 0.008       # the entering push to the seat, guarded at 6 N (was a fixed 8 s)
PUSH_MIN_S = 2.0
ABOVE_RIM_M = 0.003          # first probe; later ones start 1.5 mm over the MEASURED rim (CAD rim read 1.3 mm low)
SEATED_DZ_M = 0.003          # within this of the predicted seat = in
SEATED_DZ_WEDGE_M = 0.001    # ... for a part CAD says can wedge (line_green: an end on the base top holds its grasp point
                             # +2.3..3.5 mm up; its clean seats push to -0.5..-2.0 mm)
ENTER_DZ_M = 0.004           # stopped this far below the rim contact height = partly in the pocket
# Probe grid (2026-10-21 23:10): a hexagonal lattice around the aim, nearest first. u_brown's pocket admits the part
# within about +-1 x +-0.5 mm, so the lattice pitch (covering radius pitch/sqrt(3)) must stay ~1 mm; the old 2/4/6 mm
# rings could step over it. The aim comes from the hover servo (repeats to ~0.5 mm across runs).
# 2026-10-21 23:50: with the aim from the hover servo on the SAME-FRAME marker pose (within ~0.5 mm / 0.5 deg) a 0.5 mm
# lattice within 1.5 mm (19 dips, ~2.5 min) entered on dip 17; the 1 mm / 3 mm grid (41 dips) sat 4.9 mm from the
# pocket and could not. Yaw dips are off by default: twisting a pressed part slips it in the pads (TOUCH_YAWS_DEG=0.5,-0.5).
# 10-22 03:55: measured from the clear height (seat on the far side of the base, markers out of view lower down) the
# servo's aim for u_orange was 2.1 mm from the pocket - a 0.7 mm grid out to 3 mm entered at (-1.1, -1.8) mm on dip 31.
# Out to 2.5 mm at 0.6 mm pitch (nearest first, so a close aim still enters early: u_brown on dips 0-17)
GRID_PITCH_M = float(os.environ.get('TOUCH_PITCH_MM', '0.6')) / 1000.0
GRID_RMAX_M = float(os.environ.get('TOUCH_RMAX_MM', '2.5')) / 1000.0
CORRIDOR_PITCH_M = 0.0005
CORRIDOR_ALONG_M = 0.003
CORRIDOR_ACROSS_M = 0.0025      # retracts ride over the measured contact (guarded), so across steps are safe
# A part CAD says can wedge (FMB1: line_green) searches ALONG its own long axis: base1's pocket gives the 140 mm bar
# +-1.4 mm along (+-0.8..1.4 across), and 59 of 74 line_green inserts (10-14..10-23) were released with one end on the
# base top (+7 mm), the bar 2-8 mm off along its axis - never touching anything: its rim is only 7 mm over the seat, so
# the descent's end (rim - 5 mm = seat + 2 mm) counted as 'in'. Steps under twice the play, out to the errors seen.
WEDGE_ALONG_PITCH_M = 0.002
WEDGE_ALONG_M = 0.006
PUSH_SPRING_N = 3.0          # a push that reaches the floor WITHOUT its 6 N guard firing but carried this much: an end bearing
                             # on something while the bar pivots in the jaws (10-23 wrench traces: line_green wedged in run 61
                             # rose 1.0 N/mm from +1.3 mm to 4.5 N; its clean push in run 60 peaked 1.4 N; seats that bottom
                             # rise 18-34 N/mm and trip the guard - u_brown/u_orange/yellow, 6 pushes). Wedge search only.



ENTRY_PRIOR_N, ENTRY_PRIOR_MIN, ENTRY_PRIOR_MAD_M = 3, 3, 0.001   # last 3 (was 6): 10-23 22:11 on, line_green entered ~2 mm off the 6-median (-2.6, -1.0) and took 72-93 probes; the last 3 put it at (-0.75, -0.67)


def _entry_prior(obj):
    """(dx, dy) m: the median entry offset of this part's last successful inserts (their 'push' probe), when there are
    at least ENTRY_PRIOR_MIN and they agree (MAD <= 1 mm on each axis); else None."""
    import glob as _g
    offs = []
    for f in sorted(_g.glob(str(LOGS / f'insert_{obj}_*.meta.json')), key=os.path.getmtime, reverse=True):
        try:
            m = json.load(open(f))
        except Exception:
            continue
        if m.get('outcome') != 'success':
            continue
        ent = [p_ for p_ in m.get('probes', []) if p_.get('push')]
        # an insert that entered without a flagged push (10-23 21:51 / 22:11: seated, no 'push' probe) entered at its first
        # probe below the rim contact - those two were the line_green entries that had moved ~2 mm off the old prior
        ent = ent or [p_ for p_ in m.get('probes', []) if p_.get('z_above_seat_mm', 99) < 10.0][:1]
        if ent:
            offs.append((ent[-1]['dx_mm'] / 1000.0, ent[-1]['dy_mm'] / 1000.0))
        if len(offs) >= ENTRY_PRIOR_N:
            break
    if len(offs) < ENTRY_PRIOR_MIN:
        return None
    A = np.asarray(offs)
    med = np.median(A, axis=0)
    if np.any(np.median(np.abs(A - med), axis=0) > ENTRY_PRIOR_MAD_M) or np.hypot(*med) < 0.0003:
        return None
    return float(med[0]), float(med[1])

def corridor_grid(axis_xy, pitch, along, across):
    """(dx, dy) points on a rectangle along/across a world unit axis, nearest first."""
    u = np.asarray(axis_xy, float) / np.linalg.norm(axis_xy); v = np.array([-u[1], u[0]])
    na, nc = int(round(along / pitch)), int(round(across / pitch))
    # across first, along second (10-22 motion trace: astride the bar the part is held ACROSS it - the yellow entered at
    # 0.0 mm along / 2.0 mm across on probe 46 of the radius order; this order reaches it on probe 9)
    ij = sorted(((i, j) for i in range(-na, na + 1) for j in range(-nc, nc + 1)),
                key=lambda t: (abs(t[0]), abs(t[1]), t[0] < 0, t[1] < 0))
    return [tuple(i * pitch * u + j * pitch * v) for i, j in ij]


def _corridor_axis_world(base, over, base_quat):
    """World xy direction of the long axis of the seated part(s) the held part threads over (CAD footprint)."""
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base, _surface_points
    P = np.vstack([R.from_quat(load_seat_pose_in_base(base, o)[1]).apply(_surface_points(o, 0.003))
                   + load_seat_pose_in_base(base, o)[0] for o in over])
    ext = P[:, :2].max(0) - P[:, :2].min(0)
    ax_b = np.array([1.0, 0.0, 0.0]) if ext[0] >= ext[1] else np.array([0.0, 1.0, 0.0])
    return R.from_quat(base_quat).apply(ax_b)[:2]


YAWS_DEG = tuple(float(v) for v in os.environ['TOUCH_YAWS_DEG'].split(',')) if os.environ.get('TOUCH_YAWS_DEG') else ()


def hex_grid(pitch, rmax):
    """(dx, dy) lattice points within rmax of the origin, nearest first."""
    pts, n = [], int(math.ceil(rmax / pitch)) + 1
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = pitch * (i + 0.5 * j), pitch * (math.sqrt(3) / 2) * j
            if math.hypot(x, y) <= rmax + 1e-9:
                pts.append((x, y))
    return sorted(pts, key=lambda p: (round(math.hypot(*p), 6), math.atan2(p[1], p[0])))


def _env():
    return dict(os.environ, PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}")


def _result(out):
    lines = out.splitlines()
    try:
        i = lines.index('__RESULT_JSON__'); return json.loads(lines[i + 1])
    except Exception:
        return {'result': 'failure', 'error': (lines[-1] if lines else 'no output')}


IN_PROCESS = os.environ.get('TOUCH_INPROC', '1') == '1'
_NODE = {}
SHORT_MOVE_M = 0.005          # probe steps and retracts up to this: timed by distance, not the 2.5 s floor
SHORT_SPEED_M_S = 0.005
SHORT_MIN_S = 0.6
# the guarded descent to the probe start goes no faster than this within NEAR_RIM_M of it: the guard's 51-sample median
# (~100 ms, against the F/T glitch) lags a hard contact, and seated parts stand up to ~40 mm over line_green's rim. Run 82d:
# the in-hand base re-read failed, the arm stayed at 0.56 m and came down 330 mm at 70 mm/s onto a part top 29 mm over the
# rim - -7 to -75 N in 57 ms, robot protective stop before the guard fired; 10-23 21:51 touched the same height at 35 mm/s
# and the guard stopped it at 6.4 N
NEAR_RIM_M = 0.06
NEAR_RIM_SPEED_M_S = 0.015   # mean; the s-curve peaks ~1.9x -> under the 35 mm/s the guard caught


def _node():
    """One ROS node for every probe (10-22 motion trace: 3 processes per probe = ~4.5 s of a 9.5 s probe)."""
    if 'n' not in _NODE:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from primitives.move_cartesian import Mover
        rclpy.init()
        from geometry_msgs.msg import WrenchStamped
        n = Mover(); n.tcp = None; n.W = None
        n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: setattr(n, 'tcp', m.pose), 5)
        # the full wrench (torque too) for the insert's wrench trace - the Mover keeps force only
        n.create_subscription(WrenchStamped, '/force_torque_sensor_broadcaster/wrench', lambda m: setattr(n, 'W', (
            m.wrench.force.x, m.wrench.force.y, m.wrench.force.z, m.wrench.torque.x, m.wrench.torque.y, m.wrench.torque.z)), 50)
        _NODE['n'] = n
    return _NODE['n']


def close_node():
    """Release the probes' ROS context: a caller that goes on to start its own (seat_check.capture_world calls
    rclpy.init) must not find one already up (run 29d: 'Context.init() must only be called once' after the taps)."""
    n = _NODE.pop('n', None)
    if n is not None:
        import rclpy
        try:
            n.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


_TFH = {}   # the insert's timing rows (still audit / chain-of-events trace): one row per move
_WFH = {}   # the insert's wrench trace (<stem>.wrench.csv): every guarded-loop sample of every probe / push (10-23: line_green
            # wedged in runs 58/59 with probes identical to clean runs - the pivot in the jaws shows in torque, not recorded)


def _wrench_sample(label):
    fh = _WFH.get('fh')
    n = _NODE.get('n')

    def rec(t, dF, F, q):
        _WFH['dfmax'] = max(_WFH.get('dfmax', 0.0), float(dF))     # the push's peak load (PUSH_SPRING_N)
        if fh is None:
            return
        W = getattr(n, 'W', None) or (float('nan'),) * 6
        p = getattr(n, 'tcp', None)
        z = p.position.z if p is not None else float('nan')
        fh.write(f"{time.time():.3f},{label},{z:.5f},{dF:.2f}," + ",".join(f"{v:.3f}" for v in W) + "," +
                 ",".join(f"{v:.4f}" for v in q) + "\n")
    return rec


def move(xyz, quat, max_force=None, min_duration=None, min_z=0.15, duration=None):
    """A straight guarded/unguarded move. In process (one node, primitives.move_cartesian.plan_move) unless
    TOUCH_INPROC=0. duration: exact seconds - given for the short probe moves (<= SHORT_MOVE_M) by distance."""
    t0 = time.time()
    try:
        return _move(xyz, quat, max_force, min_duration, min_z, duration)
    finally:
        fh = _TFH.get('fh')
        if fh is not None:
            fh.write(json.dumps({'kind': 'touch_move', 't0': t0, 't1': time.time(),
                                 'label': f"{'guarded' if max_force else 'free'} to z {1000 * float(xyz[2]):.1f}"}) + '\n')


def _move(xyz, quat, max_force=None, min_duration=None, min_z=0.15, duration=None):
    if IN_PROCESS:
        from primitives.move_cartesian import plan_move
        return plan_move(_node(), xyz, ('quat', list(quat)), max_force=max_force, min_duration=min_duration or 0.0,
                         min_z=min_z, duration=duration, on_sample=_wrench_sample(f"to_z_{1000 * float(xyz[2]):.1f}"))
    argv = [sys.executable, '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
            *[f"{v:.5f}" for v in xyz], '--quat', *[f"{v:.6f}" for v in quat], '--min-z', f"{min_z:.3f}"]
    if max_force is not None:
        argv += ['--max-force', f"{max_force:g}"]
    if min_duration is not None:
        argv += ['--min-duration', f"{min_duration:.2f}"]
    p = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True, timeout=120, env=_env())
    return _result(p.stdout)


def tcp_now():
    if IN_PROCESS:
        import rclpy
        n = _node(); n.tcp = None; t = time.time()
        while n.tcp is None and time.time() - t < 3:
            rclpy.spin_once(n, timeout_sec=0.02)
        p = n.tcp
        return (np.array([p.position.x, p.position.y, p.position.z]),
                np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]))
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
    _TFH['fh'] = open(LOGS / f"touch_{a.object_name}_{time.strftime('%Y%m%d_%H%M%S')}.timing.jsonl", 'a', buffering=1)
    from compliant_insertion_studio.wrapper.cad_lookup import predict_tcp_at_seat
    from compliant_insertion_studio.wrapper.cad_geometry import _load_base_top_z_local
    from primitives.shared.config import held_tool_offset_for

    p0, q0 = tcp_now()
    pred = predict_tcp_at_seat(base_name=a.base_name, object_name=a.object_name, grasp_id=a.grasp_id,
                               base_world_xyz=a.final_base_pos, base_world_quat_xyzw=a.final_base_orientation,
                               held_quat_xyzw=a.current_object_orientation, ee_orientation_xyzw=list(q0),
                               flange_offset_m=float(held_tool_offset_for(a.object_name)[2]))
    seat = np.array(pred['predicted_tcp_at_seat']['xyz_m'], float)
    # the hover servo aligned the part to the pocket PHYSICALLY (image yaw); the chain's seat quat would turn it back
    q_seat = np.array(pred['predicted_tcp_at_seat']['quat_xyzw'], float) if os.environ.get('TOUCH_USE_SEAT_QUAT') == '1' else np.asarray(q0, float)
    # rim contact height: the part's lowest point at the seat sits below the base top by (base top - seat bottom)
    from compliant_insertion_studio.wrapper.cad_geometry import expected_contact_tcp_z
    base_top_w = a.final_base_pos[2] + _load_base_top_z_local(a.base_name)
    try:
        rim_z = expected_contact_tcp_z(surface_world_z=base_top_w, object_name=a.object_name, grasp_id=a.grasp_id,
                                       R_grasp_quat_xyzw=a.current_object_orientation,
                                       predicted_tcp_z_at_seat=float(seat[2]),
                                       flange_offset_world_z=-float(held_tool_offset_for(a.object_name)[2]))
    except Exception:
        rim_z = float(seat[2]) + 0.031
    ts = time.strftime('%Y%m%d_%H%M%S'); stem = LOGS / f'insert_{a.object_name}_{ts}'
    meta = {'object': a.object_name, 'base': a.base_name, 'grasp_id': a.grasp_id, 'router': 'touch_insert',
            'predicted_seat_xyz': [float(v) for v in seat], 'rim_tcp_z': float(rim_z), 'probes': []}
    open(str(stem) + '.csv', 'w').write('t_s,tcp_x,tcp_y,tcp_z\n')
    try:
        _WFH['fh'] = open(str(stem) + '.wrench.csv', 'w', buffering=1)
        _WFH['fh'].write('t_s,move,tcp_z,dF,fx,fy,fz,tx,ty,tz,q0,q1,q2,q3,q4,q5\n')
    except Exception:
        _WFH.pop('fh', None)
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
        cur, _ = tcp_now() if IN_PROCESS else (None, None)
        d_ = float(np.linalg.norm(np.array([x, y, start_z]) - cur)) if cur is not None else 1.0
        if cur is not None and cur[2] > start_z + NEAR_RIM_M + 0.005:
            r0 = move([x, y, start_z + NEAR_RIM_M], q, max_force=MAX_FORCE_N)
            if r0.get('result') == 'failure':
                return None
            cur, _ = tcp_now()
            d_ = float(np.linalg.norm(np.array([x, y, start_z]) - cur))
        if d_ > SHORT_MOVE_M:
            r1 = move([x, y, start_z], q, max_force=MAX_FORCE_N, min_duration=d_ / NEAR_RIM_SPEED_M_S,
                      duration=d_ / NEAR_RIM_SPEED_M_S)
        else:
            r1 = move([x, y, start_z], q, max_force=MAX_FORCE_N,          # guarded: 13:39 an unguarded descent here put the
                  duration=max(SHORT_MIN_S, d_ / SHORT_SPEED_M_S))   # yellow on line_green
        if r1.get('result') == 'failure' and not r1.get('contact'):
            return None
        # down only as far as the rim can tell: ENTER_DZ (+1 mm) below the lowest rim known (measured, else CAD) - past it
        # without contact the part is in the pocket mouth and the entering push takes it to the seat. The descent to
        # 2 mm under the seat (36-54 mm) touched after ~1.5 mm, inside its slow start (10-22 motion trace)
        rim_ = min(rim_z, best[0]) if best is not None else rim_z
        z_end = max(float(seat[2]) - 0.002, rim_ - ENTER_DZ_M - 0.001) if IN_PROCESS else float(seat[2]) - 0.002
        dur = (start_z - z_end) / SPEED_M_S
        r2 = move([x, y, z_end], q, max_force=MAX_FORCE_N, min_duration=dur, duration=dur if IN_PROCESS else None)
        pz, _ = tcp_now()
        z = float(pz[2])
        rec = {'dx_mm': round(1000 * dx, 2), 'dy_mm': round(1000 * dy, 2), 'dyaw_deg': dyaw_deg,
               'z_above_seat_mm': round(1000 * (z - seat[2]), 2), 'contact': bool(r2.get('contact')),
               'error': r2.get('error')}
        meta['probes'].append(rec)
        print(f"[touch] probe ({rec['dx_mm']:+.1f}, {rec['dy_mm']:+.1f}) mm yaw {dyaw_deg:+.1f}: stopped "
              f"{rec['z_above_seat_mm']:+.1f} mm above the seat" + (" (contact)" if rec['contact'] else ""), flush=True)
        return z

    def seat_z():
        # The rim contacts MEASURE the grasp depth: the part meets the rim where the chain says (rim - seat) above the
        # seat, so a rim read higher/lower than CAD moves the seat with it (10-22 12:25: u_orange's rim read +34.2 vs
        # CAD +31.5 - held 2.7 mm lower in the jaws; it seated at +5.0 against the uncorrected seat and was called
        # 'entering'). Within -3..+6 mm; the corrected seat goes to the meta for drive-mate's at-rest judge.
        if best is not None and -0.003 <= best[0] - rim_z <= 0.006:
            return float(seat[2] + (best[0] - rim_z))
        return float(seat[2])

    def outcome(z):
        if z is None:
            return 'error'
        if z - seat_z() <= (SEATED_DZ_WEDGE_M if wedge_check else SEATED_DZ_M):
            return 'in'
        if z < rim_z - ENTER_DZ_M:
            return 'entering'
        return 'rim'

    best = None                                         # (z, dx, dy) lowest rim contact
    rmax = GRID_RMAX_M
    try:
        from compliant_insertion_studio.wrapper.cad_geometry import seated_under
        over = seated_under(a.base_name, a.object_name)
    except Exception:
        over = []
    if over:
        # threading over seated parts (CAD): astride them the part is held to +-1 mm ACROSS their long axis, but free ALONG
        # it - so the lattice runs CORRIDOR_ALONG_M along the corridor and CORRIDOR_ACROSS_M across (13:25 yellow: 37 probes
        # within 1.5 mm all on the rim at +2.8 mm, legs astride line_green - the miss was along the bar)
        ax = _corridor_axis_world(a.base_name, over, a.final_base_orientation)
        pts = corridor_grid(ax, CORRIDOR_PITCH_M, CORRIDOR_ALONG_M, CORRIDOR_ACROSS_M)
        rmax = CORRIDOR_ALONG_M
        print(f"[touch] astride {over}: lattice +-{1000*CORRIDOR_ALONG_M:.1f} mm along the corridor "
              f"({np.round(ax, 3).tolist()}), +-{1000*CORRIDOR_ACROSS_M:.1f} across", flush=True)
    else:
        pts = hex_grid(GRID_PITCH_M, rmax)
    try:
        from compliant_insertion_studio.wrapper.cad_geometry import wedge_possible
        wedge_check = bool(wedge_possible(a.base_name, a.object_name))
    except Exception:
        wedge_check = False
    # CAD says it can rest off its seat looking seated (an end on the base top): it is 'in' only within
    # SEATED_DZ_WEDGE_M, and a push held up short steps on along the lattice - also when it threads over seated parts
    # (10-24: line_green, astride the U parts, was accepted at +2.33 mm by the 3 mm test and certify read the end wedge
    # +7.5 mm, pitch -3.3 deg, in runs 87 / 89 / 89b / 90). The lattice change stays for a part not threading over others.
    wedge_search = not over and wedge_check
    if wedge_search:
        ax = _corridor_axis_world(a.base_name, [a.object_name], a.final_base_orientation)
        pts = corridor_grid(ax, WEDGE_ALONG_PITCH_M, WEDGE_ALONG_M, 0.0)
        rmax = WEDGE_ALONG_M
        print(f"[touch] can wedge (CAD): along its own axis {np.round(ax, 3).tolist()} +-{1000*WEDGE_ALONG_M:.0f} mm "
              f"every {1000*WEDGE_ALONG_PITCH_M:.0f} mm, each probe pushed to the seat", flush=True)
    prior = _entry_prior(a.object_name)
    if prior is not None:
        # the search starts where this part's recent inserts entered (a measured prior, not a per-part constant: the
        # lattice is the same, re-centred and re-ordered; the points around the aim stay in it). line_green 10-23: entries
        # 0.5-2.7 mm off the aim, median (-1.9, -0.7) mm, and the lattice spiralled out from the aim (52 probes in run 49)
        px, py = prior
        shifted = [(px + dx, py + dy) for dx, dy in pts]
        rest = [q for q in pts if min(np.hypot(q[0] - s_[0], q[1] - s_[1]) for s_ in shifted) > 0.00025]
        pts = sorted(shifted + rest, key=lambda q: np.hypot(q[0] - px, q[1] - py))
        print(f"[touch] entry prior ({1000*px:+.1f}, {1000*py:+.1f}) mm from the last inserts - the search starts there",
              flush=True)
    print(f"[touch] {len(pts)} probe points, out to {1000*rmax:.1f} mm", flush=True)
    pushes = []                                         # (z after the entering push, dx, dy) - the wedge search's fallback
    for dx, dy in pts:
        z = probe(dx, dy)
        st = outcome(z)
        if st == 'in' and z is not None and not meta['probes'][-1]['contact'] and z > seat_z() + 0.0005:
            st = 'entering'                             # stopped over the seat touching nothing proves nothing: push on
        if st == 'in':
            return finish(stem, meta, 'success')
        if st == 'entering':
            # in the pocket mouth: push on to the seat (6 N guard), then hand on to the judge
            q = (R.from_euler('z', 0.0, degrees=True) * R.from_quat(q_seat)).as_quat()
            # timed by its distance (run 43: a fixed 8 s for ~28 mm, 3.5 mm/s); the 6 N guard still stops it on the floor
            cz_ = float(tcp_now()[0][2]) if IN_PROCESS else float(seat[2]) + 0.03
            _WFH['dfmax'] = 0.0
            r3 = move([seat[0] + dx, seat[1] + dy, seat[2] - 0.002], q, max_force=6.0,
                      min_duration=max(PUSH_MIN_S, (cz_ - float(seat[2]) + 0.002) / PUSH_SPEED_M_S))
            pz, _ = tcp_now(); z2 = float(pz[2])
            dfm = float(_WFH.get('dfmax', 0.0)) if IN_PROCESS else 0.0
            spring = wedge_search and outcome(z2) == 'in' and not r3.get('contact') and dfm >= PUSH_SPRING_N
            meta['probes'].append({'dx_mm': round(1000 * dx, 2), 'dy_mm': round(1000 * dy, 2), 'dyaw_deg': 0.0, 'push': True,
                                   'z_above_seat_mm': round(1000 * (z2 - seat[2]), 2), 'contact': bool(r3.get('contact')),
                                   'error': r3.get('error'), 'dF_max_N': round(dfm, 2), 'spring': bool(spring)})
            print(f"[touch] entered at ({1000*dx:+.1f}, {1000*dy:+.1f}) mm: pushed to {1000*(z2 - seat[2]):+.1f} mm above the seat"
                  f" (peak {dfm:.1f} N)", flush=True)
            if spring:
                # at the floor but it carried the load of an end on the rim: the bar pivoted in the jaws (looks seated, one end
                # +8 mm high - runs 58/59/61) - the next point along the bar, as for a push held up short (straight up first)
                print(f"[touch] the push carried {dfm:.1f} N without bottoming (>= {PUSH_SPRING_N:.0f} N): an end bearing, "
                      f"the bar pivoting in the jaws - the next point along the bar", flush=True)
                pushes.append((z2, dx, dy))                    # the fallback if no point along the bar seats clean
                move([seat[0] + dx, seat[1] + dy, start_z], q_seat, max_force=None)
                continue
            if outcome(z2) == 'in' or not wedge_check:
                return finish(stem, meta, 'success' if outcome(z2) == 'in' else 'entering')
            # held up short of the seat: an end on the base top - the next point along the bar (straight up first)
            pushes.append((z2, dx, dy))
            move([seat[0] + dx, seat[1] + dy, start_z], q_seat, max_force=None)
            continue
        if z is None:
            return finish(stem, meta, 'failure')
        if best is None or z < best[0]:
            best = (z, dx, dy)
            sz_ = seat_z()
            meta['seat_z_measured'] = sz_ if sz_ != float(seat[2]) else None
            meta['rim_measured_above_cad_mm'] = round(1000 * (best[0] - rim_z), 2)
        # lateral steps 1.5 mm over the lowest contact MEASURED - never below a contact (14:37: the first probe stopped on
        # line_green's top at +50.3, the retract went to the planned +34.5 into it, protective stop), guarded
        start_z = best[0] + 0.0015
        cur, _ = tcp_now() if IN_PROCESS else (None, None)
        d_ = abs(float(cur[2]) - start_z) if cur is not None else 1.0
        # straight UP off a contact needs no guard - and a guard cannot work there: its reference is taken while the
        # part still presses at ~4 N, lifting off unloads it and trips it at once (probe bench 10-22: the retract moved
        # 0.2 mm). A retract that is not upward stays guarded (14:37: one went below a contact on line_green's top).
        up = cur is not None and start_z > float(cur[2])
        move([seat[0] + dx, seat[1] + dy, start_z], q_seat, max_force=None if up else MAX_FORCE_N,
             duration=max(SHORT_MIN_S, d_ / SHORT_SPEED_M_S) if d_ <= SHORT_MOVE_M else None)   # retract before any step
    if pushes:
        # nothing seated along the bar: back to the lowest push and leave it there - the seat check's recovery (nudge)
        # takes it from the state the insert always left before
        z2, dx, dy = min(pushes)
        q = R.from_quat(q_seat).as_quat()
        move([seat[0] + dx, seat[1] + dy, start_z], q_seat)
        move([seat[0] + dx, seat[1] + dy, seat[2] - 0.002], q, max_force=6.0, min_duration=8.0)
        print(f"[touch] no point along the bar seated - left at the lowest ({1000*dx:+.1f}, {1000*dy:+.1f}) mm, "
              f"{1000*(z2 - seat[2]):+.1f} mm above the seat", flush=True)
        return finish(stem, meta, 'entering')
    bx, by = (best[1], best[2]) if best else (0.0, 0.0)
    for dyaw in YAWS_DEG:
        z = probe(bx, by, dyaw)
        if outcome(z) == 'in':
            return finish(stem, meta, 'success')
        move([seat[0] + bx, seat[1] + by, start_z], q_seat)
    return finish(stem, meta, 'not_found')


def finish(stem, meta, result):
    meta['outcome'] = result
    # the final TCP as the trace's one row: record_seat_offset (fmb1_loop.insert_summary) reads where the part ended from
    # the CSV - the empty one meant no U-part seat was ever recorded on the touch route, and line_green, which threads
    # between them (+-1.1 mm), aimed at the bare base (10-22 15:48: it landed rolled 6.7 deg on a U edge, +6.5 mm high)
    try:
        p_, _ = tcp_now()
        with open(str(stem) + '.csv', 'a') as fh:
            fh.write(f"{time.time():.3f},{p_[0]:.6f},{p_[1]:.6f},{p_[2]:.6f}\n")
    except Exception:
        pass
    if meta.get('seat_z_measured') is not None:              # the seat corrected by the measured rim (grasp depth)
        meta['predicted_seat_xyz_chain'] = list(meta['predicted_seat_xyz'])
        meta['predicted_seat_xyz'] = meta['predicted_seat_xyz'][:2] + [meta['seat_z_measured']]
    json.dump(meta, open(str(stem) + '.meta.json', 'w'), indent=1)
    print('__RESULT_JSON__'); print(json.dumps({'result': 'success' if result == 'success' else 'failure',
                                                'touch_outcome': result, 'n_probes': len(meta['probes']),
                                                'meta': str(stem) + '.meta.json'}))
    print('__END_RESULT_JSON__', flush=True)
    return 0 if result == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
