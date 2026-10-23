#!/usr/bin/env python3
"""Short sideways nudge of a seated part, from CAD + the measured base (2026-10-20, FMB3 line_brown).

    python3 -u ablations/nudge_part.py --object line_brown --base base3 --toward +x --at-long -58 [--push-mm 2] [--dry-run]

A bar cocked in its channel (one end riding a channel lip) drops in when that end is pushed sideways
off the lip (operator, 2026-10-20: "these line parts will need a nudge from the other two directions
as well"). Everything comes from CAD in the base frame and the base read:

  * the part's side face (CAD top face centre/half-width, seat_check.top_face), the push height: pad
    centre at the part's top (fingertips 15 mm lower, above the base top - never on it)
  * start: closed finger block `--gap-mm` clear of the face, lowered from safe height, guarded (stops on
    touch; then lifts and starts further out)
  * taps (default): out to the CAD face (`--push-mm` past it, default 0) at the arm's short-move speed and
    straight back, --max-force as a stop. Never a sustained push: the base is not fixed, and a slow push
    dragged base3 4 mm / 1.7 deg (operator, 2026-10-20: "a little tap would have prevented the drag")
  * back off, safe height, re-read the base, depth seat check

--toward: the direction the part is pushed, in the BASE frame (+x / -x / +y / -y). --at-long: where along
the part's long axis (mm, from the top-face centre) the fingers push - at the high end.
The closed finger block is 20.2 mm (across the fingers) x 22.8 mm (closing axis); the wrist is turned so
the 20.2 mm side faces the push.
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

BLOCK_HALF_PUSH_M = 0.0101          # closed finger block, half of its 20.2 mm side (faces the push)
BLOCK_HALF_SIDE_M = 0.0114          # half of the closing-axis extent (22.8 mm closed)
MAX_FORCE_N = 4.0                   # stop on this sustained force change (operator's limit 5 N)
PUSH_MM_MAX = 4.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--object', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--toward', required=True, choices=['+x', '-x', '+y', '-y'])
    ap.add_argument('--at-long', type=float, default=0.0, help='mm along the long axis from the top-face centre')
    ap.add_argument('--push-mm', type=float, default=0.0,
                    help='how far past the CAD face the block goes (0 = to the face: a part with no clearance in its '
                         'seat pushes the BASE with anything more - 1 mm turned base3 1.7 deg, 2026-10-20)')
    ap.add_argument('--gap-mm', type=float, default=3.0, help='start clearance from the measured face')
    ap.add_argument('--face-offset-mm', type=float, default=0.0,
                    help='measured shift of the face toward the block (+ = the part sits that much closer)')
    ap.add_argument('--max-force', type=float, default=MAX_FORCE_N)
    ap.add_argument('--taps', type=int, default=2, help='quick taps (out and straight back) - the default')
    ap.add_argument('--slow-push', action='store_true',
                    help='one slow push instead of taps (dragged base3 4 mm / 1.7 deg, 2026-10-20 - avoid)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if not -2.0 <= a.push_mm <= PUSH_MM_MAX:
        sys.exit(f"--push-mm must be in [-2, {PUSH_MM_MAX}]")
    from seat_check import top_face, check
    from replay_real_assembly import observe_base_pose, run_subprocess
    from primitives.shared.config import RG2_TIP_CLOSED_M, RG2_PAD_HALF_M, SAFE_HEIGHT
    f = top_face(a.object, a.base)
    ob = observe_base_pose(a.base)
    if ob is None:
        sys.exit(f"{a.base} not observed")
    (bx, by, bz), bq = ob
    Rb = R.from_quat(bq)
    axis = 0 if a.toward[1] == 'x' else 1                    # push axis (base frame)
    sgn = 1.0 if a.toward[0] == '+' else -1.0
    long_ax = 1 if f['long_axis'] == 'y' else 0
    if axis == long_ax:
        half = f['half_long_m']; along = 0.0                   # pushing an end along the part
    else:
        half = f['half_cross_m']; along = a.at_long / 1000.0
    c = np.array([f['cx'], f['cy']])
    face = c[axis] - sgn * half                                      # the CAD face the taps stop at (base frame)
    meas = face - sgn * a.face_offset_mm / 1000.0                     # where that face is now (offset toward the block)
    pad_z = f['z']                                          # pad centre at the part's top: tips 14.9 mm below
    def base_pt(push_coord):
        p = np.zeros(3); p[axis] = push_coord; p[1 - axis] = c[1 - axis] + (along if axis != long_ax else 0.0)
        p[2] = pad_z
        return p
    start_b = base_pt(meas - sgn * (a.gap_mm / 1000.0 + BLOCK_HALF_PUSH_M))
    push_b = base_pt(face + sgn * (a.push_mm / 1000.0 - BLOCK_HALF_PUSH_M))
    W = lambda p: np.array([bx, by, bz]) + Rb.apply(p)
    fz = lambda p: p[2] + RG2_TIP_CLOSED_M - RG2_PAD_HALF_M
    # wrist: top-down (0,1,0,0) closes along world x, so its 20.2 mm side faces y; push along x -> turn 90 deg
    yaw_base = float(Rb.as_euler('xyz')[2])
    q = (R.from_euler('z', yaw_base + (np.pi / 2 if axis == 0 else 0.0)) * R.from_quat([0, 1, 0, 0])).as_quat()
    tips_z_base = pad_z - RG2_PAD_HALF_M
    plan = dict(object=a.object, base=a.base, toward=a.toward, face_base_mm=round(1000 * face, 1),
                start_base_mm=np.round(1000 * start_b, 1).tolist(), push_base_mm=np.round(1000 * push_b, 1).tolist(),
                tips_z_base_mm=round(1000 * tips_z_base, 1), part_top_mm=round(1000 * f['z'], 1),
                start_world=np.round(W(start_b), 4).tolist(), push_world=np.round(W(push_b), 4).tolist())
    print(f"[nudge] plan {json.dumps(plan)}", flush=True)
    if a.dry_run:
        return 0
    Q = [f"{v:.6f}" for v in q]
    X = lambda p, z: [f"{p[0]:.4f}", f"{p[1]:.4f}", f"{z:.4f}"]
    go = lambda argv, label, t=60: run_subprocess(['python3', '-u', '-m', *argv], label=label, timeout_s=t)
    for argv, label in [(['primitives.control_gripper', 'close', '--mode', 'real'], 'close'),
                        (['primitives.move_to_safe_height', '--mode', 'real'], 'safe height')]:
        if go(argv, label).get('result') == 'failure':
            print(f"!!! {label} failed"); return 1
    for extra in (0.0, 0.005, 0.010):
        s = W(start_b - np.eye(3)[axis] * sgn * extra)
        r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(s, SAFE_HEIGHT + 0.08), '--quat', *Q],
               'over the start')
        if r.get('result') == 'failure':
            print(f"!!! over the start failed: {r.get('error')}"); return 1
        r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(s, fz(s)), '--quat', *Q,
                '--min-z', '0.15', '--max-force', f"{a.max_force}", '--min-duration', '5'], f'lower ({1000*extra:.0f} mm further out)')
        if not r.get('contact'):
            break
        print(f"[nudge] touched lowering ({r.get('error')}) -> lift, further out", flush=True)
        go(['primitives.move_to_safe_height', '--mode', 'real'], 'lift')
    else:
        print("!!! no clear spot to lower the fingers"); return 1
    if r.get('result') == 'failure':
        print(f"!!! lower failed: {r.get('error')}"); return 1
    time.sleep(1.5)                                          # settle before the force baseline
    p = W(push_b)
    if a.slow_push:
        r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(p, fz(p)), '--quat', *Q,
                '--min-z', '0.15', '--max-force', f"{a.max_force}", '--min-duration', '4'], f'push {a.push_mm:g} mm past the face')
        print(f"[nudge] push: {'stopped: ' + str(r.get('error')) if r.get('contact') else 'reached the distance cap'}", flush=True)
    else:
        # Taps (operator 2026-10-20: "a little tap would have prevented the drag on the base"): out to the
        # CAD face at the arm's own short-move speed and straight back - contact lasts a fraction of a
        # second, so the part is knocked off its lip before friction carries the base along.
        for k in range(1, a.taps + 1):
            r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(p, fz(p)), '--quat', *Q,
                    '--min-z', '0.15', '--max-force', f"{a.max_force}"], f'tap {k}')
            print(f"[nudge] tap {k}: {'stopped: ' + str(r.get('error')) if r.get('contact') else 'reached the face'}"
                  f" ({r.get('duration_s', r.get('t_after_start_s'))} s)", flush=True)
            b = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(s, fz(s)), '--quat', *Q,
                    '--min-z', '0.15'], f'tap {k} back')
            if b.get('result') == 'failure':
                print(f"!!! tap {k} back failed: {b.get('error')}"); return 1
            time.sleep(1.5)                                  # settle before the next force baseline
    for argv, label in [(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(s, fz(s) + 0.02), '--quat', *Q,
                          '--min-z', '0.15'], 'back off'),
                        (['primitives.move_to_safe_height', '--mode', 'real'], 'safe height')]:
        if go(argv, label).get('result') == 'failure':
            print(f"!!! {label} failed"); return 1
    ob2 = observe_base_pose(a.base)
    if ob2 is not None:
        d = 1000 * np.linalg.norm(np.array(ob2[0][:2]) - np.array(ob[0][:2]))
        dy = np.degrees((R.from_quat(ob2[1]) * Rb.inv()).magnitude())
        print(f"[nudge] base moved {d:.1f} mm / {dy:.2f} deg during the nudge", flush=True)
        ob = ob2
    res = check(a.object, ob, move=True, out_dir=REPO / 'ablations/logs/seat_checks', base=a.base)
    print(f"[seat-check] {a.object} after the nudge: "
          f"{json.dumps({k: v for k, v in res.items() if k not in ('profile_mm', 'observed_base')})}", flush=True)
    return 0 if res.get('ok') else 2


if __name__ == '__main__':
    sys.exit(main())
