#!/usr/bin/env python3
"""Tip an upright part flat with closed fingers (2026-10-19).

    python3 -u ablations/tip_part.py PART [--layout ablations/logs/staging_layout.json] [--table-z Z]

Operator's method: like the line_green nudge, the closed fingertips strike the TOP of the standing
part at an angle. From safe height the fingers go straight down to a start point beside the part
(25 mm before its near face, 10 mm above its top) — a guarded lowering: touching anything there
means the plan is wrong, so it lifts and stops. Then ONE guarded diagonal stroke through the top
edge, down and across, to 30 mm past the far face and 45 mm above the table (above the 21 mm the
part lies at once tipped, so the fingers never touch it on the table). The stroke stops at
TIP_FORCE_N (operator: <= 5 N); the parts are light and fall long before that — a stop means
something held it. Straight back up after.

Strike point along the part: U parts stand open-top (two 21 mm legs, 28 mm gap) -> a leg; yellow
-> its solid top centre (staging_layout.STAND). The wrist turns so the closed finger block's
narrow side (20 mm, as in the nudge) leads the stroke.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))

TIP_FORCE_N = 5.0
START_BEFORE_M = 0.025         # fingertip block centre before the near face
START_BELOW_TOP_M = 0.012      # ... and this far below the part's top (beside it, in clear space)
END_PAST_M = 0.030             # stroke end past the far face
END_TIP_Z_M = 0.045            # stroke end: fingertips this high above the table
# The stroke meets the near face ~16 mm below the top (COM is at 35 mm): starting 10 mm ABOVE the
# top only clipped yellow's top edge — the contact pushed down, 5.9 N, no tip (2026-10-19).


def go(argv, label, t=90):
    env = dict(os.environ, PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}")
    p = subprocess.run([sys.executable, '-u', '-m', *argv], capture_output=True, text=True, timeout=t,
                       env=env, cwd=str(REPO))
    js = [l for l in (p.stdout + p.stderr).splitlines() if l.startswith('{"result')]
    res = json.loads(js[-1]) if js else {'result': 'failure', 'error': f'rc={p.returncode}, no result'}
    print(f">>> {label}: {res.get('result')} {res.get('error', '')}"[:200], flush=True)
    return res


def stroke_quat(d):
    """Top-down, turned so a side of the closed finger block leads along d — tool y or tool x,
    whichever needs the smaller wrist turn (<= 45 deg): a 90 deg turn plus the move over the start
    went over move_cartesian's 1.5 rad joint-step limit (2026-10-19)."""
    cands = [math.atan2(-d[0], d[1]), math.atan2(-d[1], -d[0])]          # tool y || d, tool x || d
    cands += [c + math.pi for c in cands]
    th = min(((c + math.pi) % (2 * math.pi) - math.pi for c in cands), key=abs)
    return (R.from_euler('z', th) * R.from_quat([0.0, 1.0, 0.0, 0.0])).as_quat()


def tip(p, table_z, say=print):
    """p: one part of a staging_layout plan (place_xy, yaw, tip_dir, strike_x). Returns
    {'ok': bool, 'stopped': bool, 'error': str|None}; the camera check of the result is the caller's."""
    from primitives.shared.config import RG2_TIP_CLOSED_M, SAFE_HEIGHT
    from staging_layout import STAND
    g = STAND[p['name']]
    c, yaw, d = np.array(p['place_xy']), p['yaw'], np.array(p['tip_dir'])
    u = np.array([math.cos(yaw), math.sin(yaw)])
    s = c + u * p.get('strike_x', 0.0)
    start = s - d * (g['T'] / 2 + START_BEFORE_M)
    end = s + d * (g['T'] / 2 + END_PAST_M)
    z_start = table_z + g['H'] - START_BELOW_TOP_M + RG2_TIP_CLOSED_M      # flange z
    z_end = table_z + END_TIP_Z_M + RG2_TIP_CLOSED_M
    q = [f"{v:.6f}" for v in stroke_quat(d)]
    X = lambda xy, z: [f"{xy[0]:.4f}", f"{xy[1]:.4f}", f"{z:.4f}"]
    say(f"  [tip] {p['name']}: strike ({1000*s[0]:.0f}, {1000*s[1]:.0f}) mm toward ({d[0]:+.2f}, {d[1]:+.2f}); "
        f"fingertips {1000*(z_start-RG2_TIP_CLOSED_M-table_z):.0f} -> {1000*(z_end-RG2_TIP_CLOSED_M-table_z):.0f} mm "
        f"above the table, stop at {TIP_FORCE_N:g} N")
    for argv, label in [(['primitives.control_gripper', 'close', '--mode', 'real'], 'tip close'),
                        (['primitives.move_to_safe_height', '--mode', 'real'], 'tip safe height'),
                        (['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(start, SAFE_HEIGHT + 0.03),
                          '--keep-orientation'], 'tip over the start'),
                        (['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(start, SAFE_HEIGHT + 0.03),
                          '--quat', *q], 'tip turn the wrist in place')]:
        if go(argv, label).get('result') == 'failure':
            return dict(ok=False, stopped=False, error=label)
    time.sleep(1.5)                                                       # settle before the force baseline
    r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(start, z_start), '--quat', *q,
            '--min-z', '0.15', '--max-force', f"{TIP_FORCE_N}", '--min-duration', '5'], 'tip lower to the start')
    if r.get('contact') or r.get('result') == 'failure':
        go(['primitives.move_to_safe_height', '--mode', 'real'], 'tip lift (start not clear)')
        return dict(ok=False, stopped=True, error=f"touched something lowering to the start: {r.get('error')}")
    time.sleep(1.5)
    r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(end, z_end), '--quat', *q,
            '--min-z', '0.15', '--max-force', f"{TIP_FORCE_N}", '--min-duration', '6'],
           f'tip stroke (until {TIP_FORCE_N:g} N)')
    stopped = bool(r.get('contact'))
    say(f"  [tip] {p['name']}: stroke {'STOPPED at ' + str(r.get('error')) if stopped else 'ran to its end'}")
    up = go(['primitives.move_to_safe_height', '--mode', 'real'], 'tip back up')
    if up.get('result') == 'failure':
        return dict(ok=False, stopped=stopped, error='tip back up')
    go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(end, SAFE_HEIGHT), '--quat', '0.0', '1.0', '0.0', '0.0'],
       'tip turn the wrist back')
    return dict(ok=True, stopped=stopped, error=None)


def is_flat(quat_xyzw, tol_deg=30.0):
    """Lying on a large face: the part's thickness axis (object z, the 21 mm CAD height) vertical."""
    return abs(R.from_quat(quat_xyzw).as_matrix()[2, 2]) >= math.cos(math.radians(tol_deg))


def plan_from_observation(name, say=print):
    """A standing part as the camera sees it now (call from the staging view): its centre and yaw
    from the localizer, tipped toward whichever side keeps the landing inside the staging region and
    farthest from the other parts (>= 25 mm). None if neither side fits or it is not standing."""
    from spacing_check import observe_parts, PARTS, outline
    from staging_layout import part_plan, inside_region, _sep_axis_gap, CLEARANCE_M, SWEEP_CLEARANCE_M
    seen = observe_parts(PARTS)
    if name not in seen:
        say(f"{name} not seen"); return None
    (x, y, _), q = seen[name]
    if is_flat(q):
        say(f"{name} already lies flat"); return None
    Rm = R.from_quat(q).as_matrix()
    u = Rm[:2, 0] / np.linalg.norm(Rm[:2, 0])
    others = [np.asarray(outline(n, seen[n]))[:, :2] for n in seen if n != name]
    best = None
    for sign in (1.0, -1.0):
        p = part_plan(name, True, (x, y), math.atan2(u[1], u[0]), sign)
        if not (inside_region(p['final']) and inside_region(p['sweep'])):
            continue
        gap = min([_sep_axis_gap(p['final'], o) for o in others] or [1.0])
        sweep_gap = min([_sep_axis_gap(p['sweep'], o) for o in others] or [1.0])
        if gap >= CLEARANCE_M and sweep_gap >= SWEEP_CLEARANCE_M and (best is None or gap > best[0]):
            best = (gap, p)
    if best is None:
        say(f"{name}: no tip side lands inside the staging region and clear of the others"); return None
    say(f"  [tip] {name} seen standing at ({1000*x:.0f}, {1000*y:.0f}) mm; tip toward "
        f"({best[1]['tip_dir'][0]:+.2f}, {best[1]['tip_dir'][1]:+.2f}), {1000*best[0]:.0f} mm clear")
    return best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('part')
    ap.add_argument('--layout', default=str(REPO / 'ablations/logs/staging_layout.json'))
    ap.add_argument('--table-z', type=float, default=None)
    ap.add_argument('--observe', action='store_true',
                    help='tip the part where the camera sees it now (arm over the staging region), '
                         'toward the free side; then check it lies flat')
    a = ap.parse_args()
    if a.observe:
        from staging_layout import look, STAGING_VIEW, TOP_DOWN_Q
        from spacing_check import observe_parts
        tz, _ = look()
        if tz is None:
            return 1
        p = plan_from_observation(a.part)
        if p is None:
            return 1
        r = tip(p, tz)
        print(json.dumps(r))
        if not r['ok']:
            return 1
        go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *[f"{v:.4f}" for v in STAGING_VIEW],
            '--quat', *map(str, TOP_DOWN_Q)], 'staging view')
        time.sleep(1.5)
        seen = observe_parts([a.part])
        flat = a.part in seen and is_flat(seen[a.part][1])
        print(f"[tip] {a.part}: {'FLAT' if flat else 'NOT flat' if a.part in seen else 'not seen'} after the tip")
        return 0 if flat else 2
    L = json.load(open(a.layout))
    p = next((q for q in L['parts'] if q['name'] == a.part), None)
    if p is None or not p.get('tip'):
        print(f"{a.part}: not a part to tip in {a.layout}"); return 1
    from primitives.shared.config import TABLE_HEIGHT
    tz = a.table_z if a.table_z is not None else L.get('table_z', TABLE_HEIGHT)
    r = tip(p, tz)
    print(json.dumps(r))
    return 0 if r['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
