#!/usr/bin/env python3
"""Move the (empty) base on the table with the arm: slide it and turn it to a target pose (2026-10-20).

Operator: "you already have the primitives ... tapping or the primitives you used to tip can be used to
slide the base, even correct angles ... you can even spin the base". For M2's cycles (base moved ~3 cm,
then turned ~5 deg) without a person at the table. Run it between a disassembly and the reassembly: the
base is empty, nothing seated can tip.

    python3 -u ablations/move_base.py --base base3 --dy 30                # slide 30 mm along world +y
    python3 -u ablations/move_base.py --base base3 --dyaw 5               # turn +5 deg about its centre
    python3 -u ablations/move_base.py --base base3 --to X_MM Y_MM YAW_DEG # to an absolute pose
    [--dry-run] [--tol-mm 4] [--tol-deg 1] [--max-pushes 12]

Closed loop: read the base (6-dof CAD fit to depth, empty gripper, standard view) -> the remaining error ->
one push -> read again. A push: closed fingers, wrist turned so the 20.2 mm side of the finger block faces
the push, lowered from safe height beside the face (guarded: stops on touch, then starts further out),
fingertips TIP_ABOVE_TABLE_M over the table (below the base top, CAD), then a straight push guarded at
MAX_FORCE_N, at most STEP_MAX_MM past the face; back off; lift.
  slide: the middle of the face opposite the move, along the move (the larger error axis of the base frame)
  turn:  the same face but LEVER_MM off its middle: an off-centre push turns the base as it slides; the
         lever's side sets the sense. Turn first (a turn also slides it), then slide.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))

TIP_ABOVE_TABLE_M = 0.008       # fingertips over the table during the push (base is 35 mm tall)
GAP_M = 0.012                   # finger face this far off the base face before the push
STEP_MAX_MM = 50.0              # one push moves the finger face at most this far past the base face
PUSH_SPEED_M_S = 0.008          # slow enough that the base stops when the fingers stop
MAX_FORCE_N = 12.0              # guard for the push (the empty base slides at a few N)
LEVER_MM = 55.0                 # push point off the face middle for a turn
BLOCK_HALF_PUSH_M = 0.0101      # closed finger block: half of the side that faces the push
VIEW_Z = 0.56


def read_base(base):
    from replay_real_assembly import observe_base_pose
    from base_icp import read_icp
    from base_depth import measure
    ob = observe_base_pose(base, timeout_s=8.0)
    if ob is None:
        return None, 'not seen'
    oi, info = read_icp(base, ob)
    if oi is None:
        od, _ = measure(ob, base)
        oi = od or ob
        info['fallback'] = 'outline' if od else 'marker'
    return oi, info


def go(argv, label, timeout=90):
    r = subprocess.run(['python3', '-u', '-m', *argv], cwd=str(REPO), capture_output=True, text=True, timeout=timeout)
    out = r.stdout
    res = {}
    if '__RESULT_JSON__' in out:
        try:
            res = json.loads(out.split('__RESULT_JSON__')[-1].split('__END_RESULT_JSON__')[0].strip())
        except Exception:
            pass
    res.setdefault('result', 'success' if r.returncode == 0 else 'failure')
    print(f"    {label}: {res.get('result')}{' - ' + str(res.get('error')) if res.get('error') else ''}", flush=True)
    return res


def push(base, ob, face_dir_b, along_mm, step_mm, dry):
    """One guarded push on the face whose outward normal is -face_dir_b (base frame unit x/y), moving the base
    along +face_dir_b. along_mm: push point offset along the face from its middle (lever for a turn)."""
    from spacing_check import BASE_SIDE_M
    from primitives.shared.config import RG2_TIP_CLOSED_M, SAFE_HEIGHT, TABLE_HEIGHT
    (bx, by, bz), bq = ob
    Rb = R.from_quat(bq)
    h = BASE_SIDE_M[base] / 2
    n = np.array([face_dir_b[0], face_dir_b[1], 0.0])                  # push direction, base frame
    t = np.array([-n[1], n[0], 0.0])                                    # along the face
    face_c = -n * h + t * (along_mm / 1000.0)                           # point on the face (base frame)
    table_z = bz - 0.0175 if abs(bz - TABLE_HEIGHT) > 0.03 else TABLE_HEIGHT
    tip_z = table_z + TIP_ABOVE_TABLE_M
    fz = tip_z + RG2_TIP_CLOSED_M                                       # flange z: tips at tip_z, top-down
    start_b = face_c - n * (GAP_M + BLOCK_HALF_PUSH_M)
    end_b = face_c + n * (step_mm / 1000.0 - BLOCK_HALF_PUSH_M)
    W = lambda p: np.array([bx, by]) + Rb.apply(p)[:2]
    s_w, e_w = W(start_b), W(end_b)
    push_w = Rb.apply(n)[:2]
    yaw = float(np.arctan2(push_w[1], push_w[0]))
    # top-down (0,1,0,0) closes along world x -> its 20.2 mm side faces y; make that side face the push
    q = (R.from_euler('z', yaw - np.pi / 2) * R.from_quat([0, 1, 0, 0])).as_quat()
    Q = [f"{v:.6f}" for v in q]
    X = lambda p, z: [f"{p[0]:.4f}", f"{p[1]:.4f}", f"{z:.4f}"]
    print(f"  push: face {np.round(face_dir_b, 2).tolist()} lever {along_mm:+.0f} mm, step {step_mm:.0f} mm, "
          f"start {np.round(1000 * s_w, 1).tolist()} -> {np.round(1000 * e_w, 1).tolist()} mm, tips {1000*(tip_z-table_z):.0f} mm over the table", flush=True)
    if dry:
        return True
    for argv, label in [(['primitives.control_gripper', 'close', '--mode', 'real'], 'close'),
                        (['primitives.move_to_safe_height', '--mode', 'real'], 'safe height'),
                        (['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(s_w, SAFE_HEIGHT + 0.05), '--quat', *Q], 'over the start')]:
        if go(argv, label).get('result') == 'failure':
            return False
    for extra in (0.0, 0.010, 0.020):
        s2 = s_w - push_w * extra
        r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(s2, fz), '--quat', *Q, '--min-z', '0.13',
                '--max-force', '5', '--min-duration', '5'], f'lower beside the face (+{1000*extra:.0f} mm out)')
        if not r.get('contact'):
            break
        go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(s2, SAFE_HEIGHT + 0.05), '--quat', *Q], 'lift')
    else:
        return False
    if r.get('result') == 'failure':
        return False
    time.sleep(1.5)                                                     # settle before the force baseline
    # slow (PUSH_SPEED_M_S): a 3 s push of 5 mm moved the base 10 mm - it slid on after the fingers stopped
    dur = max(3.0, float(np.linalg.norm(e_w - s2)) / PUSH_SPEED_M_S)
    r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(e_w, fz), '--quat', *Q, '--min-z', '0.13',
            '--max-force', f"{MAX_FORCE_N}", '--min-duration', f"{dur:.1f}"], 'push')
    go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(s2, fz + 0.01), '--quat', *Q, '--min-z', '0.13'], 'back off')
    go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(s2, SAFE_HEIGHT + 0.05), '--quat', *Q], 'lift')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True)
    ap.add_argument('--dx', type=float, default=0.0, help='mm, world')
    ap.add_argument('--dy', type=float, default=0.0, help='mm, world')
    ap.add_argument('--dyaw', type=float, default=0.0, help='deg')
    ap.add_argument('--to', type=float, nargs=3, default=None, metavar=('X_MM', 'Y_MM', 'YAW_DEG'))
    ap.add_argument('--tol-mm', type=float, default=5.0)
    ap.add_argument('--tol-deg', type=float, default=1.0)
    ap.add_argument('--max-pushes', type=int, default=4)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    from primitives.shared.config import STAGING_REGION

    def view(ob):
        if a.dry_run:
            return True
        (bx, by, _), _ = ob
        return go(['primitives.move_cartesian', '--mode', 'real', '--xyz', f"{bx:.4f}", f"{by:.4f}", f"{VIEW_Z:.2f}",
                   '--quat', '0', '1', '0', '0'], 'view the base').get('result') != 'failure'

    ob, info = read_base(a.base)
    if ob is None:
        sys.exit(f"{a.base} not read: {info}")
    view(ob); time.sleep(1.5); ob, info = read_base(a.base)
    x0, y0 = 1000 * ob[0][0], 1000 * ob[0][1]; yaw0 = float(R.from_quat(ob[1]).as_euler('xyz', degrees=True)[2])
    tgt = a.to if a.to else (x0 + a.dx, y0 + a.dy, yaw0 + a.dyaw)
    print(f"[move_base] {a.base} at ({x0:.1f}, {y0:.1f}) mm {yaw0:.2f} deg -> target ({tgt[0]:.1f}, {tgt[1]:.1f}) mm {tgt[2]:.2f} deg  [{json.dumps(info)}]", flush=True)
    sx, sy = STAGING_REGION['x'], STAGING_REGION['y']
    if not (1000 * sx[0] + 80 < tgt[0] < 1000 * sx[1] - 80 and 1000 * sy[0] + 80 < tgt[1] < 1000 * sy[1] - 80):
        print("  !!! target outside the staging/assembly area - not moving"); return 1
    first_dir = None
    for k in range(1, a.max_pushes + 1):
        (x, y, _), q = ob
        yaw = float(R.from_quat(q).as_euler('xyz', degrees=True)[2])
        ex, ey = tgt[0] - 1000 * x, tgt[1] - 1000 * y
        eyaw = (tgt[2] - yaw + 180) % 360 - 180
        print(f"[move_base] {k}: error ({ex:+.1f}, {ey:+.1f}) mm, {eyaw:+.2f} deg", flush=True)
        if np.hypot(ex, ey) <= a.tol_mm and abs(eyaw) <= a.tol_deg:
            print(f"[move_base] DONE in {k-1} pushes: ({1000*x:.1f}, {1000*y:.1f}) mm {yaw:.2f} deg"); return 0
        Rb = R.from_euler('z', yaw, degrees=True)
        e_b = Rb.inv().apply([ex, ey, 0.0])[:2]                            # error in the base frame
        if abs(eyaw) > a.tol_deg:
            # turn: push the face across the larger slide error (so the slide comes along), off-centre
            ax = int(np.argmax(np.abs(e_b))) if np.hypot(*e_b) > a.tol_mm else 1
            d = np.zeros(2); d[ax] = 1.0 if (e_b[ax] >= 0 or np.hypot(*e_b) <= a.tol_mm) else -1.0
            # a push along +d at a point offset +t from the face middle (t = d rotated +90) turns the base -z
            lever = -np.sign(eyaw) * LEVER_MM
            step = float(np.clip(abs(eyaw) * 1.5 + 3, 4, STEP_MAX_MM))
        else:
            ax = int(np.argmax(np.abs(e_b)))
            d = np.zeros(2); d[ax] = float(np.sign(e_b[ax]))
            lever = 0.0
            step = float(np.clip(abs(e_b[ax]), 3, STEP_MAX_MM))
            # operator 2026-10-20: "for a slide in one direction why are we iterating so much when you know how
            # much to slide it by" - one push of the whole distance; a top-up only from the SAME side; an
            # overshoot is accepted (pushing back from the other side oscillated +-5 mm for 10 pushes)
            if first_dir is None:
                first_dir = d.copy()
            elif np.dot(d, first_dir) < 0:
                print(f"[move_base] overshot by {np.hypot(*e_b):.1f} mm - accepted (no push back)"); return 0
        if not push(a.base, ob, d, lever, step, a.dry_run):
            print("  !!! push failed - stopping"); return 1
        if a.dry_run:
            return 0
        if not view(ob):
            return 1
        time.sleep(1.5)
        nb, info = read_base(a.base)
        if nb is None:
            print(f"  !!! base not read after the push: {info}"); return 1
        print(f"    read: {json.dumps(info)}", flush=True)
        ob = nb
    print("[move_base] not at the target after the pushes allowed"); return 2


if __name__ == '__main__':
    sys.exit(main())
