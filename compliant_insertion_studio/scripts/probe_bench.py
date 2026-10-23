#!/usr/bin/env python3
"""Bench the touch insert's probe moves with NOTHING gripped (2026-10-22, motion-safety rule: new motion code is tested
empty first): guarded dips of the closed fingers onto the bare table at the clear area, the same move() / tcp_now() the
touch insert uses - in one process (TOUCH_INPROC=1, the new path) or one process per move (TOUCH_INPROC=0, the old one).

    TOUCH_INPROC=1 python3 -u compliant_insertion_studio/scripts/probe_bench.py --n 9
    TOUCH_INPROC=0 python3 -u compliant_insertion_studio/scripts/probe_bench.py --n 3

Each dip: short guarded step to the next lattice point 1.5 mm over the last contact, guarded descent (4 N) toward 9 mm
under the first contact (it stops on the table), retract. Prints per-dip seconds and contact heights; refuses when the
gripper reports a grasp.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'ablations')]


def gripper_holding():
    p = subprocess.run(['ros2', 'topic', 'echo', '--once', '/gripper_status'], capture_output=True, text=True, timeout=15)
    return 'Grasp:True' in p.stdout, p.stdout.strip().splitlines()[:1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xy', type=float, nargs=2, default=[-0.320, -0.500], help='bare table spot (clear area)')
    ap.add_argument('--hover-z', type=float, default=0.170, help='flange z ~19 mm over a closed-tip table touch '
                    '(TABLE_HEIGHT + RG2_TIP_CLOSED_M = 0.151)')
    ap.add_argument('--n', type=int, default=9)
    a = ap.parse_args()
    held, st = gripper_holding()
    if held:
        print(f"refused: the gripper reports a grasp ({st}) - this bench runs with nothing gripped"); return 1
    import compliant_insertion_studio.wrapper.touch_insert as T
    from compliant_insertion_studio.wrapper.touch_insert import hex_grid
    print(f"in-process: {T.IN_PROCESS}", flush=True)
    q = [0.0, 1.0, 0.0, 0.0]
    x0, y0 = a.xy
    subprocess.run([sys.executable, '-u', '-m', 'primitives.control_gripper', 'close', '--mode', 'real'], cwd=str(REPO),
                   capture_output=True, timeout=30)
    r = T.move([x0, y0, 0.30], q)                       # over the spot at safe height first, then straight down
    if r.get('result') != 'failure':
        r = T.move([x0, y0, a.hover_z], q)
    if r.get('result') == 'failure':
        print('hover not reached', r); return 1
    rows, start_z, first = [], a.hover_z, None
    for k, (dx, dy) in enumerate(hex_grid(0.0006, 0.0025)[:a.n]):
        t0 = time.time()
        cur, _ = T.tcp_now()
        d_ = float(np.linalg.norm(np.array([x0 + dx, y0 + dy, start_z]) - cur))
        r1 = T.move([x0 + dx, y0 + dy, start_z], q, max_force=T.MAX_FORCE_N,
                    duration=max(T.SHORT_MIN_S, d_ / T.SHORT_SPEED_M_S) if (T.IN_PROCESS and d_ <= T.SHORT_MOVE_M) else None)
        t1 = time.time()
        z_end = (first - 0.009) if first is not None else 0.150
        dur = (start_z - z_end) / T.SPEED_M_S
        r2 = T.move([x0 + dx, y0 + dy, z_end], q, max_force=T.MAX_FORCE_N, min_duration=dur,
                    duration=dur if T.IN_PROCESS else None)
        t2 = time.time()
        pz, _ = T.tcp_now(); z = float(pz[2])
        if not r2.get('contact'):
            print(f"dip {k}: NO contact (ended at {z:.4f}) - stopping", r2); break
        first = z if first is None else first
        start_z = z + 0.0015
        cur, _ = T.tcp_now()
        r3 = T.move([x0 + dx, y0 + dy, start_z], q, max_force=T.MAX_FORCE_N,
                    duration=max(T.SHORT_MIN_S, abs(float(cur[2]) - start_z) / T.SHORT_SPEED_M_S) if T.IN_PROCESS else None)
        t3 = time.time()
        rows.append(dict(k=k, step_s=round(t1 - t0, 2), descend_s=round(t2 - t1, 2), retract_s=round(t3 - t2, 2),
                         total_s=round(t3 - t0, 2), contact_z=round(z, 5), r1=r1.get('result'), r3=r3.get('result'),
                         F=r2.get('F_now')))
        print(json.dumps(rows[-1]), flush=True)
    T.move([x0, y0, a.hover_z + 0.03], q)
    if rows:
        tot = [r_['total_s'] for r_ in rows]
        print(f"\n{len(rows)} dips, median {np.median(tot):.2f} s/dip, contact z spread "
              f"{1000 * np.ptp([r_['contact_z'] for r_ in rows]):.2f} mm (in-process {T.IN_PROCESS})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
