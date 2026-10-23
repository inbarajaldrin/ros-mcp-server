#!/usr/bin/env python3
"""Bench the smart taps (seams.tap_probe) with NOTHING gripped (motion-safety rule): closed tips on the bare table at
the clear area as the "top face" (seated height = the table). A map should read ~0 mm at every point; a press on the
rigid table must come back 'blocked' (the force rises, the height does not move) - the verdict the recovery relies on.

    python3 -u compliant_insertion_studio/scripts/tap_bench.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'ablations')]


def main():
    sys.path.insert(0, str(REPO / 'compliant_insertion_studio' / 'scripts'))
    from probe_bench import gripper_holding
    held, st = gripper_holding()
    if held:
        print(f"refused: the gripper reports a grasp ({st})"); return 1
    from primitives.shared.config import TABLE_HEIGHT, RG2_TIP_CLOSED_M
    from seams.tap_probe import Tapper, Point
    import compliant_insertion_studio.wrapper.touch_insert as T
    subprocess.run([sys.executable, '-u', '-m', 'primitives.control_gripper', 'close', '--mode', 'real'], cwd=str(REPO),
                   capture_output=True, timeout=30)
    x0, y0 = -0.320, -0.500
    pts = [Point(n, np.array([x0 + dx, y0 + dy, TABLE_HEIGHT])) for n, dx, dy in
           (('mid', 0, 0), ('end-', 0, -0.05), ('end+', 0, 0.05), ('edge-', -0.01, 0), ('edge+', 0.01, 0))]
    tp = Tapper(pts, float(RG2_TIP_CLOSED_M), 0.0015, press_below_m=0.0008)   # the table touch is at flange 0.151
    r = T.move([x0, y0, 0.30], [0.0, 1.0, 0.0, 0.0])                # over the spot at safe height, then down
    if r.get('result') == 'failure':
        print('not reached', r); return 1
    t0 = time.time()
    m = tp.map()
    t1 = time.time()
    print(f"map {m} mm in {t1 - t0:.1f} s ({(t1 - t0) / len(pts):.2f} s/tap)", flush=True)
    v = tp.press(pts[0])
    print(f"press on the table: {v} {tp.log[-1]} ({time.time() - t1:.1f} s)", flush=True)
    T.move([x0, y0, 0.30], [0.0, 1.0, 0.0, 0.0])
    print(json.dumps(tp.log))
    ok = all(abs(e) < 1.5 for e in m.values()) and v == 'blocked'
    print('BENCH', 'OK' if ok else 'NOT OK')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
