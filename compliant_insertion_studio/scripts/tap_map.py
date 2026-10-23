#!/usr/bin/env python3
"""Tap-map a part where it sits in the base (seams.tap_probe): a fresh base read, then <= 3 N measuring taps at its ends,
middle and edges against the seated CAD top. Measures only - presses nothing.

    python3 -u compliant_insertion_studio/scripts/tap_map.py line_green [--base base1]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'ablations')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part'); ap.add_argument('--base', default='base1')
    a = ap.parse_args()
    from disassemble import observe_base
    ob = observe_base(a.base)
    if ob is None:
        print('base not read'); return 1
    print(f"base {[round(1000 * v, 1) for v in ob[0]]} mm", flush=True)
    subprocess.run([sys.executable, '-u', '-m', 'primitives.control_gripper', 'close', '--mode', 'real'], cwd=str(REPO),
                   capture_output=True, timeout=30)
    from seams.tap_probe import Tapper
    tp = Tapper.for_part(a.part, ob, a.base)
    m = tp.map()
    import compliant_insertion_studio.wrapper.touch_insert as T
    c = T.tcp_now()[0]
    T.move([c[0], c[1], 0.32], [0.0, 1.0, 0.0, 0.0])
    print(f"MAP {a.part}: {json.dumps(m)} mm vs the seated CAD top (band +-{1000 * tp.band_m:.1f}); seated={tp.seated()}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
