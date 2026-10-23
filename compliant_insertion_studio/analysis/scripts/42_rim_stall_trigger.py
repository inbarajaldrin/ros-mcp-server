#!/usr/bin/env python3
"""Replay the proposed SEARCH "stuck on rim" trigger against raw insert CSVs.

Proposed trigger (staged/032-search-rim-unload): while on the rim, the TCP has moved
less than STALL_XY_M in xy over the last STALL_WINDOW_S. The at-robot trigger also
requires the spiral reference to be paused (lag-pause); that column is not in the CSV,
so this replay is STRICTER than the real trigger — if it stays silent on good runs, the
real one does too.

On rim  := tcp_z > predicted_seat_z + ABOVE_SEAT_M for the whole window and the
           window's MEDIAN |fz| > CONTACT_N (pressed on something, not yet in the slot).
           A median, not every sample: the raw wrench carries impulses and dips, and a
           per-sample test broke on tonight's 146 s freeze. Starts GRACE_S after first
           contact (force-mode startup transient).

Ground truth is raw: seated iff the lowest ACTIVE tcp_z came within 5 mm of the
predicted seat (CLAUDE.md rule), not the FSM's outcome label.

Usage: 42_rim_stall_trigger.py DIR [DIR...]     (reads insert_*.csv + .meta.json)
"""
import csv
import glob
import json
import math
import os
import sys

STALL_WINDOW_S = 2.0
STALL_XY_M = 0.0003
CONTACT_N = 3.0
ABOVE_SEAT_M = 0.010
GRACE_S = 1.0


def analyse(base):
    meta = json.load(open(base + '.meta.json'))
    seat_z = meta['cad_prediction']['predicted_tcp_at_seat']['xyz_m'][2]
    rows = [r for r in csv.DictReader(open(base + '.csv')) if r.get('phase') == 'ACTIVE']
    if not rows:
        return None
    t = [float(r['t_s']) for r in rows]
    x = [float(r['tcp_x']) for r in rows]
    y = [float(r['tcp_y']) for r in rows]
    z = [float(r['tcp_z']) for r in rows]
    fz = [float(r['fz']) for r in rows]
    seated = min(z) - seat_z < 0.005

    on_rim = [abs(f) > CONTACT_N and zz > seat_z + ABOVE_SEAT_M for f, zz in zip(fz, z)]
    first = next((i for i, o in enumerate(on_rim) if o), None)
    if first is None:
        return dict(seated=seated, rim_s=0.0, trigger_s=None, frozen_s=0.0)
    t0 = t[first]
    rim_s = sum(t[i] - t[i - 1] for i in range(1, len(t)) if on_rim[i] and on_rim[i - 1])

    trigger_s, frozen_s, j = None, 0.0, first
    for i in range(first, len(t)):
        if t[i] - t0 < GRACE_S + STALL_WINDOW_S:
            continue
        while t[i] - t[j] > STALL_WINDOW_S:
            j += 1
        if not all(zz > seat_z + ABOVE_SEAT_M for zz in z[j:i + 1]):
            continue
        if sorted(abs(f) for f in fz[j:i + 1])[(i - j) // 2] <= CONTACT_N:
            continue
        ext = math.hypot(max(x[j:i + 1]) - min(x[j:i + 1]), max(y[j:i + 1]) - min(y[j:i + 1]))
        if ext < STALL_XY_M:
            if trigger_s is None:
                trigger_s = t[i] - t0
            frozen_s += t[i] - t[i - 1]
    return dict(seated=seated, rim_s=rim_s, trigger_s=trigger_s, frozen_s=frozen_s)


def main():
    bases = sorted({os.path.splitext(p)[0] for d in sys.argv[1:]
                    for p in glob.glob(os.path.join(d, 'insert_*.meta.json'))})
    bases = [b[:-5] if b.endswith('.meta') else b for b in bases]
    print(f"{'run':40s} {'seated':>6s} {'on-rim s':>8s} {'trigger @':>9s} {'frozen s':>8s}")
    fp = 0
    for b in bases:
        if not os.path.exists(b + '.csv'):
            continue
        r = analyse(b)
        if r is None:
            continue
        trig = f"{r['trigger_s']:.1f}s" if r['trigger_s'] is not None else '-'
        flag = ''
        if r['seated'] and r['trigger_s'] is not None:
            flag = '  <-- fired on a run that seated'
            fp += 1
        print(f"{os.path.basename(b):40s} {str(r['seated']):>6s} {r['rim_s']:8.1f} {trig:>9s} "
              f"{r['frozen_s']:8.1f}{flag}")
    print(f"\nfired on seated runs: {fp}")


if __name__ == '__main__':
    main()
