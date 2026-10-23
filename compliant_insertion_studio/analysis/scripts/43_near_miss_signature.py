#!/usr/bin/env python3
"""Is there a signal when the SEARCH spiral passes NEAR the pocket, before it drops in?

Question (2026-10-07): the spiral often passes close to the pocket on an earlier turn and
only drops in a full turn later. If a near-miss leaves a recognisable signature, the
search could switch to local exploration at that point instead of completing the turn.

For every insert that searched on the rim and then seated (raw criterion: lowest tcp_z
within 5 mm of the predicted seat), using only raw CSV columns:
  hole_xy   := TCP xy at the deepest sample (peg in the slot)
  window    := first rim contact + GRACE_S  ..  final drop (z leaves the rim plateau by
               DROP_M for good)
  per sample: d = |tcp_xy - hole_xy|, rel_fz = |fz| / median |fz| over the window,
              z_dip = median rim z - tcp_z  (positive = lower than the rim plateau)
Then (1) pooled signal-vs-distance bins, (2) near-miss passes: local minima of d below
NEAR_M before the final approach, and the time from the first close pass to the drop.

Usage: 43_near_miss_signature.py DIR [DIR...] [--object u_brown]
"""
import csv
import glob
import json
import math
import os
import statistics as st
import sys

CONTACT_N = 3.0
ABOVE_SEAT_M = 0.010
GRACE_S = 1.0
DROP_M = 0.003
NEAR_M = 0.003
PLATEAU_M = 0.003
CLOSE_M = 0.0015
BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 99.0)]   # mm


def load(base):
    m = json.load(open(base + '.meta.json'))
    seat = m['cad_prediction']['predicted_tcp_at_seat']['xyz_m']
    rows = [r for r in csv.DictReader(open(base + '.csv')) if r.get('phase') == 'ACTIVE']
    if not rows:
        return None
    T = [float(r['t_s']) for r in rows]
    X = [float(r['tcp_x']) for r in rows]
    Y = [float(r['tcp_y']) for r in rows]
    Z = [float(r['tcp_z']) for r in rows]
    F = [abs(float(r['fz'])) for r in rows]
    if min(Z) - seat[2] >= 0.005:
        return None                                    # did not seat
    k = min(range(len(Z)), key=lambda i: Z[i])
    hole = (X[k], Y[k])
    on = [f > CONTACT_N and z > seat[2] + ABOVE_SEAT_M for f, z in zip(F, Z)]
    first = next((i for i, o in enumerate(on) if o), None)
    if first is None:
        return None
    # rim plateau = median z over on-rim samples before the deepest point
    rim_z = st.median([Z[i] for i in range(first, k) if on[i]] or [Z[first]])
    # final drop = last time z was still on the plateau before falling to the seat
    end = max((i for i in range(first, k) if Z[i] > rim_z - DROP_M), default=first)
    start = next((i for i in range(first, end) if T[i] - T[first] >= GRACE_S), None)
    if start is None or T[end] - T[start] < 2.0:
        return None                                    # seated on touchdown — no search
    # On the rim plateau only: the force-mode startup spike can register as contact
    # while the TCP is still 40-70 mm up on the approach, and those samples pass over
    # the hole xy at height.
    idx = [i for i in range(start, end + 1) if abs(Z[i] - rim_z) < PLATEAU_M]
    if len(idx) < 50:
        return None
    fmed = st.median([F[i] for i in idx])
    samples = [(T[i] - T[idx[0]], 1000 * math.hypot(X[i] - hole[0], Y[i] - hole[1]),
                F[i] / fmed, 1000 * (rim_z - Z[i])) for i in idx]
    return dict(search_s=T[end] - T[idx[0]], samples=samples)


def passes(samples):
    """Local minima of distance below NEAR_M, >= 1 s apart, excluding the last 1 s."""
    t_end = samples[-1][0]
    out = []
    for i in range(1, len(samples) - 1):
        t, d = samples[i][0], samples[i][1]
        if d < 1000 * NEAR_M and t < t_end - 1.0 and d <= samples[i - 1][1] and d <= samples[i + 1][1]:
            if not out or t - out[-1][0] > 1.0:
                out.append(samples[i])
            elif d < out[-1][1]:
                out[-1] = samples[i]
    return out


def main():
    obj = None
    dirs = []
    a = sys.argv[1:]
    while a:
        x = a.pop(0)
        if x == '--object':
            obj = a.pop(0)
        else:
            dirs.append(x)
    bases = sorted({p[:-len('.meta.json')] for d in dirs
                    for p in glob.glob(os.path.join(d, 'insert_*.meta.json'))
                    if obj is None or os.path.basename(p).startswith(f'insert_{obj}_')})
    pooled = {b: [] for b in BINS}
    print(f"{'run':34s} {'search s':>8s} {'near passes (t s, d mm, rel|fz|, z dip mm)':s}")
    gains = []
    for b in bases:
        if not os.path.exists(b + '.csv'):
            continue
        r = load(b)
        if r is None:
            continue
        for _, d, f, zd in r['samples']:
            for lo, hi in BINS:
                if lo <= d < hi:
                    pooled[(lo, hi)].append((f, zd))
        ps = passes(r['samples'])
        desc = '  '.join(f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.2f}, {p[3]:+.2f})" for p in ps) or '-'
        close = [p for p in ps if p[1] < 1000 * CLOSE_M]
        if close:
            gains.append(r['search_s'] - close[0][0])
        print(f"{os.path.basename(b):34s} {r['search_s']:8.1f}  {desc}")
    print("\npooled by distance to the final hole:")
    print(f"  {'d (mm)':>9s} {'n':>6s} {'rel|fz| med':>11s} {'p10':>6s} {'z dip med':>9s} {'p90':>6s}")
    for (lo, hi), v in pooled.items():
        if not v:
            continue
        fs = sorted(x[0] for x in v)
        zs = sorted(x[1] for x in v)
        p = lambda arr, q: arr[min(len(arr) - 1, int(q * len(arr)))]
        print(f"  {lo:4.1f}-{hi:<4.1f} {len(v):6d} {st.median(fs):11.2f} {p(fs, .10):6.2f} "
              f"{st.median(zs):+9.2f} {p(zs, .90):+6.2f}")
    if gains:
        print(f"\nruns with an earlier pass within {1000*CLOSE_M:.1f} mm: {len(gains)}; time from that "
              f"pass to the drop: median {st.median(gains):.1f} s, max {max(gains):.1f} s")


if __name__ == '__main__':
    main()
