#!/usr/bin/env python3
"""Every interval the arm stood still in a run, and what was running then (operator 10-23: "arm being still is the
thing you got to also check in a run ... for unnecessary reasons").

    python3 ablations/still_audit.py ablations/logs/replay_<ts>.timing.jsonl [--min 0.8] [--top 25]

Reads the run's motion recorder (replay_<ts>.motion.jsonl: TCP moving/still transitions) and the timing rows of the
replay AND of every child that writes its own (drive_align_*.timing.jsonl in compliant_insertion_studio/logs) inside
the run's window. Each still interval is attributed to the innermost timing row(s) overlapping it; time no row covers
is '(gap)' - the replay's own overhead between steps. Prints the intervals and the totals per kind of row.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def rows_of(path):
    out = []
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if 't0' in r and 't1' in r:
            out.append(r)
    return out


def kind(r):
    """A label without part names / indices, so repeats add up."""
    lab = re.sub(r'(u_brown|u_orange|line_green|inverted_u_yellow|line_\w+|fork_\w+|base\d)', '<part>', str(r.get('label', '')))
    lab = re.sub(r'#\d+', '', lab)
    lab = re.sub(r'\d+(\.\d+)?', 'N', lab)
    return f"{r.get('kind', '?')}: {lab}"[:70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('timing')
    ap.add_argument('--min', type=float, default=0.8, help='only still intervals at least this long (s)')
    ap.add_argument('--top', type=int, default=25)
    a = ap.parse_args()
    motion = a.timing.replace('.timing.jsonl', '.motion.jsonl')
    M = [json.loads(l) for l in open(motion)]
    rows = rows_of(a.timing)
    t_lo, t_hi = min(r['t0'] for r in rows), max(r['t1'] for r in rows)
    for f in glob.glob(str(REPO / 'compliant_insertion_studio' / 'logs' / '*.timing.jsonl')):
        if os.path.getmtime(f) < t_lo:
            continue
        child = [r for r in rows_of(f) if t_lo <= r['t0'] <= t_hi]
        for r in child:
            r['label'] = f"[{Path(f).name.split('_2026')[0]}] {r.get('label', '')}"
        rows += child
    still = []
    for i, m in enumerate(M):
        if not m.get('moving') and i + 1 < len(M):
            s0, s1 = m['t'], M[i + 1]['t']
            if s1 - s0 >= a.min and t_lo <= s0 <= t_hi:
                still.append((s0, s1))
    total = defaultdict(float)
    listing = []
    for s0, s1 in still:
        # innermost rows: those overlapping, sorted by length; attribute each second to the shortest row covering it
        ov = [r for r in rows if r['t1'] > s0 and r['t0'] < s1 and (r['t1'] - r['t0']) < 600]
        ov.sort(key=lambda r: r['t1'] - r['t0'])
        step = 0.05; t = s0; parts = defaultdict(float)
        while t < s1:
            cov = next((r for r in ov if r['t0'] <= t < r['t1'] and r.get('kind') not in ('part', 'seam')), None)
            cov = cov or next((r for r in ov if r['t0'] <= t < r['t1']), None)
            parts[kind(cov) if cov else '(gap: no timing row - replay overhead)'] += step
            t += step
        for k, v in parts.items():
            total[k] += v
        top = sorted(parts.items(), key=lambda kv: -kv[1])[:2]
        listing.append((s1 - s0, s0 - t_lo, top))
    n = sum(s1 - s0 for s0, s1 in still)
    print(f"run window {t_hi - t_lo:.0f} s, still intervals >= {a.min} s: {len(still)}, {n:.0f} s still in them")
    print("\nstill time by what was running (all intervals):")
    for k, v in sorted(total.items(), key=lambda kv: -kv[1])[:a.top]:
        print(f"  {v:6.1f} s  {k}")
    print(f"\nlongest still intervals:")
    for d, at, top in sorted(listing, key=lambda x: -x[0])[:a.top]:
        print(f"  {d:5.1f} s at +{at:6.1f}  " + ' | '.join(f"{k} {v:.1f}s" for k, v in top))


if __name__ == '__main__':
    main()
