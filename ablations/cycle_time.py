#!/usr/bin/env python3
"""Where a replay's time goes (2026-10-21, Speed track T0).

    python3 ablations/cycle_time.py ablations/logs/replay_<ts>.timing.jsonl [--top 15] [--json]

Reads the replay's timing rows (one per subprocess / read / seam call / part) and the motion recorder's transitions
(same stem, .motion.jsonl). For the whole run and per part: wall time, time the arm was actually moving (TCP speed >
2 mm/s), and the rest; per block kind and label: count, wall, moving, still. Blocks overlap only by nesting (a part
contains its subprocesses), so per-part numbers use the part rows and per-block numbers the leaf rows; time inside a
part but in no leaf block is the replay's own in-process work ("in-process").
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def moving_intervals(motion_rows):
    iv, start = [], None
    for r in motion_rows:
        if r.get('end'):
            break
        if r.get('moving') and start is None:
            start = r['t']
        elif not r.get('moving') and start is not None:
            iv.append((start, r['t'])); start = None
    if start is not None:
        iv.append((start, motion_rows[-1]['t']))
    return iv


def moving_in(iv, t0, t1):
    return sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in iv)


def union_len(spans):
    tot, cur = 0.0, None
    for a, b in sorted(spans):
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return tot + (cur[1] - cur[0] if cur else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('timing'); ap.add_argument('--top', type=int, default=15); ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    rows = load(a.timing)
    mp = Path(a.timing.replace('.timing.jsonl', '.motion.jsonl'))
    iv = moving_intervals(load(mp)) if mp.exists() else []
    parts = [r for r in rows if r['kind'] == 'part']
    nonpart = [r for r in rows if r['kind'] != 'part']
    # count a block once: drop rows nested inside another non-part row (observe_base -> observe)
    leaves = [r for r in nonpart if not any(o is not r and o['t0'] <= r['t0'] and r['t1'] <= o['t1']
                                            and (o['t1'] - o['t0']) > (r['t1'] - r['t0']) for o in nonpart)]
    t0 = min(r['t0'] for r in rows); t1 = max(r['t1'] for r in rows)
    out = {'run': {'wall_s': t1 - t0, 'moving_s': moving_in(iv, t0, t1)}, 'parts': [], 'blocks': []}
    for p in parts:
        inner = [r for r in leaves if r['t0'] >= p['t0'] - 0.01 and r['t1'] <= p['t1'] + 0.01]
        covered = union_len([(r['t0'], r['t1']) for r in inner])
        w = p['t1'] - p['t0']
        out['parts'].append({'part': p['label'], 'wall_s': w, 'moving_s': moving_in(iv, p['t0'], p['t1']),
                             'in_process_s': w - covered, 'n_blocks': len(inner)})
    agg = defaultdict(lambda: {'n': 0, 'wall_s': 0.0, 'moving_s': 0.0})
    for r in leaves:
        lab = r['label'].split('#')[-1] if '#' in r['label'] else r['label']     # "u_brown#6 move_to_grasp" -> "6 move_to_grasp"
        key = f"{r['kind']}: {' '.join(lab.split()[1:]) or lab}" if r['kind'] == 'subprocess' and '#' in r['label'] else f"{r['kind']}: {lab}"
        g = agg[key]; g['n'] += 1; g['wall_s'] += r['t1'] - r['t0']; g['moving_s'] += moving_in(iv, r['t0'], r['t1'])
    out['blocks'] = sorted(({'block': k, **v, 'still_s': v['wall_s'] - v['moving_s']} for k, v in agg.items()),
                           key=lambda b: -b['still_s'])
    if a.json:
        print(json.dumps(out, indent=1)); return
    R = out['run']
    print(f"run: wall {R['wall_s']:.0f} s, arm moving {R['moving_s']:.0f} s ({100 * R['moving_s'] / max(R['wall_s'], 1e-9):.0f}%)"
          + ('' if iv else '   [no motion recording]'))
    for p in out['parts']:
        print(f"  {p['part']:<20} wall {p['wall_s']:6.0f} s  moving {p['moving_s']:5.0f} s  "
              f"still {p['wall_s'] - p['moving_s']:5.0f} s  (in-process {p['in_process_s']:4.0f} s, {p['n_blocks']} blocks)")
    print(f"\nblocks by time with the arm still (top {a.top}):")
    print(f"  {'block':<58} {'n':>3} {'wall':>7} {'moving':>7} {'still':>7}")
    for b in out['blocks'][:a.top]:
        print(f"  {b['block'][:58]:<58} {b['n']:>3} {b['wall_s']:7.1f} {b['moving_s']:7.1f} {b['still_s']:7.1f}")


if __name__ == '__main__':
    main()
