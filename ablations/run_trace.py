#!/usr/bin/env python3
"""Chain-of-events page for one FMB1 cycle (operator 10-23: "a graph for chain of events to trace any unnecessary
work so we know what an action is used for").

    python3 ablations/run_trace.py RUN_TIMING [--reset R1 R2 ...] [--purpose PURPOSE_JSON (default compliant_insertion_studio/docs/purpose_map.json)] [--out FILE.html]

RUN_TIMING   ablations/logs/replay_<ts>.timing.jsonl (its .motion.jsonl beside it)
--reset      the cycle's disassemble_<ts>.timing.jsonl files (default: those in the 30 min before the run)
--purpose    a purpose map (list of {action, produces, consumed_by, needed, why, cheaper_alternative} + "edges")

Writes one self-contained HTML page: a timeline (lanes: reset, replay, align child, touch child) with the arm's still
intervals shaded, still seconds per action with what each action is for, and the producer -> consumer chain.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS_A = REPO / 'ablations' / 'logs'
LOGS_C = REPO / 'compliant_insertion_studio' / 'logs'
PARTS = r'(u_brown|u_orange|line_green|inverted_u_yellow|line_\w+|fork_\w+|base\d)'


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


def motion_of(timing_path):
    p = str(timing_path).replace('.timing.jsonl', '.motion.jsonl')
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def kind(r):
    lab = re.sub(PARTS, '<part>', str(r.get('label', '')))
    lab = re.sub(r'#\d+', '', lab)
    lab = re.sub(r'-?\d+(\.\d+)?', 'N', lab)
    return f"{r.get('kind', '?')}: {lab}"[:80]


def category(r):
    k, lab = r.get('kind', ''), str(r.get('label', '')).lower()
    if k in ('camera', 'observe', 'observe_many', 'observe_base', 'base_read', 'depth_pick', 'brightness', 'part_view') \
            or any(w in lab for w in ('still_frame', 'read_icp', 'certify')):
        return 'sense'
    if 'gripper' in lab:
        return 'gripper'
    if k in ('move_inproc', 'touch_move', 'move') or any(w in lab for w in ('move', 'view', 'hover', 'clear', 'lift',
                                                                              'descend', 'land', 'carry', 'push', 'rotate')):
        return 'move'
    if k == 'seam':
        return 'seam'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('timing')
    ap.add_argument('--reset', nargs='*')
    ap.add_argument('--purpose', default=str(REPO / 'compliant_insertion_studio' / 'docs' / 'purpose_map.json'))
    ap.add_argument('--out', default=str(REPO / 'ablations' / 'logs' / 'run_trace.html'))
    ap.add_argument('--title', default='')
    a = ap.parse_args()
    run_rows = rows_of(a.timing)
    r0, r1 = min(r['t0'] for r in run_rows), max(r['t1'] for r in run_rows)
    if a.reset is not None:
        resets = a.reset
    else:
        # this cycle's reset: the disassemblies after the previous replay ended (and before this one started)
        prev_end = max([max((r['t1'] for r in rows_of(f)), default=0) for f in glob.glob(str(LOGS_A / 'replay_*.timing.jsonl'))
                        if f != a.timing and os.path.getmtime(f) < r0] or [r0 - 1800])
        resets = sorted(f for f in glob.glob(str(LOGS_A / 'disassemble_*.timing.jsonl'))
                        if prev_end < os.path.getmtime(f) <= r0 + 5 and rows_of(f) and min(r['t0'] for r in rows_of(f)) > prev_end)
    lanes = []                                        # (lane, rows)
    reset_rows, motion = [], []
    for f in resets:
        rr = rows_of(f)
        reset_rows += rr
        motion += motion_of(f)
    if reset_rows:
        lanes.append(('reset (disassembly, base push)', reset_rows))
    lanes.append(('assembly (replay)', [r for r in run_rows if r.get('kind') not in ('part',)]))
    motion += motion_of(a.timing)
    t_lo = min([r0] + [r['t0'] for r in reset_rows])
    for lane, pat in (('align (drive_align child)', 'drive_align_*.timing.jsonl'), ('insert (touch child)', 'touch_*.timing.jsonl')):
        rows = []
        for f in glob.glob(str(LOGS_C / pat)):
            if os.path.getmtime(f) >= r0:
                rows += [r for r in rows_of(f) if r0 <= r['t0'] <= r1]
        if rows:
            lanes.append((lane, rows))
    parts = [{'label': r['label'], 't0': r['t0'] - t_lo, 't1': r['t1'] - t_lo} for r in run_rows if r.get('kind') == 'part']
    motion.sort(key=lambda m: m['t'])
    still = []
    for i, m in enumerate(motion):
        if not m.get('moving') and i + 1 < len(motion):
            s0, s1 = m['t'], motion[i + 1]['t']
            if s1 - s0 >= 0.3:
                still.append([round(s0 - t_lo, 2), round(s1 - t_lo, 2)])
    # still seconds per action kind (innermost row covering each 0.05 s)
    all_rows = [r for _, rows in lanes for r in rows]
    per = defaultdict(lambda: [0.0, 0.0, 0])          # still s, wall s, count
    for r in all_rows:
        per[kind(r)][1] += r['t1'] - r['t0']; per[kind(r)][2] += 1
    for s0, s1 in still:
        t = s0 + t_lo
        while t < s1 + t_lo:
            cov = [r for r in all_rows if r['t0'] <= t < r['t1'] and r.get('kind') not in ('seam',) and r['t1'] - r['t0'] < 600]
            cov = cov or [r for r in all_rows if r['t0'] <= t < r['t1']]
            if cov:
                k = kind(min(cov, key=lambda r: r['t1'] - r['t0']))
            elif t < r0:
                k = '(reset, between scripts: base push, process starts)'
            else:
                k = '(gap: no timing row - script overhead)'
            per[k][0] += 0.1
            t += 0.1
    purpose = {}
    edges = []
    if a.purpose and os.path.exists(a.purpose):
        P = json.load(open(a.purpose))
        entries = P.get('entries', P) if isinstance(P, dict) else P
        edges = P.get('edges', []) if isinstance(P, dict) else []
        for e in entries:
            if isinstance(e, dict) and e.get('action'):
                purpose[e['action']] = e
        if isinstance(P, list):
            edges = next((x['edges'] for x in P if isinstance(x, dict) and 'edges' in x), [])
    data = {
        'title': a.title or f"Cycle {Path(a.timing).stem.replace('.timing', '')}",
        'span': round(max(r1, max([r['t1'] for r in reset_rows], default=r1)) - t_lo, 1),
        'assembly': [round(r0 - t_lo, 1), round(r1 - t_lo, 1)],
        'lanes': [{'name': n, 'bars': [{'l': str(r.get('label', ''))[:90], 'k': r.get('kind', ''), 'c': category(r),
                                         'kk': kind(r), 't0': round(r['t0'] - t_lo, 2), 't1': round(r['t1'] - t_lo, 2)}
                                        for r in rows]} for n, rows in lanes],
        'parts': parts, 'still': still,
        'per': sorted([{'k': k, 'still': round(v[0], 1), 'wall': round(v[1], 1), 'n': v[2]} for k, v in per.items()],
                      key=lambda x: -x['still']),
        'purpose': purpose, 'edges': edges,
    }
    html = open(REPO / 'ablations' / 'run_trace_template.html').read().replace('/*DATA*/null', json.dumps(data))
    open(a.out, 'w').write(html)
    print(a.out)


if __name__ == '__main__':
    main()
