#!/usr/bin/env python3
"""FMB1 hands-off loop: robot disassembles, then assembles; repeat and record (2026-10-14).

    python3 -u ablations/fmb1_loop.py --cycles 3 [--start-with assemble] [--layout staging [--all-flat]]

Each cycle: ablations/disassemble.py (seated parts -> free table spots, with the insert's own
grasp), then replay_real_assembly.py (one command, insert recovery + line_green seat check).
Fails closed: the first non-zero exit stops the loop with the arm where that step left it.
Per cycle, appends to ablations/logs/fmb1_loop.jsonl: exit codes, per-part insert outcome and
the insert's xy correction (seat TCP - CAD-predicted seat; a direct measure of grasp error),
recoveries used and seat-check results.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
J = REPO / 'ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json'
LOGS = REPO / 'compliant_insertion_studio/logs'
OUT = REPO / 'ablations/logs/fmb1_loop.jsonl'


def run(argv, log):
    print(f"\n### {' '.join(argv[2:])}  (log {log})", flush=True)
    with open(log, 'w') as fh:
        p = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
    print(f"### rc={p.returncode}", flush=True)
    return p.returncode


def insert_summary(since):
    """Per insert meta written after `since`: outcome and the xy correction the insert made."""
    out = []
    for m in sorted(LOGS.glob('insert_*.meta.json'), key=lambda p: p.stat().st_mtime):
        if m.stat().st_mtime < since:
            continue
        d = json.load(open(m))
        pred = (d.get('cad_prediction') or {}).get('predicted_tcp_at_seat', {}).get('xyz_m') or d.get('predicted_seat_xyz')
        csv_p = Path(d.get('csv_path') or str(m).replace('.meta.json', '.csv'))
        dev = None
        try:
            rows = [r for r in csv.DictReader(open(csv_p)) if r['tcp_x'] not in ('', 'nan')]
            last = np.array([float(rows[-1][k]) for k in ('tcp_x', 'tcp_y', 'tcp_z')])
            dev = [round(1000 * float(v), 2) for v in last - np.array(pred)] if pred else None
        except Exception:
            pass
        out.append(dict(file=m.name, object=d.get('object'), outcome=d.get('outcome'),
                        reason=d.get('outcome_reason'), seat_minus_pred_mm=dev))
    return out


def grep(path, *keys):
    try:
        return [l.strip()[:300] for l in open(path) if any(k in l for k in keys)]
    except FileNotFoundError:
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cycles', type=int, default=1)
    ap.add_argument('--start-with', choices=['disassemble', 'assemble'], default='disassemble',
                    help='disassemble when the parts are seated now; assemble when they lie on the table')
    ap.add_argument('--layout', choices=['spots', 'staging'], default='spots',
                    help="disassembly layout: spots (upright, home view) or staging (operator's generator "
                         "idea: staging region, drawn yaws, tipped flat; one tippable part upright per cycle)")
    ap.add_argument('--all-flat', action='store_true', help='with --layout staging: tip all three')
    a = ap.parse_args()
    if os.nice(0) < 5:                     # the UR driver has no RT priority here: run below it
        os.nice(5 - os.nice(0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    py = [sys.executable, '-u']
    for c in range(1, a.cycles + 1):
        ts = time.strftime('%Y%m%d_%H%M%S'); t0 = time.time()
        rec = dict(cycle=c, start=ts)
        if c > 1 or a.start_with == 'disassemble':
            log = OUT.parent / f'loop_{ts}_disassemble.log'
            rec['disassemble_rc'] = run(py + [str(REPO / 'ablations/disassemble.py'), '--assembly-json', 'x',
                                              '--layout', a.layout, '--cycle', str(c - 1)]
                                             + (['--all-flat'] if a.all_flat else []), log)
            rec['disassemble_fail'] = grep(log, 'FAILED', 'stopping', 'contact')
            if rec['disassemble_rc'] != 0:
                json.dump(rec, open(OUT, 'a')); open(OUT, 'a').write('\n')
                print(json.dumps(rec, indent=1)); return 1
        log = OUT.parent / f'loop_{ts}_assemble.log'
        t_asm = time.time()
        rec['assemble_rc'] = run(py + [str(REPO / 'ablations/replay_real_assembly.py'), '--assembly-json', str(J)], log)
        rec['inserts'] = insert_summary(t_asm)
        rec['recoveries'] = grep(log, 'recovery ', 'seat-check', 'staging check')
        rec['completed'] = grep(log, 'Completed:', 'FAILED at:')
        rec['duration_s'] = round(time.time() - t0)
        with open(OUT, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        print(json.dumps(rec, indent=1), flush=True)
        if rec['assemble_rc'] != 0:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
