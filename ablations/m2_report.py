#!/usr/bin/env python3
"""M2 evidence: the aim error over base position, and sigmas, from the logs of FMB1 cycles (10-23).

    python3 ablations/m2_report.py /tmp/fmb1_55.log /tmp/fmb1_56.log ... [--md compliant_insertion_studio/docs/M2_ERROR_MAP.md]

Per cycle and part, from the replay's console log and the drive-align logs it names:
  base     drive-mate's measured base for the mate (x, y mm; yaw from the align record)
  r0       the first servo reading (U parts: marker servo; line_green / yellow: depth servo) - what the aim missed by
           before any correction, i.e. the vision-chain error at that base pose
  moved    the servo's total aim correction = the aim error at that base pose (r0 is after the servo prior)
  slide    the insert's measured seat minus its prediction (x, y, z mm) - what the probes / channel still had to find
  depth    ExecuteMate's certified seat depth (mm)
Sigmas: per part, the spread of r0, slide and depth over the cycles, and of the base read (drive-mate's per-mate reads
of the same cycle's base: it does not move between mates unless an insert pushed it).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

PARTS = ('u_brown', 'u_orange', 'line_green', 'inverted_u_yellow')


def cycle(log):
    txt = Path(log).read_text(errors='replace')
    out = {'log': log, 'parts': {}}
    m = re.search(r'move_base\] DONE in \d+ pushes: \(([-0-9.]+), ([-0-9.]+)\) mm ([-0-9.]+) deg', txt)
    out['base_set'] = [float(m.group(1)), float(m.group(2)), float(m.group(3))] if m else None
    m = re.search(r'wall (\d+) s', txt)
    for p in PARTS:
        d = {}
        m = re.search(rf'\[drive_mate\] fmb1/mate:\d+: measured base \(([-0-9.]+), ([-0-9.]+)\) mm, grasp \d+[^\n]*\n(?:[^\n]*\n){{0,3}}?'
                      rf'[^\n]*aligned by [^\n]*drive_align_{p}_(\d{{8}}_\d{{6}})\.log', txt)
        al = re.findall(rf'drive_align_{p}_(\d{{8}}_\d{{6}})\.log', txt)
        if al:
            d['align_log'] = f'compliant_insertion_studio/logs/drive_align_{p}_{al[-1]}.log'
        bm = re.findall(rf'\[drive_mate\] fmb1/mate:\d+: measured base \(([-0-9.]+), ([-0-9.]+)\) mm[^\n]*\n(?:(?!\[drive_mate\] fmb1/mate:\d+: measured)[^\n]*\n)*?[^\n]*drive_align_{p}_', txt)
        if bm:
            d['base'] = [float(bm[-1][0]), float(bm[-1][1])]
        sl = re.findall(rf'\[seated-offsets\] {p}: seat - prediction \[([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\] mm', txt)
        if sl:
            d['slide'] = [float(v) for v in sl[-1]]
        out['parts'][p] = d
    for addr, dep in re.findall(r'\[execute_mate\] fmb1/mate:0(\d): placed=True refusal=\'\'\s+depth=([-0-9.e]+)', txt):
        out['parts'][PARTS[int(addr)]]['depth'] = round(1000 * float(dep), 2)
    for p, d in out['parts'].items():
        f = d.get('align_log')
        if not f or not Path(f).exists():
            continue
        a = Path(f).read_text(errors='replace')
        r0 = re.search(rf'{p} round 0: (?:part )?\[([-0-9.]+), ([-0-9.]+)\] mm', a)
        if r0:
            d['r0'] = [float(r0.group(1)), float(r0.group(2))]
        mv = re.findall(rf'\[servo\] {p}: aim moved by \(([-0-9.]+), ([-0-9.]+)\) mm', a) or \
            re.findall(rf'\[dservo\] {p}: (?:converged, aim|aim shifted \()([-0-9.]+)(?: / |, )([-0-9.]+)', a)
        if mv:
            d['moved'] = [float(mv[-1][0]), float(mv[-1][1])]
        if 'misread' in a:
            d['note'] = 'servo misread -> held-part aim'
    return out


def sd(rows):
    a = np.array([r for r in rows if r is not None], float)
    return None if len(a) < 2 else np.round(a.std(axis=0, ddof=1), 2).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--md')
    a = ap.parse_args()
    cs = [cycle(l) for l in a.logs]
    lines = ['| cycle | base set (x, y mm, yaw) | part | measured base | r0 (mm) | servo moved | slide (x, y, z) | depth |',
             '|---|---|---|---|---|---|---|---|']
    for c in cs:
        for p in PARTS:
            d = c['parts'][p]
            lines.append(f"| {Path(c['log']).stem} | {c['base_set'] or 'layout'} | {p} | {d.get('base')} | {d.get('r0')} | "
                         f"{d.get('moved')} | {d.get('slide')} | {d.get('depth')} |" + (f" {d['note']}" if d.get('note') else ''))
    lines += ['', '| part | sigma aim error = servo moved (mm) | sigma slide (mm) | sigma depth (mm) | n |', '|---|---|---|---|---|']
    for p in PARTS:
        ds = [c['parts'][p] for c in cs]
        dep = [[d['depth']] for d in ds if 'depth' in d]
        lines.append(f"| {p} | {sd([d.get('moved') for d in ds])} | {sd([d.get('slide') for d in ds])} | {sd(dep)} | {len(ds)} |")
    txt = '\n'.join(lines)
    print(txt)
    if a.md:
        Path(a.md).write_text('# M2: aim error over base position (FMB1, 10-23)\n\n' + __doc__.split('\n\n', 1)[1] + '\n' + txt + '\n')
        print(f'\nwritten {a.md}')


if __name__ == '__main__':
    main()
