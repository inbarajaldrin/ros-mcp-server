#!/usr/bin/env python3
"""Collect the real cell's measured base1 / part poses from the FMB1 run logs (read-only).

    python3 -I collect_cell_facts.py /tmp/fmb1_*.log > ../evidence/run_log_poses.json

Every number is copied from a log line the cell printed; nothing is estimated. Frames:
  UR `base`   as logged (mm, deg).
  twin world  p_world = (-x, -y, z + 80 mm), yaw_world = yaw + 180 deg (UR base -> base_link is pi about z;
              base_link origin 0.080 m above the table-top plane = world z 0; primitives/shared/config.py:3).
Lines read:
  [move_base] DONE in N pushes: (x, y) mm yaw deg           the base after the reset's push (ablations/move_base.py:297 area)
  [base] base1 observed at (x, y, z) mm, yaw Y deg          the replay's applied base read (replay_real_assembly.py:2909)
  [base] CAD fit ...: {json}  /  [move_base] base1 at ... [{json}]   depth 6-dof CAD fit (z_mm, roll_pitch_deg, rms_mm)
  [spot] <part>: free spot in the staging region -> flange (x, y) mm   disassembly set-down (ablations/disassemble.py)
  [spot] <part>: not seen from the staging view - set down at (x, y) mm
  [pick] <part> grasp G: ... (pose used: (x, y, z) mm rpy (r, p, y))  the observed part pose the pick judged (replay:598)
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

R_DONE = re.compile(r'\[move_base\] DONE in (\d+) pushes: \(([-\d.]+), ([-\d.]+)\) mm ([-\d.]+) deg')
R_OBS = re.compile(r'\[base\] base1 observed at \(([-\d.]+), ([-\d.]+), ([-\d.]+)\) mm, yaw ([-\d.]+) deg')
R_FIT = re.compile(r'(?:\[base\] CAD fit[^:]*: |\[move_base\] base1 at .*?\[)(\{"rms_mm".*?\})')
R_SPOT = re.compile(r'\[spot\] (\w+): (?:free spot in the staging region -> flange \(([-\d.]+), ([-\d.]+)\) mm'
                    r'|not seen from the staging view - set down at \(([-\d.]+), ([-\d.]+)\) mm)')
R_PICK = re.compile(r'\[pick\] (\w+) grasp (\d+): .*?pose used: \(([-\d.]+), ([-\d.]+), ([-\d.]+)\) mm '
                    r'rpy \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)')
PARTS = ('u_brown', 'u_orange', 'line_green', 'inverted_u_yellow')


def world(x, y, z=None, yaw=None):
    out = {'x_mm': round(-x, 1), 'y_mm': round(-y, 1)}
    if z is not None:
        out['z_mm'] = round(z + 80.0, 1)
    if yaw is not None:
        out['yaw_deg'] = round(((yaw + 180.0 + 180.0) % 360.0) - 180.0, 2)
    return out


def summ(vals):
    if not vals:
        return None
    return {'n': len(vals), 'min': round(min(vals), 2), 'max': round(max(vals), 2), 'mean': round(st.mean(vals), 2),
            'median': round(st.median(vals), 2), 'stdev': round(st.stdev(vals), 2) if len(vals) > 1 else None}


def main(paths):
    done, obs, fits, spots, picks = [], [], [], [], []
    for p in sorted(paths):
        name = Path(p).name
        txt = Path(p).read_text(errors='replace')
        if 'base1' not in txt:
            continue
        for ln, line in enumerate(txt.splitlines(), 1):
            src = f'{p}:{ln}'
            m = R_DONE.search(line)
            if m:
                done.append(dict(src=src, pushes=int(m[1]), x=float(m[2]), y=float(m[3]), yaw=float(m[4])))
            m = R_OBS.search(line)
            if m:
                obs.append(dict(src=src, x=float(m[1]), y=float(m[2]), z=float(m[3]), yaw=float(m[4])))
            m = R_FIT.search(line)
            if m and ('base1' in line or '[base] CAD fit' in line):
                try:
                    d = json.loads(m[1]); d['src'] = src; fits.append(d)
                except Exception:
                    pass
            m = R_SPOT.search(line)
            if m:
                x, y = (m[2], m[3]) if m[2] else (m[4], m[5])
                spots.append(dict(src=src, part=m[1], x=float(x), y=float(y), seen=bool(m[2])))
            m = R_PICK.search(line)
            if m:
                picks.append(dict(src=src, part=m[1], grasp=int(m[2]), x=float(m[3]), y=float(m[4]), z=float(m[5]),
                                  roll=float(m[6]), pitch=float(m[7]), yaw=float(m[8])))
    # applied base reads at replay start: drop the far-off outliers of a bench test (yaw < 0: the -53 deg tests)
    obs_ok = [o for o in obs if o['yaw'] > 0]
    out = {
        'files_scanned': len(paths),
        'move_base_done': {'n': len(done), 'x_mm': summ([d['x'] for d in done]), 'y_mm': summ([d['y'] for d in done]),
                           'yaw_deg': summ([d['yaw'] for d in done]), 'rows': done},
        'replay_base_observed': {
            'n_all': len(obs), 'n_used': len(obs_ok),
            'excluded': [o for o in obs if o['yaw'] <= 0],
            'x_mm': summ([o['x'] for o in obs_ok]), 'y_mm': summ([o['y'] for o in obs_ok]),
            'z_mm_as_logged': summ([o['z'] for o in obs_ok]), 'yaw_deg': summ([o['yaw'] for o in obs_ok]),
            'world': {'x_mm': summ([-o['x'] for o in obs_ok]), 'y_mm': summ([-o['y'] for o in obs_ok]),
                      'yaw_deg': summ([o['yaw'] + 180.0 for o in obs_ok])},
            'rows': obs},
        'base_cad_fit': {'n': len(fits), 'z_mm': summ([f['z_mm'] for f in fits if 'z_mm' in f]),
                         'roll_deg': summ([f['roll_pitch_deg'][0] for f in fits if 'roll_pitch_deg' in f]),
                         'pitch_deg': summ([f['roll_pitch_deg'][1] for f in fits if 'roll_pitch_deg' in f]),
                         'rms_mm': summ([f['rms_mm'] for f in fits if 'rms_mm' in f]),
                         'xy_minus_marker_mm': {
                             'dx': summ([f['xy_mm'][0] - f['marker_mm'][0] for f in fits if 'marker_mm' in f]),
                             'dy': summ([f['xy_mm'][1] - f['marker_mm'][1] for f in fits if 'marker_mm' in f])}},
        'spots': {}, 'picks': {},
    }
    for part in PARTS:
        S = [s for s in spots if s['part'] == part]
        P = [q for q in picks if q['part'] == part]
        out['spots'][part] = {'n': len(S), 'x_mm': summ([s['x'] for s in S]), 'y_mm': summ([s['y'] for s in S]),
                              'distinct_xy_mm': sorted({(s['x'], s['y']) for s in S}),
                              'world_x_mm': summ([-s['x'] for s in S]), 'world_y_mm': summ([-s['y'] for s in S])}
        out['picks'][part] = {'n': len(P), 'grasp_ids': sorted({q['grasp'] for q in P}),
                              'x_mm': summ([q['x'] for q in P]), 'y_mm': summ([q['y'] for q in P]),
                              'z_mm': summ([q['z'] for q in P]), 'world_z_mm': summ([q['z'] + 80.0 for q in P]),
                              'roll_deg': summ([q['roll'] for q in P]), 'pitch_deg': summ([q['pitch'] for q in P]),
                              'stances_rp_deg': sorted({(round(q['roll']), round(q['pitch'])) for q in P}),
                              'rows': P}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
