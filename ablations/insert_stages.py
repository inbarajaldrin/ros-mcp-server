#!/usr/bin/env python3
"""Where does each part spend its insert time, in CAD terms?

For every logged insert of a part: the TCP height above its predicted seat, sampled over the ACTIVE
phase, binned per mm and summed over runs (time-weighted). Against it, the CAD stage heights for that
part — the surfaces its lowest point meets on the way down (base rim, the seated parts' tops, the seat),
from the seated meshes (rotated by the assembly quaternion). A dwell at a stage height = the part
hanging on that surface; a dwell between stages = wedged where CAD has no surface (a yaw/xy miss
inside an opening).

Usage: python3 ablations/insert_stages.py [--since 2026-10-17] [--part line_green]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
LOGS = REPO / 'compliant_insertion_studio/logs'
ORDER = ['u_brown', 'u_orange', 'line_green', 'inverted_u_yellow']


def seated_extent(name):
    from compliant_insertion_studio.wrapper import cad_geometry as cg
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    V = cg._load_obj_verts(name)
    p, q = load_seat_pose_in_base('base1', name)
    Vw = R.from_quat(q).apply(V) + p
    return float(Vw[:, 2].min()), float(Vw[:, 2].max())


def stages_for(part):
    """{label: height above the seat, m} of the surfaces the part's lowest point meets (base frame)."""
    from compliant_insertion_studio.wrapper import cad_geometry as cg
    bottom, _ = seated_extent(part)
    rim = cg._load_base_top_z_local('base1')
    st = {'BASE_RIM': rim - bottom}
    for other in ORDER[:ORDER.index(part)]:
        _, top = seated_extent(other)
        st[f'top of {other}'] = top - bottom
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default='2026-10-17')
    ap.add_argument('--part', default=None)
    a = ap.parse_args()
    t0 = time.mktime(time.strptime(a.since, '%Y-%m-%d'))
    for part in ORDER:
        if a.part and part != a.part:
            continue
        hist = {}
        n_runs = 0
        for mp in sorted(LOGS.glob(f'insert_{part}_*.meta.json')):
            if mp.stat().st_mtime < t0:
                continue
            m = json.load(open(mp))
            pred = (m.get('cad_prediction') or {}).get('predicted_tcp_at_seat', {}).get('xyz_m') or m.get('predicted_seat_xyz')
            c = Path(m.get('csv_path') or str(mp).replace('.meta.json', '.csv'))
            if pred is None or not c.exists():
                continue
            rows = [r for r in csv.DictReader(open(c)) if r.get('tcp_x') not in ('', 'nan', None) and r['phase'] == 'ACTIVE']
            if len(rows) < 5:
                continue
            n_runs += 1
            t = np.array([float(r['t_s']) for r in rows])
            z = np.array([float(r['tcp_z']) for r in rows]) - pred[2]
            dt = np.diff(t, append=t[-1])
            for h, w in zip(np.round(1000 * z).astype(int), dt):
                hist[h] = hist.get(h, 0.0) + w
        if not n_runs:
            continue
        st = stages_for(part)
        total = sum(hist.values())
        print(f"\n== {part}: {n_runs} inserts, {total:.0f} s in ACTIVE. CAD stages above the seat: "
              + ", ".join(f"{k} {1000*v:+.1f} mm" for k, v in st.items()))
        # share of time within +-2 mm of each stage, at the seat, and elsewhere
        def share(lo, hi):
            return 100 * sum(w for h, w in hist.items() if lo <= h <= hi) / total
        print(f"   at the seat (<= +3 mm): {share(-99, 3):5.1f} %")
        for k, v in st.items():
            print(f"   on {k} ({1000*v:+.0f} mm, +-2): {share(round(1000*v)-2, round(1000*v)+2):5.1f} %")
        # the biggest dwells not explained by a stage
        expl = set()
        for v in st.values():
            expl.update(range(round(1000 * v) - 2, round(1000 * v) + 3))
        other = sorted(((w, h) for h, w in hist.items() if h > 3 and h not in expl), reverse=True)[:4]
        print("   unexplained dwells (height above seat, share): " + ", ".join(f"{h:+d} mm {100*w/total:.1f} %" for w, h in other))


if __name__ == '__main__':
    main()
