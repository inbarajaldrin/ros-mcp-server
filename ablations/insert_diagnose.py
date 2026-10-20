#!/usr/bin/env python3
"""Classify one insert from its raw trace: was the time on the rim yaw, xy aim, or the search itself?

  yaw    sustained |Tz| during the search (part twisted against the slot walls) - the part sits until the
         spiral happens to free it
  xy     |Tz| small, lateral force persistently one way, the drop comes after a net slide in that direction
  search |Tz| small and |F_lat| small, yet a long spiral before the drop: the search wandered, not blocked

Prints one line per insert. --watch prints a line for every new insert meta as soon as it is written
(for following a run live). Thresholds are first guesses from the 2026-10-17..20 data, to be refined.

  python3 ablations/insert_diagnose.py --watch           # live, one line per insert
  python3 ablations/insert_diagnose.py --last 6 --part u_brown
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

LOGS = Path(__file__).resolve().parents[1] / 'compliant_insertion_studio/logs'
TZ_YAW_NM = 0.06          # median |Tz| during the search above this = twisted (11:01 over-rotation: 0.17)
FLAT_XY_N = 1.5           # median lateral force above this, persistently one way = xy
PERSIST = 0.6             # |mean lateral vector| / mean |lateral| above this = "one way"


def diagnose(meta_path):
    m = json.load(open(meta_path))
    c = Path(m.get('csv_path') or str(meta_path).replace('.meta.json', '.csv'))
    r = [x for x in csv.DictReader(open(c)) if x.get('tcp_x') not in ('', 'nan', None) and x['phase'] == 'ACTIVE']
    if len(r) < 20:
        return None
    t = np.array([float(x['t_s']) for x in r]); z = np.array([float(x['tcp_z']) for x in r])
    fx = np.array([float(x['fx']) for x in r]); fy = np.array([float(x['fy']) for x in r])
    fz = np.array([float(x['fz']) for x in r]); tz = np.array([float(x['tz']) for x in r])
    xy = np.array([[float(x['tcp_x']), float(x['tcp_y'])] for x in r])
    ic = next((i for i in range(len(r)) if abs(fz[i]) > 3.0), None)
    if ic is None:
        return dict(run=meta_path.name, verdict='no contact')
    # search window: from contact until the part starts to drop (z within 4 mm of its final depth)
    drop = next((i for i in range(ic, len(r)) if z[i] < z.min() + 0.004), len(r) - 1)
    w = slice(ic, max(drop, ic + 1))
    rim_s = t[drop] - t[ic]
    tz_med = float(np.median(tz[w]))
    lat = np.c_[fx[w], fy[w]]
    lat_mag = float(np.median(np.linalg.norm(lat, axis=1)))
    persist = float(np.linalg.norm(lat.mean(0)) / (np.linalg.norm(lat, axis=1).mean() + 1e-9))
    slide = 1000 * (xy[drop] - xy[ic])
    if rim_s < 2.0:
        verdict = 'clean (dropped in)'
    elif abs(tz_med) > TZ_YAW_NM:
        verdict = f"YAW (Tz {tz_med:+.3f} N*m: part {'under' if tz_med > 0 else 'over'}-rotated vs the slot)"
    elif lat_mag > FLAT_XY_N and persist > PERSIST:
        verdict = f"XY (lateral {lat_mag:.1f} N one way, slide ({slide[0]:+.1f}, {slide[1]:+.1f}) mm)"
    else:
        verdict = f"SEARCH (Tz {tz_med:+.3f}, lateral {lat_mag:.1f} N, slide {np.linalg.norm(slide):.1f} mm)"
    return dict(run=meta_path.name.replace('insert_', '').replace('.meta.json', ''), part=m.get('object'),
                rim_s=rim_s, tz=tz_med, lat=lat_mag, persist=persist, slide=slide, verdict=verdict)


def line(d):
    if d is None:
        return None
    if 'rim_s' not in d:
        return f"{d['run']}: {d['verdict']}"
    return f"{d['run']:34s} rim {d['rim_s']:5.1f} s  Tz {d['tz']:+.3f}  lat {d['lat']:.1f} N  -> {d['verdict']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--last', type=int, default=5)
    ap.add_argument('--part', default=None)
    a = ap.parse_args()
    if a.watch:
        seen = {p.name for p in LOGS.glob('insert_*.meta.json')}
        while True:
            for p in sorted(LOGS.glob('insert_*.meta.json'), key=lambda p: p.stat().st_mtime):
                if p.name in seen:
                    continue
                seen.add(p.name)
                time.sleep(1.0)                      # let the CSV close
                try:
                    out = line(diagnose(p))
                except Exception as e:
                    out = f"{p.name}: not diagnosed ({e})"
                if out:
                    print(out, flush=True)
            time.sleep(2.0)
    metas = sorted(LOGS.glob(f"insert_{a.part or ''}*.meta.json"), key=lambda p: p.stat().st_mtime)[-a.last:]
    for p in metas:
        out = line(diagnose(p))
        if out:
            print(out)


if __name__ == '__main__':
    main()
