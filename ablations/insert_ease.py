#!/usr/bin/env python3
"""How easily did each insert go in? Per insert, from the raw logs (never the FSM's own label):

  attempts      prismatic: attempt=N in outcome_reason (3 = all retries); compliant: always 1
  contact->seat seconds from the first sustained contact (|Fz| > CONTACT_N for CONTACT_S) to the
                last ACTIVE sample
  dev (base)    seat TCP - CAD-predicted TCP at seat, mm, rotated into the base frame (the EE yaw
                at hover = the base yaw the insert aimed with); the sign pattern across runs is the
                aim bias, the spread is the grasp/read noise
  lat work      integral of |F_lat| over the contact->seat window, N*s: how much the search pushed
  z stalls      dwells (> STALL_S at one height, > 5 mm above the seat) before the seat: the stages
                the part hung on

Usage: python3 ablations/insert_ease.py [--since 2026-10-17] [--last 6] [--part u_brown]
Game plan 2026-10-19: a fix counts when these improve across cycles, not when a run says "success".
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

LOGS = Path(__file__).resolve().parents[1] / 'compliant_insertion_studio/logs'
CONTACT_N, CONTACT_S = 2.0, 0.2
STALL_S = 2.0
TD = R.from_quat([0, 1, 0, 0])           # top-down flange, yaw 0: the inserts' reference


def ee_yaw(q):
    return (TD.inv() * R.from_quat(q)).as_euler('xyz')[2]


def one(meta_path):
    m = json.load(open(meta_path))
    c = Path(m.get('csv_path') or str(meta_path).replace('.meta.json', '.csv'))
    if not c.exists():
        return None
    rows = [r for r in csv.DictReader(open(c)) if r.get('tcp_x') not in ('', 'nan', None)]
    if len(rows) < 5:
        return None
    t = np.array([float(r['t_s']) for r in rows])
    z = np.array([float(r['tcp_z']) for r in rows])
    xy = np.array([[float(r['tcp_x']), float(r['tcp_y'])] for r in rows])
    f = np.array([[float(r['fx']), float(r['fy']), float(r['fz'])] for r in rows])
    act = np.array([r['phase'] == 'ACTIVE' for r in rows])
    if not act.any():
        return None
    ia = np.nonzero(act)[0]
    # first sustained contact inside ACTIVE
    i_c = None
    for i in ia:
        j = np.searchsorted(t, t[i] + CONTACT_S)
        if j > i and (np.abs(f[i:j, 2]) > CONTACT_N).all():
            i_c = i
            break
    i_end = ia[-1]
    dt = np.diff(t, prepend=t[0])
    lat = float(np.sum(np.hypot(f[:, 0], f[:, 1])[i_c:i_end] * dt[i_c:i_end])) if i_c is not None else float('nan')
    # dwells above the seat before the end
    pred = (m.get('cad_prediction') or {}).get('predicted_tcp_at_seat', {}).get('xyz_m') or m.get('predicted_seat_xyz')
    stalls = 0
    if pred is not None and i_c is not None:
        k = i_c
        while k < i_end:
            j = k
            while j < i_end and abs(z[j] - z[k]) < 0.0015:
                j += 1
            if t[j] - t[k] > STALL_S and z[k] - pred[2] > 0.005:
                stalls += 1
            k = j + 1
    dev = None
    if pred is not None:
        d = (xy[i_end] - np.array(pred[:2])) * 1000
        hq = (m.get('hover_pose_world') or {}).get('quat_xyzw') or [float(rows[0][k]) for k in ('tcp_qx', 'tcp_qy', 'tcp_qz', 'tcp_qw')]
        yaw = ee_yaw(hq)
        Rb = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])   # world -> base
        dev = Rb @ d
    reason = m.get('outcome_reason') or ''
    mm = re.search(r'attempt=(\d+)', reason)
    attempts = 3 if 'exhausted' in reason else (int(mm.group(1)) if mm else 1)
    return dict(run=meta_path.name.replace('insert_', '').replace('.meta.json', ''), part=m.get('object'),
                outcome=m.get('outcome'), attempts=attempts,
                contact_to_seat=(t[i_end] - t[i_c]) if i_c is not None else float('nan'),
                dev=dev, lat_work=lat, stalls=stalls, t=meta_path.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default=None, help='YYYY-MM-DD')
    ap.add_argument('--last', type=int, default=0, help='only the last N inserts per part')
    ap.add_argument('--part', default=None)
    a = ap.parse_args()
    t0 = time.mktime(time.strptime(a.since, '%Y-%m-%d')) if a.since else 0
    recs = []
    for p in sorted(LOGS.glob('insert_*.meta.json'), key=lambda p: p.stat().st_mtime):
        if p.stat().st_mtime < t0:
            continue
        try:
            r = one(p)
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
            continue
        if r and (not a.part or r['part'] == a.part):
            recs.append(r)
    by = {}
    for r in recs:
        by.setdefault(r['part'], []).append(r)
    print(f"{'run':34s} {'out':7s} {'att':>3s} {'c->seat s':>9s} {'dev base x,y mm':>16s} {'lat N*s':>8s} {'stalls':>6s}")
    for part, rs in by.items():
        rs = rs[-a.last:] if a.last else rs
        for r in rs:
            dv = f"{r['dev'][0]:+5.1f},{r['dev'][1]:+5.1f}" if r['dev'] is not None else '   -'
            print(f"{r['run']:34s} {str(r['outcome'])[:7]:7s} {r['attempts']:3d} {r['contact_to_seat']:9.1f} {dv:>16s} {r['lat_work']:8.1f} {r['stalls']:6d}")
        ok = [r for r in rs if r['outcome'] == 'success']
        a1 = sum(1 for r in ok if r['attempts'] == 1)
        devs = np.array([r['dev'] for r in ok if r['dev'] is not None])
        print(f"  == {part}: {len(ok)}/{len(rs)} seated, attempt-1 {a1}/{len(rs)}; contact->seat median "
              f"{np.nanmedian([r['contact_to_seat'] for r in ok]) if ok else float('nan'):.1f} s; "
              f"dev median ({np.median(devs[:, 0]) if len(devs) else float('nan'):+.1f}, {np.median(devs[:, 1]) if len(devs) else float('nan'):+.1f}) mm, "
              f"spread (p10-p90) {np.percentile(devs[:, 0], 90) - np.percentile(devs[:, 0], 10) if len(devs) else float('nan'):.1f}/"
              f"{np.percentile(devs[:, 1], 90) - np.percentile(devs[:, 1], 10) if len(devs) else float('nan'):.1f} mm; "
              f"lat work median {np.nanmedian([r['lat_work'] for r in ok]) if ok else float('nan'):.1f} N*s\n")


if __name__ == '__main__':
    main()
