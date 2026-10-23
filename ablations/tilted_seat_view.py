#!/usr/bin/env python3
"""Seat height of a part from a CLOSE, TILTED depth view (2026-10-20).

    python3 -u ablations/tilted_seat_view.py --object line_red --base base3 [--tilt 25] [--dist 0.47] [--from -x]

Operator: the lights are overhead, so a camera looking straight down at a glossy top (line_red's red
paint) sees the light's reflection and the D415 depth goes wrong there (a seated line_red read 3 mm high at
one end from 0.56 m straight above; matte line_brown read flat from the same view). Tilting the camera
by 20-30 deg takes it out of the mirror direction; closer (the D415 min range is ~0.45 m) gives more points.

The camera is aimed at the part's CAD top-face centre from --dist, tilted --tilt deg, from the side --from
(base frame: the camera sits on that side and looks across). The bar top is measured against the base top
plane fitted in the same cloud, along the part's long axis. Prints the profile; returns to safe height.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))


def aimed_flange(tgt, side_dir_world, tilt_deg, dist, X_CAM, yaw):
    """Flange pose with the camera's optical axis through tgt, the camera displaced toward side_dir_world."""
    best = None
    for ax in ('x', 'y'):
        for sg in (1, -1):
            Rf = (R.from_euler('z', yaw) * R.from_quat([0, 1, 0, 0]) * R.from_euler(ax, sg * np.radians(tilt_deg))).as_matrix()
            Rc = Rf @ X_CAM[:3, :3]
            pc = tgt - dist * Rc[:, 2]
            horiz = (pc - tgt)[:2]
            score = float(np.dot(horiz / (np.linalg.norm(horiz) + 1e-9), side_dir_world[:2]))
            pf = pc - Rf @ X_CAM[:3, 3]
            if best is None or score > best[0]:
                best = (score, pf, R.from_matrix(Rf).as_quat())
    return best[1], best[2], best[0]


def profile(obj, base, ob, Pw):
    from seat_check import TOP_FACE
    f = TOP_FACE[(obj, base)]
    (bx, by, bz), bq = ob
    Pb = (Pw - np.array([bx, by, bz])) @ R.from_quat(bq).as_matrix()
    m = (np.abs(Pb[:, 0]) < 0.078) & (np.abs(Pb[:, 1]) < 0.078) & (np.abs(Pb[:, 2] - 0.0175) < 0.004)
    m &= np.ones(len(Pb), bool)
    B = Pb[m]
    A = np.c_[B[:, 0], B[:, 1], np.ones(len(B))]
    coef, *_ = np.linalg.lstsq(A, B[:, 2], rcond=None)
    long_ax = 1 if f['long_axis'] == 'y' else 0
    rows = []
    for s in np.arange(-60, 61, 10):
        c = np.array([f['cx'], f['cy']]); c[long_ax] += s / 1000.0
        sel = (np.abs(Pb[:, long_ax] - c[long_ax]) < 0.004) & (np.abs(Pb[:, 1 - long_ax] - c[1 - long_ax]) < 0.007) \
            & (Pb[:, 2] > f['z'] - 0.012) & (Pb[:, 2] < f['z'] + 0.015)
        if sel.sum() < 15:
            rows.append((int(s), None, int(sel.sum()))); continue
        z = np.percentile(Pb[sel, 2], 60)
        base_here = coef[0] * c[0] + coef[1] * c[1] + coef[2]
        rows.append((int(s), round(1000 * (z - base_here) - 1000 * (f['z'] - 0.0175), 1), int(sel.sum())))
    return rows, coef


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--object', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--tilt', type=float, default=25.0)
    ap.add_argument('--dist', type=float, default=0.47)
    ap.add_argument('--from', dest='side', default='-x', choices=['+x', '-x', '+y', '-y'])
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    from seat_check import X_CAM, TOP_FACE, capture_world
    from replay_real_assembly import observe_base_pose
    from primitives.shared.config import SAFE_HEIGHT
    ob = observe_base_pose(a.base)
    (bx, by, bz), bq = ob
    Rb = R.from_quat(bq)
    f = TOP_FACE[(a.object, a.base)]
    tgt = np.array([bx, by, bz]) + Rb.apply([f['cx'], f['cy'], f['z']])
    d = np.zeros(3); d['xy'.index(a.side[1])] = 1.0 if a.side[0] == '+' else -1.0
    side_w = Rb.apply(d)
    yaw = float(Rb.as_euler('xyz')[2])
    pf, qf, sc = aimed_flange(tgt, side_w, a.tilt, a.dist, X_CAM, yaw)
    print(f"[tilted] camera from {a.side} at {a.tilt:g} deg, {a.dist:g} m: flange {np.round(pf, 4).tolist()} (side match {sc:.2f})", flush=True)
    if a.dry_run:
        return 0
    def go(argv, label):
        r = subprocess.run(['python3', '-u', '-m', *argv], cwd=str(REPO), capture_output=True, text=True, timeout=90)
        ok = r.returncode == 0
        if not ok:
            print(f"!!! {label}: {(r.stdout + r.stderr).strip().splitlines()[-1][:200]}")
        return ok
    if not go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *[f"{v:.4f}" for v in pf],
               '--quat', *[f"{v:.6f}" for v in qf], '--min-z', '0.15'], 'tilted view'):
        return 1
    time.sleep(1.5)
    cap = capture_world(n=15)
    ob2 = observe_base_pose(a.base, timeout_s=6.0) or ob          # the base read from this view too
    rows, coef = profile(a.object, a.base, ob, cap[0]) if cap else ([], None)
    print(f"[tilted] {a.object}: bar top - CAD (mm, vs the base top plane) along the bar: "
          + ' '.join(f"{s:+d}:{'-' if v is None else f'{v:+.1f}'}" for s, v, n in rows), flush=True)
    print(json.dumps({'object': a.object, 'profile': rows,
                      'base_marker_here_mm': [round(1000 * ob2[0][0], 1), round(1000 * ob2[0][1], 1)]}), flush=True)
    go(['primitives.move_cartesian', '--mode', 'real', '--xyz', f"{pf[0]:.4f}", f"{pf[1]:.4f}", f"{SAFE_HEIGHT + 0.1:.4f}",
        '--quat', '0', '1', '0', '0'], 'back up, top-down')
    return 0


if __name__ == '__main__':
    sys.exit(main())
