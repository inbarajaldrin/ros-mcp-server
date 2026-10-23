#!/usr/bin/env python3
"""Where the gap between a seated part's legs really is, from depth against CAD (2026-10-21).

    python3 -u ablations/notch_depth.py --base base1 --parts u_brown u_orange      # from where the arm is

line_green must drop between the legs of the seated U parts. At FMB1-parked, depth put u_brown's gap at x -9..+15 mm
(centre +3) where CAD has -13..+13 (centre 0); the marker 'follow' read called it +0.6 mm, so the bar aimed at x 0 and
landed on the leg (contact at the U tops, 39.5 mm above the seat, every attempt). Operator: use CAD and depth.

For each part: a band +-BAND_MM around its CAD seat y (base frame), the height profile across x from one still depth
capture (90th percentile per 2 mm bin, against the base top), the gap = the low run (below base top + GAP_BELOW_MM)
around the CAD gap centre; the same profile from the CAD mesh gives the CAD gap. Offset = measured centre - CAD centre
(base frame x, across the part's long axis). Returns {part: {'dx_m', 'width_mm', 'cad_width_mm'}} or an error.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'ablations')]
BAND_MM, BIN_MM, SPAN_MM, GAP_BELOW_MM = 6.0, 1.0, 40.0, 10.0
MIN_BINS_SEEN, WIDTH_TOL_MM = 60, 3.0


def _gap(xs, hs, around=0.0):
    """(lo, hi) x of the contiguous run with heights below the threshold that contains / is nearest `around`."""
    low = [x for x, h in zip(xs, hs) if h is not None and h < GAP_BELOW_MM]
    if not low:
        return None
    low = np.array(sorted(low)); runs = np.split(low, np.nonzero(np.diff(low) > 1.5 * BIN_MM)[0] + 1)
    best = min(runs, key=lambda r: 0 if r[0] <= around <= r[-1] else min(abs(r[0] - around), abs(r[-1] - around)))
    return float(best[0]), float(best[-1])


def measure(base, base_pose, parts, Pw=None):
    import seat_check as sc
    from compliant_insertion_studio.wrapper.cad_geometry import _load_base_top_z_local
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    from grasp_depth import load_mesh, surface_points
    if Pw is None:
        cap = sc.capture_world(n=10)
        if cap is None:
            return {'error': 'no still depth capture'}
        Pw = cap[0]
    (bx, by, bz), bq = base_pose
    Rb = R.from_quat(bq); Pb = Rb.inv().apply(Pw - np.array([bx, by, bz])); bt = _load_base_top_z_local(base)
    xs = np.arange(-SPAN_MM, SPAN_MM + 1e-9, BIN_MM)
    out = {}
    for p in parts:
        sp, sq = load_seat_pose_in_base(base, p); yc = 1000 * sp[1]
        V, F = load_mesh(p); W = surface_points(V, F, 0.001) @ R.from_quat(sq).as_matrix().T + sp
        cb = W[np.abs(1000 * W[:, 1] - yc) < BAND_MM]
        hc = [None if (s := cb[np.abs(1000 * cb[:, 0] - x) < BIN_MM / 2]).shape[0] < 1 else 1000 * (s[:, 2].max() - bt) for x in xs]
        # CAD: no part material over a bin = the channel floor (below the base top)
        hc = [h if h is not None else -5.0 for h in hc]
        band = Pb[np.abs(1000 * Pb[:, 1] - yc) < BAND_MM]
        hm = [None if (s := band[np.abs(1000 * band[:, 0] - x) < BIN_MM / 2]).shape[0] < 3 else
              float(np.percentile(1000 * (s[:, 2] - bt), 90)) for x in xs]
        gc = _gap(xs, hc); gm = _gap(xs, hm, around=0.0 if gc is None else (gc[0] + gc[1]) / 2)
        if gc is None or gm is None:
            out[p] = {'error': 'gap not found' + (' in CAD' if gc is None else ' in depth')}; continue
        seen = sum(h is not None for h in hm)
        wm, wc = gm[1] - gm[0] + BIN_MM, gc[1] - gc[0] + BIN_MM
        if seen < MIN_BINS_SEEN or abs(wm - wc) > WIDTH_TOL_MM:
            # occluded or misfound (u_orange behind a held bar: 17 bins, a 3 mm 'gap'): not a measurement
            out[p] = {'error': f'gap not measured (width {wm:.0f} vs CAD {wc:.0f} mm, {seen} bins seen)'}; continue
        dx = ((gm[0] + gm[1]) - (gc[0] + gc[1])) / 2
        out[p] = {'dx_m': round(dx / 1000, 4), 'gap_mm': [round(gm[0], 1), round(gm[1], 1)],
                  'cad_gap_mm': [round(gc[0], 1), round(gc[1], 1)], 'width_mm': round(gm[1] - gm[0] + BIN_MM, 1),
                  'cad_width_mm': round(gc[1] - gc[0] + BIN_MM, 1), 'bins_seen': seen}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--parts', nargs='+', required=True)
    a = ap.parse_args()
    lb = json.load(open(REPO / 'ablations/logs/last_base_pose.json'))[a.base]
    print(json.dumps(measure(a.base, (tuple(lb[0]), tuple(lb[1])), a.parts), indent=1))


if __name__ == '__main__':
    main()
