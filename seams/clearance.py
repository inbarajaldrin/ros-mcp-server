"""A seated part's xy clearance from CAD (AR3 sigma gate, decision 2026-10-22: drive-mate moves on the vision seam's
held-part / target read "when the target's sigma exceeds the clearance").

The narrowest horizontal gap between the part, in its seated pose, and what surrounds it in the band it is inserted
through - the base's pocket walls and the parts seated before it - from the meshes' surface points (base frame, above
the part's lowest 0.8 mm so the floor it rests on is not a wall). 10-23, FMB1 on base1: u_brown 1.30 mm, u_orange 0.50,
line_green 1.05 (measured on the arm: +-1.1 mm across the U openings), inverted_u_yellow 0.07 (it straddles line_green).
Cached per (base, part, parts seated before) in ablations/logs/cad_clearance.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / 'ablations' / 'logs' / 'cad_clearance.json'
FLOOR_BAND_M = 0.0008
STEP_M = 0.0008


def _pts(name, base, step=STEP_M):
    sys.path[:0] = [str(REPO), str(REPO / 'ablations')]
    from scipy.spatial.transform import Rotation as R
    from grasp_depth import load_mesh, surface_points
    V, F = load_mesh(name)
    P = surface_points(V, F, step)
    if name == base:
        return P
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    p, q = load_seat_pose_in_base(base, name)
    return R.from_quat(q).apply(P) + np.asarray(p)


def xy_clearance_m(base, part, seated_before):
    """The narrowest xy gap (m) of `part` seated on `base` to the base and the parts in `seated_before`."""
    key = f"{base}|{part}|{','.join(sorted(seated_before))}"
    try:
        c = json.load(open(CACHE))
    except Exception:
        c = {}
    if key in c:
        return float(c[key])
    from scipy.spatial import cKDTree
    P = _pts(part, base)
    O = np.vstack([_pts(base, base)] + [_pts(o, base) for o in seated_before])
    lo, hi = P[:, 2].min() + FLOOR_BAND_M, min(P[:, 2].max(), O[:, 2].max())
    Pb, Ob = P[(P[:, 2] > lo) & (P[:, 2] < hi)], O[(O[:, 2] > lo) & (O[:, 2] < hi)]
    if not len(Pb) or not len(Ob):
        val = float('inf')                       # nothing around it in its band: no lateral constraint
    else:
        d, _ = cKDTree(Ob[:, :2]).query(Pb[:, :2])
        val = float(d.min())
    c[key] = val
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(c, open(CACHE, 'w'), indent=1)
    except Exception:
        pass
    return val
