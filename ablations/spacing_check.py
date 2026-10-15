#!/usr/bin/env python3
"""Are the table parts spaced so an open gripper can reach each one? (2026-10-15)

    python3 -u ablations/spacing_check.py          # arm at home: every table part in view

Each part's CAD outline (aruco-grasp-annotator mesh) is placed at its camera-observed pose and
projected on the table; base1 is its 161 mm square at the observed base pose. Every table part
must stand MIN_GAP_MM clear of the base and of every other part: an RG2 finger at the pre-grasp
width reaches ~18-21 mm past a part's side, so a part closer than that to anything gets a finger
on it. 2026-10-15: line_green fell out of the jaws during a pull-out and lay 1 cm from base1;
nothing checked, and the next pick put an open finger on the base (protective stop).

Run by replay_real_assembly at startup (before any pick) and by disassemble.py after the last
part is set down. Fails closed: a violation stops the run, naming the pair and the gap.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))

MIN_GAP_MM = 25.0
MESH = Path.home() / 'Documents/aruco-grasp-annotator/data/models/{}.obj'
BASE_SIDE_M = {'base1': 0.161}


def _mesh_xyz(name):
    V = [list(map(float, l.split()[1:4])) for l in open(str(MESH).format(name)) if l.startswith('v ')]
    return np.array(V) / 100.0                                       # cm -> m


def outline(name, pose):
    """Convex xy outline (K, 2) of `name` at pose ((x, y, z), quat) — the part's table footprint."""
    (p, q) = pose
    P = R.from_quat(q).apply(_mesh_xyz(name))[:, :2] + np.asarray(p)[:2]
    return P[ConvexHull(P).vertices]


def base_outline(pose, base='base1'):
    (p, q) = pose
    h = BASE_SIDE_M[base] / 2
    sq = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]])
    return R.from_quat(q).apply(sq)[:, :2] + np.asarray(p)[:2]


def _densify(poly, step=0.001):
    pts = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(2, int(np.linalg.norm(b - a) / step))
        pts.append(a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a))
    return np.vstack(pts)


def _inside(pts, poly):
    hull = ConvexHull(poly)
    return np.all(pts @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9, axis=1)


def gap_mm(a, b):
    """Smallest distance between two convex outlines (0 when they overlap)."""
    if _inside(a, b).any() or _inside(b, a).any():
        return 0.0
    A, B = _densify(a), _densify(b)
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min()
    return float(1000 * d)


def check(parts, base_pose, base='base1', min_gap_mm=MIN_GAP_MM):
    """parts: {name: pose}. Parts whose centre is inside the base square are seated: they are part
    of the base obstacle, not table parts. Returns (ok, report)."""
    B = base_outline(base_pose, base)
    table = {n: outline(n, p) for n, p in parts.items()
             if not _inside(np.asarray(p[0])[None, :2], B)[0]}
    gaps = []
    names = sorted(table)
    for i, a in enumerate(names):
        gaps.append((a, base, gap_mm(table[a], B)))
        for b in names[i + 1:]:
            gaps.append((a, b, gap_mm(table[a], table[b])))
    bad = [(a, b, round(g, 1)) for a, b, g in gaps if g < min_gap_mm]
    rep = {'ok': not bad, 'min_gap_mm': min_gap_mm, 'table_parts': names,
           'gaps_mm': {f"{a}|{b}": round(g, 1) for a, b, g in gaps}, 'too_close': bad}
    return not bad, rep


def observe_parts(names, timeout_s=8.0):
    """{name: ((x, y, z), quat)} for the parts the camera sees now (median of samples)."""
    from replay_real_assembly import observe_object_orientation
    out = {}
    for n in names:
        obs = observe_object_orientation(n, timeout_s, n_samples=5)
        S = obs.get('samples') or []
        if obs.get('error') or not S:
            continue
        P = np.median(np.array([s['xyz'] for s in S]), axis=0)
        Q = np.array([s['quat_xyzw'] for s in S]); Q *= np.sign(Q @ Q[0])[:, None]
        q = Q.mean(axis=0); q /= np.linalg.norm(q)
        out[n] = (tuple(P), tuple(q))
    return out


PARTS = ('u_brown', 'u_orange', 'line_green', 'inverted_u_yellow')


def main():
    from replay_real_assembly import observe_base_pose
    ob = observe_base_pose('base1')
    if ob is None:
        print(json.dumps({'ok': False, 'error': 'base1 not observed'})); return 1
    parts = observe_parts(PARTS)
    ok, rep = check(parts, ob)
    rep['unseen'] = [n for n in PARTS if n not in parts]
    print(json.dumps(rep))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
