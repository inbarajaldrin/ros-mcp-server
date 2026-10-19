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
CLUSTER_MM = 10.0            # pose samples within this belong to the same reading
AMBIG_MM = 25.0              # two readings further apart than this: ask depth which one is real
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
        obs = observe_object_orientation(n, timeout_s, n_samples=12)
        S = obs.get('samples') or []
        if obs.get('error') or not S:
            continue
        # The localizer can alternate between two poses for one part (a half-hidden second marker:
        # yellow read at (293, -503) and (202, -517) in the same second, 2026-10-17); a median of
        # the mix put it on base1 — a false 'too close'. Keep the largest cluster of samples within
        # CLUSTER_MM of each other.
        from primitives.shared.config import TABLE_HEIGHT
        S = [x for x in S if x['xyz'][2] >= TABLE_HEIGHT - 0.010]   # below the table = a misread (move_to_grasp)
        if not S:
            continue
        XY = np.array([s['xyz'][:2] for s in S])
        groups, left = [], set(range(len(XY)))
        while left:                                     # greedy clusters of mutually-near samples
            i = min(left)
            g = [j for j in left if np.linalg.norm(XY[j] - XY[i]) < CLUSTER_MM / 1000]
            groups.append(g); left -= set(g)
        cands = []
        for g in sorted(groups, key=len, reverse=True):
            if len(g) < 2:
                continue
            P = np.median(np.array([S[j]['xyz'] for j in g]), axis=0)
            Q = np.array([S[j]['quat_xyzw'] for j in g]); Q *= np.sign(Q @ Q[0])[:, None]
            q = Q.mean(axis=0); q /= np.linalg.norm(q)
            cands.append((tuple(P), tuple(q)))
        if not cands:
            continue
        if len(cands) > 1 and np.linalg.norm(np.subtract(cands[0][0][:2], cands[1][0][:2])) > AMBIG_MM / 1000:
            cands = [_depth_pick(n, cands)]             # readings disagree: let depth say which is real
        out[n] = cands[0]
    return out


_DEPTH = {}


def _depth_pick(name, cands):
    """Of several candidate poses (marker readings that disagree), the one whose CAD outline has
    the most depth points standing above the table — a misread outline lands on bare table or on
    base1, not on the part. Depth is captured once per process from the current (home) view."""
    try:
        if 'Pw' not in _DEPTH:
            from seat_check import capture_world
            cap = capture_world(n=5)
            _DEPTH['Pw'] = cap[0] if cap else None
        Pw = _DEPTH['Pw']
        if Pw is None:
            return cands[0]
        table = np.percentile(Pw[:, 2], 30)
        high = Pw[(Pw[:, 2] > table + 0.008) & (Pw[:, 2] < table + 0.12)][:, :2]
        scores = []
        for c in cands:
            poly = outline(name, c)
            inside = _inside(high, poly).mean() if len(high) else 0.0
            area = ConvexHull(poly).volume
            scores.append(_inside(high, poly).sum() / max(area, 1e-6))
        k = int(np.argmax(scores))
        print(f"[spacing] {name}: marker readings disagree {[tuple(np.round(1000*np.array(c[0][:2]))) for c in cands]} "
              f"-> depth picks {tuple(np.round(1000*np.array(cands[k][0][:2])))}", flush=True)
        return cands[k]
    except Exception as e:
        print(f"[spacing] {name}: depth disambiguation failed ({e}) — using the largest cluster")
        return cands[0]


PARTS = ('u_brown', 'u_orange', 'line_green', 'inverted_u_yellow')
GRASPS = Path.home() / 'Documents/aruco-grasp-annotator/data/grasp_points/{}_grasp_points.json'
FINGER_T_M, FINGER_W_M = 0.0134, 0.0242          # RG2 finger block: thickness (closing axis) x width
FINGER_GAP_MIN_MM = 5.0


def finger_outlines(name, pose, gid, open_mm):
    """The two open-finger blocks' xy outlines for grasp `gid` of `name` at `pose`: centred
    open/2 + T/2 either side of the grasp point along the part's closing axis (object x)."""
    g = next(g for g in json.load(open(str(GRASPS).format(name)))['grasp_points'] if g['id'] == gid)
    (p, q) = pose
    Rq = R.from_quat(q)
    c = Rq.apply([g['position'][k] for k in 'xyz'])[:2] + np.asarray(p)[:2]
    ax = Rq.apply([1.0, 0.0, 0.0])[:2]
    if np.linalg.norm(ax) < 0.5:          # object x near vertical (bar on its side): the fingers
        ax = Rq.apply([0.0, 0.0, 1.0])[:2]  # close across the horizontal cross-section axis instead
    ax /= np.linalg.norm(ax)
    pe = np.array([-ax[1], ax[0]])
    out = []
    for s in (-1, 1):
        m = c + s * ax * (open_mm / 2000 + FINGER_T_M / 2)
        out.append(np.array([m + u * ax * FINGER_T_M / 2 + v * pe * FINGER_W_M / 2
                             for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]))
    return out


def grasp_width_mm(name, gid):
    """Opening grasp `gid` needs, closing along the part's x axis — move_to_grasp turns the wrist to
    the part's own yaw, so the fingers close along object x (2026-10-19, every logged pick). From the
    annotator's grasp_validity; None if the grasp is not valid along x. u_brown: 1 and 3 (legs) 35 mm,
    2 (the 70 mm bridge) 84 mm."""
    g = next(g for g in json.load(open(str(GRASPS).format(name)))['grasp_points'] if g['id'] == gid)
    return (g.get('grasp_validity') or {}).get('x_axis_gripper_width_mm')


def clear_grasp(name, pose, base_pose, others, open_mm=None, base='base1'):
    """The grasp id of `name` whose open fingers stand farthest clear of the base and the other
    parts' outlines (>= FINGER_GAP_MIN_MM), each at the opening THAT grasp needs (grasp_width_mm);
    (gid, report). gid None if none is clear. 2026-10-19: every grasp was scored at the first grasp's
    35 mm, so u_brown's bridge grasp (needs 84 mm) looked clear and the 35 mm-open fingers came down
    on the part (protective stop). open_mm: fallback for grasps without validity data."""
    obst = [('base1', base_outline(base_pose, base))] + [(n, outline(n, p)) for n, p in others.items() if n != name]
    rep, best = {}, None
    ids = sorted(g['id'] for g in json.load(open(str(GRASPS).format(name)))['grasp_points'])
    for gid in ids:
        w = grasp_width_mm(name, gid) or open_mm
        if w is None:
            rep[gid] = 'not valid along x'; continue
        g = min(gap_mm(f, o) for f in finger_outlines(name, pose, gid, w) for _, o in obst)
        rep[gid] = round(g, 1)
        if g >= FINGER_GAP_MIN_MM and (best is None or g > rep[best]):
            best = gid
    return best, rep


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
