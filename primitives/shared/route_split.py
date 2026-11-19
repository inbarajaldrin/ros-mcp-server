"""Split a refused joint-linear move into short joint-linear segments (lead approval 2026-11-19).

rotate_object sends one joint target and the controller interpolates a straight line in joint space.  In MuJoCo
r48e that line swept the v3.1 camera mount through the forearm / wrist_1 zone for a flat u_orange held camera-up.
Moving the joint groups one after another (arm 0-2, wrist_1, wrist_2, wrist_3) reaches the same end pose along a
different path.  Every segment starts from where the previous one ended, so nothing is solved from a fixed seed,
and each segment is checked by the caller's full rule set.
"""
from __future__ import annotations

import itertools

import numpy as np

GROUPS = {"arm": (0, 1, 2), "wrist_1": (3,), "wrist_2": (4,), "wrist_3": (5,)}


def candidate_routes(start, target):
    """Waypoint lists start -> ... -> target: the direct line first, then every ordered split of the joint
    groups that move, fewest segments first."""
    start, target = np.asarray(start, float), np.asarray(target, float)
    moving = [g for g, idx in GROUPS.items() if np.max(np.abs(target[list(idx)] - start[list(idx)])) > 1e-6]
    yield [start, target]
    for n_parts in range(2, len(moving) + 1):
        seen = set()
        for order in itertools.permutations(moving):
            # partition `order` into n_parts consecutive blocks
            for cuts in itertools.combinations(range(1, len(order)), n_parts - 1):
                blocks = [order[a:b] for a, b in zip((0,) + cuts, cuts + (len(order),))]
                key = tuple(tuple(sorted(b)) for b in blocks)
                if key in seen:
                    continue
                seen.add(key)
                q, wps = start.copy(), [start.copy()]
                for block in blocks:
                    q = q.copy()
                    for g in block:
                        q[list(GROUPS[g])] = target[list(GROUPS[g])]
                    wps.append(q)
                yield wps


def admissible_route(start, target, segment_refusal):
    """The admissible route of least total joint travel (ties: fewer segments), or (None, reasons).

    segment_refusal(a, b, is_last) -> None or a typed reason."""
    best, reasons = None, []
    for wps in candidate_routes(start, target):
        segs = list(zip(wps[:-1], wps[1:]))
        why = None
        for i, (a, b) in enumerate(segs):
            why = segment_refusal(a, b, i == len(segs) - 1)
            if why:
                break
        if why:
            reasons.append(why)
            continue
        travel = float(sum(np.sum(np.abs(b - a)) for a, b in segs))
        key = (round(travel, 6), len(segs))
        if best is None or key < best[0]:
            best = (key, wps)
        if len(segs) == 1:
            break                                   # the direct line is admissible: nothing beats it
    return (best[1], None) if best else (None, reasons)
