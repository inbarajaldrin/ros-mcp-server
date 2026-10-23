#!/usr/bin/env python3
"""Smart taps (2026-10-22, operator: "use force to smartly tap - you know where it is seating and the force
feedback"): the per-seat recovery's taps as MEASUREMENTS against the seated CAD, read through their force record.

    tp = Tapper.for_part(obj, base_pose, base)      # top-face points from the seated CAD (seat_check.TOP_FACE)
    m = tp.map()                                    # tap ends, middle and both edges: height error of each vs seated
    r = tp.press(pt)                                # press one point <= 5 N, logging height against force:
                                                    #   seated (reached the band) | moving (went down, still high) |
                                                    #   blocked (the force rose, the height did not move)
    ok = tap_recover(obj, base_pose, base, say)     # map -> press the highest point -> re-tap -> ... -> verdict

Before: a tap went close -> safe height -> over the point -> a 10 s guarded descent -> one height -> safe height
(~20 s), the result was a camera re-check (20-30 s), and a press that did not move the part was repeated
unchanged (10-22 logs: line_green's -x edge +4.3 mm, pressed at 5 N -> +4.2, pressed again -> +4.2). Now the tip
stays a few mm over the top between taps (one ROS node, ~2.5 s a tap), the seated CAD says what every height
should be, the force record says whether a press moved the part, and a blocked point is not pressed again.
The seat verdict is seams.acceptance's band (one read-sigma in), the same rule ExecuteMate certifies with.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

TAP_N = 3.0                  # a measuring tap: contact, not a push (3 N did not move a wedged bar, 10-20)
PRESS_N = 4.5                # the guard trips past it; the peak stays under the operator's 5 N (bench: 5.0 -> 5.12 N)
OVER_M = 0.003               # the tip waits this far over the highest height seen
SPEED_M_S = 0.004            # descent speed of taps and presses
PRESS_BELOW_M = 0.0015       # a press may go this far under the seated top (a cocked part rolls down past it)
MOVED_MIN_M = 0.0005         # a press that travelled this far after first contact moved the part
CONTACT_N = 1.0              # first contact in the force record
EDGE_IN_M = 0.004            # edge taps this far in from the CAD top-face edge
END_IN_M = 0.008             # end taps this far in from the CAD top-face ends
MAX_PRESSES = 6
Q_DOWN = (0.0, 1.0, 0.0, 0.0)


@dataclass
class Point:
    name: str
    xyz_top: np.ndarray        # the seated CAD top face here, world (m)
    err_m: Optional[float] = None
    blocked: int = 0


@dataclass
class Tapper:
    points: List[Point]
    tip_m: float               # flange z - tip z, fingers closed
    band_m: float              # |err| within this = seated here
    log: list = field(default_factory=list)
    press_below_m: float = PRESS_BELOW_M
    min_z: float = 0.15

    @staticmethod
    def for_part(obj, base_pose, base='base1', band_m=None):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ablations'))
        from seat_check import TOP_FACE
        from primitives.shared.config import RG2_TIP_CLOSED_M
        from seams.acceptance import lim_z_m, margin_m
        f = TOP_FACE[(obj, base)]
        (bx, by, bz), bq = base_pose
        Rb = R.from_quat(bq)

        def w(dx, dy):
            return np.array([bx, by, bz]) + Rb.apply([f['cx'] + dx, f['cy'] + dy, f['z']])
        long_y = f.get('long_axis', 'y') == 'y'
        hl, hc = float(f['half_long_m']), float(f['half_cross_m'])
        al = max(hl - END_IN_M, 0.0)
        ac = max(hc - EDGE_IN_M, 0.0)
        pts = [('mid', 0.0, 0.0)]
        if hl >= 2 * hc:                                    # a bar: both ends; else the corners' midpoints
            pts += [('end-', -al, 0.0), ('end+', al, 0.0)]
        pts += [('edge-', 0.0, -ac), ('edge+', 0.0, ac)]
        # base-frame (dx, dy) of each point: along the long axis by a, across it by c
        P = [Point(n, w(c, a) if long_y else w(a, c)) for n, a, c in pts]
        if band_m is None:
            lim = lim_z_m(base)
            band_m = (lim - margin_m()) if lim is not None else 0.0015
        return Tapper(P, float(RG2_TIP_CLOSED_M), float(band_m))

    # -- motion (one node, the touch insert's) --------------------------------------------------------------------
    def _move(self, xyz, force=None, duration=None, on_sample=None):
        from compliant_insertion_studio.wrapper.touch_insert import _node
        from primitives.move_cartesian import plan_move
        return plan_move(_node(), list(map(float, xyz)), ('quat', list(Q_DOWN)), max_force=force, min_z=self.min_z,
                         duration=duration, on_sample=on_sample)

    def _z(self):
        from compliant_insertion_studio.wrapper.touch_insert import tcp_now
        return float(tcp_now()[0][2])

    def _over(self, p: Point):
        """Over p, OVER_M above the highest height seen (or its seated height + 10 mm before any)."""
        hi = max([q.err_m for q in self.points if q.err_m is not None], default=0.010)
        return np.array([p.xyz_top[0], p.xyz_top[1], p.xyz_top[2] + self.tip_m + max(hi, 0.0) + OVER_M])

    def go_over(self, p: Point):
        tgt = self._over(p)
        z = self._z()
        if z < tgt[2] - 0.0005:                              # straight up first, never sideways low
            from compliant_insertion_studio.wrapper.touch_insert import tcp_now
            c = tcp_now()[0]
            self._move([c[0], c[1], tgt[2]], duration=max(0.6, (tgt[2] - z) / 0.005))
        c = self._z()
        return self._move(tgt, force=TAP_N, duration=None if abs(c - tgt[2]) > 0.005 else 0.8)

    # -- measurements ---------------------------------------------------------------------------------------------
    def _descend(self, p: Point, force):
        """Guarded descent at p to PRESS_BELOW_M under its seated top. Returns (err_m at stop, force record)."""
        rec = []
        z_seat = p.xyz_top[2] + self.tip_m
        start = self._z()
        tgt = np.array([p.xyz_top[0], p.xyz_top[1], z_seat - self.press_below_m])
        dur = max(0.6, (start - tgt[2]) / SPEED_M_S)

        def on_sample(t, dF, F, q):
            rec.append((t, dF))
        r = self._move(tgt, force=force, duration=dur, on_sample=on_sample)
        z = self._z()
        # the descent's height at each sample from the planned constant-speed profile is not exact (s-curve); the record
        # keeps (t, dF) and the stop height - first contact time vs stop time says how far it went under load
        return z - z_seat, rec, r, start

    def tap(self, p: Point):
        self.go_over(p)
        err, rec, r, _ = self._descend(p, TAP_N)
        p.err_m = float(err)
        self._move(self._over(p), duration=0.8)              # straight back up (unguarded: lifting off a contact)
        self.log.append(dict(kind='tap', point=p.name, err_mm=round(1000 * err, 2), contact=bool(r.get('contact'))))
        return err

    def map(self):
        return {p.name: round(1000 * self.tap(p), 2) for p in self.points}

    def press(self, p: Point):
        """Press p at <= PRESS_N; the verdict from where it stopped and how the force built."""
        self.go_over(p)
        err0 = p.err_m
        err, rec, r, start = self._descend(p, PRESS_N)
        t_c = next((t for t, dF in rec if dF > CONTACT_N), None)
        t_end = rec[-1][0] if rec else None
        # travel under load: contact-to-stop time x the commanded speed (constant-speed descent)
        loaded_m = SPEED_M_S * (t_end - t_c) if (t_c is not None and t_end is not None) else 0.0
        moved = (err0 is not None and err0 - err > MOVED_MIN_M) or loaded_m > 2 * MOVED_MIN_M
        verdict = 'seated' if abs(err) <= self.band_m else ('moving' if moved else 'blocked')
        p.err_m = float(err)
        if verdict == 'blocked':
            p.blocked += 1
        self._move(self._over(p), duration=0.8)
        self.log.append(dict(kind='press', point=p.name, err_before_mm=None if err0 is None else round(1000 * err0, 2),
                             err_mm=round(1000 * err, 2), loaded_mm=round(1000 * loaded_m, 2), verdict=verdict,
                             peak_N=round(max([dF for _, dF in rec], default=0.0), 2)))
        return verdict

    def seated(self):
        return all(p.err_m is not None and abs(p.err_m) <= self.band_m for p in self.points)


def tap_recover(obj, base_pose, base, say=print, tapper: Optional[Tapper] = None):
    """Map the top, press the highest point not yet blocked twice, re-tap it, repeat; True when every point is inside
    the band. False with the map when presses no longer move it (the caller's other recoveries: nudge, reseat)."""
    tp = tapper or Tapper.for_part(obj, base_pose, base)
    t0 = time.time()
    m = tp.map()
    say(f"  [taps] {obj}: map {m} mm (band +-{1000 * tp.band_m:.1f})")
    for k in range(MAX_PRESSES):
        if tp.seated():
            say(f"  [taps] {obj}: seated by the taps ({time.time() - t0:.0f} s): {tp.log[-1] if tp.log else ''}")
            return True, tp
        cand = [p for p in tp.points if p.err_m is not None and p.err_m > tp.band_m and p.blocked < 2]
        if not cand:
            say(f"  [taps] {obj}: every high point is blocked - not pressing again ({[(p.name, round(1000 * p.err_m, 1)) for p in tp.points]})")
            return False, tp
        p = max(cand, key=lambda q: q.err_m)
        v = tp.press(p)
        say(f"  [taps] {obj}: press {p.name} -> {tp.log[-1]}")
        if v != 'blocked':
            for q in tp.points:                               # the part moved: re-measure the others
                if q is not p:
                    tp.tap(q)
            say(f"  [taps] {obj}: map now {[(q.name, round(1000 * q.err_m, 1)) for q in tp.points]} mm")
    return tp.seated(), tp
