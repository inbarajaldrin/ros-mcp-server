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
PRESS_N = 4.0                # the guard trips past it; peaks overshoot it by up to 0.7 N (run 32: 4.5 -> 5.2 N, bench 5.0 -> 5.12):
                             # 4.0 keeps the peak under the operator's 5 N
OVER_M = 0.003               # the tip waits this far over the highest height seen
SPEED_M_S = 0.004            # descent speed of taps and presses
PRESS_BELOW_M = 0.0015       # a press may go this far under the seated top (a cocked part rolls down past it)
MOVED_MIN_M = 0.0005         # a press that travelled this far after first contact moved the part
CONTACT_N = 1.0              # first contact in the force record
EDGE_IN_M = 0.004            # edge taps this far in from the CAD top-face edge
FINGER_HALF_M = 0.012        # the closed finger block's half size (20.2 mm block) + margin
END_IN_M = 0.010             # end taps this far in from the CAD top-face ends: 8 mm put line_green's end tap inside the U
                             # part's opening, where the 20 mm finger block lands on the U arm beside the bar (+10.6 mm,
                             # 10-23 00:08); 25 mm still leaves a 45 mm lever for the pitch
MAX_PRESSES = 6
TAP_FAST = __import__('os').environ.get('TAP_FAST', '1') == '1'   # travel timed by distance (benched 10-23: 9.9 -> 6.9 s a tap)
FAST_UP_M_S, FAST_ACROSS_M_S, FAST_DOWN_M_S = 0.020, 0.030, 0.012  # lift unguarded; across and down guarded at TAP_N
CLEAR_M = 0.025              # sideways travel between tap points happens this far over the part's seated top: its
                             # neighbours stand up to +19.6 mm over line_green's (10-22: travel 3-7 mm over it ran the tips
                             # into them and slid the free base 56, 128, 40, 59 mm in four tap sessions - base-read audit)
VALID_M = (-0.005, 0.012)    # a tap reading outside this is not on the part (line_green's neighbours stand +19.6 mm
                             # over its top; the floor stops a seated part, so far below is a miss): never pressed
Q_DOWN = (0.0, 1.0, 0.0, 0.0)


@dataclass
class Point:
    name: str
    xyz_top: np.ndarray        # the seated CAD top face here, world (m)
    err_m: Optional[float] = None
    blocked: int = 0
    missed: bool = False         # the descent went to its floor without contact: the tip was not over the part
    along_m: float = 0.0         # along the long axis from the CAD top-face centre (the certifier extrapolates the ends)


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
        long_y = f.get('long_axis', 'y') == 'y'
        hl, hc = float(f['half_long_m']), float(f['half_cross_m'])
        # tap points from the CAD: the closed finger block (FINGER_HALF_M each way) must not overlap any seated
        # neighbour's footprint (10-23 00:08-00:12: line_green's end taps 8 and 25 mm in from its ends landed inside the
        # U parts' openings, on their arms: +10.6 / +12.3 mm). Ends: scanned inward from END_IN_M until clear.
        boxes = _neighbour_boxes(obj, base, top_z=float(f['z']))

        def b_xy(a_, c_):                                   # base-frame (x, y) of along a_, across c_
            return (f['cx'] + c_, f['cy'] + a_) if long_y else (f['cx'] + a_, f['cy'] + c_)

        def clear(a_, c_):
            x, y = b_xy(a_, c_)
            return all(x + FINGER_HALF_M < x0 or x - FINGER_HALF_M > x1 or y + FINGER_HALF_M < y0 or y - FINGER_HALF_M > y1
                       for x0, x1, y0, y1 in boxes)

        def scan(sign, c_=0.0):
            a_ = hl - END_IN_M
            while a_ >= 0.010:
                if clear(sign * a_, c_):
                    return sign * a_
                a_ -= 0.002
            return None
        pts = []
        a_mid = 0.0 if clear(0.0, 0.0) else next((s_ * a_ for a_ in np.arange(0.002, hl, 0.002) for s_ in (1, -1)
                                                    if clear(s_ * a_, 0.0)), None)
        if a_mid is not None:
            pts.append(('mid', a_mid, 0.0))
        if hl >= 2 * hc:                                    # a bar: both ends where the block is clear of neighbours
            for nm, sg in (('end-', -1), ('end+', 1)):
                a_ = scan(sg)
                if a_ is not None and (a_mid is None or abs(a_ - a_mid) >= 0.015):
                    pts.append((nm, a_, 0.0))
        ac = max(hc - EDGE_IN_M, 0.0)
        for nm, sg in (('edge-', -1), ('edge+', 1)):
            if a_mid is not None and clear(a_mid, sg * ac):
                pts.append((nm, a_mid, sg * ac))

        def w(dx, dy):
            return np.array([bx, by, bz]) + Rb.apply([dx, dy, f['z']])
        P = [Point(n, w(*b_xy(a_, c_)), along_m=float(a_)) for n, a_, c_ in pts]
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
        """Over p: straight up to the clear height (CLEAR_M over the seated top), sideways there, straight down (guarded)
        to OVER_M over the highest height seen. Never sideways near the part or its neighbours."""
        from compliant_insertion_studio.wrapper.touch_insert import tcp_now
        tgt = self._over(p)
        c = tcp_now()[0]
        def dur(d_m, v):                                    # TAP_FAST: timed by distance (the floor held each at ~2.5 s)
            return max(0.6, d_m / v) if TAP_FAST else None
        if np.hypot(c[0] - tgt[0], c[1] - tgt[1]) > 0.0005:
            z_clear = max(float(p.xyz_top[2]) + self.tip_m + CLEAR_M, float(tgt[2]))
            if c[2] < z_clear - 0.0005:
                self._move([c[0], c[1], z_clear], duration=dur(z_clear - c[2], FAST_UP_M_S))
            self._move([tgt[0], tgt[1], z_clear], force=TAP_N,
                       duration=dur(float(np.hypot(c[0] - tgt[0], c[1] - tgt[1])), FAST_ACROSS_M_S))
        c = self._z()
        if abs(c - tgt[2]) <= 0.005:
            return self._move(tgt, force=TAP_N, duration=0.8)
        return self._move(tgt, force=TAP_N, duration=dur(abs(c - tgt[2]), FAST_DOWN_M_S))

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
        # no contact = the descent reached its floor in the air: not a height (run 28c: five taps 'read' -1.45 mm, the
        # floor, contact False - the tips were beside the bar - and the map called it seated)
        p.missed = not bool(r.get('contact'))
        p.err_m = None if p.missed else float(err)
        self._move(self._over(p), duration=0.8)              # straight back up (unguarded: lifting off a contact)
        self.log.append(dict(kind='tap', point=p.name, err_mm=round(1000 * err, 2), contact=bool(r.get('contact'))))
        return p.err_m

    def map(self):
        out = {}
        for p in self.points:
            e = self.tap(p)
            out[p.name] = 'missed' if e is None else round(1000 * e, 2)
        return out

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
        peak = max([dF for _, dF in rec], default=0.0)
        if peak < CONTACT_N:                                 # never touched: the tip is not over the part
            verdict = 'missed'
            p.missed, p.err_m = True, None
        else:
            verdict = 'seated' if abs(err) <= self.band_m else ('moving' if moved else 'blocked')
            p.err_m = float(err)
        if verdict == 'blocked':
            p.blocked += 1
        self._move(self._over(p), duration=0.8)
        self.log.append(dict(kind='press', point=p.name, err_before_mm=None if err0 is None else round(1000 * err0, 2),
                             err_mm=round(1000 * err, 2), loaded_mm=round(1000 * loaded_m, 2), verdict=verdict,
                             peak_N=round(max([dF for _, dF in rec], default=0.0), 2)))
        return verdict

    def valid(self, p: Point):
        return (not p.missed) and p.err_m is not None and VALID_M[0] <= p.err_m <= VALID_M[1]

    def seated(self):
        """Every measured point inside the band, and the middle and (on a bar) both ends among them; a missed point is
        unknown, not a pass."""
        need = {'mid', 'end-', 'end+'} & {p.name for p in self.points}
        meas = [p for p in self.points if self.valid(p)]
        return need <= {p.name for p in meas} and all(abs(p.err_m) <= self.band_m for p in meas) and \
            not self.off_part()

    def off_part(self):
        """Points that read a height the part cannot have (another part's top, far below): the pose is off - stop."""
        return [p.name for p in self.points if (not p.missed) and p.err_m is not None and not self.valid(p)]

    def missed(self):
        return [p.name for p in self.points if p.missed]


def tap_recover(obj, base_pose, base, say=print, tapper: Optional[Tapper] = None):
    """tap_recover_ with the probes' ROS context released afterwards (the caller's next camera read starts its own)."""
    from compliant_insertion_studio.wrapper.touch_insert import close_node
    try:
        return tap_recover_(obj, base_pose, base, say, tapper)
    finally:
        close_node()


def _neighbour_boxes(obj, base, margin=0.001, top_z=None):
    """Base-frame xy boxes (x0, x1, y0, y1) of the parts seated before `obj` (seated CAD) that stand higher than
    top_z - 2 mm: only those can meet the finger block (the yellow sits astride line_green, whose top is below its own)."""
    from compliant_insertion_studio.wrapper.cad_geometry import get_seated_so_far
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base, _surface_points
    out = []
    for o in get_seated_so_far(base, obj):
        sp, sq = load_seat_pose_in_base(base, o)
        P = R.from_quat(sq).apply(_surface_points(o, 0.003)) + sp
        if top_z is not None and P[:, 2].max() < top_z - 0.002:
            continue
        out.append((P[:, 0].min() - margin, P[:, 0].max() + margin, P[:, 1].min() - margin, P[:, 1].max() + margin))
    return out


def close_fingers():
    """Taps and presses are made with the fingers CLOSED (one 20 mm tip block). Run 29d / 30: the pipeline tapped right
    after the release, fingers at the release width - they straddled the bar (middle 'missed'), the end taps landed on
    the U parts beside it (+12 mm). The standalone tap scripts closed the gripper first and hit every point."""
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, '-u', '-m', 'primitives.control_gripper', 'close', '--mode', 'real'], cwd=str(repo),
                   capture_output=True, timeout=30)


def tap_recover_(obj, base_pose, base, say=print, tapper: Optional[Tapper] = None):
    """Map the top, press the highest point not yet blocked twice, re-tap it, repeat; True when every point is inside
    the band. False with the map when presses no longer move it (the caller's other recoveries: nudge, reseat)."""
    tp = tapper or Tapper.for_part(obj, base_pose, base)
    close_fingers()
    t0 = time.time()
    m = tp.map()
    say(f"  [taps] {obj}: map {m} mm (band +-{1000 * tp.band_m:.1f})")
    for k in range(MAX_PRESSES):
        if tp.seated():
            say(f"  [taps] {obj}: seated by the taps ({time.time() - t0:.0f} s): {tp.log[-1] if tp.log else ''}")
            return True, tp
        if tp.off_part():
            say(f"  [taps] {obj}: taps off the part {tp.off_part()} (the base pose is off) - not pressing")
            return False, tp
        if {'mid', 'end-', 'end+'} & set(tp.missed()):
            say(f"  [taps] {obj}: missed {tp.missed()} - the map cannot judge it, not pressing")
            return False, tp
        cand = [p for p in tp.points if tp.valid(p) and p.err_m > tp.band_m and p.blocked < 2]
        if not cand:
            say(f"  [taps] {obj}: every high point is blocked - not pressing again ({[(p.name, round(1000 * p.err_m, 1)) for p in tp.points]})")
            return False, tp
        p = max(cand, key=lambda q: q.err_m)
        v = tp.press(p)
        say(f"  [taps] {obj}: press {p.name} -> {tp.log[-1]}")
        if v == 'blocked':
            # the highest point does not give under a vertical press: the lower ones will not seat it either (runs 31 and
            # 32: line_green end- blocked, then every edge press blocked too - 6 presses, ~60 s - and one lateral nudge
            # seated it). Hand it to the caller's lateral recovery now.
            say(f"  [taps] {obj}: the highest point {p.name} is blocked - not pressing the lower ones")
            return False, tp
        if v != 'blocked':
            for q in tp.points:                               # the part moved: re-measure the others
                if q is not p:
                    tp.tap(q)
            say(f"  [taps] {obj}: map now {[(q.name, round(1000 * q.err_m, 1)) for q in tp.points]} mm")
    return tp.seated(), tp
