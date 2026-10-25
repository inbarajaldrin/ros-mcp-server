"""The recovery supervisor beside drive-mate (AR3, decision 2026-10-22: "nudge is not part of compliant insertion, it is
part of the recovery method"): the depth seat check, release + nudge, edge press, reseat, and the taps that judge a
seat on a free base. The planner calls drive-mate, then this when ExecuteMate refuses, then ExecuteMate again.
Moved out of replay_real_assembly.py 10-23."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation as R_

import cell_io
from cell_io import (REPO_ROOT, RUN_BASE, SEATED, observe_base_pose, run_subprocess)  # noqa: F401


def BASE_FIXED_CFG() -> bool:
    from primitives.shared.config import BASE_FIXED
    return bool(BASE_FIXED)


CROSS_END_DIFF_MM = 1.5                   # one end this much higher than the other ...


CROSS_EDGE_DIFF_MM = 0.8                  # ... and at that end one long edge this much higher: riding a channel lip


CROSS_NUDGE_AT_MM = 12.0                  # push this far in from the high end (line_brown 2026-10-20: -58 of 70)


CROSS_LOW_EDGE_MAX_MM = 3.0               # ... and its other edge still near the seat (a lip carries one edge only)


def cross_nudge_plan(res):
    """(toward, at_long_mm) for a bar whose high end rides a channel lip, from the depth check: that end high
    and, at it, the edge on the lip side high - push from that side toward the other, at the high end
    (line_brown 2026-10-20: -y end +1.2, -x edge +2 mm -> toward +x at -58 mm; seated after). None if not."""
    ends = res.get('end_heights_mm'); per = (res.get('cross') or {}).get('end_edge_heights_mm'); f = res.get('face') or {}
    if not ends or not per or not f:
        return None
    hi = 1 if ends[1] > ends[0] else 0
    if ends[hi] - ends[1 - hi] < CROSS_END_DIFF_MM or per[hi] is None or None in per[hi]:
        return None
    lo_edge, hi_edge = per[hi]
    if abs(hi_edge - lo_edge) < CROSS_EDGE_DIFF_MM:
        return None
    if min(lo_edge, hi_edge) > CROSS_LOW_EDGE_MAX_MM:
        # both edges up: the whole end rests on the base top past its pocket - the axial push (10-14 line_green fix),
        # not a lip. 10-22 09:35: edges +7.1/+8.9 called a lip, the sideways nudge lowered onto the seated U part
        # beside line_green twice (7.9 / 7.7 N, stopped)
        return None
    cross_ax = 'x' if f['long_axis'] == 'y' else 'y'
    toward = ('+' if lo_edge > hi_edge else '-') + cross_ax      # away from the high edge
    at = (1 if hi else -1) * (1000 * f['half_long_m'] - CROSS_NUDGE_AT_MM)
    return toward, at


RESEAT_MAX = 1


# Tilted line_green = one end resting on base1's top past the end of its pocket (the 140 mm bar
# has almost no axial play). Operator's fix, verified 2026-10-14: gripper closed, from safe height,
# lower beside the high end, push ALONG the bar toward the slot in small steps at <= 5 N (base1 is
# glued to the table), re-check depth after each step. Regrasps and pressing did not work.
NUDGE_MAX_FORCE_N = 4.0              # push until the bar is blocked (seated or stuck): operator's rule,


NUDGE_PUSH_PAST_MM = 15.0            # travel cap (operator): finger face at most this far past the estimated end


NUDGE_FIRST_END_MM = 4.0             # depth end this high after an insert: a wedge - the nudge before any tap


NUDGE_SEAT_OVER_MM = 2.0             # ...and never past the seated CAD end by more than this: line_green has no axial


NUDGE_TRIES = 3


NUDGE_SHIFT_MM = 3.0                 # axial shift that means an end rides past its pocket


NUDGE_END_DIFF_MM = 2.5              # one end this much higher than the other: nudge, not reseat


NUDGE_START_GAPS_MM = (10.0, 20.0, 30.0)   # finger block this far past the depth end; next on contact


NUDGE_PAD_Z_BASE_M = 0.037           # pad centre height (base frame): tips 22 mm, base top 17.5 mm


NUDGE_AXIAL_BIAS_MM = 7.0          # depth puts a raised end ~7 mm past the real one (edge smear on


LAST_SEAT_RES = {}             # obj -> the last depth seat check (the ladder's final word when it runs out)


def _seat_check_accepted(obj, ob, **kw):
    """seat_check.check, passing only what ExecuteMate would certify, one read-sigma inside its band (seams.acceptance:
    one rule for both - attempt 24 stopped its recovery at an end +2.0 mm the certifier then refused)."""
    from seat_check import check as _check
    res = _check(obj, ob, **kw)
    LAST_SEAT_RES[obj] = res
    if res.get('ok') and res.get('height_err_mm') is not None:
        from seams.acceptance import margin_m, seat_inside
        inside, why = seat_inside(kw.get('base') or RUN_BASE['name'], obj, float(res['height_err_mm']) / 1000.0,
                                  res.get('end_heights_mm') or [], margin=margin_m())
        if not inside:
            res = dict(res, ok=False, acceptance=why)
            print(f"  [seat-check] {obj}: seat check passes it, the certifier would not ({why}) -> recovery goes on",
                  flush=True)
    return res


def nudge_into_slot(obj, ob, res, say, out_dir) -> bool:
    """Push a tilted bar's raised end along its axis into its slot (operator's method): closed
    fingers lowered beside the high end from safe height, then ONE guarded push toward the slot
    that stops at NUDGE_MAX_FORCE_N — the bar slides while it is free and the force rises once it
    is blocked (seated against the pocket end, or stuck). Depth re-check after every try; the next
    try starts from the new measurement. The base is re-fitted by every check (7th-axis creep)."""
    if not BASE_FIXED_CFG() and os.environ.get('NUDGE_FREE_BASE', '0') != '1':
        # on a free base the push never blocks - the base slides first: attempt 27's three pushes ran their travel cap
        # and moved base1 ~50 mm while the bar was seated (tap map). Vertical presses (seams.tap_probe) only.
        say(f"  [nudge] {obj}: free base - no lateral push (NUDGE_FREE_BASE=1 to allow)")
        return False
    check = _seat_check_accepted
    from primitives.shared.config import RG2_TIP_CLOSED_M, RG2_PAD_HALF_M, SAFE_HEIGHT
    base = [ob]
    f = res.get('face') or {'cx': 0.0, 'cy': 0.0, 'z': NUDGE_PAD_Z_BASE_M, 'long_axis': 'y'}
    def world(yb):          # yb: along the part's long axis from its CAD top-face centre (line_green: base y)
        (bx, by, bz), bq = base[0]
        p = np.array([f['cx'], f['cy'] + yb, f['z']]) if f['long_axis'] == 'y' else np.array([f['cx'] + yb, f['cy'], f['z']])
        return np.array([bx, by, bz]) + R_.from_quat(bq).as_matrix() @ p
    fz = lambda p: p[2] + RG2_TIP_CLOSED_M - RG2_PAD_HALF_M          # flange z, closed, top-down
    half = 10.1                                                      # closed finger block half-width (mm)
    q = ['0.0', '1.0', '0.0', '0.0']
    X = lambda p, z: [f"{p[0]:.4f}", f"{p[1]:.4f}", f"{z:.4f}"]
    go = lambda argv, label, t=60: run_subprocess(['python3', '-u', '-m', *argv], label=label, timeout_s=t)
    for k in range(1, NUDGE_TRIES + 1):
        ends, ext = res.get('end_heights_mm'), (res.get('axial') or {}).get('extent_mm')
        if not ends or not ext:
            say(f"  [nudge] {obj}: no end heights / axial extent -> cannot nudge"); return False
        sgn = 1.0 if ends[1] > ends[0] else -1.0                     # +1: the +y (base) end is high
        raw_mm = ext[1] if sgn > 0 else ext[0]                       # depth's end: never short of the real one
        end_mm = raw_mm - sgn * NUDGE_AXIAL_BIAS_MM                  # corrected (depth smears a raised end out)
        face_mm = end_mm - sgn * NUDGE_PUSH_PAST_MM
        if f.get('half_long_m'):                                     # the seated end bounds it (no axial stop to meet)
            face_mm = sgn * max(sgn * face_mm, 1000 * float(f['half_long_m']) - NUDGE_SEAT_OVER_MM)
        push = world((face_mm + sgn * half) / 1000.0)
        say(f"  [nudge] try {k}: {'+y' if sgn > 0 else '-y'} end high, depth end {raw_mm:+.0f} mm (base) "
            f"-> push toward the slot until {NUDGE_MAX_FORCE_N:g} N (finger face to {face_mm:+.0f}: at most "
            f"{NUDGE_PUSH_PAST_MM:g} mm past {end_mm:+.0f}, {NUDGE_SEAT_OVER_MM:g} mm past the seated end)")
        for argv, label in [
                (['primitives.control_gripper', 'close', '--mode', 'real'], 'nudge close'),
                (['primitives.move_to_safe_height', '--mode', 'real'], 'nudge safe height')]:
            if go(argv, label).get('result') == 'failure':
                say(f"\n!!! {label} failed"); return False
        # Lower in clear space past the RAW depth end (the correction above moved a -y end 7 mm onto
        # the bar, 2026-10-15). Fingers must never land on the part: on contact, lift, step 10 mm
        # further out, lower again.
        for gap in NUDGE_START_GAPS_MM:
            start = world((raw_mm + sgn * (half + gap)) / 1000.0)
            r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(start, SAFE_HEIGHT + 0.08), '--quat', *q],
                   'nudge over the start')
            if r.get('result') == 'failure':
                say(f"\n!!! nudge over the start failed: {r.get('error', '')}"); return False
            r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(start, fz(start)), '--quat', *q,
                    '--min-z', '0.15', '--max-force', f"{NUDGE_MAX_FORCE_N}", '--min-duration', '5'],
                   f'nudge lower beside the end ({gap:g} mm clear)')
            if not r.get('contact'):
                break
            say(f"  [nudge] try {k}: touched something lowering {gap:g} mm past the end ({r.get('error')}) "
                f"-> lift, further out")
            if go(['primitives.move_to_safe_height', '--mode', 'real'], 'nudge lift').get('result') == 'failure':
                return False
        else:
            say(f"\n!!! {obj}: no clear spot past the end to lower the fingers"); return False
        if r.get('result') == 'failure':
            say(f"\n!!! nudge lower beside the end failed: {r.get('error', '')}"); return False
        time.sleep(1.5)                                              # settle before the force baseline
        r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(push, fz(push)), '--quat', *q,
                '--min-z', '0.15', '--max-force', f"{NUDGE_MAX_FORCE_N}", '--min-duration', '10'],
               f'nudge push {k} (until {NUDGE_MAX_FORCE_N:g} N)')
        say(f"  [nudge] push {k}: {'stopped at ' + str(r.get('error')) if r.get('contact') else 'reached the travel cap'}")
        for argv, label in [
                (['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(start, fz(start) + 0.02), '--quat', *q,
                  '--min-z', '0.15'], 'nudge back off'),
                (['primitives.move_to_safe_height', '--mode', 'real'], 'nudge safe height')]:
            if go(argv, label).get('result') == 'failure':
                say(f"\n!!! {label} failed"); return False
        res = check(obj, base[0], move=True, out_dir=out_dir, base=RUN_BASE['name'])
        base[0] = res.get('observed_base') or base[0]
        say(f"  [seat-check] {obj} after nudge {k}: {json.dumps({k_: v for k_, v in res.items() if k_ not in ('profile_mm', 'observed_base')})}")
        if res.get('ok'):
            _record_seat_check_error(obj, res)     # attempt 23: only the first-pass and after-taps passes recorded it
            return True
        if _level_but_high(res) or abs(res.get('roll_deg', 0.0)) >= ROLL_PRESS_DEG:
            say(f"  [nudge] {obj}: level along it now but high/rolled - stop nudging, edge press next")
            return False
    return False


EDGE_OFFSET_MM = 8.0                 # tap/press point off the bar's centreline (26 mm bar, 20 mm block)


EDGE_TAP_N = 3.0                     # measuring tap: a 3 N centre tap did not move the wedged bar


EDGE_PRESS_N = 4.0                   # guard under the operator's 5 N limit: presses overshoot by up to 0.7 N (run 32)


EDGE_DIFF_MIN_MM = 0.8               # edges closer than this: no roll to fix -> do not guess


ROLL_PRESS_DEG = 5.0                 # depth roll this large: a cocked bar -> edge press (a seated bar reads ~-3 deg: bias)


def _level_but_high(res) -> bool:
    """Depth says: level along the bar, centred along its axis, but the whole top is high —
    the bar is cocked across its width (roll), which depth cannot measure (16 mm strip; it reads
    -4 deg on every seat, seated or not: 54/54 checks, 2026-10-17)."""
    if 'pitch_deg' not in res or res.get('ok'):
        return False
    shift = abs(float((res.get('axial') or {}).get('shift_mm', 0.0)))
    return (abs(res['pitch_deg']) <= res.get('limits', {}).get('pitch_deg', 1.5) and shift < NUDGE_SHIFT_MM
            and res.get('height_err_mm', 0.0) > res.get('limits', {}).get('height_mm', 2.5))


def press_high_edge(obj, ob, say, out_dir) -> bool:
    """Rare roll wedge (operator, 2026-10-17: a bar cocked across its width sat 3.6 mm high by touch;
    a 5 N press on its high edge, off-centre at the grasp point, seated it). Find the high edge
    MECHANICALLY — tap both edges at 3 N — press only that edge at <= 5 N, re-check. No guessing:
    if the edges read the same there is no roll and it stops."""
    from tap_check import tap_once
    check = _seat_check_accepted
    from seat_check import TOP_FACE as _TF
    across_x = _TF[(obj, RUN_BASE['name'])].get('long_axis', 'y') == 'y'          # edges lie across the long axis
    off = lambda s: dict(offset_x_mm=s * EDGE_OFFSET_MM) if across_x else dict(offset_y_mm=s * EDGE_OFFSET_MM)
    taps = {s: tap_once(obj, ob, **off(s), force=EDGE_TAP_N, say=say, base=RUN_BASE['name']) for s in (-1, 1)}
    if not all(t['contact'] and t['err_mm'] is not None for t in taps.values()):
        say(f"  [edge] {obj}: an edge tap found no contact {taps} -> not pressing"); return False
    hi = max(taps, key=lambda s: taps[s]['err_mm'])
    diff = taps[hi]['err_mm'] - taps[-hi]['err_mm']
    if diff < EDGE_DIFF_MIN_MM:
        say(f"  [edge] {obj}: edges level ({taps[-1]['err_mm']:+.1f} / {taps[1]['err_mm']:+.1f} mm) — "
            f"not a roll, not pressing"); return False
    say(f"  [edge] {obj}: {'+x' if hi > 0 else '-x'} edge high by {diff:.1f} mm -> press it at {EDGE_PRESS_N:g} N")
    tap_once(obj, ob, **off(hi), force=EDGE_PRESS_N, press_mm=4.0, say=say, base=RUN_BASE['name'])
    res = check(obj, ob, move=True, out_dir=out_dir, base=RUN_BASE['name'])
    say(f"  [seat-check] {obj} after the edge press: "
        f"{json.dumps({k: v for k, v in res.items() if k not in ('profile_mm', 'observed_base')})}")
    if res.get('ok'):
        _record_seat_check_error(obj, res)
    return bool(res.get('ok'))


def _record_seat_check_error(obj, res):
    """A passed depth seat check measures where the part really sits ACROSS its slot (cross centre) and along it (axial
    shift), base frame, mm: written to the verify ledger as rel_err_mm so the part threading over it next is aimed at it
    (_follow_seated_general). 13:39/13:54 the yellow, aimed at its CAD seat, came down on line_green's top twice -
    line_green sat 1.7 mm across from CAD (its check), the yellow straddles it with +-1 mm."""
    try:
        f = res.get('face') or {}
        cross = float((res.get('cross') or {}).get('centre_mm'))
        along = float((res.get('axial') or {}).get('shift_mm', 0.0))
        ext_ = (res.get('axial') or {}).get('extent_mm')
        if not (ext_ and f.get('half_long_m') and abs((ext_[1] - ext_[0]) - 2000 * f['half_long_m']) <= RESEAT_SPAN_TOL_MM):
            # a depth span that is not the part's own length is smeared, not a place (10-24 17:45: line_green seated flat,
            # span -70..+90 = 160 mm for its 140 -> 'shift +10 mm' was recorded for the yellow's aim)
            along = 0.0
        e = [cross, along] if f.get('long_axis', 'y') == 'y' else [along, cross]
        SEATED.set_seat_error(obj, e, 'depth seat check', touch_t=True)
        print(f"  [seat-check] {obj}: measured seat error {e} mm (base frame) recorded for the parts over it", flush=True)
    except Exception as ex:
        print(f"  [seat-check] {obj}: seat error not recorded ({ex})", flush=True)


def side_tap(obj, res, say) -> bool:
    """Measure -> tap -> re-check, in one constant direction (operator 10-24: "moving the base is fine ... measure the
    wedge and do manoeuvres to nudge it with constant direction and recheck"): each round taps the part along its
    length from the high end toward its slot (ablations/nudge_part.py, quick taps), the end face taken from the latest
    depth check when its span is the part's own length, the stroke going SIDE_TAP_STEP_MM further past the CAD face
    each round (to nudge_part's 4 mm cap); nudge_part re-reads the base and runs the depth seat check after every
    round. True when a check passes."""
    f = res.get('face') or {}
    ends = res.get('end_heights_mm')
    if not ends or not f.get('half_long_m'):
        return False
    hi = 1 if ends[1] > ends[0] else 0                              # the high end (0: -long, 1: +long) - kept all rounds
    ax = f.get('long_axis', 'y')
    toward = ('-' if hi == 1 else '+') + ax                         # constant: from the high end toward the other
    import seat_check as _sc
    for k in range(SIDE_TAP_ROUNDS):
        ext = (res.get('axial') or {}).get('extent_mm')
        off = 0.0
        if ext and abs((ext[1] - ext[0]) - 2000 * f['half_long_m']) <= RESEAT_SPAN_TOL_MM:
            cad_end = (1 if hi == 1 else -1) * 1000 * f['half_long_m']
            off = float(np.clip(abs(ext[hi] - cad_end), 0.0, 4.0))
        push = min(4.0, SIDE_TAP_STEP_MM * k)
        say(f"  [side-tap] {obj} round {k + 1}/{SIDE_TAP_ROUNDS}: taps toward {toward}, end {off:.1f} mm past CAD "
            f"(depth span {ext}), stroke {push:.1f} mm past the CAD face")
        r = run_subprocess([sys.executable, '-u', str(Path(__file__).resolve().parent / 'nudge_part.py'), '--object', obj,
                            '--base', RUN_BASE['name'], '--toward', toward, '--face-offset-mm', f"{off:.1f}",
                            '--push-mm', f"{push:.1f}", '--taps', '2'], label=f'side taps {obj} round {k + 1}', timeout_s=300)
        if r.get('result') != 'failure' and r.get('returncode', 0) == 0:
            say(f"  [side-tap] {obj}: seated after round {k + 1}"); return True
        # the depth check nudge_part ran is in its output only - measure again for the next round's end face
        ob_ = observe_base_pose(RUN_BASE['name'])
        if ob_ is None:
            break
        res = _sc.check(obj, ob_, move=True, out_dir=None, base=RUN_BASE['name'])
        if res.get('ok'):
            say(f"  [side-tap] {obj}: seated after round {k + 1}"); return True
        e_ = res.get('end_heights_mm') or ends
        say(f"  [side-tap] {obj} after round {k + 1}: ends {e_} mm, span {(res.get('axial') or {}).get('extent_mm')}")
    say(f"  [side-tap] {obj}: not seated after {SIDE_TAP_ROUNDS} rounds")
    return False


def _seat_by_taps(obj, args, say):
    """The seat judged by taps (operator 10-22: "use force to tap smartly - you know where it seats and the force
    feedback"): one depth view - it re-fits the base and measures where the part sits ACROSS its slot (what the parts
    over it follow) - then seams.tap_probe maps the top against the seated CAD and presses only what is high, <= 4.5 N,
    vertical. Attempt 27: the depth check called line_green's end +1.5/+1.7/+1.9 mm high, three lateral nudges pushed
    the free base ~50 mm, and the taps then read the bar seated (-1.35..+0.05 mm). True / False; None: the taps could
    not judge it (taps off the part) - the depth ladder decides."""
    import seat_check as _sc
    from seams.tap_probe import tap_recover
    out_dir = REPO_ROOT / 'ablations' / 'logs' / 'seat_checks'
    res = _sc.check(obj, args.observed_base, move=True, out_dir=out_dir, base=RUN_BASE['name'])
    ob = res.get('observed_base') or args.observed_base
    args.observed_base = ob
    # the taps go where drive-mate last MEASURED the base (its align read, top-down), not the depth re-fit from the
    # seat check's tilted view: run 28c's taps from that re-fit all landed beside the bar (no contact at any point)
    ob = getattr(args, 'measured_base', None) or ob
    say(f"  [seat-check] {obj} (depth, for the base and the across error): "
        f"{json.dumps({k: v for k, v in res.items() if k not in ('profile_mm', 'observed_base')})}")
    ends_ = res.get('end_heights_mm') or []
    if res.get('height_err_mm') is not None:
        from seams.acceptance import seat_inside
        inside_, why_ = seat_inside(RUN_BASE['name'], obj, float(res['height_err_mm']) / 1000.0, ends_)
        if inside_:
            # depth already passes it by the certifier's own rule: the taps would only confirm a seat (run 38b: line_green
            # seated from the insert, ends +1.5 / -0.2 mm, 49 s of taps) - ExecuteMate measures it next anyway
            say(f"  [seat-check] {obj}: depth passes it ({why_}) - no taps")
            if (res.get('cross') or {}).get('centre_mm') is not None:
                _record_seat_check_error(obj, res)
            SEATED.mark(obj, True, how='depth seat check')
            return True
    from compliant_insertion_studio.wrapper.cad_geometry import wedge_possible
    if wedge_possible(RUN_BASE['name'], obj) and ends_ and max(ends_) > NUDGE_FIRST_END_MM and (res.get('axial') or {}).get('extent_mm'):
        # an end this high is a wedge depth cannot mistake (runs 31/32 after the insert: +7.7 / +8.1 mm; seated bars read
        # +1.5..+2.5 with depth's tilt): the map and a press before the nudge only confirmed it (~60 s, every press
        # blocked) - nudge first, the taps judge after it
        say(f"  [taps] {obj}: depth shows an end {max(ends_):+.1f} mm high (> {NUDGE_FIRST_END_MM:g}: a wedge) -> "
            f"one lateral nudge first, then the taps judge")
        ok, tp, blocked = False, None, True
    else:
        ok, tp = tap_recover(obj, ob, RUN_BASE['name'], say)
        if tp.off_part() or ({'mid', 'end-', 'end+'} & set(tp.missed())):
            return None                                      # the taps could not judge it: the depth ladder
        blocked = not ok and any(p.blocked for p in tp.points)
        if blocked:
            say(f"  [taps] {obj}: vertical presses blocked -> one lateral nudge, then the taps judge")
    if blocked and not BASE_FIXED_CFG():
        # free base: a slow push drags the base with the part (runs 89/89b/90/91/92: the 10 s push 'reached the travel
        # cap', frames 18:19 show base1 moving, base1 turned ~20 deg over these recoveries). Operator 10-24: "tap it into
        # place from the side ... when it's not fixed a small tap will seat it and then you can check if it seated" - the
        # taps of nudge_part.py: from the high end along the part toward its slot, out to the CAD end face (corrected by
        # where depth sees that end) at the arm's short-move speed and straight back, then a depth seat check.
        if side_tap(obj, res, say):
            return True
        say(f"  [taps] {obj}: the side taps did not seat it -> the depth ladder and its reseat")
        return None
    if blocked:
        # the vertical presses are blocked (an end riding a lip springs back: 22:30 line_green +4.2 -> +2.7 -> +3.8 -> +3.8);
        # ONE lateral nudge frees it (22:35: depth ends +6.1 -> +0.6, taps then -0.8..-1.1), the taps judge after it -
        # never a second nudge on a depth reading (attempt 27: three, chasing +1.5-1.9 mm, moved the base ~50 mm)
        prev = os.environ.get('NUDGE_FREE_BASE')
        os.environ['NUDGE_FREE_BASE'] = '1'
        global NUDGE_TRIES
        tries0, NUDGE_TRIES = NUDGE_TRIES, 1
        try:
            nudge_into_slot(obj, args.observed_base, res, say, out_dir)
        finally:
            NUDGE_TRIES = tries0
            if prev is None:
                os.environ.pop('NUDGE_FREE_BASE', None)
            else:
                os.environ['NUDGE_FREE_BASE'] = prev
        ob2 = observe_base_pose(RUN_BASE['name']) or ob       # the push moves a free base: tap where it is now
        args.observed_base = args.measured_base = ob2
        ok, tp = tap_recover(obj, ob2, RUN_BASE['name'], say)
        if tp.off_part() or ({'mid', 'end-', 'end+'} & set(tp.missed())):
            # a point off the part right after a push: the base read lags the push (run 31: end+ on a U arm, +13.7 mm,
            # the other three points seated) - one more base read from its own view, one more map
            ob3 = observe_base_pose(RUN_BASE['name'])
            if ob3 is not None:
                args.observed_base = args.measured_base = ob3
                ok, tp = tap_recover(obj, ob3, RUN_BASE['name'], say)
        if tp.off_part() or ({'mid', 'end-', 'end+'} & set(tp.missed())):
            return None
    if ok and not blocked and (res.get('cross') or {}).get('centre_mm') is not None:
        # the across error the parts over it follow - from this depth read only if nothing moved the part since (after a
        # nudge it is the pre-nudge read: run 32 recorded [-0.2, -3.0] mm of a wedged bar the taps then found seated)
        _record_seat_check_error(obj, res)
    if ok:
        # with its own time and the points' places along the bar: ExecuteMate judges the ends from it while nothing has
        # moved since (seams.vision_ops.tap_ends)
        SEATED.mark(obj, True, how='taps', tap_map_mm={p.name: ('missed' if p.err_m is None else round(1000 * p.err_m, 2)) for p in tp.points},
                    tap_t=time.time(), tap_along_mm={p.name: round(1000 * p.along_m, 1) for p in tp.points})
    return ok


def _taps_mark(obj, args, say) -> bool:
    """Tap the part where drive-mate last measured the base; True (and the timed tap map in the ledger, which ExecuteMate
    judges the ends from) when the taps find it seated."""
    from seams.tap_probe import tap_recover
    ob_ = getattr(args, 'measured_base', None) or getattr(args, 'observed_base', None)
    if ob_ is None:
        return False
    say(f"  [taps] {obj}: the certifier read an end over its band - contact measures the ends")
    ok, tp = tap_recover(obj, ob_, RUN_BASE['name'], say)
    if not ok and (tp.off_part() or ({'mid', 'end-', 'end+'} & set(tp.missed()))):
        # taps beside the part: the base read they were placed from is stale by mm (run 50: end- and edge+ missed on a
        # bar depth read seated, ends +1.55 / +0.13 mm) - read the base where it is now and map once more
        ob2 = observe_base_pose(RUN_BASE['name'])
        if ob2 is not None:
            say(f"  [taps] {obj}: taps off/beside the part {tp.off_part() or tp.missed()} - base re-read, mapping again")
            args.observed_base = args.measured_base = ob2
            ok, tp = tap_recover(obj, ob2, RUN_BASE['name'], say)
    if ok:
        SEATED.mark(obj, True, how='taps', tap_map_mm={p.name: ('missed' if p.err_m is None else round(1000 * p.err_m, 2)) for p in tp.points},
                    tap_t=time.time(), tap_along_mm={p.name: round(1000 * p.along_m, 1) for p in tp.points})
    return bool(ok)


def seat_check_and_reseat(obj, state, args, say) -> bool:
    """The depth seat check (+ nudges / edge press); a pass is recorded in the verify ledger as a depth
    confirmation, which verify_seated accepts when the markers cannot see the part (2026-10-20: fork_orange,
    depth -0.08 mm from its CAD height, failed the run on 'marker not detected'). On a free base the taps judge
    the seat first (_seat_by_taps); the depth ladder runs only when they could not."""
    if (os.environ.get('SEAT_BY_TAPS', '1') == '1' and not BASE_FIXED_CFG()
            and getattr(args, 'observed_base', None) is not None):
        ok_t = _seat_by_taps(obj, args, say)
        if ok_t:
            return True
        if ok_t is None:
            say(f"  [seat-check] {obj}: the taps could not judge it -> the depth ladder")
        else:
            # run 33: after the nudge line_green sat +4.8 mm level and rolled (a long edge on the channel lip), its high
            # edge blocked at 4 N - the ladder's reseat with the insert grasp is the recovery left (run 31: it worked)
            say(f"  [seat-check] {obj}: the taps could not seat it -> the depth ladder (edge press, reseat)")
    ok = _seat_check_and_reseat(obj, state, args, say)
    if not ok:
        # the recovery ladder stops one read-sigma inside the certifier's band; when it has run out, a last reading inside
        # the band itself goes on to ExecuteMate, which decides (run 31: an end at +1.53 mm stopped the run, 0.03 mm over
        # the ladder's +-1.5)
        r_ = LAST_SEAT_RES.get(obj) or {}
        if r_.get('height_err_mm') is not None:
            from seams.acceptance import seat_inside
            inside, why = seat_inside(RUN_BASE['name'], obj, float(r_['height_err_mm']) / 1000.0, r_.get('end_heights_mm') or [])
            if inside:
                say(f"  [seat-check] {obj}: the ladder ran out inside the certifier's band ({why}) - ExecuteMate decides")
                ok = True
    if ok:
        SEATED.mark(obj, True, how='depth seat check')                  # merged: keeps rel_err_mm
    return ok


def _seat_check_and_reseat(obj, state, args, say) -> bool:
    """Depth-check the seated part; if it sits tilted or high, take it with the insert's own
    grasp (the pose the insert released it from), and insert it again. Fails closed."""
    check = _seat_check_accepted
    from disassemble import seat_grasp, SEAT_EE_QUAT, HOVER_DZ
    ob = getattr(args, 'observed_base', None)
    if ob is None:
        say(f"  [seat-check] {obj}: no observed base -> skipped"); return True
    out_dir = REPO_ROOT / 'ablations' / 'logs' / 'seat_checks'
    for attempt in range(RESEAT_MAX + 1):
        lift = False          # a press / nudge left it high: lift it with the insert grasp and insert it again
        res = check(obj, ob, move=True, out_dir=out_dir, base=RUN_BASE['name'])
        ob = res.get('observed_base') or ob                    # re-fitted from the check's own view
        args.observed_base = ob
        say(f"  [seat-check] {obj}: {json.dumps({k: v for k, v in res.items() if k not in ('profile_mm', 'observed_base')})}")
        if 'pitch_deg' in res and not res.get('ok') and (_level_but_high(res) or abs(res.get('roll_deg', 0.0)) >= ROLL_PRESS_DEG
                                                           and abs(res['pitch_deg']) <= res.get('limits', {}).get('pitch_deg', 1.5)):
            # cocked across its width (operator 2026-10-20: "this is a roll issue ... we already faced a roll solve for
            # fmb1"): the FMB1 edge press - tap both edges, press the high one - not axial nudges
            say(f"  [seat-check] {obj}: level along it but {res.get('height_err_mm', 0):+.1f} mm high, roll {res.get('roll_deg', 0):+.1f} deg "
                f"-> edge press (FMB1 roll fix)")
            if press_high_edge(obj, ob, say, out_dir):
                return True
            if attempt >= RESEAT_MAX:
                say(f"\n!!! {obj} still not seated after the edge press"); return False
            lift = True
        # a seat the check passes is not touched (operator 10-22: no nudges on a seated part; 14:02 the lip rule nudged an
        # ok line_green sideways)
        plan = cross_nudge_plan(res) if (not lift and 'pitch_deg' in res and not res.get('ok')) else None
        if not BASE_FIXED_CFG() and os.environ.get('NUDGE_FREE_BASE', '0') != '1':
            plan = None                                       # no sideways push on a free base (see nudge_into_slot)
        if abs(float((res.get('axial') or {}).get('shift_mm', 0.0))) >= NUDGE_SHIFT_MM:
            plan = None        # shifted along its length: the axial nudge first (line_red 2026-10-20: -16 mm,
                               # the sideways taps did nothing, the axial push seated it)
        if plan is not None and attempt < RESEAT_MAX:            # nudge, then the next pass re-checks
            # an end riding a channel lip: the depth limits call it ok (line_brown 2026-10-20, the operator
            # saw it unseated) - push that end sideways off the lip, to the CAD face (the base is not fixed)
            toward, at = plan
            say(f"  [seat-check] {obj}: high end on a channel lip -> sideways nudge toward {toward} at {at:+.0f} mm")
            r = run_subprocess(['python3', '-u', 'ablations/nudge_part.py', '--object', obj, '--base', RUN_BASE['name'],
                                f'--toward={toward}', '--at-long', f"{at:.0f}"], label=f'nudge {obj} {toward}', timeout_s=240)
            ob = observe_base_pose(RUN_BASE['name']) or ob
            args.observed_base = ob
            continue
        if res.get('ok') and not lift:
            _record_seat_check_error(obj, res)
            return True
        if not lift and res.get('mode', '').startswith('median height'):
            if res.get('ok'):
                return True
            say(f"\n!!! {obj}: top {res.get('height_err_mm'):+.1f} mm from its CAD height - not seated"); return False
        if 'pitch_deg' not in res:
            say(f"  [seat-check] {obj}: could not measure -> continuing"); return True
        shift = abs(float((res.get('axial') or {}).get('shift_mm', 0.0)))
        ends = res.get('end_heights_mm') or [0.0, 0.0]
        if lift:
            pass
        elif (abs(res['pitch_deg']) > res.get('limits', {}).get('pitch_deg', 1.5)
                or shift >= NUDGE_SHIFT_MM or abs(ends[1] - ends[0]) >= NUDGE_END_DIFF_MM):
            # tilted, or shifted along its axis = an end on base1's top past the pocket end; a
            # regrasp holds it where it is. 2026-10-15: a bar 4-7 mm high and +5 mm along its axis
            # (pitch only 1.2 deg) went to the top-down reseat, which cannot move it, and failed.
            ok = False
            shift = float((res.get('axial') or {}).get('shift_mm', 0.0))
            f_ = res.get('face') or {}
            if abs(shift) >= NUDGE_SHIFT_MM and f_:
                # taps along the bar to its CAD end face (operator 2026-10-20: a slow push drags the unfixed
                # base - cycle 1's axial push moved base3 10 mm); start clear of the MEASURED end
                lax = f_.get('long_axis', 'y')
                # direction from the HIGH end toward the slot (as nudge_into_slot): the extent midpoint is smeared
                # by a raised end (cycle 2: -y end 6.7 mm high, extent midpoint +7 -> taps pushed it further -y)
                ends_ = res.get('end_heights_mm') or [0.0, 0.0]
                toward = ('+' if ends_[0] > ends_[1] else '-') + lax
                say(f"  [seat-check] {obj}: shifted {shift:+.0f} mm along its length -> taps toward {toward} to the CAD end face")
                run_subprocess(['python3', '-u', 'ablations/nudge_part.py', '--object', obj, '--base', RUN_BASE['name'],
                                f'--toward={toward}', '--face-offset-mm', f"{abs(shift) + 3:.0f}", '--taps', '3'],
                               label=f'tap {obj} {toward}', timeout_s=300)
                res2 = check(obj, ob, move=True, out_dir=out_dir, base=RUN_BASE['name'])
                ob = res2.get('observed_base') or ob
                say(f"  [seat-check] {obj} after the taps: {json.dumps({k: v for k, v in res2.items() if k not in ('profile_mm', 'observed_base')})}")
                ok = bool(res2.get('ok'))
                res = res2 if 'pitch_deg' in res2 else res
                if ok:
                    _record_seat_check_error(obj, res2)
            if not ok:
                ok = nudge_into_slot(obj, ob, res, say, out_dir)      # the slow push as the fallback
            if not ok:
                res2 = check(obj, ob, move=True, out_dir=out_dir, base=RUN_BASE['name'])
                if _level_but_high(res2) or abs(res2.get('roll_deg', 0.0)) >= ROLL_PRESS_DEG:   # level but high: a roll
                    say(f"  [seat-check] {obj}: level but {res2['height_err_mm']:+.1f} mm high after the nudge -> edge taps")
                    ok = press_high_edge(obj, ob, say, out_dir)
                if 'pitch_deg' in res2:
                    res = res2
            if ok:
                return ok
            if attempt >= RESEAT_MAX:
                say(f"\n!!! {obj} still not seated after the nudge"); return False
            # run 73: line_green lay level ~5 mm high (taps: mid +4.8, both edges +4.2/+5.3, the high one blocked at
            # 4 N) after the axial taps, the slow push and the edge check - nothing a push fixes; lift and insert again
            lift = True
        elif _level_but_high(res):
            say(f"  [seat-check] {obj}: level and centred but {res['height_err_mm']:+.1f} mm high -> edge taps")
            ok = press_high_edge(obj, ob, say, out_dir)
            if ok:
                return ok
            if attempt >= RESEAT_MAX:
                say(f"\n!!! {obj} still not seated after the edge press"); return False
            lift = True
        if attempt == RESEAT_MAX:
            say(f"\n!!! {obj} still not flat after {RESEAT_MAX} reseat(s)"); return False
        from primitives.shared.config import get_grasp_id_for_assembly, get_gripper_width_mm
        gid = get_grasp_id_for_assembly(obj); w = get_gripper_width_mm(obj, gid)
        tcp = seat_grasp(obj, gid, ob, base_name=RUN_BASE['name'])
        # grip it where it LIES along its length, not at the CAD seat point: re-inserted from a grip at the seat point,
        # a bar lying off along its axis goes back to the same spot (10-24: line_green spanned -72..+68 mm in 13 wedged
        # checks - 140 mm, CAD's length, 2 mm toward -y: its -y end on base1's wedge; the reseat gripped at 0 and put it
        # there again in run 91). Only when the depth span is the part's own length (the smeared reads are not).
        ax_ = (res.get('axial') or {}).get('extent_mm'); f_ = res.get('face') or {}
        if ax_ and f_.get('half_long_m') and abs((ax_[1] - ax_[0]) - 2000 * f_['half_long_m']) <= RESEAT_SPAN_TOL_MM:
            sh = float(np.clip((ax_[0] + ax_[1]) / 2.0, -RESEAT_SHIFT_MAX_MM, RESEAT_SHIFT_MAX_MM)) / 1000.0
            v_ = [0.0, sh, 0.0] if f_.get('long_axis', 'y') == 'y' else [sh, 0.0, 0.0]
            dw = R_.from_quat(ob[1]).apply(v_)
            tcp = [tcp[0] + dw[0], tcp[1] + dw[1], tcp[2]]
            say(f"  [seat-check] {obj}: lies {1000 * sh:+.1f} mm along its length (depth span {ax_} mm) -> reseat grip there")
        # the part sits high: grip it where the insert held it, not that much deeper (run 73: +5..7 mm on the rim - at
        # the seat height the fingertips would reach past its underside); depth reads over the taps, so this errs high
        lift_h = min(max(float(res.get('height_err_mm') or 0.0), 0.0), 8.0) / 1000.0
        tcp = [tcp[0], tcp[1], tcp[2] + lift_h]
        from disassemble import grasp_offset_for
        os.environ['GRASP_TOOL_OFFSET_M'] = f"{grasp_offset_for(obj):.5f}"   # the insert's grasp depth
        q = [str(v) for v in SEAT_EE_QUAT]
        held = (R_.from_quat(ob[1]) * R_.from_quat(_seat_quat(obj, RUN_BASE['name']))).as_quat()
        (bx, by, bz), bq = ob
        # the seat height drive-mate inserts at (UNI): learned from the part's certified inserts, else the table
        from seams import seat_learn as _sl
        dz, _n = _sl.learned_dz(obj)
        if dz is None:
            from primitives.shared.config import get_object_base_offset_m
            dz = float(get_object_base_offset_m(obj)[2])
        steps = [
            (['primitives.move_cartesian', '--mode', 'real', '--xyz', f"{tcp[0]:.4f}", f"{tcp[1]:.4f}",
              f"{tcp[2] + HOVER_DZ:.4f}", '--quat', *q], 'reseat hover', 60),
            (['primitives.control_gripper', f"{w:g}", '--mode', 'real'], 'reseat open', 30),
            (['primitives.move_cartesian', '--mode', 'real', '--xyz', f"{tcp[0]:.4f}", f"{tcp[1]:.4f}",
              f"{tcp[2]:.4f}", '--quat', *q, '--min-z', '0.15', '--max-force', '12', '--min-duration', '6'],
             'reseat descend', 60),
            (['primitives.control_gripper', 'close', '--mode', 'real'], 'reseat close', 30),
            (['primitives.translate_object', '--mode', 'real', '--object-name', obj, '--base-name', RUN_BASE['name'],
              '--insert', '--insertion-type', 'compliant', '--grasp-id', str(gid),
              '--current-object-orientation', *[f"{v:.6f}" for v in held],
              '--final-base-pos', f"{bx:.6f}", f"{by:.6f}", f"{bz + dz:.6f}",
              '--final-base-orientation', *[f"{v:.6f}" for v in bq]], 'reseat insert', 600),
            (['primitives.control_gripper', f"{w:g}", '--mode', 'real'], 'reseat release', 30),
            (['primitives.move_to_safe_height', '--mode', 'real'], 'reseat lift', 60),
        ]
        say(f"  [seat-check] {obj} {'still high' if lift else 'not flat'} -> reseat {attempt + 1}/{RESEAT_MAX} with the insert "
            f"grasp (grip {1000 * lift_h:+.1f} mm, seat dz {1000 * dz:+.2f} mm)")
        for argv, label, t in steps:
            r = run_subprocess(['python3', '-u', '-m', *argv], label=label, timeout_s=t)
            if r.get('result') == 'failure':
                say(f"\n!!! {label} failed: {r.get('error', '')}"); return False
    return False


SIDE_TAP_ROUNDS = 4          # measure -> tap -> re-check rounds on a free base
SIDE_TAP_STEP_MM = 1.0       # each round's stroke goes this much further past the CAD face (nudge_part caps at 4)
RESEAT_SPAN_TOL_MM = 4.0     # the depth span must match the part's CAD length this well to be read as its place
RESEAT_SHIFT_MAX_MM = 5.0    # ... and the grip moves at most this far along the part


def _seat_quat(obj, base_name='base1'):
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    return load_seat_pose_in_base(base_name, obj)[1]
