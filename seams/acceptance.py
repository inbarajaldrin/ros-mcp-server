#!/usr/bin/env python3
"""When a seat is accepted (2026-10-22): ONE rule for ExecuteMate's certification and the planner's per-seat recovery.

    lim = lim_z_m('base1')                     # the graph's declared half-width (z) + 2 sigma of the seat-depth read
    ok, why = seat_inside('base1', 'line_green', z_err_m, end_heights_mm)

Before: the depth seat check passed a seat at |centre| <= 2.5 mm and ends <= 4.0 mm (a long face), ExecuteMate at
2.0 mm for both (ends judged only where CAD says the part can wedge looking seated). Attempt 24 (10-22 17:08): the
taps left line_green with one end +2.0 mm, the seat check passed it and stopped the recovery, ExecuteMate refused it
(not_at_seat) - the run stopped on a seat one more tap would have fixed.
"""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
_B = None


def bindings():
    global _B
    if _B is None:
        _B = yaml.safe_load(open(REPO / 'seams' / 'bindings.yaml'))
    return _B


def assembly_of(base: str):
    B = bindings()
    return next((k for k, v in B.items() if isinstance(v, dict) and v.get('reference') == base), None)


def lim_z_m(base: str):
    """None when the base has no bound assembly (no declared acceptance: the caller's own limits stand)."""
    a = assembly_of(base)
    if a is None:
        return None
    B = bindings()
    return float(B[a]['acceptance']['halfwidth_m'][2]) + 2 * float(B['sigma']['seat_depth']['z'])


def margin_m():
    """One sigma of the seat-depth read: recovery stops this far inside the band, so the certifier's own read (another
    view, another sample) does not land a borderline seat outside it (attempt 24: end +2.0 mm in a +-2.0 mm band)."""
    return float(bindings()['sigma']['seat_depth']['z'])


def seat_inside(base: str, obj: str, z_err_m: float, ends_mm=(), margin=0.0, ends_by_contact=False):
    """(inside, why) - centre within the band; both end heights too where the part can wedge looking seated (CAD).
    margin (m): judge against the band narrowed by it (the recovery's stop rule: margin_m()).
    ends_by_contact: the ends come from taps (true heights, no view bias) - judged on the band itself, not against a
    depth centre that carries the bias (run 52: tap ends -1.45 mm vs a depth centre +1.20 refused a seated bar)."""
    lim = lim_z_m(base)
    if lim is None:
        return True, 'no declared acceptance for this base'
    lim -= margin
    from compliant_insertion_studio.wrapper.cad_geometry import wedge_possible
    if abs(z_err_m) > lim:
        return False, f'centre {1000 * z_err_m:+.2f} mm outside +-{1000 * lim:.1f} mm'
    if ends_mm and wedge_possible(base, obj):
        # the ends against the part's OWN measured centre (the centre is judged absolutely above): a wedge lifts one end
        # relative to the rest, while the depth view's bias lifts the whole part (line_green: seated bars read ends within
        # 1.63 mm of their centre over 40 readings, wedged ones 3.9-4.5 mm; on an absolute band the bias alone pushed
        # seated ends to +2.0..2.7 mm and the certifier refused seated bars, 10-23 runs 40/41/48/50)
        c_mm = 0.0 if ends_by_contact else 1000 * z_err_m      # tap ends are true heights: against the seat itself
        worst = max(ends_mm, key=lambda e: abs(e - c_mm))
        if abs(worst - c_mm) / 1000.0 > lim:
            ref = 'the seat (by contact)' if ends_by_contact else f'the centre ({c_mm:+.2f})'
            return False, (f'an end at {worst:+.2f} mm is {worst - c_mm:+.2f} mm from {ref}, outside '
                           f'+-{1000 * lim:.1f} mm (the part can wedge here)')
    return True, f'within +-{1000 * lim:.1f} mm'
