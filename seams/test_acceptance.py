"""seams.acceptance: one seat rule for ExecuteMate and the planner's recovery (2026-10-22). Offline (CAD for wedge_possible)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from seams.acceptance import lim_z_m, margin_m, seat_inside  # noqa: E402


def test_band_is_the_graphs_plus_two_sigma():
    assert abs(lim_z_m('base1') - 0.002) < 1e-12 and lim_z_m('no_such_base') is None


def test_ends_judged_against_the_centre():
    """10-23: a bar's ends are judged against its own measured centre - depth lifts the whole bar (seated: ends within
    1.63 mm of the centre over 40 readings), a wedge lifts one end (3.9-4.5 mm). attempt 24's +2.0 / -0.56 at a +0.72
    centre is a seated bar (both ends 1.28 mm off it); run 48's +2.39 end at +1.26 too."""
    assert seat_inside('base1', 'line_green', 0.00072, [2.0, -0.56], margin=margin_m())[0]
    assert seat_inside('base1', 'line_green', 0.00126, [2.39, 0.1])[0]


def test_a_wedged_end_still_fails():
    """one end lifted relative to the rest: an end 3.0 mm over a centre still inside the band is refused."""
    ok, why = seat_inside('base1', 'line_green', 0.0015, [4.5, -1.2])
    assert not ok and 'end' in why


def test_ends_judged_only_where_the_part_can_wedge():
    """u_orange cannot wedge looking seated (CAD): its noisy end heights are reported, not judged."""
    assert seat_inside('base1', 'u_orange', 0.0005, [2.4, -1.8], margin=margin_m())[0]
    assert not seat_inside('base1', 'u_orange', 0.0026, [], margin=0.0)[0]


def test_tap_ends_judged_against_the_seat():
    """run 52: taps read the ends -1.40 / -1.45 mm (true heights), depth read the centre +1.20 (the view's bias) - a
    seated bar; contact ends are judged on the band itself, not against the biased centre."""
    assert seat_inside('base1', 'line_green', 0.0012, [-1.40, -1.45], ends_by_contact=True)[0]
    ok, why = seat_inside('base1', 'line_green', 0.0012, [-1.40, 4.5], ends_by_contact=True)
    assert not ok and 'end' in why
