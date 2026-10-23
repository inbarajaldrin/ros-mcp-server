"""seams.acceptance: one seat rule for ExecuteMate and the planner's recovery (2026-10-22). Offline (CAD for wedge_possible)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from seams.acceptance import lim_z_m, margin_m, seat_inside  # noqa: E402


def test_band_is_the_graphs_plus_two_sigma():
    assert abs(lim_z_m('base1') - 0.002) < 1e-12 and lim_z_m('no_such_base') is None


def test_attempt_24_line_green_keeps_recovering():
    """end +2.0 mm: inside the certifier's band (by 0), not inside the recovery's (one sigma in) - one more tap."""
    assert seat_inside('base1', 'line_green', 0.00072, [2.0, -0.56])[0]
    ok, why = seat_inside('base1', 'line_green', 0.00072, [2.0, -0.56], margin=margin_m())
    assert not ok and 'end' in why


def test_ends_judged_only_where_the_part_can_wedge():
    """u_orange cannot wedge looking seated (CAD): its noisy end heights are reported, not judged."""
    assert seat_inside('base1', 'u_orange', 0.0005, [2.4, -1.8], margin=margin_m())[0]
    assert not seat_inside('base1', 'u_orange', 0.0026, [], margin=0.0)[0]
