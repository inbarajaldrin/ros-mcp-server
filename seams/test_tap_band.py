import numpy as np

from seams.acceptance import lim_z_m, margin_m
from seams.tap_probe import Point, Tapper


def _tapper(errs_mm):
    lim = lim_z_m('base1')
    return Tapper([Point(n, np.zeros(3), err_m=e / 1000.0) for n, e in errs_mm.items()], 0.2, lim - margin_m(), low_m=lim)


def test_low_points_down_to_the_certifier_band_are_seated():
    """run 52: a level bar tapped -1.38..-1.62 mm - the pocket floor stops it, a press cannot lower it."""
    assert _tapper({'mid': -1.38, 'end-': -1.39, 'end+': -1.62, 'edge-': -1.39, 'edge+': -1.42}).seated()


def test_high_side_keeps_the_recovery_margin():
    assert not _tapper({'mid': -1.38, 'end-': -1.39, 'end+': 1.62, 'edge-': -1.39, 'edge+': -1.42}).seated()


def test_under_the_certifier_band_is_not_seated():
    assert not _tapper({'mid': -1.38, 'end-': -1.39, 'end+': -2.3, 'edge-': -1.39, 'edge+': -1.42}).seated()
