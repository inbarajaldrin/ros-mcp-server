"""exposure_keeper.decide: lights off -> manual long exposure; lit again -> back to auto; no flapping on one sample."""
import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / 'compliant_insertion_studio/scripts/exposure_keeper.py'
_s = importlib.util.spec_from_file_location('exposure_keeper', _p)
K = importlib.util.module_from_spec(_s)
_s.loader.exec_module(K)


def run(state, exposure, means):
    streak, out = 0, []
    for m in means:
        state, params, streak = K.decide(state, m, exposure, streak)
        exposure = params.get('rgb_camera.exposure', exposure)
        out.append((state, params))
    return state, exposure, out


def test_lights_off_switches_to_manual_long_exposure_after_hold():
    state, exposure, out = run('auto', 166, [8, 8, 8])
    assert state == 'manual' and exposure == K.EXP_MAX
    assert out[-1][1]['rgb_camera.enable_auto_exposure'] is False and out[-1][1]['rgb_camera.gain'] == K.GAIN_DARK


def test_one_dark_frame_does_not_switch():
    state, _, out = run('auto', 166, [8, 120, 8, 120])
    assert state == 'auto' and all(not p for _, p in out)


def test_lights_on_returns_to_auto_on_the_first_saturated_frame():
    """2026-11-21 14:53: lights back on, the dark setting read 254 / 253 / 243 while the old rule stepped the exposure down."""
    state, exposure, out = run('manual', K.EXP_MAX, [254])
    assert state == 'auto'
    assert out[0][1] == {'rgb_camera.enable_auto_exposure': True, 'rgb_camera.gain': K.GAIN_LIT}


def test_bright_but_not_saturated_manual_still_scales_down():
    state, exposure, out = run('manual', K.EXP_MAX, [200])
    assert state == 'manual' and exposure < K.EXP_MAX


def test_manual_in_band_is_left_alone():
    state, exposure, out = run('manual', K.EXP_MAX, [51, 60, 120, 169])
    assert state == 'manual' and exposure == K.EXP_MAX and all(not p for _, p in out)


def test_dim_but_not_black_raises_exposure_within_cap():
    state, exposure, _ = run('manual', 2000, [30])
    assert state == 'manual' and exposure == 6000


def test_lit_room_at_short_manual_exposure_returns_to_auto():
    """2026-11-20: lights on, the keeper scaled to exposure 156 with mean 154 (in band) and stayed manual with gain 128."""
    state, exposure, out = run('manual', 156, [154, 154, 154])
    assert state == 'auto'
    assert {'rgb_camera.enable_auto_exposure': True, 'rgb_camera.gain': K.GAIN_LIT} in [p for _, p in out]


def test_dark_room_long_exposure_stays_manual():
    state, exposure, out = run('manual', K.EXP_MAX, [60, 60, 60])
    assert state == 'manual' and all(not p for _, p in out)
