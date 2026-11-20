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


def test_lights_on_scales_exposure_down_then_returns_to_auto():
    state, exposure, out = run('manual', K.EXP_MAX, [254] * 12)
    assert state == 'auto'
    assert {'rgb_camera.enable_auto_exposure': True, 'rgb_camera.gain': K.GAIN_LIT} in [p for _, p in out]
    assert exposure == K.EXP_MIN


def test_manual_in_band_is_left_alone():
    state, exposure, out = run('manual', K.EXP_MAX, [51, 60, 120, 169])
    assert state == 'manual' and exposure == K.EXP_MAX and all(not p for _, p in out)


def test_dim_but_not_black_raises_exposure_within_cap():
    state, exposure, _ = run('manual', 2000, [30])
    assert state == 'manual' and exposure == 6000
