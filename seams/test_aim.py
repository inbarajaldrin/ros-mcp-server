"""seams.aim keeps the replay's aim (2026-10-22, AR1): for every part and every record state, aim_for_insert gives the
pose replay_real_assembly.build_argv sent as --final-base-pos / --final-base-orientation before the move (golden values
in testdata/aim_golden.json, taken from build_argv at 5b94f9e). Offline (CAD meshes only).

    python3 -m pytest -q seams/test_aim.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'ablations')]

from seams import aim as A  # noqa: E402

BASE1 = ((0.1463903944510883, -0.5188579307381027, -0.0625), (0.0, 0.0, 0.19213076781321956, 0.9813693331562295))
PARTS = ('u_brown', 'u_orange', 'line_green', 'inverted_u_yellow')
NOW = 1792710000.0

OFFSETS = {'u_brown': {'seat_minus_pred_mm': [0.14, 0.01, 1.47], 't': 'x'},
           'u_orange': {'seat_minus_pred_mm': [-2.3, 1.6, 0.4], 't': 'x'}}
LEDGER_CAM = {'u_brown': {'verified': True, 't': NOW - 60, 'rel_err_mm': [0.8, -0.3]},
              'u_orange': {'verified': True, 't': NOW - 60, 'rel_err_mm': [-1.4, 0.6]},
              'line_green': {'verified': True, 't': NOW - 60, 'rel_err_mm': [1.2, -0.7]}}
LEDGER_OLD = {k: dict(v, t=NOW - 9000) for k, v in LEDGER_CAM.items()}
HISTORY = {'u_brown': [[-1.0, 0.5], [-2.0, 1.0], [-1.5, 0.2], [9.0, -9.0]],
           'u_orange': [[0.4, -3.0], [1.0, -2.0]],
           'inverted_u_yellow': [[0.3, 0.1], [0.2, 0.2], [0.4, 0.0]]}

RECORDS = {'none': ({}, {}, {}), 'offsets': (OFFSETS, {}, {}), 'camera': (OFFSETS, LEDGER_CAM, {}),
           'stale': (OFFSETS, LEDGER_OLD, HISTORY), 'all': (OFFSETS, LEDGER_CAM, HISTORY)}
SWITCHES = [dict(), dict(SEAMS_DRIVE='1'), dict(FOLLOW_YAW='0'), dict(NO_AIM_BIAS='1'),
            dict(INSERT_YAW_OFFSET_DEG='1.5'), dict(SEAMS_DRIVE='1', FOLLOW_YAW='0', INSERT_YAW_OFFSET_DEG='-2')]


GOLDEN = json.load(open(REPO / 'seams/testdata/aim_golden.json'))['cases']


@pytest.mark.parametrize('obj,rec,sw', list(itertools.product(PARTS, RECORDS, range(len(SWITCHES)))))
def test_same_aim_as_build_argv(obj, rec, sw):
    """GOLDEN = build_argv before seams.aim (5b94f9e) on the same records and switches."""
    g = GOLDEN[f"{obj}|{rec}|{sw}"]
    off, led, hist = RECORDS[rec]
    f = A.load_facts(env=SWITCHES[sw], now=NOW)
    f = A.AimFacts(seated_offsets=off, ledger=led, seat_history=hist, one_node=f.one_node, no_aim_bias=f.no_aim_bias,
                   follow_yaw=f.follow_yaw, yaw_exp_deg=f.yaw_exp_deg, now=NOW)
    a = A.aim_for_insert(obj, 'base1', BASE1, f)
    assert np.allclose(a.argv_pos(), g['pos'], atol=2e-6), (a.argv_pos(), g['pos'], a.corrections)   # argv had 6 decimals
    assert np.allclose(a.quat, g['quat'], atol=2e-6) or np.allclose(a.quat, -np.array(g['quat']), atol=2e-6)
    assert np.allclose(a.shift('bias'), g['bias'], atol=1e-12)


def test_measured_base_unchanged_without_records():
    a = A.aim_for_insert('u_brown', 'base1', BASE1, A.AimFacts(now=NOW))
    assert a.corrections == () and a.pos == BASE1[0] and np.allclose(a.quat, BASE1[1])


def test_follow_needs_both_u_parts():
    f = A.AimFacts(seated_offsets={'u_brown': OFFSETS['u_brown']}, one_node=True, now=NOW)
    a = A.aim_for_insert('line_green', 'base1', BASE1, f)
    assert a.corrections == () and any('not all recorded' in n for n in a.notes)


def test_one_node_follows_the_inserts_not_the_camera():
    f = A.AimFacts(seated_offsets=OFFSETS, ledger=LEDGER_CAM, one_node=True, now=NOW)
    c = A.aim_for_insert('line_green', 'base1', BASE1, f).corrections[0]
    assert c.name == 'follow' and 'from the inserts' in c.detail


@pytest.mark.parametrize('obj,rec,sw', [(o, r, s) for o in PARTS for r in ('none', 'camera', 'all') for s in (0, 1, 4)])
def test_replay_sends_the_aim(obj, rec, sw, tmp_path, monkeypatch):
    """The replay's build_argv (wired to seams.aim) still sends the golden pose, and keeps the Aim for drive-mate."""
    sys.argv = ['replay_real_assembly']
    import replay_real_assembly as rr
    off, led, hist = RECORDS[rec]
    import seams.seated_parts as SP
    for name, data in (('seated_offsets', off), ('assembly_state', led), ('seat_offset_history', hist)):
        (tmp_path / f'{name}.json').write_text(json.dumps(data))
    monkeypatch.setattr(SP, 'LOGS', tmp_path)
    for k in ('SEAMS_DRIVE', 'FOLLOW_YAW', 'NO_AIM_BIAS', 'INSERT_YAW_OFFSET_DEG'):
        monkeypatch.delenv(k, raising=False)
    for k, v in SWITCHES[sw].items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(A.time, 'time', lambda: NOW)
    st = rr.AssemblyState()
    st.held_quat, st.last_grasp_id = (0.0, 0.0, 0.0, 1.0), 1
    argv, _ = rr.build_argv('translate_object', dict(action='insert', object_name=obj, base_name='base1'), st,
                            observed_base=BASE1)
    i = argv.index('--final-base-pos')
    g = GOLDEN[f"{obj}|{rec}|{sw}"]
    assert np.allclose([float(v) for v in argv[i + 1:i + 4]], g['pos'], atol=2e-6)
    assert st.aim is not None and st.aim.obj == obj
    assert np.allclose(np.array(st.aim.pos) + [0, 0, st.aim.seat_dz_m], g['pos'], atol=2e-6)


@pytest.mark.parametrize('obj,rec,sw', [(o, r, s) for o in PARTS for r in RECORDS for s in range(len(SWITCHES))])
def test_corrections_reapplied_on_a_servoed_base(obj, rec, sw):
    """drive-mate applies the planner's corrections (made on the measured base) to the base its servo arrived at: the
    same aim as aim_for_insert run on that servoed base (what the replay did before AR3)."""
    off, led, hist = RECORDS[rec]
    f0 = A.load_facts(env=SWITCHES[sw], now=NOW)
    f = A.AimFacts(seated_offsets=off, ledger=led, seat_history=hist, one_node=f0.one_node, no_aim_bias=f0.no_aim_bias,
                   follow_yaw=f0.follow_yaw, yaw_exp_deg=f0.yaw_exp_deg, now=NOW)
    from scipy.spatial.transform import Rotation as R
    for dx, dy, dyaw in ((0.0, 0.0, 0.0), (-0.0089, 0.0009, -3.48), (0.0021, -0.0047, 2.1)):
        servoed = ((BASE1[0][0] + dx, BASE1[0][1] + dy, BASE1[0][2]),
                   tuple((R.from_euler('z', dyaw, degrees=True) * R.from_quat(BASE1[1])).as_quat()))
        want = A.aim_for_insert(obj, 'base1', servoed, f)
        made = A.aim_for_insert(obj, 'base1', BASE1, f)
        seat = A.Cad().seat_pos('base1', obj)
        got = A.apply_corrections(servoed, json.loads(A.corrections_json(made)), seat_in_base=seat)
        # line_green's follow from the inserts projects WORLD seat offsets onto the base axes (and its slot yaw from
        # them): on a base turned by the servo that projection differs - <= 0.08 mm / 0.10 deg here (servo turns <= 4 deg,
        # the cap), inside the U corridor's +-1.1 mm. drive-mate cannot redo it (no seat records behind the seam)
        assert np.allclose(got[0], want.pos, atol=1e-4), (got[0], want.pos)
        assert np.degrees((R.from_quat(got[1]) * R.from_quat(want.quat).inv()).magnitude()) < 0.15
