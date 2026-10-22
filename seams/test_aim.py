"""seams.aim keeps the replay's aim (2026-10-22, AR1): for every part and every record state, aim_for_insert gives the
pose replay_real_assembly.build_argv sent as --final-base-pos / --final-base-orientation. Offline (CAD meshes only).

    python3 -m pytest -q seams/test_aim.py
"""
import importlib
import itertools
import json
import os
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


@pytest.fixture(scope='module')
def rr():
    sys.argv = ['replay_real_assembly']
    return importlib.import_module('replay_real_assembly')


def _old_aim(rr, obj, tmp, records, env, monkeypatch):
    """What build_argv sent before seams.aim: run it on the same records (files) and switches (env)."""
    off, led, hist = records
    for name, data in (('SEATED_OFFSETS', off), ('ASSEMBLY_STATE', led), ('SEAT_HISTORY', hist)):
        p = tmp / f'{name}.json'
        p.write_text(json.dumps(data))
        monkeypatch.setattr(rr, name, p)
    for k in ('SEAMS_DRIVE', 'FOLLOW_YAW', 'NO_AIM_BIAS', 'INSERT_YAW_OFFSET_DEG'):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(rr.time, 'time', lambda: NOW)
    rr._APPLIED_BIAS.clear()
    st = rr.AssemblyState()
    st.held_quat, st.last_grasp_id = (0.0, 0.0, 0.0, 1.0), 1
    argv, _ = rr.build_argv('translate_object', dict(action='insert', object_name=obj, base_name='base1'), st,
                            observed_base=BASE1)
    i = argv.index('--final-base-pos'); j = argv.index('--final-base-orientation')
    return [float(v) for v in argv[i + 1:i + 4]], [float(v) for v in argv[j + 1:j + 5]], dict(rr._APPLIED_BIAS)


@pytest.mark.parametrize('obj,rec,sw', list(itertools.product(PARTS, RECORDS, range(len(SWITCHES)))))
def test_same_aim_as_build_argv(rr, obj, rec, sw, tmp_path, monkeypatch):
    env = SWITCHES[sw]
    p_old, q_old, bias_old = _old_aim(rr, obj, tmp_path, RECORDS[rec], env, monkeypatch)
    off, led, hist = RECORDS[rec]
    f = A.load_facts(env=env, now=NOW)
    f = A.AimFacts(seated_offsets=off, ledger=led, seat_history=hist, one_node=f.one_node, no_aim_bias=f.no_aim_bias,
                   follow_yaw=f.follow_yaw, yaw_exp_deg=f.yaw_exp_deg, now=NOW)
    a = A.aim_for_insert(obj, 'base1', BASE1, f)
    # build_argv prints 6 decimals
    assert np.allclose(a.argv_pos(), p_old, atol=2e-6), (a.argv_pos(), p_old, a.corrections)
    assert np.allclose(a.quat, q_old, atol=2e-6) or np.allclose(a.quat, -np.array(q_old), atol=2e-6)
    assert np.allclose(a.shift('bias'), bias_old.get(obj, (0.0, 0.0)), atol=1e-12)


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
