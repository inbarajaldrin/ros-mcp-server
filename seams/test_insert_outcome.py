"""seams.insert_outcome judges an insert as drive-mate did (2026-10-22, AR2): 22 recorded inserts (touch, prismatic
stash, compliant FSM; seat / wedged / above / below / unsettled / no_trace), each at its trace end and at four resting
TCPs, against what drive_mate_server.outcome_from_meta + judge_at_rest gave at 2c5d602 (testdata/insert_outcomes.json).
The same comparison over the last 400 recorded inserts (1984 judgements) had 0 differences. Offline.

    python3 -m pytest -q seams/test_insert_outcome.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from seams import insert_outcome as O  # noqa: E402

CASES = json.load(open(REPO / 'seams/testdata/insert_outcomes.json'))['cases']


@pytest.mark.parametrize('case', CASES, ids=[c['name'] for c in CASES])
def test_same_judgement_as_drive_mate(case):
    for e in case['expect']:
        rest = None if e['rest'] is None else (np.array(e['rest'][0]), e['rest'][1])
        o = O.judge(case['meta'], case['rows'], rest).as_dict()
        assert o['stage_reached'] == e['stage_reached']
        assert bool(o['seated']) == bool(e['seated'])
        for k in ('slide', 'measured_correction', 'seat_xyz'):
            assert (o[k] is None) == (e.get(k) is None), k
            if e.get(k) is not None:
                assert np.allclose(o[k], e[k], atol=1e-12), k
        assert o['trace_stage'] == e.get('trace_stage')


def _write(tmp, name, meta, rows):
    mp = tmp / f'{name}.meta.json'
    mp.write_text(json.dumps(meta))
    with open(tmp / f'{name}.csv', 'w') as fh:
        fh.write('t_s,tcp_x,tcp_y,tcp_z\n')
        for i, r in enumerate(rows):
            fh.write(f'{i},{r[0]},{r[1]},{r[2]}\n')
    return str(mp)


def test_the_named_trace_wins_over_a_newer_one(tmp_path, monkeypatch):
    """A newer meta of the same part (a concurrent or later insert) must not be judged instead of the one named."""
    monkeypatch.setattr(O, 'LOGS', tmp_path)
    seat = [0.1, -0.5, 0.2]
    mine = _write(tmp_path, 'insert_u_brown_1', dict(predicted_seat_xyz=seat), [seat] * 3)
    other = _write(tmp_path, 'insert_u_brown_2', dict(predicted_seat_xyz=seat), [[0.1, -0.5, 0.23]] * 3)
    os.utime(other, (os.path.getmtime(mine) + 10,) * 2)
    out = f"noise\n__RESULT_JSON__\n{json.dumps({'result': 'success', 'meta': mine})}\n__END_RESULT_JSON__\n"
    o = O.outcome_of(out, 'u_brown', 0.0)
    assert o.found_by == 'result' and o.meta_path == mine and o.stage == 'seat'
    o2 = O.outcome_of('no result block', 'u_brown', 0.0)
    assert o2.found_by == 'mtime' and o2.meta_path == other and o2.stage == 'above'


def test_no_trace_is_said():
    o = O.outcome_of('', 'nothing_by_this_name', 1e12)
    assert o.stage == 'no_trace' and o.found_by == 'none' and not o.seated


def test_empty_trace_judged_at_rest():
    seat = (0.1, -0.5, 0.2)
    assert O.judge(dict(predicted_seat_xyz=seat), [], None).stage == 'no_trace'
    o = O.judge(dict(predicted_seat_xyz=seat), [], (np.array([0.1003, -0.5, 0.2014]), True))
    assert o.stage == 'seat' and o.trace_stage == 'no_trace' and o.judged_on == 'resting TCP'


@pytest.fixture
def rr(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / 'ablations'))
    sys.argv = ['replay_real_assembly']
    import replay_real_assembly as rr_
    from seams.seated_parts import SeatedParts
    monkeypatch.setattr(rr_, 'SEATED', SeatedParts(tmp_path))
    rr_._APPLIED_BIAS.clear()
    return rr_


def test_replay_wedge_from_the_drive_mate_result(rr):
    wedged = dict(stage_reached='wedged', measured_correction=[0.003, -0.004, 0.0052])
    w = rr._wedged_near_seat('line_green', 0.0, wedged)
    assert w == pytest.approx({'dz_mm': 5.2, 'dxy_mm': 5.0})
    assert rr._wedged_near_seat('line_green', 0.0, dict(stage_reached='above', measured_correction=[0, 0, 0.03])) is None


def test_replay_records_the_seat_from_the_drive_mate_result(rr):
    """No log file read: the outcome drive-mate returned is the record (plus the bias the aim applied)."""
    rr._APPLIED_BIAS['u_orange'] = (0.001, -0.002)
    rr.record_seat_offset('u_orange', 1e12, dict(measured_correction=[-0.0007, -0.0005, 0.0002]))
    d = rr.SEATED.seated_offsets()['u_orange']['seat_minus_pred_mm']
    assert d == pytest.approx([0.3, -2.5, 0.2])
    assert rr.SEATED.seat_history()['u_orange'] == [pytest.approx([0.3, -2.5])]


def test_replay_says_when_nothing_recorded(rr, capsys):
    rr.record_seat_offset('u_orange', 1e12, None)
    assert 'NOT recorded' in capsys.readouterr().out and rr.SEATED.seated_offsets() == {}
