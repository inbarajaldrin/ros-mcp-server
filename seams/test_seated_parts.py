"""seams.seated_parts (2026-10-22, AR4): merged per-part writes, the seat error kept across verdicts, atomic under
concurrent writers. Offline.

    python3 -m pytest -q seams/test_seated_parts.py
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from seams.seated_parts import SeatedParts  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_a_verdict_keeps_the_seat_error(tmp_path):
    """Attempt 21: verify_seated replaced line_green's entry and dropped its seat error; the yellow aimed at CAD."""
    c = Clock(); S = SeatedParts(tmp_path, c)
    S.set_seat_error('line_green', [1.7, -0.4], 'depth seat check')
    c.t = 1010
    S.mark('line_green', True, how='marker xy 0.8 mm')
    c.t = 1020
    S.mark('line_green', True, how='execute_mate', depth_mm=0.78)
    e = S.entry('line_green')
    assert e['rel_err_mm'] == [1.7, -0.4] and e['rel_err_t'] == 1000 and e['rel_err_src'] == 'depth seat check'
    assert e['how'] == 'execute_mate' and e['t'] == 1020 and e['depth_mm'] == 0.78


def test_a_failed_read_is_not_a_new_verification(tmp_path):
    c = Clock(); S = SeatedParts(tmp_path, c)
    S.mark('u_brown', True, how='execute_mate')
    c.t = 2000
    S.mark('u_brown', False)
    e = S.entry('u_brown')
    assert e['verified'] is False and e['t'] == 1000 and e['how'] == 'execute_mate'


def test_seat_offsets_history_and_forget(tmp_path):
    S = SeatedParts(tmp_path, Clock())
    for i in range(25):
        S.record_seat_offset('u_orange', [i, -i, 0.5])
    assert S.seated_offsets()['u_orange']['seat_minus_pred_mm'] == [24, -24, 0.5]
    assert len(S.seat_history()['u_orange']) == 20 and S.seat_history()['u_orange'][-1] == [24, -24]
    S.mark('u_orange', True, how='x')
    S.forget('u_orange')
    assert 'u_orange' not in S.ledger() and 'u_orange' not in S.seated_offsets()
    assert len(S.seat_history()['u_orange']) == 20          # the prior stays


def _writer(logs, part, n):
    S = SeatedParts(Path(logs))
    for i in range(n):
        S.set_seat_error(part, [i, i], 'test')
        S.mark(part, True, how=f'w{i}')


def test_concurrent_writers_lose_nothing(tmp_path):
    """Two processes writing different parts: each keeps its own entry (load-modify-dump lost them before)."""
    ps = [mp.Process(target=_writer, args=(str(tmp_path), p, 40)) for p in ('a', 'b', 'c')]
    [p.start() for p in ps]; [p.join() for p in ps]
    led = json.load(open(tmp_path / 'assembly_state.json'))
    assert {k: (v['how'], v['rel_err_mm']) for k, v in led.items()} == {p: ('w39', [39, 39]) for p in 'abc'}
