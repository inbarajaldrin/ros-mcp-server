"""seams.servo_law reproduces the hover servo (2026-10-22, AR3 step 1): every session in the replay logs of 10-21/22
(testdata/servo_sessions.json, parsed by seams.servo_log) is replayed through the law - the same wrist turns, the same
stops, and the logged final 'aim moved by' (sessions since 1e2803c, 10-22 06:00, when the law took its present form) within the logs' 0.1 mm rounding (amplified by the learned gain). Offline.

    python3 -m pytest -q seams/test_servo_law.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from seams import servo_law as L  # noqa: E402

L.TURN_AND_STEP = False          # the logs were made by the law without it
L.GAIN_MATRIX = False            # ...and with the scalar clear-height gain
L.SERVO_MAX_YAW_DEG = 4.0        # ...and with the 4 deg yaw cap (6 since run 28)

import re  # noqa: E402

ALL = json.load(open(REPO / 'seams/testdata/servo_sessions.json'))
# the law as it stands dates from 1e2803c (10-22 06:00, the clear-height bias); earlier sessions ran earlier laws (the
# 10-21 ones had no bias, no gain learning, no far-edge rule) and are kept only as data for trying new laws
LAW_SINCE = '20261022_0600'


def _stamp(src):
    m = re.search(r'(\d{8}_\d{6})', src)
    return m.group(1) if m else ''


SESSIONS = [s for s in ALL if _stamp(s['source']) >= LAW_SINCE]


def replay(sess):
    """Feed the logged readings to the law; return (aim or stop, turns made, mismatches)."""
    law = L.HoverServoLaw(np.zeros(2))
    far, turns, bad = None, [], []
    ev = list(sess['events'])
    i = 0
    while i < len(ev):
        e = ev[i]; i += 1
        if e['kind'] == 'clear':
            law.measuring_from_clear()
        elif e['kind'] == 'low':
            law.measuring_low()
        elif e['kind'] == 'far':
            far = e['far_mm']
        elif e['kind'] == 'stop':
            if e['k'] != law.k:
                bad.append(f"stop logged at round {e['k']}, law at {law.k}")
            break
        elif e['kind'] == 'part':
            if e['k'] != law.k:
                bad.append(f"round {e['k']} logged, law at {law.k}")
            info = dict(spread_px=e['spread'], yaw_img_deg=e['yaw'], d_far_only_mm=far)
            far = None
            act = law.reading(np.array(e['d_mm']) / 1000.0, info)
            if isinstance(act, L.Turn):
                nxt = ev[i] if i < len(ev) else None
                if nxt is None or nxt['kind'] != 'turn':
                    bad.append(f"law turns {act.deg:+.2f} at round {law.k}, log does not")
                    break
                if abs(nxt['deg'] - act.deg) > 0.011:
                    bad.append(f"turn {act.deg:+.2f} vs logged {nxt['deg']:+.2f}")
                i += 1
                turns.append(act.deg)
                law.turned(act.deg); law.next_round()
            elif isinstance(act, L.Step):
                law.next_round()
            else:                                   # Converged / Stop: the loop ends
                if i < len(ev) and ev[i]['kind'] in ('part', 'turn'):
                    bad.append(f"law ended ({act}) at round {law.k}, log goes on")
                break
        elif e['kind'] == 'turn':
            bad.append('turn logged without a reading before it')
    return law.finish(), turns, bad


@pytest.mark.parametrize('sess', SESSIONS, ids=[f"{s['source'][7:22]}-{s['obj']}-{j}" for j, s in enumerate(SESSIONS)])
def test_law_reproduces_the_logged_servo(sess):
    out, turns, bad = replay(sess)
    assert not bad, bad
    assert isinstance(out, L.Aim)
    # readings are logged to 0.1 mm; a learned gain of 0.4 amplifies that rounding up to 2.5x
    assert np.allclose(np.array(out.dxy) * 1000, sess['aim'], atol=0.35), (np.round(np.array(out.dxy) * 1000, 2), sess['aim'])


def test_turn_and_step_steps_with_the_turn(monkeypatch):
    monkeypatch.setattr(L, 'TURN_AND_STEP', True)
    law = L.HoverServoLaw(np.zeros(2))
    act = law.reading(np.array([0.0075, 0.0005]), dict(spread_px=[2, 1], yaw_img_deg=-1.4))
    assert isinstance(act, L.Turn) and np.allclose(act.xy, [-0.0075, -0.0005])
    law.turned(act.deg); law.next_round()
    act2 = law.reading(np.array([0.0004, 0.0001]), dict(spread_px=[2, 1], yaw_img_deg=-0.1))
    assert isinstance(act2, L.Converged)
    out = law.finish()
    assert np.allclose(out.dxy, np.array([-0.0075, -0.0005]) + L.SERVO_BIAS_M)


def test_matrix_law_converges_from_clear(monkeypatch):
    """The 2x2 clear-height law on its own model (seat at the origin, part held off by e0): the reading is
    G @ (xy - pocket) + turn effect; it ends within 0.5 mm of the pocket in <= 3 readings for the logged first readings."""
    monkeypatch.setattr(L, 'GAIN_MATRIX', True)
    monkeypatch.setattr(L, 'TURN_AND_STEP', True)
    import numpy as np
    G = L.SERVO_G_CLEAR
    for pocket_mm, yaw0 in [((1.2, -2.0), -3.3), ((-3.0, 1.0), 0.0), ((2.5, 2.5), 1.5), ((-0.4, -0.3), 0.0)]:
        pocket = np.array(pocket_mm) / 1000.0
        law = L.HoverServoLaw((0.0, 0.0))
        law.measuring_from_clear(L.SERVO_G_YAW_DEG)
        yaw_left, n = yaw0, 0
        while True:
            d = G @ (law.xy - pocket)
            n += 1
            act = law.reading(d, {'yaw_img_deg': yaw_left, 'spread_px': [2.0, 1.0]})
            if isinstance(act, L.Converged):
                break
            assert not isinstance(act, L.Stop), act
            if isinstance(act, L.Turn):
                law.turned(act.deg); yaw_left -= act.deg
                pocket = pocket - L.SERVO_TURN_READ_CLEAR * act.deg @ np.linalg.inv(G).T   # the turn moves the reading
            law.next_round()
        out = law.finish()
        assert isinstance(out, L.Aim)
        assert n <= 3, (pocket_mm, n)
        assert np.hypot(*(np.array(out.dxy) - L.SERVO_BIAS_CLEAR_G_M - pocket)) < 0.0005
