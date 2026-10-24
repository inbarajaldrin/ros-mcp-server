"""Each part's seat height, learned from where its own certified inserts ended (UNI, operator 10-23: "from all the inserts
today you already know where it sits").

The insert aims the TCP at the CAD seat + dz. dz used to be a hand table from May (PER_OBJECT_BASE_OFFSET_M: line_green
+5.8 mm, yellow -0.9, the U parts -0.6..-1.2 with the settle); now it is measured:

  every insert records (applied dz, end TCP z - predicted, floor?)   - drive-mate, after the insert
  the certifier marks it ok when the part certified on the FIRST check (no nudge / taps: those mean the insert's end
  pose was not the seated one)                                         - the planner, after ExecuteMate

  learned dz = median over the last N ok records of  applied + end_rel (+ FLOOR_MARGIN when the push ran to its floor)

A push that bottoms on contact (U parts, yellow) ends AT the seat. One that runs to its floor without contact (line_green:
the bar seats by itself and the TCP goes FLOOR_MARGIN under the aim) ends that margin under the aim it was given - adding
the margin back keeps the floor where it has seated the part, so the value cannot walk down run after run.
None until LEARN_MIN agree within LEARN_MAD_M: the caller then aims at the CAD seat (dz 0) - no per-part table.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

FILE = Path(__file__).resolve().parents[1] / 'ablations' / 'logs' / 'seat_z_history.json'
LEARN_N, LEARN_MIN, LEARN_MAD_M = 6, 3, 0.001
FLOOR_MARGIN_M = 0.002          # touch_insert pushes to the aim - 2 mm (guarded): ending there = it never bottomed
FLOOR_AT_M = -0.0015            # an end this far under the aim or more was the floor, not a contact


def _load(path=None):
    try:
        return json.load(open(path or FILE))
    except Exception:
        return {}


def _save(d, path=None):
    p = Path(path or FILE)
    tmp = p.with_suffix('.tmp')
    json.dump(d, open(tmp, 'w'), indent=1)
    os.replace(tmp, p)


def record_insert(obj, applied_dz_m, end_rel_m, seated, path=None, t=None):
    """After an insert: what dz it was aimed with and where the TCP ended vs that aim. ok is decided by the certifier."""
    if end_rel_m is None or not np.isfinite(end_rel_m):
        return
    d = _load(path)
    d.setdefault(obj, []).append({'t': t or time.time(), 'applied_dz_m': float(applied_dz_m), 'end_rel_m': float(end_rel_m),
                                  'floor': bool(end_rel_m <= FLOOR_AT_M), 'seated': bool(seated), 'ok': None})
    d[obj] = d[obj][-50:]
    _save(d, path)


def mark_certified(obj, ok, path=None, max_age_s=900.0):
    """The planner's verdict for this part's latest insert: certified on the first check (True) or not (False)."""
    d = _load(path)
    rows = d.get(obj) or []
    if rows and rows[-1].get('ok') is None and time.time() - float(rows[-1].get('t', 0)) < max_age_s:
        rows[-1]['ok'] = bool(ok)
        _save(d, path)


def seat_of(row):
    """The dz at which this insert's part sat right."""
    return row['applied_dz_m'] + row['end_rel_m'] + (FLOOR_MARGIN_M if row.get('floor') else 0.0)


def learned_dz(obj, path=None):
    """(dz m, n) from the last LEARN_N ok inserts, or (None, n) when fewer than LEARN_MIN or they disagree."""
    rows = [r for r in (_load(path).get(obj) or []) if r.get('ok') and r.get('seated')][-LEARN_N:]
    if len(rows) < LEARN_MIN:
        return None, len(rows)
    v = np.array([seat_of(r) for r in rows])
    med = float(np.median(v))
    if float(np.median(np.abs(v - med))) > LEARN_MAD_M:
        return None, len(rows)
    return med, len(rows)
