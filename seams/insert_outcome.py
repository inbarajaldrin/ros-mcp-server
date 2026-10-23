#!/usr/bin/env python3
"""Insert outcome (2026-10-22, AR2): what one insert did, judged once, from the trace the insert NAMED.

    o = outcome_of(output_text, obj, since, rest)      # drive-mate, after the insert process exits
    o.stage, o.seated, o.slide, o.correction, o.trace_path, o.found_by

Before this module three readers each found "this insert's" trace by globbing insert_<obj>_*.meta.json for the newest
modification time against their own clock, each handled the two meta schemas (or not: the replay's wedge check read only
predicted_seat_xyz and silently returned None for the compliant FSM's), and each failed silently - the empty touch CSV
(attempt 22) recorded no U seat and line_green aimed at the bare base. Now:

  found_by   'result'  the insert printed its meta path in __RESULT_JSON__ (touch_insert does)
             'mtime'   fallback for routes that do not yet: the newest meta since `since` - said, not hidden
             'none'    no trace: stage no_trace
  stage      seat | wedged (still, 0 < dz <= 7 mm, within 5 mm in xy: on a channel edge) | above | below |
             unsettled (not still) | no_trace - judged at rest when the resting TCP is given, else on the trace's end
"""
from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / 'compliant_insertion_studio' / 'logs'
SEATED_DZ_M = 0.005              # CLAUDE.md: |tcp_z - predicted_seat| < 5 mm and motion stopped
WEDGE_DZ_M, WEDGE_DXY_M = 0.007, 0.005    # stalled <= 7 mm over the seat, <= 5 mm in xy (cycle 3: 3.8-4.4 mm, 5 mm high = on the channel edge)


@dataclass(frozen=True)
class InsertOutcome:
    stage: str
    seated: bool = False
    seat_xyz: Optional[Tuple[float, float, float]] = None     # predicted seat TCP (the meta's, either schema)
    end_xyz: Optional[Tuple[float, float, float]] = None      # where it was judged: resting TCP, else the trace's end
    still: Optional[bool] = None
    slide: Optional[Tuple[float, float]] = None               # end - seat, xy (m)
    correction: Optional[Tuple[float, float, float]] = None   # end - seat (m)
    trace_stage: Optional[str] = None                          # the trace's own reading when judged at rest
    judged_on: str = ''                                        # 'resting TCP' | 'trace end' | ''
    trace_path: str = ''
    meta_path: str = ''
    found_by: str = 'none'
    fsm_outcome: Optional[str] = None
    router: Optional[str] = None

    def as_dict(self):
        """The drive-mate result's keys (stage_reached, measured_correction, ...) plus the rest."""
        d = asdict(self)
        d['stage_reached'] = d.pop('stage')
        d['measured_correction'] = d.pop('correction')
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def stage_of(seat, end, still) -> str:
    d = np.asarray(end, float) - np.asarray(seat, float); dxy = float(np.hypot(d[0], d[1]))
    if not still:
        return 'unsettled'
    if abs(d[2]) < SEATED_DZ_M:
        return 'seat'
    if 0 < d[2] <= WEDGE_DZ_M and dxy <= WEDGE_DXY_M:
        return 'wedged'
    return 'above' if d[2] > 0 else 'below'


def seat_of(meta: dict):
    """The predicted seat TCP: the prismatic stash and touch_insert write predicted_seat_xyz, the compliant FSM
    cad_prediction.predicted_tcp_at_seat.xyz_m."""
    s = meta.get('predicted_seat_xyz') or ((meta.get('cad_prediction') or {}).get('predicted_tcp_at_seat') or {}).get('xyz_m')
    return tuple(float(v) for v in s) if s else None


def judge(meta: dict, trace: Sequence, rest=None, meta_path='', found_by='none') -> InsertOutcome:
    """Pure: the meta, the trace's TCP rows (N x 3, m) and optionally the resting TCP (xyz, still)."""
    trace_path = meta_path.replace('.meta.json', '.csv') if meta_path else ''
    base = dict(trace_path=trace_path, meta_path=meta_path, found_by=found_by,
                fsm_outcome=meta.get('outcome'), router=meta.get('router'))
    seat = seat_of(meta)
    if seat is None:
        return InsertOutcome('no_trace', **base)
    P = np.asarray(trace, float).reshape(-1, 3)[-50:]
    if len(P):
        t_still = float(np.ptp(P[:, 2])) < 0.001
        t_stage = stage_of(seat, P[-1], t_still)
    else:
        # the insert's quickest success writes its meta but no rows: only the resting TCP can judge it
        # (2026-10-21: line_green +2.3 mm, 0.3 mm in xy, reported no_trace)
        t_still, t_stage = None, 'no_trace'
    if rest is not None:
        end, still = np.asarray(rest[0], float), bool(rest[1])
        stage, judged_on, trace_stage = stage_of(seat, end, still), 'resting TCP', t_stage
    elif len(P):
        end, still, stage, judged_on, trace_stage = P[-1], t_still, t_stage, 'trace end', None
    else:
        return InsertOutcome('no_trace', seat_xyz=seat, **base)
    d = end - np.asarray(seat)
    return InsertOutcome(stage, seated=stage == 'seat', seat_xyz=seat, end_xyz=tuple(float(v) for v in end), still=still,
                         slide=(float(d[0]), float(d[1])), correction=tuple(float(v) for v in d),
                         trace_stage=trace_stage, judged_on=judged_on, **base)


def meta_path_from_output(text: str) -> Optional[str]:
    """The meta the insert named in its last __RESULT_JSON__ block ('meta'), if it named one."""
    blocks = text.split('__RESULT_JSON__')[1:]
    for b in reversed(blocks):
        line = b.strip().splitlines()[0] if b.strip() else ''
        try:
            mp = json.loads(line).get('meta')
        except Exception:
            continue
        if mp:
            return str(mp)
    return None


def find_meta(obj: str, since: float, output_text: str = ''):
    """(meta path, found_by)."""
    mp = meta_path_from_output(output_text) if output_text else None
    if mp and os.path.exists(mp):
        return mp, 'result'
    metas = [m for m in glob.glob(str(LOGS / f'insert_{obj}_*.meta.json')) if os.path.getmtime(m) >= since - 1]
    if metas:
        return max(metas, key=os.path.getmtime), 'mtime'
    return None, 'none'


def read_trace(meta_path: str):
    """(meta dict, TCP rows N x 3) - rows empty if the CSV has none."""
    m = json.load(open(meta_path))
    rows = []
    try:
        for x in csv.DictReader(open(meta_path.replace('.meta.json', '.csv'))):
            if x.get('tcp_z') not in ('', 'nan', None):
                rows.append([float(x['tcp_x']), float(x['tcp_y']), float(x['tcp_z'])])
    except FileNotFoundError:
        pass
    return m, rows


def outcome_of(output_text: str, obj: str, since: float, rest=None) -> InsertOutcome:
    mp, how = find_meta(obj, since, output_text)
    if mp is None:
        return InsertOutcome('no_trace', found_by='none')
    m, rows = read_trace(mp)
    return judge(m, rows, rest, meta_path=mp, found_by=how)
