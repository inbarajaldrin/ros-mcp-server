#!/usr/bin/env python3
"""Seated parts (2026-10-22, AR4): one owner for what the cell knows about parts already in the base.

    S = SeatedParts()                      # files under ablations/logs (a test passes a temp dir)
    S.mark(part, verified=True, how='execute_mate', depth_mm=0.8)    # merges: keeps the recorded seat error
    S.set_seat_error(part, [dx, dy], source='depth seat check')     # base frame, mm, with its own time + source
    S.record_seat_offset(part, [dx, dy, dz])                          # where the insert seated vs the prediction
    S.forget(part)                                                    # pulled out (disassemble): every record of it
    S.ledger(), S.seated_offsets(), S.seat_history()                  # read views (seams.aim.load_facts)

Before this module seven functions in the replay each loaded assembly_state.json, edited it and dumped the whole map
back - some after tens of seconds of arm moves, so a write made in between was lost - and verify_seated replaced a
part's whole entry, dropping its recorded seat error (attempt 21: the yellow then aimed at line_green's CAD seat).
Every operation here re-reads the file under a lock, changes one part, and replaces the file atomically.

Ledger entry (keys kept from before): verified, t (when the verdict was made), how, depth_mm?, rel_err_mm?
(base frame [x, y] mm) + rel_err_t, rel_err_src (new: the seat error's own time and source).
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional, Sequence

LOGS = Path(__file__).resolve().parents[1] / 'ablations/logs'
HISTORY_N = 20


class SeatedParts:
    def __init__(self, logs: Optional[Path] = None, clock: Callable[[], float] = time.time):
        self.logs = Path(logs or LOGS)
        self.clock = clock
        self.ledger_path = self.logs / 'assembly_state.json'          # {part: {verified, t, how, ...}}
        self.offsets_path = self.logs / 'seated_offsets.json'          # {part: {seat_minus_pred_mm, t}}
        self.history_path = self.logs / 'seat_offset_history.json'     # {part: [[dx, dy] mm, ...]}

    # -- reads --------------------------------------------------------------------------------------------------
    @staticmethod
    def _read(p: Path) -> dict:
        try:
            return json.load(open(p))
        except Exception:
            return {}

    def ledger(self) -> dict:
        return self._read(self.ledger_path)

    def seated_offsets(self) -> dict:
        return self._read(self.offsets_path)

    def seat_history(self) -> dict:
        return self._read(self.history_path)

    def entry(self, part: str) -> dict:
        return dict(self.ledger().get(part) or {})

    # -- writes -------------------------------------------------------------------------------------------------
    @contextmanager
    def _edit(self, p: Path):
        """Read-modify-write one file under an exclusive lock; the new content replaces it atomically."""
        self.logs.mkdir(parents=True, exist_ok=True)
        with open(str(p) + '.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = self._read(p)
            yield d
            tmp = Path(str(p) + f'.tmp{os.getpid()}')
            with open(tmp, 'w') as fh:
                json.dump(d, fh, indent=1)
            os.replace(tmp, p)

    def mark(self, part: str, verified: bool, how: Optional[str] = None, **extra) -> dict:
        """A verdict on `part` (merged: other keys, the seat error among them, are kept). verified=False keeps the
        previous time and how (a failed read is not a new verification)."""
        with self._edit(self.ledger_path) as d:
            e = dict(d.get(part) or {})
            e['verified'] = bool(verified)
            if verified:
                e['t'] = self.clock()
                if how is not None:
                    e['how'] = how
            e.update(extra)
            d[part] = e
            return dict(e)

    def set_seat_error(self, part: str, err_mm: Sequence[float], source: str, touch_t: bool = False) -> None:
        """Where `part` really sits vs its seat, base frame [x, y] mm. touch_t: also refresh the entry's t (the depth
        seat check did, before this module; the follow reads did not)."""
        with self._edit(self.ledger_path) as d:
            e = dict(d.get(part) or {})
            e['rel_err_mm'] = [float(v) for v in err_mm]
            e['rel_err_t'] = self.clock()
            e['rel_err_src'] = source
            if touch_t:
                e['t'] = self.clock()
            d[part] = e

    def record_seat_offset(self, part: str, mm: Sequence[float]) -> None:
        """Seat TCP - predicted TCP of `part`'s insert (world mm): the latest + the history the seat bias reads."""
        tot = [float(v) for v in mm]
        with self._edit(self.offsets_path) as d:
            d[part] = dict(seat_minus_pred_mm=tot, t=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.clock())))
        with self._edit(self.history_path) as h:
            h[part] = (list(h.get(part) or []) + [tot[:2]])[-HISTORY_N:]

    def forget(self, part: str, verdict: bool = True, offset: bool = True) -> None:
        """`part` left the base: drop its verdict and its seat offset (the history stays - it is the part's prior)."""
        if verdict:
            with self._edit(self.ledger_path) as d:
                d.pop(part, None)
        if offset:
            with self._edit(self.offsets_path) as d:
                d.pop(part, None)
