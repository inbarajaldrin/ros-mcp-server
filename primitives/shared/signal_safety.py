"""Safe-state cleanup that a stop signal cannot cut short (CLAUDE.md: SIGTERM cleanup must reach a safe-state exit,
even mid-force-mode; review r4 P1-1).

Under goal_guard.init_ros every SIGINT/SIGTERM raises KeyboardInterrupt wherever the main thread is.  A second
signal arriving during a cleanup (the caller's group SIGTERM, a forwarded SIGINT) would abort it halfway - inside a
subprocess.run, which SIGKILLs its CLI child.  deferred_signals() records such signals instead, for the duration of
the cleanup; the caller then exits as it was going to.  Processes the cleanup starts should run in their own session
(start_new_session=True) so the group's signals do not reach them either.
"""
from __future__ import annotations

import contextlib
import signal
import sys
import threading

STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


@contextlib.contextmanager
def deferred_signals(signals=STOP_SIGNALS):
    """Record SIGINT/SIGTERM instead of raising while the block runs (main thread only).  Yields the list of
    signal numbers received; the previous handlers are restored afterwards, nothing is re-raised."""
    received: list[int] = []
    if threading.current_thread() is not threading.main_thread():
        yield received
        return

    def record(signum, frame):
        received.append(signum)
        print(f"[safe-state] signal {signum} deferred: the cleanup finishes first", file=sys.stderr, flush=True)

    previous = {s: signal.signal(s, record) for s in signals}
    try:
        yield received
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


import ctypes as _ctypes

_LIBC = _ctypes.CDLL("libc.so.6", use_errno=True)   # loaded here: a preexec_fn must not import (fork + lock)
_PR_SET_PDEATHSIG = 1


def die_with_parent():
    """preexec_fn: the child gets SIGKILL when its parent dies (Linux PR_SET_PDEATHSIG), so an orphaned CLI call
    (`ros2 control` waits forever for the controller manager) cannot apply a stale switch later (review r5 P3-2)."""
    _LIBC.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL))
