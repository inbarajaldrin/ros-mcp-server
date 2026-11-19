"""Run a child primitive and stop it the CLAUDE.md way: SIGINT (its goal_guard cancels and confirms its arm goal),
then SIGTERM after a grace, never SIGKILL (subprocess.run SIGKILLs a child 0.25 s after the caller is interrupted)."""
from __future__ import annotations

import signal
import subprocess


def stop_child(proc, grace_s, already_signalled=False):
    """SIGINT (unless the child already has it), wait grace_s; SIGTERM, wait grace_s.  Returns the return code
    (None if it is still alive).  A child in the caller's process group already got the group's SIGINT: a second
    one would land inside its goal_guard cancel (review r4 P2-1), so it only gets its grace, then SIGTERM."""
    for sig in ((signal.SIGTERM,) if already_signalled else (signal.SIGINT, signal.SIGTERM)):
        if already_signalled and sig == signal.SIGTERM:
            try:
                return proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                pass
        if proc.poll() is not None:
            return proc.returncode
        proc.send_signal(sig)
        try:
            return proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            continue
    return proc.poll()


def run_child(cmd, timeout_s, grace_s=None):
    """(returncode, stdout, stderr, timed_out).  On a timeout or a KeyboardInterrupt the child is stopped with
    stop_child; a KeyboardInterrupt is re-raised after the child has ended (or the graces ran out)."""
    if grace_s is None:
        from primitives.shared.config import CHILD_STOP_GRACE_S
        grace_s = CHILD_STOP_GRACE_S
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
        return proc.returncode, out, err, False
    except subprocess.TimeoutExpired:
        rc = stop_child(proc, grace_s)
        out, err = proc.communicate() if rc is not None else ("", "")
        return rc, out, err, True
    except KeyboardInterrupt:
        stop_child(proc, grace_s, already_signalled=True)   # the child is in our group: it has the signal too
        raise
