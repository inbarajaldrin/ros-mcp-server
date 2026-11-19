"""Run a child primitive and stop it the CLAUDE.md way: SIGINT (its goal_guard cancels and confirms its arm goal),
then SIGTERM after a grace, never SIGKILL (subprocess.run SIGKILLs a child 0.25 s after the caller is interrupted)."""
from __future__ import annotations

import signal
import subprocess


def stop_child(proc, grace_s):
    """SIGINT, wait grace_s; SIGTERM, wait grace_s.  Returns the return code (None if it is still alive)."""
    for sig in (signal.SIGINT, signal.SIGTERM):
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
        stop_child(proc, grace_s)
        raise
