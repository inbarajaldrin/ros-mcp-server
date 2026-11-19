"""Review r4 P1-1: a stopped insert child reaches its safe state (force mode off, controllers restored) whatever
signals follow.  Stand-ins with prismatic's exit shape, built on the real deferred_signals, translate_object's real
_run_child/_settle_child, and cell_io's real _stop_group (dashboard replaced: nothing touches the network)."""
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# prismatic's shape: SIGINT/SIGTERM raise (goal_guard), the finally runs the safe-state cleanup with signals
# deferred, its CLI call (a 2.5 s stand-in for `ros2 control switch_controllers`) in its own session.
CHILD = textwrap.dedent('''
    import signal, subprocess, sys, time
    sys.path.insert(0, {root!r})
    from primitives.shared.signal_safety import deferred_signals
    log = open(sys.argv[1], "a", buffering=1)
    def raise_interrupt(sig, frame):
        log.write("got " + signal.Signals(sig).name + "\\n")
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, raise_interrupt)
    signal.signal(signal.SIGTERM, raise_interrupt)
    try:
        print("started", flush=True)                    # inside the try, as prismatic's switch now is
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        log.write("interrupted\\n")
    finally:
        with deferred_signals() as deferred:
            log.write("stop_force_mode\\n")
            subprocess.run(["sleep", "2.5"], start_new_session=True)
            log.write("controllers restored\\n")
        log.write("deferred " + str(len(deferred)) + "\\n")
''')

# translate_object's shape: the leader runs the child through the real _run_child.
LEADER = textwrap.dedent('''
    import sys
    sys.path.insert(0, {root!r})
    import tests.ros_stubs as s; s.install()
    from primitives import translate_object as t
    t.CHILD_CLEANUP_S, t.CHILD_TERM_GRACE_S = 6.0, 2.0
    try:
        t._run_child([sys.executable, sys.argv[1], sys.argv[2]])
    except KeyboardInterrupt:
        pass
''')


def _spawn(tmp_path, leader=True):
    child, log = tmp_path / "child.py", tmp_path / "child.log"
    child.write_text(CHILD.format(root=str(ROOT)))
    if leader:
        lead = tmp_path / "leader.py"
        lead.write_text(LEADER.format(root=str(ROOT)))
        cmd = [sys.executable, str(lead), str(child), str(log)]
    else:
        cmd = [sys.executable, str(child), str(log)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, start_new_session=True, cwd=ROOT)
    proc.stdout.readline()                                                # started
    return proc, log


def _lines(log):
    return log.read_text().split("\n")


def test_the_restore_completes_under_a_second_sigint(tmp_path):
    proc, log = _spawn(tmp_path, leader=False)
    os.killpg(proc.pid, signal.SIGINT)                                    # the group's stop
    time.sleep(1.0)
    os.kill(proc.pid, signal.SIGINT)                                      # a second one, mid-restore
    assert proc.wait(timeout=15) is not None
    lines = _lines(log)
    assert "controllers restored" in lines and "deferred 1" in lines, lines


def test_the_restore_completes_under_the_outer_sigterm(tmp_path):
    proc, log = _spawn(tmp_path, leader=False)
    os.killpg(proc.pid, signal.SIGINT)
    time.sleep(1.0)
    os.killpg(proc.pid, signal.SIGTERM)                                   # the caller's group SIGTERM, mid-restore
    proc.wait(timeout=15)
    lines = _lines(log)
    assert "controllers restored" in lines and "deferred 1" in lines, lines


def test_the_leader_does_not_forward_a_second_sigint(tmp_path):
    proc, log = _spawn(tmp_path, leader=True)
    os.killpg(proc.pid, signal.SIGINT)                                    # cell_io's group SIGINT
    proc.wait(timeout=20)
    lines = _lines(log)
    assert lines.count("got SIGINT") == 1 and "controllers restored" in lines, lines


def test_stop_group_waits_for_a_grandchild_still_cleaning_up(tmp_path, monkeypatch):
    from ablations import cell_io
    monkeypatch.setattr(cell_io, "dashboard_stop", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no dashboard stop for a group that finished its cleanup")))
    proc, log = _spawn(tmp_path, leader=True)
    t0 = time.monotonic()
    how = cell_io._stop_group(proc, 8.0, "safe-state test")
    assert how == "SIGINT" and time.monotonic() - t0 >= 2.0               # it waited for the 2.5 s restore
    assert "controllers restored" in _lines(log)
