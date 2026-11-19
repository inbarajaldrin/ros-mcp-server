"""Review r2: a caller's timeout stops a primitive with SIGINT first (its goal_guard cancels the arm goal), SIGTERM
only after the grace, never SIGKILL; and move_to_clear_area never sends a route its caller's timeout could cut."""
import sys
import textwrap
import types

import pytest

from ablations import cell_io
from primitives.shared import config

CHILD = textwrap.dedent('''
    import signal, sys, time
    log = open(sys.argv[1], "a")
    def on(sig, frame):
        log.write(signal.Signals(sig).name + "\\n"); log.flush()
        if sys.argv[2] == "obey" or sig == signal.SIGTERM:
            sys.exit(0)
    signal.signal(signal.SIGINT, on)
    signal.signal(signal.SIGTERM, on)
    print("started", flush=True)
    while True:
        time.sleep(0.05)
''')


@pytest.fixture
def no_robot(monkeypatch):
    """The safety watcher polls the real robot's dashboard; the test must never reach the network."""
    monkeypatch.setattr(cell_io, "robot_safety_status", lambda *a, **k: "NORMAL")
    monkeypatch.setattr(config, "CHILD_STOP_GRACE_S", 1.5)


@pytest.mark.parametrize("behaviour,expected", [("obey", ["SIGINT"]), ("ignore_int", ["SIGINT", "SIGTERM"])])
def test_a_timed_out_primitive_gets_sigint_first(no_robot, tmp_path, behaviour, expected):
    script, log = tmp_path / "child.py", tmp_path / "signals.log"
    script.write_text(CHILD)
    result = cell_io._run_subprocess_once([sys.executable, str(script), str(log), behaviour], "stop-order test",
                                          timeout_s=1)
    assert result["result"] == "failure" and "timeout" in result["error"]
    assert log.read_text().split() == expected


def test_clear_area_refuses_a_route_its_caller_could_cut():
    import tests.ros_stubs as ros_stubs
    ros_stubs.install()
    import time
    from primitives.core import move_to_clear_area as mca
    node = object.__new__(mca.MoveToClearArea)
    sent = []
    node.action_client = types.SimpleNamespace(send_goal_async=lambda *a, **k: sent.append(a))
    node.get_logger = lambda: types.SimpleNamespace(error=lambda *a: None, info=lambda *a: None)
    node.error_message, node.operation_success, node.operation_complete = None, None, False
    node.trajectory_sent, node.goal_sent_at = False, None

    def goal(seconds):
        stamp = types.SimpleNamespace(sec=int(seconds), nanosec=0)
        return types.SimpleNamespace(trajectory=types.SimpleNamespace(points=[types.SimpleNamespace(time_from_start=stamp)]))

    node.started_at = time.monotonic() - 150.0                     # 150 s of planning already
    node._dispatch(goal(120.0))                                    # 150 + 10 + 130 + 5 > 300 - 20
    assert node.error_message.startswith("ROUTE_OUTLASTS_CALLER_TIMEOUT") and not sent and not node.trajectory_sent
    assert node.operation_complete and node.operation_success is False
    node.error_message, node.operation_complete = None, False
    node.started_at = time.monotonic() - 5.0
    node._send_goal_future = None
    node.goal_response = lambda f: None
    try:
        node._dispatch(goal(20.0))
    except AttributeError:
        pass                                                       # the stand-in future has no callbacks
    assert node.error_message is None and sent
