"""The cell's Grip adapter on prompt2assembly's current contract (Grip.action at robot-integration 3b9eeaf).

`holding` is a witnessed hold of `expect`; on a rig the witness is the measured width against the declared span.
Pure: seams.grip_adapter.judge() from the readings (no ROS, no gripper)."""
from pathlib import Path

import tests.ros_stubs as ros_stubs

ros_stubs.install()
from seams.grip_adapter import SPAN_TOL_M, judge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_the_action_is_prompt2assemblys_current_one():
    text = (ROOT / "ros2_pkgs/pa_msgs/pa_gripper_msgs/action/Grip.action").read_text()
    assert "string not_holding_reason" in text and "string not_holding_detail" in text
    assert "reached | holding | not_holding | slip | refused | unreachable" in text


def test_a_close_that_stalls_at_the_declared_span_is_a_hold():
    j = judge("64.0", 0.064, 0.086, 0.0701, True, "u_brown", 0.070, 0)
    assert j["holding"] and j["not_holding_reason"] == "" and j["verdict"] == "holding"
    assert j["identity_verified"] and j["identity_authority"] == "width_vs_span"


def test_a_stall_off_the_declared_span_is_not_a_hold():
    j = judge("64.0", 0.064, 0.086, 0.070 + SPAN_TOL_M + 0.002, True, "u_brown", 0.070, 0)
    assert not j["holding"] and j["not_holding_reason"] == "width_off_band" and j["verdict"] == "not_holding"
    assert "from the declared span" in j["not_holding_detail"]


def test_a_close_that_does_not_stall_is_empty():
    j = judge("close", 0.0, 0.086, 0.0005, False, "u_brown", 0.070, 0)
    assert not j["holding"] and j["not_holding_reason"] == "empty"


def test_a_stall_with_no_declared_span_is_unknown_never_a_hold():
    for expect, span in (("u_brown", 0.0), ("none", 0.070)):
        j = judge("64.0", 0.064, 0.086, 0.0701, True, expect, span, 0)
        assert not j["holding"] and j["not_holding_reason"] == "unknown" and j["identity_authority"] in ("none", "width_vs_span")


def test_an_open_is_not_a_close_even_when_it_overshoots():
    """10-21: a release to 35 mm read 37.1 and the old adapter called it 'holding'."""
    j = judge("35.0", 0.035, 0.0300, 0.0371, False, "none", 0.0, 0)
    assert not j["holding"] and j["not_holding_reason"] == "not_a_close"
    j = judge("open", 0.100, 0.070, 0.0995, False, "none", 0.0, 0)
    assert not j["holding"] and j["not_holding_reason"] == "not_a_close" and j["verdict"] == "reached"


def test_no_width_reading_is_unknown_and_a_refused_goal_is_not_a_close():
    j = judge("close", 0.0, 0.086, None, True, "u_brown", 0.070, 0)
    assert not j["holding"] and j["not_holding_reason"] == "unknown"
    j = judge("64.0", 0.064, None, 0.086, False, "u_brown", 0.070, 2)
    assert not j["holding"] and j["not_holding_reason"] == "not_a_close" and j["verdict"] == "refused"


def test_holding_and_its_reason_are_exclusive():
    cases = [("64.0", 0.064, 0.086, w, f, e, s, 0) for w in (None, 0.0005, 0.064, 0.0701, 0.080)
             for f in (False, True) for e in ("u_brown", "none") for s in (0.0, 0.070)]
    for c in cases:
        j = judge(*c)
        assert j["holding"] == (j["not_holding_reason"] == ""), c


def test_a_failed_close_is_never_a_hold():
    """Review w8-realio r1 P2-1: control_gripper exits 1 on 'fingers asymmetric' (one pad on the part); even at a
    width in the band and stalled, that is not a hold."""
    j = judge("64.0", 0.064, 0.086, 0.0701, True, "u_brown", 0.070, 1)
    assert not j["holding"] and j["not_holding_reason"] == "unknown" and "exited 1" in j["not_holding_detail"]


def test_stop_run_stops_the_board_port_runner():
    text = (ROOT / "compliant_insertion_studio/scripts/stop_run.sh").read_text()
    import re
    for pat_name in ("REPLAY_PAT", "ASSEMBLY_PAT"):
        line = next(l for l in text.splitlines() if l.startswith(pat_name + "="))
        pat = line.split("=", 1)[1].strip().strip('"').replace("${A}", "ablations").replace("${S}", "seams") \
            .replace("${M}", "primitives").replace("${B}", "boardport")
        assert re.search(pat, "/usr/bin/python3 -u /home/x/fmb-cell-lab/planners/boardport/run_board.py --real")
