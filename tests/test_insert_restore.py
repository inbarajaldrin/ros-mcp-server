"""An insert leaves the arm commandable or the run stops (lead 2026-11-20, w5b-insert-restore; review r1).

Behaviour, not source text:
- prismatic's cleanup stops force mode first (only when it may be on, with bounded waits), retries the restore, and
  when the restore still fails halts motion and exits EXIT_RESTORE_FAILED;
- translate_object maps exit 3 to a typed INSERT_RESTORE_FAILED;
- cell_io.post_insert_gate latches the run (_DASHBOARD_STOPPED) and the latched run_subprocess never re-sends the
  program (the end-of-run cleanup included);
- replay_object, driven offline, runs NO step after a failed gate, at every insert site (first try, outer try,
  recovery reinsert)."""
import argparse
import sys
import types
from pathlib import Path

import pytest

import tests.ros_stubs as ros_stubs

ros_stubs.install()
from primitives._real_mode_stash import prismatic_peg_insertion as pp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ablations"))
LIST_RESTORED = """joint_state_broadcaster  joint_state_broadcaster/JointStateBroadcaster  active
scaled_joint_trajectory_controller  ur_controllers/ScaledJointTrajectoryController  \x1b[92mactive\x1b[0m
force_mode_controller  ur_controllers/ForceModeController  inactive
passthrough_trajectory_controller  ur_controllers/PassthroughTrajectoryController  inactive"""
# what the URSim rig showed after an induced restore failure on dev (receipts lg-press-fail / yl-press-fail)
LIST_LEFT_SWITCHED = """scaled_joint_trajectory_controller  ur_controllers/ScaledJointTrajectoryController  inactive
force_mode_controller  ur_controllers/ForceModeController  active
passthrough_trajectory_controller  ur_controllers/PassthroughTrajectoryController  active"""


# ---- prismatic's cleanup --------------------------------------------------------------------------------------------
def _cleanup_env(monkeypatch, switch_results, stop_ok=True, maybe_on=True):
    calls = []
    results = list(switch_results)

    def switch(activate, deactivate):
        calls.append(("switch",))
        return results.pop(0) if results else False
    monkeypatch.setattr(pp, "switch_controllers", switch)
    monkeypatch.setattr(pp, "RESTORE_BACKOFF_S", (0.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr(pp.time, "sleep", lambda s: None)
    monkeypatch.setattr(pp, "halt_motion_dashboard", lambda host=None: calls.append(("dashboard_stop",)) or True)
    monkeypatch.setattr(pp.rclpy, "ok", lambda: True, raising=False)
    monkeypatch.setattr(pp.rclpy, "shutdown", lambda: calls.append(("shutdown",)), raising=False)
    import primitives.shared.goal_guard as gg
    monkeypatch.setattr(gg, "cancel_active_goals", lambda *a, **k: calls.append(("cancel_goals",)))
    node = types.SimpleNamespace(force_mode_maybe_on=maybe_on)
    node.stop_force_mode = lambda timeout=5.0: calls.append(("stop_force_mode", timeout)) or stop_ok
    return node, calls


def test_restore_is_retried_with_backoff(monkeypatch):
    node, calls = _cleanup_env(monkeypatch, [False, False, True])
    assert pp.safe_state_cleanup(node) is True
    assert [c[0] for c in calls].count("switch") == 3 and ("dashboard_stop",) not in calls


def test_force_mode_is_stopped_first_and_a_failed_restore_halts_motion(monkeypatch, capsys):
    node, calls = _cleanup_env(monkeypatch, [])                      # every switch refused
    assert pp.safe_state_cleanup(node) is False
    names = [c[0] for c in calls]
    assert names.index("stop_force_mode") < names.index("switch")
    assert names.count("switch") == len(pp.RESTORE_BACKOFF_S)
    assert names.index("dashboard_stop") > max(i for i, n in enumerate(names) if n == "switch")
    assert "INSERT_RESTORE_FAILED" in capsys.readouterr().out


def test_force_mode_stop_is_bounded_and_retried_when_not_acknowledged(monkeypatch):
    node, calls = _cleanup_env(monkeypatch, [True], stop_ok=False)
    pp.safe_state_cleanup(node)
    stops = [c for c in calls if c[0] == "stop_force_mode"]
    assert len(stops) == 3 and all(c[1] == 2.0 for c in stops)


def test_no_force_mode_stop_once_it_was_acknowledged_off(monkeypatch, capsys):
    # review r1 P2-2: the common failure path stops force mode itself; the cleanup does not stop it again
    node, calls = _cleanup_env(monkeypatch, [True], maybe_on=False)
    assert pp.safe_state_cleanup(node) is True
    assert not [c for c in calls if c[0] == "stop_force_mode"]
    assert "not active" in capsys.readouterr().out


def test_a_failed_restore_overrides_the_exit_code(monkeypatch):
    monkeypatch.setattr(pp.sys, "argv", ["prismatic", "--object-name", "line_green", "--base-name", "base1",
                                         "--grasp-id", "1", "--current-object-orientation", "0", "0", "0", "1"])
    monkeypatch.setattr(pp.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(pp, "switch_controllers", lambda a, d: True)
    monkeypatch.setattr(pp.time, "sleep", lambda s: None)
    monkeypatch.setattr(pp, "init_ros", lambda **k: None)
    monkeypatch.setattr(pp, "PrismaticPegInsertion", lambda **k: types.SimpleNamespace(execute=lambda: True))
    monkeypatch.setattr(pp, "safe_state_cleanup", lambda node: False)
    import signal
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    assert pp.main() == pp.EXIT_RESTORE_FAILED == 3


# ---- translate_object ----------------------------------------------------------------------------------------------
def test_translate_object_maps_exit_3_to_a_typed_error():
    from primitives.translate_object import prismatic_insert_result
    a = types.SimpleNamespace(mode="real", object_name="line_green", base_name="base1")
    r3 = prismatic_insert_result(3, a)
    assert r3["result"] == "failure" and r3["error"].startswith("INSERT_RESTORE_FAILED")
    assert prismatic_insert_result(1, a)["error"] == "prismatic_peg_insertion exited 1"
    assert prismatic_insert_result(0, a) == {**r3, "result": "success", "error": None} or \
        "error" not in prismatic_insert_result(0, a)


# ---- cell_io: the gate and the latch -------------------------------------------------------------------------------
@pytest.fixture
def cio(monkeypatch):
    import cell_io
    monkeypatch.setattr(cell_io, "_DASHBOARD_STOPPED", None)
    monkeypatch.setattr(cell_io, "_SAFETY_STOP", None)
    stops = []
    monkeypatch.setattr(cell_io, "dashboard_stop", lambda host=None, timeout=3.0: stops.append(host) or True)
    cell_io._test_stops = stops
    return cell_io


def test_controller_state_check(cio):
    assert cio.controllers_restored(list_text=LIST_RESTORED) == (True, "restored")
    ok, why = cio.controllers_restored(list_text=LIST_LEFT_SWITCHED)
    assert ok is False and "scaled_joint_trajectory_controller inactive" in why and "force_mode_controller active" in why
    ok, why = cio.controllers_restored(list_text="")                 # unreadable: fail closed
    assert ok is False and why.startswith("controller state unreadable")


def test_gate_latches_when_the_controllers_are_left_switched(cio, monkeypatch):
    monkeypatch.setattr(cio, "controllers_restored", lambda: cio.parse_controller_states and
                        (False, "scaled_joint_trajectory_controller inactive"))
    why = cio.post_insert_gate({"result": "success"})
    assert why.startswith("CONTROLLERS_NOT_RESTORED") and cio._DASHBOARD_STOPPED == why and cio._test_stops


def test_gate_latches_on_insert_restore_failed_without_asking_the_controllers(cio, monkeypatch):
    monkeypatch.setattr(cio, "controllers_restored", lambda: pytest.fail("must not be asked"))
    why = cio.post_insert_gate({"result": "failure", "error": "INSERT_RESTORE_FAILED: prismatic ..."})
    assert why.startswith("INSERT_RESTORE_FAILED") and cio._DASHBOARD_STOPPED == why


def test_gate_reports_a_protective_stop_as_itself(cio, monkeypatch):
    monkeypatch.setattr(cio, "_SAFETY_STOP", "PROTECTIVE_STOP")
    monkeypatch.setattr(cio, "controllers_restored", lambda: pytest.fail("must not be asked"))
    assert cio.post_insert_gate({"result": "failure"}) == "robot PROTECTIVE_STOP during the insert"
    assert cio._DASHBOARD_STOPPED is None and not cio._test_stops


def test_gate_passes_a_commandable_arm(cio, monkeypatch):
    monkeypatch.setattr(cio, "controllers_restored", lambda: (True, "restored"))
    assert cio.post_insert_gate({"result": "success"}) is None and cio._DASHBOARD_STOPPED is None


def test_latched_run_never_resends_the_program(cio, monkeypatch):
    # review r1 P1-1: the end-of-run cleanup's move after a gate failure
    ran = []
    monkeypatch.setattr(cio, "_DASHBOARD_STOPPED", "CONTROLLERS_NOT_RESTORED after an insert (test)")
    monkeypatch.setattr(cio, "_run_subprocess_once", lambda *a, **k: ran.append(a) or {"result": "success"})
    monkeypatch.setattr(cio.subprocess, "run", lambda *a, **k: ran.append(("subprocess.run", a)))
    r = cio.run_subprocess(["python3", "-m", "primitives.move_to_safe_height"], label="cleanup move_to_safe_height")
    assert r["result"] == "failure" and r["error"].startswith("DASHBOARD_STOPPED") and ran == []


# ---- replay_object, driven offline ---------------------------------------------------------------------------------
INSERT = "ros-mcp-server__translate_object(action='insert', object_name='u_orange', base_name='base1', mode='real')"
TAIL = ["ros-mcp-server__control_gripper(command='open', mode='real')", "ros-mcp-server__move_to_safe_height(mode='real')"]


def _replay_args():
    import replay_real_assembly as rra

    class Grab(Exception):
        pass
    orig = argparse.ArgumentParser.parse_args

    def grab(self, a=None, n=None):
        raise Grab(orig(self, ["--assembly-json", "ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json"]))
    argparse.ArgumentParser.parse_args = grab
    try:
        rra.main()
    except Grab as g:
        return g.args[0]
    finally:
        argparse.ArgumentParser.parse_args = orig


def _drive(monkeypatch, insert_results, controller_states):
    """replay_object on [insert, open, lift]: each insert subprocess returns the next of insert_results and each gate
    sees the next controller state.  Returns (ok, labels of every subprocess the replay ran, the latch)."""
    import cell_io
    import replay_real_assembly as rra
    monkeypatch.setattr(cell_io, "_DASHBOARD_STOPPED", None)
    monkeypatch.setattr(cell_io, "_SAFETY_STOP", None)
    ran, inserts, states = [], list(insert_results), list(controller_states)

    def run(argv, label, timeout_s=600, log_fh=None):
        ran.append(label)
        if "--insert" in argv:
            return inserts.pop(0)
        return {"result": "success"}
    for mod in (rra, cell_io):
        monkeypatch.setattr(mod, "run_subprocess", run)
    for name in ("snapshot_robot_state",):
        monkeypatch.setattr(rra, name, lambda *a, **k: {})
    for name in ("_record_snapshot", "pre_insert_base_read", "held_yaw_calibration", "reread_base", "_seen_here",
                 "_gentle_place_down"):
        monkeypatch.setattr(rra, name, lambda *a, **k: None, raising=False)
    monkeypatch.setattr(rra, "one_node", lambda: False)
    monkeypatch.setattr(rra, "camera_frame_age", lambda: 0.0)                 # the recovery's regrasp: camera streaming,
    monkeypatch.setattr(rra, "placed", lambda obj: None)                      # not at the clear area, a depth target
    monkeypatch.setattr(rra, "depth_pick_target", lambda *a, **k: ((0.3, -0.4, 0.1), (0.0, 0.0, 0.0, 1.0)))
    monkeypatch.setattr(cell_io, "controllers_restored", lambda: states.pop(0))
    monkeypatch.setattr(cell_io, "dashboard_stop", lambda host=None, timeout=3.0: ran.append("DASHBOARD_STOP") or True)
    args = _replay_args()
    args.part_at_staging = True
    state = rra.AssemblyState()
    state.held_quat, state.last_grasp_id = (0.0, 0.0, 0.0, 1.0), 1
    ok = rra.replay_object({"object_name": "u_orange", "tool_sequence": [INSERT] + TAIL}, state, args)
    return ok, ran, cell_io._DASHBOARD_STOPPED


GOOD, BAD = (True, "restored"), (False, "scaled_joint_trajectory_controller inactive")


def test_replay_stops_after_a_failed_gate_on_the_first_try(monkeypatch):
    ok, ran, latch = _drive(monkeypatch, [{"result": "success"}], [BAD])
    assert ok is False and latch.startswith("CONTROLLERS_NOT_RESTORED")
    assert ran == ["u_orange#1 translate_object", "DASHBOARD_STOP"]          # no open, no lift, nothing after


def test_replay_stops_after_a_failed_gate_on_an_outer_try(monkeypatch):
    ok, ran, latch = _drive(monkeypatch, [{"result": "failure", "error": "not seated"}, {"result": "success"}],
                            [GOOD, BAD])
    assert ok is False and latch.startswith("CONTROLLERS_NOT_RESTORED")
    assert ran[-2:] == ["u_orange#1try2 translate_object", "DASHBOARD_STOP"]


def test_replay_stops_after_insert_restore_failed_on_the_recovery_reinsert(monkeypatch):
    fail = {"result": "failure", "error": "not seated"}
    restore_failed = {"result": "failure", "error": "INSERT_RESTORE_FAILED: prismatic could not restore ..."}
    ok, ran, latch = _drive(monkeypatch, [fail, fail, fail, restore_failed], [GOOD, GOOD, GOOD])
    assert ok is False and latch.startswith("INSERT_RESTORE_FAILED")
    assert ran[-2:] == ["u_orange#1rec10 translate_object", "DASHBOARD_STOP"]
