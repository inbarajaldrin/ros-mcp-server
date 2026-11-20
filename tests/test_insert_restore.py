"""An insert must leave the arm commandable (lead 2026-11-20, w5b-insert-restore, after the URSim stop test): prismatic's
cleanup stops force mode first, retries the controller restore with backoff, and when the restore still fails halts
motion (dashboard stop) and exits EXIT_RESTORE_FAILED; translate_object reports it typed; the replay checks the
controllers after every insert and stops the run."""
import types
from pathlib import Path

import tests.ros_stubs as ros_stubs

ros_stubs.install()
from primitives._real_mode_stash import prismatic_peg_insertion as pp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LIST_RESTORED = """joint_state_broadcaster  joint_state_broadcaster/JointStateBroadcaster  active
scaled_joint_trajectory_controller  ur_controllers/ScaledJointTrajectoryController  \x1b[92mactive\x1b[0m
force_mode_controller  ur_controllers/ForceModeController  inactive
passthrough_trajectory_controller  ur_controllers/PassthroughTrajectoryController  inactive"""
# what the URSim rig showed after an induced restore failure on dev (receipts lg-press-fail / yl-press-fail)
LIST_LEFT_SWITCHED = """scaled_joint_trajectory_controller  ur_controllers/ScaledJointTrajectoryController  inactive
force_mode_controller  ur_controllers/ForceModeController  active
passthrough_trajectory_controller  ur_controllers/PassthroughTrajectoryController  active"""


def _cleanup_env(monkeypatch, switch_results, stop_ok=True):
    calls = []
    results = list(switch_results)

    def switch(activate, deactivate):
        calls.append(("switch", tuple(activate), tuple(deactivate)))
        return results.pop(0) if results else False
    monkeypatch.setattr(pp, "switch_controllers", switch)
    monkeypatch.setattr(pp, "RESTORE_BACKOFF_S", (0.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr(pp.time, "sleep", lambda s: None)
    monkeypatch.setattr(pp, "halt_motion_dashboard", lambda host=None: calls.append(("dashboard_stop",)) or True)
    monkeypatch.setattr(pp.rclpy, "ok", lambda: True, raising=False)
    monkeypatch.setattr(pp.rclpy, "shutdown", lambda: calls.append(("shutdown",)), raising=False)
    import primitives.shared.goal_guard as gg
    monkeypatch.setattr(gg, "cancel_active_goals", lambda *a, **k: calls.append(("cancel_goals",)))
    node = types.SimpleNamespace(stop_force_mode=lambda: calls.append(("stop_force_mode",)) or stop_ok)
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


def test_force_mode_stop_is_retried_when_not_acknowledged(monkeypatch):
    node, calls = _cleanup_env(monkeypatch, [True], stop_ok=False)
    pp.safe_state_cleanup(node)
    assert [c[0] for c in calls].count("stop_force_mode") == 3


def test_a_failed_restore_overrides_the_exit_code(monkeypatch):
    # main() with the ROS side stubbed: the insert "succeeds", the cleanup cannot restore -> EXIT_RESTORE_FAILED
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


def test_controller_state_check():
    from ablations import cell_io
    assert cell_io.controllers_restored(list_text=LIST_RESTORED) == (True, "restored")
    ok, why = cell_io.controllers_restored(list_text=LIST_LEFT_SWITCHED)
    assert ok is False and "scaled_joint_trajectory_controller inactive" in why and "force_mode_controller active" in why
    ok, why = cell_io.controllers_restored(list_text="")                 # unreadable: fail closed
    assert ok is False and why.startswith("controller state unreadable")


def test_translate_object_and_the_replay_act_on_it():
    translate = (ROOT / "primitives/translate_object.py").read_text()
    assert "if r.returncode == 3:" in translate and '"INSERT_RESTORE_FAILED: prismatic could not restore' in translate
    replay = (ROOT / "ablations/replay_real_assembly.py").read_text()
    i = replay.index("        result = execute(step_idx, step, tool_name, kw)\n        if result is None:")
    block = replay[i:i + 1400]
    assert "ok_ctrl, why_ctrl = cell_io.controllers_restored()" in block
    assert "cell_io.dashboard_stop(" in block and "CONTROLLERS_NOT_RESTORED" in block and "return False" in block
