"""drive-mate gates its own inserts and latches (lead 2026-11-20, w8-drivemate-gate).

Behaviour, driven offline: DriveMateServer._execute with the insert subprocess, the dashboard and the controller list
stubbed (tests.ros_stubs; nothing joins a ROS graph).
- a commandable arm after the insert: no refusal, no stop, not latched;
- the controllers left switched / a prismatic INSERT_RESTORE_FAILED / a protective stop: refusal INSERT_GATE, the
  dashboard stop (not for a protective stop: the robot already halted), the server latched, and the NEXT goal refused
  DRIVE_MATE_LATCHED before anything starts;
- a non-zero exit with the arm commandable: refusal insert_rc, not latched, `seated` stays the trace's verdict;
- reset_latch clears only when the robot is NORMAL and the controllers read restored;
- the replay's cell_io.post_insert_gate latches on drive-mate's INSERT_GATE / DRIVE_MATE_LATCHED without a second stop.
"""
import json
import threading
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

import tests.ros_stubs as ros_stubs

ros_stubs.install()
import seams.drive_mate_server as dms  # noqa: E402
from ablations import cell_io  # noqa: E402

RESTORED = (True, "restored")
LEFT_SWITCHED = (False, "scaled_joint_trajectory_controller inactive; force_mode_controller active")


def _goal():
    pose = NS(position=NS(x=0.03, y=-0.45, z=-0.06), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))
    return NS(mate_id="fmb1/mate:00", corrections_json="",
              target=NS(header=NS(frame_id="base"), pose=pose, sigma=[0.0, 0.0, 0.0, 0.0]),
              held=NS(header=NS(frame_id="tool0"), pose=pose))


class _GH:
    def __init__(self):
        self.request, self.is_cancel_requested, self.ended = _goal(), False, None

    def publish_feedback(self, fb):
        pass

    def abort(self):
        self.ended = "aborted"

    def succeed(self):
        self.ended = "succeeded"

    def canceled(self):
        self.ended = "canceled"


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """A server built without Node.__init__, and the world it sees: rc / output of the insert, the controllers, the
    robot's safety status; every dashboard stop and every insert start recorded."""
    world = NS(rc=0, text="", controllers=RESTORED, safety="NORMAL", stops=[], starts=[], children=[])
    srv = object.__new__(dms.DriveMateServer)
    srv.program_running = True
    srv.tcp = NS(position=NS(x=0.03, y=-0.45, z=0.2), orientation=NS(x=1.0, y=0.0, z=0.0, w=0.0))
    srv.proc, srv.busy, srv.latch = None, threading.Lock(), dms.InsertLatch(tmp_path / "latch.json")
    log = NS(info=lambda *a: None, warn=lambda *a: None, error=lambda *a: None)
    srv.get_logger = lambda: log
    srv.resting_tcp = lambda *a, **k: (np.zeros(3), True)
    srv._sigma_gate = lambda g, o, b: (False, "test")
    monkeypatch.setenv("DRIVE_ALIGN", "0")
    monkeypatch.setattr(dms, "resolve", lambda m: (("u_brown", "base1"), ""))
    monkeypatch.setattr(dms, "grasp_from_held", lambda o, p, q: (11, 0.2, 0.0))
    monkeypatch.setattr(dms, "vision_pose", lambda *a, **k: None)
    monkeypatch.setattr(dms, "LOGS", tmp_path)

    class _Msg:
        pass
    monkeypatch.setattr(dms, "DriveMate", NS(Result=_Msg, Feedback=_Msg))
    monkeypatch.setattr(dms, "outcome_of", lambda text, obj, t0, rest=None: NS(
        found_by="result", meta_path="",
        as_dict=lambda: {"stage_reached": "seat", "seated": True, "trace_path": "t.csv"}))
    import seams.seat_learn as sl
    monkeypatch.setattr(sl, "learned_dz", lambda obj: (None, 0))
    monkeypatch.setattr(sl, "record_insert", lambda *a, **k: None)

    class _Popen:
        """Each start takes the next (rc, text) from world.children, else world.rc / world.text."""
        def __init__(self, argv, cwd=None, env=None, stdout=None, stderr=None, start_new_session=False):
            world.starts.append(argv)
            rc, text = world.children.pop(0) if world.children else (world.rc, world.text)
            stdout.write(text)
            stdout.flush()
            self.returncode, self.pid = rc, 0

        def poll(self):
            return self.returncode
    monkeypatch.setattr(dms.subprocess, "Popen", _Popen)
    monkeypatch.setattr(cell_io, "_DASHBOARD_STOPPED", None)
    monkeypatch.setattr(cell_io, "_SAFETY_STOP", None)
    monkeypatch.setattr(cell_io, "robot_safety_status", lambda host=None, timeout=1.5: world.safety)
    monkeypatch.setattr(cell_io, "controllers_restored", lambda *a, **k: world.controllers)
    monkeypatch.setattr(cell_io, "dashboard_stop", lambda host=None, timeout=3.0: world.stops.append(host) or True)
    return srv, world


def _run(srv):
    gh = _GH()
    res = srv._execute(gh)
    return res, gh


def _result_text(result):
    return "progress...\n__RESULT_JSON__\n" + json.dumps(result) + "\n__END_RESULT_JSON__\n"


def test_a_commandable_arm_passes_and_does_not_latch(rig):
    srv, world = rig
    world.text = _result_text({"result": "success"})
    res, gh = _run(srv)
    assert res.refusal == "" and res.seated is True and gh.ended == "succeeded"
    assert srv.latch.reason is None and world.stops == []


def test_controllers_left_switched_stop_the_arm_latch_and_refuse_the_next_goal(rig):
    srv, world = rig
    world.text, world.controllers = _result_text({"result": "failure"}), LEFT_SWITCHED
    res, gh = _run(srv)
    assert res.refusal.startswith("INSERT_GATE: CONTROLLERS_NOT_RESTORED") and gh.ended == "aborted"
    assert world.stops and srv.latch.reason
    world.controllers = RESTORED                          # even a healthy arm: the latch holds until reset
    res2, gh2 = _run(srv)
    assert res2.refusal.startswith("DRIVE_MATE_LATCHED: CONTROLLERS_NOT_RESTORED") and gh2.ended == "aborted"
    assert len(world.starts) == 1                         # the second goal started nothing


def test_insert_restore_failed_is_typed_and_latches_without_asking_the_controllers(rig, monkeypatch):
    srv, world = rig
    world.rc = 1
    world.text = _result_text({"result": "failure", "error": "INSERT_RESTORE_FAILED: prismatic could not restore"})
    monkeypatch.setattr(cell_io, "controllers_restored", lambda *a, **k: pytest.fail("must not be asked"))
    res, gh = _run(srv)
    assert res.refusal.startswith("INSERT_GATE: INSERT_RESTORE_FAILED") and srv.latch.reason and world.stops


def test_a_protective_stop_is_reported_as_itself_and_latches(rig):
    srv, world = rig
    world.safety, world.text = "PROTECTIVE_STOP", _result_text({"result": "failure"})
    res, gh = _run(srv)
    assert res.refusal == "INSERT_GATE: robot PROTECTIVE_STOP during the insert" and srv.latch.reason
    assert world.stops == []                              # the robot already halted


def test_a_nonzero_exit_on_a_commandable_arm_is_carried_not_latched(rig):
    srv, world = rig
    world.rc, world.text = 1, _result_text({"result": "failure", "error": "FSM ended in SEARCH timeout"})
    res, gh = _run(srv)
    assert res.refusal == "insert_rc: 1 (FSM ended in SEARCH timeout)" and srv.latch.reason is None
    assert res.seated is True and gh.ended == "succeeded" and world.stops == []
    world.text = "no result block at all\n"
    res, _ = _run(srv)
    assert res.refusal == "insert_rc: 1 (translate_object exited 1)"


def test_reset_latch_only_after_the_arm_is_restored(rig):
    srv, world = rig
    srv.latch.trip("CONTROLLERS_NOT_RESTORED after an insert (test)")
    world.controllers = LEFT_SWITCHED
    r = srv.on_reset_latch(None, NS(success=None, message=None))
    assert r.success is False and "stop_run.sh" in r.message and srv.latch.reason
    world.controllers, world.safety = RESTORED, "PROTECTIVE_STOP"
    r = srv.on_reset_latch(None, NS(success=None, message=None))
    assert r.success is False and "PROTECTIVE_STOP" in r.message and srv.latch.reason
    world.safety, srv.program_running = "NORMAL", False                 # r1 P2-2: the program must run (stop_run.sh)
    r = srv.on_reset_latch(None, NS(success=None, message=None))
    assert r.success is False and "program is not running" in r.message and srv.latch.reason
    srv.program_running = True
    r = srv.on_reset_latch(None, NS(success=None, message=None))
    assert r.success is True and srv.latch.reason is None
    world.text = _result_text({"result": "success"})
    res, gh = _run(srv)
    assert res.refusal == "" and gh.ended == "succeeded"


def test_the_replay_latches_on_drive_mates_gate_and_latch_without_a_second_stop(monkeypatch):
    monkeypatch.setattr(cell_io, "_SAFETY_STOP", None)
    stops = []
    monkeypatch.setattr(cell_io, "dashboard_stop", lambda host=None, timeout=3.0: stops.append(host) or True)
    monkeypatch.setattr(cell_io, "controllers_restored", lambda *a, **k: pytest.fail("must not be asked"))
    for err in ("INSERT_GATE: CONTROLLERS_NOT_RESTORED after an insert (x)", "DRIVE_MATE_LATCHED: INSERT_RESTORE_FAILED"):
        monkeypatch.setattr(cell_io, "_DASHBOARD_STOPPED", None)
        why = cell_io.post_insert_gate({"result": "failure", "error": err})
        assert why == f"drive-mate: {err}" and cell_io._DASHBOARD_STOPPED == why
    assert stops == []


# ---- round 2 (review r1) ---------------------------------------------------------------------------------------
@pytest.fixture
def aligning(rig, monkeypatch):
    """The rig with the align step on (a free base, the sigma gate asking for the servo)."""
    import importlib
    import primitives.shared.config as cfg
    srv, world = rig
    monkeypatch.setenv("DRIVE_ALIGN", "1")
    monkeypatch.setattr(importlib, "reload", lambda m: m)
    monkeypatch.setattr(cfg, "BASE_FIXED", False, raising=False)
    srv._sigma_gate = lambda g, o, b: (True, "test: servo")
    return srv, world


def _align_text(**kw):
    out = dict(aim_base=None, measured_base=None, held_quat=None, route="servo", measured=True, safety="",
               grasp_depth_fix_m=None, dashboard_stopped="")
    out.update(kw)
    return "aligning...\n__ALIGN__ " + json.dumps(out) + "\n"


def test_the_align_is_gated_and_a_switched_arm_stops_before_the_insert(aligning):
    srv, world = aligning
    world.controllers = LEFT_SWITCHED
    world.children = [(0, _align_text())]
    res, gh = _run(srv)
    assert res.refusal.startswith("INSERT_GATE: CONTROLLERS_NOT_RESTORED") and srv.latch.reason and world.stops
    assert len(world.starts) == 1 and "seams.drive_align" in world.starts[0]          # the insert never started


def test_the_aligns_own_dashboard_stop_latches(aligning):
    srv, world = aligning
    world.children = [(3, _align_text(dashboard_stopped="cell_io sent the dashboard `stop` during 'turn_wrist'"))]
    res, gh = _run(srv)
    assert res.refusal.startswith("INSERT_GATE: the align sent the dashboard stop") and srv.latch.reason
    assert len(world.starts) == 1


@pytest.mark.parametrize("rc, text", [(1, _align_text(error="no marker")), (0, "aligning...\n__ALIGN__ {not json\n"),
                                      (0, "no result line\n"), (2, _align_text(safety="PROTECTIVE_STOP"))])
def test_a_failed_align_on_a_commandable_arm_refuses_and_never_inserts(aligning, rc, text):
    srv, world = aligning
    world.children = [(rc, text)]
    res, gh = _run(srv)
    assert res.refusal.startswith("align_failed") and gh.ended == "aborted" and srv.latch.reason is None
    assert len(world.starts) == 1 and world.stops == []


def test_an_exception_judging_the_insert_fails_closed(rig, monkeypatch):
    """r1 P1-2: a truncated trace makes outcome_of raise; the goal must not end open (no latch, empty refusal)."""
    srv, world = rig
    world.text = _result_text({"result": "success"})
    monkeypatch.setattr(dms, "outcome_of", lambda *a, **k: (_ for _ in ()).throw(ValueError("truncated CSV")))
    res, gh = _run(srv)
    assert res.refusal.startswith("INSERT_GATE: gate crashed: trace unreadable") and "arm checked commandable" in res.refusal and srv.latch.reason
    assert gh.ended == "aborted" and world.stops == []                  # the controllers read restored: no stop
    world.controllers = LEFT_SWITCHED                                    # unrestored: the stop goes out too
    srv.latch.reason = None
    res, gh = _run(srv)
    assert res.refusal.startswith("INSERT_GATE: CONTROLLERS_NOT_RESTORED") and world.stops     # the gate ran FIRST


def test_an_exception_while_the_child_runs_fails_closed_and_signals_it(rig, monkeypatch):
    srv, world = rig
    world.controllers = LEFT_SWITCHED
    killed = []
    monkeypatch.setattr(dms.os, "killpg", lambda pid, sig: killed.append(sig))
    monkeypatch.setattr(dms.DriveMateServer, "_term_later", staticmethod(lambda p: killed.append("term_later")))

    class _Running:
        def __init__(self, *a, **k):
            world.starts.append(a[0]); self.pid, self.returncode = 0, None

        def poll(self):
            return None
    monkeypatch.setattr(dms.subprocess, "Popen", _Running)
    gh = _GH()
    gh.publish_feedback = lambda fb: (_ for _ in ()).throw(RuntimeError("feedback publisher gone"))
    res = srv._execute(gh)
    assert res.refusal.startswith("INSERT_GATE: gate crashed: insert") and srv.latch.reason and world.stops
    assert killed[0] == dms.signal.SIGINT and "term_later" in killed and srv.proc is None


def test_an_unreadable_safety_status_fails_closed(rig):
    srv, world = rig
    world.safety, world.text = None, _result_text({"result": "success"})
    res, gh = _run(srv)
    assert res.refusal.startswith("INSERT_GATE: robot safety status unreadable") and srv.latch.reason and world.stops


def test_reduced_safety_is_cell_ios_ok(rig):
    srv, world = rig
    world.safety, world.text = "REDUCED", _result_text({"result": "success"})
    res, gh = _run(srv)
    assert res.refusal == "" and srv.latch.reason is None


def test_a_cancel_before_the_insert_starts_nothing(rig):
    srv, world = rig
    gh = _GH()
    gh.is_cancel_requested = True
    srv._execute(gh)
    assert gh.ended == "canceled" and world.starts == []


def test_reset_answers_at_once_when_not_latched(rig, monkeypatch):
    srv, world = rig
    monkeypatch.setattr(cell_io, "controllers_restored", lambda *a, **k: pytest.fail("must not be asked"))
    monkeypatch.setattr(cell_io, "robot_safety_status", lambda *a, **k: pytest.fail("must not be asked"))
    r = srv.on_reset_latch(None, NS(success=None, message=None))
    assert r.success is True and r.message == "not latched"


def test_the_replay_maps_a_seated_gate_refusal_to_a_failure_and_latches(monkeypatch):
    """r1 P1-1 through the replay's own mapping: seated=True and INSERT_GATE must still latch the replay."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ablations"))
    import replay_real_assembly as rra
    out = {"seated": True, "stage_reached": "seat", "refusal": "INSERT_GATE: CONTROLLERS_NOT_RESTORED (x)"}
    mapped = rra.map_drive_mate_result(out)
    assert mapped["result"] == "failure" and mapped["error"].startswith("INSERT_GATE")
    import cell_io as replay_cell_io                                      # the replay's own module object
    monkeypatch.setattr(replay_cell_io, "_DASHBOARD_STOPPED", None)
    monkeypatch.setattr(replay_cell_io, "_SAFETY_STOP", None)
    monkeypatch.setattr(replay_cell_io, "controllers_restored", lambda *a, **k: pytest.fail("must not be asked"))
    assert replay_cell_io.post_insert_gate(mapped).startswith("drive-mate: INSERT_GATE")
    monkeypatch.setattr(replay_cell_io, "_DASHBOARD_STOPPED", None)      # the refusal field alone latches too
    assert replay_cell_io.post_insert_gate(dict(out, error="")).startswith("drive-mate: INSERT_GATE")
    ok = rra.map_drive_mate_result({"seated": True, "refusal": "insert_rc: 1 (x)"})
    assert ok["result"] == "success" and ok["error"] == ""                # a commandable non-zero exit: as before


def test_result_of_reads_the_last_result_block():
    text = _result_text({"result": "failure", "error": "a"}) + _result_text({"result": "success"})
    assert dms.result_of(text) == {"result": "success"} and dms.result_of("nothing") == {}


# ---- review r2 (before the merge) -----------------------------------------------------------------------------------
def _cancelled_mid_insert(srv, world, monkeypatch):
    polls = []

    class _Slow:
        def __init__(self, argv, cwd=None, env=None, stdout=None, stderr=None, start_new_session=False):
            world.starts.append(argv); self.pid = 0; self.returncode = None

        def poll(self):
            polls.append(1)
            if len(polls) >= 3:
                self.returncode = -2
            return self.returncode
    monkeypatch.setattr(dms.subprocess, "Popen", _Slow)
    monkeypatch.setattr(dms.time, "sleep", lambda s: None)
    gh = _GH()

    def feedback(fb):
        gh.is_cancel_requested = True
    gh.publish_feedback = feedback
    return srv._execute(gh), gh


def test_a_cancel_with_the_program_running_runs_the_full_gate(rig, monkeypatch):
    """r3 P1: a Ctrl-C'd replay or a client timeout cancels with the program RUNNING; force mode may be left on, so
    the full gate runs, stop included, and the server latches."""
    srv, world = rig
    srv.program_running, world.controllers = True, LEFT_SWITCHED
    res, gh = _cancelled_mid_insert(srv, world, monkeypatch)
    assert res.refusal.startswith("DRIVE_MATE_CANCELLED: CONTROLLERS_NOT_RESTORED") and world.stops
    assert srv.latch.reason and gh.ended == "canceled"


def test_a_cancel_with_the_program_already_stopped_latches_without_a_stop(rig, monkeypatch):
    """r2 P2-A: stop_run.sh's step 0 stopped the program; it owns the restore, so the server sends no stop."""
    srv, world = rig
    srv.program_running, world.controllers = False, LEFT_SWITCHED
    res, gh = _cancelled_mid_insert(srv, world, monkeypatch)
    assert res.refusal.startswith("DRIVE_MATE_CANCELLED") and "already stopped" in res.refusal
    assert world.stops == [] and srv.latch.reason and gh.ended == "canceled"


def test_the_latch_persists_across_a_restart_and_only_a_reset_removes_it(tmp_path):
    f = tmp_path / "latch.json"
    a = dms.InsertLatch(f)
    a.trip("CONTROLLERS_NOT_RESTORED (x)")
    assert f.exists() and dms.InsertLatch(f).reason == "CONTROLLERS_NOT_RESTORED (x)"     # a new server: latched
    assert a.reset("NORMAL", (False, "x"), True).startswith("not reset") and f.exists()
    assert a.reset("NORMAL", (True, "restored"), True) == "" and not f.exists()
    assert dms.InsertLatch(f).reason is None
    f.write_text("{not json")
    assert dms.InsertLatch(f).reason.startswith("latch file present")                      # unreadable: latched
    f.write_text("{}")
    assert dms.InsertLatch(f).reason.startswith("latch file present")                      # presence is the latch


def test_the_startup_check_reads_the_latch_file_even_with_the_server_down(tmp_path):
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ablations"))
    import replay_real_assembly as rra
    f = tmp_path / "latch.json"
    f.write_text(json.dumps({"reason": "INSERT_RESTORE_FAILED (x)"}))
    assert rra.drive_mate_startup_refusal({"latched": False}, f).startswith("DRIVE_MATE_LATCHED (file")
    assert rra.drive_mate_startup_refusal({"error": "no server"}, f).startswith("DRIVE_MATE_LATCHED (file")
    assert rra.drive_mate_startup_refusal({"latched": False}, tmp_path / "none.json") is None


def test_a_stopped_program_after_any_motion_child_latches(rig):
    """r2 P3-A: a child that sent its own stop (no typed result) is caught by the program state."""
    srv, world = rig
    srv.program_running = False
    world.rc, world.text = 1, "Traceback ... (no result block)\n"
    res, gh = _run(srv)
    assert res.refusal == "INSERT_GATE: the external-control program was stopped during the insert"
    assert srv.latch.reason and gh.ended == "aborted"


def test_the_latch_status_is_read_only(rig):
    srv, world = rig
    r = srv.on_latch_status(None, NS(success=None, message=None))
    assert r.success is True and r.message == "not latched"
    srv.latch.trip("CONTROLLERS_NOT_RESTORED (x)")
    r = srv.on_latch_status(None, NS(success=None, message=None))
    assert r.success is False and r.message == "CONTROLLERS_NOT_RESTORED (x)" and srv.latch.reason


def test_a_seams_run_refuses_at_startup_on_a_latched_or_absent_drive_mate():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ablations"))
    import replay_real_assembly as rra
    none = Path("/nonexistent/drive_mate_latch.json")
    assert rra.drive_mate_startup_refusal({"latched": True, "reason": "x"}, none) == "DRIVE_MATE_LATCHED: x"
    assert rra.drive_mate_startup_refusal({"error": "no server"}, none).startswith("DRIVE_MATE_UNAVAILABLE")
    assert rra.drive_mate_startup_refusal({"latched": False, "reason": ""}, none) is None
    assert "Do NOT open" in rra.DRIVE_MATE_LATCH_RECOVERY and "ABOVE the base" in rra.DRIVE_MATE_LATCH_RECOVERY


def test_stop_run_reports_the_latch_and_never_clears_it():
    text = (Path(__file__).resolve().parents[1] / "compliant_insertion_studio/scripts/stop_run.sh").read_text()
    assert "/ur5e/drive_mate/latch_status" in text
    calls = [l for l in text.splitlines() if "ros2 service call /ur5e/drive_mate/reset_latch" in l]
    assert calls and all(l.strip().startswith("echo") for l in calls)      # printed, never called
    assert text.index("if ! prog_running; then") < text.index('"  Arm is stopped and commandable.')


def test_one_latch_path_for_the_server_the_replay_and_any_tree(monkeypatch, tmp_path):
    """r4 P2-1: the latch path is per machine (FMB_STATE_DIR / ~/.local/state/fmb_cell), not per tree: a server from
    another tree starts latched, and every startup check reads the same file."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ablations"))
    import replay_real_assembly as rra
    from seams import latch_state
    monkeypatch.setenv("FMB_STATE_DIR", str(tmp_path / "state"))
    assert latch_state.latch_path() == tmp_path / "state" / "drive_mate_latch.json"
    a = dms.InsertLatch()
    assert a.path == latch_state.latch_path() and a.reason is None
    a.trip("CONTROLLERS_NOT_RESTORED (x)")
    assert dms.InsertLatch().reason == "CONTROLLERS_NOT_RESTORED (x)"       # a restarted server, any tree
    assert rra.drive_mate_startup_refusal({"latched": False}).startswith("DRIVE_MATE_LATCHED (file")
    assert a.reset("NORMAL", (True, "restored"), True) == "" and rra.drive_mate_startup_refusal(
        {"latched": False}) is None


def test_an_unwritable_state_dir_refuses_the_run_and_a_failed_write_is_loud(monkeypatch, tmp_path, capsys):
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ablations"))
    import replay_real_assembly as rra
    blocker = tmp_path / "file_not_dir"
    blocker.write_text("x")
    monkeypatch.setenv("FMB_STATE_DIR", str(blocker / "state"))              # under a regular file: unwritable
    assert rra.drive_mate_startup_refusal({"latched": False}).startswith("DRIVE_MATE_STATE_DIR_UNWRITABLE")
    a = dms.InsertLatch()
    a.trip("x")
    assert a.reason == "x" and a.write_error and "NOT written" in capsys.readouterr().err
