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
    world = NS(rc=0, text="", controllers=RESTORED, safety="NORMAL", stops=[], starts=[])
    srv = object.__new__(dms.DriveMateServer)
    srv.tcp = NS(position=NS(x=0.03, y=-0.45, z=0.2), orientation=NS(x=1.0, y=0.0, z=0.0, w=0.0))
    srv.proc, srv.busy, srv.latch = None, threading.Lock(), dms.InsertLatch()
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
        def __init__(self, argv, cwd=None, env=None, stdout=None, stderr=None, start_new_session=False):
            world.starts.append(argv)
            stdout.write(world.text)
            stdout.flush()
            self.returncode, self.pid = world.rc, 0

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
    world.safety = "NORMAL"
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


def test_result_of_reads_the_last_result_block():
    text = _result_text({"result": "failure", "error": "a"}) + _result_text({"result": "success"})
    assert dms.result_of(text) == {"result": "success"} and dms.result_of("nothing") == {}
