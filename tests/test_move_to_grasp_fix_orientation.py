"""move_to_grasp's fix-orientation (the path every regrasp takes) checks the lift, the swing and the hover before
sending anything (review r1 P2-1, behaviour, not source order): a camera-rule miss and a collision both refuse with
nothing sent and the typed reason kept for the result JSON; a clear swing is sent and completes."""
import types

import numpy as np
from scipy.spatial.transform import Rotation as R

import tests.ros_stubs as ros_stubs

ros_stubs.install()
from primitives import move_to_grasp as mtg  # noqa: E402
from primitives.shared import ik  # noqa: E402
from primitives.shared.config import GRIPPER_CENTER_TOOL_OFFSET  # noqa: E402
from primitives.shared.ik import dh_params, forward_kinematics  # noqa: E402

# MuJoCo r48c: u_orange's place-down end (camera toward the table, the mount 32 mm into it; gripper centre over the
# active clear area) and the high carry before it.
R48C_PLACE_DOWN = [-0.641109, -0.916988, 2.454026, -1.577309, -0.633258, -3.091288]
R48C_HIGH_CARRY = [-0.640855, -1.780168, 2.188927, -0.448674, -0.633091, -3.091574]
HOVER = [0.7748, -1.2917, 1.6872, -1.9656, -1.566, -0.7949]


class Point:
    def __init__(self, positions=None, velocities=None, time_from_start=None):
        self.positions = positions


class Done:
    """A future that is already done: the goal is accepted and succeeds (status 4)."""
    def __init__(self, value):
        self.value = value

    def add_done_callback(self, cb):
        cb(self)

    def result(self):
        return self.value


class Logger:
    def __init__(self):
        self.errors = []

    def info(self, *a, **k):
        pass

    warn = info

    def error(self, msg, *a, **k):
        self.errors.append(msg)


def _node(q):
    T = forward_kinematics(dh_params, np.asarray(q, float))
    quat = R.from_matrix(T[:3, :3]).as_quat()
    sent = []

    def send_goal_async(goal):
        sent.append(goal)
        handle = types.SimpleNamespace(accepted=True,
                                       get_result_async=lambda: Done(types.SimpleNamespace(status=4)))
        return Done(handle)

    node = types.SimpleNamespace(
        current_joint_angles=np.asarray(q, float),
        current_ee_pose=types.SimpleNamespace(pose=types.SimpleNamespace(
            position=types.SimpleNamespace(x=T[0, 3], y=T[1, 3], z=T[2, 3]),
            orientation=types.SimpleNamespace(x=quat[0], y=quat[1], z=quat[2], w=quat[3]))),
        action_client=types.SimpleNamespace(send_goal_async=send_goal_async))
    gc = T[:3, :3] @ GRIPPER_CENTER_TOOL_OFFSET + T[:3, 3]
    return node, sent, (float(gc[0]), float(gc[1]))


def _run(monkeypatch, q, hover=None):
    monkeypatch.setattr(mtg, "JointTrajectoryPoint", Point)
    if hover is not None:                                 # pin the hover IK (the start is high: no lift IK)
        def solve(self, **kw):
            self._best_result = type("Result", (), {"cost": 0.0})()
            return np.asarray(hover, float)
        monkeypatch.setattr(ik.IKSolver, "solve", solve)
    node, sent, xy = _node(q)
    log = Logger()
    ok = mtg.fix_orientation_inprocess(node, xy, log)
    return ok, node, sent, log


def test_a_camera_rule_miss_refuses_with_nothing_sent(monkeypatch):
    # real lift and hover IK from r48c's place-down: the mount is in the table at the start of the lift
    ok, node, sent, log = _run(monkeypatch, R48C_PLACE_DOWN)
    assert ok is False and sent == []
    assert node.fix_orientation_error.startswith("fix-orient: CAMERA_CLEARANCE")
    assert "[fix-orientation leg 1]" in node.fix_orientation_error


def test_a_colliding_swing_refuses_with_nothing_sent(monkeypatch):
    low = list(HOVER)
    low[1] = -0.3                                          # shoulder down: the arm meets the table along the swing
    ok, node, sent, log = _run(monkeypatch, R48C_HIGH_CARRY, hover=low)
    assert ok is False and sent == []
    assert node.fix_orientation_error.startswith("fix-orient: the turn to face-down collides")


def test_a_clear_swing_is_sent_and_completes(monkeypatch):
    ok, node, sent, log = _run(monkeypatch, R48C_HIGH_CARRY)   # real hover IK over the clear area
    assert ok is True and len(sent) == 1, log.errors
    assert node.fix_orientation_error is None


def test_the_refusal_reason_reaches_move_to_grasps_result():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "primitives/move_to_grasp.py").read_text()
    assert ('"Auto fix-orientation failed before move_to_object: "\n'
            '                                      + (getattr(node, "fix_orientation_error", None) or "see the log")') in src
