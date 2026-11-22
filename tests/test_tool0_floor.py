"""R40, the operator's tool0 floor (primitives/shared/tool0_floor.py): the pure check, the trajectory check, the send
guard and move_cartesian.plan_move's refusal - all before anything is sent.  Offline: ROS is stubbed (tests.ros_stubs)."""
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import tests.ros_stubs as ros_stubs

ros_stubs.install()
from primitives.shared import tool0_floor as tf  # noqa: E402

DOWN = R.from_euler("x", 180, degrees=True)


def pose(z, tilt_deg=0.0, axis="x"):
    T = np.eye(4)
    T[:3, :3] = (R.from_euler(axis, tilt_deg, degrees=True) * DOWN).as_matrix()
    T[:3, 3] = [0.40, -0.30, z]
    return T


def horizontal(z):
    T = np.eye(4)
    T[:3, :3] = (R.from_euler("x", 90, degrees=True) * DOWN).as_matrix()      # tool z horizontal
    T[:3, 3] = [0.40, -0.30, z]
    return T


def test_the_constants_are_r40s():
    assert (tf.TOOL0_FLOOR_TOP_DOWN_M, tf.TOOL0_FLOOR_ANY_M, tf.TOP_DOWN_MAX_DEG) == (0.1478, -0.0002, 10.0)


def test_top_down_below_147_8_is_refused():
    why = tf.r40_refusal(pose(0.1477))
    assert why.startswith("R40_TOOL0_FLOOR: tool0 z 147.7 mm < 147.8 mm (top-down")
    assert tf.r40_refusal(pose(0.1000)).startswith("R40_TOOL0_FLOOR")


def test_top_down_at_the_october_insert_minimum_and_at_the_floor_is_allowed():
    assert tf.r40_refusal(pose(0.1829)) is None           # u_brown's lowest October insert
    assert tf.r40_refusal(pose(0.1478)) is None           # the floor itself


def test_horizontal_at_minus_0_1_mm_is_allowed_and_at_minus_1_mm_refused():
    assert tf.r40_refusal(horizontal(-0.0001)) is None
    why = tf.r40_refusal(horizontal(-0.0010))
    assert why.startswith("R40_TOOL0_FLOOR: tool0 z -1.0 mm < -0.2 mm (any orientation")


def test_the_top_down_split_is_10_deg_of_tool_z_tilt():
    assert tf.is_top_down(pose(0.1, 9.9)) and tf.is_top_down(pose(0.1, 9.9, "y"))
    assert not tf.is_top_down(pose(0.1, 10.1))
    assert tf.r40_refusal(pose(0.1000, 9.9)) is not None  # still top-down: the 147.8 floor
    assert tf.r40_refusal(pose(0.1000, 10.1)) is None     # tilted past 10 deg: the any-orientation floor


def test_a_non_finite_pose_is_refused():
    T = pose(0.2)
    T[2, 3] = np.nan
    assert "not checkable" in tf.r40_refusal(T)


def test_path_refusal_names_the_lowest_offending_pose():
    why = tf.path_refusal([pose(0.30), pose(0.146), pose(0.140), pose(0.20)], "waypoint")
    assert "tool0 z 140.0 mm" in why and why.endswith("at waypoint 3/4")
    assert tf.path_refusal([pose(0.30), pose(0.1829)]) is None


@pytest.fixture
def fake_fk(monkeypatch):
    """FK stand-in: tool0 top-down at z = shoulder_pan (rad read as m), so a trajectory's verdict is set by hand."""
    import primitives.shared.ik as ik
    monkeypatch.setattr(ik, "forward_kinematics", lambda dh, q: pose(float(q[0])))


NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def test_trajectory_refusal_checks_every_point_by_joint_name(fake_fk):
    assert tf.trajectory_refusal(NAMES, [[0.30, 0, 0, 0, 0, 0], [0.1829, 0, 0, 0, 0, 0]]) is None
    why = tf.trajectory_refusal(NAMES, [[0.30, 0, 0, 0, 0, 0], [0.1470, 0, 0, 0, 0, 0]])
    assert why.startswith("R40_TOOL0_FLOOR: tool0 z 147.0 mm") and why.endswith("trajectory point 2/2")
    # any joint order, prefixed names (DroidStack instance)
    names = ["ur5e_0_" + n for n in reversed(NAMES)]
    assert tf.trajectory_refusal(names, [[0, 0, 0, 0, 0, 0.1470]]).startswith("R40_TOOL0_FLOOR: tool0 z 147.0")
    assert tf.trajectory_refusal(names, [[0, 0, 0, 0, 0, 0.30]]) is None


def test_a_trajectory_that_cannot_be_checked_is_refused(fake_fk):
    assert "not checkable" in tf.trajectory_refusal(NAMES[:5], [[0.3] * 5])
    assert "not checkable" in tf.trajectory_refusal(NAMES, [[0.3] * 5])
    assert "not checkable" in tf.trajectory_refusal(NAMES, [])


def test_trajectory_refusal_agrees_with_the_real_fk():
    from primitives.shared.ik import dh_params, forward_kinematics
    rng = np.random.default_rng(40)
    seen = set()
    for _ in range(300):
        q = rng.uniform(-np.pi, np.pi, 6)
        expect = tf.r40_refusal(forward_kinematics(dh_params, q))
        got = tf.trajectory_refusal(NAMES, [list(q)])
        assert (expect is None) == (got is None)
        seen.add(expect is None)
    assert seen == {True, False}


@pytest.fixture
def guarded_client(monkeypatch):
    """A stand-in ActionClient class in rclpy.action, the send guard installed on it fresh; sends recorded."""
    sent = []

    class FollowJointTrajectory:
        pass

    class GripperCommand:
        pass

    class ActionClient:
        def __init__(self, action_type):
            self._action_type = action_type

        def send_goal_async(self, goal, *a, **k):
            sent.append(goal)
            return "future"

    monkeypatch.setattr(sys.modules["rclpy.action"], "ActionClient", ActionClient, raising=False)
    monkeypatch.setattr(tf, "_send_guard_installed", False)
    tf.install_send_guard()
    return NS(sent=sent, arm=ActionClient(FollowJointTrajectory), grip=ActionClient(GripperCommand))


def _goal(*zs):
    pts = [NS(positions=[z, 0, 0, 0, 0, 0]) for z in zs]
    return NS(trajectory=NS(joint_names=NAMES, points=pts))


def test_the_send_guard_refuses_an_arm_goal_under_the_floor_before_the_send(guarded_client, fake_fk, monkeypatch):
    monkeypatch.setenv(tf.ENV, "1")
    with pytest.raises(tf.Tool0FloorRefused, match="R40_TOOL0_FLOOR: tool0 z 140.0 mm"):
        guarded_client.arm.send_goal_async(_goal(0.30, 0.140))
    assert guarded_client.sent == []
    assert guarded_client.arm.send_goal_async(_goal(0.30, 0.1829)) == "future" and len(guarded_client.sent) == 1


def test_the_send_guard_is_off_without_the_env_and_ignores_non_arm_goals(guarded_client, fake_fk, monkeypatch):
    monkeypatch.delenv(tf.ENV, raising=False)
    guarded_client.arm.send_goal_async(_goal(0.10))                      # not enforced: sent
    monkeypatch.setenv(tf.ENV, "1")
    guarded_client.grip.send_goal_async(NS(command=NS(position=0.0)))    # a gripper goal: never checked
    assert len(guarded_client.sent) == 2


def test_plan_move_refuses_under_the_floor_before_anything_is_sent(monkeypatch):
    import primitives.move_cartesian as mc
    monkeypatch.setattr(mc, "forward_kinematics", lambda dh, q: pose(float(q[0])))
    monkeypatch.setattr(mc, "camera_path_refusal", lambda T0, wps: None)
    wps = [[z, 0, 0, 0, 0, 0] for z in np.linspace(0.20, 0.14, 30)]
    monkeypatch.setattr(mc, "compute_cartesian_waypoints_ik", lambda *a, **k: wps)

    def no_send(*a, **k):
        raise AssertionError("a goal was sent")
    n = NS(q=np.array([0.20, 0, 0, 0, 0, 0]), ac=NS(wait_for_server=lambda timeout_sec=None: True, send_goal_async=no_send))
    monkeypatch.setattr(mc.rclpy, "spin_once", lambda node, **k: setattr(node, "q", np.array(wps[0])), raising=False)
    monkeypatch.setenv(tf.ENV, "1")
    out = mc.plan_move(n, [0.40, -0.30, 0.14], ("keep", None), min_z=-1.0)
    assert out["result"] == "failure" and out["refusal"] == "R40_TOOL0_FLOOR"
    assert out["error"].startswith("R40_TOOL0_FLOOR: tool0 z 140.0 mm") and out["error"].endswith("waypoint 30/30")
