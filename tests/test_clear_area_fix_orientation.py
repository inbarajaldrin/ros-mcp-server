"""move_to_clear_area --fix-orientation checks its swing to the face-down hover before sending anything (lead
2026-11-20): table/self collision (check_trajectory_collision, as move_to_grasp's fix_orientation_inprocess does),
then the camera rule and the cable caps (camera_up.hover_route_refusal).  Fail closed."""
import numpy as np

import tests.ros_stubs as ros_stubs

ros_stubs.install()
from primitives.core import move_to_clear_area as mca  # noqa: E402
from primitives.shared import ik  # noqa: E402

# MuJoCo r48c's high carry (sideways, 0.4 m up) and fix-orientation's hover over the clear area from its fixed seeds
HIGH_CARRY = [-0.640855, -1.780168, 2.188927, -0.448674, -0.633091, -3.091574]
HOVER = [0.7748, -1.2917, 1.6872, -1.9656, -1.566, -0.7949]


class Logger:
    def __init__(self):
        self.errors = []

    def info(self, *a, **k):
        pass

    warn = info

    def error(self, msg, *a, **k):
        self.errors.append(msg)


def _node():
    n = object.__new__(mca.MoveToClearArea)
    n.mode, n.robot_mode = "fix_orientation", "real"
    n.current_joint_angles = np.array(HIGH_CARRY)
    n.target_gripper_center_position = [-0.320, -0.500, 0.32]
    n.read_current_ee_pose = lambda: {"position": [0.3, -0.4, 0.40], "orientation": [0.0, 0.7071, 0.0, 0.7071]}
    log = Logger()
    n.get_logger = lambda: log
    n.sent = []
    n._dispatch = lambda goal: n.sent.append(goal)
    n.operation_success, n.operation_complete, n.error_message = None, False, None
    return n, log


def _run_with_hover(monkeypatch, hover):
    def solve(self, **kw):
        self._best_result = type("Result", (), {"cost": 0.0})()
        return np.array(hover, float)
    monkeypatch.setattr(ik.IKSolver, "solve", solve)
    n, log = _node()
    n.move_to_clear_space()
    return n, log


def test_a_colliding_swing_is_refused_before_anything_is_sent(monkeypatch):
    low = list(HOVER)
    low[1] = -0.3                                # shoulder down: the arm meets the table along the swing
    assert mca.check_trajectory_collision(np.array(HIGH_CARRY), np.array(low))
    n, log = _run_with_hover(monkeypatch, low)
    assert n.sent == []
    assert n.operation_success is False and n.operation_complete is True
    assert n.error_message.startswith("Fix-orientation refused: the turn to face-down collides")


def test_a_clear_swing_is_sent(monkeypatch):
    n, log = _run_with_hover(monkeypatch, HOVER)
    assert len(n.sent) == 1 and n.error_message is None, log.errors


def test_a_swing_clear_of_the_table_but_past_the_cable_limit_is_refused(monkeypatch):
    wound = list(HOVER)
    wound[5] = -4.2
    n, log = _run_with_hover(monkeypatch, wound)
    assert n.sent == [] and "ROUTE_WRIST3_LIMIT" in n.error_message
