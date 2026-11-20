"""rotate_object's declared budget (lead 2026-11-20): before any motion, planning so far + the planned motion + 10 s
must fit the caller's timeout, else ROTATE_OUTLASTS_CALLER and nothing is sent; the replays pass their timeout as the
budget (replay_real_assembly: 90 s)."""
import types

import numpy as np
from scipy.spatial.transform import Rotation as R

import tests.ros_stubs as ros_stubs

ros_stubs.install()
from primitives import rotate_object as ro  # noqa: E402
from primitives.shared.ik import dh_params, forward_kinematics  # noqa: E402

# a face-down hover over the table at safe height, holding inverted_u_yellow as the twin saw it lying (rpy -180, 0, 68.3)
HOVER = np.array([0.7748, -1.2917, 1.6872, -1.9656, -1.566, -0.7949])
YELLOW_FLAT = R.from_euler("xyz", [-180.0, 0.0, 68.3], degrees=True).as_quat()


class Logger:
    def __init__(self):
        self.lines = []

    def info(self, msg, *a, **k):
        self.lines.append(msg)

    warn = warning = error = info


def _node(monkeypatch, elapsed_s, budget_s):
    n = object.__new__(ro.ReorientForAssembly)
    n.mode = "real"
    n.assembly_config, n.assembly_json_file, n.loaded_base_name = {}, None, None
    n.symmetry_dir = ro.SYMMETRY_DIR
    n.current_poses = {}
    T = forward_kinematics(dh_params, HOVER)
    q = R.from_matrix(T[:3, :3]).as_quat()
    n.current_ee_pose = types.SimpleNamespace(pose=types.SimpleNamespace(
        position=types.SimpleNamespace(x=T[0, 3], y=T[1, 3], z=T[2, 3]),
        orientation=types.SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3])))
    n.current_joint_angles = HOVER.copy()
    n.joint_names = ro.BARE_ARM_JOINTS
    n.cardinal_error_threshold_initial, n.cardinal_error_threshold_max = 45.0, 180.0
    n.cardinal_error_threshold_increment = 10.0
    n.current_cardinal_error_threshold = 45.0
    n.tcp_to_gripper_center_offset = ro.GRIPPER_CENTER_TOOL_OFFSET
    n.rotate_about_gripper_center = ro.ROTATE_ABOUT_GRIPPER_CENTER
    for k in ("error_message", "object_name", "base_name", "initial_object_orientation_quat",
              "initial_object_orientation_rpy_deg", "final_object_orientation_quat", "final_object_orientation_rpy_deg",
              "initial_ee_orientation_quat", "initial_ee_orientation_rpy_deg", "final_ee_orientation_quat",
              "final_ee_orientation_rpy_deg", "target_orientation_quat", "target_orientation_rpy_deg",
              "alignment_error_deg"):
        setattr(n, k, None)
    log = Logger()
    n.get_logger = lambda: log
    n.read_current_joint_angles = lambda: n.current_joint_angles.copy()
    n.budget_s = budget_s
    sent = []
    n.execute_trajectory = lambda trajectory: sent.append(trajectory) or True
    n.action_client = types.SimpleNamespace(send_goal_async=lambda goal: sent.append(goal))
    # hermetic: no measured base yaw from this PC's logs; the route planner answers at once with the direct line (the
    # slow planner is the process clock below: as if planning had already taken `elapsed_s`)
    monkeypatch.setattr(ro, "_measured_base_yaw", lambda *a, **k: None)
    monkeypatch.setattr(ro.ReorientForAssembly, "_plan_reorient_route",
                        lambda self, start, target, held: ([np.asarray(start, float), np.asarray(target, float)], None))
    monkeypatch.setattr(ro, "process_elapsed_s", lambda: elapsed_s)
    return n, sent, log


def test_the_budget_rule():
    assert ro.outlasts_caller_refusal(80.0, 12.0, None) is None              # no budget: no check
    assert ro.outlasts_caller_refusal(30.0, 12.0, 90.0) is None              # 52 s <= 90 s
    why = ro.outlasts_caller_refusal(70.0, 12.0, 90.0)                       # 92 s > 90 s
    assert why.startswith("ROTATE_OUTLASTS_CALLER") and "nothing sent" in why


def test_a_slow_planner_refuses_before_any_motion_is_sent(monkeypatch):
    n, sent, log = _node(monkeypatch, elapsed_s=85.0, budget_s=90.0)
    assert n.reorient_for_target("inverted_u_yellow", "base1", list(YELLOW_FLAT)) is False
    assert sent == []
    assert n.error_message.startswith("ROTATE_OUTLASTS_CALLER"), n.error_message


def test_the_same_reorient_within_its_budget_is_sent(monkeypatch):
    n, sent, log = _node(monkeypatch, elapsed_s=5.0, budget_s=90.0)
    assert n.reorient_for_target("inverted_u_yellow", "base1", list(YELLOW_FLAT)) is True, n.error_message
    assert len(sent) == 1


def test_the_replays_pass_their_timeout_as_the_budget():
    from ablations import replay_real_assembly as rra
    argv, timeout = rra.build_argv("rotate_object", {"object_name": "u_orange", "base_name": "base1"},
                                   rra.AssemblyState())
    assert timeout == rra.ROTATE_TIMEOUT_S == 90
    assert argv[argv.index("--budget-s") + 1] == "90"
    from ablations import replay_sim_assembly as rsa
    argv, timeout = rsa._command_for_step(
        "ros-mcp-server__rotate_object(object_name='u_orange', base_name='base1', mode='sim')")
    assert argv[argv.index("--budget-s") + 1] == str(timeout)
