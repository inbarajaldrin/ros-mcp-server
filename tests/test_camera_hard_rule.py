"""Camera away from the table: hard, fail-closed rule (lead 2026-11-19, operator after r48c)."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from primitives.shared import camera_up as cu
from primitives.shared.config import TABLE_HEIGHT
from primitives.shared.ik import dh_params, forward_kinematics

ROOT = Path(__file__).resolve().parents[1]
# MuJoCo r48c: u_orange place-down end (camera toward the table, D415 through it) and the high carry before it.
R48C_PLACE_DOWN = [-0.641109, -0.916988, 2.454026, -1.577309, -0.633258, -3.091288]
R48C_HIGH_CARRY = [-0.640855, -1.780168, 2.188927, -0.448674, -0.633091, -3.091574]


def _pose(R_flange, gripper_height_m):
    T = np.eye(4)
    T[:3, :3] = R_flange
    T[:3, 3] = [0.3, -0.4, TABLE_HEIGHT + gripper_height_m] - R_flange @ np.array([0, 0, 0.2286])
    return T


def test_r48c_place_down_is_refused_and_the_high_carry_is_not():
    reason = cu.camera_stance_refusal(forward_kinematics(dh_params, R48C_PLACE_DOWN))
    assert reason and reason.startswith("CAMERA_CLEARANCE")
    assert cu.camera_stance_refusal(forward_kinematics(dh_params, R48C_HIGH_CARRY)) is None
    # ...but the same stance as a place-down (carried to the table) is refused by the side rule
    assert cu.camera_stance_refusal(forward_kinematics(dh_params, R48C_HIGH_CARRY), near_table=True) \
        .startswith("CAMERA_TOWARD_TABLE")


def test_side_rule_near_the_table_only_for_sideways_stances():
    # Sideways stance: tool z horizontal (world +x); tool y (the camera side, +Y) world up or down.
    def sideways(camera_up):
        y = np.array([0.0, 0.0, 1.0 if camera_up else -1.0])
        z = np.array([1.0, 0.0, 0.0])
        return np.column_stack([np.cross(y, z), y, z])
    assert cu.camera_z(sideways(True)) > 0.99 and cu.camera_z(sideways(False)) < -0.99
    assert cu.camera_stance_refusal(_pose(sideways(True), 0.10)) is None
    assert cu.camera_stance_refusal(_pose(sideways(False), 0.10)).startswith("CAMERA_CLEARANCE")   # mount in the table
    assert cu.camera_stance_refusal(_pose(sideways(False), 0.148)).startswith("CAMERA_TOWARD_TABLE")  # clear, but side
    assert cu.camera_stance_refusal(_pose(sideways(False), 0.40)) is None          # high: side rule off
    top_down = R.from_euler("x", 180, degrees=True).as_matrix()
    assert cu.camera_stance_refusal(_pose(top_down, 0.05)) is None                # top-down pick: allowed


def test_every_gate_is_wired_and_fails_closed():
    grasp = (ROOT / "primitives/move_to_grasp.py").read_text()
    assert "return 'refuse'" in grasp and grasp.count("CAMERA_TOWARD_TABLE:") == 2
    assert "falling back to the other wrist" not in grasp
    assert "camera_route_refusal(self.current_joint_angles, joint_angles, near_table_end=True)" in \
        (ROOT / "primitives/rotate_object.py").read_text()
    clear = (ROOT / "primitives/core/move_to_clear_area.py").read_text()
    assert "camera_route_refusal(start_joints, target_joints, near_table_end=True)" in clear
    assert "camera_route_refusal(home, tj, near_table_end=True)" in clear
    assert "camera_stance_refusal(forward_kinematics(dh_params, np.asarray(waypoints[-1], float)))" in \
        (ROOT / "primitives/core/move_down.py").read_text()
    assert '"--base-name", "base1"' in (ROOT / "ablations/replay_sim_assembly.py").read_text()
