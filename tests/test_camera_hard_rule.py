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
    assert "camera_route_refusal(a, b, near_table_end=is_last)" in (ROOT / "primitives/rotate_object.py").read_text()
    clear = (ROOT / "primitives/core/move_to_clear_area.py").read_text()
    assert "camera_route_refusal(start_joints, target_joints, near_table_end=True)" in clear
    assert "camera_route_refusal(home, tj, near_table_end=True)" in clear
    assert "camera_stance_refusal(forward_kinematics(dh_params, np.asarray(waypoints[-1], float)))" in \
        (ROOT / "primitives/core/move_down.py").read_text()
    assert '"--base-name", "base1"' in (ROOT / "ablations/replay_sim_assembly.py").read_text()




def test_mount_vs_arm_refuses_r48c_yellow_reorient_and_admits_flown_poses():
    # r48c (MuJoCo) t=255.2 s: a camera-up yellow reorient that drove the v3.1 mount into the forearm/wrist_1.
    q_bad = [2.0117, -1.5065, 1.7891, -0.8216, -3.1744, 0.0055]
    from primitives.shared.camera_up import camera_mount_arm_refusal
    home = [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
    assert camera_mount_arm_refusal(home) is None
    assert camera_mount_arm_refusal(R48C_HIGH_CARRY) is None
    reason = camera_mount_arm_refusal(q_bad)
    assert reason is not None and reason.startswith("CAMERA_MOUNT_ARM_COLLISION")
    assert cu.camera_route_refusal(home, q_bad) is not None


def test_mount_surface_sampling_bounds_the_true_box_distance():
    """Review P2-1: corners and centres miss mid-face/mid-edge points (up to ~16 mm on the 32 mm boxes).  The face
    grid (<= 5 mm) with its gap subtracted never overstates the clearance."""
    import numpy as np
    from primitives.shared.camera_geometry import CAMERA_MOUNT_BOXES_TOOL, mount_surface_tool
    surface, gap, _owner = mount_surface_tool()
    assert gap <= 0.0036
    boxes = [(np.asarray(lo), np.asarray(hi)) for bs in CAMERA_MOUNT_BOXES_TOOL.values() for lo, hi in bs]
    corners = np.array([[x, y, z] for lo, hi in boxes for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])] + [(lo + hi) / 2 for lo, hi in boxes])
    rng = np.random.default_rng(0)
    lo_all = np.min([lo for lo, _ in boxes], axis=0) - 0.03
    hi_all = np.max([hi for _, hi in boxes], axis=0) + 0.03
    worst_corner_miss = 0.0
    for p in rng.uniform(lo_all, hi_all, size=(400, 3)):
        exact = min(float(np.linalg.norm(np.maximum(np.maximum(lo - p, 0.0), p - hi))) for lo, hi in boxes)
        if exact == 0.0:
            continue                                    # inside a box
        sampled = float(np.min(np.linalg.norm(surface - p, axis=1)))
        assert sampled - gap <= exact + 1e-9 and exact <= sampled + 1e-9
        worst_corner_miss = max(worst_corner_miss, float(np.min(np.linalg.norm(corners - p, axis=1))) - exact)
    assert worst_corner_miss > 0.010                    # what the corner sampling could hide


def test_mount_vs_arm_prefilter_matches_the_full_surface():
    """The bounding-sphere prefilter only skips boxes that clear the margin: same verdict as every sample."""
    import numpy as np
    from primitives.shared.camera_geometry import mount_surface_tool
    from primitives.shared.camera_up import CAMERA_MOUNT_ARM_MARGIN_M, MOUNT_ARM_LINKS, camera_mount_arm_refusal
    from primitives.shared.collision import UR5E_LINK_RADII, compute_all_joint_positions
    from primitives.shared.ik import dh_params, forward_kinematics
    surface, gap, _owner = mount_surface_tool()
    rng = np.random.default_rng(3)
    refused = 0
    for _ in range(300):
        q = np.array([1.9, -1.4, 1.7, -0.8, -3.1, 0.0]) + rng.normal(0, 0.8, 6)
        T = forward_kinematics(dh_params, q)
        W = surface @ T[:3, :3].T + T[:3, 3]
        J = [np.asarray(p) for p in compute_all_joint_positions(q)]
        brute = False
        for link in range(MOUNT_ARM_LINKS):
            a, ab = J[link], J[link + 1] - J[link]
            t = np.clip(((W - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
            d = float(np.min(np.linalg.norm(W - (a + t[:, None] * ab), axis=1))) - gap - UR5E_LINK_RADII[link]
            if d < CAMERA_MOUNT_ARM_MARGIN_M:
                brute = True
                break
        assert (camera_mount_arm_refusal(q) is not None) == brute
        refused += brute
    assert 0 < refused < 300
