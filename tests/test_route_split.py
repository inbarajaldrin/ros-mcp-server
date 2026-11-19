"""Split reorient routes (lead approval 2026-11-19): r48e's u_orange reorient, camera kept up."""
from pathlib import Path

import numpy as np

from primitives.shared.camera_up import camera_route_refusal
from primitives.shared.route_split import admissible_route, candidate_routes

ROOT = Path(__file__).resolve().parents[1]
# MuJoCo r48d/r48e: the planned single-point u_orange reorient (start measured, target recovered from the
# interpolated commands and the logged 2.75 rad max joint change).
START = np.array([1.89727, -1.30194, 1.700762, -1.8321, -1.803826, 1.924308])
TARGET = np.array([1.963492, -1.267425, 1.652992, -0.238788, -4.553826, -0.180144])
# MuJoCo r48g: the held u_orange's ground-truth pose (base_link) at the refused clear-area carry.
R48G_PART_P = [0.279324, -0.725317, 0.298128]
R48G_PART_XYZW = [0.70435737, -0.00378221, 0.0045327, -0.70982099]


def rule(a, b, is_last):
    return camera_route_refusal(a, b, near_table_end=is_last)


def test_direct_line_is_refused_and_a_detour_route_is_admitted():
    """With the conservative link radii and the mount's faces sampled (no corner-only blind spot), every
    joint-group split of r48e's reorient sweeps the mount within 10 mm of the forearm too; a +90 deg wrist_3
    detour turns it away."""
    assert camera_route_refusal(START, TARGET, near_table_end=True).startswith("CAMERA_MOUNT_ARM_COLLISION")
    route, _ = admissible_route(START, TARGET, rule)
    assert route is not None and len(route) == 4
    assert np.allclose(route[0], START) and np.allclose(route[-1], TARGET)
    assert np.isclose(route[1][5] - START[5], np.pi / 2) and np.allclose(route[1][:5], START[:5])
    assert max(abs(w[5]) for w in route) <= 4.0
    travel = sum(np.sum(np.abs(b - a)) for a, b in zip(route[:-1], route[1:]))
    assert travel <= np.sum(np.abs(TARGET - START)) + np.pi + 1e-9      # the direct travel plus the detour out and back


def test_an_admissible_direct_line_is_kept_and_none_refuses():
    small = START + np.array([0, 0, 0, 0, 0, 0.1])
    route, _ = admissible_route(START, small, lambda a, b, last: camera_route_refusal(a, b))
    assert len(route) == 2
    route, reasons = admissible_route(START, TARGET, lambda a, b, last: "ALWAYS_REFUSED")
    assert route is None and reasons and all(r == "ALWAYS_REFUSED" for r in reasons)
    # group splits + the wrist_3 detours that keep |wrist_3| <= 4 rad (+135 and +180 deg exceed it from 1.92 rad)
    assert sum(1 for _ in candidate_routes(START, TARGET)) == 75 + 4


def _reorient_node():
    """rotate_object's planner on an instance built without a ROS node (its checks are pure geometry)."""
    import tests.ros_stubs as ros_stubs
    ros_stubs.install()
    import primitives.rotate_object as ro
    node = object.__new__(ro.ReorientForAssembly)
    node.tcp_to_gripper_center_offset = ro.GRIPPER_CENTER_TOOL_OFFSET
    return node


# Both stances pass the reorient IK rules and the direct joint line is clear of the table and the arm, but moving
# wrist_1 + wrist_2 first stops at a stance that self-collides (DH capsules, conservative radii).
SPLIT_START = np.array([1.914, -1.7228, 2.761, -2.7811, -1.4732, 2.5738])
SPLIT_TARGET = np.array([2.163, -1.668, 1.633, -0.1665, -3.7406, -0.1339])
SPLIT_MIDDLE = np.array([1.914, -1.7228, 2.761, -0.1665, -3.7406, 2.5738])


def test_rotate_object_refuses_a_split_whose_middle_stance_self_collides(monkeypatch):
    from primitives.shared import route_split
    from primitives.shared.collision import check_self_collision, check_trajectory_collision
    from primitives.shared.config import TABLE_HEIGHT
    node = _reorient_node()
    assert not node._check_collision(SPLIT_START) and not node._check_collision(SPLIT_TARGET)
    assert not check_trajectory_collision(SPLIT_START, SPLIT_TARGET, z_threshold=TABLE_HEIGHT)
    assert check_self_collision(SPLIT_MIDDLE)
    # Offer the planner only that split, as if the camera rule had refused everything else.
    monkeypatch.setattr(route_split, "candidate_routes",
                        lambda start, target, vias=(): iter([[start, SPLIT_MIDDLE.copy(), target]]))
    route, reasons = node._plan_reorient_route(SPLIT_START, SPLIT_TARGET, None)
    assert route is None
    assert reasons and reasons[0].startswith("ARM_TABLE_OR_SELF_COLLISION")


def test_rotate_object_routes_pass_the_arm_checks_on_every_segment_and_stance():
    from primitives.shared.collision import check_trajectory_collision
    from primitives.shared.config import TABLE_HEIGHT
    node = _reorient_node()
    for start, target in ((START, TARGET), (SPLIT_START, SPLIT_TARGET)):
        route, reasons = node._plan_reorient_route(start, target, None)
        assert route is not None, reasons
        assert np.allclose(route[0], start) and np.allclose(route[-1], target)
        for a, b in zip(route[:-1], route[1:]):
            assert not check_trajectory_collision(a, b, z_threshold=TABLE_HEIGHT, num_samples=20)
            assert not node._check_collision(b)
            assert camera_route_refusal(a, b) is None


def test_rotate_object_refuses_a_stance_the_reorient_ik_rules_refuse(monkeypatch):
    """A route stance that clears the arm capsules but fails another IK rule (here: the planner is told the
    middle stance faces the robot) is refused too."""
    from primitives.shared import route_split
    node = _reorient_node()
    middle = START.copy()
    middle[5] = TARGET[5]
    real_check = node._check_collision
    monkeypatch.setattr(node, "_check_collision", lambda q: np.allclose(q, middle) or real_check(q))
    monkeypatch.setattr(route_split, "candidate_routes",
                        lambda start, target, vias=(): iter([[start, middle.copy(), target]]))
    route, reasons = node._plan_reorient_route(START, TARGET, None)
    assert route is None and reasons[0].startswith("ARM_STANCE_REFUSED")


def test_r48f_clear_area_carry_needs_a_wrist3_detour():
    """MuJoCo r48f: camera up, the clear-area carry's wrist_2 sweep passes the mount through the forearm /
    wrist_1 zone on every direct or group-split route; a wrist_3 detour turns the mount away."""
    from primitives.shared.collision import check_trajectory_collision
    from primitives.shared.config import HOME_JOINTS, TABLE_COLLISION_MARGIN_SIDEWAYS, TABLE_HEIGHT
    start = np.array([1.9577104621800272, -1.2703251056487066, 1.6572930970066544,
                      -0.3777497059349086, -4.313742523402642, 0.003576458218420753])
    target = np.array([-0.640768, -1.767599, 2.185283, -0.420121, -0.633356, 0.003576])

    def seg(a, b, last):
        if check_trajectory_collision(a, b, z_threshold=TABLE_HEIGHT - TABLE_COLLISION_MARGIN_SIDEWAYS, num_samples=20):
            return "ARM"
        return camera_route_refusal(a, b, near_table_end=last)

    assert seg(start, target, True) is not None
    home = np.array(HOME_JOINTS, float)
    assert seg(start, home, False) or seg(home, target, True)
    route, _ = admissible_route(start, target, seg)
    assert route is not None and len(route) == 4
    assert abs(route[1][5] - start[5]) > 1.0 and np.allclose(route[-1], target)
    assert max(abs(w[5]) for w in route) <= 4.0


def test_r48g_carry_needs_home_plus_a_wrist3_detour_for_the_held_part():
    """MuJoCo r48g: the detours alone swing the held u_orange into the upper arm, the via-home route sweeps the
    mount through wrist_1; via home WITH a wrist_3 detour passes every rule."""
    from scipy.spatial.transform import Rotation as Rot
    from primitives.shared.collision import check_trajectory_collision
    from primitives.shared.config import HOME_JOINTS, TABLE_COLLISION_MARGIN_SIDEWAYS, TABLE_HEIGHT
    from primitives.shared.held_part_collision import held_part_model, held_part_route_refusal
    from primitives.shared.ik import dh_params, forward_kinematics
    start = np.array([1.9577104621800272, -1.2703251056487066, 1.6572930970066544,
                      -0.3777497059349086, -4.313742523402642, 0.003576458218420753])
    target = np.array([-0.640768, -1.767599, 2.185283, -0.420121, -0.633356, 0.003576])
    T_part = np.eye(4)
    T_part[:3, :3] = Rot.from_quat(R48G_PART_XYZW).as_matrix()
    T_part[:3, 3] = R48G_PART_P
    held = held_part_model("u_orange", forward_kinematics(dh_params, start), T_part=T_part)

    def seg(a, b, last):
        if check_trajectory_collision(a, b, z_threshold=TABLE_HEIGHT - TABLE_COLLISION_MARGIN_SIDEWAYS, num_samples=20):
            return "ARM"
        return camera_route_refusal(a, b, near_table_end=last) or held_part_route_refusal(a, b, held)

    route, _ = admissible_route(start, target, seg)
    assert route is None
    route, _ = admissible_route(start, target, seg, vias=(np.array(HOME_JOINTS, float),))
    assert route is not None and len(route) == 5 and np.allclose(route[2][:5], np.array(HOME_JOINTS)[:5])
