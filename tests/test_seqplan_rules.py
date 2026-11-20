"""seqplan's one checker: the hard limits and the rules every planned route keeps (rules.route_refusal).
Review r1 of w8-realpath-prereq: the gripper and fingertips against the table (P1-2), rotate_object's stance rules at
every stance, the last included (P1-3), translate_object's travel caps (P1-4), sampling per distance (P2-4)."""
import numpy as np
import pytest
from scipy.optimize import minimize

from primitives.seqplan import rules
from primitives.seqplan.rules import CABLE_WINDOW_DEG, JOINT_LIMITS_RAD, route_refusal
from primitives.shared.config import GRIPPER_CENTER_TOOL_OFFSET, HOME_JOINTS, TABLE_HEIGHT
from primitives.shared.ik import dh_params, forward_kinematics

HOME = np.array(HOME_JOINTS, float)


def _fk(q):
    return forward_kinematics(dh_params, np.asarray(q, float))


def _gc(q):
    T = _fk(q)
    return T[:3, :3] @ GRIPPER_CENTER_TOOL_OFFSET + T[:3, 3]


def _top_down_at(dz):
    """Top-down, the home gripper centre's xy, the gripper centre dz above the table (the reviewer's check1)."""
    target = _gc(HOME).copy()
    target[2] = TABLE_HEIGHT + dz

    def f(q):
        T = _fk(q)
        return 1e3 * np.sum((T[:3, :3] @ GRIPPER_CENTER_TOOL_OFFSET + T[:3, 3] - target) ** 2) + np.sum(
            (T[:3, 2] - [0, 0, -1]) ** 2)
    return minimize(f, HOME, method="BFGS").x


def test_a_waypoint_past_a_joint_limit_is_refused():
    """r49: a face-right descent took wrist_1 from -358 to -393 deg; the sim's controller (and the UR) refuse it."""
    a = np.radians([15, -36, 34, -358, -75, 0])
    b = np.radians([15, -16, 51, -393, -75, 0])
    assert route_refusal([a, b], leg="approach").startswith("JOINT_LIMIT: joint 3")
    assert JOINT_LIMITS_RAD[2] == np.pi and all(v == 2 * np.pi for i, v in enumerate(JOINT_LIMITS_RAD) if i != 2)


def test_a_waypoint_outside_the_cable_window_is_refused():
    lo, hi = CABLE_WINDOW_DEG
    a = np.radians([90, -90, 90, -90, -90, 0])
    b = a.copy()
    b[5] = np.radians(hi + 5)
    assert route_refusal([a, b], leg="approach").startswith(("CABLE_WINDOW", "WRIST3_LIMIT"))


def test_the_leg_must_be_stated():
    with pytest.raises(TypeError):
        route_refusal([HOME, HOME])
    with pytest.raises(ValueError):
        route_refusal([HOME, HOME], leg="anything")


def test_an_empty_gripper_driven_into_the_table_is_refused_P1_2():
    hover = _top_down_at(0.10)
    assert route_refusal([hover, _top_down_at(0.03)], leg="approach", opening_m=0.0) is None   # tips ~28 mm up
    for dz in (0.0, -0.02, -0.05):                                     # the reviewer's cases: all were admitted
        why = route_refusal([hover, _top_down_at(dz)], leg="approach", opening_m=0.0)
        assert why and why.startswith(("GRIPPER_TABLE", "EXTENDED_GRIPPER")), (dz, why)


def test_the_extended_gripper_check_is_rotate_objects_P1_3():
    import tests.ros_stubs as ros_stubs
    ros_stubs.install()
    from primitives import rotate_object as ro
    node = object.__new__(ro.ReorientForAssembly)
    node.tcp_to_gripper_center_offset = GRIPPER_CENTER_TOOL_OFFSET
    rng = np.random.default_rng(3)
    from primitives.shared.gripper_rules import extended_gripper_refusal
    agree = 0
    for _ in range(300):
        q = HOME + rng.normal(0, 0.8, 6)
        assert (extended_gripper_refusal(q) is not None) == bool(node.check_extended_gripper_collision(q))
        agree += 1
    assert agree == 300


def test_the_orientation_rules_apply_at_every_stance_the_last_included_P1_3():
    from primitives.shared.gripper_rules import orientation_refusal
    T = np.eye(4)
    T[:3, 2] = [0.0, 0.8, -0.6]
    assert orientation_refusal(T).startswith("EE_FACING_ROBOT")
    T[:3, 2] = [-0.8, 0.0, -0.6]
    assert orientation_refusal(T).startswith("CAMERA_TOWARD_MINUS_X")
    T[:3, 2] = [0.0, 0.0, -1.0]
    assert orientation_refusal(T) is None


def test_ee_below_base_binds_transit_and_insert_legs_not_approach_legs_P1_3():
    low = _top_down_at(0.010)                                         # a pick: the flange below 0.1625 m, tips 8 mm up
    assert _fk(low)[2, 3] < 0.1625
    assert route_refusal([low, low], leg="approach", opening_m=0.0) is None
    for leg in ("transit", "insert"):
        assert route_refusal([low, low], leg=leg, opening_m=0.0).startswith("EE_BELOW_BASE")


def test_translate_objects_travel_caps_bind_every_leg_P1_4():
    a = HOME.copy()
    b = HOME.copy()
    b[0] += np.radians(185)
    assert route_refusal([a, b], leg="transit").startswith("TRAVEL_CAP: joint 0")
    c = HOME.copy()
    c[5] = 3.25
    assert rules.travel_cap_refusal(HOME, c).startswith("TRAVEL_CAP: wrist_3")


def test_sampling_is_per_distance_P2_4():
    a = HOME.copy()
    b = HOME.copy()
    b[0] += 1.0
    assert rules._samples(a, b) == 50
    assert rules._samples(a, a) == 1
