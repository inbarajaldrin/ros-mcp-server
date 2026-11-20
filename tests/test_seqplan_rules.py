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
        assert why and why.startswith(("GRIPPER_TABLE", "EXTENDED_GRIPPER", "EE_BELOW_BASE")), (dz, why)


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
    assert rules.travel_cap_refusal(HOME, c) is None                  # wrist_3's 3.2 rad cap is the insert's (r2 P1-A)


def test_sampling_is_per_distance_P2_4():
    a = HOME.copy()
    b = HOME.copy()
    b[0] += 1.0
    assert rules._samples(a, b) == 50
    assert rules._samples(a, a) == 1


def test_an_insert_leg_is_capped_from_its_start_as_translate_object_does_r2_P1_A():
    a = HOME.copy()
    for j in (3, 4):                                                   # wrist_1, wrist_2 +3.5 rad in one segment
        b = HOME.copy()
        b[j] += 3.5
        assert rules.insert_cap_refusal(a, b).startswith(f"INSERT_TRAVEL_CAP: joint {j}")
    mid, end = HOME.copy(), HOME.copy()
    mid[0] += 0.9 * np.pi
    end[0] += 1.8 * np.pi                                              # 0.9 pi twice: 1.8 pi from the start
    assert rules.travel_cap_refusal(a, mid) is None and rules.travel_cap_refusal(mid, end) is None
    assert rules.insert_cap_refusal(a, end).startswith("INSERT_TRAVEL_CAP: joint 0")
    w3 = HOME.copy()
    w3[5] += 3.25
    assert rules.insert_cap_refusal(a, w3).startswith("INSERT_TRAVEL_CAP: wrist_3")


def _tilted(deg):
    """A flange pose with the tool tilted `deg` in the jaw plane (tool x - z), 0.25 m above the table."""
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, :3] = R.from_euler("y", 180 - deg, degrees=True).as_matrix()
    T[2, 3] = TABLE_HEIGHT + 0.25
    return T


def test_an_unknown_opening_is_judged_over_the_stroke_r2_P1_B():
    from primitives.shared.gripper_rules import OPENING_GRID_MM, finger_boxes_tool, gripper_points_tool
    rng = np.random.default_rng(5)
    from scipy.spatial.transform import Rotation as R
    fine = np.arange(0.0, 110.0 + 1e-9, 0.01)
    worst = 0.0
    for k in range(200):
        Rm = R.random(random_state=rng).as_matrix() if k else _tilted(45)[:3, :3]
        zs_fine = min(float(np.min(np.array([[x, y, z] for lo, hi in finger_boxes_tool(w / 1000)
                                             for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                                   @ Rm[2])) for w in fine[::10])
        z_grid = float(np.min(gripper_points_tool(None) @ Rm[2]))
        worst = max(worst, z_grid - zs_fine)
    assert worst < 0.00005, worst                                      # within 0.05 mm of 0.1 mm sampling
    assert len(OPENING_GRID_MM) > 100


def test_the_45_deg_under_read_is_caught_r2_P1_B():
    from primitives.shared.gripper_rules import gripper_points_tool, gripper_table_refusal
    T = _tilted(45)
    z_grid = float(np.min(gripper_points_tool(None) @ T[2, :3]))
    z_ext = min(float(np.min(gripper_points_tool(w) @ T[2, :3])) for w in (0.0, 0.110))
    assert z_ext - z_grid > 0.010                                       # the extremes alone read >10 mm too high
    T[2, 3] = TABLE_HEIGHT + 0.001 - z_grid                             # lowest corner 1 mm above the table
    assert gripper_table_refusal(T, None).startswith("GRIPPER_TABLE")


def test_the_approach_exemption_holds_only_on_dense_segments_near_the_table_r2_P2_B():
    low = _top_down_at(0.010)
    assert route_refusal([low, low], leg="approach", opening_m=0.0) is None          # a pick stance: exempt
    far = HOME.copy()
    far[0] += 0.6
    why = route_refusal([far, low], leg="approach", opening_m=0.0)                   # a carry labelled approach
    assert why and why.startswith(("EE_BELOW_BASE", "COMPACT", "GRIPPER_TABLE", "EXTENDED", "ARM_")), why
