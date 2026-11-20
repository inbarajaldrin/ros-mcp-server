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


def _top_down_at(dz, seed=None):
    """Top-down, the home gripper centre's xy, the gripper centre dz above the table (the reviewer's check1)."""
    target = _gc(HOME).copy()
    target[2] = TABLE_HEIGHT + dz

    def f(q):
        T = _fk(q)
        return 1e3 * np.sum((T[:3, :3] @ GRIPPER_CENTER_TOOL_OFFSET + T[:3, 3] - target) ** 2) + np.sum(
            (T[:3, 2] - [0, 0, -1]) ** 2)
    return minimize(f, HOME if seed is None else seed, method="BFGS").x


def test_a_waypoint_past_a_joint_limit_is_refused():
    """r49: a face-right descent took wrist_1 from -358 to -393 deg; the sim's controller (and the UR) refuse it."""
    a = np.radians([15, -36, 34, -358, -75, 0])
    b = np.radians([15, -16, 51, -393, -75, 0])
    assert route_refusal([a, b], leg="approach", ends_in_grip=False).startswith("JOINT_LIMIT: joint 3")
    assert JOINT_LIMITS_RAD[2] == np.pi and all(v == 2 * np.pi for i, v in enumerate(JOINT_LIMITS_RAD) if i != 2)


def test_a_waypoint_outside_the_cable_window_is_refused():
    lo, hi = CABLE_WINDOW_DEG
    a = np.radians([90, -90, 90, -90, -90, 0])
    b = a.copy()
    b[5] = np.radians(hi + 5)
    assert route_refusal([a, b], leg="approach", ends_in_grip=False).startswith(("CABLE_WINDOW", "WRIST3_LIMIT"))


def test_the_leg_must_be_stated():
    with pytest.raises(TypeError):
        route_refusal([HOME, HOME])
    with pytest.raises(TypeError):                                   # r4 P2-1: ends_in_grip is required too
        route_refusal([HOME, HOME], leg="transit")
    with pytest.raises(ValueError):
        route_refusal([HOME, HOME], leg="transit", ends_in_grip=None)
    with pytest.raises(ValueError):
        route_refusal([HOME, HOME], leg="anything", ends_in_grip=False)


def test_an_empty_gripper_driven_into_the_table_is_refused_P1_2():
    hover = _top_down_at(0.10)
    assert route_refusal([hover, _top_down_at(0.03)], leg="approach", opening_m=0.0, ends_in_grip=False) is None   # tips ~28 mm up
    for dz in (0.0, -0.02, -0.05):                                     # the reviewer's cases: all were admitted
        why = route_refusal([hover, _top_down_at(dz)], leg="approach", opening_m=0.0, ends_in_grip=False)
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
    assert route_refusal([low, low], leg="approach", opening_m=0.0, ends_in_grip=False) is None
    for leg in ("transit", "insert"):
        assert route_refusal([low, low], leg=leg, opening_m=0.0, ends_in_grip=False).startswith("EE_BELOW_BASE")


def test_translate_objects_travel_caps_bind_every_leg_P1_4():
    a = HOME.copy()
    b = HOME.copy()
    b[0] += np.radians(185)
    assert route_refusal([a, b], leg="transit", ends_in_grip=False).startswith("TRAVEL_CAP: joint 0")
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
    """rg2_geometry's unknown opening (None) is the union over the stroke: never higher than any sampled opening."""
    from scipy.spatial.transform import Rotation as R
    from primitives.shared.gripper_rules import gripper_points_tool
    rng = np.random.default_rng(5)
    widths = np.arange(0.0, 0.110 + 1e-9, 0.0025)
    per_w = [gripper_points_tool(w) for w in widths]
    P_none = gripper_points_tool(None)
    for k in range(200):
        Rm = R.random(random_state=rng).as_matrix() if k else _tilted(45)[:3, :3]
        assert float(np.min(P_none @ Rm[2])) <= min(float(np.min(P @ Rm[2])) for P in per_w) + 1e-6


def test_the_45_deg_under_read_is_caught_r2_P1_B():
    from primitives.shared.gripper_rules import gripper_points_tool, gripper_table_refusal
    T = _tilted(45)
    z_grid = float(np.min(gripper_points_tool(None) @ T[2, :3]))
    z_ext = min(float(np.min(gripper_points_tool(w) @ T[2, :3])) for w in (0.0, 0.110))
    assert z_ext - z_grid > 0.005                                       # the extremes alone read too high
    T[2, 3] = TABLE_HEIGHT + 0.001 - z_grid                             # lowest corner 1 mm above the table
    assert gripper_table_refusal(T, None).startswith("GRIPPER_TABLE")


def test_the_approach_exemption_holds_only_on_dense_segments_near_the_table_r2_P2_B():
    low = _top_down_at(0.010)
    assert route_refusal([low, low], leg="approach", opening_m=0.0, ends_in_grip=False) is None          # a pick stance: exempt
    far = HOME.copy()
    far[0] += 0.6
    why = route_refusal([far, low], leg="approach", opening_m=0.0, ends_in_grip=False)                   # a carry labelled approach
    assert why and why.startswith(("EE_BELOW_BASE", "COMPACT", "GRIPPER_TABLE", "EXTENDED", "ARM_")), why


def test_the_housing_is_checked_against_the_table_r3():
    from primitives.shared.gripper_rules import gripper_table_refusal
    T = _tilted(90)                                                    # tool horizontal: the housing hangs lowest
    from primitives.shared.collision import RG2_BODY_BOX
    T[2, 3] = TABLE_HEIGHT + RG2_BODY_BOX[1][1] - 0.001                # the housing's side 1 mm into the table
    why = gripper_table_refusal(T, 0.0)
    assert why and why.startswith("GRIPPER_TABLE")


def _lift_from(dz_start):
    """A straight vertical lift (IK per 2 mm, as a planner's Cartesian line)."""
    W = [_top_down_at(dz_start)]
    for k in range(1, 26):
        W.append(_top_down_at(dz_start + 0.002 * k, W[-1]))
    return W


def test_a_lift_from_a_move_down_contact_is_exempt_only_while_it_rises_r3():
    from primitives.shared.gripper_rules import gripper_low_z
    W = _lift_from(0.0015)                                             # closed tips ~1 mm under the 2 mm floor
    T = _fk(W[0])
    assert gripper_low_z(T, 0.0) - TABLE_HEIGHT < 0.002
    assert route_refusal(W, leg="approach", opening_m=0.0, ends_in_grip=False).startswith("GRIPPER_TABLE")      # without the ruling
    assert route_refusal(W, leg="approach", opening_m=0.0, starts_at_contact=True, ends_in_grip=False) is None
    down = list(reversed(W))                                           # descending from it: never exempt
    assert route_refusal(down, leg="approach", opening_m=0.0, starts_at_contact=True, ends_in_grip=False) is not None
    deep = _lift_from(-0.012)                                          # still under the floor after 10 mm of rise
    why = route_refusal(deep, leg="approach", opening_m=0.0, starts_at_contact=True, ends_in_grip=False)
    assert why and why.startswith("LIFT_FROM_CONTACT"), why


def test_the_approach_exemption_needs_a_straight_segment_r3():
    low = _top_down_at(0.010)
    side = low.copy()
    side[5] += 0.04                                                    # a dense step that turns the tool
    assert not rules.straight_segment(low, side)
    assert route_refusal([low, side], leg="approach", opening_m=0.0, ends_in_grip=False).startswith("EE_BELOW_BASE")
    up = _top_down_at(0.04, low)
    assert rules.straight_segment(low, up)


# ---- review r3 P1-1 / P3-3 (round 4) ------------------------------------------------------------------------------
def _top_down_xy(dz, dx=0.0, seed=None):
    target = _gc(HOME).copy()
    target[0] += dx
    target[2] = TABLE_HEIGHT + dz

    def f(q):
        T = _fk(q)
        return 1e3 * np.sum((T[:3, :3] @ GRIPPER_CENTER_TOOL_OFFSET + T[:3, 3] - target) ** 2) + np.sum(
            (T[:3, 2] - [0, 0, -1]) ** 2)
    return minimize(f, HOME if seed is None else seed, method="BFGS").x


def _descent_to(dz, start=0.03):
    """A dense vertical descent (IK per 2 mm) ending with the gripper centre dz above the table."""
    W = [_top_down_xy(dz + start)]
    for k in range(1, int(round(start / 0.002)) + 1):
        W.append(_top_down_xy(dz + start - 0.002 * k, seed=W[-1]))
    return W


def test_a_pick_is_judged_where_the_jaws_close_r4():
    """Review r3 P1-1, the u_orange case: a 95.2 mm raw approach whose open tips end 3 mm above the table passes as an
    approach, but the close to ~30 mm raw drives the tips > 20 mm down, into the table.  A route that ends in a Grip is
    judged at the close width (grasp_stance_refusal); unknown close width: fully closed."""
    from primitives.core.move_plan import move_refusal
    from primitives.shared import rg2_geometry as g
    from primitives.shared.gripper_rules import grasp_stance_refusal
    open_raw, close_raw = 0.0952, 0.0302
    gc_off = float(GRIPPER_CENTER_TOOL_OFFSET[2])
    assert g.tip_z_tool(close_raw) - g.tip_z_tool(open_raw) > 0.020        # the close drives the tips > 20 mm down
    W = _descent_to(0.003 - (gc_off - g.tip_z_tool(open_raw)))             # open tips 3 mm above the table
    assert route_refusal(W, leg="approach", opening_m=open_raw, ends_in_grip=False) is None     # judged at the approach opening only
    why = route_refusal(W, leg="approach", opening_m=open_raw, close_width_m=close_raw, ends_in_grip=True)
    assert why and why.startswith("GRASP_STANCE"), why
    assert route_refusal(W, leg="approach", opening_m=open_raw, ends_in_grip=True).startswith("GRASP_STANCE")
    assert grasp_stance_refusal(_fk(W[-1]), close_raw).startswith("GRASP_STANCE")
    move = {"waypoints": [w.tolist() for w in W], "dense": True, "contact": False, "ends_in_grip": True, "opening_m": open_raw,
            "close_width_m": close_raw}
    assert move_refusal(move).startswith("GRASP_STANCE")
    W_ok = _descent_to(0.003 - (gc_off - g.tip_z_tool(close_raw)))         # the CLOSED tips end 3 mm above the table
    assert route_refusal(W_ok, leg="approach", opening_m=open_raw, close_width_m=close_raw, ends_in_grip=True) is None


def test_the_lift_exemption_itself_needs_a_straight_segment_r4():
    """Review r3 P3-3: a level drag along the table from a contact stance is refused by the lift rule itself, not only
    by EE-below-base."""
    a = _top_down_xy(0.0015)
    W = [a]
    for k in range(1, 31):
        W.append(_top_down_xy(0.0015, dx=0.002 * k, seed=W[-1]))
    why = route_refusal(W, leg="approach", opening_m=0.0, starts_at_contact=True, ends_in_grip=False)
    assert why and why.startswith("LIFT_FROM_CONTACT: not a straight"), why
