"""move_plan: a planned move is sent exactly as planned (lead 2026-11-19: never wrap a joint target), and only when it
may be (review r1 of w8-realpath-prereq: the gate keys on the target, P1-1; no contact leg, P1-5; explicit flags,
P2-2; the final stance checked, P2-3; a tight start, P2-1; Cartesian timing, P1-5; no 10 s cap, P3-4)."""
import math

import numpy as np
from scipy.optimize import minimize

from primitives.core.move_plan import (CONTACT_CLEAR_M, START_TOL_RAD, move_refusal, plan_trajectory, start_refusal,
                                       target_refusal)
from primitives.shared.config import CART_VEL_LIMIT, GRIPPER_CENTER_TOOL_OFFSET, HOME_JOINTS, TABLE_HEIGHT
from primitives.shared.ik import dh_params, forward_kinematics

HOME = np.array(HOME_JOINTS, float)


def _gc(q):
    T = forward_kinematics(dh_params, np.asarray(q, float))
    return T[:3, :3] @ GRIPPER_CENTER_TOOL_OFFSET + T[:3, 3]


def _top_down_at(dz, seed=HOME):
    target = _gc(HOME).copy()
    target[2] = TABLE_HEIGHT + dz

    def f(q):
        T = forward_kinematics(dh_params, q)
        return 1e3 * np.sum((T[:3, :3] @ GRIPPER_CENTER_TOOL_OFFSET + T[:3, 3] - target) ** 2) + np.sum(
            (T[:3, 2] - [0, 0, -1]) ** 2)
    return minimize(f, seed, method="BFGS").x


def _line(q0, q1, n):
    return [list(q0 + (k / (n - 1)) * (q1 - q0)) for k in range(n)]


def test_the_gate_keys_on_where_the_goal_goes_P1_1():
    assert target_refusal("sim", True, {}) is None                              # the simulator's controller
    assert target_refusal("real", True, {}).startswith("MODE_TARGET_MISMATCH")
    # the real arm's controller (no ROS_MCP_DROIDSTACK_SIM): refused unless approved, whatever --mode says
    assert target_refusal("sim", False, {}).startswith("MODE_TARGET_MISMATCH")
    assert target_refusal("sim", False, {"ROS_MCP_SEQPLAN_REAL": "1"}).startswith("MODE_TARGET_MISMATCH")
    assert target_refusal("real", False, {}).startswith("SEQPLAN_REAL_NOT_APPROVED")
    assert target_refusal("real", False, {"ROS_MCP_SEQPLAN_REAL": "1"}) is None


def test_the_flags_are_required_and_a_contact_leg_is_refused_P2_2_P1_5():
    W = [list(HOME), list(HOME + 0.1)]
    assert move_refusal({"waypoints": W, "dense": False}).startswith("MOVE_FLAG_MISSING: 'contact'")
    assert move_refusal({"waypoints": W, "contact": False, "ends_in_grip": False}).startswith("MOVE_FLAG_MISSING: 'dense'")
    assert move_refusal({"waypoints": W, "dense": False, "contact": True, "ends_in_grip": False}).startswith("CONTACT_LEG")
    assert move_refusal({"waypoints": W, "dense": False, "contact": False, "ends_in_grip": False}) is None


def test_a_dense_leg_with_a_long_step_or_a_bad_waypoint_is_refused_P2_2_P3_2():
    W = _line(HOME, HOME + 0.2, 5)                                          # 0.05 rad steps
    assert move_refusal({"waypoints": W, "dense": True, "contact": False, "ends_in_grip": False}).startswith("DENSE_STEP")
    bad = [list(HOME), [float("nan")] * 6]
    assert move_refusal({"waypoints": bad, "dense": False, "contact": False, "ends_in_grip": False}).startswith("WAYPOINT_INVALID")
    assert start_refusal(HOME, [float("nan")] * 6).startswith("PLAN_START_INVALID")


def test_a_descent_to_the_table_needs_move_downs_contact_stop_P1_5():
    hover = _top_down_at(0.12)
    low = _top_down_at(0.010, hover)
    W = _line(hover, low, 1 + int(np.ceil(np.max(np.abs(low - hover)) / 0.015)))
    why = move_refusal({"waypoints": W, "dense": True, "contact": False, "ends_in_grip": False, "opening_m": 0.0})
    assert why.startswith("DESCENT_NEEDS_CONTACT_STOP"), why
    up = list(reversed(W))                                                  # the lift away is fine
    assert move_refusal({"waypoints": up, "dense": True, "contact": False, "ends_in_grip": False, "opening_m": 0.0}) is None
    assert CONTACT_CLEAR_M == 0.050


def test_the_final_stance_is_checked_against_the_table_P2_3():
    hover = _top_down_at(0.12)
    under = _top_down_at(-0.02, hover)
    why = move_refusal({"waypoints": [list(hover), list(under)], "dense": False, "contact": False, "ends_in_grip": False})
    assert why.startswith("GRIPPER_TABLE") and why.endswith("(final stance)")


def test_a_wrist3_target_past_180_deg_is_sent_unwrapped():
    start = np.zeros(6)
    target = np.array([0, 0, 0, 0, 0, math.radians(-300.0)])
    pts = plan_trajectory([start, target], dense=False)
    assert len(pts) == 1 and math.isclose(pts[0][0][5], math.radians(-300.0)) and pts[0][1] == [0.0] * 6
    from primitives.shared.velocity_profiles import compute_duration
    far = plan_trajectory([start, np.array([8.0, 0, 0, 0, 0, 0])], dense=False)[0][2]
    assert far > 10.0 and math.isclose(far, compute_duration(joint_distance=8.0, profile="s_curve",
                                                             max_duration=math.inf))   # no 10 s cap (P3-4)


def test_the_start_check_compares_unwrapped_values_and_is_tight_P2_1():
    plan_start = np.array([0, 0, 0, 0, 0, math.radians(-190.0)])
    measured_twin = np.array([0, 0, 0, 0, 0, math.radians(170.0)])           # the same pose, the other way round
    why = start_refusal(measured_twin, plan_start)
    assert why and why.startswith("PLAN_START_MISMATCH") and "360.00 deg apart" in why
    assert start_refusal(HOME + np.array([0, 0, 0, 0.001, 0.001, 0.002]), HOME) is None   # a settled arm
    assert start_refusal(HOME + 0.002, HOME).startswith("PLAN_START_MISMATCH: the gripper centre")  # 3.1 mm
    assert start_refusal(HOME + 0.005, HOME).startswith("PLAN_START_MISMATCH")   # was admitted at 0.01 rad
    assert START_TOL_RAD == 0.003


def test_a_route_stops_at_every_waypoint_and_a_dense_leg_keeps_its_samples():
    W = [np.zeros(6), np.full(6, 0.3), np.full(6, -0.2)]
    stops = plan_trajectory(W, dense=False)
    assert [p[0] for p in stops] == [list(map(float, W[1])), list(map(float, W[2]))]
    assert all(p[1] == [0.0] * 6 for p in stops) and stops[0][2] < stops[1][2]
    dense = plan_trajectory([HOME + 0.01 * k for k in range(10)], dense=True)
    assert len(dense) == 10 and np.allclose(dense[-1][0], HOME + 0.09)


def test_a_dense_leg_is_timed_at_the_cells_cartesian_rate_P1_5():
    hover = _top_down_at(0.40)
    low = _top_down_at(0.10, hover)
    W = _line(hover, low, 1 + int(np.ceil(np.max(np.abs(low - hover)) / 0.015)))
    T = plan_trajectory(W, dense=True)[-1][2]
    assert T >= 0.30 / CART_VEL_LIMIT                                       # 0.30 m at move_down's 0.0798 m/s



def test_ends_in_grip_is_a_required_flag_r4():
    """Lead 2026-11-20 (r4 P2-1): like dense and contact, a move states whether it ends in a Grip."""
    from primitives.core.move_plan import move_refusal
    from primitives.shared.config import HOME_JOINTS
    W = [list(HOME_JOINTS), list(HOME_JOINTS)]
    assert move_refusal({"waypoints": W, "dense": False, "contact": False}).startswith("MOVE_FLAG_MISSING: 'ends_in_grip'")
    assert move_refusal({"waypoints": W, "dense": False, "contact": False, "ends_in_grip": None}).startswith(
        "MOVE_FLAG_MISSING: 'ends_in_grip'")
