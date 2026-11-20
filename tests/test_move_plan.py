"""move_plan: a planned move is sent exactly as planned (lead 2026-11-19: never wrap a joint target)."""
import math

import numpy as np

from primitives.core.move_plan import plan_trajectory, start_refusal


def test_a_wrist3_target_past_180_deg_is_sent_unwrapped():
    start = np.zeros(6)
    target = np.array([0, 0, 0, 0, 0, math.radians(-300.0)])
    pts = plan_trajectory([start, target], dense=False)
    assert len(pts) == 1 and math.isclose(pts[0][0][5], math.radians(-300.0)) and pts[0][1] == [0.0] * 6


def test_the_start_check_compares_unwrapped_values():
    plan_start = np.array([0, 0, 0, 0, 0, math.radians(-190.0)])
    measured_twin = np.array([0, 0, 0, 0, 0, math.radians(170.0)])           # the same pose, the other way round
    why = start_refusal(measured_twin, plan_start)
    assert why and why.startswith("PLAN_START_MISMATCH") and "360.00 deg apart" in why
    assert start_refusal(plan_start + 0.005, plan_start) is None


def test_a_route_stops_at_every_waypoint_and_a_dense_leg_keeps_its_samples():
    W = [np.zeros(6), np.full(6, 0.3), np.full(6, -0.2)]
    stops = plan_trajectory(W, dense=False)
    assert [p[0] for p in stops] == [list(map(float, W[1])), list(map(float, W[2]))]
    assert all(p[1] == [0.0] * 6 for p in stops) and stops[0][2] < stops[1][2]
    dense = plan_trajectory([np.zeros(6) + 0.01 * k for k in range(10)], dense=True)
    assert len(dense) == 10 and np.allclose(dense[-1][0], 0.09)
