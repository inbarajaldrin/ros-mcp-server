"""seqplan's one checker: the hard limits every planned waypoint must keep (rules.route_refusal)."""
import numpy as np

from primitives.seqplan.rules import CABLE_WINDOW_DEG, JOINT_LIMITS_RAD, route_refusal


def test_a_waypoint_past_a_joint_limit_is_refused():
    """r49: a face-right descent took wrist_1 from -358 to -393 deg; the sim's controller (and the UR) refuse it."""
    a = np.radians([15, -36, 34, -358, -75, 0])
    b = np.radians([15, -16, 51, -393, -75, 0])
    why = route_refusal([a, b], samples=1)
    assert why.startswith("JOINT_LIMIT: joint 3")
    assert JOINT_LIMITS_RAD[2] == np.pi and all(v == 2 * np.pi for i, v in enumerate(JOINT_LIMITS_RAD) if i != 2)


def test_a_waypoint_outside_the_cable_window_is_refused():
    lo, hi = CABLE_WINDOW_DEG
    a = np.radians([90, -90, 90, -90, -90, 0])
    b = a.copy()
    b[5] = np.radians(hi + 5)
    why = route_refusal([a, b], samples=1)
    assert why.startswith(("CABLE_WINDOW", "WRIST3_LIMIT"))
