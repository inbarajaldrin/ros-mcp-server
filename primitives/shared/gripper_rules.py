"""Rules on the RG2 itself (fingers, gripper centre) and on the tool's orientation, for every planned stance.

The arm checks (collision.check_trajectory_collision) end at RG2_BODY_BOX (0.1436 m from the flange): the fingers out
to ~0.23 m were in no route check (review r1 of w8-realpath-prereq, P1-2).  These are pure geometry, no ROS:

- gripper_table_refusal: every corner of both finger boxes stays GRASP_TIP_CLEARANCE_M (2 mm, move_to_grasp's
  fingertip clearance) above the table.  A finger box (tool frame, opening w): x +-[w/2, w/2 + 20 mm] (pad, link and
  cover), y +-10.1 mm (the pad's 20.2 mm width), z 0.14 m to the fingertip (rg2_tip_length_m(w)).  The nominal
  gripper centre (0.2286 m) is not a material point: it lies 2.3 mm above the closed fingertips and in the empty
  space between the jaws when they are open, so the finger boxes bound it.
  Opening unknown (review r2 P1-B): the corners over OPENING_GRID_MM.  The fingertip depth is concave in the opening
  (4-bar), so a tilted tool's lowest corner can sit at an intermediate opening (14.7 mm under the extremes at 45
  deg).  A corner's height is linear in w/2 plus n_z * tip(w); sampled at step h it is under-read by at most
  h^2 / 8 * max|tip''| over the step.  tip''(w) = -55 / 110^2 * (1 - u^2)^(-3/2) mm/mm^2, u = (w + 3.96) / 110:
  0.13 /mm at 100 mm, steep only near the stroke's end, hence 1 mm steps to 100 mm and 0.1 mm steps to 110 mm;
  tests/test_seqplan_rules.py holds the grid within 0.05 mm of 0.01 mm sampling over random orientations.
- extended_gripper_refusal: rotate_object.check_extended_gripper_collision, the same numbers (the gripper centre below
  the table; the gripper centre inside the base column, r < 0.14 m below z 0.20 m; the flange -> gripper-centre
  segment within 0.11 m of the upper arm).  tests/test_seqplan_rules.py holds it to rotate_object's own method.
- orientation_refusal: rotate_object's EE-facing-robot (tool z . y > 0.5) and camera-toward--X (tool z . x < -0.5).
"""
from __future__ import annotations

import numpy as np

FINGER_THICK_M = 0.020
FINGER_HALF_WIDTH_M = 0.0101
FINGER_TOP_Z_M = 0.14
RG2_OPEN_MAX_M = 0.110
OPENING_GRID_MM = np.concatenate([np.arange(0.0, 100.0, 1.0), np.arange(100.0, 110.0 + 1e-9, 0.1)])


def finger_boxes_tool(opening_m):
    """Both finger boxes (lo, hi) in the tool0 frame at an opening (m)."""
    from primitives.shared.config import rg2_tip_length_m
    tip = rg2_tip_length_m(1000.0 * float(opening_m))
    h = float(opening_m) / 2.0
    return [(np.array([h, -FINGER_HALF_WIDTH_M, FINGER_TOP_Z_M]), np.array([h + FINGER_THICK_M, FINGER_HALF_WIDTH_M, tip])),
            (np.array([-h - FINGER_THICK_M, -FINGER_HALF_WIDTH_M, FINGER_TOP_Z_M]), np.array([-h, FINGER_HALF_WIDTH_M, tip]))]


def _corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


_GRID_POINTS = None


def gripper_points_tool(opening_m=None):
    """The finger-box corners (tool0) at `opening_m`, or over OPENING_GRID_MM when the opening is unknown."""
    global _GRID_POINTS
    if opening_m is not None:
        return np.concatenate([_corners(lo, hi) for lo, hi in finger_boxes_tool(float(opening_m))])
    if _GRID_POINTS is None:
        _GRID_POINTS = np.concatenate([_corners(lo, hi) for w in OPENING_GRID_MM
                                       for lo, hi in finger_boxes_tool(w / 1000.0)])
    return _GRID_POINTS


def gripper_table_refusal(T_flange, opening_m=None):
    """None, or a typed GRIPPER_TABLE refusal: the gripper centre or a fingertip is under the 2 mm clearance."""
    from primitives.shared.config import GRASP_TIP_CLEARANCE_M, TABLE_HEIGHT
    T = np.asarray(T_flange, float)
    z = gripper_points_tool(opening_m) @ T[2, :3] + T[2, 3]
    low = float(z.min()) - TABLE_HEIGHT
    if low < GRASP_TIP_CLEARANCE_M:
        return (f"GRIPPER_TABLE: the gripper/fingertips reach {1000 * low:+.1f} mm above the table "
                f"(needs {1000 * GRASP_TIP_CLEARANCE_M:.0f})")
    return None


def extended_gripper_refusal(q):
    """rotate_object.check_extended_gripper_collision as a pure function (same thresholds)."""
    from primitives.shared.collision import compute_all_joint_positions, segment_distance
    from primitives.shared.config import GRIPPER_CENTER_TOOL_OFFSET, TABLE_HEIGHT
    from primitives.shared.ik import dh_params, forward_kinematics
    q = np.asarray(q, float)
    jp = compute_all_joint_positions(q)
    T = forward_kinematics(dh_params, q)
    ee = T[:3, 3]
    tip = ee + T[:3, :3] @ np.asarray(GRIPPER_CENTER_TOOL_OFFSET, float)
    min_safe = 0.04 + 0.02
    if tip[2] < TABLE_HEIGHT:
        return f"EXTENDED_GRIPPER: gripper centre {1000 * (tip[2] - TABLE_HEIGHT):.1f} mm below the table"
    if np.linalg.norm(tip[:2]) < 0.08 + min_safe and tip[2] < 0.20:
        return "EXTENDED_GRIPPER: gripper centre inside the base column"
    d = segment_distance(ee, tip, np.array(jp[1]), np.array(jp[2]))
    if d < 0.05 + min_safe:
        return f"EXTENDED_GRIPPER: gripper {1000 * d:.0f} mm from the upper arm (needs {1000 * (0.05 + min_safe):.0f})"
    return None


def orientation_refusal(T_flange):
    """rotate_object's EE-facing-robot (tool z . y > 0.5) and camera-toward--X (tool z . x < -0.5) rules."""
    z = np.asarray(T_flange, float)[:3, 2]
    if z[1] > 0.5:
        return f"EE_FACING_ROBOT: tool z . y = {z[1]:.2f} (> 0.5)"
    if z[0] < -0.5:
        return f"CAMERA_TOWARD_MINUS_X: tool z . x = {z[0]:.2f} (< -0.5)"
    return None
