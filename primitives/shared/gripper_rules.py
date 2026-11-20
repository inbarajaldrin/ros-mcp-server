"""Rules on the RG2 itself (housing, quick changer, fingers) and on the tool's orientation, for every planned stance.

The arm checks (collision.check_trajectory_collision, arm_table_refusal) cover the arm; "the gripper is the grasp's own
business" there (review r1 of w8-realpath-prereq, P1-2).  These are pure geometry, no ROS:

- gripper_table_refusal: every point of gripper_points_tool stays GRASP_TIP_CLEARANCE_M (2 mm, move_to_grasp's
  fingertip clearance) above the table.  The points (tool0):
  * the finger boxes from primitives.shared.rg2_geometry (the ONE RG2 finger model, shared with collision.py's
    tier 1: the six pinned finger-link meshes posed by the 4-bar, each box enclosing its link over the opening's
    range and padded for the travel between samples; opening None = the whole stroke).  Its closed fingertip is the
    mesh's 234.2 mm below the flange, 2.7 mm deeper than config's measured RG2_TIP_CLOSED_M; the deeper figure
    stands until the real fingertip is measured (lead ruling 2026-11-20);
  * the RG2 housing (collision.RG2_BODY_BOX, flange z 0.014-0.144 m) and the quick changer (r 0.035 m, z 0-0.0185 m)
    as box corners (review r2 round 3: nothing between the fingers and the flange is left unchecked).
  The nominal gripper centre (0.2286 m) is not a material point: it lies in the empty space between open jaws and
  above the closed fingertips, so the finger boxes bound it.  The model unavailable (a mesh missing or not its pin):
  RG2_MODEL_UNAVAILABLE, a refusal.
- extended_gripper_refusal: rotate_object.check_extended_gripper_collision, the same numbers (the gripper centre below
  the table; the gripper centre inside the base column, r < 0.14 m below z 0.20 m; the flange -> gripper-centre
  segment within 0.11 m of the upper arm).  tests/test_seqplan_rules.py holds it to rotate_object's own method.
- orientation_refusal: rotate_object's EE-facing-robot (tool z . y > 0.5) and camera-toward--X (tool z . x < -0.5).
"""
from __future__ import annotations

import numpy as np

RG2_OPEN_MAX_M = 0.110
QUICK_CHANGER_BOX_TOOL = ((-0.035, -0.035, 0.0), (0.035, 0.035, 0.0185))


def _corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def finger_boxes_tool(opening_m=None):
    """The ONE RG2 finger model (rg2_geometry.finger_boxes_tool); None when the model is unavailable."""
    from primitives.shared.rg2_geometry import finger_boxes_tool as fb
    return fb(opening_m)


def _housing_points():
    from primitives.shared.collision import RG2_BODY_BOX
    c, h = np.asarray(RG2_BODY_BOX[0], float), np.asarray(RG2_BODY_BOX[1], float)
    return np.concatenate([_corners(c - h, c + h), _corners(*map(np.asarray, QUICK_CHANGER_BOX_TOOL))])


def gripper_points_tool(opening_m=None):
    """The finger-box corners over the opening's range plus the housing and quick-changer corners (tool0), or None
    when the finger model is unavailable."""
    from primitives.shared.rg2_geometry import worst_finger_points_tool
    f = worst_finger_points_tool(opening_m)
    if f is None:
        return None
    return np.concatenate([np.asarray(f, float), _housing_points()])


def gripper_low_z(T_flange, opening_m=None):
    """The lowest gripper point's height in the base frame, or None (model unavailable)."""
    P = gripper_points_tool(opening_m)
    if P is None:
        return None
    T = np.asarray(T_flange, float)
    return float((P @ T[2, :3]).min() + T[2, 3])


def gripper_table_refusal(T_flange, opening_m=None):
    """None, or a typed refusal: a gripper point under the 2 mm clearance, or the finger model unavailable."""
    from primitives.shared.config import GRASP_TIP_CLEARANCE_M, TABLE_HEIGHT
    z = gripper_low_z(T_flange, opening_m)
    if z is None:
        from primitives.shared.rg2_geometry import why_unavailable
        return f"RG2_MODEL_UNAVAILABLE: {why_unavailable()}"
    low = z - TABLE_HEIGHT
    if low < GRASP_TIP_CLEARANCE_M:
        return (f"GRIPPER_TABLE: the gripper/fingertips reach {1000 * low:+.1f} mm above the table "
                f"(needs {1000 * GRASP_TIP_CLEARANCE_M:.0f})")
    return None


def extended_gripper_refusal(q, table_check=True):
    """rotate_object.check_extended_gripper_collision as a pure function (same thresholds).  table_check=False drops
    its first clause (the nominal gripper centre below the table): that point is empty space between open jaws (a pick
    with the jaws open is refused by it at -0.4 mm), and gripper_table_refusal checks the fingers themselves."""
    from primitives.shared.collision import compute_all_joint_positions, segment_distance
    from primitives.shared.config import GRIPPER_CENTER_TOOL_OFFSET, TABLE_HEIGHT
    from primitives.shared.ik import dh_params, forward_kinematics
    q = np.asarray(q, float)
    jp = compute_all_joint_positions(q)
    T = forward_kinematics(dh_params, q)
    ee = T[:3, 3]
    tip = ee + T[:3, :3] @ np.asarray(GRIPPER_CENTER_TOOL_OFFSET, float)
    min_safe = 0.04 + 0.02
    if table_check and tip[2] < TABLE_HEIGHT:
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
