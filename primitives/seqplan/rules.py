"""The one checker (R29): every rule, one geometry, for every route a planner may fly.

route_refusal(waypoints, leg=..., ...) is the only place a planner applies the rules.  It calls the cell's rule
modules unchanged, at every sample of every segment, the route's LAST stance included:

- per waypoint: every joint inside its hardware limit (JOINT_LIMITS_RAD), |wrist_3| <= WRIST3_LIMIT_RAD, and wrist_3
  inside the camera cable's CABLE_WINDOW_DEG (absolute joint value, never wrapped);
- per segment, the travel caps: no joint more than one turn (held_part_collision.joint_travel_refusal), an arm joint
  (pan, shoulder, elbow) at most pi in one segment (TRAVEL_CAP_ARM_RAD); on an "insert" leg, translate_object's own
  insert-hover caps measured from the ROUTE's start, as it measures them (review r2 P1-A): joints 0-4 at most pi,
  wrist_3 at most 3.2 rad (TRAVEL_CAP_WRIST3_RAD);
- along each segment, sampled at most SAMPLE_STEP_RAD apart on any joint (both ends included):
  * collision.check_trajectory_collision: the arm against the table and itself (DH capsules, the two-tier physical
    arm + RG2 body, the camera mount vs the arm);
  * gripper_rules.gripper_table_refusal: the RG2's finger boxes (rg2_geometry, the one finger model) and its housing
    and quick-changer corners 2 mm above the table, at the leg's opening (unknown: the boxes enclose the whole
    stroke); the nominal gripper centre is not a material point (gripper_rules);
  * a route that ends in a Grip (ends_in_grip, required): its LAST stance also at the close width (close_width_m;
    None = fully closed)
    (gripper_rules.grasp_stance_refusal, review r3 P1-1: the fingertips go down as the jaws close);
  * gripper_rules.extended_gripper_refusal and orientation_refusal: rotate_object's extended-gripper checks (base
    column, upper arm; its "gripper centre below the table" clause is left to gripper_table_refusal, which checks
    the fingers themselves) and its EE-facing-robot / camera-toward--X rules;
  * camera_up.camera_route_refusal: the camera side near the table, the mount/D415 clearance, the mount vs the arm;
  * held_part_collision.held_part_route_refusal: the held part against the arm and the table;
- per stance, by leg (`leg` is required, no default: a caller states what it flies):
  * "transit" (a joint route between hovers) and "insert" (the seat approach and descent): collision.
    check_ee_below_base and check_compact_configuration at EVERY sample, the last included (rotate_object checks
    every route-segment end, translate_object every insert waypoint);
  * "approach" (a pick, place-down or lift leg near the table): not those two, but only on a STRAIGHT DENSE segment
    near the table (review r2 P2-B, r3): at most APPROACH_DENSE_STEP_RAD on any joint, the tool's orientation kept
    (<= 1 deg), the gripper centre moving vertically or along the tool's own axis (<= 2 deg), and within
    APPROACH_NEAR_TABLE_M of the table.  That is move_down's and move_to_safe_height's motion (vertical, orientation
    kept); a carry labelled "approach" is held to both rules as a transit is.
- starts_at_contact (a lift from a stance move_down reached, its force stop the authority there; lead ruling
  2026-11-20): while the gripper is under its 2 mm floor at the route's start, the floor is replaced by: the segment
  is straight (straight_segment: vertical or along the tool axis, orientation kept; review r3 P3-3), its lowest point
  rises monotonically, and it clears the floor within LIFT_FROM_CONTACT_M of gripper-centre travel.
Fingertip figures (review r3): these rules use the RG2 model's fingertip (234.2 mm closed); config's measured
RG2_TIP_CLOSED_M (230.86 mm) and rg2_tip_length_m stay in dev's move_down default floor and move_to_grasp.  If the
mesh is right, dev's closed-tip --to-table floor puts the tips 2.3 mm into the table (force stop only); the caliper
measurement settles which figure is right.
The table threshold of each leg mirrors the primitive that flies it (z_threshold): move_to_clear_area and the
place-down descent TABLE_HEIGHT - TABLE_COLLISION_MARGIN_*, every other primitive TABLE_HEIGHT.
The camera's table floor is the real 30 mm (camera_up.CAMERA_CLEARANCE_M).
A descent toward a part or the table also needs a contact stop: move_plan refuses to fly one (it has none); fly it
with move_down.
"""
from __future__ import annotations

import math

import numpy as np

from primitives.shared.camera_up import camera_route_refusal
from primitives.shared.collision import (check_compact_configuration, check_ee_below_base,
                                         check_trajectory_collision)
from primitives.shared.config import TABLE_HEIGHT
from primitives.shared.gripper_rules import extended_gripper_refusal, gripper_table_refusal, orientation_refusal
from primitives.shared.held_part_collision import held_part_route_refusal, joint_travel_refusal
from primitives.shared.ik import WRIST3_LIMIT_RAD, dh_params, forward_kinematics

SAMPLE_STEP_RAD = 0.02                   # review r1 P2-4: sampled per distance, not per segment
# The camera's table floor is the real 30 mm everywhere (operator ruling R33, 2026-11-19).
REAL_CAMERA_FLOOR_M = 0.030             # camera_up.CAMERA_CLEARANCE_M: stances under it are flagged in the plan
CABLE_WINDOW_DEG = (-340.0, 320.0)       # wrist_3 absolute joint value (lead 2026-11-19, after the strap)
# The UR5e's joint limits (rad): +-2pi, the elbow +-pi (the arm's hardware and the sim's arm binding agree).
JOINT_LIMITS_RAD = (2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi)
TRAVEL_CAP_ARM_RAD = np.pi               # translate_object.py's insert-hover caps (review r1 P1-4)
TRAVEL_CAP_WRIST3_RAD = 3.2
LEGS = ("transit", "insert", "approach")
APPROACH_DENSE_STEP_RAD = 0.05
APPROACH_STRAIGHT_DEG = 2.0
LIFT_FROM_CONTACT_M = 0.010
APPROACH_NEAR_TABLE_M = 0.250   # a pre-place hover: 0.15 m above a horizontal stand-up, gripper centre ~0.22 m up


def waypoint_refusal(w):
    """The per-waypoint hard limits (joint limits, |wrist_3|, the cable window), or None."""
    w = np.asarray(w, float)
    if w.shape != (6,) or not np.all(np.isfinite(w)):
        return f"WAYPOINT_INVALID: {w.tolist()}"
    over = np.abs(w) > np.asarray(JOINT_LIMITS_RAD) + 1e-9
    if over.any():
        j = int(np.argmax(over))
        return f"JOINT_LIMIT: joint {j} at {np.degrees(w[j]):+.1f} deg outside +-{np.degrees(JOINT_LIMITS_RAD[j]):.0f}"
    if abs(float(w[5])) > WRIST3_LIMIT_RAD:
        return f"WRIST3_LIMIT: wrist_3 {float(w[5]):+.2f} rad outside +-{WRIST3_LIMIT_RAD} rad"
    w3 = float(np.degrees(w[5]))
    if not CABLE_WINDOW_DEG[0] <= w3 <= CABLE_WINDOW_DEG[1]:
        return f"CABLE_WINDOW: wrist_3 {w3:+.0f} deg outside [{CABLE_WINDOW_DEG[0]:+.0f}, {CABLE_WINDOW_DEG[1]:+.0f}]"
    return None


def travel_cap_refusal(a, b):
    """An arm joint (pan, shoulder, elbow) at most pi in one segment (every leg)."""
    d = np.abs(np.asarray(b, float) - np.asarray(a, float))
    for j in range(3):
        if d[j] > TRAVEL_CAP_ARM_RAD + 1e-9:
            return f"TRAVEL_CAP: joint {j} moves {np.degrees(d[j]):.0f} deg in one segment (cap 180)"
    return None


def insert_cap_refusal(start, w):
    """translate_object's insert-hover caps from the route's start (translate_object.py: max(delta[:5]) > pi, wrist_3
    > 3.2 rad, measured from the current stance to each waypoint)."""
    d = np.abs(np.asarray(w, float) - np.asarray(start, float))
    j = int(np.argmax(d[:5]))
    if d[j] > TRAVEL_CAP_ARM_RAD + 1e-9:
        return f"INSERT_TRAVEL_CAP: joint {j} {np.degrees(d[j]):.0f} deg from the insert's start (cap 180)"
    if d[5] > TRAVEL_CAP_WRIST3_RAD + 1e-9:
        return f"INSERT_TRAVEL_CAP: wrist_3 {d[5]:.2f} rad from the insert's start (cap {TRAVEL_CAP_WRIST3_RAD})"
    return None


def straight_segment(a, b):
    """The segment keeps the tool's orientation and moves the gripper centre vertically or along the tool axis."""
    from primitives.shared.config import GRIPPER_CENTER_TOOL_OFFSET
    Ta, Tb = forward_kinematics(dh_params, np.asarray(a, float)), forward_kinematics(dh_params, np.asarray(b, float))
    Rd = Ta[:3, :3].T @ Tb[:3, :3]
    if math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rd) - 1) / 2)))) > 1.0:
        return False
    off = np.asarray(GRIPPER_CENTER_TOOL_OFFSET, float)
    d = (Tb[:3, :3] @ off + Tb[:3, 3]) - (Ta[:3, :3] @ off + Ta[:3, 3])
    n = float(np.linalg.norm(d))
    if n < 1e-4:
        return True
    cos_tol = math.cos(math.radians(APPROACH_STRAIGHT_DEG))
    return abs(d[2]) / n >= cos_tol or abs(float(d @ Ta[:3, 2])) / n >= cos_tol


def stance_refusal(q, leg, opening_m=None, dense_segment=False, skip_gripper_table=False):
    """The rules every stance keeps (gripper vs table, extended gripper, orientation), plus EE-below-base and
    compact for transit and insert legs, and for approach legs away from the table or off a dense segment."""
    from primitives.shared.config import GRIPPER_CENTER_TOOL_OFFSET
    T = forward_kinematics(dh_params, np.asarray(q, float))
    why = ((None if skip_gripper_table else gripper_table_refusal(T, opening_m))
           or extended_gripper_refusal(q, table_check=False) or orientation_refusal(T))
    if why:
        return why
    near = float((T[:3, :3] @ np.asarray(GRIPPER_CENTER_TOOL_OFFSET, float) + T[:3, 3])[2]) - TABLE_HEIGHT \
        < APPROACH_NEAR_TABLE_M
    if leg in ("transit", "insert") or not (dense_segment and near):
        if check_ee_below_base(q):
            return "EE_BELOW_BASE: the flange is below the base's height on a transit / insert leg"
        if check_compact_configuration(q):
            return "COMPACT_CONFIGURATION: the wrist is within 0.20 m of the shoulder"
    return None


def _samples(a, b):
    return max(1, int(math.ceil(float(np.max(np.abs(np.asarray(b, float) - np.asarray(a, float)))) / SAMPLE_STEP_RAD - 1e-9)))


def route_refusal(waypoints, *, leg, held=None, near_table_end=False, z_threshold=TABLE_HEIGHT, opening_m=None,
                  starts_at_contact=False, close_width_m=None, ends_in_grip):
    """None when every segment of the joint route passes every rule, else the first typed refusal (with the segment).
    `leg`: "transit" | "insert" | "approach" (see the module).  `opening_m`: the gripper's raw opening along the
    route, None when unknown (the finger boxes then enclose the whole stroke).  ends_in_grip / close_width_m: the
    route ends where a Grip closes (a pick), to close_width_m raw (None: fully closed); its last stance is judged at
    that width too (grasp_stance_refusal).  ends_in_grip is REQUIRED, like leg (lead 2026-11-20, r4 P2-1): a caller
    states whether its route ends in a Grip."""
    if leg not in LEGS:
        raise ValueError(f"route_refusal: leg must be one of {LEGS}, not {leg!r}")
    if not isinstance(ends_in_grip, bool):
        raise ValueError(f"route_refusal: ends_in_grip must be true or false, not {ends_in_grip!r}")
    W = [np.asarray(w, float) for w in waypoints]
    if not W:
        return "ROUTE_EMPTY"
    for w in W:
        why = waypoint_refusal(w)
        if why:
            return why
    if len(W) == 1:
        W = [W[0], W[0]]
    from primitives.shared.config import GRASP_TIP_CLEARANCE_M, GRIPPER_CENTER_TOOL_OFFSET
    from primitives.shared.gripper_rules import gripper_low_z
    off = np.asarray(GRIPPER_CENTER_TOOL_OFFSET, float)
    floor = TABLE_HEIGHT + GRASP_TIP_CLEARANCE_M
    rising = bool(starts_at_contact)
    T0 = forward_kinematics(dh_params, W[0])
    gc0, prev_low = T0[:3, :3] @ off + T0[:3, 3], -np.inf
    for i, (a, b) in enumerate(zip(W[:-1], W[1:])):
        last = i == len(W) - 2
        n = _samples(a, b)
        why = travel_cap_refusal(a, b) or joint_travel_refusal(a, b, prefix="ROUTE" if held is None else "HELD_ROUTE")
        if not why and leg == "insert":
            why = insert_cap_refusal(W[0], b)
        dense = (float(np.max(np.abs(b - a))) <= APPROACH_DENSE_STEP_RAD + 1e-12) and straight_segment(a, b)
        if not why and check_trajectory_collision(a, b, z_threshold=z_threshold, num_samples=n):
            why = "ARM_TABLE_OR_SELF_COLLISION"
        if not why:
            for k in range(n + 1):
                q = a + (k / n) * (b - a)
                skip = False
                if rising:
                    Tq = forward_kinematics(dh_params, q)
                    low = gripper_low_z(Tq, opening_m)
                    if low is not None and low < floor:
                        rise = float((Tq[:3, :3] @ off + Tq[:3, 3])[2] - gc0[2])
                        if not straight_segment(a, b):
                            why = "LIFT_FROM_CONTACT: not a straight vertical / tool-axis segment under the floor"
                        elif low < prev_low - 1e-6:
                            why = f"LIFT_FROM_CONTACT: the gripper descends ({1000 * (low - prev_low):+.2f} mm) under its floor"
                        elif rise > LIFT_FROM_CONTACT_M:
                            why = (f"LIFT_FROM_CONTACT: still under the 2 mm floor after {1000 * rise:.1f} mm of rise "
                                   f"(allowed {1000 * LIFT_FROM_CONTACT_M:.0f})")
                        if why:
                            why = f"{why} (alpha={k / n:.3f})"
                            break
                        prev_low, skip = low, True
                    else:
                        rising = False
                why = stance_refusal(q, leg, opening_m, dense, skip_gripper_table=skip)
                if why:
                    why = f"{why} (alpha={k / n:.3f})"
                    break
        if not why:
            why = (camera_route_refusal(a, b, near_table_end=near_table_end and last, num_samples=n)
                   or held_part_route_refusal(a, b, held, num_samples=n))
        if not why and last and ends_in_grip:
            from primitives.shared.gripper_rules import grasp_stance_refusal
            why = grasp_stance_refusal(forward_kinematics(dh_params, b), close_width_m)
        if why:
            return f"{why} (segment {i + 1}/{len(W) - 1})"
    return None


def camera_floor_flags(waypoints):
    """[(waypoint index, camera clearance m)] for the waypoints whose mount/D415 comes closer to the table than the
    real path's floor (REAL_CAMERA_FLOOR_M)."""
    from primitives.shared.camera_up import mount_lowest_above_table
    out = []
    for i, w in enumerate(waypoints):
        T = forward_kinematics(dh_params, np.asarray(w, float))
        c = mount_lowest_above_table(T[:3, :3], T[:3, 3], TABLE_HEIGHT, REAL_CAMERA_FLOOR_M)
        if c < REAL_CAMERA_FLOOR_M:
            out.append((i, c))
    return out
