"""The one checker (R29): every rule, one geometry, for every route the planner may fly.

route_refusal(waypoints, ...) is the only place the planner applies the rules.  It calls the cell's rule modules
unchanged:
- collision.check_trajectory_collision: the arm against the table and itself along each joint line (DH capsules,
  the physical arm + gripper, the camera mount vs the arm), 20 samples per segment, both ends included;
- rotate_object's stance rules on a reorient stance (EE below the base, compact, extended gripper is in the physical
  check) where the caller asks for them;
- camera_up.camera_route_refusal: the camera side near the table, the mount/D415 clearance, mount vs arm;
- held_part_collision.held_part_route_refusal: the held part against the arm and the table, and joint travel;
- every joint inside its hardware limit (JOINT_LIMITS_RAD) and |wrist_3| <= WRIST3_LIMIT_RAD at every waypoint (a joint line is monotonic per joint, so the ends bound it);
- the camera cable (lead 2026-11-19, the operator's real sweeps, lab-notes/2026-11-19-cable-scan): wrist_3 inside
  CABLE_WINDOW_DEG (absolute joint value, never wrapped) at every waypoint.  With the strap on the wrist housing the
  D415's USB 3 link held from -361 to +320 deg; UR's hardware limit is +-360.  dev's |wrist_3| <= WRIST3_LIMIT_RAD
  (4 rad, 229 deg) is tighter and still applies.  The per-part cumulative wrist_3 budget is the planner's (a path sum).
The table threshold of each leg mirrors the primitive that flies it on dev (z_threshold): move_to_clear_area and the
place-down descent TABLE_HEIGHT - TABLE_COLLISION_MARGIN_*, every other primitive TABLE_HEIGHT.
The camera's table floor is the real 30 mm (camera_up.CAMERA_CLEARANCE_M); camera_floor_flags() still lists any
planned stance under it (none should remain).
"""
from __future__ import annotations

import numpy as np

from primitives.shared.camera_up import camera_route_refusal
from primitives.shared.collision import (check_compact_configuration, check_ee_below_base,
                                         check_trajectory_collision)
from primitives.shared.config import TABLE_HEIGHT
from primitives.shared.held_part_collision import held_part_route_refusal
from primitives.shared.ik import WRIST3_LIMIT_RAD

TRAJECTORY_SAMPLES = 20
# The camera's table floor is the real 30 mm everywhere (operator ruling R33, 2026-11-19: place-downs release at
# move_down's floor, which keeps the camera well above it; the derived 12 mm sim floor is withdrawn).
REAL_CAMERA_FLOOR_M = 0.030             # camera_up.CAMERA_CLEARANCE_M: stances under it are flagged in the plan
CABLE_WINDOW_DEG = (-340.0, 320.0)       # wrist_3 absolute joint value (lead 2026-11-19, after the strap)


# The UR5e's joint limits (rad): +-2pi, the elbow +-pi (the arm's hardware and the sim's arm binding agree).  r49
# (MuJoCo, 2026-11-19): a planned face-right descent took wrist_1 from -358 to -393 deg and the controller refused it.
JOINT_LIMITS_RAD = (2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi)


def route_refusal(waypoints, held=None, near_table_end=False, stance_rules=False, samples=TRAJECTORY_SAMPLES,
                  z_threshold=TABLE_HEIGHT):
    """None when every segment of the joint route passes every rule, else the first typed refusal.  `samples`: per
    joint-linear segment (both ends included); a dense Cartesian leg (waypoints ~5 mm apart) uses 1, its
    waypoints being the samples."""
    W = [np.asarray(w, float) for w in waypoints]
    for w in W:
        over = np.abs(w) > np.asarray(JOINT_LIMITS_RAD) + 1e-9         # a joint line is monotonic: ends bound it
        if over.any():
            j = int(np.argmax(over))
            return f"JOINT_LIMIT: joint {j} at {np.degrees(w[j]):+.1f} deg outside +-{np.degrees(JOINT_LIMITS_RAD[j]):.0f}"
        if abs(float(w[5])) > WRIST3_LIMIT_RAD:
            return f"WRIST3_LIMIT: wrist_3 {float(w[5]):+.2f} rad outside +-{WRIST3_LIMIT_RAD} rad"
        w3 = float(np.degrees(w[5]))
        if not CABLE_WINDOW_DEG[0] <= w3 <= CABLE_WINDOW_DEG[1]:
            return f"CABLE_WINDOW: wrist_3 {w3:+.0f} deg outside [{CABLE_WINDOW_DEG[0]:+.0f}, {CABLE_WINDOW_DEG[1]:+.0f}] (provisional)"
    for i, (a, b) in enumerate(zip(W[:-1], W[1:])):
        last = i == len(W) - 2
        if check_trajectory_collision(a, b, z_threshold=z_threshold, num_samples=samples):
            return f"ARM_TABLE_OR_SELF_COLLISION (segment {i + 1}/{len(W) - 1})"
        if stance_rules and not last and (check_ee_below_base(b) or check_compact_configuration(b)):
            return f"ARM_STANCE_REFUSED: in-between stance {i + 1} is below the base or compact"
        why = (camera_route_refusal(a, b, near_table_end=near_table_end and last, num_samples=max(1, 2 * samples))
               or held_part_route_refusal(a, b, held, num_samples=max(1, 2 * samples)))
        if why:
            return f"{why} (segment {i + 1}/{len(W) - 1})"
    return None


def camera_floor_flags(waypoints):
    """[(waypoint index, camera clearance m)] for the waypoints whose mount/D415 comes closer to the table than the
    real path's floor (REAL_CAMERA_FLOOR_M): admissible in the sim plan, NOT on the real arm until delta_real is
    bounded."""
    from primitives.shared.camera_up import mount_lowest_above_table
    from primitives.shared.ik import dh_params, forward_kinematics
    out = []
    for i, w in enumerate(waypoints):
        T = forward_kinematics(dh_params, np.asarray(w, float))
        c = mount_lowest_above_table(T[:3, :3], T[:3, 3], TABLE_HEIGHT, REAL_CAMERA_FLOOR_M)   # exact under 30 mm
        if c < REAL_CAMERA_FLOOR_M:
            out.append((i, c))
    return out
