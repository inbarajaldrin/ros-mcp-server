"""The face-down --to-table floor (flange z), pure: no ROS import (review w8-floor-notilt r1 P3).  move_down uses it
(and re-exports it); the replay's own floor mirror calls it, so the two cannot drift.

floor_width_mm: the width a descent's floor is judged at.  to_table_floor_z: the floor at that width, or with none.
"""
import numpy as np

PLACE_TIP_CLEARANCE_M = 0.001
RAW_MINUS_GAP_MM = 9.2          # /gripper_width (raw) - /gripper_width_offset (the rubber gap): SETUP.md section 6
# The no-width sweep: openings 0-110 mm raw every 10 mm; rg2_geometry's boxes at w cover [w - 1, w + 10.2] mm, so the
# grid tiles the whole stroke with no gap.
SWEEP_OPENINGS_M = tuple(round(0.010 * k, 3) for k in range(12))


def floor_width_mm(opening_mm=None, close_width_mm=None, measured_mm=None):
    """The width the --to-table floor is judged at: the NARROWEST raw width the jaws reach at this stance (review r3
    P1-1: closing drives the RG2's fingertips down, so a pick is judged where it closes, as move_to_grasp.py:462
    judges it at the part's grip width).  None when no width is given (dev's closed-tip floor, every existing caller);
    otherwise the smallest of the descent's opening, the measured /gripper_width (review r3 P2: a commanded open that
    was not reached) and the close width the stance ends in (the part's grasp width; fully closed, 0, when not
    given)."""
    if opening_mm is None and close_width_mm is None:
        return None
    given = [0.0 if close_width_mm is None else float(close_width_mm)]
    given += [float(v) for v in (opening_mm, measured_mm) if v is not None]
    return max(0.0, min(given))



def to_table_floor_z(R_flange, opening_mm=None):
    """The face-down --to-table floor (flange z).
    Without a width: the LOWEST RG2 point over the whole stroke - each opening's finger boxes (SWEEP_OPENINGS_M, the
    boxes of an opening w covering [w - 1, w + 10.2] mm, so the grid tiles 0-110 mm) plus the housing / quick changer -
    GRASP_TIP_CLEARANCE_M (2 mm) above the table in the ACTUAL orientation, never below the closed fingertips (config's
    RG2_TIP_CLOSED_M) 1 mm above it - tilt-aware (lead 2026-11-20; vertical: 4.3 mm above dev's old floor).
    With a raw width (floor_width_mm's: the NARROWEST the jaws reach - the close width, never just the approach
    opening; fully closed when the measured width is missing): the flange z that puts the LOWEST point of the RG2 -
    rg2_geometry's finger boxes at that width plus the housing and quick-changer corners (gripper_rules'
    gripper_points_tool, the geometry every other rule uses) - GRASP_TIP_CLEARANCE_M (2 mm) above the table in the
    ACTUAL orientation, so it is exact at any face-down tilt (review w8-floor-width r1 P1); and never below config's
    width-aware figure along the tool axis (rg2_tip_length_m at the rubber gap raw - 9.2 mm, move_to_grasp's
    pad-centre floor) - the deeper fingertip holds until the caliper measurement (lead rulings 2026-11-20).
    The model unavailable: dev's closed-tip floor (fail safe)."""
    from primitives.shared.config import (GRASP_TIP_CLEARANCE_M, RG2_TIP_CLOSED_M, TABLE_HEIGHT,
                                          rg2_tip_length_m)
    R = np.asarray(R_flange, float)
    closed = TABLE_HEIGHT + PLACE_TIP_CLEARANCE_M - float((R @ np.array([0.0, 0.0, RG2_TIP_CLOSED_M]))[2])
    from primitives.shared.gripper_rules import gripper_points_tool
    if opening_mm is None:
        # no width (lead 2026-11-20, the tilted-descent hazard): the LOWEST RG2 point over the WHOLE stroke in the
        # actual orientation, 2 mm clear; never below dev's closed-tip figure.  Tilt-aware: the closed-tip figure alone
        # put the RG2 11-50 mm into the table at 20-60 deg.  The model unavailable: the closed-tip floor.
        # PER-OPENING finger boxes swept across the stroke (review w8-floor-notilt r1 P2): the deepest of each
        # opening's lowest point.  One box over the whole stroke combined the closed fingertip's depth with the open
        # finger's reach - a corner no real jaw position has, 10.9 / 17.3 / 29.9 mm too high at 5 / 10 / 20 deg.
        lows = []
        for w in SWEEP_OPENINGS_M:
            P = gripper_points_tool(w)
            if P is None:
                return closed
            lows.append(float((np.asarray(P, float) @ R[2, :]).min()))
        return max(closed, TABLE_HEIGHT + GRASP_TIP_CLEARANCE_M - min(lows))
    raw = max(0.0, float(opening_mm))
    P = gripper_points_tool(raw / 1000.0)
    if P is None:
        return closed
    lowest = float((np.asarray(P, float) @ R[2, :]).min())          # the lowest RG2 point below the flange, in base z
    exact = TABLE_HEIGHT + GRASP_TIP_CLEARANCE_M - lowest
    along_axis = TABLE_HEIGHT + GRASP_TIP_CLEARANCE_M - float(
        (R @ np.array([0.0, 0.0, rg2_tip_length_m(max(0.0, raw - RAW_MINUS_GAP_MM))]))[2])
    return max(exact, along_axis)
