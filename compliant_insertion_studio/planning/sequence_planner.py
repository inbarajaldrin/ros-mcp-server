#!/usr/bin/env python3
"""Derive an assembly step sequence from the OBSERVED start orientation.

Replaces replaying a fixed recorded tool_sequence. The recorded lists bake in a
pass count that was correct only for the orientation the part happened to be in
when the gold trajectory was captured; a part resting on a different face needs
a different number of passes, and a part displaced mid-run has no way to
re-enter the fixed list.

THE RULE
--------
The gripper takes a resting part from above, so it occupies whichever object
face points up at rest. To insert, the gripper must again be above the part,
which means the face it holds has to point up in the target orientation too.

    single pass   pick -> rotate -> insert
                  when the object axis pointing up at rest also points up in
                  some fold-symmetry-equivalent target

    two passes    pick -> rotate -> place -> pick -> rotate -> insert
                  otherwise: the first pass sets the part down on the face that
                  exposes the correct face upward, the second takes it there

Fold symmetry matters because a part may be visually identical under several
orientations; only one of them needs to satisfy the up-face condition.

Worked example (FMB1): u_brown, u_orange and inverted_u_yellow all target
RPY (-90, 0, 0) -- on edge -- so a flat-resting part changes faces and needs two
passes. line_green targets (0, 0, 0) -- flat -- so the face never changes and one
pass suffices. That reproduces the recorded 15/15/15/8-step sequences.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.spatial.transform import Rotation as R

from primitives.shared.fold_symmetry import equivalent_orientations

WORLD_UP = np.array([0.0, 0.0, 1.0])

# A face counts as "up" within this cone. Generous because the camera's
# orientation estimate is noisy and a part resting on a face is physically
# snapped to it anyway.
UP_TOLERANCE_DEG = 30.0

SINGLE_PASS = ["control_gripper", "move_to_grasp", "control_gripper",
               "move_to_safe_height", "rotate_object", "translate_object",
               "control_gripper", "move_to_safe_height"]


def up_axis_in_object_frame(R_object_world: np.ndarray) -> np.ndarray:
    """Object-frame axis that points along world +Z for a part in this pose.

    That is the face the gripper takes when it descends from above.
    """
    return R_object_world.T @ WORLD_UP


def single_pass_feasible(R_start_world: np.ndarray,
                         R_target_world: np.ndarray,
                         fold_data,
                         tolerance_deg: float = UP_TOLERANCE_DEG):
    """True when the grasped face still points up in some equivalent target.

    Returns (feasible, best_angle_deg, R_target_used).
    """
    u = up_axis_in_object_frame(R_start_world)
    cos_tol = np.cos(np.radians(tolerance_deg))

    best_angle = 180.0
    best_target = None
    feasible = False

    for R_equiv in equivalent_orientations(R_target_world, fold_data):
        alignment = float(np.dot(R_equiv @ u, WORLD_UP))
        angle = np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))
        if angle < best_angle:
            best_angle, best_target = angle, R_equiv
        if alignment > cos_tol:
            feasible = True

    return feasible, best_angle, best_target


def plan_passes(R_start_world: np.ndarray,
                R_target_world: np.ndarray,
                fold_data,
                tolerance_deg: float = UP_TOLERANCE_DEG) -> int:
    """Number of pick->rotate->place passes needed. 1 or 2."""
    feasible, _, _ = single_pass_feasible(
        R_start_world, R_target_world, fold_data, tolerance_deg)
    return 1 if feasible else 2


def plan_sequence(R_start_world: np.ndarray,
                  R_target_world: np.ndarray,
                  fold_data,
                  tolerance_deg: float = UP_TOLERANCE_DEG) -> list[str]:
    """Tool sequence for a part observed at R_start_world.

    The two-pass form is the single-pass form run twice: the first pass ends by
    placing the part down at the clear area, which is simply another start state
    the second pass consumes.
    """
    passes = plan_passes(R_start_world, R_target_world, fold_data, tolerance_deg)
    if passes == 1:
        return list(SINGLE_PASS)
    # Second pass re-enters at move_to_grasp: the gripper is already open from
    # the release that ended pass 1.
    return list(SINGLE_PASS) + list(SINGLE_PASS[1:])


def quat_to_matrix(q) -> np.ndarray:
    """Accept (x, y, z, w) as a sequence or an object with .x/.y/.z/.w."""
    if hasattr(q, "x"):
        q = [q.x, q.y, q.z, q.w]
    return R.from_quat(np.asarray(q, dtype=float)).as_matrix()


# ---------------------------------------------------------------------------
# Emitting a tool_sequence the replay executor can consume
# ---------------------------------------------------------------------------

def _set_translate_action(call: str, action: str) -> str:
    return re.sub(r"action='[a-z_]+'", f"action='{action}'", call)


def split_recorded_passes(recorded: list[str]) -> tuple[list[str], list[str]]:
    """Split a recorded tool_sequence into its first pass and any second pass.

    A pass ends with the translate_object that either places the part down or
    inserts it, followed by the gripper release and the retract.
    """
    ends = [i for i, c in enumerate(recorded) if "translate_object" in c]
    if not ends:
        raise ValueError("recorded sequence contains no translate_object step")
    first_end = ends[0] + 3          # translate, control_gripper, move_to_safe_height
    return recorded[:first_end], recorded[first_end:]


def build_tool_sequence(recorded: list[str], passes: int) -> list[str]:
    """Re-shape a recorded sequence to the derived number of passes.

    Argument values (gripper widths, grasp ids) are inherited from the recording;
    only the STRUCTURE is derived. Returns the recording untouched when it already
    has the right shape, so this is a no-op for parts resting as they were taught.
    """
    first, second = split_recorded_passes(recorded)

    if passes == 1:
        # The single pass inserts directly instead of placing the part down.
        return [_set_translate_action(c, "insert") if "translate_object" in c else c
                for c in first]

    # Two passes: first places the part down, second re-enters at move_to_grasp
    # (the gripper is already open from the release that ended pass 1).
    pass1 = [_set_translate_action(c, "place_down") if "translate_object" in c else c
             for c in first]
    if second:
        return pass1 + second
    pass2 = [_set_translate_action(c, "insert") if "translate_object" in c else c
             for c in first[1:]]
    return pass1 + pass2
