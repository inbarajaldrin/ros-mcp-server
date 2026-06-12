#!/usr/bin/env python3
"""Unified boundary-predicate registry — the one named surface for "is the robot
in state X?" checks, sim+real, consumed in-process by the MCP server / triggers
(e.g. the Step-3 commit_object) and by primitives that already hold state.

Each predicate ties to a single SOURCE OF TRUTH and returns (ok: bool, reason: str).
The reason is a terse DIAGNOSTIC (measured value vs threshold) for server logs; the
agent-facing remedy ("Move to safe height", "Move home") is composed by the caller,
which alone knows phase / object / first-vs-final context.

Registry
--------
  is_at_safe_height(state)        SELF-CONTAINED — get_robot_state() + the
                                  move_to_safe_height gripper-center invariant.
  is_home(state)                  SELF-CONTAINED — get_robot_state() + HOME_POSE (EE-pose
                                  match to what move_home drives to, not HOME_JOINTS).

  is_object_grasped(...)          SOURCE: queries/verify_grasp.py        ✓ wired
  is_assembled(...)               SOURCE: queries/verify_assembly.py     ✓ wired
  is_disassembled(...)            SOURCE: queries/verify_disassembly.py  ✓ wired
  is_order_correct(...)           order PRE-GATE candidate (NOT wired yet). Replaces the
                                  scattered, pre-action-only _assert_assembly_order; works
                                  pre-action AND at commit. Wire into the pre-gate registry
                                  later (overall pre-gates + per-primitive pre-gates).
  (more)                          for motion primitives that lack a clean boundary,
                                  added after surveying the primitive surface.

The two SELF-CONTAINED predicates live here in full because no prior primitive owns
them. The SOURCE-backed predicates keep their verdict logic AT the source primitive
and are exposed here as pure normalizers over that source's result dict — the same
{"result": "success"|"failure", ...} dict signal_phase_complete already consumes
(signal_phase_complete.py:185). The caller (e.g. commit_object) runs the
verify_* call via its injected callback and passes the dict in; this module stays
ROS-free and unit-testable.

Provenance: gripper-center-z + the safe-height invariant mirror
primitives/move_to_safe_height.py:254-259; constants from primitives/shared/config.py.
"""
import math

from scipy.spatial.transform import Rotation as R

from primitives.shared.config import (
    SAFE_HEIGHT,
    GRIPPER_CENTER_TOOL_OFFSET,
    HOME_POSE,
)

# move_to_safe_height parks the GRIPPER CENTER at this z. Calibrated convention:
# SAFE_HEIGHT is the face-down FLANGE target, and the gripper center hangs
# GRIPPER_CENTER_TOOL_OFFSET[2] below it (see move_to_safe_height.py:257-259,
# facedown_offset_z = -GRIPPER_CENTER_TOOL_OFFSET[2]). The pipeline is calibrated
# around this; the predicate matches it rather than relitigating it.
SAFE_GRIPPER_CENTER_Z = float(SAFE_HEIGHT - GRIPPER_CENTER_TOOL_OFFSET[2])

SAFE_HEIGHT_TOL_M = 0.02       # gripper center may sit this far below target and still pass
HOME_POS_TOL_M = 0.02          # EE position may sit this far from HOME_POSE and still pass
HOME_ORI_TOL_DEG = 3.0         # EE orientation may differ this much from HOME_POSE and still pass


def _finite_seq(xs, n: int) -> bool:
    """True iff xs is a length-n sequence of finite numbers (fail-closed guard)."""
    try:
        if xs is None or len(xs) != n:
            return False
        return all(math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


def gripper_center_z(ee_pos, ee_quat) -> float:
    """Base-frame Z (m) of the gripper center given the flange pose.

    Rigid body: gripper_center = flange_pos + R_flange @ offset_tool, so the Z is
    flange_z + (R @ offset)[2]. Orientation-aware on purpose (a tilted EE has a
    different flange->center dz than face-down). Raises ValueError on a degenerate
    (zero-norm / NaN) quaternion.
    """
    R_fk = R.from_quat(ee_quat).as_matrix()
    offset_world = R_fk @ GRIPPER_CENTER_TOOL_OFFSET
    return float(ee_pos[2] + offset_world[2])


def is_at_safe_height(state: dict, tol_m: float = SAFE_HEIGHT_TOL_M):
    """True if the gripper center is at/above the safe-height target (within tol below).

    Matches the invariant move_to_safe_height enforces across all orientations:
    gripper_center_z == SAFE_GRIPPER_CENTER_Z. `state` is get_robot_state() output
    (uses ee_pos, ee_quat). Returns (ok, reason); fails closed on missing/malformed pose.
    """
    if not isinstance(state, dict):
        return False, "robot state unavailable (not a dict)"
    ee_pos = state.get("ee_pos")
    ee_quat = state.get("ee_quat")
    if not _finite_seq(ee_pos, 3) or not _finite_seq(ee_quat, 4):
        return False, "ee pose unavailable or malformed (ee_pos/ee_quat)"
    try:
        gc_z = gripper_center_z(ee_pos, ee_quat)
    except ValueError as e:
        return False, f"invalid ee_quat: {e}"
    threshold = SAFE_GRIPPER_CENTER_Z - tol_m
    ok = gc_z >= threshold
    return ok, (
        f"gripper center z={gc_z * 1000:.0f}mm "
        f"{'>=' if ok else '<'} safe {SAFE_GRIPPER_CENTER_Z * 1000:.0f}mm (tol {tol_m * 1000:.0f}mm)"
    )


def is_home(state: dict, pos_tol_m: float = HOME_POS_TOL_M, ori_tol_deg: float = HOME_ORI_TOL_DEG):
    """True if the END-EFFECTOR is at the home pose move_home drives to (HOME_POSE).

    SOURCE-OF-TRUTH ALIGNMENT: move_home targets the Cartesian HOME_POSE (EE tool frame)
    via IK, and that pose may resolve to any of several joint branches — so a joint-space
    match against a single HOME_JOINTS is WRONG: it rejects valid homes move_home actually
    produces (e.g. Cartesian move_home lands at HOME_POSE with joint[0]/[5] off HOME_JOINTS).
    This checks the EE pose itself (position + orientation), so is_home agrees with what
    move_home does. `state` is get_robot_state() output (uses ee_pos, ee_quat). Returns
    (ok, reason); fails closed on missing/malformed pose.
    """
    if not isinstance(state, dict):
        return False, "robot state unavailable (not a dict)"
    ee_pos = state.get("ee_pos")
    ee_quat = state.get("ee_quat")
    if not _finite_seq(ee_pos, 3) or not _finite_seq(ee_quat, 4):
        return False, "ee pose unavailable or malformed (ee_pos/ee_quat)"
    try:
        R_meas = R.from_quat(ee_quat)
    except ValueError as e:
        return False, f"invalid ee_quat: {e}"
    target_pos = [float(v) for v in HOME_POSE[0:3]]
    R_target = R.from_euler("xyz", HOME_POSE[3:6], degrees=True)
    pos_err = math.dist([float(v) for v in ee_pos], target_pos)
    ori_err_deg = math.degrees((R_target.inv() * R_meas).magnitude())
    ok = pos_err <= pos_tol_m and ori_err_deg <= ori_tol_deg
    return ok, (
        f"EE {'at home' if ok else 'not home'}: "
        f"pos off {pos_err * 1000:.0f}mm (tol {pos_tol_m * 1000:.0f}mm), "
        f"ori off {ori_err_deg:.1f}deg (tol {ori_tol_deg:.1f}deg)"
    )


# --- Source-backed predicates: pure normalizers over a verify_* result dict ---------
#
# Each takes the {"result": "success"|"failure", "object_name"?, "error"?} dict that
# the source's server callback returns (NOT the raw (bool, dict) tuple the verify_*.py
# methods return internally — that is normalized upstream). The caller orchestrates the
# verify_* subprocess/callback and passes the dict; this keeps the module ROS-free.

def _source_verdict(result, ok_reason: str, fail_reason: str):
    """Normalize a verify_* SOURCE result dict into (ok, reason).

    Keys on result["result"] == "success" — the exact contract consumed at
    signal_phase_complete.py:185. On success the reason is ok_reason; on failure it is
    the source's own "error" text if present, else fail_reason (per-object failure
    dicts carry metrics but often no top-level "error"). Fail-closed: any non-dict /
    missing "result" is treated as failure.
    """
    if not isinstance(result, dict):
        return False, "no verify result (fail-closed)"
    ok = result.get("result") == "success"
    obj = result.get("object_name")
    detail = ok_reason if ok else (result.get("error") or fail_reason)
    return ok, (f"[{obj}] {detail}" if obj else detail)


def is_object_grasped(verify_grasp_result: dict):
    """Holding / grasp verdict — SOURCE: queries/verify_grasp.py (sim+real).
    Pass the verify_grasp callback's result dict. Returns (ok, reason)."""
    return _source_verdict(verify_grasp_result, "grasp verified", "object not grasped")


def is_assembled(verify_assembly_result: dict):
    """Per-object assembled verdict — SOURCE: queries/verify_assembly.py.
    Pass the verify_assembly callback's result dict. Returns (ok, reason)."""
    return _source_verdict(
        verify_assembly_result, "in correct assembly pose", "not in correct assembly pose"
    )


def is_disassembled(verify_disassembly_result: dict):
    """Per-object disassembled verdict — SOURCE: queries/verify_disassembly.py.
    Pass the verify_disassembly callback's result dict. Returns (ok, reason)."""
    return _source_verdict(
        verify_disassembly_result, "removed from assembly pose", "still in assembly pose"
    )


def is_order_correct(object_name, ordered_objects, completed_objects, remedy_verb="complete"):
    """Order PRE-GATE — are all of object_name's PREDECESSORS already done?

    Works at BOTH application points: pre-action (X not yet done -> "predecessors done"
    == "can I start X?") and commit (X done -> still checks predecessors). That is why it
    can replace the scattered, pre-action-only _assert_assembly_order in the future
    pre-gate registry. NOT wired at the commit boundary: the order pre-gate blocks
    out-of-order picks before they execute, so a commit can never be reached out of order.
    Inputs:
      ordered_objects   the phase sequence (assembly_order, or its reverse for disassembly)
      completed_objects the done set (seated / removed), or None if the order oracle was
                        unavailable -> fail-closed (except first-in-sequence, no predecessors).
      remedy_verb       "Seat" / "Remove" for the message.
    Returns (ok, reason). Unknown object (not in the map) -> ungated (liveness).
    """
    if not ordered_objects or object_name not in ordered_objects:
        return True, "object not in order map (ungated)"
    k = ordered_objects.index(object_name)
    if k == 0:
        return True, "order ok (first in sequence)"  # no predecessors -> oracle irrelevant
    if completed_objects is None:
        return False, "order check unavailable. Retry."
    done = set(completed_objects)
    missing = [o for o in ordered_objects[:k] if o not in done]
    if missing:
        return False, f"'{object_name}' out of order. {remedy_verb} '{missing[0]}' first."
    return True, "order ok"


def is_checkpoint_saved(object_name, ordered_objects, saved_checkpoints):
    """Checkpoint PRE-GATE — has object_name's scene-state checkpoint been saved?

    Folds in the phase-baseline ("init") check: it is the same test (a named
    scene-state file present), only for the first object. Mirrors is_order_correct
    — the CALLER passes the phase-appropriate ordered_objects (assembly_order, or
    its reverse for disassembly, exactly like _assert_assembly_order's
    `removal = list(reversed(order))` at server_core.py) — so this predicate
    carries NO phase logic.

    Checkpoint naming: the FIRST object in the sequence shares the baseline name
    "init" (k==0); every later object's checkpoint is named after the object.
    An empty/None object_name asks only "is 'init' saved?" — the motion pre-gate's
    question, which has no specific object.

    Inputs:
      ordered_objects     the phase sequence (assembly_order, or its reverse for
                          disassembly). Ignored when object_name is empty.
      saved_checkpoints   any iterable of checkpoint basenames present (sans .json).
    Returns (ok, reason). Fail-closed (missing checkpoint -> False). Unknown object
    (not in the map) -> ungated (liveness), matching is_order_correct.
    """
    saved = set(saved_checkpoints or ())
    # Motion pre-gate path: no object -> only the baseline "init" is required.
    if not object_name or not str(object_name).strip():
        return ("init" in saved), (
            "init checkpoint saved" if "init" in saved
            else "save scene state as 'init' before any motion this phase."
        )
    if not ordered_objects or object_name not in ordered_objects:
        return True, "object not in order map (ungated)"
    required = "init" if ordered_objects.index(object_name) == 0 else object_name
    if required in saved:
        return True, f"checkpoint '{required}' saved"
    return False, f"save scene state as '{required}' before committing '{object_name}'."
