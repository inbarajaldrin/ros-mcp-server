"""
CAD-derived target pose lookup for Phase 5 v2 (refactored 2026-05-04).

Computes per-attempt prediction of where the TCP (wrist flange) should end up
at success, using the SAME flow as `primitives/translate_object.py`'s
real-mode insert (`translate_for_target_real`):

    R_target_abs = R_base_world @ R_target_in_base                  (CAD: assembly)
    grasp_offset_in_object = grasp_points[obj][grasp_id].position   (CAD: grasp)

    # Fold symmetry — pick the equivalent of R_target_abs that's CLOSEST to
    # the held object's current orientation; that's the rotation we use to
    # rotate grasp_offset into world.
    equivalents = fold_symmetry.equivalent_orientations(R_target_abs, fold_data)
    R_grasp_rotation = argmin_{R_eq in equivalents} (||R_eq @ grasp_offset
                                                       - R_held @ grasp_offset||
                                                      + tiebreak: angle_dist(R_eq, R_held))

    grasp_world_offset    = R_grasp_rotation @ grasp_offset_in_object
    target_gripper_center = base_world_xyz + R_base_world @ pos_in_base
                            + grasp_world_offset
    tool_offset_world     = R_EE_world @ GRIPPER_CENTER_TOOL_OFFSET   (face-down EE)
    target_flange         = target_gripper_center − tool_offset_world

This matches the SAME math the rest of the codebase uses — no more "+0.2286 in
object's local z" bug. The held_quat (current object orientation in world)
and R_EE_world (current EE orientation, FK-derived or fed from /tcp_pose)
are now required inputs because they determine T_grasp implicitly.

Why the v1 cad_lookup was wrong (ARCHIVED for posterity):
  v1 did `T_world_tcp = T_world_tip @ T_tip_flange` where T_tip_flange was a
  +0.2286 translation along z. T_world_tip's rotation came from the OBJECT's
  assembly orientation (e.g. -90° about X for u_orange), so the +0.2286 ended
  up applied along WORLD +Y instead of WORLD -Z (the actual gripper face-down
  tool axis). For u_orange the predicted TCP was reported 228 mm off in y/z;
  the predicate's tcp_z_reached_predicted check therefore never fired even on
  successful inserts. Diagnosed 2026-05-04 by comparing actual final TCP at
  the operator-confirmed successful run (043239) to v1's predicted_tcp_at_seat.

Inputs (required):
  base_name, object_name, grasp_id    — CAD asset identifiers
  base_world_xyz, base_world_quat_xyzw — base pose in world. Use config-fixed
                                          values from primitives.shared.config
                                          (operator decision: camera has noise,
                                          base is physically pinned).
  held_quat_xyzw                       — current OBJECT orientation in world
                                          (== --current-object-orientation arg).
                                          Used for fold-symmetry best-match.
  ee_orientation_xyzw                  — current EE (TCP / wrist flange)
                                          orientation in world. Used for the
                                          tool-Z direction when subtracting the
                                          flange offset. For Phase 5 (face-down
                                          gripper) typically ≈ R_y(180).

Outputs (a dict ready to drop into the meta JSON):
  predicted_tcp_at_seat:        {xyz_m, quat_xyzw}   — flange/TCP target
  predicted_grasp_tip_at_seat:  {xyz_m, quat_xyzw}   — gripper jaw center target
  T_world_object_seat:          {xyz_m, quat_xyzw}   — where object should end up
  fold_symmetry_used:           {applied, R_eq_quat_xyzw, pos_error_mm, angle_error_deg}
  grasp_point_in_object:        {xyz_m}              — for traceability
  cad_provenance:               {assembly_path, grasp_path, flange_offset_m}

Reference: https://github.com/<repo>/blob/main/primitives/translate_object.py
(`translate_for_target_real`, lines 734-895)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# Search paths — match what translate_object reads.
ASSEMBLY_INDEX_DIR = Path.home() / "Documents" / "aruco-grasp-annotator" / "data"
GRASP_POINTS_DIR   = Path.home() / "Documents" / "aruco-grasp-annotator" / "data" / "grasp_points"

# OnRobot RG2 mounted on UR5e — flange (EE) → gripper-center distance along
# the tool-frame +Z axis. Matches primitives.shared.config.GRIPPER_CENTER_TOOL_OFFSET.
GRIPPER_CENTER_TOOL_OFFSET = np.array([0.0, 0.0, 0.2286])
GRIPPER_CENTER_TOOL_OFFSET_M = float(GRIPPER_CENTER_TOOL_OFFSET[2])


# ---------------------------------------------------------------------------
# CAD index loaders — unchanged from v1
# ---------------------------------------------------------------------------

def _resolve_assembly_path(base_name: str) -> Path:
    if base_name.startswith("base") and base_name[4:].isdigit():
        candidate = ASSEMBLY_INDEX_DIR / f"fmb_assembly{base_name[4:]}.json"
        if candidate.exists():
            return candidate
    for p in sorted(ASSEMBLY_INDEX_DIR.glob("fmb_assembly*.json")):
        d = json.load(open(p))
        for c in d.get("components", []):
            if c.get("name") == base_name:
                return p
    raise FileNotFoundError(f"No assembly index found for base {base_name!r}")


def load_seat_pose_in_base(base_name: str, object_name: str) -> tuple[np.ndarray, np.ndarray]:
    path = _resolve_assembly_path(base_name)
    d = json.load(open(path))
    comp = next((c for c in d["components"] if c.get("name") == object_name), None)
    if comp is None:
        raise KeyError(f"{object_name!r} not in {path.name}; "
                       f"available: {[c.get('name') for c in d['components']]}")
    pos = comp["position"]
    quat = comp["rotation"]["quaternion"]
    return (
        np.array([pos["x"], pos["y"], pos["z"]], dtype=float),
        np.array([quat["x"], quat["y"], quat["z"], quat["w"]], dtype=float),
    )


def load_grasp_offset_in_object(object_name: str, grasp_id: int) -> np.ndarray:
    path = GRASP_POINTS_DIR / f"{object_name}_grasp_points.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}")
    d = json.load(open(path))
    grasp = next((g for g in d["grasp_points"] if g.get("id") == grasp_id), None)
    if grasp is None:
        raise KeyError(f"grasp_id={grasp_id} not in {path.name}; "
                       f"available: {[g.get('id') for g in d['grasp_points']]}")
    p = grasp["position"]
    return np.array([p["x"], p["y"], p["z"]], dtype=float)


# ---------------------------------------------------------------------------
# Fold-symmetry — lazy import (so cad_lookup import doesn't pull primitives/)
# ---------------------------------------------------------------------------

def _load_fold_data(object_name: str):
    """Returns fold_data dict or None if no symmetry file. Mirrors what
    `translate_object.translate_for_target_real` does."""
    try:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from primitives.shared.fold_symmetry import load_symmetry_data
        from utils.data_path_finder import get_symmetry_dir
        return load_symmetry_data(object_name, str(get_symmetry_dir()))
    except Exception:
        return None


def _equivalent_orientations(R_target_abs: np.ndarray, fold_data) -> list[np.ndarray]:
    if fold_data is None:
        return [R_target_abs]
    try:
        from primitives.shared.fold_symmetry import equivalent_orientations
        return list(equivalent_orientations(R_target_abs, fold_data))
    except Exception:
        return [R_target_abs]


def _select_best_fold_equivalent(
    R_target_abs: np.ndarray,
    R_held: np.ndarray,
    grasp_offset: np.ndarray,
    fold_data,
) -> tuple[np.ndarray, float, float]:
    """Same selection rule as translate_object.translate_for_target_real:
    pick the equivalent of R_target_abs that minimises ||R_eq @ grasp_offset −
    R_held @ grasp_offset|| (= position error of the gripper center if we use
    R_eq as the grasp rotation), with angular distance to R_held as tie-break.
    Returns (best_R_eq, pos_error_m, angle_error_rad).
    """
    equivalents = _equivalent_orientations(R_target_abs, fold_data)
    held_offset = R_held @ grasp_offset
    best_R_eq = R_target_abs
    best_pos_err = float("inf")
    best_ang_err = float("inf")
    for R_eq in equivalents:
        cand_offset = R_eq @ grasp_offset
        pos_err = float(np.linalg.norm(cand_offset - held_offset))
        ang_err = float((R.from_matrix(R_held).inv() *
                          R.from_matrix(R_eq)).magnitude())
        if pos_err < best_pos_err - 1e-9 or (
            abs(pos_err - best_pos_err) < 1e-9 and ang_err < best_ang_err
        ):
            best_pos_err = pos_err
            best_ang_err = ang_err
            best_R_eq = R_eq
    return best_R_eq, best_pos_err, best_ang_err


# ---------------------------------------------------------------------------
# Main entry point — refactored 2026-05-04
# ---------------------------------------------------------------------------

def predict_tcp_at_seat(
    *,
    base_name: str,
    object_name: str,
    grasp_id: int,
    base_world_xyz: np.ndarray | list,
    base_world_quat_xyzw: np.ndarray | list,
    held_quat_xyzw: np.ndarray | list,
    ee_orientation_xyzw: np.ndarray | list,
    flange_offset_m: float = GRIPPER_CENTER_TOOL_OFFSET_M,
) -> dict:
    """Predict TCP (wrist flange) pose at insertion success via the
    translate_object real-mode flow.

    See module docstring for math. All quaternions are xyzw; all positions are
    meters in the robot base_link / world frame.
    """
    base_world_xyz       = np.asarray(base_world_xyz, dtype=float)
    base_world_quat      = np.asarray(base_world_quat_xyzw, dtype=float)
    held_quat            = np.asarray(held_quat_xyzw, dtype=float)
    ee_orientation       = np.asarray(ee_orientation_xyzw, dtype=float)

    # 1. Load CAD: assembly target (object pose in base) + grasp offset (in object)
    seat_pos_in_base, seat_quat_in_base = load_seat_pose_in_base(base_name, object_name)
    grasp_offset_in_object               = load_grasp_offset_in_object(object_name, grasp_id)

    # 2. Project to world
    R_base       = R.from_quat(base_world_quat).as_matrix()
    R_target_rel = R.from_quat(seat_quat_in_base).as_matrix()
    R_target_abs = R_base @ R_target_rel
    target_object_position_abs = base_world_xyz + R_base @ seat_pos_in_base

    # 3. Fold symmetry — pick best equivalent that matches the held orientation
    R_held = R.from_quat(held_quat).as_matrix()
    fold_data = _load_fold_data(object_name)
    R_grasp_rotation, pos_err_m, ang_err_rad = _select_best_fold_equivalent(
        R_target_abs, R_held, grasp_offset_in_object, fold_data,
    )

    # 4. Gripper center at success = target_object_position + rotated grasp offset
    grasp_world_offset    = R_grasp_rotation @ grasp_offset_in_object
    target_gripper_center = target_object_position_abs + grasp_world_offset

    # 5. Flange (TCP) at success = gripper_center − R_EE @ tool_offset_local.
    # For face-down gripper (Phase 5 standard), R_EE rotates tool +Z to world −Z,
    # so tool_offset_world = (0, 0, −0.2286), and target_flange = gripper_center
    # + (0, 0, +0.2286). Generalised:
    R_ee = R.from_quat(ee_orientation).as_matrix()
    tool_offset_world = R_ee @ (np.array([0.0, 0.0, flange_offset_m]))
    target_flange = target_gripper_center - tool_offset_world

    # The TCP orientation at success matches the EE orientation we provided
    # (compliance descent doesn't change EE rotation).
    target_tcp_quat = ee_orientation

    return {
        "predicted_tcp_at_seat": {
            "xyz_m":     target_flange.tolist(),
            "quat_xyzw": target_tcp_quat.tolist(),
        },
        "predicted_grasp_tip_at_seat": {
            "xyz_m":     target_gripper_center.tolist(),
            "quat_xyzw": R.from_matrix(R_grasp_rotation).as_quat().tolist(),
        },
        "T_world_object_seat": {
            "xyz_m":     target_object_position_abs.tolist(),
            "quat_xyzw": R.from_matrix(R_target_abs).as_quat().tolist(),
        },
        "fold_symmetry_used": {
            "applied":           fold_data is not None,
            "R_eq_quat_xyzw":    R.from_matrix(R_grasp_rotation).as_quat().tolist(),
            "pos_error_mm":      float(pos_err_m * 1000.0),
            "angle_error_deg":   float(np.degrees(ang_err_rad)),
        },
        "grasp_point_in_object": {
            "xyz_m": grasp_offset_in_object.tolist(),
        },
        "cad_provenance": {
            "assembly_path":   str(_resolve_assembly_path(base_name)),
            "grasp_path":      str(GRASP_POINTS_DIR / f"{object_name}_grasp_points.json"),
            "flange_offset_m": float(flange_offset_m),
        },
    }


if __name__ == "__main__":
    # Smoke test for u_orange grasp_id=1, base1 at config-fixed pose,
    # face-down EE (R_y(180) ≈ quat(0, 1, 0, 0)), held part at -90° about X.
    out = predict_tcp_at_seat(
        base_name="base1",
        object_name="u_orange",
        grasp_id=1,
        base_world_xyz=[0.0, -0.4, -0.0625],
        base_world_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        held_quat_xyzw=[-0.7071067811865475, 0.0, 0.0, 0.7071067811865477],   # R_x(-90°)
        ee_orientation_xyzw=[0.0, 1.0, 0.0, 0.0],                              # R_y(180°), face-down
    )
    print(json.dumps(out, indent=2))
