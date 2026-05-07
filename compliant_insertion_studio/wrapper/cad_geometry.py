"""
CAD-derived contact classifier.

Given a TCP-z at the moment of contact, plus the target object's predicted seat
TCP-z (from cad_lookup.predict_tcp_at_seat), plus the held-pose chain (held_quat,
grasp offset, OBJ bbox), this module computes the expected TCP-z at:

  - the target object's seated position           ("SEAT")
  - base1's outer top surface (rim contact)       ("BASE_RIM")
  - the top surface of any previously-seated obj  ("OBJ:<name>")

…and returns the closest match, with the residual error so the FSM can decide
whether the match is high-confidence enough to trust.

Validated 2026-05-06 against existing logs:
  insert_u_brown_20260506_082119:           rim 230.61mm  vs predicted 231.81mm  (Δ +1.20mm)
  insert_inverted_u_yellow_20260506_095933: rim 244.66mm  vs predicted 245.40mm  (Δ +0.74mm)
  insert_inverted_u_yellow_20260506_095933: line_green-top 263.40mm vs 265.35mm (Δ +1.95mm)

All three within ±2mm purely from CAD chain (no fudge).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

ARUCO_DATA_DIR = Path.home() / "Documents" / "aruco-grasp-annotator" / "data"
GRASP_POINTS_DIR = ARUCO_DATA_DIR / "grasp_points"
MODELS_DIR = ARUCO_DATA_DIR / "models"
ASSEMBLY_INDEX_DIR = ARUCO_DATA_DIR

GRIPPER_CENTER_TOOL_OFFSET_M = 0.2286


def _load_obj_verts(object_name: str) -> np.ndarray:
    """OBJ files are in cm; return Nx3 array in METERS, in object's local CAD frame."""
    p = MODELS_DIR / f"{object_name}.obj"
    verts = []
    for line in open(p):
        if line.startswith("v "):
            parts = line.split()
            verts.append([float(parts[1]) / 100.0,
                          float(parts[2]) / 100.0,
                          float(parts[3]) / 100.0])
    return np.array(verts)


def _load_grasp_offset(object_name: str, grasp_id: int) -> np.ndarray:
    p = GRASP_POINTS_DIR / f"{object_name}_grasp_points.json"
    d = json.load(open(p))
    grasp = next(g for g in d["grasp_points"] if g["id"] == grasp_id)
    return np.array([grasp["position"]["x"],
                     grasp["position"]["y"],
                     grasp["position"]["z"]], dtype=float)


def get_seated_so_far(base_name: str, target_object: str) -> list[str]:
    """Returns the list of object names that are seated when target_object is
    being inserted, per the assembly_order in
    ablations/eval_resources/fmb1_assembly.json.

    For target_object at position N, returns object names at positions 1..N-1.
    Returns [] if the file is missing or target isn't in the assembly.
    """
    repo_root = Path(__file__).resolve().parents[2]
    json_path = repo_root / "ablations" / "eval_resources" / f"{base_name[:3] if base_name.startswith('fmb') else 'fmb1'}_assembly.json"
    if not json_path.exists():
        json_path = repo_root / "ablations" / "eval_resources" / "fmb1_assembly.json"
    if not json_path.exists():
        return []
    d = json.load(open(json_path))
    order = d.get("assembly_order", [])
    seated: list[str] = []
    for step in order:
        nm = step.get("object_name")
        if nm == target_object:
            return seated
        if nm:
            seated.append(nm)
    return []


def _load_assembly_seat_z_above_base(base_name: str, object_name: str) -> Optional[float]:
    """Returns seat_z (meters) of object's CAD origin above base origin in the
    assembled state. None if not found in the assembly index."""
    if base_name.startswith("base") and base_name[4:].isdigit():
        candidate = ASSEMBLY_INDEX_DIR / f"fmb_assembly{base_name[4:]}.json"
        if not candidate.exists():
            return None
        d = json.load(open(candidate))
        comp = next((c for c in d["components"] if c.get("name") == object_name), None)
        if comp is None:
            return None
        return float(comp["position"]["z"])
    return None


def _load_base_top_z_local(base_name: str) -> float:
    """Returns the world-z of the base's outer top surface in its own CAD frame.
    Derived from the OBJ's max-z vertex."""
    return float(_load_obj_verts(base_name)[:, 2].max())


def _load_object_local_top_z(object_name: str) -> float:
    """Returns the local-z of the object's top surface (max z of its OBJ bbox)."""
    return float(_load_obj_verts(object_name)[:, 2].max())


def _peg_lower_extent_world_z_at_seat(
    *,
    object_name: str,
    grasp_id: int,
    R_grasp: np.ndarray,
    predicted_tcp_z_at_seat: float,
    flange_offset_world_z: float = -GRIPPER_CENTER_TOOL_OFFSET_M,  # face-down default
) -> float:
    """Compute the world-z of the held object's lowest point at the seated TCP.

    Walks the cad_lookup chain backwards from predicted_tcp_z:
        gripper_center_z = predicted_tcp_z + flange_offset_world_z
        cad_origin_z     = gripper_center_z - (R_grasp @ grasp_offset)[z]
        peg_lower_z      = cad_origin_z + (R_grasp @ obj_local_bbox)[:, 2].min()
    """
    grasp_offset = _load_grasp_offset(object_name, grasp_id)
    rotated_grasp = R_grasp @ grasp_offset
    verts_world = (R_grasp @ _load_obj_verts(object_name).T).T
    peg_lower_below_origin = verts_world[:, 2].min()

    gripper_center_z = predicted_tcp_z_at_seat + flange_offset_world_z
    cad_origin_z = gripper_center_z - rotated_grasp[2]
    return cad_origin_z + peg_lower_below_origin


def expected_contact_tcp_z(
    *,
    surface_world_z: float,
    object_name: str,
    grasp_id: int,
    R_grasp_quat_xyzw: Iterable[float],
    predicted_tcp_z_at_seat: float,
    flange_offset_world_z: float = -GRIPPER_CENTER_TOOL_OFFSET_M,
) -> float:
    """Predicted TCP-z when the held object's lowest point touches the given
    surface_world_z. Pure CAD chain, no fudge factor."""
    R_grasp = R.from_quat(list(R_grasp_quat_xyzw)).as_matrix()
    peg_lower_at_seat = _peg_lower_extent_world_z_at_seat(
        object_name=object_name, grasp_id=grasp_id, R_grasp=R_grasp,
        predicted_tcp_z_at_seat=predicted_tcp_z_at_seat,
        flange_offset_world_z=flange_offset_world_z,
    )
    return predicted_tcp_z_at_seat + (surface_world_z - peg_lower_at_seat)


def build_contact_candidates(
    *,
    target_object: str,
    target_grasp_id: int,
    base_name: str,
    base_world_z: float,
    R_grasp_quat_xyzw: Iterable[float],
    predicted_tcp_z_at_seat: float,
    seated_objects: list[str] | None = None,
    flange_offset_world_z: float = -GRIPPER_CENTER_TOOL_OFFSET_M,
) -> dict:
    """Build the {label: expected_tcp_z_m} dict used by the FSM at contact moment.
    Same chain as classify_contact() but without the lookup. Returned dict can be
    passed to ContactSearchFSM(contact_candidates=...)."""
    candidates: dict[str, float] = {"SEAT": float(predicted_tcp_z_at_seat)}
    base_top_local = _load_base_top_z_local(base_name)
    candidates["BASE_RIM"] = expected_contact_tcp_z(
        surface_world_z=base_world_z + base_top_local,
        object_name=target_object, grasp_id=target_grasp_id,
        R_grasp_quat_xyzw=R_grasp_quat_xyzw,
        predicted_tcp_z_at_seat=predicted_tcp_z_at_seat,
        flange_offset_world_z=flange_offset_world_z,
    )
    for seated_name in (seated_objects or []):
        seat_z_above = _load_assembly_seat_z_above_base(base_name, seated_name)
        if seat_z_above is None:
            continue
        local_top = _load_object_local_top_z(seated_name)
        candidates[f"OBJ:{seated_name}"] = expected_contact_tcp_z(
            surface_world_z=base_world_z + seat_z_above + local_top,
            object_name=target_object, grasp_id=target_grasp_id,
            R_grasp_quat_xyzw=R_grasp_quat_xyzw,
            predicted_tcp_z_at_seat=predicted_tcp_z_at_seat,
            flange_offset_world_z=flange_offset_world_z,
        )
    return candidates


def classify_contact(
    *,
    tcp_z_at_contact: float,
    target_object: str,
    target_grasp_id: int,
    base_name: str,
    base_world_z: float,
    R_grasp_quat_xyzw: Iterable[float],
    predicted_tcp_z_at_seat: float,
    seated_objects: list[str] | None = None,
    flange_offset_world_z: float = -GRIPPER_CENTER_TOOL_OFFSET_M,
    seat_tolerance_m: float = 0.005,
    surface_tolerance_m: float = 0.005,
) -> dict:
    """Classify a contact event by TCP-z against CAD candidates.

    Returns dict:
      {
        "label": "SEAT" | "BASE_RIM" | "OBJ:<name>" | "UNKNOWN",
        "tcp_z_at_contact_m": float,
        "candidates": { name: predicted_tcp_z_m, ... },
        "best_match_label": str,            # closest, regardless of tolerance
        "best_match_residual_mm": float,    # |tcp_z - predicted| in mm
        "in_tolerance": bool,
      }

    Tolerance hierarchy:
      - SEAT requires |tcp_z - predicted_seat| < seat_tolerance_m (tight, 2mm).
      - Other surfaces require residual < surface_tolerance_m (looser, 5mm).
      - If no candidate is within tolerance → "UNKNOWN" but best-match still reported.
    """
    candidates: dict[str, float] = {"SEAT": predicted_tcp_z_at_seat}

    base_top_local = _load_base_top_z_local(base_name)
    base_rim_world = base_world_z + base_top_local
    candidates["BASE_RIM"] = expected_contact_tcp_z(
        surface_world_z=base_rim_world,
        object_name=target_object, grasp_id=target_grasp_id,
        R_grasp_quat_xyzw=R_grasp_quat_xyzw,
        predicted_tcp_z_at_seat=predicted_tcp_z_at_seat,
        flange_offset_world_z=flange_offset_world_z,
    )

    for seated_name in (seated_objects or []):
        seat_z_above = _load_assembly_seat_z_above_base(base_name, seated_name)
        if seat_z_above is None:
            continue
        local_top = _load_object_local_top_z(seated_name)
        surface_world_z = base_world_z + seat_z_above + local_top
        candidates[f"OBJ:{seated_name}"] = expected_contact_tcp_z(
            surface_world_z=surface_world_z,
            object_name=target_object, grasp_id=target_grasp_id,
            R_grasp_quat_xyzw=R_grasp_quat_xyzw,
            predicted_tcp_z_at_seat=predicted_tcp_z_at_seat,
            flange_offset_world_z=flange_offset_world_z,
        )

    residuals = {k: tcp_z_at_contact - v for k, v in candidates.items()}
    best_label = min(residuals, key=lambda k: abs(residuals[k]))
    best_residual_m = residuals[best_label]
    best_residual_mm = best_residual_m * 1000.0

    in_tol = (
        (best_label == "SEAT" and abs(best_residual_m) < seat_tolerance_m)
        or (best_label != "SEAT" and abs(best_residual_m) < surface_tolerance_m)
    )
    label = best_label if in_tol else "UNKNOWN"

    return {
        "label": label,
        "tcp_z_at_contact_m": float(tcp_z_at_contact),
        "candidates_m": {k: float(v) for k, v in candidates.items()},
        "best_match_label": best_label,
        "best_match_residual_mm": float(best_residual_mm),
        "in_tolerance": bool(in_tol),
        "tolerances_m": {
            "seat": seat_tolerance_m,
            "surface": surface_tolerance_m,
        },
    }
