"""Ruling A (lead, 2026-11-19): sim seats are scored on the final pose after release."""
import ast
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ablations.replay_sim_assembly import derive_sim_sequence
from ablations.sim_seat_scorer import seat_error
from primitives.shared.fold_symmetry import load_symmetry_data
from utils.data_path_finder import (find_assembly_json_by_base_name, get_assembly_data_dir,
                                    get_symmetry_dir)

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = {c["name"]: c for c in json.load(open(find_assembly_json_by_base_name(
    "base1", str(get_assembly_data_dir()))))["components"]}
ASSEMBLY = json.load(open(ROOT / "ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json"))


def _seated(name, T_base, dz=0.0, extra=None):
    c = COMPONENTS[name]
    p, q = c["position"], c["rotation"]["quaternion"]
    R_seat = T_base[:3, :3] @ R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    if extra is not None:
        R_seat = R_seat @ extra
    T = np.eye(4)
    T[:3, :3] = R_seat
    T[:3, 3] = T_base[:3, 3] + T_base[:3, :3] @ np.array([p["x"], p["y"], p["z"]]) + [0, 0, dz]
    return T


def test_seat_error_measures_offset_and_accepts_fold_equivalents():
    T_base = np.eye(4)
    T_base[:3, :3] = R.from_euler("z", 0.3).as_matrix()
    T_base[:3, 3] = [0.03, -0.45, -0.06]
    fold = load_symmetry_data("u_brown", str(get_symmetry_dir()))
    mm, deg = seat_error(_seated("u_brown", T_base), T_base, COMPONENTS["u_brown"], fold)
    assert mm < 1e-9 and deg < 1e-6
    mm, deg = seat_error(_seated("u_brown", T_base, dz=0.00113), T_base, COMPONENTS["u_brown"], fold)
    assert abs(mm - 1.13) < 1e-6 and deg < 1e-6
    # u_brown is 2-fold about its y axis: the flipped part is seated.
    flipped = _seated("u_brown", T_base, extra=R.from_euler("y", np.pi).as_matrix())
    assert seat_error(flipped, T_base, COMPONENTS["u_brown"], fold)[1] < 1e-6
    tilted = _seated("u_brown", T_base, extra=R.from_euler("x", 1.5, degrees=True).as_matrix())
    assert abs(seat_error(tilted, T_base, COMPONENTS["u_brown"], fold)[1] - 1.5) < 1e-6


def test_derive_sim_sequence_uses_the_production_planner():
    row = next(r for r in ASSEMBLY["assembly_order"] if r["object_name"] == "u_brown")
    seated_q = R.from_quat([COMPONENTS["u_brown"]["rotation"]["quaternion"][k] for k in "xyzw"])
    one, note_one = derive_sim_sequence("u_brown", row["tool_sequence"], "base1", seated_q.as_quat().tolist())
    assert "1 pass," in note_one and len(one) == 8
    flat = (R.from_euler("x", 90, degrees=True) * seated_q).as_quat().tolist()
    two, note_two = derive_sim_sequence("u_brown", row["tool_sequence"], "base1", flat)
    assert "2 passes" in note_two and len(two) == 15
    assert any("place_down" in step for step in two)


def test_translate_sim_insert_no_longer_gates_on_the_held_pose():
    source = (ROOT / "primitives/translate_object.py").read_text()
    assert "Pre-release held pose (logged, not gated)" in source
    assert "sim insert contact is not seated" not in source
    assert '"gated": False' in source


def test_replay_scores_each_part_then_rescored_all():
    tree = ast.parse((ROOT / "ablations/replay_sim_assembly.py").read_text())
    text = ast.unparse(tree)
    assert "__score_final_seat__" in text and "__rescore_all_seats__" in text
    assert "derive_sequence" in text and "default=True" in text
