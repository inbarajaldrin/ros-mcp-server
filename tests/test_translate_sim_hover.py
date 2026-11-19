import ast
import math
from pathlib import Path


TRANSLATE = Path(__file__).resolve().parents[1] / "primitives/translate_object.py"


def test_hover_ik_admits_the_whole_interpolated_path_and_prefers_short_travel():
    """An endpoint-safe IK answer is not a safe arm move to the insert hover."""

    source = TRANSLATE.read_text()
    tree = ast.parse(source)
    solve = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "compute_ik_with_current_seed"
    )
    body = ast.get_source_segment(source, solve)
    assert "nearest_joint_equivalent(joints, self.current_joint_angles)" in body
    assert "INSERT_HOVER_MAX_ARM_JOINT_TRAVEL_RAD" in body
    assert "INSERT_HOVER_MAX_WRIST3_TRAVEL_RAD" in body
    assert "WRIST3_LIMIT_RAD" in body
    assert 'rejected["travel"]' in body
    assert "check_trajectory_collision(" in body
    assert 'rejected["path"]' in body
    assert "safe.sort(key=lambda item: (item[1], item[0]))" in body


def test_sim_insert_reobserves_the_friction_held_object_before_descent():
    source = TRANSLATE.read_text()
    assert "Post-hover object error" in source
    assert "clearance_consumed_m <= SIM_INSERT_CLEARANCE_COMMIT_M" in source
    cad_corner_radius_m = math.sqrt(0.035**2 + 0.035**2 + 0.0105**2)
    measured_consumption = 0.000063 + cad_corner_radius_m * math.sin(math.radians(0.774))
    assert measured_consumption < 0.0011
    outside_consumption = 0.000063 + cad_corner_radius_m * math.sin(math.radians(1.2))
    assert outside_consumption > 0.0011
    assert "SIM_INSERT_PART_CORNER_RADIUS_M = 0.0506" in source
    assert "SIM_INSERT_ORIENTATION_COMMIT_DEG" not in source
    assert "correction_gain=1.0" in source
    assert "correct_orientation=False" in source
    assert "dxyz_mm=" in source
    assert "rotvec_deg=" in source
    assert "Insert-chain commit:" in source
    assert "base_gt_vs_used" in source
    assert "held_gt_vs_used" in source
    assert "commit_tcp_vs_required_from_gt" in source
    assert "settle_sim_observation(node, args.object_name, 1.0)" in source
    assert "settle_sim_observation(node, args.object_name, 0.5)" in source
    assert "pose.header = msg.header" in source
    assert 'SIM_INSERT_HOVER_BIAS_M if object_name == "u_brown" else 0.0' in source
    assert "run_move_down(mode='sim', to_table=True, require_contact=True)" in source
    assert "Pre-release held pose (logged, not gated)" in source  # ruling A, 2026-11-19
    assert "seat_mm > 1.0 or seat_deg > 1.0" not in source  # gated after release (sim_seat_scorer)

