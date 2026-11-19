"""Split reorient routes (lead approval 2026-11-19): r48e's u_orange reorient, camera kept up."""
from pathlib import Path

import numpy as np

from primitives.shared.camera_up import camera_route_refusal
from primitives.shared.route_split import admissible_route, candidate_routes

ROOT = Path(__file__).resolve().parents[1]
# MuJoCo r48d/r48e: the planned single-point u_orange reorient (start measured, target recovered from the
# interpolated commands and the logged 2.75 rad max joint change).
START = np.array([1.89727, -1.30194, 1.700762, -1.8321, -1.803826, 1.924308])
TARGET = np.array([1.963492, -1.267425, 1.652992, -0.238788, -4.553826, -0.180144])


def rule(a, b, is_last):
    return camera_route_refusal(a, b, near_table_end=is_last)


def test_direct_line_is_refused_and_a_split_route_is_admitted():
    assert camera_route_refusal(START, TARGET, near_table_end=True).startswith("CAMERA_MOUNT_ARM_COLLISION")
    route, _ = admissible_route(START, TARGET, rule)
    assert route is not None and len(route) > 2
    assert np.allclose(route[0], START) and np.allclose(route[-1], TARGET)
    for a, b in zip(route[:-1], route[1:]):
        moved = np.abs(b - a) > 1e-9
        assert moved.any()
    # same end pose, no more travel than the direct line (axis splits move each joint once)
    travel = sum(np.sum(np.abs(b - a)) for a, b in zip(route[:-1], route[1:]))
    assert travel <= np.sum(np.abs(TARGET - START)) + 1e-9


def test_an_admissible_direct_line_is_kept_and_none_refuses():
    small = START + np.array([0, 0, 0, 0, 0, 0.1])
    route, _ = admissible_route(START, small, lambda a, b, last: camera_route_refusal(a, b))
    assert len(route) == 2
    route, reasons = admissible_route(START, TARGET, lambda a, b, last: "ALWAYS_REFUSED")
    assert route is None and reasons and all(r == "ALWAYS_REFUSED" for r in reasons)
    assert sum(1 for _ in candidate_routes(START, TARGET)) == 75


def test_rotate_object_uses_the_split_route_with_the_full_rule_set():
    source = (ROOT / "primitives/rotate_object.py").read_text()
    assert "admissible_route(self.current_joint_angles, joint_angles, segment_refusal)" in source
    assert "held_part_route_refusal(a, b, held)" in source
    assert "camera_route_refusal(a, b, near_table_end=is_last)" in source
    assert "no admissible route (direct or split by joint group)" in source
