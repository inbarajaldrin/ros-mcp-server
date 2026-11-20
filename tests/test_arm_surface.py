"""Tier 2 of the physical checks: UR's real surface (arm_surface), review w5-enclose r1.

- the exact point-to-triangle distance;
- flown stances: the 10-15 yellow recovery (#13rec8, the real arm ~2 mm from itself) stays refused; an ordinary
  10-15 yellow grasp end stance, which the enclosing capsules refuse, is admitted on the real surfaces;
- tier 2 can only admit a pair tier 1 refused (same pairs, same margin);
- fail closed: meshes missing or not the pinned ones -> every tier-1 refusal stands;
- the table: the arm's surface against the table plane.
"""
from pathlib import Path

import numpy as np
import pytest

from primitives.shared import arm_surface as S
from primitives.shared import collision as C
from primitives.shared.config import HOME_JOINTS

DRIVES_A_UR = Path("/opt/ros/humble/share/ur_robot_driver").is_dir()
needs_meshes = pytest.mark.skipif(not S.VISUAL_DIR.is_dir() and not DRIVES_A_UR,
                                  reason=f"UR's visual meshes not installed ({S.VISUAL_DIR}) and this PC drives no UR")
# replay_inverted_u_yellow_20261015_121344: the recovery's translate_object end stance (flown; the real upper arm
# passed ~2 mm from the wrist_2 housing)
NEAR_MISS = np.array([0.876917, -1.553163, 2.861902, -1.31836, 0.884919, -3.139337])
# replay_inverted_u_yellow_20261015_094845: inverted_u_yellow#2 move_to_grasp end stance (flown)
GRASP = np.array([1.695824, -1.778614, 2.781019, -2.572538, -1.566053, 3.264019])


@pytest.fixture
def fresh_surface(monkeypatch):
    """arm_surface with its load state reset (restored after the test)."""
    monkeypatch.setattr(S, "_STATE", {"links": {}})
    return S


def test_point_triangle_distance_is_exact():
    rng = np.random.default_rng(1)
    T = rng.normal(0, 0.05, (200, 3, 3))
    P = rng.normal(0, 0.08, (200, 3))
    d = S.point_triangle_distance(P, T)
    u, v = np.meshgrid(np.linspace(0, 1, 301), np.linspace(0, 1, 301))
    m = (u + v) <= 1
    u, v = u[m], v[m]
    for i in range(len(P)):
        X = T[i, 0] + u[:, None] * (T[i, 1] - T[i, 0]) + v[:, None] * (T[i, 2] - T[i, 0])
        brute = float(np.min(np.linalg.norm(X - P[i], axis=1)))
        assert d[i] <= brute + 1e-12                     # never above a point of the triangle
        assert brute - d[i] < 1e-3                       # and the grid's nearest point is close to it


@needs_meshes
def test_the_10_15_yellow_recovery_stays_refused():
    assert min(C.physical_self_clearances(NEAR_MISS))[0] < C.PHYSICAL_SELF_MARGIN_M
    why = C.physical_self_collision_refusal(NEAR_MISS)
    assert why and why.startswith("PHYSICAL_SELF_COLLISION") and "real surfaces" in why
    assert C.check_self_collision(NEAR_MISS)


@needs_meshes
def test_an_ordinary_flown_grasp_the_capsules_refuse_is_admitted_on_the_real_surfaces():
    worst = min(C.physical_self_clearances(GRASP))
    assert worst[0] < 0.0 and "upper arm" in worst[1] and "wrist_1 (housing)" in worst[2]
    assert C.physical_self_collision_refusal(GRASP) is None


@needs_meshes
def test_tier2_only_ever_admits_a_pair_tier1_refused():
    """Over random configurations: a configuration tier 1 admits is admitted, and every pair tier 2 admitted had its
    real-surface bound >= the margin (so the two-tier verdict is never looser than tier 1 on a pair it did not refine)."""
    rng = np.random.default_rng(5)
    for _ in range(60):
        q = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-2.5, -0.5), rng.uniform(0.3, 2.8),
                      rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)])
        tier1 = [c for c in C.physical_self_clearances(q) if c[0] < C.PHYSICAL_SELF_MARGIN_M]
        why = C.physical_self_collision_refusal(q)
        if not tier1:
            assert why is None or not why.startswith("PHYSICAL_SELF")
            continue
        frames, caps = C._physical_bodies(q)
        capsule_of = {label: (a, b, r) for a, b, r, label in caps}
        bounds = [S.pair_lower_bound(la, lb, frames, capsule_of, C.PHYSICAL_SELF_MARGIN_M) for _d, la, lb in tier1]
        all_clear = all(b is not None and b >= C.PHYSICAL_SELF_MARGIN_M for b in bounds)
        assert (why is None or not why.startswith("PHYSICAL_SELF")) == all_clear
        for (d1, _la, _lb), b in zip(tier1, bounds):
            if b is not None and b < C.PHYSICAL_SELF_MARGIN_M:
                assert b >= d1 - 1e-9 - 2 * S.ALLOWANCE_PER_BODY_M - 0.003      # real surfaces lie inside the capsules


def test_missing_meshes_fail_closed(fresh_surface, monkeypatch, tmp_path):
    monkeypatch.setattr(S, "VISUAL_DIR", tmp_path)
    assert not S.available() and "FileNotFoundError" in S.why_unavailable()
    why = C.physical_self_collision_refusal(GRASP)
    assert why and "real-surface check unavailable" in why


def test_meshes_that_are_not_the_pinned_ones_fail_closed(fresh_surface, monkeypatch, tmp_path):
    for name in S.MESH_SHA256:
        (tmp_path / name).write_text("not UR's mesh")
    monkeypatch.setattr(S, "VISUAL_DIR", tmp_path)
    assert not S.available() and "is not the pinned" in S.why_unavailable()
    assert C.check_self_collision(GRASP)


@needs_meshes
def test_the_arm_surface_is_checked_against_the_table():
    assert C.arm_table_refusal(np.array(HOME_JOINTS)) is None
    # home with the shoulder lifted down past horizontal: the forearm and wrists sink below the table plane
    low = np.array(HOME_JOINTS) + np.array([0, 1.2, 0, 0, 0, 0])
    why = C.arm_table_refusal(low)
    assert why and why.startswith("ARM_TABLE_COLLISION")
    assert C.check_collision_with_table(low, z_threshold=C.TABLE_HEIGHT - 1.0)      # whatever the DH threshold


def test_the_table_check_fails_closed_without_the_meshes(fresh_surface, monkeypatch, tmp_path):
    monkeypatch.setattr(S, "VISUAL_DIR", tmp_path)
    low = np.array(HOME_JOINTS) + np.array([0, 1.2, 0, 0, 0, 0])
    why = C.arm_table_refusal(low)
    assert why and why.startswith("ARM_TABLE_COLLISION")
