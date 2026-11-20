"""Tier 2 for the camera-mount rule and the tier-2 early exit (lead 2026-11-20, w5b-tier2).

Stances from the u_orange diagnosis (fmb-cell-lab mujoco/evidence/w5b-uorange-reorient-diagnosis): exact mount-to-
UR5e-mesh clearances were measured there (mount box faces vs the pinned meshes, exact point-to-triangle)."""
import numpy as np
import pytest

from primitives.shared import arm_surface as A
from primitives.shared import camera_up as cu
from primitives.shared import collision as C

pytestmark = pytest.mark.skipif(not A.available(), reason="pinned UR5e meshes not installed (tier 2 unavailable)")


def _at(start, target, alpha):
    a, b = np.asarray(start, float), np.asarray(target, float)
    return a + alpha * (b - a)


# 10-24 10:19 u_orange#5 as flown: the capsule says 7.8 mm at alpha 0.575; the real forearm mesh is >= 21 mm away
L1019 = ([2.04036, -0.98078, 1.22732, -1.81671, -1.56601, 0.43071], [2.05205, -0.97607, 1.2199, -0.24438, -4.62251, 3.14295])
# 10-24 09:28 u_orange#5 as flown: capsule -1.0 mm at alpha 0.30; real mesh 8.8 mm (under the 10 mm margin)
L0928 = ([2.04006, -1.05613, 1.34713, -1.86118, -1.56601, -0.1588], [2.06132, -1.04722, 1.33347, -0.28673, -4.55958, -3.14026])
# the twin's seed-48 direct line: a real near collision (mesh <= ~1.5 mm) at alpha 0.625
TWIN = ([1.8915394, -1.3049388, 1.7048768, -1.9697987, -1.5659160, 2.1062887], [1.9573, -1.27193, 1.66131, -0.38907, -4.31629, 0.00206])


def _tier1_only(q, monkeypatch):
    monkeypatch.setattr(A, "mount_lower_bound", lambda *a, **k: None)
    monkeypatch.setattr(A, "why_unavailable", lambda: "test")
    return cu.camera_mount_arm_refusal(q)


def test_tier2_admits_a_capsule_refusal_the_real_mesh_clears(monkeypatch):
    q = _at(*L1019, 0.575)
    assert cu.camera_mount_arm_refusal(q) is None
    assert _tier1_only(q, monkeypatch).startswith("CAMERA_MOUNT_ARM_COLLISION")   # tier 1 alone refused it


def test_tier2_keeps_a_real_shortfall_refused():
    why = cu.camera_mount_arm_refusal(_at(*L0928, 0.30))
    assert why.startswith("CAMERA_MOUNT_ARM_COLLISION") and "real surfaces" in why


def test_tier2_keeps_a_near_collision_refused():
    why = cu.camera_mount_arm_refusal(_at(*TWIN, 0.625))
    assert why.startswith("CAMERA_MOUNT_ARM_COLLISION") and "real surfaces <" in why          # decided early


def test_fail_closed_without_the_meshes(monkeypatch):
    why = _tier1_only(_at(*L1019, 0.575), monkeypatch)
    assert "real-surface check unavailable" in why


def test_tier2_is_only_asked_about_what_tier1_refused(monkeypatch):
    monkeypatch.setattr(A, "mount_lower_bound", lambda *a, **k: pytest.fail("tier 2 asked about a tier-1 admit"))
    home = [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
    assert cu.camera_mount_arm_refusal(home) is None


@pytest.mark.parametrize("q", [_at(*TWIN, 0.625), _at(*L0928, 0.30), [1.5708, -0.2, 3.1, -1.5708, -1.5708, 0.0]])
def test_the_early_exit_never_changes_a_verdict(monkeypatch, q):
    fast_mount, fast_self = cu.camera_mount_arm_refusal(q), C.physical_self_collision_refusal(q)
    monkeypatch.setattr(A, "EARLY_EXIT", False)
    slow_mount, slow_self = cu.camera_mount_arm_refusal(q), C.physical_self_collision_refusal(q)
    assert (fast_mount is None) == (slow_mount is None) and (fast_self is None) == (slow_self is None)


def test_an_early_refusal_is_not_reported_as_a_lower_bound():
    q = [1.5708, -0.2, 3.1, -1.5708, -1.5708, 0.0]          # elbow folded into the shoulder: deep pairs
    why = C.physical_self_collision_refusal(q)
    assert why and ("real surfaces <" in why or "real surfaces >=" in why)
    if "real surfaces <" in why:
        assert "sample pair" in why


def test_preload_loads_every_link():
    th = A.preload_async()
    th.join(timeout=120)
    assert not th.is_alive() and all(A._STATE["links"].get(link) is not None for link in A.DH_FRAME)


# ---- the RG2 housing tier 2 ------------------------------------------------------------------------------------------
ROUTES_75_78 = {   # u_orange seed-48 reorient, segment 1 of the two routes a09377a admitted (diagnosis, lab b42319a)
    "75": ([1.891539, -1.304939, 1.704877, -1.969799, -1.565916, 3.677085],
           [1.9573, -1.27193, 1.66131, -0.38907, -4.31629, 1.572856]),
    "78": ([1.891539, -1.304939, 1.704877, -1.969799, -1.565916, -1.035304],
           [1.9573, -1.27193, 1.66131, -0.38907, -4.31629, -3.139533]),
}


def _rg2_pair(q):
    frames, caps = C._physical_bodies(np.asarray(q, float))
    capsule_of = {lab: (a, b, r) for a, b, r, lab in caps}
    return A.pair_lower_bound("forearm 1/8", "RG2 body", frames, capsule_of, 0.010)


needs_rg2 = pytest.mark.skipif(A.rg2_housing() is None, reason="the cell's RG2 mesh is not the pinned one")


@needs_rg2
@pytest.mark.parametrize("route, alpha", [("75", 0.6), ("75", 0.65), ("78", 0.6), ("78", 0.65)])
def test_routes_75_78_stances_are_admitted_on_the_real_housing(route, alpha):
    """The box refused these (forearm vs RG2 body, capsules ~-17.6 mm, box bound 8.2-8.3 mm); the real housing is
    18.7-21.3 mm away (exact mesh): admitted, allowances included."""
    q = _at(*ROUTES_75_78[route], alpha)
    assert [c for c in C.physical_self_clearances(q) if c[0] < 0.010 and "RG2 body" in c[1:]], "tier 1 must refuse"
    assert _rg2_pair(q) >= 0.010 and C.physical_self_collision_refusal(q) is None


@needs_rg2
@pytest.mark.parametrize("offset_mm, refused", [(4.0, True), (30.0, False)])
def test_a_real_housing_near_collision_is_refused(offset_mm, refused):
    """Forearm surface points placed at a known distance off the real housing (a housing piece's centre pushed out
    along its normal): 4 mm is a near collision - refused; 30 mm clears the margin with allowances."""
    T, cent, rho = A.rg2_housing()
    k = int(np.argmax(np.abs(cent[:, 0])))                 # a piece on the housing's side wall
    n = np.cross(T[k, 1] - T[k, 0], T[k, 2] - T[k, 0]); n /= np.linalg.norm(n)
    if n @ (cent[k] - cent.mean(axis=0)) < 0:
        n = -n                                             # outward
    L = (cent[k] + offset_mm / 1000.0 * n)[None, :]
    d = A._rg2_lower_bound(L, 0.0, 0.010)
    assert (d < 0.010) == refused


def test_rg2_tier2_fails_closed_to_the_box(monkeypatch):
    monkeypatch.setattr(A, "rg2_housing", lambda: None)
    q = _at(*ROUTES_75_78["75"], 0.6)
    assert C.physical_self_collision_refusal(q).startswith("PHYSICAL_SELF_COLLISION")     # the box refuses it


def test_the_disk_cache_round_trips_and_a_bad_file_is_recomputed(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "CACHE_DIR", tmp_path)
    exact = A._load_link("forearm_link")
    first = A._load_link_cached("forearm_link")                 # computes and writes
    files = list(tmp_path.glob("forearm_link-*.npz"))
    assert len(files) == 1
    again = A._load_link_cached("forearm_link")                 # reads
    for lab in exact:
        assert np.array_equal(again[lab].samples, exact[lab].samples) and np.array_equal(again[lab].pieces, exact[lab].pieces)
        assert again[lab].gap == exact[lab].gap and again[lab].frame == exact[lab].frame
    files[0].write_bytes(b"not a cache")                        # corrupt: recomputed, never trusted
    fixed = A._load_link_cached("forearm_link")
    assert all(np.array_equal(fixed[lab].pieces, exact[lab].pieces) for lab in exact)
    assert set(first) == set(exact)
