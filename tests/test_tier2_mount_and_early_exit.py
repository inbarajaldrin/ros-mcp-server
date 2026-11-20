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
# 10-24 09:28 u_orange#5 as flown: capsule -1.0 mm at alpha 0.30; UR mesh vs the mount's boxes 8.8 mm, vs the exact
# mount meshes 10.08 mm (admitted: it was flown on the real cell); the route stays refused (camera toward the table)
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


def test_tier2_keeps_a_real_shortfall_refused(monkeypatch):
    """The twin's direct u_orange line at 0.525: the exact meshes overlap (-0.6 mm after allowances)."""
    monkeypatch.setattr(A, "EARLY_EXIT", False)
    why = cu.camera_mount_arm_refusal(_at(*TWIN, 0.525))
    assert why.startswith("CAMERA_MOUNT_ARM_COLLISION") and "real surfaces >=" in why


def test_the_flown_0928_stance_is_admitted_on_the_exact_mount_mesh_and_its_route_stays_refused():
    """Tier 2's mount side is the exact pinned mount/D415 meshes (lead 2026-11-20): the flown 10-24 09:28 stance clears
    at 10.08 mm (the mount boxes' faces said 8.8); the route is still refused at its end (camera toward the table)."""
    from primitives.shared.camera_up import camera_route_refusal
    assert cu.camera_mount_arm_refusal(_at(*L0928, 0.30)) is None
    why = camera_route_refusal(np.asarray(L0928[0]), np.asarray(L0928[1]), near_table_end=True)
    assert why and why.startswith("CAMERA_TOWARD_TABLE")


def test_mount_tier2_fails_closed_without_the_pinned_mount_meshes(monkeypatch, tmp_path):
    from primitives.shared import camera_geometry as CG
    CG.mount_mesh_triangles_tool.cache_clear()                 # camera_geometry caches the meshes per name
    try:
        monkeypatch.setattr(CG, "MOUNT_MESH_DIR", tmp_path)
        assert A.mount_mesh() is None
        why = cu.camera_mount_arm_refusal(_at(*L0928, 0.30))
        assert why and "real-surface check unavailable" in why
    finally:
        CG.mount_mesh_triangles_tool.cache_clear()


def test_extra_mount_solids_are_always_in_tier2(monkeypatch):
    """The hook for parts the pinned meshes lack (the D415 USB plug, w8 P3-6): a box placed on the arm's surface makes
    tier 2 refuse a stance it otherwise admits."""
    from primitives.shared.collision import dh_link_frames
    q = _at(*L0928, 0.30)
    assert cu.camera_mount_arm_refusal(q) is None
    frames = dh_link_frames(q)
    S, _ = A._world(A.body("forearm (wrist_1 end) 1/2"), frames)
    R6, p6 = frames[6][:3, :3], frames[6][:3, 3]
    centre = ((S - p6) @ R6)[0]                                 # a UR surface sample, in tool0
    monkeypatch.setattr(A, "MOUNT_EXTRA_SOLIDS_TOOL", [(centre, np.array([0.005, 0.005, 0.005]))])
    assert A.mount_lower_bound("forearm (wrist_1 end) 1/2", frames, 0.010) < 0.010


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
    C.set_gripper_openings((20.4,))              # the twin's measured close on u_orange (20.37 mm), as the replay passes
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


def test_the_rg2_tier1_shapes_enclose_the_pinned_rg2_mesh():
    """Lead ruling 2026-11-20: tier 1 must be strictly conservative (tier 2 only reconsiders tier-1 refusals).  The
    pinned RG2 mesh, at every height (flange frame, mounted rpy 0 0 pi): its HOUSING (the mesh minus the six finger
    links) inside RG2_BODY_BOX or the quick-changer capsule alone - so tier 1 holds at any opening (review r1 P3-3) -
    and its fingers inside their own boxes over the stroke (the mesh's fingers are at one opening of it; the
    per-opening boxes are tested in test_rg2_geometry).  The housing's top (148.05 mm) is inside the box."""
    import hashlib
    from scipy.spatial import cKDTree
    from primitives.shared import rg2_geometry as F
    raw = A.RG2_MESH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != A.RG2_MESH_SHA256 or F.model() is None:
        pytest.skip("the cell's RG2 mesh or finger links are not the pinned ones")
    V = np.array([[float(x) for x in l.split()[1:4]] for l in raw.decode().splitlines() if l.startswith("v ")])
    V = np.unique((V @ np.diag([-1.0, -1.0, 1.0])).round(7), axis=0)
    fingers = np.vstack([t.reshape(-1, 3) for t in F.posed(F.model()[1]).values()])
    is_finger = cKDTree(fingers).query(V)[0] < 1e-6
    housing, finger_v = V[~is_finger], V[is_finger]
    assert len(housing) > 1000 and len(finger_v) > 1000
    c, h = np.asarray(C.RG2_BODY_BOX[0]), np.asarray(C.RG2_BODY_BOX[1])
    in_box = np.all(np.abs(housing - c) <= h, axis=1)
    _f, a, b, r, _l = next(cap for cap in C.PHYSICAL_WRIST_CAPSULES if cap[4] == "quick changer")
    a, b = np.asarray(a), np.asarray(b)
    t = np.clip(((housing - a) @ (b - a)) / float((b - a) @ (b - a)), 0.0, 1.0)
    in_qc = np.linalg.norm(housing - (a + t[:, None] * (b - a)), axis=1) <= r
    outside = housing[~(in_box | in_qc)]
    assert len(outside) == 0, f"{len(outside)} housing vertices outside tier 1, e.g. {np.round(outside[:3] * 1000, 1)} mm"
    assert 0.1480 < housing[:, 2].max() <= c[2] + h[2]
    inside = np.zeros(len(finger_v), bool)
    for fc, fh in F.boxes(None).values():
        inside |= np.all(np.abs(finger_v - fc) <= fh, axis=1)
    assert inside.all()


# 10-15 flown yellow steps (replay_inverted_u_yellow_20261015_*.snapshots.jsonl, joints before -> after)
YELLOW_1015_5 = ([1.696132, -2.23406, 2.432704, -1.768843, -1.566006, 3.264195],
                 [1.188665, -2.082274, 2.375037, -0.287314, -0.374753, 3.136008])
YELLOW_1015_13REC8_END = [0.876917, -1.553163, 2.861902, -1.31836, 0.884919, -3.139337]


@needs_rg2
def test_the_housing_is_the_mesh_without_its_fingers_at_every_height():
    """Tier 2's RG2 body is the pinned mesh minus the six finger links (lead ruling 2026-11-20, tier 2 only): 10000
    triangles before subdivision, up to the housing's top (148.05 mm); tier 1 keeps the box."""
    T, _c, _r = A.rg2_housing()
    assert 0.1480 < T[:, :, 2].max() < 0.1481
    assert T[:, :, 2].max() <= C.RG2_BODY_BOX[0][2] + C.RG2_BODY_BOX[1][2]


@needs_rg2
def test_the_10_15_yellow_rotate_is_admitted_at_its_measured_width_and_refused_without_it():
    """The flown 10-15 yellow #5 rotate: at its measured width (43.4 mm) the housing is >= 29.85 mm and the fingers
    >= 11.99 mm away (after allowances): admitted all along.  Width unknown (the whole stroke): refused."""
    a, b = (np.asarray(x) for x in YELLOW_1015_5)
    C.set_gripper_openings((43.4,))
    assert all(C.physical_self_collision_refusal(a + t * (b - a)) is None for t in np.linspace(0, 1, 41))
    C.set_gripper_openings(None)
    whys = [C.physical_self_collision_refusal(a + t * (b - a)) for t in np.linspace(0, 1, 41)]
    assert any(w and "RG2 finger" in w for w in whys)


@needs_rg2
def test_the_10_15_wrist2_overlap_stays_refused():
    for w in ((43.4,), None):
        C.set_gripper_openings(w)
        why = C.physical_self_collision_refusal(YELLOW_1015_13REC8_END)
        assert why and "wrist_2" in why, (w, why)


@pytest.mark.parametrize("q", [_at(*TWIN, 0.525), _at(*L0928, 0.30), _at(*L0928, 0.35), _at(*L1019, 0.575)])
def test_the_box_prefilter_is_a_lower_bound_and_never_changes_a_verdict(monkeypatch, q):
    """mount_lower_bound keeps the padded boxes' exact distance for samples that clear the margin by it; the boxes
    enclose the meshes, so the result is <= the true sample-to-mesh distance (brute force over every sample, less the
    same gap and allowances), and its verdict is the true one's."""
    from scipy.spatial import cKDTree
    from primitives.shared.collision import dh_link_frames, physical_link_capsules
    monkeypatch.setattr(A, "EARLY_EXIT", False)
    q = np.asarray(q, float)
    frames = dh_link_frames(q)
    T, cent, rho, tree = A.mount_mesh()
    R6, p6 = frames[6][:3, :3], frames[6][:3, 3]
    lo, hi = A._mount_boxes()
    checked = 0
    for lab in [lab for *_x, lab in physical_link_capsules(q)]:
        surf = A.body(lab)
        if surf is None:
            continue
        S, _ = A._world(surf, frames)
        L = (S - p6) @ R6
        dbox = np.linalg.norm(np.maximum(np.maximum(lo[None] - L[:, None], L[:, None] - hi[None]), 0.0), axis=2).min()
        if dbox > 0.03:
            continue                                # far from the mount: the prefilter decides nothing there
        checked += 1
        got = A.mount_lower_bound(lab, frames, 0.010)
        dc, _ = tree.query(L, k=1)
        true = float(A._exact_min(L, T, cent, dc + rho).min()) - surf.gap - 2 * A.ALLOWANCE_PER_BODY_M
        assert got <= true + 1e-9, lab
        assert (got >= 0.010) == (true >= 0.010), lab
    assert checked >= 1
