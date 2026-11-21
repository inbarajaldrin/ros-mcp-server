"""The RG2's finger links per opening (primitives/shared/rg2_geometry.py): one model for the physical self checks and
the gripper-vs-table rule (lead 2026-11-20; agreed by w5b and w8).  Module-only: no collision / arm_surface."""
import hashlib
import math

import numpy as np
import pytest

from primitives.shared import rg2_geometry as F

pytestmark = pytest.mark.skipif(F.model() is None, reason="the RG2 finger meshes are not the pinned ones")


def test_the_links_at_the_modelled_opening_are_the_pinned_obj():
    from scipy.spatial import cKDTree
    raw = F.RG2_OBJ.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == F.RG2_OBJ_SHA256
    V = np.array([[float(x) for x in l.split()[1:4]] for l in raw.decode().splitlines() if l.startswith("v ")])
    tree = cKDTree(np.unique((V @ np.diag([-1.0, -1.0, 1.0])).round(7), axis=0))     # the obj, tool0 frame
    corr = F.tip_correction_m()
    for name, T in F.posed(F.model()[1]).items():
        if "finger" in name and "knuckle" not in name:
            T = T - np.array([0.0, 0.0, corr])        # the pinned obj is the CAD; the model carries the measured tip
        d, _ = tree.query(T.reshape(-1, 3))
        assert d.max() < 1e-6, name


def test_the_closed_fingertip_is_the_measured_one():
    """Lead ruling 2026-11-21: the closed tips at flange - 230.86 mm (config.RG2_TIP_CLOSED_M, measured on the cell);
    the meshes were 2.70 mm deeper at every width."""
    from primitives.shared.config import RG2_TIP_CLOSED_M, rg2_tip_length_m
    assert abs(F.tip_correction_m() + 0.00270) < 0.0001
    for gap_mm in (0.0, 15.0, 21.0, 42.7, 71.8):
        L = F.posed(F.alpha_of_raw((gap_mm + 9.2) / 1000.0))
        deep = max(float(X[..., 2].max()) for n, X in L.items())
        assert abs(deep - rg2_tip_length_m(gap_mm)) < 0.0002, gap_mm
    closed = F.posed(F.ALPHA_MIN)
    assert abs(max(float(X[..., 2].max()) for X in closed.values()) - RG2_TIP_CLOSED_M) < 1e-6


def test_the_outer_knuckle_carries_the_finger_pivot():
    """Parallelogram: the outer knuckle's distal pivot (the finger link's origin at the modelled opening), turned with
    the knuckle, lands where the finger's translation puts it - at every opening."""
    links, alpha0, _ = F.model()
    for side, k, f in ((-1, "lo_knuckle", "l_finger"), (1, "ro_knuckle", "r_finger")):
        P0 = np.asarray(F.LINK_ORIGIN[f][0])[None, :]
        for a in np.linspace(F.ALPHA_MIN, F.ALPHA_MAX, 9):
            via_knuckle = F._pose_points({k: P0}, a, alpha0)[k]
            via_finger = F._pose_points({f: P0}, a, alpha0)[f]
            assert np.allclose(via_knuckle, via_finger, atol=1e-9)


def test_the_pad_gap_follows_the_arc_model():
    """Rg2Arc: rubber gap = raw - 2 x 4.6 mm.  The posed pads' inner faces agree within 0.6 mm across the stroke."""
    for raw in (0.0092, 0.030, 0.060, 0.0844, 0.110):
        P = F.posed(F.alpha_of_raw(raw))
        gap = P["l_finger"][..., 0].min() - P["r_finger"][..., 0].max()
        assert abs(gap - (raw - 2 * F.ARC_OFFSET_M)) < 0.0006, (raw, gap)
    try:
        from onrobot_rg2_control.rg2_arc import Rg2Arc
    except Exception:
        return
    arc = Rg2Arc()
    for raw_mm in (10.0, 40.0, 84.4, 109.0):
        assert abs(F.alpha_of_raw(raw_mm / 1000.0) - arc.alpha(arc.theta_of_raw_width(raw_mm))) < 1e-9


@pytest.mark.parametrize("openings", [None, (21.0,), (56.7,), (100.0,), (35.0, 84.0), (("held", 43.4),),
                                      (("held", 21.0), 100.8)])
def test_rg2_finger_boxes_enclose(openings):
    """Every pinned finger-link vertex, posed by the 4-bar at random openings BETWEEN the samples and at the
    samples, lies inside its box (finger_link_boxes over the opening set; None = the whole stroke)."""
    B = F.boxes(openings)
    links, alpha0, _ = F.model()
    rng = np.random.default_rng(5)
    for lo, hi in F.alpha_ranges(openings):
        angles = list(rng.uniform(lo, hi, 25)) + F.alpha_samples(openings)[::7]
        for a in angles:
            for name, T in F._pose_points(links, a, alpha0).items():
                c, h = B[F.LABELS[name]]
                assert np.all(np.abs(T.reshape(-1, 3) - c) <= h + 1e-9), (name, math.degrees(a))


def test_the_whole_stroke_box_contains_every_link_point_at_any_known_opening():
    full = F.boxes(None)
    links, alpha0, _ = F.model()
    for w in (10.0, 40.0, 70.0, 100.0):
        for a in F.alpha_samples((w,)):
            for name, T in F._pose_points(links, a, alpha0).items():
                c, h = full[F.LABELS[name]]
                assert np.all(np.abs(T.reshape(-1, 3) - c) <= h + 1e-9), (name, w)


def test_the_shared_api():
    """finger_boxes_tool (per finger, the union of its links), worst points, tip depth; opening in metres."""
    two = F.finger_boxes_tool(0.035)
    links = F.finger_link_boxes_tool(0.035)
    assert len(two) == 2 and len(links) == 6
    for (lo, hi), side in zip(two, ("L", "R")):
        for k, (llo, lhi) in links.items():
            if f"finger {side} " in k:
                assert np.all(lo <= llo + 1e-12) and np.all(hi >= lhi - 1e-12)
    assert F.worst_finger_points_tool(None).shape == (16, 3)
    assert F.tip_z_tool(0.0092) > F.tip_z_tool(0.110)            # closed fingers reach deeper
    assert F.tip_z_tool(None) >= F.tip_z_tool(0.0092) - 1e-12


def test_to_link_frame_inverts_the_pose():
    links, alpha0, _ = F.model()
    for name, T in links.items():
        for a in (F.ALPHA_MIN, alpha0 + 0.1, F.ALPHA_MAX):
            P = F._pose_points({name: T}, a, alpha0)[name]
            assert np.abs(F.to_link_frame(name, P, a, alpha0) - T).max() < 1e-12, (name, a)
