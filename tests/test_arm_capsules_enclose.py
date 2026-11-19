"""The physical arm capsules ENCLOSE the UR5e's real surface (lead ruling 2026-11-19, w5-enclose).

The surface is ur_description's visual meshes (UR's own, pinned by sha256 in arm_surface.MESH_SHA256), placed by
ur_description's ur5e chain (arm_surface.CHAIN / VISUAL_ORIGIN, as xacro expands it) and expressed in each link's DH
frame.  That transform is checked to be the same at every configuration, so the proof in the link frame holds for all
configurations.  The surface is cut into pieces (every edge <= 20 mm); each piece must lie wholly inside ONE capsule of
its link, at least 0.4 mm inside (the fit pads 0.5 mm; a capsule is convex, a piece is the convex hull of its corners),
which proves the link's surface lies inside the union of its capsules.  The base: inside the base cylinder.
The proof is on the nominal DH chain; the cell places the capsules with the calibrated one: the largest displacement
of a mesh point between the two joint transforms is checked against arm_surface.CALIBRATION_ALLOWANCE_M (the inset
and the 10 mm margin absorb it).
Where the meshes are absent the test skips, except on a PC that drives a UR (ur_robot_driver installed): there it
fails, as it does when a mesh differs from the pinned hash.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from primitives.shared import arm_surface as S
from primitives.shared import collision as C
from primitives.shared.config import DH_DELTAS, DH_PARAMS_NOMINAL
from primitives.shared.ik import dh_transform

VISUAL = S.VISUAL_DIR
DRIVES_A_UR = Path("/opt/ros/humble/share/ur_robot_driver").is_dir()
MAX_EDGE_M = 0.02
INSET_M = 0.0004
DH_FRAME = {link: k for link, k in S.DH_FRAME.items() if k}


def _dh_frames(q):
    return S.nominal_dh_frames(q)


def _link_T(link, q):
    return S.urdf_link_T(link, q)


_dae_triangles = S.dae_triangles
RZ = S.RZ
VISUAL_ORIGIN = S.VISUAL_ORIGIN


def _subdivide(T):
    return S.subdivide(T, MAX_EDGE_M)


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _capsules_of(frame):
    return [(np.asarray(a, float), np.asarray(b, float), r, label)
            for k, a, b, r, label in tuple(C.PHYSICAL_LINK_CAPSULES) + tuple(C.PHYSICAL_WRIST_CAPSULES)
            if k == frame and label != "quick changer"]


def test_the_meshes_are_present_and_pinned_where_the_cell_runs():
    if not VISUAL.is_dir() and not DRIVES_A_UR:
        pytest.skip(f"UR's visual meshes not installed ({VISUAL}) and this PC drives no UR")
    for name, digest in S.MESH_SHA256.items():
        assert (VISUAL / name).is_file(), f"{VISUAL / name} missing on a PC that drives a UR"
        assert hashlib.sha256((VISUAL / name).read_bytes()).hexdigest() == digest, f"{name} is not the pinned mesh"


needs_meshes = pytest.mark.skipif(not VISUAL.is_dir() and not DRIVES_A_UR,
                                  reason=f"UR's visual meshes not installed ({VISUAL}) and this PC drives no UR")


@needs_meshes
@pytest.mark.parametrize("link", list(DH_FRAME))
def test_every_piece_of_the_link_surface_lies_inside_one_capsule(link):
    k = DH_FRAME[link]
    rng = np.random.default_rng(k)
    Ms = [np.linalg.inv(_dh_frames(q)[k]) @ RZ @ _link_T(link, q) @ VISUAL_ORIGIN[link][0]
          for q in rng.uniform(-3, 3, (8, 6))]
    assert max(float(np.abs(M - Ms[0]).max()) for M in Ms) < 1e-6          # the link frame relation is constant
    T = _dae_triangles(VISUAL / VISUAL_ORIGIN[link][1])
    P = _subdivide(T @ Ms[0][:3, :3].T + Ms[0][:3, 3])
    caps = _capsules_of(k)
    assert caps, f"no capsules declared for {link}"
    outside = np.min([_seg_dist(P.reshape(-1, 3), a, b).reshape(-1, 3).max(axis=1) - r for a, b, r, _ in caps], axis=0)
    assert float(outside.max()) <= -INSET_M, f"{link}: a surface piece lies {1000 * outside.max():+.2f} mm from a capsule wall"


@needs_meshes
def test_the_base_surface_lies_inside_the_base_cylinder():
    T = _dae_triangles(VISUAL / VISUAL_ORIGIN["base_link_inertia"][1])
    M = RZ @ _link_T("base_link_inertia", np.zeros(6)) @ VISUAL_ORIGIN["base_link_inertia"][0]
    P = T.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]                       # the cylinder is convex: corners suffice
    assert float(np.max(C.base_cylinder_distance(P))) <= 1e-6          # float32 noise at the mounting plane z = 0


@needs_meshes
@pytest.mark.parametrize("link", list(DH_FRAME))
def test_calibrated_dh_moves_no_mesh_point_more_than_the_allowance(link):
    """The cell places each link with the calibrated DH (DH_DELTAS); the mesh is fixed to its frame by the nominal
    chain.  Every mesh point moves at most CALIBRATION_ALLOWANCE_M between the two joint transforms (any angle)."""
    k = DH_FRAME[link]
    M = S.mesh_to_dh_frame(link)
    P = np.unique(_dae_triangles(VISUAL / VISUAL_ORIGIN[link][1]).reshape(-1, 3), axis=0) @ M[:3, :3].T + M[:3, 3]
    Ph = np.c_[P, np.ones(len(P))]
    nom = DH_PARAMS_NOMINAL[k - 1]
    cal = tuple(n + d for n, d in zip(nom, DH_DELTAS[k - 1]))
    worst = max(float(np.linalg.norm(Ph @ (dh_transform(q + cal[0], *cal[1:]) - dh_transform(q + nom[0], *nom[1:])).T,
                                     axis=1).max()) for q in np.linspace(-np.pi, np.pi, 73))
    assert worst <= S.CALIBRATION_ALLOWANCE_M
