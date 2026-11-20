"""The camera hard rule's mount/D415 geometry ENCLOSES the CAD (lead 2026-11-19).

The v3.1 saddles and the D415 body (assets/cell/camera_mount_v31, sha256-pinned in camera_geometry) are placed in tool0
by the composite chain.  Cut into pieces (every edge <= 3 mm), each piece must lie wholly inside ONE fitted box padded
by CAMERA_MOUNT_BOX_PAD_M (a box is convex), so the padded boxes' union contains the mount's surface.  The clearance
to the table is then measured on those boxes, and below the floor on the exact lowest CAD vertex.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from primitives.shared import camera_geometry as CG
from primitives.shared import camera_up as CU

MAX_EDGE_M = 0.003


def _subdivide(T):
    done = []
    while len(T):
        e = np.max(np.stack([np.linalg.norm(T[:, i] - T[:, (i + 1) % 3], axis=1) for i in range(3)]), axis=0)
        done.append(T[e <= MAX_EDGE_M])
        T = T[e > MAX_EDGE_M]
        if not len(T):
            break
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        T = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1), np.stack([ca, bc, c], 1),
                            np.stack([ab, bc, ca], 1)])
    return np.concatenate(done)


@pytest.mark.parametrize("name", list(CG.MOUNT_MESH_SHA256))
def test_every_piece_of_the_cad_lies_inside_one_padded_box(name):
    P = _subdivide(CG.mount_mesh_triangles_tool(name))       # raises if the mesh is not the pinned one
    V = P.reshape(-1, 3)
    p = CG.CAMERA_MOUNT_BOX_PAD_M
    best = np.full(len(P), np.inf)
    for lo, hi in CG._all_boxes():
        lo, hi = np.asarray(lo) - p, np.asarray(hi) + p
        out = np.linalg.norm(np.maximum(np.maximum(lo - V, V - hi), 0.0), axis=1).reshape(-1, 3).max(axis=1)
        best = np.minimum(best, out)
    assert float(best.max()) <= 0.0, f"{name}: a piece lies {1000 * best.max():.2f} mm outside every padded box"


def test_the_chain_puts_the_cad_where_the_model_has_it():
    """The meshes' extent in tool0 matches the envelopes measured on the lab model's visual meshes (to 0.1 mm)."""
    for name, (lo, hi) in CU.CAMERA_GEOMETRY_TOOL.items():
        V = CG.mount_mesh_triangles_tool(name).reshape(-1, 3)
        assert np.allclose(V.min(axis=0), lo, atol=1e-4) and np.allclose(V.max(axis=0), hi, atol=1e-4)


def test_the_clearance_is_the_lowest_cad_point_where_the_boxes_are_under_the_floor():
    rng = np.random.default_rng(2)
    V = CG.mount_mesh_vertices_tool()
    C = CG.padded_box_corners_tool()
    for _ in range(200):
        Rf = R.random(random_state=rng).as_matrix()
        pf = np.array([0.4, -0.3, rng.uniform(-0.05, 0.2)])
        boxes = float(np.min(C @ Rf[2] + pf[2]) + 0.08)
        exact = float(np.min(V @ Rf[2] + pf[2]) + 0.08)
        got = CU.mount_lowest_above_table(Rf, pf, -0.08)
        assert exact >= boxes - 1e-12                      # the padded boxes enclose the CAD
        assert got == pytest.approx(boxes if boxes >= CU.CAMERA_CLEARANCE_M else exact)


def test_missing_meshes_fall_back_to_the_padded_boxes(monkeypatch, tmp_path):
    monkeypatch.setattr(CG, "MOUNT_MESH_DIR", tmp_path)
    CG.mount_mesh_triangles_tool.cache_clear()
    CG.mount_mesh_vertices_tool.cache_clear()
    try:
        assert CG.mount_mesh_vertices_tool() is None
        Rf = R.from_euler("x", 100, degrees=True).as_matrix()
        pf = np.array([0.4, -0.3, -0.05])
        boxes = float(np.min(CG.padded_box_corners_tool() @ Rf[2] + pf[2]) + 0.08)
        assert CU.mount_lowest_above_table(Rf, pf, -0.08) == pytest.approx(boxes)
    finally:
        CG.mount_mesh_triangles_tool.cache_clear()
        CG.mount_mesh_vertices_tool.cache_clear()
