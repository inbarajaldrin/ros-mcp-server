"""v3.1 camera mount + D415 geometry in the tool0 (flange) frame, for the camera hard rule.

Boxes (min, max; m) fitted to the 2 mm-sampled surface of the CAD meshes mount_upper / mount_lower / d415
(fmb-cell-lab 7840ac8, mujoco/evidence/camera-mount-boxes-tool0.json): their union contains the surface,
without the empty corners of one bounding box per mesh.
"""
import functools
from pathlib import Path

import numpy as np

CAMERA_MOUNT_BOXES_TOOL = {
    "mount_upper": [
        ((-0.044, 0.0008, 0.0721), (-0.0179, 0.0196, 0.1041)),
        ((-0.0333, 0.0197, 0.0721), (-0.0178, 0.024, 0.1041)),
        ((-0.0178, 0.018, 0.0721), (0.0083, 0.0196, 0.1041)),
        ((-0.0178, 0.0197, 0.069), (0.0084, 0.0385, 0.1041)),
        ((-0.017, 0.0386, 0.0647), (0.003, 0.0574, 0.0938)),
        ((-0.017, 0.0574, 0.0603), (0.003, 0.0763, 0.0806)),
        ((-0.017, 0.0763, 0.0583), (0.0084, 0.0952, 0.0757)),
        ((-0.017, 0.0952, 0.0654), (0.0084, 0.1141, 0.0824)),
        ((0.0084, 0.0008, 0.0721), (0.0346, 0.0196, 0.1041)),
        ((0.0084, 0.0197, 0.069), (0.0346, 0.0385, 0.1041)),
        ((0.024, 0.0386, 0.0647), (0.03, 0.0574, 0.0928)),
        ((0.024, 0.0574, 0.0603), (0.03, 0.0763, 0.0805)),
        ((0.0084, 0.0763, 0.0583), (0.0346, 0.0952, 0.0758)),
        ((0.0084, 0.0952, 0.0655), (0.0345, 0.1141, 0.0824)),
        ((0.0346, 0.0008, 0.0721), (0.044, 0.0196, 0.1041)),
        ((0.0347, 0.0198, 0.0721), (0.04, 0.024, 0.0841)),
        ((0.0346, 0.0794, 0.0583), (0.0608, 0.0952, 0.0758)),
        ((0.0346, 0.0952, 0.0655), (0.0608, 0.1141, 0.0824)),
        ((0.0608, 0.0794, 0.0583), (0.087, 0.0952, 0.0757)),
        ((0.0608, 0.0952, 0.0654), (0.087, 0.1141, 0.0824)),
    ],
    "mount_lower": [
        ((-0.0333, -0.024, 0.0721), (-0.022, -0.0124, 0.1041)),
        ((-0.044, -0.0124, 0.0721), (-0.0273, -0.0007, 0.1041)),
        ((-0.022, -0.024, 0.0721), (-0.0, -0.018, 0.1041)),
        ((-0.0, -0.024, 0.0721), (0.022, -0.018, 0.1041)),
        ((0.022, -0.024, 0.0721), (0.0333, -0.0124, 0.1041)),
        ((0.0272, -0.0124, 0.0721), (0.044, -0.0007, 0.1041)),
    ],
    "d415": [
        ((-0.0145, 0.076, 0.0711), (0.0185, 0.1022, 0.097)),
        ((0.0185, 0.076, 0.0711), (0.0515, 0.1022, 0.097)),
        ((0.0515, 0.076, 0.0711), (0.0845, 0.1022, 0.097)),
    ],
}


MOUNT_SURFACE_SPACING_M = 0.005
# The boxes were fitted to 2 mm surface samples: up to 1.42 mm of the CAD surface lies outside them.  Padded by
# this much they ENCLOSE the CAD (tests/test_camera_mount_boxes_enclose.py: every 3 mm piece of the three meshes lies
# wholly inside one padded box).
CAMERA_MOUNT_BOX_PAD_M = 0.0015

# The CAD meshes (assets/cell/camera_mount_v31, rg2_base_link frame, mm) and the chain that puts them in tool0
# (robots/ur5e_rg2/composite.yaml): tool0 -> quick_changer_link rpy (0, 0, -pi/2) -> rg2_base_link xyz (0, 0, 0.0181)
# rpy (0, -pi/2, 0).
MOUNT_MESH_DIR = Path(__file__).resolve().parents[2] / "assets" / "cell" / "camera_mount_v31"
MOUNT_MESH_SHA256 = {
    "mount_upper": ("V31_upper_saddle_camera_tower_assembly_coordinates.stl",
                    "48c1ae253b5b7bc70cc79b5a8a005d37b892ac1fca7734859ba50ef529dbc7d4"),
    "mount_lower": ("V31_lower_saddle_captive_M4_assembly_coordinates.stl",
                    "66cb783ec19458d456c15118a573bce9b6f543f39fcffe37889e8ff7e3415a40"),
    "d415": ("D415_body_rg2_base_link.stl", "e1fc8ca2ab3f38685270412b1f2bcd3c5066e3ce1a0a384239dde4f55cf630a5"),
}


def tool0_from_rg2_base():
    from scipy.spatial.transform import Rotation as R
    A = np.eye(4)
    A[:3, :3] = R.from_euler("xyz", [0.0, 0.0, -np.pi / 2]).as_matrix()
    B = np.eye(4)
    B[:3, :3] = R.from_euler("xyz", [0.0, -np.pi / 2, 0.0]).as_matrix()
    B[:3, 3] = [0.0, 0.0, 0.0181]
    return A @ B


def read_binary_stl(path):
    """(n, 3, 3) triangles of a binary STL, in its own units."""
    import struct
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return a["v"].astype(float)


@functools.lru_cache(maxsize=None)
def mount_mesh_triangles_tool(name):
    """The CAD triangles of mount_upper / mount_lower / d415 in tool0 (m); raises when a file is missing or is not
    the pinned one (the callers fail closed)."""
    import hashlib
    fname, digest = MOUNT_MESH_SHA256[name]
    data = (MOUNT_MESH_DIR / fname).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{fname} is not the pinned mesh")
    M = tool0_from_rg2_base()
    return read_binary_stl(MOUNT_MESH_DIR / fname) / 1000.0 @ M[:3, :3].T + M[:3, 3]


@functools.lru_cache(maxsize=None)
def mount_mesh_vertices_tool():
    """Every distinct CAD vertex of the mount and the D415 in tool0 (m), or None when the meshes are unavailable."""
    try:
        return np.unique(np.concatenate([mount_mesh_triangles_tool(n).reshape(-1, 3) for n in MOUNT_MESH_SHA256]),
                         axis=0)
    except Exception:                                            # noqa: BLE001 - unavailable: tier 1 stands
        return None


@functools.lru_cache(maxsize=None)
def padded_box_corners_tool():
    """The 8 corners of every fitted box, padded by CAMERA_MOUNT_BOX_PAD_M (the boxes' union encloses the CAD)."""
    p = CAMERA_MOUNT_BOX_PAD_M
    out = []
    for lo, hi in _all_boxes():
        for x in (lo[0] - p, hi[0] + p):
            for y in (lo[1] - p, hi[1] + p):
                for z in (lo[2] - p, hi[2] + p):
                    out.append((x, y, z))
    return np.array(out)


def _axis_samples(lo, hi, spacing):
    n = max(1, int(np.ceil((hi - lo) / spacing)))
    return np.linspace(lo, hi, n + 1), (hi - lo) / n


@functools.lru_cache(maxsize=None)
def mount_surface_tool(spacing=MOUNT_SURFACE_SPACING_M, pad=CAMERA_MOUNT_BOX_PAD_M):
    """Every box face sampled on a grid of at most `spacing` (corners and edges included), tool frame (N, 3),
    and the sampling gap: every surface point lies within `gap` of a sample, so for any convex obstacle the true
    distance is at least (the nearest sample's distance - gap).  A box's closest point to a capsule axis is on
    its surface (mid-face or mid-edge, not only a corner).  The boxes are padded by `pad` (default
    CAMERA_MOUNT_BOX_PAD_M: their union then encloses the CAD; review r1 P3-7)."""
    pts, owner, gap = [], [], 0.0
    for b, (lo, hi) in enumerate(_all_boxes(pad)):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        grids, steps = zip(*(_axis_samples(lo[k], hi[k], spacing) for k in range(3)))
        for k in range(3):                              # the two faces normal to axis k
            u, v = [j for j in range(3) if j != k]
            gap = max(gap, 0.5 * float(np.hypot(steps[u], steps[v])))
            U, V = np.meshgrid(grids[u], grids[v], indexing="ij")
            for w in (lo[k], hi[k]):
                face = np.empty((U.size, 3))
                face[:, k], face[:, u], face[:, v] = w, U.ravel(), V.ravel()
                pts.append(face)
                owner.append(np.full(U.size, b))
    pts, owner = np.concatenate(pts), np.concatenate(owner)
    pts.setflags(write=False)
    owner.setflags(write=False)
    return pts, gap, owner


def _all_boxes(pad=0.0):
    return [(tuple(np.asarray(lo, float) - pad), tuple(np.asarray(hi, float) + pad))
            for boxes in CAMERA_MOUNT_BOXES_TOOL.values() for lo, hi in boxes]


@functools.lru_cache(maxsize=None)
def mount_box_spheres_tool(pad=CAMERA_MOUNT_BOX_PAD_M):
    """Centre (B, 3) and half-diagonal (B,) of every (padded) box, tool frame: a box whose sphere clears an obstacle
    by a margin clears it by at least that margin."""
    boxes = np.array(_all_boxes(pad), float)            # (B, 2, 3)
    centres = boxes.mean(axis=1)
    radii = 0.5 * np.linalg.norm(boxes[:, 1] - boxes[:, 0], axis=1)
    centres.setflags(write=False)
    radii.setflags(write=False)
    return centres, radii
