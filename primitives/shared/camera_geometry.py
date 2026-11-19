"""v3.1 camera mount + D415 geometry in the tool0 (flange) frame, for the camera hard rule.

Boxes (min, max; m) fitted to the 2 mm-sampled surface of the CAD meshes mount_upper / mount_lower / d415
(fmb-cell-lab 7840ac8, mujoco/evidence/camera-mount-boxes-tool0.json): their union contains the surface,
without the empty corners of one bounding box per mesh.
"""
import functools

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


def _axis_samples(lo, hi, spacing):
    n = max(1, int(np.ceil((hi - lo) / spacing)))
    return np.linspace(lo, hi, n + 1), (hi - lo) / n


@functools.lru_cache(maxsize=None)
def mount_surface_tool(spacing=MOUNT_SURFACE_SPACING_M):
    """Every box face sampled on a grid of at most `spacing` (corners and edges included), tool frame (N, 3),
    and the sampling gap: every surface point lies within `gap` of a sample, so for any convex obstacle the true
    distance is at least (the nearest sample's distance - gap).  A box's closest point to a capsule axis is on
    its surface (mid-face or mid-edge, not only a corner)."""
    pts, owner, gap = [], [], 0.0
    for b, (lo, hi) in enumerate(_all_boxes()):
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


def _all_boxes():
    return [box for boxes in CAMERA_MOUNT_BOXES_TOOL.values() for box in boxes]


@functools.lru_cache(maxsize=None)
def mount_box_spheres_tool():
    """Centre (B, 3) and half-diagonal (B,) of every box, tool frame: a box whose sphere clears an obstacle by a
    margin clears it by at least that margin."""
    boxes = np.array(_all_boxes(), float)               # (B, 2, 3)
    centres = boxes.mean(axis=1)
    radii = 0.5 * np.linalg.norm(boxes[:, 1] - boxes[:, 0], axis=1)
    centres.setflags(write=False)
    radii.setflags(write=False)
    return centres, radii
