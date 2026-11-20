"""
Collision detection utilities for UR5e robot.

Provides table collision, self-collision, and configuration safety checks
using DH-based forward kinematics and capsule-based link models.
"""

import re

import numpy as np

from primitives.shared.config import DH_PARAMS, TABLE_HEIGHT
from primitives.shared.ik import dh_transform

# UR5e link capsule radii (m) about the DH joint-origin segments: base, upper arm,
# forearm, wrist_1, wrist_2, wrist_3.  Conservative estimates, larger than the
# measured link radii.  The capsules sit on the DH segments, not on the physical
# links: the physical upper arm runs about 0.13 m off its DH segment along the
# shoulder axis, an approximation these radii do NOT cover.  One geometry for the
# real arm and the simulator; the single source for self-collision rejection,
# rotate_object's clearance ranking, the camera mount-vs-arm rule and the held-part
# check.  Do not shrink it to admit a stance.
UR5E_LINK_RADII = (0.075, 0.065, 0.055, 0.045, 0.045, 0.040)


# The arm's real surface, enclosed (lead ruling 2026-11-19, w5-enclose): capsules fitted to ur_description's VISUAL
# meshes (UR's own surface meshes) in this DH chain's link frames, where each mesh is fixed (checked to 1e-9).  Every
# piece of each link's surface (subdivided to <= 20 mm) lies wholly inside one of its capsules, >= 0.5 mm inside
# (tubes are split along their axis into 8 capsules and joint housings into 2, so a tapered or flaring part is
# not given its thickest radius):
# tests/test_arm_capsules_enclose.py proves it; the fit is fmb-cell-lab mujoco/arm-capsules/fit_surface_capsules.py.
# They replace the twin's Menagerie capsules, which left the real surface up to 27 mm outside (upper arm).  Used by
# the physical self-check (with the DH check), and - alone - by the camera mount and held-part checks against the arm.
# (DH frame, end a, end b, radius m, label)
PHYSICAL_LINK_CAPSULES = (
    (1, (0.0, -0.0639, 0.0), (0.0, 0.01408, 0.0), 0.06031, "shoulder (pan housing) 1/2"),
    (1, (0.0, -0.00761, 0.0), (0.0, 0.07152, 0.0), 0.06061, "shoulder (pan housing) 2/2"),
    (1, (0.0, 0.0, -0.0601), (0.0, 0.0, 0.0136), 0.06046, "shoulder (lift housing) 1/2"),
    (1, (0.0, 0.0, 0.00124), (0.0, 0.0, 0.0744), 0.06051, "shoulder (lift housing) 2/2"),
    (2, (0.425, 0.0, 0.0744), (0.425, 0.0, 0.15017), 0.06034, "upper arm (shoulder end) 1/2"),
    (2, (0.425, 0.0, 0.13433), (0.425, 0.0, 0.20983), 0.06085, "upper arm (shoulder end) 2/2"),
    (2, (-0.05959, 0.0, 0.1384), (0.0131, 0.0, 0.1384), 0.05938, "upper arm 1/8"),
    (2, (0.00166, 0.0, 0.1384), (0.07763, 0.0, 0.1384), 0.06082, "upper arm 2/8"),
    (2, (0.07693, 0.0, 0.1384), (0.15467, 0.0, 0.1384), 0.05885, "upper arm 3/8"),
    (2, (0.13866, 0.0, 0.1384), (0.21412, 0.0, 0.1384), 0.04391, "upper arm 4/8"),
    (2, (0.21188, 0.0, 0.1384), (0.28735, 0.0, 0.1384), 0.04391, "upper arm 5/8"),
    (2, (0.27134, 0.0, 0.1384), (0.34904, 0.0, 0.1384), 0.05885, "upper arm 6/8"),
    (2, (0.34838, 0.0, 0.1384), (0.42435, 0.0, 0.1384), 0.06036, "upper arm 7/8"),
    (2, (0.41453, 0.0, 0.1384), (0.4856, 0.0, 0.1384), 0.06068, "upper arm 8/8"),
    (2, (0.0, 0.0, 0.0602), (0.0, 0.0, 0.14383), 0.06026, "upper arm (elbow end) 1/2"),
    (2, (0.0, 0.0, 0.12481), (0.0, 0.0, 0.20983), 0.0611, "upper arm (elbow end) 2/2"),
    (3, (0.3922, 0.0, -0.04058), (0.3922, 0.0, 0.01153), 0.04903, "forearm (elbow end) 1/2"),
    (3, (0.3922, 0.0, 0.00515), (0.3922, 0.0, 0.06022), 0.0616, "forearm (elbow end) 2/2"),
    (3, (-0.03887, 0.0, 0.0077), (0.0308, 0.0, 0.0077), 0.03934, "forearm 1/8"),
    (3, (0.0104, 0.0, 0.0077), (0.08407, 0.0, 0.0077), 0.0401, "forearm 2/8"),
    (3, (0.08386, 0.0, 0.0077), (0.14597, 0.0, 0.0077), 0.03924, "forearm 3/8"),
    (3, (0.14535, 0.0, 0.0077), (0.20788, 0.0, 0.0077), 0.03925, "forearm 4/8"),
    (3, (0.20684, 0.0, 0.0077), (0.26979, 0.0, 0.0077), 0.03934, "forearm 5/8"),
    (3, (0.26833, 0.0, 0.0077), (0.3284, 0.0, 0.0077), 0.0496, "forearm 6/8"),
    (3, (0.32832, 0.0, 0.0077), (0.39489, 0.0, 0.0077), 0.06129, "forearm 7/8"),
    (3, (0.38312, 0.0, 0.0077), (0.45055, 0.0, 0.0077), 0.05877, "forearm 8/8"),
    (3, (0.0, 0.0, -0.04945), (0.0, 0.0, 0.01166), 0.04031, "forearm (wrist_1 end) 1/2"),
    (3, (0.0, 0.0, -0.00285), (0.0, 0.0, 0.06045), 0.03896, "forearm (wrist_1 end) 2/2"),
    (4, (0.0, -0.07355, 0.0), (0.0, -0.05448, 0.0), 0.03826, "wrist_1 1/8"),
    (4, (0.0, -0.06547, 0.0), (0.0, -0.03549, 0.0), 0.03827, "wrist_1 2/8"),
    (4, (0.0, -0.05713, 0.0), (0.0, -0.02155, 0.0), 0.04037, "wrist_1 3/8"),
    (4, (0.0, -0.0407, 0.0), (0.0, -0.00857, 0.0), 0.03918, "wrist_1 4/8"),
    (4, (0.0, -0.02736, 0.0), (0.0, -0.00141, 0.0), 0.03911, "wrist_1 5/8"),
    (4, (0.0, 0.01448, 0.0), (0.0, 0.02612, 0.0), 0.03934, "wrist_1 7/8"),
    (4, (0.0, 0.02281, 0.0), (0.0, 0.03862, 0.0), 0.03707, "wrist_1 8/8"),
    (4, (0.0, 0.0, -0.05629), (0.0, 0.0, 0.00163), 0.04055, "wrist_1 (housing) 1/2"),
    (4, (0.0, 0.0, -0.00799), (0.0, 0.0, 0.05359), 0.03907, "wrist_1 (housing) 2/2"),
)

def dh_link_frames(joint_angles):
    """The base frame and the six DH link frames (7 4x4 transforms)."""
    T = np.eye(4)
    frames = [T.copy()]
    for i, (theta, d, a, alpha) in enumerate(DH_PARAMS):
        T = T @ dh_transform(joint_angles[i] + theta, d, a, alpha)
        frames.append(T.copy())
    return frames


def physical_link_capsules(joint_angles, frames=None):
    """[(a, b, radius, label)] of PHYSICAL_LINK_CAPSULES in the base frame (frames: dh_link_frames, if known)."""
    frames = dh_link_frames(joint_angles) if frames is None else frames
    out = []
    for k, a, b, r, label in PHYSICAL_LINK_CAPSULES:
        F = frames[k]
        out.append((F[:3, :3] @ np.asarray(a) + F[:3, 3], F[:3, :3] @ np.asarray(b) + F[:3, 3], r, label))
    return out


# The rest of the physical arm and the gripper, for the physical self-collision check (review r2 P1-1): the twin's
# wrist_2 / wrist_3 capsules and cylinder, the quick changer and the RG2 body (fmb_cell.xml), in this DH chain's
# frames (fixed to < 1.3 mm).  Cylinders are kept as capsules (a superset).  (DH frame, end a, end b, radius m, label)
PHYSICAL_WRIST_CAPSULES = (
    (5, (0.0, -0.03908, 0.0), (0.0, -0.02734, 0.0), 0.03793, "wrist_2 1/8"),
    (5, (0.0, -0.02955, 0.0), (0.0, -0.01636, 0.0), 0.03955, "wrist_2 2/8"),
    (5, (0.0, -0.01983, 0.0), (0.0, -0.01341, 0.0), 0.03903, "wrist_2 3/8"),
    (5, (0.0, 0.00112, 0.0), (0.0, 0.01695, 0.0), 0.03915, "wrist_2 5/8"),
    (5, (0.0, 0.00815, 0.0), (0.0, 0.03159, 0.0), 0.03944, "wrist_2 6/8"),
    (5, (0.0, 0.01436, 0.0), (0.0, 0.04611, 0.0), 0.03942, "wrist_2 7/8"),
    (5, (0.0, 0.02645, 0.0), (0.0, 0.04611, 0.0), 0.03986, "wrist_2 8/8"),
    (5, (0.0, 0.0, -0.05673), (0.0, 0.0, 0.0032), 0.04011, "wrist_2 (housing) 1/2"),
    (5, (0.0, 0.0, -0.00884), (0.0, 0.0, 0.05315), 0.03894, "wrist_2 (housing) 2/2"),
    (6, (0.0, 0.0, -0.04575), (0.0, 0.0, -0.01726), 0.04456, "wrist_3 1/2"),
    (6, (0.0, 0.0, -0.02738), (0.0, 0.0, 4e-05), 0.0382, "wrist_3 2/2"),
    (6, (0.0, 0.0, 0.0005), (0.0, 0.0, 0.0185), 0.035, "quick changer"),      # the gripper's, not UR's (twin)
)
# The RG2 body: a box in the flange frame (centre, half sizes along the flange x, y, z), measured exactly.
RG2_BODY_BOX = ((0.0, 0.0003, 0.0786), (0.035, 0.035, 0.065))
# The UR5e base: a cylinder about the base z axis, (radius m, top z m) - the bounds of ur_description's
# meshes/ur5e/collision/base.stl (r 75.6 mm, z 0..99.1 mm).  The DH base capsule (J0-J1, r 75 mm) reaches z 237 mm
# with its cap, above any physical base; it stays in the DH check only.
UR5E_BASE_CYLINDER = (0.0761, 0.0996)   # + 0.5 mm: encloses the visual base mesh too (tests)
PHYSICAL_SELF_MARGIN_M = 0.010
_RG2_SURFACE = []


def _rg2_surface(spacing=0.005):
    """The RG2 body's faces sampled on a grid of at most `spacing` (flange frame), and the sampling gap."""
    if not _RG2_SURFACE:
        centre, half = np.asarray(RG2_BODY_BOX[0]), np.asarray(RG2_BODY_BOX[1])
        lo, hi = centre - half, centre + half
        n = [max(1, int(np.ceil((hi[k] - lo[k]) / spacing))) for k in range(3)]
        grids = [np.linspace(lo[k], hi[k], n[k] + 1) for k in range(3)]
        steps = [(hi[k] - lo[k]) / n[k] for k in range(3)]
        pts, gap = [], 0.0
        for k in range(3):
            u, v = [j for j in range(3) if j != k]
            gap = max(gap, 0.5 * float(np.hypot(steps[u], steps[v])))
            U, V = np.meshgrid(grids[u], grids[v], indexing="ij")
            for w in (lo[k], hi[k]):
                face = np.empty((U.size, 3))
                face[:, k], face[:, u], face[:, v] = w, U.ravel(), V.ravel()
                pts.append(face)
        _RG2_SURFACE.extend([np.concatenate(pts), gap])
    return _RG2_SURFACE[0], _RG2_SURFACE[1]
# Kinematic order of each body (pairs less than two apart are not checked: they share a joint).  A joint housing
# (a capsule along DH frame k's z axis, the axis of joint k+1) joins links k and k+1, so it ranks k + 0.5.
_SELF_ORDER = {"shoulder (pan housing)": 0.5, "shoulder (lift housing)": 1.5,
               "upper arm (shoulder end)": 1.5, "upper arm": 2, "upper arm (elbow end)": 2.5,
               "forearm (elbow end)": 2.5, "forearm": 3, "forearm (wrist_1 end)": 3.5,
               "wrist_1": 4, "wrist_1 (housing)": 4.5, "wrist_2": 5, "wrist_2 (housing)": 5.5,
               "wrist_3": 6, "quick changer": 6, "RG2 body": 6}


def _order(label):
    """A body's kinematic order; a tube's bin capsules ("forearm 2/4") rank with their tube."""
    return _SELF_ORDER[re.sub(r" \d+/\d+$", "", label)]


def _segment_convex_distance(A, B, point_distance):
    """Exact distance from segments A[i]-B[i] to a convex body given by point_distance(P (..., 3)) -> (...).  The
    distance to a convex set along a line is convex in the segment parameter, so the minimum lies within one step of
    the best sample of a grid: a 65-point grid, then three 33-point refinements around the best (step 1/262144 of
    the segment), vectorised over segments."""
    D = B - A

    def dist(T):                                        # T: (n, k) parameters -> (n, k) distances
        return point_distance(A[:, None, :] + T[:, :, None] * D[:, None, :])

    n = len(A)
    T = np.tile(np.linspace(0.0, 1.0, 65), (n, 1))
    step = 1.0 / 64
    for _ in range(3):
        d = dist(T)
        k = np.argmin(d, axis=1)
        best_t, best_d = T[np.arange(n), k], d[np.arange(n), k]
        T = np.clip(best_t[:, None] + np.linspace(-step, step, 33)[None, :], 0.0, 1.0)
        step /= 16
    return np.minimum(best_d, dist(T).min(axis=1))


def _segment_box_distance(A, B, centre, half):
    """Exact distance from segments A[i]-B[i] (box frame) to an axis-aligned box."""
    return _segment_convex_distance(
        A, B, lambda P: np.linalg.norm(np.maximum(np.abs(P - centre) - half, 0.0), axis=-1))


def base_cylinder_distance(P):
    """Distance (m) from points P (base frame) to the UR5e base cylinder UR5E_BASE_CYLINDER (0 inside)."""
    P = np.asarray(P, float)
    radius, top = UR5E_BASE_CYLINDER
    radial = np.maximum(np.hypot(P[..., 0], P[..., 1]) - radius, 0.0)
    vertical = np.maximum(np.maximum(-P[..., 2], P[..., 2] - top), 0.0)
    return np.hypot(radial, vertical)


def _segment_base_distance(A, B):
    """Exact distance from segments (base frame) to the base cylinder UR5E_BASE_CYLINDER."""
    radius, top = UR5E_BASE_CYLINDER

    def point_distance(P):
        radial = np.maximum(np.hypot(P[..., 0], P[..., 1]) - radius, 0.0)
        vertical = np.maximum(np.maximum(-P[..., 2], P[..., 2] - top), 0.0)
        return np.hypot(radial, vertical)
    return _segment_convex_distance(A, B, point_distance)


def _segments_distance(P1, Q1, P2, Q2):
    """Exact closest distances between segments P1[i]-Q1[i] and P2[i]-Q2[i] (Ericson, Real-Time Collision
    Detection 5.1.9), vectorised."""
    d1, d2, r = Q1 - P1, Q2 - P2, P1 - P2
    a, e = np.einsum("ij,ij->i", d1, d1), np.einsum("ij,ij->i", d2, d2)
    f, c, b = np.einsum("ij,ij->i", d2, r), np.einsum("ij,ij->i", d1, r), np.einsum("ij,ij->i", d1, d2)
    a, e = np.maximum(a, 1e-18), np.maximum(e, 1e-18)
    denom = a * e - b * b
    s = np.where(denom > 1e-18, np.clip((b * f - c * e) / np.where(denom > 1e-18, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm((P1 + d1 * s[:, None]) - (P2 + d2 * t[:, None]), axis=1)


def _physical_bodies(joint_angles):
    frames = dh_link_frames(joint_angles)
    caps = [(a, b, r, label) for a, b, r, label in physical_link_capsules(joint_angles, frames)]
    for k, a, b, r, label in PHYSICAL_WRIST_CAPSULES:
        F = frames[k]
        caps.append((F[:3, :3] @ np.asarray(a) + F[:3, 3], F[:3, :3] @ np.asarray(b) + F[:3, 3], r, label))
    return frames, caps


_PAIRS_CACHE = {}


def physical_self_clearances(joint_angles):
    """[(clearance m, body a, body b)] for every non-adjacent pair of the physical arm and gripper: the base
    cylinder, the physical link capsules, the wrist capsules, the quick changer and the RG2 body."""
    frames, caps = _physical_bodies(joint_angles)
    labels = tuple(c[3] for c in caps)
    if labels not in _PAIRS_CACHE:
        ii, jj = zip(*[(i, j) for i in range(len(caps)) for j in range(i + 1, len(caps))
                       if abs(_order(labels[i]) - _order(labels[j])) >= 2])
        far = [i for i, l in enumerate(labels) if _order(l) <= 4]
        _PAIRS_CACHE[labels] = (np.array(ii), np.array(jj), np.array(far))
    ii, jj, far = _PAIRS_CACHE[labels]
    A = np.array([c[0] for c in caps])
    B = np.array([c[1] for c in caps])
    R = np.array([c[2] for c in caps])
    d = _segments_distance(A[ii], B[ii], A[jj], B[jj]) - R[ii] - R[jj]
    out = [(float(x), labels[i], labels[j]) for x, i, j in zip(d, ii, jj)]
    # the RG2 body against everything two or more links away from the flange
    R6, p6 = frames[6][:3, :3], frames[6][:3, 3]
    db = _segment_box_distance((A[far] - p6) @ R6, (B[far] - p6) @ R6,
                               np.asarray(RG2_BODY_BOX[0]), np.asarray(RG2_BODY_BOX[1])) - R[far]
    out += [(float(x), labels[i], "RG2 body") for x, i in zip(db, far)]
    # the base cylinder against everything two or more links above it (the upper arm cannot reach it: its tube
    # runs 138 mm out along the shoulder axis, outside the 75.6 mm base)
    above = np.array([i for i, l in enumerate(labels) if _order(l) >= 2.5])
    dbase = _segment_base_distance(A[above], B[above]) - R[above]
    out += [(float(x), "base", labels[i]) for x, i in zip(dbase, above)]
    # ... and against the RG2 body: its faces sampled at <= 5 mm, less the sampling gap (a lower bound)
    surface, gap = _rg2_surface()
    radius, top = UR5E_BASE_CYLINDER
    W = surface @ R6.T + p6
    radial = np.maximum(np.hypot(W[:, 0], W[:, 1]) - radius, 0.0)
    vertical = np.maximum(np.maximum(-W[:, 2], W[:, 2] - top), 0.0)
    out.append((float(np.min(np.hypot(radial, vertical))) - gap, "base", "RG2 body"))
    return out


_PHYS_CACHE = {}


def physical_self_collision_refusal(joint_angles, margin=PHYSICAL_SELF_MARGIN_M):
    """Memoized per joint vector and margin (N-1, w5b-tier2); the key includes the tier-2 settings and the mount
    check (tests swap them)."""
    from primitives.shared import arm_surface
    from primitives.shared import camera_up
    key = (tuple(float(x) for x in np.asarray(joint_angles, float)), float(margin), id(arm_surface.pair_lower_bound),
           id(arm_surface.rg2_housing), id(arm_surface.mount_lower_bound), id(camera_up.camera_mount_arm_refusal),
           id(arm_surface.body), arm_surface.state_token())
    if key not in _PHYS_CACHE:
        if len(_PHYS_CACHE) > 20000:
            _PHYS_CACHE.clear()
        _PHYS_CACHE[key] = _physical_self_collision_refusal(joint_angles, margin)
    return _PHYS_CACHE[key]


def _physical_self_collision_refusal(joint_angles, margin=PHYSICAL_SELF_MARGIN_M):
    """None, or a typed reason: two physical bodies of the arm and gripper closer than the margin, or the camera
    mount/D415 within its margin of the arm (camera_up.camera_mount_arm_refusal).

    Two tiers (review w5-enclose r1): the enclosing capsules first; a pair they refuse is admitted only when the
    lower bound on its real surfaces' distance (arm_surface.pair_lower_bound: UR's meshes, allowances included)
    clears the same margin.  Tier 2 unavailable (meshes missing or not the pinned ones): the refusal stands."""
    from primitives.shared import arm_surface
    q = np.asarray(joint_angles, float)
    tier1 = sorted(c for c in physical_self_clearances(q) if c[0] < margin)
    if tier1:
        frames, caps = _physical_bodies(q)
        capsule_of = {label: (a, b, r) for a, b, r, label in caps}
        for d1, la, lb in tier1:
            d2 = arm_surface.pair_lower_bound(la, lb, frames, capsule_of, margin)
            if d2 is None:
                return (f"PHYSICAL_SELF_COLLISION: {la} vs {lb} {1000 * d1:.1f} mm apart (needs {1000 * margin:.0f}; "
                        f"real-surface check unavailable: {arm_surface.why_unavailable()})")
            if d2 < margin:
                if getattr(d2, "early", False):                   # decided before the exact pass (N-1)
                    return (f"PHYSICAL_SELF_COLLISION: {la} vs {lb}: real surfaces < {1000 * margin:.0f} mm apart "
                            f"(a surface sample pair at {1000 * d2:.1f} mm after allowances; capsules {1000 * d1:.1f})")
                return (f"PHYSICAL_SELF_COLLISION: {la} vs {lb}: real surfaces >= {1000 * d2:.1f} mm apart "
                        f"(capsules {1000 * d1:.1f}; needs {1000 * margin:.0f})")
    from primitives.shared.camera_up import camera_mount_arm_refusal
    return camera_mount_arm_refusal(q)


def arm_table_refusal(joint_angles, margin=PHYSICAL_SELF_MARGIN_M):
    """None, or a typed reason: the arm's real surface (shoulder to wrist_3) within the margin of the table plane.

    Tier 1: each enclosing capsule's lowest point (its lower end's z less its radius).  Tier 2 for a capsule tier 1
    refuses: the lowest vertex of UR's mesh pieces in it, exact, less the allowance (arm_surface.table_lower_bound).
    The base stands on the table and is not checked; the gripper is the grasp's own business."""
    from primitives.shared import arm_surface
    q = np.asarray(joint_angles, float)
    frames = dh_link_frames(q)
    caps = list(physical_link_capsules(q, frames))
    for k, a, b, r, label in PHYSICAL_WRIST_CAPSULES:
        if label != "quick changer":
            F = frames[k]
            caps.append((F[:3, :3] @ np.asarray(a) + F[:3, 3], F[:3, :3] @ np.asarray(b) + F[:3, 3], r, label))
    for a, b, r, label in sorted(caps, key=lambda c: min(c[0][2], c[1][2]) - c[2]):
        d1 = min(float(a[2]), float(b[2])) - r - TABLE_HEIGHT
        if d1 >= margin:
            break
        d2 = arm_surface.table_lower_bound(label, frames, TABLE_HEIGHT)
        if d2 is None:
            return (f"ARM_TABLE_COLLISION: {label} capsule {1000 * d1:.1f} mm above the table (needs "
                    f"{1000 * margin:.0f}; real-surface check unavailable: {arm_surface.why_unavailable()})")
        if d2 < margin:
            return (f"ARM_TABLE_COLLISION: {label} real surface {1000 * d2:.1f} mm above the table "
                    f"(capsule {1000 * d1:.1f}; needs {1000 * margin:.0f})")
    return None


def compute_all_joint_positions(joint_angles):
    """
    Compute the 3D positions of all joints given joint angles.

    Returns a list of 7 [x, y, z] positions (base + 6 joints).
    """
    joint_positions = []
    T = np.eye(4)

    # Base position (always at origin)
    joint_positions.append(T[:3, 3].copy())

    # Compute position of each joint
    for i, (theta, d, a, alpha) in enumerate(DH_PARAMS):
        T_i = dh_transform(joint_angles[i] + theta, d, a, alpha)
        T = np.dot(T, T_i)
        joint_positions.append(T[:3, 3].copy())

    return joint_positions


def check_collision_with_table(joint_angles, z_threshold=TABLE_HEIGHT, verbose=False, logger=None):
    """
    Check if any part of the robot goes below the table: the DH joint origins against z_threshold, and the arm's
    enclosed real surface against the table plane (arm_table_refusal, review w5-enclose r1 P2-1).

    Args:
        joint_angles: Array of 6 joint angles
        z_threshold: Minimum allowed Z position (meters)
        verbose: If True, log which joint caused collision
        logger: ROS logger instance for verbose output

    Returns:
        True if collision detected (any joint below threshold, or the arm's surface within 10 mm of the table:
        arm_table_refusal), False otherwise
    """
    table = arm_table_refusal(joint_angles)
    if table:
        if verbose and logger:
            logger.warn(f"Collision detected: {table}")
        return True
    joint_positions = compute_all_joint_positions(joint_angles)

    for i, pos in enumerate(joint_positions):
        if pos[2] < z_threshold:
            if verbose and logger:
                logger.warn(
                    f"Collision detected: Joint {i} at Z={pos[2]*1000:.1f}mm "
                    f"(threshold: {z_threshold*1000:.1f}mm)"
                )
            return True

    return False


def segment_distance(p1, p2, p3, p4):
    """
    Compute minimum distance between two line segments (p1-p2) and (p3-p4).
    Used for capsule-based collision detection between robot links.

    Args:
        p1, p2: Start and end points of first segment (numpy arrays)
        p3, p4: Start and end points of second segment (numpy arrays)

    Returns:
        Minimum distance between the two segments
    """
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3

    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    EPSILON = 1e-8

    if a < EPSILON and e < EPSILON:
        return np.linalg.norm(p1 - p3)

    if a < EPSILON:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e < EPSILON:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b

            if abs(denom) > EPSILON:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0

            t = (b * s + f) / e

            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    closest1 = p1 + s * d1
    closest2 = p3 + t * d2

    return np.linalg.norm(closest1 - closest2)


def check_self_collision(joint_angles, verbose=False, logger=None):
    """
    Check if the robot configuration causes self-collision: the physical arm, gripper and camera mount
    (physical_self_collision_refusal), and the DH link capsules between non-adjacent links.

    Args:
        joint_angles: Array of 6 joint angles
        verbose: If True, log collision details
        logger: ROS logger instance for verbose output

    Returns:
        True if self-collision detected, False otherwise
    """
    # Both models (review r2 P1-1): the DH capsules (conservative, arm links only) and the physical arm, gripper
    # and camera mount (physical_self_collision_refusal).  Either refusing is a self-collision.
    physical = physical_self_collision_refusal(joint_angles)
    if physical:
        if verbose and logger:
            logger.warn(f"Self-collision detected: {physical}")
        return True
    return dh_self_collision(joint_angles, verbose=verbose, logger=logger)


def dh_self_collision(joint_angles, verbose=False, logger=None):
    """The DH capsule self-collision check alone (UR5E_LINK_RADII between non-adjacent DH segments, 10 mm margin)."""
    link_radii = UR5E_LINK_RADII

    safety_margin = 0.01  # 1cm extra margin

    joint_positions = compute_all_joint_positions(joint_angles)

    num_links = len(joint_positions) - 1

    for i in range(num_links):
        for j in range(i + 2, num_links):  # Skip adjacent links
            p1 = np.array(joint_positions[i])
            p2 = np.array(joint_positions[i + 1])
            p3 = np.array(joint_positions[j])
            p4 = np.array(joint_positions[j + 1])

            dist = segment_distance(p1, p2, p3, p4)
            min_dist = link_radii[i] + link_radii[j] + safety_margin

            if dist < min_dist:
                if verbose and logger:
                    logger.warn(
                        f"Self-collision detected: Link {i} and Link {j} "
                        f"distance={dist*1000:.1f}mm < threshold={min_dist*1000:.1f}mm"
                    )
                return True

    return False


def check_ee_below_base(joint_angles, z_threshold=0.1625, verbose=False, logger=None):
    """
    Check if the end-effector goes below the robot base height.

    Args:
        joint_angles: Array of 6 joint angles
        z_threshold: Minimum allowed EE Z position (meters).
                    Default 0.1625 is the robot base height (first DH d parameter).
        verbose: If True, log details
        logger: ROS logger instance for verbose output

    Returns:
        True if EE is below threshold, False otherwise
    """
    joint_positions = compute_all_joint_positions(joint_angles)
    ee_pos = joint_positions[-1]

    if ee_pos[2] < z_threshold:
        if verbose and logger:
            logger.warn(
                f"EE below base: Z={ee_pos[2]*1000:.1f}mm < threshold={z_threshold*1000:.1f}mm"
            )
        return True

    return False


def check_compact_configuration(joint_angles, min_wrist_shoulder_xy=0.20, verbose=False, logger=None):
    """
    Check if the robot configuration is too compact (wrist too close to shoulder).

    Detects problematic configurations where the arm is folded back on itself,
    causing the wrist to be physically close to the shoulder/base area.

    Args:
        joint_angles: Array of 6 joint angles
        min_wrist_shoulder_xy: Minimum allowed XY distance (meters) between
                               wrist2 and shoulder. Default 0.20m (200mm).
        verbose: If True, log details
        logger: ROS logger instance for verbose output

    Returns:
        True if configuration is too compact (should be rejected), False otherwise
    """
    joint_positions = compute_all_joint_positions(joint_angles)

    shoulder_pos = np.array(joint_positions[1])
    wrist2_pos = np.array(joint_positions[5])

    xy_dist = np.linalg.norm(wrist2_pos[:2] - shoulder_pos[:2])

    if xy_dist < min_wrist_shoulder_xy:
        if verbose and logger:
            logger.warn(
                f"Compact configuration detected: wrist-shoulder XY distance="
                f"{xy_dist*1000:.1f}mm < threshold={min_wrist_shoulder_xy*1000:.1f}mm"
            )
        return True

    return False


def check_trajectory_collision(start_joints, target_joints, z_threshold=TABLE_HEIGHT, num_samples=20, logger=None):
    """
    Check if any point along a linearly interpolated trajectory has a collision.

    Args:
        start_joints: Starting joint configuration
        target_joints: Target joint configuration
        z_threshold: Minimum allowed Z position (meters) for table collision
        num_samples: Number of samples along the trajectory to check
        logger: ROS logger instance for warnings

    Returns:
        True if collision detected along trajectory, False otherwise
    """
    for i in range(num_samples + 1):
        alpha = i / num_samples
        interpolated_joints = start_joints + alpha * (target_joints - start_joints)
        if check_collision_with_table(interpolated_joints, z_threshold=z_threshold):
            if logger:
                logger.warn(f"Trajectory table collision at alpha={alpha:.2f}")
            return True
        if check_self_collision(interpolated_joints):
            if logger:
                logger.warn(f"Trajectory self-collision at alpha={alpha:.2f}")
            return True
    return False
