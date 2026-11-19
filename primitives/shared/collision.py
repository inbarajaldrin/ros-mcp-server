"""
Collision detection utilities for UR5e robot.

Provides table collision, self-collision, and configuration safety checks
using DH-based forward kinematics and capsule-based link models.
"""

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


# The physical link capsules, for the checks of things the arm carries (held part, camera mount) against the
# arm.  The DH segments above are not on the physical links: the upper-arm tube runs 0.138 m off its DH segment
# along the elbow axis, and there the DH capsule overstated the clearance by up to 128 mm (fmb-cell-lab
# mujoco/capsule_model_check.py, review P2-2).  These are the twin's UR5e collision capsules (fmb_cell.xml, the
# MuJoCo Menagerie geometry), expressed in this DH chain's link frames, where they are fixed to < 1 mm:
# (DH frame index, end a, end b, radius m, arm link label).  The checks use BOTH models; neither replaces the other.
PHYSICAL_LINK_CAPSULES = (
    (1, (0.0002, -0.1002, 0.0), (0.0002, 0.0198, 0.0), 0.060, "shoulder"),
    (2, (0.4251, 0.0, 0.1584), (0.4251, 0.0, 0.0384), 0.060, "upper arm (shoulder end)"),
    (2, (0.4251, 0.0, 0.1384), (0.0251, 0.0, 0.1384), 0.050, "upper arm"),
    (3, (0.3924, 0.0, 0.1477), (0.3924, 0.0, 0.0277), 0.055, "forearm (elbow end)"),
    (3, (0.3824, 0.0, 0.0077), (0.0024, 0.0, 0.0079), 0.038, "forearm"),
    (4, (0.0, -0.005, 0.0), (0.0, -0.145, 0.0), 0.040, "wrist_1"),
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
    (5, (0.0, -0.0004, 0.0), (0.0, 0.1196, 0.001), 0.040, "wrist_2"),
    (5, (0.0, 0.0, 0.0605), (0.0, 0.0, -0.0195), 0.040, "wrist_2 (joint)"),
    (6, (0.0, 0.0, -0.0391), (0.0, 0.0, 0.0009), 0.040, "wrist_3"),
    (6, (0.0, 0.0, 0.0005), (0.0, 0.0, 0.0185), 0.035, "quick changer"),
)
# The RG2 body: a box in the flange frame (centre, half sizes along the flange x, y, z), measured exactly.
RG2_BODY_BOX = ((0.0, 0.0003, 0.0786), (0.035, 0.035, 0.065))
# The UR5e base: a cylinder about the base z axis, (radius m, top z m) - the bounds of ur_description's
# meshes/ur5e/collision/base.stl (r 75.6 mm, z 0..99.1 mm).  The DH base capsule (J0-J1, r 75 mm) reaches z 237 mm
# with its cap, above any physical base; it stays in the DH check only.
UR5E_BASE_CYLINDER = (0.0756, 0.0991)
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
# Kinematic order of each body; pairs closer than two links apart are not checked (they share a joint).
_SELF_ORDER = {"shoulder": 1, "upper arm (shoulder end)": 2, "upper arm": 2, "forearm (elbow end)": 3,
               "forearm": 3, "wrist_1": 4, "wrist_2": 5, "wrist_2 (joint)": 5, "wrist_3": 6, "quick changer": 6,
               "RG2 body": 6}


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
                       if abs(_SELF_ORDER[labels[i]] - _SELF_ORDER[labels[j]]) >= 2])
        far = [i for i, l in enumerate(labels) if _SELF_ORDER[l] <= 4]
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
    above = np.array([i for i, l in enumerate(labels) if _SELF_ORDER[l] >= 3])
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


def physical_self_collision_refusal(joint_angles, margin=PHYSICAL_SELF_MARGIN_M):
    """None, or a typed reason: two physical bodies of the arm and gripper closer than the margin, or the camera
    mount/D415 within its margin of the arm (camera_up.camera_mount_arm_refusal)."""
    worst = min(physical_self_clearances(joint_angles))
    if worst[0] < margin:
        return (f"PHYSICAL_SELF_COLLISION: {worst[1]} vs {worst[2]} {1000 * worst[0]:.1f} mm apart "
                f"(needs {1000 * margin:.0f})")
    from primitives.shared.camera_up import camera_mount_arm_refusal
    return camera_mount_arm_refusal(joint_angles)


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
    Check if any part of the robot (all joints) goes below the table.

    Args:
        joint_angles: Array of 6 joint angles
        z_threshold: Minimum allowed Z position (meters)
        verbose: If True, log which joint caused collision
        logger: ROS logger instance for verbose output

    Returns:
        True if collision detected (any joint below threshold), False otherwise
    """
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
