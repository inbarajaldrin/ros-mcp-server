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


def physical_link_capsules(joint_angles):
    """[(a, b, radius, label)] of PHYSICAL_LINK_CAPSULES in the base frame."""
    frames = dh_link_frames(joint_angles)
    out = []
    for k, a, b, r, label in PHYSICAL_LINK_CAPSULES:
        F = frames[k]
        out.append((F[:3, :3] @ np.asarray(a) + F[:3, 3], F[:3, :3] @ np.asarray(b) + F[:3, 3], r, label))
    return out


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
    Check if the robot configuration causes self-collision.
    Models links as capsules and checks distances between non-adjacent links.

    Args:
        joint_angles: Array of 6 joint angles
        verbose: If True, log collision details
        logger: ROS logger instance for verbose output

    Returns:
        True if self-collision detected, False otherwise
    """
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
