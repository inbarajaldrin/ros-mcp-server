"""
Collision detection utilities for UR5e robot.

Provides table collision, self-collision, and configuration safety checks
using DH-based forward kinematics and capsule-based link models.
"""

import numpy as np

from primitives.shared.config import DH_PARAMS, TABLE_HEIGHT
from primitives.shared.ik import dh_transform


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
    # UR5e approximate link radii (meters) - conservative estimates
    link_radii = [
        0.075,  # Base (joint 0-1)
        0.065,  # Shoulder to elbow (joint 1-2) - upper arm
        0.055,  # Elbow to wrist1 (joint 2-3) - forearm
        0.045,  # Wrist1 to wrist2 (joint 3-4)
        0.045,  # Wrist2 to wrist3 (joint 4-5)
        0.040,  # Wrist3 to EE (joint 5-6)
    ]

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
