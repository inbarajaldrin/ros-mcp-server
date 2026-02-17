#!/usr/bin/env python3
"""
Fold Symmetry - Shared utilities for object symmetry handling.

Objects with rotational symmetry (e.g., 2-fold, 4-fold) have multiple orientations
that look identical. This module loads symmetry data from JSON files and generates
all equivalent orientations for use in grasping, assembly verification, etc.

JSON format: Each object has a {name}_symmetry.json file with fold axes and
quaternion rotations per axis. The full symmetry group is the Cartesian product
of per-axis rotations.
"""

import json
import os
import glob

import numpy as np
from scipy.spatial.transform import Rotation as R


def load_symmetry_data(object_name, symmetry_dir):
    """
    Load fold symmetry JSON for an object.

    Tries exact name, then glob pattern.

    Args:
        object_name: Object name (e.g., "fork_yellow")
        symmetry_dir: Directory containing *_symmetry.json files

    Returns:
        Parsed JSON dict, or None if not found
    """
    patterns = [
        os.path.join(symmetry_dir, f"{object_name}_symmetry.json"),
        os.path.join(symmetry_dir, f"{object_name}*_symmetry.json"),
    ]

    for pattern in patterns:
        if '*' in pattern:
            matches = glob.glob(pattern)
            if matches:
                with open(matches[0], 'r') as f:
                    return json.load(f)
        elif os.path.exists(pattern):
            with open(pattern, 'r') as f:
                return json.load(f)
    return None


def get_symmetry_matrices(fold_data):
    """
    Generate the full symmetry group as 3x3 rotation matrices.

    Composes per-axis rotations (X x Y x Z) and deduplicates.
    Always includes identity.

    Args:
        fold_data: Parsed symmetry JSON dict, or None

    Returns:
        List of 3x3 numpy rotation matrices
    """
    if fold_data is None:
        return [np.eye(3)]

    axis_rotations = {'x': [np.eye(3)], 'y': [np.eye(3)], 'z': [np.eye(3)]}

    for axis in ['x', 'y', 'z']:
        if axis not in fold_data.get('fold_axes', {}):
            continue

        axis_data = fold_data['fold_axes'][axis]
        for q_data in axis_data.get('quaternions', []):
            q = np.array([
                q_data['quaternion']['x'],
                q_data['quaternion']['y'],
                q_data['quaternion']['z'],
                q_data['quaternion']['w']
            ])
            q = q / np.linalg.norm(q)
            R_sym = R.from_quat(q).as_matrix()

            # Skip if already present (includes identity)
            is_dup = any(np.allclose(R_sym, existing, atol=1e-6)
                         for existing in axis_rotations[axis])
            if not is_dup:
                axis_rotations[axis].append(R_sym)

    # Cartesian product of per-axis rotations
    combined = []
    seen = set()

    for Rx in axis_rotations['x']:
        for Ry in axis_rotations['y']:
            for Rz in axis_rotations['z']:
                R_combined = Rx @ Ry @ Rz
                key = tuple(R_combined.flatten().round(6))
                if key not in seen:
                    seen.add(key)
                    combined.append(R_combined)

    return combined


def equivalent_orientations(R_target, fold_data):
    """
    Generate all symmetry-equivalent target orientations.

    R_equivalent = R_target @ R_symmetry (symmetry applied in object frame)

    Args:
        R_target: Target orientation as 3x3 rotation matrix (world frame)
        fold_data: Parsed symmetry JSON dict (may be None)

    Returns:
        List of 3x3 rotation matrices
    """
    return [R_target @ R_sym for R_sym in get_symmetry_matrices(fold_data)]


def find_closest_canonical_quaternion(detected_quat, fold_data, threshold=0.15):
    """
    Find the closest canonical quaternion to the detected orientation.

    For each symmetry rotation, applies its inverse to the detected quaternion
    and checks how close the result is to identity (the canonical pose).

    Args:
        detected_quat: Detected object quaternion [x, y, z, w]
        fold_data: Parsed symmetry JSON dict
        threshold: Maximum quaternion distance for a valid match (0-1 scale)

    Returns:
        Tuple of (canonical_quat, symmetry_rotation, distance) or (None, None, distance)
    """
    detected_quat = np.array(detected_quat)
    detected_quat = detected_quat / np.linalg.norm(detected_quat)

    symmetry_quats = [R.from_matrix(m).as_quat() for m in get_symmetry_matrices(fold_data)]

    if not symmetry_quats:
        return detected_quat, np.array([0, 0, 0, 1]), 0.0

    identity = np.array([0, 0, 0, 1])
    best_distance = float('inf')
    best_symmetry = None
    best_canonical = None

    for sym_rot in symmetry_quats:
        # Inverse of unit quaternion = conjugate
        sym_inv = np.array([-sym_rot[0], -sym_rot[1], -sym_rot[2], sym_rot[3]])
        transformed = (R.from_quat(detected_quat) * R.from_quat(sym_inv)).as_quat()
        transformed = transformed / np.linalg.norm(transformed)

        # Distance via dot product (faster than geodesic, same ordering)
        dot = min(1.0, abs(np.dot(transformed, identity)))
        distance = 1.0 - dot

        if distance < best_distance:
            best_distance = distance
            best_symmetry = sym_rot
            best_canonical = transformed

    if best_distance <= threshold:
        return best_canonical, best_symmetry, best_distance
    else:
        return None, None, best_distance
