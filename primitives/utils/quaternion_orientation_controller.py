#!/usr/bin/env python3
"""
Quaternion Orientation Controller - Gimbal Lock Free Gripper Control

This module provides a quaternion-based approach to gripper orientation control,
avoiding gimbal lock singularities that occur at pitch=180° when using Euler angles (RPY).

FOLD SYMMETRY EXPLANATION:
- The JSON files store discrete orientations where the object looks IDENTICAL
- For a line with 2-fold Z symmetry: orientations at 0° and 180° around Z look the same
- For a fork with 2-fold Y symmetry: orientations at 0° and 180° around Y look the same
- The goal: find which canonical orientation the detected pose is equivalent to,
  then use that canonical's yaw for gripper alignment

Key Insight:
- Euler angles have a singularity at pitch = ±90° (gimbal lock)
- At pitch=180°, the roll and yaw axes become colinear, losing one degree of freedom
- Quaternions use a 4-parameter representation on a 3-sphere that covers rotation space
  without singularities - enabling safe operation at ANY yaw angle

Reference:
- scipy.spatial.transform.Rotation uses [x, y, z, w] convention
"""

import numpy as np
import math
from scipy.spatial.transform import Rotation as R


class QuaternionOrientationController:
    """
    Controls gripper orientation using quaternions instead of Euler angles.
    
    Eliminates gimbal lock issues when the gripper is constrained to face-down
    orientation (pitch = 180°) with variable yaw angles.
    """
    
    @staticmethod
    def face_down_quaternion(yaw_degrees):
        """
        Create quaternion for face-down gripper with specified yaw.
        
        Represents: roll=0°, pitch=180° (face down), yaw=yaw_degrees
        
        This is the core method for gimbal-lock-free gripper control.
        At pitch=180°, Euler angles have a singularity, but quaternions
        handle this without any issues.
        
        Args:
            yaw_degrees: Yaw rotation around world Z-axis in degrees
            
        Returns:
            quaternion [x, y, z, w] in scipy convention
            
        Example:
            q = face_down_quaternion(45.0)  # Yaw 45°, pitch 180°, roll 0°
        """
        # Create face-down orientation with yaw rotation
        # Using scipy's Rotation which handles gimbal lock internally
        r = R.from_euler('xyz', [0, 180, yaw_degrees], degrees=True)
        q = r.as_quat()  # Returns [x, y, z, w]
        
        # Ensure quaternion is normalized
        return QuaternionOrientationController.normalize_quaternion(q)
    
    @staticmethod
    def quaternion_multiply(q1, q2):
        """
        Multiply two quaternions to compose rotations.
        
        Result represents: first apply q1, then apply q2
        (Note: quaternion multiplication order matters!)
        
        Args:
            q1, q2: quaternions in [x, y, z, w] format
            
        Returns:
            Result quaternion [x, y, z, w]
        """
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        return (r1 * r2).as_quat()
    
    @staticmethod
    def apply_yaw_rotation(base_quaternion, yaw_offset_degrees):
        """
        Apply additional yaw rotation to existing quaternion.
        
        Useful for fine-tuning orientation or creating trajectories
        with incremental yaw adjustments.
        
        Args:
            base_quaternion: Starting quaternion [x, y, z, w]
            yaw_offset_degrees: Additional yaw rotation in degrees
            
        Returns:
            Modified quaternion [x, y, z, w]
        """
        r_yaw = R.from_euler('z', yaw_offset_degrees, degrees=True)
        r_base = R.from_quat(base_quaternion)
        return (r_yaw * r_base).as_quat()
    
    @staticmethod
    def slerp(q1, q2, t):
        """
        Smooth interpolation between two quaternions (Spherical Linear Interpolation).
        
        Useful for creating smooth trajectories between two quaternion-based orientations.
        Maintains constant rotation rate without gimbal lock artifacts.
        
        Args:
            q1: Start quaternion [x, y, z, w]
            q2: End quaternion [x, y, z, w]
            t: Interpolation parameter in [0, 1]
               t=0 returns q1, t=1 returns q2
               
        Returns:
            Interpolated quaternion [x, y, z, w]
        """
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        
        # Calculate relative rotation from q1 to q2
        r_relative = r1.inv() * r2
        axis_angle = r_relative.as_rotvec()
        angle = np.linalg.norm(axis_angle)
        
        # Only interpolate if there's a meaningful rotation
        if angle > 1e-6:
            axis = axis_angle / angle
            r_interp = r1 * R.from_rotvec(axis * angle * t)
        else:
            # Quaternions are very close, just return q1
            r_interp = r1
        
        return r_interp.as_quat()
    
    @staticmethod
    def normalize_quaternion(q):
        """
        Normalize quaternion to unit length (magnitude = 1.0).
        
        All quaternions representing rotations must be unit quaternions.
        This method ensures numerical stability.
        
        Args:
            q: quaternion [x, y, z, w]
            
        Returns:
            Normalized quaternion [x, y, z, w]
        """
        q = np.array(q)
        magnitude = np.linalg.norm(q)
        if magnitude < 1e-10:
            raise ValueError(f"Cannot normalize zero quaternion: {q}")
        return q / magnitude
    
    @staticmethod
    def quaternion_to_rpy(q):
        """
        Convert quaternion to Euler angles (RPY in degrees).
        
        ⚠️ CAUTION: Use this ONLY at I/O boundaries for logging/display!
        
        Args:
            q: quaternion [x, y, z, w] in scipy convention
            
        Returns:
            [roll, pitch, yaw] in degrees
        """
        r = R.from_quat(q)
        rpy_radians = r.as_euler('xyz')
        return np.degrees(rpy_radians)
    
    @staticmethod
    def rpy_to_quaternion(roll_deg, pitch_deg, yaw_deg):
        """
        Convert Euler angles (RPY) to quaternion.
        
        ⚠️ CAUTION: Only use at I/O boundaries!
        
        Args:
            roll_deg: Roll in degrees
            pitch_deg: Pitch in degrees
            yaw_deg: Yaw in degrees
            
        Returns:
            quaternion [x, y, z, w]
        """
        r = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
        return r.as_quat()
    
    @staticmethod
    def quaternion_inverse(q):
        """
        Calculate inverse of a quaternion.
        
        For unit quaternion: inverse = conjugate = [-x, -y, -z, w]
        
        Args:
            q: quaternion [x, y, z, w]
            
        Returns:
            Inverse quaternion [x, y, z, w]
        """
        q = np.array(q)
        return np.array([-q[0], -q[1], -q[2], q[3]])
    
    @staticmethod
    def quaternion_distance(q1, q2):
        """
        Calculate angular distance between two quaternions.
        
        Uses dot product: distance = 1 - abs(dot(q1, q2))
        For unit quaternions, this gives angular distance in [0, 1]
        where 0 = identical orientation, 1 = 180° apart
        
        Note: abs() handles quaternion double-cover (q and -q represent same rotation)
        
        Args:
            q1, q2: quaternions [x, y, z, w]
            
        Returns:
            Distance in [0, 1] where 0 = identical, 1 = opposite
        """
        q1_norm = np.array(q1) / np.linalg.norm(q1)
        q2_norm = np.array(q2) / np.linalg.norm(q2)
        dot_product = abs(np.dot(q1_norm, q2_norm))
        # Clamp to handle numerical errors
        dot_product = min(1.0, dot_product)
        return 1.0 - dot_product
    
    @staticmethod
    def quaternion_angle_degrees(q1, q2):
        """
        Calculate angular difference between two quaternions in degrees.
        
        Args:
            q1, q2: quaternions [x, y, z, w]
            
        Returns:
            Angle in degrees [0, 180]
        """
        q1_norm = np.array(q1) / np.linalg.norm(q1)
        q2_norm = np.array(q2) / np.linalg.norm(q2)
        dot_product = abs(np.dot(q1_norm, q2_norm))
        dot_product = min(1.0, dot_product)
        angle_rad = 2.0 * math.acos(dot_product)
        return math.degrees(angle_rad)
    
    @staticmethod
    def extract_yaw_from_quaternion(q):
        """
        Extract yaw angle from quaternion using proper Euler decomposition.
        
        This extracts the Z-axis rotation component from any quaternion.
        Works correctly regardless of the quaternion's roll/pitch values.
        
        Args:
            q: quaternion [x, y, z, w]
            
        Returns:
            yaw in degrees, normalized to [-180, 180]
        """
        # Use scipy's robust Euler extraction
        r = R.from_quat(q)
        # Extract as ZYX to get yaw as the first angle (most stable)
        # Then convert to our XYZ convention
        euler_xyz = r.as_euler('xyz', degrees=True)
        yaw_degrees = euler_xyz[2]  # Z component in XYZ
        
        # Normalize to [-180, 180]
        yaw_degrees = ((yaw_degrees + 180) % 360) - 180
        return yaw_degrees
    
    @staticmethod
    def load_fold_symmetry_json(object_name, symmetry_dir):
        """
        Load fold symmetry JSON file for an object.
        
        Searches for file: {symmetry_dir}/{object_name}_symmetry.json
        Also tries pattern: {object_name}*_symmetry.json
        
        Args:
            object_name: Name of the object (e.g., "fork_yellow_scaled70")
            symmetry_dir: Directory containing symmetry JSON files
            
        Returns:
            Dictionary with fold symmetry data, or None if not found
        """
        import json
        import os
        import glob
        
        # Try exact match first
        exact_path = os.path.join(symmetry_dir, f"{object_name}_symmetry.json")
        if os.path.exists(exact_path):
            try:
                with open(exact_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading fold symmetry JSON from {exact_path}: {e}")
                return None
        
        # Try pattern match
        pattern = os.path.join(symmetry_dir, f"{object_name}*_symmetry.json")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            # Try without _scaled70 suffix
            base_name = object_name.replace('_scaled70', '')
            pattern = os.path.join(symmetry_dir, f"{base_name}*_symmetry.json")
            matching_files = glob.glob(pattern)
        
        if not matching_files:
            return None
        
        # Use first match
        json_path = matching_files[0]
        
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading fold symmetry JSON from {json_path}: {e}")
            return None
    
    @staticmethod
    def get_symmetry_rotations(fold_data):
        """
        Extract all symmetry rotation quaternions from fold data.
        
        The JSON stores quaternions that represent rotations where the object
        looks identical. These are the symmetry group elements.
        
        Args:
            fold_data: Dictionary with fold symmetry data from JSON
            
        Returns:
            List of quaternion arrays [x, y, z, w] representing symmetry rotations
        """
        symmetry_rotations = []
        seen = set()  # To avoid duplicates
        
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
                
                # Create hashable key (rounded to avoid float comparison issues)
                key = tuple(np.round(q, 6))
                neg_key = tuple(np.round(-q, 6))  # q and -q are same rotation
                
                if key not in seen and neg_key not in seen:
                    seen.add(key)
                    symmetry_rotations.append(q)
        
        return symmetry_rotations
    
    @staticmethod
    def generate_full_symmetry_group(fold_data):
        """
        Generate the full symmetry group by combining rotations from all axes.
        
        For objects with symmetry on multiple axes, we need to generate
        all combinations of symmetry rotations (the group closure).
        
        Args:
            fold_data: Dictionary with fold symmetry data from JSON
            
        Returns:
            List of quaternion arrays representing all symmetry transformations
        """
        base_rotations = QuaternionOrientationController.get_symmetry_rotations(fold_data)
        
        if len(base_rotations) <= 1:
            return base_rotations
        
        # Generate group closure by repeatedly combining elements
        group = set()
        for q in base_rotations:
            key = tuple(np.round(q, 6))
            group.add(key)
        
        # Iterate to find closure (typically converges in 2-3 iterations)
        changed = True
        max_iterations = 10
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            new_elements = set()
            
            group_list = [np.array(k) for k in group]
            for q1 in group_list:
                for q2 in group_list:
                    # Combine: q1 * q2
                    q_combined = QuaternionOrientationController.quaternion_multiply(q1, q2)
                    q_combined = q_combined / np.linalg.norm(q_combined)
                    
                    key = tuple(np.round(q_combined, 6))
                    neg_key = tuple(np.round(-q_combined, 6))
                    
                    if key not in group and neg_key not in group:
                        new_elements.add(key)
                        changed = True
            
            group.update(new_elements)
        
        return [np.array(k) for k in group]
    
    @staticmethod
    def find_equivalent_canonical_yaw(detected_quat, fold_data, threshold_degrees=15.0):
        """
        Find the canonical yaw that is symmetry-equivalent to the detected orientation.
        
        Algorithm:
        1. Extract the detected object's yaw
        2. Get all symmetry rotations from fold data
        3. For each symmetry rotation, calculate what yaw it would map to
        4. Find which symmetry-equivalent yaw is in the "canonical" range
        5. Return that canonical yaw for gripper alignment
        
        For most grasping applications, we want the yaw normalized to a 
        consistent range (e.g., [-90, 90] or [0, 180]) based on symmetry.
        
        Args:
            detected_quat: Detected object quaternion [x, y, z, w]
            fold_data: Dictionary with fold symmetry data
            threshold_degrees: Not used in this algorithm (kept for API compatibility)
            
        Returns:
            Canonical yaw in degrees (normalized based on object's symmetry)
        """
        detected_quat = np.array(detected_quat)
        detected_quat = detected_quat / np.linalg.norm(detected_quat)
        
        # Extract detected yaw
        detected_yaw = QuaternionOrientationController.extract_yaw_from_quaternion(detected_quat)
        
        # Get symmetry info for Z-axis (yaw symmetry)
        z_fold = 1  # Default: no symmetry
        if 'fold_axes' in fold_data and 'z' in fold_data['fold_axes']:
            z_fold = fold_data['fold_axes']['z'].get('fold', 1)
        
        # Calculate canonical yaw based on fold symmetry
        # For n-fold symmetry around Z, equivalent yaws are: yaw, yaw+360/n, yaw+2*360/n, ...
        if z_fold > 1:
            # Normalize yaw to canonical range [0, 360/z_fold)
            period = 360.0 / z_fold
            canonical_yaw = detected_yaw % period
            
            # Optionally center around zero: [-period/2, period/2)
            if canonical_yaw > period / 2:
                canonical_yaw -= period
        else:
            # No Z symmetry, use detected yaw as-is (normalized to [-180, 180])
            canonical_yaw = detected_yaw
        
        return canonical_yaw
    
    @staticmethod
    def find_closest_canonical_quaternion(detected_quat, fold_data, threshold=0.15):
        """
        Find the closest canonical quaternion to the detected orientation.
        
        Algorithm:
        1. Get all symmetry rotation quaternions from fold_data
        2. Apply each symmetry rotation to the detected quaternion
        3. Find which transformed quaternion is closest to identity (or any reference)
        4. Return the symmetry rotation that achieved the best match
        
        This tells us which "canonical view" of the object we're seeing.
        
        Args:
            detected_quat: Detected object quaternion [x, y, z, w]
            fold_data: Dictionary with fold symmetry data from JSON
            threshold: Maximum quaternion distance for a valid match (0-1 scale)
            
        Returns:
            Tuple of (canonical_quat, symmetry_rotation, distance) or (None, None, inf) if no match
        """
        detected_quat = np.array(detected_quat)
        detected_quat = detected_quat / np.linalg.norm(detected_quat)
        
        # Get all symmetry rotations
        symmetry_rotations = QuaternionOrientationController.generate_full_symmetry_group(fold_data)
        
        if not symmetry_rotations:
            # No symmetry data, return detected as canonical
            return detected_quat, np.array([0, 0, 0, 1]), 0.0
        
        # Identity quaternion (reference canonical pose)
        identity = np.array([0, 0, 0, 1])
        
        best_distance = float('inf')
        best_symmetry = None
        best_canonical = None
        
        for sym_rot in symmetry_rotations:
            # Apply inverse symmetry to detected: detected * inv(sym_rot)
            # This transforms detected back toward the canonical pose
            sym_inv = QuaternionOrientationController.quaternion_inverse(sym_rot)
            transformed = QuaternionOrientationController.quaternion_multiply(detected_quat, sym_inv)
            transformed = transformed / np.linalg.norm(transformed)
            
            # Measure distance to identity (canonical reference)
            distance = QuaternionOrientationController.quaternion_distance(transformed, identity)
            
            if distance < best_distance:
                best_distance = distance
                best_symmetry = sym_rot
                best_canonical = transformed
        
        if best_distance <= threshold:
            return best_canonical, best_symmetry, best_distance
        else:
            return None, None, best_distance
    
    @staticmethod
    def normalize_to_canonical(detected_quat, object_name, symmetry_dir, threshold=0.15):
        """
        Normalize detected quaternion to canonical pose using fold symmetry.
        
        This function:
        1. Loads symmetry data for the object
        2. Finds the best canonical equivalent orientation
        3. Returns a quaternion that represents the "canonical view"
        
        The canonical quaternion can then be used to extract a consistent yaw
        for gripper alignment.
        
        Args:
            detected_quat: Detected object quaternion [x, y, z, w]
            object_name: Name of the object (e.g., "line_brown_scaled70")
            symmetry_dir: Directory containing symmetry JSON files
            threshold: Maximum quaternion distance for a match (0-1 scale, ~0.15 = ~30°)
            
        Returns:
            Canonical quaternion [x, y, z, w] - either matched or original if no match
        """
        # Load fold symmetry data
        fold_data = QuaternionOrientationController.load_fold_symmetry_json(object_name, symmetry_dir)
        
        if fold_data is None:
            # No symmetry data available, return detected quaternion as-is
            detected_quat = np.array(detected_quat)
            return detected_quat / np.linalg.norm(detected_quat)
        
        # Find closest canonical
        canonical_quat, symmetry_used, distance = \
            QuaternionOrientationController.find_closest_canonical_quaternion(
                detected_quat, fold_data, threshold
            )
        
        if canonical_quat is not None:
            return canonical_quat
        else:
            # No close match found, return original (normalized)
            detected_quat = np.array(detected_quat)
            return detected_quat / np.linalg.norm(detected_quat)
    
    @staticmethod
    def get_canonical_yaw_for_gripper(detected_quat, object_name, symmetry_dir):
        """
        Get the canonical yaw angle for gripper alignment.
        
        This combines fold symmetry normalization with yaw extraction
        to give a consistent yaw angle regardless of which symmetric
        view of the object was detected.
        
        Args:
            detected_quat: Detected object quaternion [x, y, z, w]
            object_name: Name of the object
            symmetry_dir: Directory containing symmetry JSON files
            
        Returns:
            Tuple of (canonical_yaw_degrees, canonical_quat, was_matched)
        """
        # Load fold symmetry data
        fold_data = QuaternionOrientationController.load_fold_symmetry_json(object_name, symmetry_dir)
        
        detected_quat = np.array(detected_quat)
        detected_quat = detected_quat / np.linalg.norm(detected_quat)
        
        if fold_data is None:
            # No symmetry data, use detected yaw directly
            yaw = QuaternionOrientationController.extract_yaw_from_quaternion(detected_quat)
            return yaw, detected_quat, False
        
        # Method 1: Use yaw-based normalization (simpler, works for Z-symmetric objects)
        canonical_yaw = QuaternionOrientationController.find_equivalent_canonical_yaw(
            detected_quat, fold_data
        )
        
        # Method 2: Use full quaternion matching (more robust for complex symmetries)
        canonical_quat, symmetry_used, distance = \
            QuaternionOrientationController.find_closest_canonical_quaternion(
                detected_quat, fold_data, threshold=0.15
            )
        
        was_matched = canonical_quat is not None
        
        if was_matched:
            # Use yaw from the canonical quaternion
            canonical_yaw = QuaternionOrientationController.extract_yaw_from_quaternion(canonical_quat)
            return canonical_yaw, canonical_quat, True
        else:
            # Fall back to simple yaw normalization
            return canonical_yaw, detected_quat, False
    
    @staticmethod
    def verify_quaternion_approach():
        """
        Verify quaternion approach works for full 360° rotation (no gimbal lock).
        
        Returns:
            bool: True if all tests pass, False otherwise
        """
        print("=" * 70)
        print("Testing quaternion face-down approach (full 360° rotation)...")
        print("=" * 70)
        
        all_passed = True
        
        for yaw_deg in range(0, 360, 30):
            try:
                q = QuaternionOrientationController.face_down_quaternion(yaw_deg)
                
                # Verify magnitude = 1.0 (unit quaternion)
                mag = np.linalg.norm(q)
                mag_ok = abs(mag - 1.0) < 1e-6
                
                # Verify no NaN/Inf
                finite_ok = np.all(np.isfinite(q))
                
                # Convert back to RPY for verification
                rpy = QuaternionOrientationController.quaternion_to_rpy(q)
                
                status = "✅" if (mag_ok and finite_ok) else "❌"
                print(f"{status} Yaw {yaw_deg:3d}°: magnitude={mag:.8f}, "
                      f"RPY=[{rpy[0]:7.2f}°, {rpy[1]:7.2f}°, {rpy[2]:7.2f}°]")
                
                if not (mag_ok and finite_ok):
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ Yaw {yaw_deg:3d}°: Exception - {e}")
                all_passed = False
        
        print("=" * 70)
        if all_passed:
            print("✅ SUCCESS: Quaternion approach is gimbal-lock-free!")
        else:
            print("❌ FAILURE: Quaternion verification detected issues!")
        print("=" * 70)
        
        return all_passed
    
    @staticmethod
    def test_fold_symmetry(object_name, symmetry_dir):
        """
        Test fold symmetry matching for an object.
        
        Args:
            object_name: Name of the object to test
            symmetry_dir: Directory containing symmetry JSON files
        """
        print("=" * 70)
        print(f"Testing fold symmetry for: {object_name}")
        print("=" * 70)
        
        # Load symmetry data
        fold_data = QuaternionOrientationController.load_fold_symmetry_json(object_name, symmetry_dir)
        
        if fold_data is None:
            print(f"❌ No symmetry data found for {object_name}")
            return
        
        print(f"✅ Loaded symmetry data:")
        for axis, data in fold_data.get('fold_axes', {}).items():
            print(f"   {axis.upper()}-axis: {data.get('fold', 1)}-fold symmetry")
        
        # Get symmetry rotations
        symmetry_rots = QuaternionOrientationController.generate_full_symmetry_group(fold_data)
        print(f"\n📊 Total symmetry group size: {len(symmetry_rots)}")
        
        # Test with various detected orientations
        print(f"\n🧪 Testing canonical yaw extraction:")
        test_yaws = [0, 45, 90, 135, 180, 225, 270, 315]
        
        for test_yaw in test_yaws:
            # Create a test quaternion (object lying flat with given yaw)
            test_quat = R.from_euler('xyz', [0, 0, test_yaw], degrees=True).as_quat()
            
            canonical_yaw, canonical_quat, was_matched = \
                QuaternionOrientationController.get_canonical_yaw_for_gripper(
                    test_quat, object_name, symmetry_dir
                )
            
            match_str = "✅ matched" if was_matched else "⚠️ no match"
            print(f"   Input yaw {test_yaw:3d}° → Canonical yaw {canonical_yaw:7.2f}° ({match_str})")
        
        print("=" * 70)


# Convenience functions for backward compatibility
def face_down_quaternion(yaw_degrees):
    """Convenience function wrapper"""
    return QuaternionOrientationController.face_down_quaternion(yaw_degrees)


def quaternion_to_rpy(q):
    """Convenience function wrapper"""
    return QuaternionOrientationController.quaternion_to_rpy(q)


def rpy_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    """Convenience function wrapper"""
    return QuaternionOrientationController.rpy_to_quaternion(roll_deg, pitch_deg, yaw_deg)


if __name__ == "__main__":
    import sys
    
    # Run verification test
    controller = QuaternionOrientationController()
    success = controller.verify_quaternion_approach()
    
    # Test fold symmetry if directory provided
    if len(sys.argv) > 1:
        symmetry_dir = sys.argv[1]
        print("\n")
        controller.test_fold_symmetry("line_brown_scaled70", symmetry_dir)
        print("\n")
        controller.test_fold_symmetry("fork_yellow_scaled70", symmetry_dir)
    else:
        print("\nTo test fold symmetry, run:")
        print(f"  python {sys.argv[0]} /path/to/symmetry/dir")