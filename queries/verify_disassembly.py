#!/usr/bin/env python3
"""
Verify Disassembly - Checks if object is NOT in assembly position relative to base (opposite of verify_assembly)

The algorithm:
1. Get current object pose and base pose
2. Calculate relative position and orientation of object relative to base
3. Compare with target position and orientation from JSON
4. Check if NOT within tolerance (opposite of assembly verification)
5. Return success if NOT in assembly position (disassembled), failure if still in assembly position
"""

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped, TransformStamped
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import time
import sys
import os

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.data_path_finder import get_assembly_data_dir, get_symmetry_dir, find_assembly_json_by_base_name

# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())
SYMMETRY_DIR = str(get_symmetry_dir())
OBJECT_TOPIC_SIM = "/objects_poses_sim"
OBJECT_TOPIC_REAL = "/objects_poses_real"
EE_TOPIC = "/tcp_pose_broadcaster/pose"

# Tolerance thresholds
POSITION_TOLERANCE = 0.07  # 7cm tolerance for position
ORIENTATION_TOLERANCE_DEG = 5.0  # 5 degrees tolerance for orientation


from primitives.shared.fold_symmetry import load_symmetry_data, equivalent_orientations

# Tolerance for checking if other objects are still assembled (same as verify_assembly)
ASSEMBLY_POSITION_TOLERANCE = 0.01  # 1cm
ASSEMBLY_ORIENTATION_TOLERANCE_DEG = 5.0  # 5 degrees


class VerifyDisassembly(Node):
    def __init__(self, base_name=None, mode='sim', base_topic=None, object_topic=None, ee_topic=EE_TOPIC):
        super().__init__('verify_disassembly')

        # Determine topic based on mode
        if object_topic is None:
            object_topic = OBJECT_TOPIC_REAL if mode == 'real' else OBJECT_TOPIC_SIM
        if base_topic is None:
            base_topic = object_topic

        # Store base name and find assembly JSON file
        self.base_name = base_name
        self.assembly_json_file = None
        self.assembly_config = {}
        self.symmetry_dir = SYMMETRY_DIR

        # Load assembly configuration if base_name is provided
        if base_name is not None:
            self.assembly_config = self.load_assembly_config(base_name)

        # Subscribers for pose data
        self.base_sub = self.create_subscription(TFMessage, base_topic, self.base_callback, 10)
        self.object_sub = self.create_subscription(TFMessage, object_topic, self.object_callback, 10)
        self.ee_sub = self.create_subscription(PoseStamped, ee_topic, self.ee_callback, 10)
        
        # Store current poses
        self.current_poses = {}
        self.current_ee_pose = None
        
    
    def load_assembly_config(self, base_name=None):
        """
        Load assembly configuration from JSON file.
        If base_name is provided, automatically finds the matching JSON file.
        
        Args:
            base_name: Optional base name to search for matching JSON file
            
        Returns:
            Assembly configuration dictionary
        """
        # If base_name is provided, find the matching JSON file
        if base_name is not None:
            json_file = find_assembly_json_by_base_name(base_name, ASSEMBLY_DATA_DIR, self.get_logger())
            if json_file:
                self.assembly_json_file = json_file
            else:
                self.get_logger().error(f"Could not find assembly JSON for base '{base_name}'")
                return {}
        
        # Use found file or fall back to default behavior
        json_file = self.assembly_json_file
        if json_file is None:
            # Fallback: try to find any assembly JSON (for backward compatibility)
            json_file = find_assembly_json_by_base_name("base", ASSEMBLY_DATA_DIR, self.get_logger())
            if json_file is None:
                self.get_logger().error("No assembly JSON file found")
                return {}
        
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.get_logger().error(f"Error loading assembly config from {json_file}: {e}")
            return {}
    
    def base_callback(self, msg):
        """Callback for base poses"""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            self.current_poses[frame_id] = transform
    
    def object_callback(self, msg):
        """Callback for object poses"""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            self.current_poses[frame_id] = transform
    
    def ee_callback(self, msg):
        """Callback for end-effector pose"""
        self.current_ee_pose = msg
    
    def transform_to_matrix(self, transform):
        """Convert ROS Transform to 4x4 transformation matrix"""
        t = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
        q = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
        r = R.from_quat(q)
        rotation_matrix = r.as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = t
        return T
    
    def pose_to_matrix(self, pose):
        """Convert ROS Pose to 4x4 transformation matrix"""
        t = np.array([pose.position.x, pose.position.y, pose.position.z])
        q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        r = R.from_quat(q)
        rotation_matrix = r.as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = t
        return T
    
    def canonicalize_euler(self, orientation):
        """Canonicalize Euler angles"""
        roll, pitch, yaw = orientation
        if abs(pitch) < 5 and abs(abs(roll) - 180) < 5:
            return np.array([0.0, 180.0, (yaw % 360) - 180])
        else:
            return orientation
    
    def matrix_to_rpy(self, T):
        """Convert 4x4 transformation matrix to position and RPY (degrees)"""
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        r = R.from_matrix(rotation_matrix)
        rpy_rad = r.as_euler('xyz')
        rpy_deg = np.degrees(rpy_rad)
        rpy_deg = self.canonicalize_euler(rpy_deg)
        return position, rpy_deg
    
    def get_object_target_position(self, object_name):
        """Get target position for object from assembly configuration (relative to base)"""
        for component in self.assembly_config.get('components', []):
            if component.get('name') == object_name:
                position = component.get('position', {})
                return np.array([position.get('x', 0), position.get('y', 0), position.get('z', 0)])
        return None

    def get_object_target_orientation(self, object_name):
        """
        Get target orientation for object from assembly configuration (relative to base),
        using the quaternion stored in the JSON.
        """
        for component in self.assembly_config.get('components', []):
            comp_name = component.get('name', '')
            if comp_name == object_name:
                rotation = component.get('rotation', {})
                quat = rotation.get('quaternion', {})
                # Default to identity if fields are missing
                return np.array([
                    quat.get('x', 0.0),
                    quat.get('y', 0.0),
                    quat.get('z', 0.0),
                    quat.get('w', 1.0),
                ])
        return None
    
    def verify_disassembly(self, object_name, base_name):
        """
        Verify if object is NOT in assembly position relative to base (opposite of verify_assembly_pose)

        Algorithm:
        1. Get current object pose and base pose
        2. Calculate relative position and orientation of object relative to base
        3. Compare with target position and orientation from JSON
        4. Check if NOT within tolerance (opposite of assembly verification)

        Args:
            object_name: Name of the object to verify
            base_name: Name of the base object

        Returns:
            Tuple: (success: bool, error_data: dict) where success = NOT in assembly position
        """
        # Wait for pose data
        if not self.current_poses:
            error_msg = "no pose data available"
            self.get_logger().error(error_msg)
            return False, {"error": error_msg}

        # Check if object exists
        if object_name not in self.current_poses:
            error_msg = f"{object_name} not found in JSON"
            self.get_logger().error(f"Object {object_name} not found in poses")
            return False, {"error": error_msg}

        # Check if base exists
        if base_name not in self.current_poses:
            error_msg = f"{base_name} not found in JSON"
            self.get_logger().error(f"Base {base_name} not found in poses")
            return False, {"error": error_msg}
        
        # Convert poses to matrices
        T_object_current = self.transform_to_matrix(self.current_poses[object_name].transform)
        T_base_current = self.transform_to_matrix(self.current_poses[base_name].transform)
        
        # Calculate relative transformation: T_object_relative = T_base^(-1) * T_object
        T_object_relative = np.linalg.inv(T_base_current) @ T_object_current
        
        # Extract relative position and orientation
        object_relative_position = T_object_relative[:3, 3]
        object_relative_rotation = R.from_matrix(T_object_relative[:3, :3])
        object_relative_rpy_rad = object_relative_rotation.as_euler('xyz')
        object_relative_rpy_deg = np.degrees(object_relative_rpy_rad)

        # Get target position and orientation from JSON (relative to base)
        target_position_relative = self.get_object_target_position(object_name)
        target_orientation_relative = self.get_object_target_orientation(object_name)

        if target_position_relative is None:
            error_msg = f"target position not found"
            self.get_logger().error(f"No target position found for {object_name} in assembly config")
            return False, {"error": error_msg}

        if target_orientation_relative is None:
            error_msg = f"target orientation not found"
            self.get_logger().error(f"No target orientation found for {object_name} in assembly config")
            return False, {"error": error_msg}
        
        # Calculate position error (vector and magnitude)
        position_error_vector = object_relative_position - target_position_relative
        position_error = np.linalg.norm(position_error_vector)
        
        # === Load fold symmetry and generate equivalent target orientations ===
        fold_data = load_symmetry_data(object_name, self.symmetry_dir)
        
        # Get target orientation as rotation matrix (from quaternion)
        target_quat = target_orientation_relative  # Already a quaternion [x, y, z, w]
        R_target_relative = R.from_quat(target_quat).as_matrix()
        target_rpy_rad = R.from_quat(target_quat).as_euler('xyz')
        target_rpy_deg = np.degrees(target_rpy_rad)

        # Generate all equivalent target orientations using fold symmetry
        equivalent_targets = equivalent_orientations(R_target_relative, fold_data)

        # Check if current orientation matches ANY equivalent target
        min_orientation_error_deg = float('inf')
        best_match_idx = -1
        best_match_error_rpy_deg = None

        for i, R_equiv_target in enumerate(equivalent_targets):
            R_equiv_rotation = R.from_matrix(R_equiv_target)
            orientation_error_rad = (object_relative_rotation.inv() * R_equiv_rotation).magnitude()
            orientation_error_deg = np.degrees(orientation_error_rad)

            if orientation_error_deg < min_orientation_error_deg:
                min_orientation_error_deg = orientation_error_deg
                best_match_idx = i
                # Get the RPY error for the best match
                equiv_rpy_rad = R_equiv_rotation.as_euler('xyz')
                equiv_rpy_deg = np.degrees(equiv_rpy_rad)
                best_match_error_rpy_deg = object_relative_rpy_deg - equiv_rpy_deg

        orientation_error_deg = min_orientation_error_deg
        orientation_error_rpy_deg = best_match_error_rpy_deg if best_match_error_rpy_deg is not None else object_relative_rpy_deg - target_rpy_deg

        # Check if within tolerance (for assembly position)
        # Only position matters for disassembly verification - orientation is informational only
        position_ok = bool(position_error <= POSITION_TOLERANCE)
        within_tolerance = position_ok  # Disassembly success depends only on position

        # Prepare error data for JSON output
        error_data = {
            "position_error_m": {
                "x": round(float(position_error_vector[0]), 2),
                "y": round(float(position_error_vector[1]), 2),
                "z": round(float(position_error_vector[2]), 2)
            },
            "orientation_error_deg": {
                "roll": round(float(orientation_error_rpy_deg[0]), 2),
                "pitch": round(float(orientation_error_rpy_deg[1]), 2),
                "yaw": round(float(orientation_error_rpy_deg[2]), 2)
            }
        }

        # For disassembly: SUCCESS if NOT in assembly position (based on position only)
        if within_tolerance:
            # Object is still in assembly position - disassembly FAILED
            self.get_logger().error("Disassembly verification failed: Object is still in assembly pose")
            self.get_logger().error(f"Position error: [{position_error_vector[0]:.6f}, {position_error_vector[1]:.6f}, {position_error_vector[2]:.6f}]m (magnitude: {position_error:.6f}m) is within tolerance ({POSITION_TOLERANCE}m)")
            return False, error_data
        else:
            # Object is NOT in assembly position - disassembly SUCCESS
            self.get_logger().info("Disassembly verification successful: Object is NOT in assembly pose")
            self.get_logger().info(f"Position error: [{position_error_vector[0]:.6f}, {position_error_vector[1]:.6f}, {position_error_vector[2]:.6f}]m (magnitude: {position_error:.6f}m) exceeds tolerance ({POSITION_TOLERANCE}m) - object moved away")
            return True, error_data

    def get_assembly_order(self, object_name):
        """Get the assembly_order for a component from the assembly config."""
        for component in self.assembly_config.get('components', []):
            if component.get('name') == object_name:
                return component.get('assembly_order')
        return None

    def get_peg_free_axis(self, object_name):
        """Check if component is a peg and get its free rotation axis.

        Pegs are axisymmetric inserts where rotation around the insertion axis
        doesn't matter for assembly verification.

        Returns:
            int or None: 0 (x), 1 (y), or 2 (z) for free rotation axis, None if not a peg
        """
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        for component in self.assembly_config.get('components', []):
            if component.get('name') == object_name:
                if component.get('subtype') == 'peg':
                    axis_str = component.get('axis')
                    return axis_map.get(axis_str)
        return None

    def compute_axis_alignment_error(self, R_current, R_target, axis_idx):
        """Compute angular error between the specified axis of current and target orientations.

        For pegs, we only care that the insertion axis points the right way,
        not about rotation around that axis.
        """
        axis_current = R_current[:, axis_idx]
        axis_target = R_target[:, axis_idx]
        dot = np.clip(np.dot(axis_current, axis_target), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    def get_max_assembly_order(self):
        """Get the highest assembly_order among non-board components."""
        max_order = 0
        for component in self.assembly_config.get('components', []):
            if component.get('type') == 'board':
                continue
            order = component.get('assembly_order', 0)
            max_order = max(max_order, order)
        return max_order

    def get_disassembly_order(self, object_name):
        """Compute disassembly order (reverse of assembly order).

        Assembly order 1..N maps to disassembly order N..1.
        Board (order 0) returns None.
        """
        assembly_order = self.get_assembly_order(object_name)
        if assembly_order is None or assembly_order == 0:
            return None
        max_order = self.get_max_assembly_order()
        return max_order - assembly_order + 1

    def check_lower_order_objects_assembled(self, object_name, base_name):
        """
        Check that all objects with a lower assembly_order than the target object
        are still in their assembled positions. If any got disturbed, the robot
        caused collateral damage during disassembly.

        Args:
            object_name: The object being disassembled
            base_name: The base object

        Returns:
            Tuple: (all_intact: bool, disturbed_objects: list of str)
        """
        target_order = self.get_assembly_order(object_name)
        if target_order is None:
            self.get_logger().warn(f"No assembly_order found for {object_name}, skipping lower-order check")
            return True, []

        disturbed = []
        components = self.assembly_config.get('components', [])

        for component in components:
            comp_name = component.get('name', '')
            comp_order = component.get('assembly_order')

            # Skip: the target object itself, the board, or components without order
            if comp_name == object_name or comp_name == base_name or comp_order is None:
                continue

            # Only check objects that should still be assembled (lower order)
            if comp_order >= target_order:
                continue

            # Check if this object is still in its assembled position
            if comp_name not in self.current_poses:
                self.get_logger().warn(f"[order check] {comp_name} (order {comp_order}) not found in poses")
                disturbed.append(comp_name)
                continue

            T_obj = self.transform_to_matrix(self.current_poses[comp_name].transform)
            T_base = self.transform_to_matrix(self.current_poses[base_name].transform)
            T_relative = np.linalg.inv(T_base) @ T_obj

            # Position check
            target_pos = self.get_object_target_position(comp_name)
            if target_pos is None:
                continue
            pos_error = np.linalg.norm(T_relative[:3, 3] - target_pos)

            # Orientation check
            target_quat = self.get_object_target_orientation(comp_name)
            if target_quat is None:
                continue
            R_target = R.from_quat(target_quat).as_matrix()
            R_current_mat = T_relative[:3, :3]

            peg_free_axis = self.get_peg_free_axis(comp_name)
            if peg_free_axis is not None:
                # For pegs: only check axis alignment, ignore rotation around insertion axis
                min_ori_error = self.compute_axis_alignment_error(R_current_mat, R_target, peg_free_axis)
            else:
                # For regular parts: use full orientation check with fold symmetry
                R_current = R.from_matrix(R_current_mat)
                fold_data = load_symmetry_data(comp_name, self.symmetry_dir)
                equiv_targets = equivalent_orientations(R_target, fold_data)
                min_ori_error = float('inf')
                for R_equiv in equiv_targets:
                    ori_error = np.degrees((R_current.inv() * R.from_matrix(R_equiv)).magnitude())
                    min_ori_error = min(min_ori_error, ori_error)

            pos_ok = pos_error <= ASSEMBLY_POSITION_TOLERANCE
            ori_ok = min_ori_error <= ASSEMBLY_ORIENTATION_TOLERANCE_DEG

            if not (pos_ok and ori_ok):
                self.get_logger().error(
                    f"[order check] {comp_name} (order {comp_order}) DISTURBED: "
                    f"pos_error={pos_error:.4f}m, ori_error={min_ori_error:.2f}°"
                )
                disturbed.append(comp_name)
            else:
                self.get_logger().info(
                    f"[order check] {comp_name} (order {comp_order}) still assembled"
                )

        return len(disturbed) == 0, disturbed

    def check_skipped_objects(self, object_name, base_name):
        """
        Check if objects that should have been disassembled first (higher assembly_order)
        are still in their assembled positions — meaning the agent skipped them.

        Args:
            object_name: The object being disassembled
            base_name: The base object

        Returns:
            Tuple: (none_skipped: bool, skipped_objects: list of str)
        """
        target_order = self.get_assembly_order(object_name)
        if target_order is None:
            return True, []

        skipped = []
        components = self.assembly_config.get('components', [])

        for component in components:
            comp_name = component.get('name', '')
            comp_order = component.get('assembly_order')

            # Skip: the target object itself, the board, or components without order
            if comp_name == object_name or comp_name == base_name or comp_order is None:
                continue

            # Only check objects with higher assembly_order (should be disassembled before this one)
            if comp_order <= target_order:
                continue

            # Check if this object is still assembled (it shouldn't be)
            if comp_name not in self.current_poses:
                # Not in poses — could be already removed, not skipped
                continue

            T_obj = self.transform_to_matrix(self.current_poses[comp_name].transform)
            T_base = self.transform_to_matrix(self.current_poses[base_name].transform)
            T_relative = np.linalg.inv(T_base) @ T_obj

            # Position check
            target_pos = self.get_object_target_position(comp_name)
            if target_pos is None:
                continue
            pos_error = np.linalg.norm(T_relative[:3, 3] - target_pos)

            # Orientation check
            target_quat = self.get_object_target_orientation(comp_name)
            if target_quat is None:
                continue
            R_target = R.from_quat(target_quat).as_matrix()
            R_current_mat = T_relative[:3, :3]

            peg_free_axis = self.get_peg_free_axis(comp_name)
            if peg_free_axis is not None:
                min_ori_error = self.compute_axis_alignment_error(R_current_mat, R_target, peg_free_axis)
            else:
                R_current = R.from_matrix(R_current_mat)
                fold_data = load_symmetry_data(comp_name, self.symmetry_dir)
                equiv_targets = equivalent_orientations(R_target, fold_data)
                min_ori_error = float('inf')
                for R_equiv in equiv_targets:
                    ori_error = np.degrees((R_current.inv() * R.from_matrix(R_equiv)).magnitude())
                    min_ori_error = min(min_ori_error, ori_error)

            pos_ok = pos_error <= ASSEMBLY_POSITION_TOLERANCE
            ori_ok = min_ori_error <= ASSEMBLY_ORIENTATION_TOLERANCE_DEG

            if pos_ok and ori_ok:
                # Still assembled — this object should have been removed first
                disassembly_order = self.get_disassembly_order(comp_name)
                self.get_logger().error(
                    f"[skip check] {comp_name} (disassembly_order {disassembly_order}) "
                    f"still assembled — should have been disassembled first"
                )
                skipped.append(comp_name)

        return len(skipped) == 0, skipped


def main(args=None):
    parser = argparse.ArgumentParser(description='Verify Disassembly - Check if object is NOT in assembly position')
    parser.add_argument('--object-name', type=str, help='Name of the object to verify (optional if --check-all is used)')
    parser.add_argument('--base-name', type=str, required=True, help='Name of the base object')
    parser.add_argument('--mode', type=str, default='sim', choices=['sim', 'real'],
                       help='Mode: sim (reads from /objects_poses_sim) or real (reads from /objects_poses_real)')
    parser.add_argument('--check-all', action='store_true', help='Check if all objects in the assembly are disassembled')
    parser.add_argument('--pretty', action='store_true', help='Pretty print output for terminal readability')
    args = parser.parse_args()

    # Validate arguments
    if not args.check_all and not args.object_name:
        parser.error("Either --object-name or --check-all must be specified")

    rclpy.init()
    node = VerifyDisassembly(base_name=args.base_name, mode=args.mode)

    success = False
    error_data = {}
    error_msg = None
    disassembled_objects = []
    still_assembled_objects = []

    try:
        # Wait for pose data (wait indefinitely until received)
        if args.check_all:
            node.get_logger().info(f"Waiting for pose data to verify all objects are disassembled for base: {args.base_name}")
        else:
            node.get_logger().info(f"Waiting for pose data for object: {args.object_name} and base: {args.base_name}")
        start_time = time.time()
        last_log_time = start_time

        while not node.current_poses:
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.1)

            # Log every 5 seconds to show we're still waiting
            current_time = time.time()
            if current_time - last_log_time >= 5.0:
                elapsed = current_time - start_time
                node.get_logger().info(f"Still waiting for pose data... ({elapsed:.1f}s elapsed)")
                last_log_time = current_time

        elapsed = time.time() - start_time
        node.get_logger().info(f"Received pose data for {len(node.current_poses)} objects (waited {elapsed:.1f}s)")

        if args.check_all:
            # Check all objects in the assembly
            components = node.assembly_config.get('components', [])

            for component in components:
                comp_name = component.get('name', '')
                # Skip the board
                if component.get('type') == 'board':
                    continue

                # Verify this component is disassembled
                try:
                    is_disassembled, _ = node.verify_disassembly(comp_name, args.base_name)
                    if is_disassembled:
                        disassembled_objects.append(comp_name)
                    else:
                        still_assembled_objects.append(comp_name)
                except Exception as e:
                    node.get_logger().debug(f"Could not verify {comp_name}: {e}")
                    still_assembled_objects.append(comp_name)

            # Success if all objects are disassembled
            success = len(still_assembled_objects) == 0

            if success:
                node.get_logger().info(f"All {len(disassembled_objects)} objects are disassembled")
            else:
                node.get_logger().error(f"Found {len(still_assembled_objects)} objects still assembled: {still_assembled_objects}")
        else:
            # Verify disassembly (opposite of assembly verification)
            object_removed, error_data = node.verify_disassembly(
                args.object_name,
                args.base_name
            )

            if object_removed:
                node.get_logger().info("Disassembly verification: SUCCESS - Object is not in assembly position")
                success = True

                errors = {}

                # Check if agent skipped higher-priority objects
                none_skipped, skipped_objects = node.check_skipped_objects(
                    args.object_name, args.base_name
                )
                if not none_skipped:
                    node.get_logger().error(
                        f"Disassembly order violation: Skipped objects that should be removed first: {skipped_objects}"
                    )
                    errors["skipped_objects"] = skipped_objects

                # Check that lower-order objects were not disturbed
                all_intact, disturbed_objects = node.check_lower_order_objects_assembled(
                    args.object_name, args.base_name
                )
                if not all_intact:
                    node.get_logger().error(
                        f"Disassembly order violation: Robot disturbed other objects: {disturbed_objects}"
                    )
                    errors["disturbed_objects"] = disturbed_objects

                if errors:
                    error_data["error"] = errors
            else:
                success = False
                node.get_logger().error("Disassembly verification: FAILED - Object is still in assembly position")

    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
        error_msg = "Interrupted by user"
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
        error_msg = str(e)
    finally:
        # Cleanup ROS
        node.destroy_node()
        rclpy.shutdown()

        # Build result dictionary
        if args.check_all:
            result = {
                "result": "success" if success else "failure",
                "base_name": args.base_name,
                "all_disassembled": success,
                "disassembled_objects": disassembled_objects,
                "still_assembled_objects": still_assembled_objects,
            }
        else:
            result = {
                "result": "success" if success else "failure",
                "object_name": args.object_name,
                "base_name": args.base_name,
            }

            # Add disassembly order (reverse of assembly order)
            disassembly_order = node.get_disassembly_order(args.object_name)
            if disassembly_order is not None:
                result["disassembly_order"] = disassembly_order

            # Add error data if available
            if error_data:
                result.update(error_data)

        # Add error message if present
        if error_msg:
            result["error"] = error_msg

        # Output JSON
        if args.pretty:
            print(json.dumps(result, indent=2))
        else:
            print("__RESULT_JSON__")
            print(json.dumps(result))
            print("__END_RESULT_JSON__")

        # Exit with appropriate code
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

