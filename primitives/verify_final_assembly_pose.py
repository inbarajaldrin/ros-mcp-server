#!/usr/bin/env python3
"""
Verify Final Assembly Pose - Checks if object is in correct position and orientation relative to base

The algorithm:
1. Get current object pose and base pose
2. Calculate relative position and orientation of object relative to base
3. Compare with target position and orientation from JSON
4. Check if within tolerance
5. Return success if match, failure if not
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
import glob

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from primitives.utils.data_path_finder import get_assembly_data_dir, get_symmetry_dir

# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())
SYMMETRY_DIR = str(get_symmetry_dir())
BASE_TOPIC = "/objects_poses_sim"
OBJECT_TOPIC = "/objects_poses_sim"
EE_TOPIC = "/tcp_pose_broadcaster/pose"

# Tolerance thresholds
POSITION_TOLERANCE = 0.01  # 1cm tolerance for position
ORIENTATION_TOLERANCE_DEG = 5.0  # 5 degrees tolerance for orientation


def find_assembly_json_by_base_name(base_name, data_dir=ASSEMBLY_DATA_DIR, logger=None):
    """
    Find the assembly JSON file that contains the given base name.
    
    Args:
        base_name: Name of the base object to search for
        data_dir: Directory to search for JSON files
        logger: Optional logger for debug output
        
    Returns:
        Path to the matching JSON file, or None if not found
    """
    if not os.path.exists(data_dir):
        if logger:
            logger.error(f"Data directory not found: {data_dir}")
        return None
    
    # Search for all JSON files in the data directory
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    # Try exact match first, then with _scaled70 suffix
    base_name_variants = [base_name, f"{base_name}_scaled70"]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)
            
            # Check if any component matches the base name
            components = config.get('components', [])
            for component in components:
                comp_name = component.get('name', '')
                if comp_name in base_name_variants:
                    return json_file
        except (json.JSONDecodeError, IOError) as e:
            # Skip invalid JSON files
            if logger:
                logger.debug(f"Skipping invalid JSON file {json_file}: {e}")
            continue
    
    if logger:
        logger.warn(f"No assembly JSON found for base '{base_name}' in {data_dir}")
    return None


class FoldSymmetry:
    """
    Proper fold symmetry handling (same as in reorient_for_assembly).
    
    The JSON stores symmetry rotations as quaternions.
    These represent rotations IN THE OBJECT FRAME that result in identical appearance.
    
    For fork with 2-fold Y symmetry:
    - Identity (0°): object as-is
    - 180° around Y: object flipped, but looks the same
    
    To generate equivalent targets:
    R_equivalent = R_target × R_symmetry  (object-frame rotation)
    """
    
    @staticmethod
    def load_symmetry_data(object_name, symmetry_dir):
        """Load fold symmetry JSON"""
        patterns = [
            os.path.join(symmetry_dir, f"{object_name}_symmetry.json"),
            os.path.join(symmetry_dir, f"{object_name}*_symmetry.json"),
            os.path.join(symmetry_dir, f"{object_name.replace('_scaled70', '')}*_symmetry.json"),
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
    
    @staticmethod
    def get_symmetry_rotations_as_matrices(fold_data):
        """
        Extract symmetry rotations as rotation matrices.
        
        Returns list of 3x3 rotation matrices representing symmetry transformations.
        Always includes identity.
        """
        if fold_data is None:
            return [np.eye(3)]
        
        symmetry_matrices = []
        seen = set()
        
        # Always include identity
        symmetry_matrices.append(np.eye(3))
        seen.add(tuple(np.eye(3).flatten().round(6)))
        
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
                
                # Convert to rotation matrix
                R_sym = R.from_quat(q).as_matrix()
                
                # Check for duplicates
                key = tuple(R_sym.flatten().round(6))
                if key not in seen:
                    seen.add(key)
                    symmetry_matrices.append(R_sym)
        
        return symmetry_matrices
    
    @staticmethod
    def generate_equivalent_target_orientations(R_target_world, fold_data, logger=None):
        """
        Generate all symmetry-equivalent target orientations.
        
        For an object with fold symmetry, multiple orientations are visually identical.
        This generates all such equivalent orientations for the assembly target.
        
        Math: R_equivalent = R_target × R_symmetry
        (Apply symmetry rotation in object's local frame)
        
        Args:
            R_target_world: Target orientation as 3x3 rotation matrix (world frame)
            fold_data: Fold symmetry data from JSON
            logger: Optional logger for debug output
            
        Returns:
            List of 3x3 rotation matrices (all equivalent target orientations)
        """
        symmetry_rotations = FoldSymmetry.get_symmetry_rotations_as_matrices(fold_data)
        
        if logger:
            logger.info(f"  Generating equivalent targets from {len(symmetry_rotations)} symmetry rotations")
        
        equivalent_targets = []
        for i, R_sym in enumerate(symmetry_rotations):
            # Apply symmetry in object frame: R_equiv = R_target × R_sym
            R_equivalent = R_target_world @ R_sym
            equivalent_targets.append(R_equivalent)
            
            if logger:
                rpy = R.from_matrix(R_equivalent).as_euler('xyz', degrees=True)
                logger.info(f"    Equivalent target {i}: RPY = [{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}]")
        
        return equivalent_targets


class VerifyFinalAssemblyPose(Node):
    def __init__(self, base_name=None, base_topic=BASE_TOPIC, object_topic=OBJECT_TOPIC, ee_topic=EE_TOPIC):
        super().__init__('verify_final_assembly_pose')
        
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
            if component.get('name') == object_name or component.get('name') == f"{object_name}_scaled70":
                position = component.get('position', {})
                return np.array([position.get('x', 0), position.get('y', 0), position.get('z', 0)])
        return None
    
    def get_object_target_orientation(self, object_name):
        """
        Get target orientation for object from assembly configuration (relative to base),
        using the quaternion stored in the JSON.
        
        The JSON structure (per component) is:
        
        "rotation": {
            "rpy": {
                "x": ...,
                "y": ...,
                "z": ...
            },
            "quaternion": {
                "x": ...,
                "y": ...,
                "z": ...,
                "w": ...
            }
        }
        
        We read the quaternion directly to avoid any RPY → quaternion conversions
        that could trigger gimbal lock.
        """
        for component in self.assembly_config.get('components', []):
            comp_name = component.get('name', '')
            if comp_name == object_name or comp_name == f"{object_name}_scaled70":
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
    
    def verify_assembly_pose(self, object_name, base_name):
        """
        Verify if object is in correct position and orientation relative to base
        
        Algorithm:
        1. Get current object pose and base pose
        2. Calculate relative position and orientation of object relative to base
        3. Compare with target position and orientation from JSON
        4. Check if within tolerance
        
        Args:
            object_name: Name of the object to verify
            base_name: Name of the base object
            
        Returns:
            True if object is in correct pose, False otherwise
        """
        # Wait for pose data
        if not self.current_poses:
            self.get_logger().error("No pose data available")
            return False
        
        # Check if object exists
        original_object_name = object_name
        if object_name not in self.current_poses:
            object_name = f"{object_name}_scaled70"
            if object_name not in self.current_poses:
                self.get_logger().error(f"Object {original_object_name} not found in poses")
                return False
        
        # Check if base exists
        original_base_name = base_name
        if base_name not in self.current_poses:
            base_name = f"{base_name}_scaled70"
            if base_name not in self.current_poses:
                self.get_logger().error(f"Base {original_base_name} not found in poses")
                return False
        
        # Convert poses to matrices
        T_object_current = self.transform_to_matrix(self.current_poses[object_name].transform)
        T_base_current = self.transform_to_matrix(self.current_poses[base_name].transform)
        
        # Calculate relative transformation: T_object_relative = T_base^(-1) * T_object
        T_object_relative = np.linalg.inv(T_base_current) @ T_object_current
        
        # Extract relative position and orientation
        object_relative_position = T_object_relative[:3, 3]
        object_relative_rotation = R.from_matrix(T_object_relative[:3, :3])
        object_relative_rpy_rad = object_relative_rotation.as_euler('xyz')
        
        # Get target position and orientation from JSON (relative to base)
        target_position_relative = self.get_object_target_position(original_object_name)
        target_orientation_relative = self.get_object_target_orientation(original_object_name)
        
        if target_position_relative is None:
            self.get_logger().error(f"No target position found for {original_object_name} in assembly config")
            return False
        
        if target_orientation_relative is None:
            self.get_logger().error(f"No target orientation found for {original_object_name} in assembly config")
            return False
        
        # Calculate position error
        position_error = np.linalg.norm(object_relative_position - target_position_relative)
        
        # === Load fold symmetry and generate equivalent target orientations ===
        fold_data = FoldSymmetry.load_symmetry_data(original_object_name, self.symmetry_dir)
        if fold_data is None:
            fold_data = FoldSymmetry.load_symmetry_data(f"{original_object_name}_scaled70", self.symmetry_dir)
        
        # Get target orientation as rotation matrix (from quaternion)
        target_quat = target_orientation_relative  # Already a quaternion [x, y, z, w]
        R_target_relative = R.from_quat(target_quat).as_matrix()
        
        # Generate all equivalent target orientations using fold symmetry
        equivalent_targets = FoldSymmetry.generate_equivalent_target_orientations(
            R_target_relative, fold_data, logger=self.get_logger() if fold_data else None
        )
        
        # Check if current orientation matches ANY equivalent target
        min_orientation_error_deg = float('inf')
        best_match_idx = -1
        
        for i, R_equiv_target in enumerate(equivalent_targets):
            R_equiv_rotation = R.from_matrix(R_equiv_target)
            orientation_error_rad = (object_relative_rotation.inv() * R_equiv_rotation).magnitude()
            orientation_error_deg = np.degrees(orientation_error_rad)
            
            if orientation_error_deg < min_orientation_error_deg:
                min_orientation_error_deg = orientation_error_deg
                best_match_idx = i
        
        orientation_error_deg = min_orientation_error_deg
        
        # Check if within tolerance
        position_ok = position_error <= POSITION_TOLERANCE
        orientation_ok = orientation_error_deg <= ORIENTATION_TOLERANCE_DEG
        
        if position_ok and orientation_ok:
            self.get_logger().info("Verification successful: Object is in correct assembly pose")
            return True
        else:
            self.get_logger().error("Verification failed: Object is NOT in correct assembly pose")
            if not position_ok:
                self.get_logger().error(f"Position error ({position_error:.6f}m) exceeds tolerance ({POSITION_TOLERANCE}m)")
            if not orientation_ok:
                self.get_logger().error(f"Orientation error ({orientation_error_deg:.2f}°) exceeds tolerance ({ORIENTATION_TOLERANCE_DEG}°)")
            return False


def main(args=None):
    parser = argparse.ArgumentParser(description='Verify Final Assembly Pose - Check if object is in correct position')
    parser.add_argument('--object-name', type=str, required=True, help='Name of the object to verify')
    parser.add_argument('--base-name', type=str, required=True, help='Name of the base object')
    args = parser.parse_args()
    
    rclpy.init()
    node = VerifyFinalAssemblyPose(base_name=args.base_name)
    
    try:
        # Wait for pose data (wait indefinitely until received)
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
        
        # Verify assembly pose
        success = node.verify_assembly_pose(
            args.object_name,
            args.base_name
        )
        
        if success:
            node.get_logger().info("Assembly pose verification: SUCCESS")
        else:
            node.get_logger().error("Assembly pose verification: FAILED - Placement failed")
        
        # Exit with appropriate code
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()

