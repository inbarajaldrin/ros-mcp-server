#!/usr/bin/env python3
"""
Get Target Object Pose

Calculates the target object pose in world frame by:
1. Loading assembly configuration from JSON
2. Reading base pose (from ROS topic in sim mode, or using default in real mode)
3. Extracting target position and orientation from JSON (relative to base)
4. Transforming target pose from base frame to world frame

Output: JSON containing only the target object pose (world frame)
"""

import sys
import os

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import numpy as np
import argparse
import glob
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R
import time

from primitives.utils.data_path_finder import get_assembly_data_dir, get_aruco_data_dir

# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())
GRASP_DATA_DIR = get_aruco_data_dir() / "grasp"
BASE_TOPIC = "/objects_poses_sim"

# Default base position and orientation (used in real mode)
DEFAULT_BASE_POSITION = [0.5, -0.37, 0.1882]  # [x, y, z] in meters
DEFAULT_BASE_ORIENTATION = [0.0, 0.0, 0.0, 1.0]  # [x, y, z, w] quaternion


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


def load_assembly_config(base_name, data_dir=ASSEMBLY_DATA_DIR):
    """
    Load the assembly configuration from JSON file.
    If base_name is provided, automatically finds the matching JSON file.

    Args:
        base_name: Base name to search for matching JSON file
        data_dir: Directory to search for JSON files

    Returns:
        Assembly configuration dictionary
    """
    json_file = find_assembly_json_by_base_name(base_name, data_dir, None)
    if json_file is None:
        print(f"Error: Could not find assembly JSON for base '{base_name}' in {data_dir}")
        return {}

    try:
        with open(json_file, 'r') as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        print(f"Error: Assembly file not found: {json_file}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Error parsing assembly JSON: {e}")
        return {}


def get_object_target_position(assembly_config, object_name):
    """
    Get target position for object from assembly configuration (relative to base).

    Args:
        assembly_config: Assembly configuration dictionary
        object_name: Name of the object

    Returns:
        numpy array [x, y, z] or None if not found
    """
    for component in assembly_config.get('components', []):
        comp_name = component.get('name', '')
        if comp_name == object_name or comp_name == f"{object_name}_scaled70":
            position = component.get('position', {})
            return np.array([
                position.get('x', 0.0),
                position.get('y', 0.0),
                position.get('z', 0.0)
            ])
    return None


def get_object_target_orientation(assembly_config, object_name):
    """
    Get target orientation for object from assembly configuration (relative to base),
    using the quaternion stored in the JSON.

    Args:
        assembly_config: Assembly configuration dictionary
        object_name: Name of the object

    Returns:
        numpy array [x, y, z, w] quaternion or None if not found
    """
    for component in assembly_config.get('components', []):
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


def load_grasp_points_data(object_name, grasp_data_dir=GRASP_DATA_DIR):
    """
    Load grasp points data for a specific object from JSON file.

    Args:
        object_name: Name of the object (will search with and without _scaled70 suffix)
        grasp_data_dir: Directory containing grasp points JSON files

    Returns:
        Dictionary with grasp points data, or None if not found
    """
    # Try with _scaled70 suffix first, then without
    object_name_variants = [
        f"{object_name}_scaled70",
        object_name,
        object_name.replace('_scaled70', '')
    ]

    for variant in object_name_variants:
        grasp_file = grasp_data_dir / f"{variant}_grasp_points_all_markers.json"
        if grasp_file.exists():
            try:
                with open(grasp_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading grasp points from {grasp_file}: {e}")
                continue

    return None


def get_grasp_point_by_id(grasp_data, grasp_id):
    """
    Get a specific grasp point by ID from grasp data.

    Args:
        grasp_data: Grasp points data dictionary
        grasp_id: ID of the grasp point to retrieve

    Returns:
        Grasp point dictionary with 'position' and 'id', or None if not found
    """
    if grasp_data is None:
        return None

    grasp_points = grasp_data.get('grasp_points', [])
    for gp in grasp_points:
        if gp.get('id') == grasp_id:
            return gp

    return None


def transform_grasp_point_to_world(grasp_point_local, object_position, object_quaternion):
    """
    Transform grasp point from object/CAD center frame to world frame.

    This follows the same transformation as grasp_points_publisher.py:
    pos_world = object_position + R_object @ pos_local

    Args:
        grasp_point_local: Dict with 'position' (x, y, z) relative to object/CAD center
        object_position: Object position in world frame [x, y, z]
        object_quaternion: Object orientation in world frame [x, y, z, w]

    Returns:
        Transformed position in world frame as numpy array [x, y, z]
    """
    # Extract local position (relative to CAD center / object frame)
    pos_local = np.array([
        grasp_point_local['position']['x'],
        grasp_point_local['position']['y'],
        grasp_point_local['position']['z']
    ])

    # Create rotation matrix from object quaternion
    r_object_world = R.from_quat(object_quaternion)
    rot_matrix = r_object_world.as_matrix()

    # Transform position to world frame
    pos_world = object_position + rot_matrix @ pos_local

    return pos_world


class BasePoseReader(Node):
    """ROS2 node to read base pose from topic"""
    def __init__(self, base_name, base_topic=BASE_TOPIC):
        super().__init__('base_pose_reader')
        self.base_name = base_name
        self.current_poses = {}
        self.pose_received = False
        
        self.subscription = self.create_subscription(
            TFMessage,
            base_topic,
            self.pose_callback,
            10
        )
    
    def pose_callback(self, msg):
        """Callback for pose data"""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            self.current_poses[frame_id] = transform
            if frame_id == self.base_name or frame_id == f"{self.base_name}_scaled70":
                self.pose_received = True
    
    def get_base_pose(self, timeout=5.0):
        """Get base pose from topic with timeout"""
        start_time = time.time()
        while rclpy.ok() and not self.pose_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        base_key = self.base_name if self.base_name in self.current_poses else f"{self.base_name}_scaled70"
        if base_key not in self.current_poses:
            return None, None
        
        transform = self.current_poses[base_key].transform
        position = np.array([
            transform.translation.x,
            transform.translation.y,
            transform.translation.z
        ])
        quaternion = np.array([
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w
        ])
        
        return position, quaternion


def main(args=None):
    parser = argparse.ArgumentParser(description='Get Target Object Pose')
    parser.add_argument('--object-name', type=str, required=True, help='Name of the object')
    parser.add_argument('--base-name', type=str, required=True, help='Name of the base object')
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: sim (reads base pose from topic) or real (uses default base pose)')

    args = parser.parse_args()
    
    # Load assembly configuration
    assembly_config = load_assembly_config(args.base_name)
    if not assembly_config:
        print(f"Error: Failed to load assembly config for base '{args.base_name}'")
        sys.exit(1)
    
    # Get base pose based on mode
    if args.mode == 'sim':
        # Sim mode: Read base pose from ROS topic
        try:
            rclpy.init()
            base_reader = BasePoseReader(args.base_name)
            base_position, base_quaternion = base_reader.get_base_pose(timeout=5.0)
            base_reader.destroy_node()
            rclpy.shutdown()

            if base_position is None or base_quaternion is None:
                print(f"Error: Could not read base pose from ROS topic.")
                sys.exit(1)
        except Exception as e:
            print(f"Error: Could not read base pose from ROS topic ({e}).")
            sys.exit(1)
    else:  # real mode
        # Real mode: Use default base position and orientation
        base_position = np.array(DEFAULT_BASE_POSITION)
        base_quaternion = np.array(DEFAULT_BASE_ORIENTATION)

    # Get target position from JSON (relative to base)
    target_position_relative = get_object_target_position(assembly_config, args.object_name)
    if target_position_relative is None:
        print(f"Error: No target position found for {args.object_name} in assembly config")
        sys.exit(1)

    # Get target orientation from JSON (relative to base)
    target_orientation_relative = get_object_target_orientation(assembly_config, args.object_name)
    if target_orientation_relative is None:
        print(f"Error: No target orientation found for {args.object_name} in assembly config")
        sys.exit(1)

    # Transform target position from base frame to world frame
    # Position transformation: target_abs = base_position + R_base @ target_relative
    R_base = R.from_quat(base_quaternion).as_matrix()
    target_object_position_abs = base_position + R_base @ target_position_relative

    # Transform target orientation from base frame to world frame
    # Orientation transformation: R_target_abs = R_base @ R_target_relative
    R_target_relative = R.from_quat(target_orientation_relative).as_matrix()
    R_target_abs = R_base @ R_target_relative
    target_object_orientation_abs = R.from_matrix(R_target_abs).as_quat()

    # Output only target object pose as JSON
    result = {
        "target_object_pose": {
            "position": {
                "x": float(target_object_position_abs[0]),
                "y": float(target_object_position_abs[1]),
                "z": float(target_object_position_abs[2])
            },
            "orientation": {
                "x": float(target_object_orientation_abs[0]),
                "y": float(target_object_orientation_abs[1]),
                "z": float(target_object_orientation_abs[2]),
                "w": float(target_object_orientation_abs[3])
            }
        }
    }

    # Commented out: Additional outputs (uncomment if needed for debugging)
    # result["object_name"] = args.object_name
    # result["base_name"] = args.base_name
    # result["mode"] = args.mode
    # result["base_pose"] = {
    #     "position": {
    #         "x": float(base_position[0]),
    #         "y": float(base_position[1]),
    #         "z": float(base_position[2])
    #     },
    #     "orientation": {
    #         "x": float(base_quaternion[0]),
    #         "y": float(base_quaternion[1]),
    #         "z": float(base_quaternion[2]),
    #         "w": float(base_quaternion[3])
    #     }
    # }
    # result["target_relative_to_base"] = {
    #     "position": {
    #         "x": float(target_position_relative[0]),
    #         "y": float(target_position_relative[1]),
    #         "z": float(target_position_relative[2])
    #     },
    #     "orientation": {
    #         "x": float(target_orientation_relative[0]),
    #         "y": float(target_orientation_relative[1]),
    #         "z": float(target_orientation_relative[2]),
    #         "w": float(target_orientation_relative[3])
    #     }
    # }

    print(json.dumps(result, indent=2))
    sys.exit(0)


if __name__ == '__main__':
    main()
