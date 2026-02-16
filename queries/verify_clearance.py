#!/usr/bin/env python3
"""
Verify Clearance - Checks if assembly objects have enough space for gripper to grab them.

Related files (clearance check pipeline):
  - queries/verify_clearance.py    (this file) — ROS query that checks poses and computes clearance
  - triggers/pre_assembly_check.py — handles failure by invoking human elicitation
  - elicitations/fix_scene.py      — Pydantic schemas for human interaction

For real-world assemblies, this script verifies:
1. All required objects for the assembly are present in the scene
2. Objects that are not yet assembled have sufficient clearance
3. No objects are too close to each other (gripper collision risk)

The algorithm:
1. Load assembly configuration by base name
2. Get current object poses from the scene
3. For each object in the assembly:
   a. Check if it's present in the scene
   b. Check if it's already assembled (using verify_assembly logic)
   c. If not assembled, compute distances to all neighboring objects
   d. Verify clearance is sufficient for gripper to operate
"""

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import time
import sys
import os
import glob

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from primitives.utils.data_path_finder import get_assembly_data_dir, get_symmetry_dir
from primitives.utils.workspace_config import GRIPPER_CENTER_TOOL_OFFSET

# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())
SYMMETRY_DIR = str(get_symmetry_dir())
OBJECT_TOPIC_REAL = "/objects_poses_real"

# Gripper dimensions (UR10e RobotIQ Hand-E gripper specifications)
# These are approximate dimensions in meters
GRIPPER_WIDTH = 0.100      # ~10cm effective gripper width for clearance check
GRIPPER_DEPTH = 0.150      # ~15cm gripper depth (fingertip to palm)
GRIPPER_FINGER_LENGTH = 0.120  # ~12cm finger length
GRIPPER_LENGTH = 0.200     # ~20cm total length (palm to fingertip)

# Clearance margin for safety (meters)
CLEARANCE_MARGIN = 0.01    # 1cm safety margin


def find_assembly_json_by_base_name(base_name, data_dir=ASSEMBLY_DATA_DIR, logger=None):
    """Find the assembly JSON file that contains the given base name."""
    if not os.path.exists(data_dir):
        if logger:
            logger.error(f"Data directory not found: {data_dir}")
        return None

    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    base_name_variants = [base_name, f"{base_name}_scaled70"]

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)

            components = config.get('components', [])
            for component in components:
                comp_name = component.get('name', '')
                if comp_name in base_name_variants:
                    return json_file
        except (json.JSONDecodeError, IOError) as e:
            if logger:
                logger.debug(f"Skipping invalid JSON file {json_file}: {e}")
            continue

    if logger:
        logger.warn(f"No assembly JSON found for base '{base_name}' in {data_dir}")
    return None


def load_object_dimensions(object_name, assembly_config):
    """
    Load dimensions for an object from assembly config.

    Expected format in component:
    "dimensions": {
        "x": width_in_meters,
        "y": depth_in_meters,
        "z": height_in_meters
    }

    If not found, returns a default small object size.
    """
    for component in assembly_config.get('components', []):
        comp_name = component.get('name', '')
        if comp_name == object_name or comp_name == f"{object_name}_scaled70":
            dims = component.get('dimensions', {})
            if dims:
                return {
                    'x': dims.get('x', 0.05),
                    'y': dims.get('y', 0.05),
                    'z': dims.get('z', 0.05)
                }

    # Default small object dimensions if not found
    return {'x': 0.05, 'y': 0.05, 'z': 0.05}


class VerifyClearance(Node):
    def __init__(self, base_name=None, mode='real'):
        super().__init__('verify_clearance')

        self.base_name_input = base_name  # User-provided base name (for finding JSON file)
        self.base_name = None  # Actual base name from JSON (identified by type="board")
        self.assembly_json_file = None
        self.assembly_config = {}
        self.symmetry_dir = SYMMETRY_DIR
        self.mode = mode

        # Load assembly configuration
        if base_name is not None:
            self.assembly_config = self.load_assembly_config(base_name)
            # Auto-detect base from assembly config
            self.base_name = self.get_base_from_config()

        # Subscribers
        object_topic = OBJECT_TOPIC_REAL if mode == 'real' else "/objects_poses_sim"
        self.object_sub = self.create_subscription(TFMessage, object_topic, self.object_callback, 10)

        # Store current poses
        self.current_poses = {}
        self.pose_received = False

    def load_assembly_config(self, base_name=None):
        """Load assembly configuration from JSON file."""
        if base_name is not None:
            json_file = find_assembly_json_by_base_name(base_name, ASSEMBLY_DATA_DIR, self.get_logger())
            if json_file:
                self.assembly_json_file = json_file
            else:
                self.get_logger().error(f"Could not find assembly JSON for base '{base_name}'")
                return {}

        json_file = self.assembly_json_file
        if json_file is None:
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

    def get_base_from_config(self):
        """
        Identify the base object from assembly config by looking for type="board".
        Returns the name of the base object.
        """
        for component in self.assembly_config.get('components', []):
            comp_type = component.get('type', '')
            if comp_type == 'board':
                base_name = component.get('name', '')
                # Remove _scaled70 suffix if present
                base_name = base_name.replace('_scaled70', '')
                self.get_logger().info(f"Identified base object from config: {base_name}")
                return base_name

        # Fallback: use the input base name if no board type found
        self.get_logger().warn(f"No component with type='board' found, using input name: {self.base_name_input}")
        return self.base_name_input

    def object_callback(self, msg):
        """Callback for object poses"""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            self.current_poses[frame_id] = transform
        self.pose_received = True

    def transform_to_matrix(self, transform):
        """Convert ROS Transform to 4x4 transformation matrix"""
        t = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        q = np.array([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
        r = R.from_quat(q)
        rotation_matrix = r.as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = t
        return T

    def get_object_position(self, object_name):
        """Get current position of object in world frame"""
        # Try original name and scaled variant
        names_to_try = [object_name, f"{object_name}_scaled70"]

        for name in names_to_try:
            if name in self.current_poses:
                transform = self.current_poses[name]
                return np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ])
        return None

    def get_object_orientation(self, object_name):
        """Get current orientation of object in world frame"""
        names_to_try = [object_name, f"{object_name}_scaled70"]

        for name in names_to_try:
            if name in self.current_poses:
                transform = self.current_poses[name]
                q = np.array([
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                ])
                return R.from_quat(q).as_matrix()
        return None

    def get_assembly_components(self):
        """Get list of all components in the assembly"""
        return [comp.get('name', '') for comp in self.assembly_config.get('components', [])]

    def is_object_present(self, object_name):
        """Check if object is in the scene"""
        return (object_name in self.current_poses or
                f"{object_name}_scaled70" in self.current_poses)

    def get_bounding_box_aabb(self, object_name, position, rotation=None):
        """
        Compute axis-aligned bounding box (AABB) for an object.

        Args:
            object_name: Name of the object
            position: Center position [x, y, z]
            rotation: Rotation matrix (3x3), if None assumes identity

        Returns:
            Dictionary with min and max corners of AABB
        """
        dims = load_object_dimensions(object_name, self.assembly_config)

        # Half dimensions
        half_x = dims['x'] / 2.0
        half_y = dims['y'] / 2.0
        half_z = dims['z'] / 2.0

        # Corner points in object local frame
        corners_local = np.array([
            [-half_x, -half_y, -half_z],
            [half_x, -half_y, -half_z],
            [-half_x, half_y, -half_z],
            [half_x, half_y, -half_z],
            [-half_x, -half_y, half_z],
            [half_x, -half_y, half_z],
            [-half_x, half_y, half_z],
            [half_x, half_y, half_z],
        ])

        # Transform to world frame if rotation provided
        if rotation is not None:
            corners_world = (rotation @ corners_local.T).T
        else:
            corners_world = corners_local

        # Translate by position
        corners_world = corners_world + position

        # Compute AABB
        aabb_min = np.min(corners_world, axis=0)
        aabb_max = np.max(corners_world, axis=0)

        return {
            'min': aabb_min,
            'max': aabb_max,
            'center': position,
            'dims': dims
        }

    def compute_min_distance_between_aabbs(self, aabb1, aabb2):
        """
        Compute minimum distance between two axis-aligned bounding boxes.

        Returns the minimum gap between the boxes. Negative value indicates overlap.
        """
        # Check distance along each axis
        distances = []

        for axis in range(3):
            min1, max1 = aabb1['min'][axis], aabb1['max'][axis]
            min2, max2 = aabb2['min'][axis], aabb2['max'][axis]

            # Distance along this axis
            if max1 < min2:
                distances.append(min2 - max1)  # aabb1 is before aabb2
            elif max2 < min1:
                distances.append(min1 - max2)  # aabb2 is before aabb1
            else:
                distances.append(0)  # Overlap along this axis

        # Minimum distance is the maximum gap (considering overlap)
        # If any axis has overlap (distance = 0), the minimum distance is 0
        min_dist = max(distances)
        return min_dist

    def check_clearance_for_object(self, object_name, exclude_objects=None):
        """
        Check if an object has sufficient clearance from others for gripper access.

        Args:
            object_name: Name of the object to check
            exclude_objects: List of object names to exclude from clearance check (e.g., assembled objects)

        Returns:
            Dictionary with clearance status and details
        """
        if exclude_objects is None:
            exclude_objects = []

        pos = self.get_object_position(object_name)
        rot = self.get_object_orientation(object_name)

        if pos is None:
            return {
                'status': 'not_present',
                'object': object_name,
                'message': f'Object {object_name} not found in scene'
            }

        aabb_target = self.get_bounding_box_aabb(object_name, pos, rot)

        # Check distances to all other objects
        min_clearance = float('inf')
        closest_object = None
        clearance_issues = []

        for other_name in self.current_poses.keys():
            # Skip self
            if other_name == object_name or other_name == f"{object_name}_scaled70":
                continue

            # Skip excluded objects (e.g., assembled objects)
            should_skip = False
            for excluded in exclude_objects:
                if other_name == excluded or other_name == f"{excluded}_scaled70":
                    should_skip = True
                    break
            if should_skip:
                continue

            # Get other object's AABB
            other_pos = self.get_object_position(other_name)
            other_rot = self.get_object_orientation(other_name)

            if other_pos is None:
                continue

            aabb_other = self.get_bounding_box_aabb(other_name, other_pos, other_rot)

            # Compute distance
            distance = self.compute_min_distance_between_aabbs(aabb_target, aabb_other)

            if distance < min_clearance:
                min_clearance = distance
                closest_object = other_name

            # Check against gripper width + margin
            required_clearance = GRIPPER_WIDTH / 2.0 + CLEARANCE_MARGIN
            if distance < required_clearance:
                clearance_issues.append({
                    'obstacle': other_name,
                    'distance': distance,
                    'required': required_clearance,
                    'deficit': required_clearance - distance
                })

        # Determine status
        if min_clearance == float('inf'):
            # No other objects in scene
            status = 'ok_isolated'
            message = 'Object is isolated in scene (sufficient clearance)'
        elif min_clearance >= GRIPPER_WIDTH / 2.0 + CLEARANCE_MARGIN:
            status = 'ok'
            message = f'Sufficient clearance (minimum gap: {min_clearance:.4f}m from {closest_object})'
        elif min_clearance >= 0:
            status = 'marginal'
            message = f'Marginal clearance (minimum gap: {min_clearance:.4f}m from {closest_object})'
        else:
            status = 'collision'
            message = f'Objects overlapping with {closest_object}'

        return {
            'status': status,
            'object': object_name,
            'message': message,
            'min_clearance': min_clearance if min_clearance != float('inf') else None,
            'closest_object': closest_object,
            'clearance_issues': clearance_issues,
            'gripper_required_width': GRIPPER_WIDTH / 2.0 + CLEARANCE_MARGIN
        }

    def verify_assembly_status(self, object_name, base_name):
        """
        Check if object is already assembled to base.
        Returns True if assembled, False if not.
        """
        # Get current poses
        obj_pos = self.get_object_position(object_name)
        base_pos = self.get_object_position(base_name)
        obj_rot = self.get_object_orientation(object_name)
        base_rot = self.get_object_orientation(base_name)

        if obj_pos is None or base_pos is None:
            return False

        # Get target position relative to base
        for component in self.assembly_config.get('components', []):
            comp_name = component.get('name', '')
            if comp_name == object_name or comp_name == f"{object_name}_scaled70":
                target_pos = component.get('position', {})
                target_pos = np.array([
                    target_pos.get('x', 0),
                    target_pos.get('y', 0),
                    target_pos.get('z', 0)
                ])
                break
        else:
            return False

        # Transform target position to world frame
        if base_rot is not None:
            target_world = base_pos + base_rot @ target_pos
        else:
            target_world = base_pos + target_pos

        # Check if current position is close to target
        position_error = np.linalg.norm(obj_pos - target_world)
        ASSEMBLY_TOLERANCE = 0.01  # 1cm tolerance

        return position_error < ASSEMBLY_TOLERANCE

    def run_verification(self):
        """Run the clearance verification and return results"""
        # Wait for pose data
        start_time = time.time()
        timeout = 5.0
        while not self.pose_received and (time.time() - start_time) < timeout:
            try:
                rclpy.spin_once(self, timeout_sec=0.1)
            except TypeError:
                # Fallback for older rclpy versions
                rclpy.spin_once(self)

        if not self.pose_received or not self.current_poses:
            return {
                'error': 'No pose data received',
                'base_input': self.base_name_input,
                'base_detected': self.base_name
            }

        # Verify base was properly detected
        if not self.base_name:
            return {
                'error': 'Could not identify base object from assembly configuration',
                'base_input': self.base_name_input,
                'assembly_file': self.assembly_json_file
            }

        results = {
            'base_input': self.base_name_input,
            'base_detected': self.base_name,
            'assembly_file': self.assembly_json_file,
            'objects_in_scene': list(self.current_poses.keys()),
            'components': [],
            'summary': {}
        }

        # Get all components in assembly
        components = self.get_assembly_components()

        if not components:
            results['error'] = f'No components found in assembly for base {self.base_name}'
            return results

        present_count = 0
        assembled_count = 0
        clearance_ok_count = 0
        missing_objects = []
        clearance_issues = []

        # First pass: identify missing, present, and assembled objects
        object_status = {}  # normalized_name -> 'missing' | 'base' | 'assembled' | 'unassembled'
        assembled_objects = []  # List of assembled object names

        for obj_name in components:
            # Normalize name (remove _scaled70 suffix for checking)
            normalized_name = obj_name.replace('_scaled70', '')

            # Check if object is present
            if not self.is_object_present(normalized_name):
                object_status[normalized_name] = 'missing'
                missing_objects.append(normalized_name)
            elif normalized_name == self.base_name:
                object_status[normalized_name] = 'base'
                present_count += 1
            else:
                # Check assembly status for non-base objects
                is_assembled = self.verify_assembly_status(normalized_name, self.base_name)
                if is_assembled:
                    object_status[normalized_name] = 'assembled'
                    assembled_objects.append(normalized_name)
                    assembled_count += 1
                    present_count += 1
                else:
                    object_status[normalized_name] = 'unassembled'
                    present_count += 1

        # Second pass: Check clearance and build results
        for obj_name in components:
            normalized_name = obj_name.replace('_scaled70', '')
            status = object_status[normalized_name]

            if status == 'missing':
                results['components'].append({
                    'name': normalized_name,
                    'status': 'missing',
                    'message': f'Object {normalized_name} not present in scene'
                })
            elif status == 'base':
                # Base is fixed - no clearance check needed for the base itself
                results['components'].append({
                    'name': normalized_name,
                    'status': 'base',
                    'message': f'Base object {normalized_name} is the reference frame (fixed position)'
                })
            elif status == 'assembled':
                results['components'].append({
                    'name': normalized_name,
                    'status': 'assembled',
                    'message': f'Object {normalized_name} is assembled to base (clearance check skipped)'
                })
            else:  # status == 'unassembled'
                # Check clearance for unassembled objects against:
                # - The base (unassembled objects must have clearance from base)
                # - Other unassembled objects
                # Exclude only assembled objects (not the base)
                clearance_result = self.check_clearance_for_object(
                    normalized_name,
                    exclude_objects=assembled_objects
                )
                results['components'].append({
                    'name': normalized_name,
                    'status': 'unassembled',
                    'clearance': clearance_result
                })

                if clearance_result['status'] in ['ok', 'ok_isolated']:
                    clearance_ok_count += 1
                else:
                    clearance_issues.append({
                        'object': normalized_name,
                        'clearance_status': clearance_result['status'],
                        'message': clearance_result['message'],
                        'issues': clearance_result.get('clearance_issues', [])
                    })

        # Summary
        results['summary'] = {
            'total_components': len(components),
            'present_in_scene': present_count,
            'missing': len(missing_objects),
            'assembled': assembled_count,
            'unassembled': present_count - assembled_count,
            'unassembled_with_ok_clearance': clearance_ok_count,
            'unassembled_with_clearance_issues': len(clearance_issues),
            'missing_objects': missing_objects,
            'clearance_issues': clearance_issues
        }

        return results


def output_result(result, pretty=False):
    """Output JSON result"""
    if pretty:
        print(json.dumps(result, indent=2, default=str))
    else:
        print("__RESULT_JSON__")
        print(json.dumps(result, default=str))
        print("__END_RESULT_JSON__")


def main():
    parser = argparse.ArgumentParser(
        description='Verify clearance for assembly objects in real world'
    )
    parser.add_argument('--base-name', required=True, help='Base object name (e.g., base1)')
    parser.add_argument('--mode', choices=['sim', 'real'], default='real',
                        help='Simulation or real robot mode')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Timeout for waiting for poses (seconds)')
    parser.add_argument('--pretty', action='store_true',
                        help='Pretty print output for terminal readability')
    args = parser.parse_args()

    # Convert base_name from hyphenated to underscored for internal use
    base_name = args.base_name.replace('-', '_')

    rclpy.init()

    success = False
    error_msg = None
    simple_result = {}

    try:
        node = VerifyClearance(base_name=base_name, mode=args.mode)
        detailed_result = node.run_verification()
        node.destroy_node()
        rclpy.shutdown()

        # Check for errors in detailed result
        if 'error' in detailed_result:
            error_msg = detailed_result['error']
            simple_result = {
                'result': 'failure',
                'base_name': base_name,
                'ready_for_assembly': False,
                'error': error_msg
            }
        else:
            # Extract summary information
            summary = detailed_result.get('summary', {})
            missing = summary.get('missing_objects', [])
            clearance_issues = summary.get('clearance_issues', [])
            objects_with_issues = [issue['object'] for issue in clearance_issues]

            # Determine success
            all_present = len(missing) == 0
            all_clearances_ok = len(objects_with_issues) == 0
            success = all_present and all_clearances_ok

            # Build simple result
            simple_result = {
                'result': 'success' if success else 'failure',
                'base_name': detailed_result.get('base_detected', base_name),
                'ready_for_assembly': success
            }

            # Add error details on failure
            if not success:
                issues = []
                if missing:
                    issues.append(f"{len(missing)} missing")
                if objects_with_issues:
                    issues.append(f"{len(objects_with_issues)} clearance issues")

                simple_result['error'] = ', '.join(issues)

                if missing:
                    simple_result['missing_objects'] = missing
                if objects_with_issues:
                    simple_result['objects_with_clearance_issues'] = objects_with_issues

        output_result(simple_result, pretty=args.pretty)
        sys.exit(0 if success else 1)

    except Exception as e:
        error_msg = str(e)
        simple_result = {
            'result': 'failure',
            'base_name': base_name,
            'ready_for_assembly': False,
            'error': error_msg
        }
        rclpy.shutdown()
        output_result(simple_result, pretty=args.pretty)
        sys.exit(1)


if __name__ == '__main__':
    main()
