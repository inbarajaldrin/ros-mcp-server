#!/usr/bin/env python3
"""
Reorient for Assembly - Proper Fold Symmetry + Extended Cardinals

CORRECT FOLD SYMMETRY USAGE:
1. Target orientation from JSON is the "canonical" assembly pose
2. Fold symmetry defines which OTHER orientations look identical
3. Generate all equivalent targets: target × each_symmetry_rotation
4. Find cardinal EE that places object closest to ANY equivalent target

KEY: The symmetry rotations define object-frame rotations that result in
identical appearance. So target rotated by symmetry = visually same assembly.
"""

import sys
import os

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import argparse
import time
import glob

from primitives.utils.ik_solver import ik_objective_quaternion, forward_kinematics, dh_params
from primitives.utils.data_path_finder import get_assembly_data_dir, get_symmetry_dir

# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())
SYMMETRY_DIR = str(get_symmetry_dir())
DEFAULT_OBJECT_TOPIC = "/objects_poses_sim"
DEFAULT_EE_TOPIC = "/tcp_pose_broadcaster/pose"


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
                    if logger:
                        logger.info(f"Found assembly JSON for base '{base_name}': {json_file}")
                    return json_file
        except (json.JSONDecodeError, IOError) as e:
            # Skip invalid JSON files
            if logger:
                logger.debug(f"Skipping invalid JSON file {json_file}: {e}")
            continue
    
    if logger:
        logger.warn(f"No assembly JSON found for base '{base_name}' in {data_dir}")
    return None


class ExtendedCardinalOrientations:
    """24 extended cardinal orientations with intermediary angles"""
    
    @staticmethod
    def get_all_extended_cardinals():
        cardinals = {}
        
        # Primary face directions (cardinal)
        primary_directions = {
            'down': (180, 0),
            'forward': (90, 0),
            'backward': (90, 180),
            'right': (90, -90),
        }
        
        # Intermediary face directions (45° increments)
        intermediary_directions = {
            'forward_right': (90, -45),
            'forward_left': (90, 45),
            'backward_right': (90, -135),
            'backward_left': (90, 135),
        }
        
        # Roll variations for primary directions (0°, 90°, 180°, 270°)
        roll_angles = [0, 90, 180, 270]
        
        # Add primary cardinal directions with roll variations (4 × 4 = 16)
        for face_name, (pitch, yaw) in primary_directions.items():
            for roll in roll_angles:
                name = f"face_{face_name}_roll{roll}"
                q = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat()
                cardinals[name] = q
        
        # Add intermediary directions with 2 roll variations each (4 × 2 = 8)
        # Using only 0° and 180° rolls for intermediaries to keep total at 24
        intermediary_rolls = [0, 180]
        for face_name, (pitch, yaw) in intermediary_directions.items():
            for roll in intermediary_rolls:
                name = f"face_{face_name}_roll{roll}"
                q = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat()
                cardinals[name] = q
        
        return cardinals
    
    @staticmethod
    def rotation_matrix_distance(R1, R2):
        """Angular distance between two rotation matrices in degrees."""
        R_diff = R1.T @ R2
        trace = np.trace(R_diff)
        cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    @staticmethod
    def get_cardinal_rpy(name):
        parts = name.split('_')
        roll = int(parts[-1].replace('roll', ''))
        
        # Reconstruct direction name (may have multiple parts like "forward_right")
        direction_parts = parts[1:-1]
        direction = '_'.join(direction_parts)
        
        pitch_yaw = {
            # Primary cardinals
            'down': (180, 0), 'up': (0, 0), 'forward': (90, 0),
            'backward': (90, 180), 'left': (90, 90), 'right': (90, -90),
            # Intermediary horizontal directions
            'forward_right': (90, -45), 'forward_left': (90, 45),
            'backward_right': (90, -135), 'backward_left': (90, 135),
        }
        
        pitch, yaw = pitch_yaw.get(direction, (0, 0))
        return (roll, pitch, yaw)
    
    @staticmethod
    def find_closest_cardinal(R_orientation, threshold_deg=10.0):
        """
        Find the closest cardinal orientation to the given rotation matrix.
        
        Args:
            R_orientation: 3x3 rotation matrix
            threshold_deg: Maximum angular distance to be considered "close" to a cardinal
            
        Returns:
            (cardinal_name, cardinal_quat, distance_deg) if within threshold, else (None, None, inf)
        """
        cardinals = ExtendedCardinalOrientations.get_all_extended_cardinals()
        
        best_name = None
        best_quat = None
        best_distance = float('inf')
        
        for card_name, card_quat in cardinals.items():
            R_cardinal = R.from_quat(card_quat).as_matrix()
            distance = ExtendedCardinalOrientations.rotation_matrix_distance(
                R_orientation, R_cardinal
            )
            
            if distance < best_distance:
                best_distance = distance
                best_name = card_name
                best_quat = card_quat
        
        if best_distance <= threshold_deg:
            return (best_name, best_quat, best_distance)
        else:
            return (None, None, best_distance)


class FoldSymmetry:
    """
    Proper fold symmetry handling.
    
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


class ReorientForAssembly(Node):
    def __init__(self, mode=None, object_topic=None, ee_topic=DEFAULT_EE_TOPIC):
        super().__init__('reorient_for_assembly')
        
        # Mode must be explicitly specified
        if mode is None:
            raise ValueError("Mode must be explicitly specified. Use 'sim' or 'real'.")
        if mode not in ['sim', 'real']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'sim' or 'real'.")
        
        self.mode = mode  # 'sim' or 'real'
        
        # Set default object topic based on mode if not provided
        if object_topic is None:
            if self.mode == 'sim':
                object_topic = DEFAULT_OBJECT_TOPIC  # "/objects_poses_sim"
            else:
                # Real mode: no object topic needed (orientations provided via arguments)
                object_topic = None
        
        self.assembly_config = {}
        self.assembly_json_file = None
        self.loaded_base_name = None
        self.symmetry_dir = SYMMETRY_DIR
        
        # Only subscribe to object topic in sim mode
        if object_topic is not None:
            self.object_sub = self.create_subscription(TFMessage, object_topic, self.object_callback, 10)
        else:
            self.object_sub = None
        
        self.ee_sub = self.create_subscription(PoseStamped, ee_topic, self.ee_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        self.current_poses = {}
        self.current_ee_pose = None
        self.current_joint_angles = None
        self.joint_angles_received = False
        self.trajectory_success = False
        self.trajectory_completed = False
        
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                           "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.action_client = ActionClient(self, FollowJointTrajectory, 
                                         '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        
        # Cardinal error threshold increment mechanism (similar to move_to_grasp)
        self.cardinal_error_threshold_initial = 45.0  # Initial threshold in degrees
        self.cardinal_error_threshold_max = 180.0  # Maximum threshold to try
        self.cardinal_error_threshold_increment = 10.0  # Increment threshold by this amount each retry
        self.current_cardinal_error_threshold = self.cardinal_error_threshold_initial
        
        self.get_logger().info(f"ReorientForAssembly initialized (Mode: {self.mode}, Fold Symmetry + 24-Cardinal with Intermediary)")
    
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
                self.loaded_base_name = base_name
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
                self.get_logger().info(f"Loaded assembly config from: {json_file}")
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.get_logger().error(f"Error loading assembly config from {json_file}: {e}")
            return {}
    
    def object_callback(self, msg):
        for transform in msg.transforms:
            self.current_poses[transform.child_frame_id] = transform
    
    def ee_callback(self, msg):
        self.current_ee_pose = msg
    
    def joint_state_callback(self, msg):
        if len(msg.name) == 6 and len(msg.position) == 6:
            joint_dict = dict(zip(msg.name, msg.position))
            positions = [joint_dict.get(name, 0) for name in self.joint_names]
            if len(positions) == 6:
                self.current_joint_angles = np.array(positions)
                self.joint_angles_received = True
    
    def get_rotation_from_transform(self, transform):
        q = np.array([transform.rotation.x, transform.rotation.y,
                      transform.rotation.z, transform.rotation.w])
        return R.from_quat(q).as_matrix()
    
    def get_rotation_from_quat(self, quat):
        return R.from_quat(quat).as_matrix()
    
    def get_pose_from_msg(self, pose_msg):
        position = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y,
                            pose_msg.pose.position.z])
        q = np.array([pose_msg.pose.orientation.x, pose_msg.pose.orientation.y,
                      pose_msg.pose.orientation.z, pose_msg.pose.orientation.w])
        return position, R.from_quat(q).as_matrix()
    
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
    
    def read_current_joint_angles(self):
        self.joint_angles_received = False
        timeout = 0
        while rclpy.ok() and not self.joint_angles_received and timeout < 100:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout += 1
        return self.current_joint_angles.copy() if self.joint_angles_received else None
    
    def compute_ik_with_current_seed(self, target_position, target_quat, max_tries=5, dx=0.001):
        target_rot = R.from_quat(target_quat).as_matrix()
        target_pose = np.eye(4)
        target_pose[:3, 3] = target_position
        target_pose[:3, :3] = target_rot
        
        if self.current_joint_angles is None:
            return None
        
        q_guess = self.current_joint_angles.copy()
        best_result, best_cost = None, float('inf')
        joint_bounds = [(-np.pi, np.pi)] * 6
        
        for i in range(max_tries):
            perturbed = target_pose.copy()
            perturbed[0, 3] += i * dx
            
            result = minimize(ik_objective_quaternion, q_guess, args=(perturbed,),
                            method='L-BFGS-B', bounds=joint_bounds)
            if result.success:
                cost = ik_objective_quaternion(result.x, perturbed)
                if cost < 0.01:
                    return result.x
                if cost < best_cost:
                    best_cost, best_result = cost, result.x
        
        if best_result is not None and best_cost < 0.1:
            return best_result
        
        # Fallback seeds - use quaternion to extract yaw component without gimbal lock
        # Extract yaw from input quaternion directly (avoids gimbal lock from RPY conversion)
        # Yaw from quaternion: yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        yaw_rad = np.arctan2(2.0 * (target_quat[3] * target_quat[2] + target_quat[0] * target_quat[1]), 
                            1.0 - 2.0 * (target_quat[1]**2 + target_quat[2]**2))
        yaw_deg = np.degrees(yaw_rad)
        
        seeds = [
            np.radians([85, -80, 90, -90, -90, yaw_deg]),
            np.radians([90, -90, 90, -90, -90, yaw_deg]),
            np.radians([0, -90, 90, -90, -90, yaw_deg]),
            np.radians([180, -90, 90, -90, -90, yaw_deg]),
        ]
        
        for seed in seeds:
            for i in range(max_tries):
                perturbed = target_pose.copy()
                perturbed[0, 3] += i * dx
                result = minimize(ik_objective_quaternion, seed, args=(perturbed,),
                                method='L-BFGS-B', bounds=joint_bounds)
                if result.success:
                    cost = ik_objective_quaternion(result.x, perturbed)
                    if cost < 0.01:
                        return result.x
                    if cost < best_cost:
                        best_cost, best_result = cost, result.x
        
        return best_result if best_cost < 0.1 else None
    
    def compute_cardinal_to_cardinal_adjustment(self, R_object_current, R_object_target_world, 
                                                 R_EE_current, R_grasp, fold_data):
        """
        If both current and target object orientations are cardinals, compute
        a targeted adjustment instead of searching all cardinals.
        
        Args:
            R_object_current: Current object orientation (3x3 matrix)
            R_object_target_world: Target object orientation (3x3 matrix)
            R_EE_current: Current EE orientation (3x3 matrix)
            R_grasp: Grasp relationship (3x3 matrix)
            fold_data: Fold symmetry data
            
        Returns:
            (success, best_quat, resulting_object_R, matched_target_R, object_error)
            or (False, None, None, None, inf) if optimization not applicable
        """
        CARDINAL_THRESHOLD = 45.0  # degrees
        
        # Check if current object is close to a cardinal
        current_cardinal_name, current_cardinal_quat, current_dist = \
            ExtendedCardinalOrientations.find_closest_cardinal(R_object_current, CARDINAL_THRESHOLD)
        
        if current_cardinal_name is None:
            self.get_logger().info(f"  Current object not in cardinal pose (closest: {current_dist:.1f}°)")
            return (False, None, None, None, float('inf'))
        
        self.get_logger().info(f"  ✓ Current object is in cardinal pose: {current_cardinal_name} (error: {current_dist:.1f}°)")
        
        # Generate equivalent targets and check if any is close to a cardinal
        equivalent_targets = FoldSymmetry.generate_equivalent_target_orientations(
            R_object_target_world, fold_data, None  # Don't log here
        )
        
        # First, find which equivalent target is closest to a cardinal (for validation)
        best_target_cardinal = None
        best_target_cardinal_quat = None
        best_target_dist = float('inf')
        
        for R_target_equiv in equivalent_targets:
            target_cardinal_name, target_cardinal_quat, target_dist = \
                ExtendedCardinalOrientations.find_closest_cardinal(R_target_equiv, CARDINAL_THRESHOLD)
            
            if target_cardinal_name is not None and target_dist < best_target_dist:
                best_target_dist = target_dist
                best_target_cardinal = target_cardinal_name
                best_target_cardinal_quat = target_cardinal_quat
        
        if best_target_cardinal is None:
            self.get_logger().info(f"  Target not in cardinal pose (closest: {best_target_dist:.1f}°)")
            return (False, None, None, None, float('inf'))
        
        self.get_logger().info(f"  ✓ Target is in cardinal pose: {best_target_cardinal} (error: {best_target_dist:.1f}°)")
        
        # Now find the equivalent target that's closest to the CURRENT object orientation
        # This ensures we make the smallest possible adjustment
        best_target_R = None
        min_distance_to_current = float('inf')
        
        for R_target_equiv in equivalent_targets:
            # Check if this equivalent target is close to a cardinal (must be valid)
            target_cardinal_name, _, target_dist = \
                ExtendedCardinalOrientations.find_closest_cardinal(R_target_equiv, CARDINAL_THRESHOLD)
            
            if target_cardinal_name is not None:
                # Calculate distance from current object to this equivalent target
                distance_to_current = ExtendedCardinalOrientations.rotation_matrix_distance(
                    R_object_current, R_target_equiv
                )
                
                if distance_to_current < min_distance_to_current:
                    min_distance_to_current = distance_to_current
                    best_target_R = R_target_equiv
        
        if best_target_R is None:
            # Fallback: use the first equivalent target that's close to a cardinal
            for R_target_equiv in equivalent_targets:
                target_cardinal_name, _, _ = \
                    ExtendedCardinalOrientations.find_closest_cardinal(R_target_equiv, CARDINAL_THRESHOLD)
                if target_cardinal_name is not None:
                    best_target_R = R_target_equiv
                    break
        
        if best_target_R is None:
            self.get_logger().error("  Failed to find valid equivalent target")
            return (False, None, None, None, float('inf'))
        
        # Log which equivalent target we're using
        target_rpy = R.from_matrix(best_target_R).as_euler('xyz', degrees=True)
        self.get_logger().info(f"  → Using equivalent target RPY: [{target_rpy[0]:.1f}, {target_rpy[1]:.1f}, {target_rpy[2]:.1f}] (closest to current: {min_distance_to_current:.1f}°)")
        
        # Compute the rotation needed to go from current object to best target
        # R_adjust_object = R_target @ R_current^T
        R_adjust_object = best_target_R @ R_object_current.T
        
        # Apply this adjustment to the EE
        # Since R_object = R_EE @ R_grasp, we have:
        # R_EE_new @ R_grasp = R_adjust_object @ (R_EE_current @ R_grasp)
        # R_EE_new @ R_grasp = R_adjust_object @ R_EE_current @ R_grasp
        # R_EE_new = R_adjust_object @ R_EE_current
        R_EE_new = R_adjust_object @ R_EE_current
        
        # Verify the result
        R_object_result = R_EE_new @ R_grasp
        object_error = ExtendedCardinalOrientations.rotation_matrix_distance(
            R_object_result, best_target_R
        )
        
        # Calculate the actual adjustment angle
        adjustment_angle = ExtendedCardinalOrientations.rotation_matrix_distance(
            R_object_current, best_target_R
        )
        
        # Always snap EE to nearest cardinal if within threshold
        EE_cardinal_name, EE_cardinal_quat, EE_cardinal_dist = \
            ExtendedCardinalOrientations.find_closest_cardinal(R_EE_new, threshold_deg=15.0)
        
        if EE_cardinal_name is not None:
            # Save original object error for logging
            original_object_error = object_error
            
            # Use the cardinal EE orientation
            R_EE_cardinal = R.from_quat(EE_cardinal_quat).as_matrix()
            R_object_from_cardinal = R_EE_cardinal @ R_grasp
            
            # Find the closest equivalent target to the object orientation from cardinal EE
            cardinal_object_error = float('inf')
            best_cardinal_target_R = None
            for R_target_equiv in equivalent_targets:
                error = ExtendedCardinalOrientations.rotation_matrix_distance(
                    R_object_from_cardinal, R_target_equiv
                )
                if error < cardinal_object_error:
                    cardinal_object_error = error
                    best_cardinal_target_R = R_target_equiv
            
            # Always snap to cardinal (prioritize EE cardinal pose over object error)
            R_EE_new = R_EE_cardinal
            R_object_result = R_object_from_cardinal
            object_error = cardinal_object_error
            best_target_R = best_cardinal_target_R
            
            if cardinal_object_error > original_object_error:
                self.get_logger().info(f"  → Snapped EE to cardinal: {EE_cardinal_name} (EE error: {EE_cardinal_dist:.1f}°, object error increased from {original_object_error:.1f}° to {cardinal_object_error:.1f}°)")
            else:
                self.get_logger().info(f"  → Snapped EE to cardinal: {EE_cardinal_name} (EE error: {EE_cardinal_dist:.1f}°, object error: {object_error:.1f}°)")
        
        # Convert to quaternion
        best_quat = R.from_matrix(R_EE_new).as_quat()
        
        # Log the adjustment
        if current_cardinal_name == best_target_cardinal:
            self.get_logger().info(f"  → Targeted adjustment: {current_cardinal_name} → {best_target_cardinal} (same cardinal, {adjustment_angle:.1f}° rotation needed)")
        else:
            self.get_logger().info(f"  → Targeted adjustment: {current_cardinal_name} → {best_target_cardinal} ({adjustment_angle:.1f}° rotation)")
        self.get_logger().info(f"  → Final object error: {object_error:.1f}°")
        
        return (True, best_quat, R_object_result, best_target_R, object_error)
    
    def find_best_cardinal_for_assembly(self, R_object_target_world, R_grasp, fold_data, R_object_current=None, R_EE_current=None):
        """
        Find the cardinal EE orientation that places the OBJECT closest 
        to a valid assembly pose (considering fold symmetry).
        
        Algorithm:
        1. Generate all equivalent target orientations using fold symmetry
        2. If current object is already close to canonical, prefer minimal adjustments:
           - Find closest equivalent target to current object
           - Calculate EE orientation that would achieve that target
           - Find closest cardinal to that EE orientation
        3. Otherwise, for each of 24 extended cardinal EE orientations:
           - Calculate resulting object orientation: R_object = R_EE × R_grasp
           - Find minimum distance to ANY equivalent target
           - Calculate rotation distance from current EE to this cardinal
        4. Return cardinal with best object alignment error, preferring smaller EE rotations
           when object errors are similar (within 5° tolerance)
        """
        # Generate all symmetry-equivalent target orientations
        equivalent_targets = FoldSymmetry.generate_equivalent_target_orientations(
            R_object_target_world, fold_data, self.get_logger()
        )
        
        cardinals = ExtendedCardinalOrientations.get_all_extended_cardinals()
        
        best_cardinal_name = None
        best_cardinal_quat = None
        best_resulting_object_R = None
        best_matched_target_R = None
        best_object_error = float('inf')
        
        # If current object orientation is provided, check if it's already close to canonical
        if R_object_current is not None:
            # Find closest equivalent target to current object
            min_distance_to_current = float('inf')
            closest_target_to_current = None
            for R_target_equiv in equivalent_targets:
                distance = ExtendedCardinalOrientations.rotation_matrix_distance(
                    R_object_current, R_target_equiv
                )
                if distance < min_distance_to_current:
                    min_distance_to_current = distance
                    closest_target_to_current = R_target_equiv
            
            # If current object is already close to canonical (within 15°), prefer minimal adjustment
            if min_distance_to_current < 15.0:
                self.get_logger().info(f"  Current object is close to canonical ({min_distance_to_current:.1f}°). Preferring minimal adjustment...")
                
                # Calculate current EE orientation from current object and grasp
                # R_object_current = R_EE_current @ R_grasp
                # So: R_EE_current = R_object_current @ R_grasp^T
                R_EE_current = R_object_current @ R_grasp.T
                
                # Calculate the minimal adjustment needed for the object: R_adjust_object = R_target @ R_current^T
                R_adjust_object = closest_target_to_current @ R_object_current.T
                
                # Apply this adjustment to the EE: R_EE_new = R_adjust_object @ R_EE_current
                R_EE_desired = R_adjust_object @ R_EE_current
                
                # Find closest cardinal to this desired EE orientation
                EE_cardinal_name, EE_cardinal_quat, EE_cardinal_dist = \
                    ExtendedCardinalOrientations.find_closest_cardinal(R_EE_desired, threshold_deg=180.0)
                
                if EE_cardinal_name is not None:
                    # Check if this cardinal gives acceptable object error
                    R_EE_cardinal = R.from_quat(EE_cardinal_quat).as_matrix()
                    R_object_from_cardinal = R_EE_cardinal @ R_grasp
                    
                    # Find closest equivalent target to this result
                    min_error_for_cardinal = float('inf')
                    best_target_for_cardinal = None
                    for R_target_equiv in equivalent_targets:
                        error = ExtendedCardinalOrientations.rotation_matrix_distance(
                            R_object_from_cardinal, R_target_equiv
                        )
                        if error < min_error_for_cardinal:
                            min_error_for_cardinal = error
                            best_target_for_cardinal = R_target_equiv
                    
                    # If this gives reasonable error, use it
                    if min_error_for_cardinal < 30.0:  # Reasonable threshold
                        self.get_logger().info(f"  → Using minimal adjustment cardinal: {EE_cardinal_name} (object error: {min_error_for_cardinal:.1f}°)")
                        return (EE_cardinal_name, EE_cardinal_quat, R_object_from_cardinal,
                                best_target_for_cardinal, min_error_for_cardinal, None)
        
        self.get_logger().info(f"  Testing {len(cardinals)} cardinals × {len(equivalent_targets)} equivalent targets...")
        
        # Collect all candidates with their errors and EE rotation distances
        candidates = []
        
        for card_name, card_quat in cardinals.items():
            # What object orientation results from this cardinal EE?
            R_EE_cardinal = R.from_quat(card_quat).as_matrix()
            R_object_result = R_EE_cardinal @ R_grasp
            
            # Find closest equivalent target
            min_error_for_cardinal = float('inf')
            best_target_for_cardinal = None
            for R_target_equiv in equivalent_targets:
                error = ExtendedCardinalOrientations.rotation_matrix_distance(
                    R_object_result, R_target_equiv
                )
                if error < min_error_for_cardinal:
                    min_error_for_cardinal = error
                    best_target_for_cardinal = R_target_equiv
            
            # Calculate rotation distance from current EE to this cardinal (if current EE is available)
            ee_rotation_distance = 0.0
            if R_EE_current is not None:
                ee_rotation_distance = ExtendedCardinalOrientations.rotation_matrix_distance(
                    R_EE_current, R_EE_cardinal
                )
            
            candidates.append((
                card_name, card_quat, R_object_result, 
                best_target_for_cardinal, min_error_for_cardinal, ee_rotation_distance
            ))
            
            if min_error_for_cardinal < best_object_error:
                best_object_error = min_error_for_cardinal
                best_cardinal_name = card_name
                best_cardinal_quat = card_quat
                best_resulting_object_R = R_object_result
                best_matched_target_R = best_target_for_cardinal
        
        # Sort candidates: first by object error, then by EE rotation distance (when errors are similar)
        # Use a tolerance of 5° - if two candidates have object errors within 5°, prefer the one with smaller EE rotation
        error_tolerance = 5.0  # degrees
        
        def sort_key(candidate):
            obj_error = candidate[4]
            ee_rotation = candidate[5]
            # Primary sort: object error (rounded to nearest tolerance)
            # Secondary sort: EE rotation distance
            error_bucket = round(obj_error / error_tolerance) * error_tolerance
            return (error_bucket, ee_rotation)
        
        candidates.sort(key=sort_key)
        
        # Update best selection if we have a better candidate (same error but smaller rotation)
        if len(candidates) > 0:
            best_candidate = candidates[0]
            best_object_error = best_candidate[4]
            best_cardinal_name = best_candidate[0]
            best_cardinal_quat = best_candidate[1]
            best_resulting_object_R = best_candidate[2]
            best_matched_target_R = best_candidate[3]
            
            # Log if we're choosing a different cardinal due to smaller rotation
            if R_EE_current is not None and len(candidates) > 1:
                # Check if there are other candidates with similar error
                for i in range(1, min(5, len(candidates))):  # Check top 5 candidates
                    other_candidate = candidates[i]
                    if abs(other_candidate[4] - best_object_error) <= error_tolerance:
                        if other_candidate[5] < best_candidate[5]:
                            # Found a candidate with similar error but smaller rotation
                            self.get_logger().info(
                                f"  → Preferring {other_candidate[0]} over {best_candidate[0]} "
                                f"(object error: {other_candidate[4]:.1f}° vs {best_candidate[4]:.1f}°, "
                                f"EE rotation: {other_candidate[5]:.1f}° vs {best_candidate[5]:.1f}°)"
                            )
                            best_candidate = other_candidate
                            best_cardinal_name = other_candidate[0]
                            best_cardinal_quat = other_candidate[1]
                            best_resulting_object_R = other_candidate[2]
                            best_matched_target_R = other_candidate[3]
                            break
        
        return (best_cardinal_name, best_cardinal_quat, best_resulting_object_R, 
                best_matched_target_R, best_object_error, candidates)
    
    def reorient_for_target(self, object_name, base_name, duration=5.0,
                            current_object_orientation=None, target_base_orientation=None):
        """Reorient EE so OBJECT ends up at a valid assembly pose."""
        
        # Load assembly config based on base_name if not already loaded for this base
        if self.loaded_base_name != base_name:
            self.assembly_config = self.load_assembly_config(base_name=base_name)
            if not self.assembly_config:
                self.get_logger().error(f"Failed to load assembly config for base '{base_name}'")
                return False
        
        self.get_logger().info(f"Reorienting {object_name} relative to {base_name}")
        self.get_logger().info("Mode: Fold Symmetry + 24-Cardinal Snap")
        
        # === Get current EE pose ===
        if self.current_ee_pose is None:
            self.get_logger().error("EE pose not available")
            return False
        ee_position, R_EE_current = self.get_pose_from_msg(self.current_ee_pose)
        
        # === Get current object orientation ===
        if current_object_orientation is not None:
            R_object_current = self.get_rotation_from_quat(current_object_orientation)
            self.get_logger().info(f"  Using provided object orientation")
        else:
            obj_key = object_name if object_name in self.current_poses else f"{object_name}_scaled70"
            if obj_key not in self.current_poses:
                self.get_logger().error(f"Object {object_name} not found")
                return False
            R_object_current = self.get_rotation_from_transform(self.current_poses[obj_key].transform)
        
        # === Log initial object orientation in RPY ===
        initial_obj_rpy = R.from_matrix(R_object_current).as_euler('xyz', degrees=True)
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"📊 INITIAL OBJECT ORIENTATION:")
        self.get_logger().info(f"   Object: {object_name}")
        self.get_logger().info(f"   RPY: [{initial_obj_rpy[0]:.1f}, {initial_obj_rpy[1]:.1f}, {initial_obj_rpy[2]:.1f}] degrees")
        self.get_logger().info("=" * 70)
        
        # === Get base orientation ===
        if target_base_orientation is not None:
            R_base = self.get_rotation_from_quat(target_base_orientation)
            self.get_logger().info(f"  Using provided base orientation")
        else:
            base_key = base_name if base_name in self.current_poses else f"{base_name}_scaled70"
            if base_key not in self.current_poses:
                self.get_logger().error(f"Base {base_name} not found")
                return False
            R_base = self.get_rotation_from_transform(self.current_poses[base_key].transform)
        
        # === Get target orientation from JSON (relative to base, quaternion) ===
        target_quat = self.get_object_target_orientation(object_name)
        if target_quat is None:
            target_quat = self.get_object_target_orientation(f"{object_name}_scaled70")
        if target_quat is None:
            self.get_logger().error(f"No target orientation for {object_name}")
            return False
        
        # === Load fold symmetry ===
        fold_data = FoldSymmetry.load_symmetry_data(object_name, self.symmetry_dir)
        if fold_data is None:
            fold_data = FoldSymmetry.load_symmetry_data(f"{object_name}_scaled70", self.symmetry_dir)
        
        self.get_logger().info("=" * 70)
        if fold_data:
            self.get_logger().info(f"  Loaded fold symmetry for {object_name}:")
            for axis, data in fold_data.get('fold_axes', {}).items():
                fold_count = data.get('fold', 1)
                if fold_count > 1:
                    self.get_logger().info(f"    {axis.upper()}-axis: {fold_count}-fold symmetry")
        else:
            self.get_logger().info("  No fold symmetry data (identity only)")
        
        # === Calculate grasp rotation ===
        # R_grasp = R_EE^T × R_object (object orientation relative to EE frame)
        R_grasp = R_EE_current.T @ R_object_current
        
        # === Transform target to world frame (for logging) ===
        # Use quaternion from JSON directly to avoid gimbal-lock-sensitive conversions.
        R_target_relative = R.from_quat(target_quat).as_matrix()
        R_object_target_world = R_base @ R_target_relative
        
        # === Transform to base-relative frame for cardinal calculation ===
        # Cardinal calculation should be done assuming base is at identity [0,0,0,1]
        # Transform current object and EE to base-relative frame
        R_object_current_base_relative = R_base.T @ R_object_current
        R_EE_current_base_relative = R_base.T @ R_EE_current
        
        # Target in base-relative frame (same as R_target_relative)
        R_object_target_base_relative = R_target_relative
        
        # Log current state
        current_obj_rpy = R.from_matrix(R_object_current).as_euler('xyz', degrees=True)
        target_world_rpy = R.from_matrix(R_object_target_world).as_euler('xyz', degrees=True)
        # For logging: derive the relative target RPY from the quaternion
        target_relative_rpy = R.from_quat(target_quat).as_euler('xyz', degrees=True)
        
        self.get_logger().info(f"  Current object RPY: [{current_obj_rpy[0]:.1f}, {current_obj_rpy[1]:.1f}, {current_obj_rpy[2]:.1f}]")
        self.get_logger().info(
            f"  Target (relative to base, RPY from quat): "
            f"[{target_relative_rpy[0]:.1f}, {target_relative_rpy[1]:.1f}, {target_relative_rpy[2]:.1f}]"
        )
        self.get_logger().info(
            f"  Target (world frame, RPY): "
            f"[{target_world_rpy[0]:.1f}, {target_world_rpy[1]:.1f}, {target_world_rpy[2]:.1f}]"
        )
        
        # === Try cardinal-to-cardinal optimization first (in base-relative frame) ===
        self.get_logger().info("-" * 70)
        self.get_logger().info("  Attempting cardinal-to-cardinal optimization...")
        (optimization_success, best_quat_base_relative, resulting_object_R_base_relative, 
         matched_target_R_base_relative, object_error) = self.compute_cardinal_to_cardinal_adjustment(
            R_object_current_base_relative, R_object_target_base_relative, R_EE_current_base_relative, R_grasp, fold_data
        )
        
        candidates = None  # Will store alternative cardinals if optimization fails
        
        if optimization_success:
            # Transform result back to world frame
            R_EE_result_base_relative = R.from_quat(best_quat_base_relative).as_matrix()
            R_EE_result_world = R_base @ R_EE_result_base_relative
            best_quat = R.from_matrix(R_EE_result_world).as_quat()
            
            # Transform resulting object and matched target back to world frame
            resulting_object_R = R_base @ resulting_object_R_base_relative
            matched_target_R = R_base @ matched_target_R_base_relative
            
            # Find the cardinal name for logging
            best_cardinal_name, _, _ = ExtendedCardinalOrientations.find_closest_cardinal(
                R_EE_result_base_relative, threshold_deg=180.0  # Check in base-relative frame
            )
            if best_cardinal_name is None:
                best_cardinal_name = "computed_adjustment"
            best_cardinal = best_cardinal_name
            best_quat_cardinal = best_quat  # Already transformed to world frame
            # Use the values already computed from optimization
            cardinal_object_error = object_error
            self.get_logger().info("  ✓ Using cardinal-to-cardinal optimization")
        else:
            # === Fall back to full search (in base-relative frame) ===
            self.get_logger().info("  → Falling back to full cardinal search...")
            (best_cardinal, best_quat_cardinal_base_relative, resulting_object_R_base_relative, 
             matched_target_R_base_relative, object_error, candidates) = self.find_best_cardinal_for_assembly(
                R_object_target_base_relative, R_grasp, fold_data, R_object_current_base_relative, R_EE_current_base_relative
            )
            
            # Transform result back to world frame
            R_EE_cardinal_base_relative = R.from_quat(best_quat_cardinal_base_relative).as_matrix()
            R_EE_cardinal_world = R_base @ R_EE_cardinal_base_relative
            best_quat_cardinal = R.from_matrix(R_EE_cardinal_world).as_quat()
            
            resulting_object_R = R_base @ resulting_object_R_base_relative
            matched_target_R = R_base @ matched_target_R_base_relative
            
            # Transform candidates back to world frame for later use
            if candidates is not None:
                transformed_candidates = []
                for card_name, card_quat_base_rel, card_obj_R_base_rel, card_target_R_base_rel, card_error, ee_rot_dist in candidates:
                    R_EE_cand_base_rel = R.from_quat(card_quat_base_rel).as_matrix()
                    R_EE_cand_world = R_base @ R_EE_cand_base_rel
                    card_quat_world = R.from_matrix(R_EE_cand_world).as_quat()
                    card_obj_R_world = R_base @ card_obj_R_base_rel
                    card_target_R_world = R_base @ card_target_R_base_rel
                    transformed_candidates.append((card_name, card_quat_world, card_obj_R_world, card_target_R_world, card_error, ee_rot_dist))
                candidates = transformed_candidates
            
            cardinal_object_error = object_error
        
        # === Try cardinals with threshold increment (ALWAYS use canonical) ===
        # Try the best cardinal first, then try alternatives if error is too high
        
        # Reset threshold for this reorientation attempt
        self.current_cardinal_error_threshold = self.cardinal_error_threshold_initial
        
        # Try candidates in order (best first)
        candidate_index = 0
        cardinal_found = False
        
        # Prepare candidate list - if we have candidates, use them; otherwise use the best one
        if candidates is not None and len(candidates) > 0:
            candidate_list = candidates
        else:
            # Create a single candidate from the best cardinal (include EE rotation distance as 6th element)
            # Use 0.0 as placeholder since we don't have R_EE_current at this point
            candidate_list = [(best_cardinal, best_quat_cardinal, resulting_object_R, matched_target_R, cardinal_object_error, 0.0)]
        
        while not cardinal_found and candidate_index < len(candidate_list):
            # Get current candidate (now includes EE rotation distance as 6th element)
            card_name, card_quat, card_object_R, card_target_R, card_error, _ = candidate_list[candidate_index]
            
            if candidate_index > 0:
                self.get_logger().info(f"  🔄 Trying alternative cardinal {candidate_index + 1}/{len(candidate_list)}: {card_name} (object error: {card_error:.1f}°)")
            
            # Recalculate object error from this cardinal to ensure consistency
            R_EE_cardinal = R.from_quat(card_quat).as_matrix()
            R_object_from_cardinal = R_EE_cardinal @ R_grasp
            
            # Find the closest equivalent target to the object orientation from cardinal EE
            equivalent_targets = FoldSymmetry.generate_equivalent_target_orientations(
                R_object_target_world, fold_data, None
            )
            cardinal_object_error = float('inf')
            best_cardinal_target_R = None
            for R_target_equiv in equivalent_targets:
                error = ExtendedCardinalOrientations.rotation_matrix_distance(
                    R_object_from_cardinal, R_target_equiv
                )
                if error < cardinal_object_error:
                    cardinal_object_error = error
                    best_cardinal_target_R = R_target_equiv
            
            # Reset threshold for this candidate
            self.current_cardinal_error_threshold = self.cardinal_error_threshold_initial
            
            # Check if object error is acceptable with threshold increment
            while cardinal_object_error > self.current_cardinal_error_threshold:
                if self.current_cardinal_error_threshold < self.cardinal_error_threshold_max:
                    old_threshold = self.current_cardinal_error_threshold
                    self.current_cardinal_error_threshold = min(
                        self.current_cardinal_error_threshold + self.cardinal_error_threshold_increment,
                        self.cardinal_error_threshold_max
                    )
                    self.get_logger().info(f"  🔄 Cardinal object error ({cardinal_object_error:.1f}°) > threshold ({old_threshold:.1f}°), incrementing to {self.current_cardinal_error_threshold:.1f}°...")
                else:
                    # Max threshold reached for this candidate - try next candidate
                    self.get_logger().info(f"  ⚠️ Max threshold reached for {card_name} (object error: {cardinal_object_error:.1f}°). Trying next candidate...")
                    break
            
            # Check if this candidate is acceptable
            if cardinal_object_error <= self.current_cardinal_error_threshold:
                # Found acceptable cardinal
                best_cardinal = card_name
                best_quat_cardinal = card_quat
                resulting_object_R = R_object_from_cardinal
                matched_target_R = best_cardinal_target_R
                cardinal_found = True
            else:
                # This candidate not acceptable, try next one
                candidate_index += 1
        
        if not cardinal_found:
            # All candidates exhausted - use the best one anyway (always use canonical)
            self.get_logger().warn(f"  ⚠️ All {len(candidate_list)} candidates exhausted. Using best cardinal anyway (object error: {cardinal_object_error:.1f}°)")
            best_cardinal = candidate_list[0][0]
            best_quat_cardinal = candidate_list[0][1]
            R_EE_cardinal = R.from_quat(best_quat_cardinal).as_matrix()
            R_object_from_cardinal = R_EE_cardinal @ R_grasp
            resulting_object_R = R_object_from_cardinal
            matched_target_R = candidate_list[0][3]
            cardinal_object_error = candidate_list[0][4]
            best_cardinal_target_R = matched_target_R
            self.current_cardinal_error_threshold = self.cardinal_error_threshold_max
        
        # Always use the cardinal (canonical orientation)
        best_quat = best_quat_cardinal
        object_error = cardinal_object_error
        
        if self.current_cardinal_error_threshold > self.cardinal_error_threshold_initial:
            self.get_logger().info(f"  → Using cardinal: {best_cardinal} (object error: {cardinal_object_error:.1f}°, threshold: {self.current_cardinal_error_threshold:.1f}°)")
        else:
            self.get_logger().info(f"  → Using cardinal: {best_cardinal} (object error: {cardinal_object_error:.1f}°)")
        
        # === Log results ===
        resulting_rpy = R.from_matrix(resulting_object_R).as_euler('xyz', degrees=True)
        matched_rpy = R.from_matrix(matched_target_R).as_euler('xyz', degrees=True)
        EE_rpy = R.from_matrix(R.from_quat(best_quat).as_matrix()).as_euler('xyz', degrees=True)
        
        self.get_logger().info("-" * 70)
        self.get_logger().info(f"  RESULT:")
        self.get_logger().info(f"    Best cardinal: {best_cardinal}")
        self.get_logger().info(f"    Cardinal RPY: {ExtendedCardinalOrientations.get_cardinal_rpy(best_cardinal)}")
        self.get_logger().info(f"    Calculated EE RPY: [{EE_rpy[0]:.1f}, {EE_rpy[1]:.1f}, {EE_rpy[2]:.1f}]")
        self.get_logger().info(f"    Matched equivalent target RPY: [{matched_rpy[0]:.1f}, {matched_rpy[1]:.1f}, {matched_rpy[2]:.1f}]")
        self.get_logger().info(f"    Resulting object RPY: [{resulting_rpy[0]:.1f}, {resulting_rpy[1]:.1f}, {resulting_rpy[2]:.1f}]")
        self.get_logger().info(f"    OBJECT alignment error: {object_error:.1f}°")
        self.get_logger().info(f"    EE Position: {ee_position} (unchanged)")
        self.get_logger().info("=" * 70)
        
        # === Redirect EE orientation if facing towards robot base ===
        # Check if EE RPY is approximately (90, 0, 180) and redirect to (90, 0, 0)
        # This prevents the EE from facing towards the robot base
        rpy_tolerance = 5.0  # degrees tolerance
        if (abs(EE_rpy[0] - 90.0) < rpy_tolerance and 
            abs(EE_rpy[1] - 0.0) < rpy_tolerance and 
            abs(EE_rpy[2] - 180.0) < rpy_tolerance):
            self.get_logger().info(f"  🔄 Redirecting EE from (90, 0, 180) to (90, 0, 0) to avoid facing robot base")
            # Create new quaternion from (90, 0, 0) RPY
            R_EE_redirected = R.from_euler('xyz', [90.0, 0.0, 0.0], degrees=True)
            best_quat = R_EE_redirected.as_quat()
            
            # Recalculate resulting object orientation with redirected EE
            R_EE_redirected_matrix = R_EE_redirected.as_matrix()
            resulting_object_R = R_EE_redirected_matrix @ R_grasp
            
            # Find closest equivalent target to verify alignment is still good
            equivalent_targets = FoldSymmetry.generate_equivalent_target_orientations(
                R_object_target_world, fold_data, None
            )
            object_error_redirected = float('inf')
            best_target_R_redirected = None
            for R_target_equiv in equivalent_targets:
                error = ExtendedCardinalOrientations.rotation_matrix_distance(
                    resulting_object_R, R_target_equiv
                )
                if error < object_error_redirected:
                    object_error_redirected = error
                    best_target_R_redirected = R_target_equiv
            
            # Update matched target and error
            matched_target_R = best_target_R_redirected
            object_error = object_error_redirected
            
            # Recalculate RPY for logging
            EE_rpy = [90.0, 0.0, 0.0]
            resulting_rpy = R.from_matrix(resulting_object_R).as_euler('xyz', degrees=True)
            matched_rpy = R.from_matrix(matched_target_R).as_euler('xyz', degrees=True)
            
            self.get_logger().info(f"  ✅ Redirected EE RPY: [{EE_rpy[0]:.1f}, {EE_rpy[1]:.1f}, {EE_rpy[2]:.1f}]")
            self.get_logger().info(f"  ✅ Updated object alignment error: {object_error:.1f}°")
        
        # === Check if error is acceptable ===
        if object_error > 30.0:
            self.get_logger().warn(f"⚠️ High alignment error ({object_error:.1f}°) - result may not be ideal")
        
        # === Compute IK ===
        if self.current_joint_angles is None:
            if self.read_current_joint_angles() is None:
                self.get_logger().error("Could not read joint angles")
                return False
        
        # Try IK with best solution first
        joint_angles = self.compute_ik_with_current_seed(ee_position.tolist(), best_quat.tolist())
        
        # If IK fails and we have alternative candidates, try them
        if joint_angles is None and candidates is not None:
            self.get_logger().warn(f"  ⚠️ IK failed for best cardinal '{best_cardinal}'. Trying alternatives...")
            for i, (card_name, card_quat, card_object_R, card_target_R, card_error, _) in enumerate(candidates[1:6], 1):  # Try top 5 alternatives
                # Snap to exact equivalent target
                R_EE_card_exact = card_target_R @ R_grasp.T
                card_quat_exact = R.from_matrix(R_EE_card_exact).as_quat()
                
                # Verify exact result
                R_object_card_exact = R_EE_card_exact @ R_grasp
                card_exact_error = ExtendedCardinalOrientations.rotation_matrix_distance(
                    R_object_card_exact, card_target_R
                )
                
                if card_exact_error < 1.0:
                    card_quat = card_quat_exact
                    card_object_R = R_object_card_exact
                    card_error = card_exact_error
                
                self.get_logger().info(f"  Trying alternative {i}: {card_name} (object error: {card_error:.1f}°)")
                joint_angles = self.compute_ik_with_current_seed(ee_position.tolist(), card_quat.tolist())
                
                if joint_angles is not None:
                    # Update to use this alternative
                    best_cardinal = card_name
                    best_quat = card_quat
                    resulting_object_R = card_object_R
                    matched_target_R = card_target_R
                    object_error = card_error
                    self.get_logger().info(f"  ✅ IK succeeded with alternative: {card_name}")
                    break
        
        if joint_angles is None:
            self.get_logger().error("IK failed for all attempted cardinals")
            return False
        
        # === Execute ===
        trajectory = {"traj1": [{
            "positions": [float(x) for x in joint_angles],
            "velocities": [0.0] * 6,
            "time_from_start": Duration(sec=int(duration))
        }]}
        
        success = self.execute_trajectory(trajectory)
        
        # === Log final object orientation in RPY ===
        if success:
            final_obj_rpy = R.from_matrix(resulting_object_R).as_euler('xyz', degrees=True)
            self.get_logger().info("=" * 70)
            self.get_logger().info(f"📊 FINAL OBJECT ORIENTATION:")
            self.get_logger().info(f"   Object: {object_name}")
            self.get_logger().info(f"   RPY: [{final_obj_rpy[0]:.1f}, {final_obj_rpy[1]:.1f}, {final_obj_rpy[2]:.1f}] degrees")
            self.get_logger().info("=" * 70)
        
        return success
    
    def execute_trajectory(self, trajectory):
        try:
            point = trajectory['traj1'][0]
            
            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names
            
            traj_point = JointTrajectoryPoint()
            traj_point.positions = point['positions']
            traj_point.velocities = [0.0] * 6
            traj_point.time_from_start = point['time_from_start']
            traj_msg.points.append(traj_point)
            
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)
            
            self.trajectory_completed = False
            self.trajectory_success = False
            
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response_callback)
            
            while rclpy.ok() and not self.trajectory_completed:
                rclpy.spin_once(self, timeout_sec=0.1)
            
            return self.trajectory_success
        except Exception as e:
            self.get_logger().error(f"Trajectory error: {e}")
            return False
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.trajectory_completed = True
            self.trajectory_success = False
            return
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result_callback)
    
    def goal_result_callback(self, future):
        result = future.result()
        self.trajectory_success = (result.status == 4)
        self.trajectory_completed = True


def main(args=None):
    parser = argparse.ArgumentParser(description='Reorient for Assembly (Fold Symmetry + 24-Cardinal with Intermediary)')
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: sim (reads from topic) or real (requires orientations)')
    parser.add_argument('--object-name', type=str, required=True,
                       help='Name of the object to reorient')
    parser.add_argument('--base-name', type=str, required=True,
                       help='Name of the base object')
    
    # In real mode, orientations are required; in sim mode, they're optional (read from topic)
    parser.add_argument('--current-object-orientation', type=float, nargs=4, metavar=('X','Y','Z','W'),
                       help='Current object orientation quaternion [x, y, z, w] (required in real mode)')
    parser.add_argument('--target-base-orientation', type=float, nargs=4, metavar=('X','Y','Z','W'),
                       help='Target base orientation quaternion [x, y, z, w] (required in real mode)')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'real':
        if args.current_object_orientation is None or args.target_base_orientation is None:
            parser.error("--current-object-orientation and --target-base-orientation are required in real mode")
    
    rclpy.init()
    node = ReorientForAssembly(mode=args.mode)
    node.action_client.wait_for_server()
    
    try:
        while node.current_ee_pose is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        
        # In sim mode, wait for poses from topic if not provided
        # In real mode, orientations should be provided via arguments
        if args.mode == 'sim' and (args.current_object_orientation is None or args.target_base_orientation is None):
            while not node.current_poses:
                rclpy.spin_once(node, timeout_sec=0.1)
        
        # Default duration is 5.0 seconds
        duration = 5.0
        
        success = node.reorient_for_target(
            args.object_name, args.base_name, duration,
            args.current_object_orientation, args.target_base_orientation
        )
        
        node.get_logger().info("✅ Success!" if success else "❌ Failed")
        
        time.sleep(0.5)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()