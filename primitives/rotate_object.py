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


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())
SYMMETRY_DIR = str(get_symmetry_dir())
DEFAULT_OBJECT_TOPIC = "/objects_poses_sim"
DEFAULT_EE_TOPIC = "/tcp_pose_broadcaster/pose"
# Default base orientation (used if --use-default-base-orientation is set)
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
        
        equivalent_targets = []
        for i, R_sym in enumerate(symmetry_rotations):
            # Apply symmetry in object frame: R_equiv = R_target × R_sym
            R_equivalent = R_target_world @ R_sym
            equivalent_targets.append(R_equivalent)
        
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

        # TCP to gripper center offset distance (from TCP to gripper center along gripper Z-axis)
        # This matches the offset used in move_to_grasp.py
        # When rotating, we keep the gripper center (where the object is) fixed, not the TCP
        self.tcp_to_gripper_center_offset = 0.24  # 0.24m = 24cm (distance from TCP to gripper center)

        # JSON output tracking
        self.error_message = None
        self.object_name = None
        self.base_name = None
        self.initial_object_orientation_quat = None
        self.initial_object_orientation_rpy_deg = None
        self.final_object_orientation_quat = None
        self.final_object_orientation_rpy_deg = None
        self.initial_ee_orientation_quat = None
        self.initial_ee_orientation_rpy_deg = None
        self.final_ee_orientation_quat = None
        self.final_ee_orientation_rpy_deg = None
        self.target_orientation_quat = None
        self.target_orientation_rpy_deg = None
        self.alignment_error_deg = None

        self.get_logger().info(f"Using {self.mode.upper()} mode")
    
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

    def canonicalize_euler(self, orientation):
        """Canonicalize Euler angles to avoid gimbal lock representation issues.
        When roll is close to ±180° and pitch is close to 0°, normalize to (0, 180, yaw)."""
        roll, pitch, yaw = orientation
        if abs(pitch) < 1 and abs(abs(roll) - 180) < 1:
            return np.array([0.0, 180.0, (yaw % 360) - 180])
        else:
            return orientation

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

    def compute_all_joint_positions(self, joint_angles):
        """
        Compute the 3D positions of all joints given joint angles.
        Returns a list of [x, y, z] positions for each joint.
        """
        # UR5e DH parameters (from ik_solver.py)
        dh_params_local = [
            (0,  0.1625,  0,     np.pi/2),
            (0,  0,      -0.425,  0),
            (0,  0,      -0.3922, 0),
            (0,  0.1333,  0,     np.pi/2),
            (0,  0.0997,  0,    -np.pi/2),
            (0,  0.0996,  0,     0)
        ]

        def dh_transform(theta, d, a, alpha):
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            return np.array([
                [ct, -st * ca,  st * sa, a * ct],
                [st,  ct * ca, -ct * sa, a * st],
                [0,   sa,       ca,      d],
                [0,   0,        0,       1]
            ])

        # Compute cumulative transformations and extract positions
        joint_positions = []
        T = np.eye(4)

        # Add base position (always at origin)
        joint_positions.append(T[:3, 3].copy())

        # Compute position of each joint
        for i, (theta, d, a, alpha) in enumerate(dh_params_local):
            T_i = dh_transform(joint_angles[i] + theta, d, a, alpha)
            T = np.dot(T, T_i)
            joint_positions.append(T[:3, 3].copy())

        return joint_positions

    def check_collision_with_table(self, joint_angles, z_threshold=-0.01, verbose=False):
        """
        Check if any part of the robot (all joints) goes below the table.

        Args:
            joint_angles: Array of 6 joint angles
            z_threshold: Minimum allowed Z position (meters).
                        Default -0.01 means 1cm below table is still allowed.
            verbose: If True, log which joint caused collision

        Returns:
            True if collision detected (any joint below threshold), False otherwise
        """
        joint_positions = self.compute_all_joint_positions(joint_angles)

        for i, pos in enumerate(joint_positions):
            if pos[2] < z_threshold:
                if verbose:
                    self.get_logger().warn(
                        f"Collision detected: Joint {i} at Z={pos[2]*1000:.1f}mm "
                        f"(threshold: {z_threshold*1000:.1f}mm)"
                    )
                return True  # Collision detected

        return False  # No collision

    def segment_distance(self, p1, p2, p3, p4):
        """
        Compute minimum distance between two line segments (p1-p2) and (p3-p4).
        Used for capsule-based collision detection between robot links.

        Args:
            p1, p2: Start and end points of first segment (numpy arrays)
            p3, p4: Start and end points of second segment (numpy arrays)

        Returns:
            Minimum distance between the two segments
        """
        d1 = p2 - p1  # Direction of segment 1
        d2 = p4 - p3  # Direction of segment 2
        r = p1 - p3

        a = np.dot(d1, d1)  # Squared length of segment 1
        e = np.dot(d2, d2)  # Squared length of segment 2
        f = np.dot(d2, r)

        EPSILON = 1e-8

        # Check if both segments are points
        if a < EPSILON and e < EPSILON:
            return np.linalg.norm(p1 - p3)

        # First segment is a point
        if a < EPSILON:
            s = 0.0
            t = np.clip(f / e, 0.0, 1.0)
        else:
            c = np.dot(d1, r)
            # Second segment is a point
            if e < EPSILON:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            else:
                # General case
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

    def check_self_collision(self, joint_angles, verbose=False):
        """
        Check if the robot configuration causes self-collision.
        Models links as capsules and checks distances between non-adjacent links.

        Args:
            joint_angles: Array of 6 joint angles
            verbose: If True, log collision details

        Returns:
            True if self-collision detected, False otherwise
        """
        # UR5e approximate link radii (meters) - conservative estimates
        # These represent the "thickness" of each link for collision purposes
        link_radii = [
            0.075,  # Base (joint 0-1)
            0.065,  # Shoulder to elbow (joint 1-2) - upper arm
            0.055,  # Elbow to wrist1 (joint 2-3) - forearm
            0.045,  # Wrist1 to wrist2 (joint 3-4)
            0.045,  # Wrist2 to wrist3 (joint 4-5)
            0.040,  # Wrist3 to EE (joint 5-6)
        ]

        # Safety margin for collision detection
        safety_margin = 0.01  # 1cm extra margin

        # Get all joint positions
        joint_positions = self.compute_all_joint_positions(joint_angles)

        # Check collisions between non-adjacent links
        # Links are defined by consecutive joint positions
        # Link i connects joint_positions[i] to joint_positions[i+1]
        num_links = len(joint_positions) - 1

        for i in range(num_links):
            for j in range(i + 2, num_links):  # Skip adjacent links (i+1)
                # Get segment endpoints
                p1 = np.array(joint_positions[i])
                p2 = np.array(joint_positions[i + 1])
                p3 = np.array(joint_positions[j])
                p4 = np.array(joint_positions[j + 1])

                # Compute distance between segments
                dist = self.segment_distance(p1, p2, p3, p4)

                # Minimum allowed distance is sum of link radii plus safety margin
                min_dist = link_radii[i] + link_radii[j] + safety_margin

                if dist < min_dist:
                    if verbose:
                        self.get_logger().warn(
                            f"Self-collision detected: Link {i} and Link {j} "
                            f"distance={dist*1000:.1f}mm < threshold={min_dist*1000:.1f}mm"
                        )
                    return True  # Collision detected

        return False  # No collision

    def check_ee_below_base(self, joint_angles, z_threshold=0.1625, verbose=False):
        """
        Check if the end-effector goes below the robot base height.

        Args:
            joint_angles: Array of 6 joint angles
            z_threshold: Minimum allowed EE Z position (meters).
                        Default 0.1625 is the robot base height (first DH d parameter).
            verbose: If True, log details

        Returns:
            True if EE is below threshold, False otherwise
        """
        joint_positions = self.compute_all_joint_positions(joint_angles)
        # EE position is the last element
        ee_pos = joint_positions[-1]

        if ee_pos[2] < z_threshold:
            if verbose:
                self.get_logger().warn(
                    f"EE below base: Z={ee_pos[2]*1000:.1f}mm < threshold={z_threshold*1000:.1f}mm"
                )
            return True  # EE too low

        return False  # EE height OK

    def check_compact_configuration(self, joint_angles, min_wrist_shoulder_xy=0.20, verbose=False):
        """
        Check if the robot configuration is too compact (wrist too close to shoulder).

        This detects problematic configurations where the arm is folded back on itself,
        causing the wrist to be physically close to the shoulder/base area even though
        standard link-to-link collision checks pass.

        Args:
            joint_angles: Array of 6 joint angles
            min_wrist_shoulder_xy: Minimum allowed XY distance (meters) between
                                   wrist2 and shoulder. Default 0.20m (200mm).
            verbose: If True, log details

        Returns:
            True if configuration is too compact (should be rejected), False otherwise
        """
        joint_positions = self.compute_all_joint_positions(joint_angles)

        # Shoulder position is at index 1 (after first joint)
        # Wrist2 position is at index 5
        shoulder_pos = np.array(joint_positions[1])
        wrist2_pos = np.array(joint_positions[5])

        # Calculate XY (horizontal) distance between wrist and shoulder
        xy_dist = np.linalg.norm(wrist2_pos[:2] - shoulder_pos[:2])

        if xy_dist < min_wrist_shoulder_xy:
            if verbose:
                self.get_logger().warn(
                    f"Compact configuration detected: wrist-shoulder XY distance="
                    f"{xy_dist*1000:.1f}mm < threshold={min_wrist_shoulder_xy*1000:.1f}mm"
                )
            return True  # Too compact

        return False  # Configuration OK

    def check_ee_facing_robot(self, R_EE, ee_position, threshold_deg=60.0):
        """
        Check if the EE tool face is pointing towards the robot base.

        The tool Z-axis (third column of rotation matrix) indicates where the tool is pointing.
        If this direction points towards the robot base (0, 0, 0), that's problematic
        because subsequent operations may be blocked.

        Args:
            R_EE: 3x3 rotation matrix of EE orientation
            ee_position: [x, y, z] position of EE
            threshold_deg: Angle threshold - if tool direction is within this angle
                          of pointing at robot base (in XY plane), it's considered facing

        Returns:
            (facing_robot, angle_to_base):
                facing_robot: True if tool is facing the robot
                angle_to_base: Angle in degrees between tool XY direction and base direction
        """
        # Tool Z-axis (where the tool is pointing)
        tool_z_axis = R_EE[:, 2]

        # Vector from EE to robot base (in XY plane only - horizontal direction)
        ee_pos = np.array(ee_position)
        vector_to_base_xy = np.array([-ee_pos[0], -ee_pos[1], 0])
        norm_xy = np.linalg.norm(vector_to_base_xy)
        if norm_xy < 1e-6:
            # EE is directly above base, can't determine facing direction
            return False, 90.0
        vector_to_base_xy_norm = vector_to_base_xy / norm_xy

        # Project tool Z-axis to XY plane
        tool_z_xy = np.array([tool_z_axis[0], tool_z_axis[1], 0])
        norm_tool_xy = np.linalg.norm(tool_z_xy)
        if norm_tool_xy < 1e-6:
            # Tool is pointing straight up or down, not facing robot horizontally
            return False, 90.0
        tool_z_xy_norm = tool_z_xy / norm_tool_xy

        # Angle between tool direction (XY) and vector to base (XY)
        dot_product = np.dot(tool_z_xy_norm, vector_to_base_xy_norm)
        angle_to_base = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

        # If angle < threshold, the tool is pointing towards the base
        facing_robot = angle_to_base < threshold_deg

        return facing_robot, angle_to_base

    def check_extended_gripper_collision(self, joint_angles, verbose=False):
        """
        Check if the extended gripper geometry would collide with the robot body.

        The gripper extends 24cm from the TCP along the gripper Z-axis. This method
        checks if that extended geometry would collide with the robot base or upper arm.

        Args:
            joint_angles: Array of 6 joint angles
            verbose: If True, log collision details

        Returns:
            True if collision detected, False otherwise
        """
        # Get joint positions and EE pose from FK
        joint_positions = self.compute_all_joint_positions(joint_angles)
        T_ee = forward_kinematics(dh_params, joint_angles)
        ee_pos = T_ee[:3, 3]
        R_ee = T_ee[:3, :3]

        # Compute gripper tip position (24cm along gripper Z-axis from TCP)
        gripper_z_axis = R_ee[:, 2]  # Third column is Z-axis
        gripper_tip = ee_pos + self.tcp_to_gripper_center_offset * gripper_z_axis

        # Minimum safe distance from robot body (gripper radius + small margin)
        gripper_radius = 0.04  # 4cm approximate gripper radius
        safety_margin = 0.02  # 2cm safety margin
        min_safe_distance = gripper_radius + safety_margin

        # Check 1: Gripper tip below table level (definite collision)
        if gripper_tip[2] < 0.0:
            if verbose:
                self.get_logger().warn(
                    f"Extended gripper collision: tip below table "
                    f"(Z: {gripper_tip[2]*1000:.1f}mm)"
                )
            return True

        # Check 2: Gripper tip inside robot base cylinder (only when very close to base)
        # Base is at origin with ~8cm radius, check only within 20cm height
        tip_xy_dist_to_base = np.linalg.norm(gripper_tip[:2])
        base_column_radius = 0.08  # 8cm actual base radius

        if tip_xy_dist_to_base < base_column_radius + min_safe_distance:
            if gripper_tip[2] < 0.20:  # Only check very close to base (below 20cm)
                if verbose:
                    self.get_logger().warn(
                        f"Extended gripper collision: tip inside base column "
                        f"(XY dist: {tip_xy_dist_to_base*1000:.1f}mm, Z: {gripper_tip[2]*1000:.1f}mm)"
                    )
                return True

        # Check 3: Gripper collision with upper arm (link 1: shoulder to elbow)
        # This is the most likely collision during rotation
        shoulder_pos = np.array(joint_positions[1])
        elbow_pos = np.array(joint_positions[2])

        # Distance from gripper segment (TCP to tip) to upper arm segment
        gripper_to_upper_arm_dist = self.segment_distance(ee_pos, gripper_tip, shoulder_pos, elbow_pos)
        upper_arm_radius = 0.05  # 5cm radius for upper arm

        if gripper_to_upper_arm_dist < upper_arm_radius + min_safe_distance:
            if verbose:
                self.get_logger().warn(
                    f"Extended gripper collision: gripper too close to upper arm "
                    f"(dist: {gripper_to_upper_arm_dist*1000:.1f}mm)"
                )
            return True

        return False

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

        # Compute current EE orientation from FK
        T_current = forward_kinematics(dh_params, self.current_joint_angles)
        R_current = T_current[:3, :3]

        # Compute orientation difference (rotation from current to target)
        R_diff = target_rot @ R_current.T
        angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))

        # Check if this is primarily a yaw-only rotation (wrist_2 near ±90°)
        # When wrist_2 is near ±90°, changing wrist_3 produces yaw rotation
        wrist_2_near_singular = abs(abs(self.current_joint_angles[4]) - np.pi/2) < np.radians(15)

        # For small orientation changes or yaw-only rotations, try wrist_3-adjusted seed first
        if wrist_2_near_singular and angle_diff < np.radians(90):
            # Compute yaw difference between current and target EE
            current_yaw = np.arctan2(R_current[1, 0], R_current[0, 0])
            target_yaw = np.arctan2(target_rot[1, 0], target_rot[0, 0])
            yaw_diff = target_yaw - current_yaw

            # Normalize to [-pi, pi]
            while yaw_diff > np.pi:
                yaw_diff -= 2 * np.pi
            while yaw_diff < -np.pi:
                yaw_diff += 2 * np.pi

            # Create a seed with wrist_3 adjusted for the yaw change
            # The sign relationship depends on wrist_2 sign:
            # - wrist_2 ≈ -90°: EE yaw change = -wrist_3 change → wrist_3_change = -yaw_diff
            # - wrist_2 ≈ +90°: EE yaw change = +wrist_3 change → wrist_3_change = +yaw_diff
            yaw_adjusted_seed = self.current_joint_angles.copy()
            if self.current_joint_angles[4] < 0:
                # wrist_2 ≈ -90°
                yaw_adjusted_seed[5] -= yaw_diff
            else:
                # wrist_2 ≈ +90°
                yaw_adjusted_seed[5] += yaw_diff

            # Try the yaw-adjusted seed
            result = minimize(ik_objective_quaternion, yaw_adjusted_seed, args=(target_pose,),
                            method='L-BFGS-B', bounds=joint_bounds)
            if result.success or ik_objective_quaternion(result.x, target_pose) < 0.01:
                cost = ik_objective_quaternion(result.x, target_pose)

                # Check for collisions (sim mode only)
                has_collision = False
                if self.mode == 'sim':
                    has_table_collision = self.check_collision_with_table(result.x, z_threshold=-0.01)
                    has_self_collision = self.check_self_collision(result.x)
                    has_ee_below_base = self.check_ee_below_base(result.x)
                    has_compact_config = self.check_compact_configuration(result.x)
                    joint_positions = self.compute_all_joint_positions(result.x)
                    ee_pos_from_fk = joint_positions[-1]
                    T_ee = forward_kinematics(dh_params, result.x)
                    R_ee = T_ee[:3, :3]
                    has_ee_facing_robot, _ = self.check_ee_facing_robot(R_ee, ee_pos_from_fk, threshold_deg=60.0)
                    has_extended_gripper_collision = self.check_extended_gripper_collision(result.x)
                    has_collision = has_table_collision or has_self_collision or has_ee_below_base or has_compact_config or has_ee_facing_robot or has_extended_gripper_collision

                if cost < 0.01 and not has_collision:
                    return result.x
                if not has_collision and cost < best_cost:
                    best_cost, best_result = cost, result.x

        for i in range(max_tries):
            perturbed = target_pose.copy()
            perturbed[0, 3] += i * dx

            result = minimize(ik_objective_quaternion, q_guess, args=(perturbed,),
                            method='L-BFGS-B', bounds=joint_bounds)
            if result.success:
                cost = ik_objective_quaternion(result.x, perturbed)

                # Check for collisions (sim mode only)
                has_collision = False
                if self.mode == 'sim':
                    # Check table collision
                    has_table_collision = self.check_collision_with_table(result.x, z_threshold=-0.01)
                    # Check self-collision
                    has_self_collision = self.check_self_collision(result.x)
                    # Check EE below robot base
                    has_ee_below_base = self.check_ee_below_base(result.x)
                    # Check compact configuration
                    has_compact_config = self.check_compact_configuration(result.x)
                    # Check if EE is facing the robot (using FK to get EE pose)
                    joint_positions = self.compute_all_joint_positions(result.x)
                    ee_pos_from_fk = joint_positions[-1]
                    T_ee = forward_kinematics(dh_params, result.x)
                    R_ee = T_ee[:3, :3]
                    has_ee_facing_robot, _ = self.check_ee_facing_robot(R_ee, ee_pos_from_fk, threshold_deg=60.0)
                    # Check extended gripper collision (24cm extension from TCP)
                    has_extended_gripper_collision = self.check_extended_gripper_collision(result.x)
                    has_collision = has_table_collision or has_self_collision or has_ee_below_base or has_compact_config or has_ee_facing_robot or has_extended_gripper_collision

                # Check if this is a good solution (low cost and no collision)
                if cost < 0.01 and not has_collision:
                    return result.x
                # Keep track of best solution (only if no collision)
                if not has_collision and cost < best_cost:
                    best_cost, best_result = cost, result.x

        # If we found any reasonable solution (without collision), use it
        if best_result is not None and best_cost < 0.1:
            return best_result

        # Fallback seeds - use quaternion to extract yaw component without gimbal lock
        # Extract yaw from input quaternion directly (avoids gimbal lock from RPY conversion)
        # Yaw from quaternion: yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        yaw_rad = np.arctan2(2.0 * (target_quat[3] * target_quat[2] + target_quat[0] * target_quat[1]),
                            1.0 - 2.0 * (target_quat[1]**2 + target_quat[2]**2))
        yaw_deg = np.degrees(yaw_rad)

        seeds = [
            # Original seeds with wrist_2 = -90
            np.radians([85, -80, 90, -90, -90, yaw_deg]),
            np.radians([90, -90, 90, -90, -90, yaw_deg]),
            np.radians([0, -90, 90, -90, -90, yaw_deg]),
            np.radians([180, -90, 90, -90, -90, yaw_deg]),
            # Additional seeds with wrist_2 = +90 (to avoid facing robot)
            np.radians([85, -80, 90, -90, 90, yaw_deg]),
            np.radians([90, -90, 90, -90, 90, yaw_deg]),
            np.radians([0, -90, 90, -90, 90, yaw_deg]),
            np.radians([180, -90, 90, -90, 90, yaw_deg]),
            # Seeds with wrist_2 = +75 (closer to user's good config)
            np.radians([85, -108, 106, 2, 76, yaw_deg]),
            np.radians([75, -108, 106, 2, 76, yaw_deg]),
            # Seeds with different wrist_1 configurations
            np.radians([85, -80, 90, 90, -90, yaw_deg]),
            np.radians([85, -80, 90, 90, 90, yaw_deg]),
        ]

        for seed in seeds:
            for i in range(max_tries):
                perturbed = target_pose.copy()
                perturbed[0, 3] += i * dx
                result = minimize(ik_objective_quaternion, seed, args=(perturbed,),
                                method='L-BFGS-B', bounds=joint_bounds)
                if result.success:
                    cost = ik_objective_quaternion(result.x, perturbed)

                    # Check for collisions (sim mode only)
                    has_collision = False
                    if self.mode == 'sim':
                        # Check table collision
                        has_table_collision = self.check_collision_with_table(result.x, z_threshold=-0.01)
                        # Check self-collision
                        has_self_collision = self.check_self_collision(result.x)
                        # Check EE below robot base
                        has_ee_below_base = self.check_ee_below_base(result.x)
                        # Check compact configuration
                        has_compact_config = self.check_compact_configuration(result.x)
                        # Check if EE is facing the robot (using FK to get EE pose)
                        joint_positions = self.compute_all_joint_positions(result.x)
                        ee_pos_from_fk = joint_positions[-1]
                        T_ee = forward_kinematics(dh_params, result.x)
                        R_ee = T_ee[:3, :3]
                        has_ee_facing_robot, _ = self.check_ee_facing_robot(R_ee, ee_pos_from_fk, threshold_deg=60.0)
                        # Check extended gripper collision (24cm extension from TCP)
                        has_extended_gripper_collision = self.check_extended_gripper_collision(result.x)
                        has_collision = has_table_collision or has_self_collision or has_ee_below_base or has_compact_config or has_ee_facing_robot or has_extended_gripper_collision

                    # Check if this is a good solution (low cost and no collision)
                    if cost < 0.01 and not has_collision:
                        return result.x
                    # Keep track of best solution (only if no collision)
                    if not has_collision and cost < best_cost:
                        best_cost, best_result = cost, result.x

        if self.mode == 'sim':
            self.get_logger().error("IK failed: couldn't find collision-free solution (table + self-collision + EE below base + compact config + EE facing robot + extended gripper) even with multiple seeds")
        else:
            self.get_logger().error("IK failed: couldn't find solution even with multiple seeds")
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

        # Generate equivalent targets first
        equivalent_targets = FoldSymmetry.generate_equivalent_target_orientations(
            R_object_target_world, fold_data, None  # Don't log here
        )
        
        # Find the equivalent target that's closest to the CURRENT object orientation
        # AND whose required EE orientation is close to a cardinal
        # This ensures we make the smallest possible adjustment using a cardinal EE
        best_target_R = None
        min_distance_to_current = float('inf')

        for R_target_equiv in equivalent_targets:
            # Calculate the EE required to achieve this target: R_EE = R_target @ R_grasp^T
            R_EE_required = R_target_equiv @ R_grasp.T

            # Check if this required EE is close to a cardinal
            ee_cardinal_name, _, ee_dist = \
                ExtendedCardinalOrientations.find_closest_cardinal(R_EE_required, CARDINAL_THRESHOLD)

            if ee_cardinal_name is not None:
                # This target can be achieved with a cardinal EE
                # Calculate distance from current object to this equivalent target
                distance_to_current = ExtendedCardinalOrientations.rotation_matrix_distance(
                    R_object_current, R_target_equiv
                )

                if distance_to_current < min_distance_to_current:
                    min_distance_to_current = distance_to_current
                    best_target_R = R_target_equiv

        if best_target_R is None:
            # No equivalent target can be achieved with a cardinal EE within threshold
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
        
        # Find nearest cardinal EE orientation - MUST snap to a cardinal
        # Use a reasonable threshold; if no cardinal is close, reject this optimization
        EE_SNAP_THRESHOLD = 45.0  # degrees - if computed EE is > 45° from all cardinals, reject
        EE_cardinal_name, EE_cardinal_quat, EE_cardinal_dist = \
            ExtendedCardinalOrientations.find_closest_cardinal(R_EE_new, threshold_deg=EE_SNAP_THRESHOLD)

        if EE_cardinal_name is None:
            # Computed EE is too far from any cardinal - reject and let full search handle it
            self.get_logger().info(f"  → Computed EE is {EE_cardinal_dist:.1f}° from nearest cardinal (> {EE_SNAP_THRESHOLD}°), falling back to full search")
            return (False, None, None, None, float('inf'))

        # Use the cardinal EE orientation (always snap to cardinal)
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

        # Use the snapped cardinal EE
        R_EE_new = R_EE_cardinal
        R_object_result = R_object_from_cardinal
        object_error = cardinal_object_error
        best_target_R = best_cardinal_target_R

        # Convert to quaternion
        best_quat = R.from_matrix(R_EE_new).as_quat()

        return (True, best_quat, R_object_result, best_target_R, object_error)
    
    def find_best_cardinal_for_assembly(self, R_object_target_world, R_grasp, fold_data, R_object_current=None, R_EE_current=None, ee_position=None, R_base=None):
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
           - Check if EE would face the robot (penalize heavily)
        4. Return cardinal with best object alignment error, preferring smaller EE rotations
           when object errors are similar (within 5° tolerance), avoiding facing-robot configs

        Args:
            R_base: If provided, used to transform cardinals to world frame for facing-robot check
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
            
            # If current object is already close to canonical (within 45°), prefer minimal adjustment
            # This threshold should match CARDINAL_THRESHOLD to ensure consistency
            if min_distance_to_current < 45.0:
                
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
                        return (EE_cardinal_name, EE_cardinal_quat, R_object_from_cardinal,
                                best_target_for_cardinal, min_error_for_cardinal, None)
        
        
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

            # Check if this cardinal would result in EE facing the robot
            facing_robot_penalty = 0.0
            if ee_position is not None:
                facing_robot, angle_to_base = self.check_ee_facing_robot(
                    R_EE_cardinal, ee_position, threshold_deg=60.0
                )
                if facing_robot:
                    # Heavy penalty: 180 - angle_to_base (closer to robot = higher penalty)
                    # This makes facing-robot cardinals much less preferred
                    facing_robot_penalty = 180.0 - angle_to_base

            candidates.append((
                card_name, card_quat, R_object_result,
                best_target_for_cardinal, min_error_for_cardinal, ee_rotation_distance,
                facing_robot_penalty
            ))
            
            if min_error_for_cardinal < best_object_error:
                best_object_error = min_error_for_cardinal
                best_cardinal_name = card_name
                best_cardinal_quat = card_quat
                best_resulting_object_R = R_object_result
                best_matched_target_R = best_target_for_cardinal
        
        # Sort candidates: penalize facing-robot, then by object error, then by EE rotation distance
        # Use a tolerance of 5° - if two candidates have object errors within 5°, prefer the one with smaller EE rotation
        error_tolerance = 5.0  # degrees

        def sort_key(candidate):
            obj_error = candidate[4]
            ee_rotation = candidate[5]
            facing_penalty = candidate[6] if len(candidate) > 6 else 0.0
            # Primary sort: facing-robot penalty (0 = not facing, >0 = facing)
            # Secondary sort: object error (rounded to nearest tolerance)
            # Tertiary sort: EE rotation distance
            facing_bucket = 1 if facing_penalty > 0 else 0  # Binary: facing or not
            error_bucket = round(obj_error / error_tolerance) * error_tolerance
            return (facing_bucket, error_bucket, ee_rotation)

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

        # Store for JSON output
        self.object_name = object_name
        self.base_name = base_name

        # Load assembly config based on base_name if not already loaded for this base
        if self.loaded_base_name != base_name:
            self.assembly_config = self.load_assembly_config(base_name=base_name)
            if not self.assembly_config:
                self.error_message = f"Failed to load assembly config for base '{base_name}'"
                self.get_logger().error(self.error_message)
                return False

        self.get_logger().info(f"Reorienting {object_name} relative to {base_name}")

        # === Get current EE pose ===
        if self.current_ee_pose is None:
            self.error_message = "EE pose data is None"
            self.get_logger().error(self.error_message)
            return False
        ee_position, R_EE_current = self.get_pose_from_msg(self.current_ee_pose)

        # Store initial EE orientation for JSON output
        initial_ee_quat = R.from_matrix(R_EE_current).as_quat()
        self.initial_ee_orientation_quat = initial_ee_quat
        initial_ee_rpy = R.from_quat(initial_ee_quat).as_euler('xyz', degrees=True)
        self.initial_ee_orientation_rpy_deg = self.canonicalize_euler(initial_ee_rpy)

        # === Get current object orientation ===
        if current_object_orientation is not None:
            R_object_current = self.get_rotation_from_quat(current_object_orientation)
        else:
            obj_key = object_name if object_name in self.current_poses else f"{object_name}_scaled70"
            if obj_key not in self.current_poses:
                self.error_message = f"Object {object_name} not found"
                self.get_logger().error(self.error_message)
                return False
            R_object_current = self.get_rotation_from_transform(self.current_poses[obj_key].transform)

        # Store initial object orientation for JSON output
        initial_quat = R.from_matrix(R_object_current).as_quat()
        self.initial_object_orientation_quat = initial_quat
        self.initial_object_orientation_rpy_deg = R.from_quat(initial_quat).as_euler('xyz', degrees=True)

        # === Get base orientation ===
        if target_base_orientation is not None:
            R_base = self.get_rotation_from_quat(target_base_orientation)
        else:
            base_key = base_name if base_name in self.current_poses else f"{base_name}_scaled70"
            if base_key not in self.current_poses:
                self.error_message = f"Base {base_name} not found"
                self.get_logger().error(self.error_message)
                return False
            R_base = self.get_rotation_from_transform(self.current_poses[base_key].transform)

        # === Get target orientation from JSON (relative to base, quaternion) ===
        target_quat = self.get_object_target_orientation(object_name)
        if target_quat is None:
            target_quat = self.get_object_target_orientation(f"{object_name}_scaled70")
        if target_quat is None:
            self.error_message = f"No target orientation for {object_name} in assembly config"
            self.get_logger().error(self.error_message)
            return False
        
        # === Load fold symmetry ===
        fold_data = FoldSymmetry.load_symmetry_data(object_name, self.symmetry_dir)
        if fold_data is None:
            fold_data = FoldSymmetry.load_symmetry_data(f"{object_name}_scaled70", self.symmetry_dir)
        
        # === Calculate grasp rotation ===
        # R_grasp = R_EE^T × R_object (object orientation relative to EE frame)
        R_grasp = R_EE_current.T @ R_object_current
        
        # === Transform target to world frame ===
        # Use quaternion from JSON directly to avoid gimbal-lock-sensitive conversions.
        R_target_relative = R.from_quat(target_quat).as_matrix()
        R_object_target_world = R_base @ R_target_relative

        # Store target orientation for JSON output
        target_quat_world = R.from_matrix(R_object_target_world).as_quat()
        self.target_orientation_quat = target_quat_world
        self.target_orientation_rpy_deg = R.from_quat(target_quat_world).as_euler('xyz', degrees=True)

        # === Transform to base-relative frame for cardinal calculation ===
        # Cardinal calculation should be done assuming base is at identity [0,0,0,1]
        # Transform current object and EE to base-relative frame
        R_object_current_base_relative = R_base.T @ R_object_current
        R_EE_current_base_relative = R_base.T @ R_EE_current
        
        # Target in base-relative frame (same as R_target_relative)
        R_object_target_base_relative = R_target_relative
        
        # Log target object orientation from JSON, then initial object and EE orientations
        target_obj_rpy = R.from_matrix(R_object_target_world).as_euler('xyz', degrees=True)
        initial_obj_rpy = R.from_matrix(R_object_current).as_euler('xyz', degrees=True)
        initial_ee_rpy = R.from_matrix(R_EE_current).as_euler('xyz', degrees=True)
        self.get_logger().info(f"Target object orientation from JSON (RPY, degrees): [{target_obj_rpy[0]:.1f}, {target_obj_rpy[1]:.1f}, {target_obj_rpy[2]:.1f}]")
        self.get_logger().info(f"Initial object orientation (RPY, degrees): [{initial_obj_rpy[0]:.1f}, {initial_obj_rpy[1]:.1f}, {initial_obj_rpy[2]:.1f}]")
        self.get_logger().info(f"Initial EE orientation (RPY, degrees): [{initial_ee_rpy[0]:.1f}, {initial_ee_rpy[1]:.1f}, {initial_ee_rpy[2]:.1f}]")
        
        # === Try cardinal-to-cardinal optimization first (in base-relative frame) ===
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
        else:
            # === Fall back to full search (in base-relative frame) ===
            # Note: ee_position is passed in world frame for facing-robot check
            (best_cardinal, best_quat_cardinal_base_relative, resulting_object_R_base_relative,
             matched_target_R_base_relative, object_error, candidates) = self.find_best_cardinal_for_assembly(
                R_object_target_base_relative, R_grasp, fold_data, R_object_current_base_relative,
                R_EE_current_base_relative, ee_position=ee_position
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
                for cand in candidates:
                    card_name = cand[0]
                    card_quat_base_rel = cand[1]
                    card_obj_R_base_rel = cand[2]
                    card_target_R_base_rel = cand[3]
                    card_error = cand[4]
                    ee_rot_dist = cand[5]
                    facing_penalty = cand[6] if len(cand) > 6 else 0.0
                    R_EE_cand_base_rel = R.from_quat(card_quat_base_rel).as_matrix()
                    R_EE_cand_world = R_base @ R_EE_cand_base_rel
                    card_quat_world = R.from_matrix(R_EE_cand_world).as_quat()
                    card_obj_R_world = R_base @ card_obj_R_base_rel
                    card_target_R_world = R_base @ card_target_R_base_rel
                    transformed_candidates.append((card_name, card_quat_world, card_obj_R_world, card_target_R_world, card_error, ee_rot_dist, facing_penalty))
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
            # Create a single candidate from the best cardinal (7 elements: name, quat, obj_R, target_R, error, ee_rot_dist, facing_penalty)
            # Use 0.0 as placeholder since we don't have R_EE_current at this point
            candidate_list = [(best_cardinal, best_quat_cardinal, resulting_object_R, matched_target_R, cardinal_object_error, 0.0, 0.0)]

        while not cardinal_found and candidate_index < len(candidate_list):
            # Get current candidate (7 elements, ignore last two: ee_rot_dist and facing_penalty)
            cand = candidate_list[candidate_index]
            card_name, card_quat, card_object_R, card_target_R, card_error = cand[0], cand[1], cand[2], cand[3], cand[4]
            
            
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
                else:
                    # Max threshold reached for this candidate - try next candidate
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
        
        # Note: Previous redirect logic for "EE facing robot base" was removed because:
        # 1. Checking only yaw=180° is too simplistic - roll changes the effective direction
        # 2. The cardinal search already finds the best orientation for correct object placement
        # 3. The redirect was causing false positives and unnecessary warnings

        # === Check if error is acceptable ===
        if object_error > 30.0:
            self.get_logger().warn(f"High alignment error ({object_error:.1f}°) - result may not be ideal")
        
        # === Compute IK ===
        if self.current_joint_angles is None:
            if self.read_current_joint_angles() is None:
                self.error_message = "Joint angles data is None"
                self.get_logger().error(self.error_message)
                return False

        # Try IK with best solution first (rotate about TCP, collision checks account for extended gripper)
        joint_angles = self.compute_ik_with_current_seed(ee_position.tolist(), best_quat.tolist())

        # If IK fails and we have alternative candidates, try them
        if joint_angles is None and candidates is not None:
            for i, cand in enumerate(candidates[1:6], 1):  # Try top 5 alternatives
                card_name, card_quat, card_object_R, card_target_R, card_error = cand[0], cand[1], cand[2], cand[3], cand[4]
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

                joint_angles = self.compute_ik_with_current_seed(ee_position.tolist(), card_quat.tolist())

                if joint_angles is not None:
                    # Update to use this alternative
                    best_cardinal = card_name
                    best_quat = card_quat
                    resulting_object_R = card_object_R
                    matched_target_R = card_target_R
                    object_error = card_error
                    break
        
        if joint_angles is None:
            self.error_message = "IK failed: couldn't find valid orientation for reorientation"
            self.get_logger().error(self.error_message)
            return False

        # Log final EE orientation (before execution)
        final_ee_rpy = R.from_quat(best_quat).as_euler('xyz', degrees=True)
        self.get_logger().info(f"Final EE orientation (RPY, degrees): [{final_ee_rpy[0]:.1f}, {final_ee_rpy[1]:.1f}, {final_ee_rpy[2]:.1f}]")

        # === Execute with single target point ===
        trajectory_points = []

        # Get current joint angles as starting point
        start_joints = self.current_joint_angles.copy()
        target_joints = np.array(joint_angles)

        # Handle joint wrapping for wrist_3 (index 5) to take shortest path
        for i in range(6):
            diff = target_joints[i] - start_joints[i]
            # If difference is more than pi, wrap around
            if diff > np.pi:
                target_joints[i] -= 2 * np.pi
            elif diff < -np.pi:
                target_joints[i] += 2 * np.pi

        # Add starting point at t=0
        trajectory_points.append({
            "positions": [float(x) for x in start_joints],
            "velocities": [0.0] * 6,
            "time_from_start": Duration(sec=0, nanosec=0)
        })

        # Add target point at t=duration
        trajectory_points.append({
            "positions": [float(x) for x in target_joints],
            "velocities": [0.0] * 6,
            "time_from_start": Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        })

        trajectory = {"traj1": trajectory_points}
        
        success = self.execute_trajectory(trajectory)

        # Store final orientations and alignment error for JSON output
        if success:
            final_obj_quat = R.from_matrix(resulting_object_R).as_quat()
            self.final_object_orientation_quat = final_obj_quat
            self.final_object_orientation_rpy_deg = R.from_quat(final_obj_quat).as_euler('xyz', degrees=True)
            self.alignment_error_deg = object_error
            final_obj_rpy = self.final_object_orientation_rpy_deg
            self.get_logger().info(f"Final object orientation (RPY, degrees): [{final_obj_rpy[0]:.1f}, {final_obj_rpy[1]:.1f}, {final_obj_rpy[2]:.1f}]")

            # Store final EE orientation (from best_quat which was used for IK)
            self.final_ee_orientation_quat = best_quat
            final_ee_rpy = R.from_quat(best_quat).as_euler('xyz', degrees=True)
            self.final_ee_orientation_rpy_deg = self.canonicalize_euler(final_ee_rpy)

        return success
    
    def execute_trajectory(self, trajectory):
        try:
            points = trajectory['traj1']

            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names

            # Add all trajectory points
            for point in points:
                traj_point = JointTrajectoryPoint()
                traj_point.positions = point['positions']
                traj_point.velocities = point.get('velocities', [0.0] * 6)
                traj_point.time_from_start = point['time_from_start']
                traj_msg.points.append(traj_point)

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)

            self.trajectory_completed = False
            self.trajectory_success = False

            self.get_logger().info(f"Trajectory with {len(points)} waypoints sent and accepted")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response_callback)

            while rclpy.ok() and not self.trajectory_completed:
                rclpy.spin_once(self, timeout_sec=0.1)

            if self.trajectory_success:
                self.get_logger().info("Movement completed successfully")
            # Error message is set in goal_result_callback or goal_response_callback

            return self.trajectory_success
        except Exception as e:
            self.error_message = f"Trajectory execution error: {e}"
            self.get_logger().error(self.error_message)
            return False

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.error_message = "External control program stopped or robot in protective stop"
            self.get_logger().error(self.error_message)
            self.trajectory_completed = True
            self.trajectory_success = False
            return
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        result = future.result()
        self.trajectory_success = (result.status == 4)
        if not self.trajectory_success:
            result_msg = result.result
            if result_msg.error_code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
                self.error_message = "Velocity or acceleration limits exceeded. The required velocity to reach the target exceeds joint velocity limits. Enable robot in URcap to fix this."
            else:
                self.error_message = f"Trajectory failed with status code {result.status}"
            self.get_logger().error(self.error_message)
        self.trajectory_completed = True

    def output_result_json(self, success):
        """Output JSON result with success/failure and orientation data"""
        result = {
            "result": "success" if success else "failure",
            "object_name": self.object_name,
            "base_name": self.base_name,
            "mode": self.mode,
            "movement_type": "rotate_object"
        }

        if success:
            # Add orientation data on success
            if self.initial_object_orientation_quat is not None:
                result["initial_object_orientation"] = {
                    "x": round(float(self.initial_object_orientation_quat[0]), 6),
                    "y": round(float(self.initial_object_orientation_quat[1]), 6),
                    "z": round(float(self.initial_object_orientation_quat[2]), 6),
                    "w": round(float(self.initial_object_orientation_quat[3]), 6)
                }
            if self.initial_object_orientation_rpy_deg is not None:
                result["initial_object_orientation_rpy_deg"] = {
                    "roll": round(float(self.initial_object_orientation_rpy_deg[0]), 4),
                    "pitch": round(float(self.initial_object_orientation_rpy_deg[1]), 4),
                    "yaw": round(float(self.initial_object_orientation_rpy_deg[2]), 4)
                }
            if self.final_object_orientation_quat is not None:
                result["final_object_orientation"] = {
                    "x": round(float(self.final_object_orientation_quat[0]), 6),
                    "y": round(float(self.final_object_orientation_quat[1]), 6),
                    "z": round(float(self.final_object_orientation_quat[2]), 6),
                    "w": round(float(self.final_object_orientation_quat[3]), 6)
                }
            if self.final_object_orientation_rpy_deg is not None:
                result["final_object_orientation_rpy_deg"] = {
                    "roll": round(float(self.final_object_orientation_rpy_deg[0]), 4),
                    "pitch": round(float(self.final_object_orientation_rpy_deg[1]), 4),
                    "yaw": round(float(self.final_object_orientation_rpy_deg[2]), 4)
                }
            if self.initial_ee_orientation_quat is not None:
                result["initial_end_effector_orientation"] = {
                    "x": round(float(self.initial_ee_orientation_quat[0]), 6),
                    "y": round(float(self.initial_ee_orientation_quat[1]), 6),
                    "z": round(float(self.initial_ee_orientation_quat[2]), 6),
                    "w": round(float(self.initial_ee_orientation_quat[3]), 6)
                }
            if self.initial_ee_orientation_rpy_deg is not None:
                result["initial_end_effector_orientation_rpy_deg"] = {
                    "roll": round(float(self.initial_ee_orientation_rpy_deg[0]), 4),
                    "pitch": round(float(self.initial_ee_orientation_rpy_deg[1]), 4),
                    "yaw": round(float(self.initial_ee_orientation_rpy_deg[2]), 4)
                }
            if self.final_ee_orientation_quat is not None:
                result["final_end_effector_orientation"] = {
                    "x": round(float(self.final_ee_orientation_quat[0]), 6),
                    "y": round(float(self.final_ee_orientation_quat[1]), 6),
                    "z": round(float(self.final_ee_orientation_quat[2]), 6),
                    "w": round(float(self.final_ee_orientation_quat[3]), 6)
                }
            if self.final_ee_orientation_rpy_deg is not None:
                result["final_end_effector_orientation_rpy_deg"] = {
                    "roll": round(float(self.final_ee_orientation_rpy_deg[0]), 4),
                    "pitch": round(float(self.final_ee_orientation_rpy_deg[1]), 4),
                    "yaw": round(float(self.final_ee_orientation_rpy_deg[2]), 4)
                }
        else:
            # Add error message on failure
            if self.error_message:
                result["error"] = self.error_message

        output_result(result)


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
                       help='Target base orientation quaternion [x, y, z, w] (required in real mode unless --use-default-base-orientation is used)')
    parser.add_argument('--use-default-base-orientation', action='store_true',
                       help=f'Use default base orientation ({DEFAULT_BASE_ORIENTATION}) (for real mode)')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'real':
        if args.current_object_orientation is None:
            parser.error("--current-object-orientation is required in real mode")
        if not args.use_default_base_orientation and args.target_base_orientation is None:
            parser.error("In real mode, either --target-base-orientation or --use-default-base-orientation must be provided")
        if args.use_default_base_orientation and args.target_base_orientation is not None:
            parser.error("Cannot use both --target-base-orientation and --use-default-base-orientation")
    
    rclpy.init()
    node = ReorientForAssembly(mode=args.mode)
    node.action_client.wait_for_server()

    success = False
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

        # Use default base orientation if flag is set
        target_base_orientation = args.target_base_orientation
        if args.use_default_base_orientation:
            target_base_orientation = DEFAULT_BASE_ORIENTATION
            node.get_logger().info(f"Using default base orientation: {target_base_orientation}")

        success = node.reorient_for_target(
            args.object_name, args.base_name, duration,
            args.current_object_orientation, target_base_orientation
        )

        if success:
            node.get_logger().info("Movement completed successfully")
        else:
            node.get_logger().error("Reorientation failed")

    except KeyboardInterrupt:
        node.error_message = "Operation interrupted by user"
    except Exception as e:
        node.error_message = f"Unexpected error: {e}"
    finally:
        try:
            node.output_result_json(success)
            time.sleep(0.5)
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()