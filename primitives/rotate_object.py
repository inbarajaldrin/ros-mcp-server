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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import json
import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import signal
import time


from primitives.shared.ik import ik_objective_quaternion, forward_kinematics, dh_params
from primitives.shared.velocity_profiles import single_point, compute_duration
from utils.data_path_finder import get_assembly_data_dir, get_symmetry_dir, find_assembly_json_by_base_name
from primitives.shared.config import TABLE_HEIGHT, ROTATE_ABOUT_GRIPPER_CENTER, GRIPPER_CENTER_TOOL_OFFSET, DEFAULT_BASE_ORIENTATION
from primitives.shared.collision import compute_all_joint_positions, check_collision_with_table, segment_distance, check_self_collision, check_ee_below_base, check_compact_configuration


def _json_dumps_decimal(obj, **kwargs):
    """json.dumps that outputs decimal notation instead of scientific notation.
    Prevents LLM tokenization issues where e.g. 1e-6 gets corrupted to 16."""
    s = json.dumps(obj, **kwargs)
    parts = re.split(r'("(?:[^"\\]|\\.)*")', s)
    for i, part in enumerate(parts):
        if not part.startswith('"'):
            parts[i] = re.sub(
                r'-?\d+\.?\d*[eE][+-]?\d+',
                lambda m: f'{float(m.group()):.10f}'.rstrip('0').rstrip('.'),
                part
            )
    return ''.join(parts)

def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(_json_dumps_decimal(result))
    print("__END_RESULT_JSON__")


# Quaternion output precision — controls how many decimal places are reported to the LLM.
# Set to 4 to avoid scientific notation (0.000001 → 0.0), or 6 for full precision (triggers SDK bug).
QUAT_DECIMAL_PLACES = 4

# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())
SYMMETRY_DIR = str(get_symmetry_dir())
DEFAULT_OBJECT_TOPIC = "/objects_poses_sim"
DEFAULT_EE_TOPIC = "/tcp_pose_broadcaster/pose"
# DEFAULT_BASE_ORIENTATION is imported from config


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
        # Roll is applied around the tool's Z-axis (body-frame Z), not the fixed X-axis.
        # This ensures all roll variants of a face direction preserve the tool Z direction.
        for face_name, (pitch, yaw) in primary_directions.items():
            R_face = R.from_euler('yz', [pitch, yaw], degrees=True).as_matrix()
            for roll in roll_angles:
                name = f"face_{face_name}_roll{roll}"
                R_tool_roll = R.from_euler('z', roll, degrees=True).as_matrix()
                q = R.from_matrix(R_face @ R_tool_roll).as_quat()
                cardinals[name] = q

        # Add intermediary directions with 2 roll variations each (4 × 2 = 8)
        # Using only 0° and 180° rolls for intermediaries to keep total at 24
        intermediary_rolls = [0, 180]
        for face_name, (pitch, yaw) in intermediary_directions.items():
            R_face = R.from_euler('yz', [pitch, yaw], degrees=True).as_matrix()
            for roll in intermediary_rolls:
                name = f"face_{face_name}_roll{roll}"
                R_tool_roll = R.from_euler('z', roll, degrees=True).as_matrix()
                q = R.from_matrix(R_face @ R_tool_roll).as_quat()
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
    def get_relevant_cardinals(equivalent_targets, R_grasp, top_k=3):
        """
        Filter cardinals to only those geometrically relevant given fold symmetry.

        For each equivalent target, computes the ideal EE orientation:
            R_EE_required = R_target @ R_grasp^T
        Then finds the top_k closest cardinals to each required EE.
        Returns only the unique set of relevant cardinals.

        Math justification: rotation distance is right-invariant, so
        dist(R_EE @ R_grasp, R_target) = dist(R_EE, R_target @ R_grasp^T).
        The cardinal closest to R_target @ R_grasp^T also minimizes object error.
        """
        all_cardinals = ExtendedCardinalOrientations.get_all_extended_cardinals()
        relevant = {}
        R_grasp_inv = R_grasp.T

        for R_target in equivalent_targets:
            R_EE_required = R_target @ R_grasp_inv
            distances = []
            for card_name, card_quat in all_cardinals.items():
                R_cardinal = R.from_quat(card_quat).as_matrix()
                dist = ExtendedCardinalOrientations.rotation_matrix_distance(
                    R_EE_required, R_cardinal
                )
                distances.append((dist, card_name, card_quat))
            distances.sort(key=lambda x: x[0])
            for i in range(min(top_k, len(distances))):
                name = distances[i][1]
                if name not in relevant:
                    relevant[name] = distances[i][2]

        return relevant

    @staticmethod
    def get_cardinal_rpy(name):
        """Extract (roll, pitch, yaw) from a cardinal name string.

        Roll is around the tool's Z-axis (body-frame Z), not the fixed X-axis.
        To reconstruct the rotation matrix, use:
            R_face = R.from_euler('yz', [pitch, yaw], degrees=True).as_matrix()
            R_tool_roll = R.from_euler('z', roll, degrees=True).as_matrix()
            R_cardinal = R_face @ R_tool_roll
        """
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


from primitives.shared.fold_symmetry import load_symmetry_data, equivalent_orientations


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

        # QoS for EE pose and joint states (VOLATILE to match UR driver publisher)
        qos_volatile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.ee_sub = self.create_subscription(PoseStamped, ee_topic, self.ee_callback, qos_volatile)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, qos_volatile)
        
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
        self.tcp_to_gripper_center_offset = GRIPPER_CENTER_TOOL_OFFSET
        self.rotate_about_gripper_center = ROTATE_ABOUT_GRIPPER_CENTER

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
    
    def validate_quaternion(self, quat, name="quaternion"):
        """Validate that quat is a unit quaternion. Returns error string or None."""
        arr = np.asarray(quat, dtype=float)
        if arr.shape != (4,):
            return f"{name} must have 4 components, got {arr.shape}"
        if np.any(np.abs(arr) > 1.0):
            return f"{name} has component(s) outside [-1, 1]: {arr.tolist()}"
        norm_sq = float(np.sum(arr ** 2))
        if abs(norm_sq - 1.0) > 0.02:
            return f"{name} norm² = {norm_sq:.4f} (expected ~1.0): {arr.tolist()}"
        return None

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
    
    def read_current_joint_angles(self):
        self.joint_angles_received = False
        timeout = 0
        while rclpy.ok() and not self.joint_angles_received and timeout < 100:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout += 1
        return self.current_joint_angles.copy() if self.joint_angles_received else None

    def check_ee_facing_robot(self, R_EE, y_threshold=0.5):
        """
        Check if the tool Z-axis points toward +Y in world frame (toward the robot).

        Args:
            R_EE: 3x3 rotation matrix of EE orientation (in world frame)
            y_threshold: Reject if tool_Z[1] > this value (default 0.5 ≈ 60°)

        Returns:
            (facing_robot, tool_z_y): True if tool points toward +Y world
        """
        tool_z = R_EE[:, 2]
        return tool_z[1] > y_threshold, tool_z[1]

    def check_camera_direction(self, R_EE, x_threshold=-0.5):
        """
        Check if the camera body extends toward -X in world frame.

        The camera is mounted along the tool Z-axis. If tool Z has a large
        negative X-component in world frame, the camera body extends toward
        -X world, which is problematic regardless of EE position.

        Args:
            R_EE: 3x3 rotation matrix of EE orientation (in world frame)
            x_threshold: Reject if tool_Z[0] < this value (default -0.5 ≈ 60°)

        Returns:
            (camera_bad, tool_z_x): True if camera direction is problematic
        """
        tool_z = R_EE[:, 2]
        return tool_z[0] < x_threshold, tool_z[0]

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
        joint_positions = compute_all_joint_positions(joint_angles)
        T_ee = forward_kinematics(dh_params, joint_angles)
        ee_pos = T_ee[:3, 3]
        R_ee = T_ee[:3, :3]

        # Compute gripper tip position (24cm along gripper Z-axis from TCP)
        gripper_z_axis = R_ee[:, 2]  # Third column is Z-axis
        gripper_tip = ee_pos + R_ee @ self.tcp_to_gripper_center_offset

        # Minimum safe distance from robot body (gripper radius + small margin)
        gripper_radius = 0.04  # 4cm approximate gripper radius
        safety_margin = 0.02  # 2cm safety margin
        min_safe_distance = gripper_radius + safety_margin

        # Check 1: Gripper tip below table level (definite collision)
        if gripper_tip[2] < TABLE_HEIGHT:
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
        gripper_to_upper_arm_dist = segment_distance(ee_pos, gripper_tip, shoulder_pos, elbow_pos)
        upper_arm_radius = 0.05  # 5cm radius for upper arm

        if gripper_to_upper_arm_dist < upper_arm_radius + min_safe_distance:
            if verbose:
                self.get_logger().warn(
                    f"Extended gripper collision: gripper too close to upper arm "
                    f"(dist: {gripper_to_upper_arm_dist*1000:.1f}mm)"
                )
            return True

        return False

    def _wrist1_quality(self, joint_angles):
        """
        Evaluate how close wrist_1 (joint index 3) is to its current value.
        Returns 0 when wrist_1 matches current, higher values when farther away.
        Uses wrapped distance so ±180° is the worst case.
        """
        w1 = joint_angles[3]
        current_w1 = self.current_joint_angles[3]
        diff = w1 - current_w1
        # Wrap to [-pi, pi]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return abs(diff)

    def _max_joint_change(self, joint_angles):
        """Max wrapped joint change from current state to candidate (radians)."""
        max_change = 0.0
        for i in range(6):
            diff = joint_angles[i] - self.current_joint_angles[i]
            # Wrap to [-pi, pi]
            while diff > np.pi:
                diff -= 2 * np.pi
            while diff < -np.pi:
                diff += 2 * np.pi
            max_change = max(max_change, abs(diff))
        return max_change

    def _min_link_clearance(self, joint_angles):
        """Minimum clearance (meters) between well-separated link capsules.

        Checks link pairs separated by 3+ links in the kinematic chain.
        Pairs separated by exactly 2 (e.g., wrist_1-link vs EE-link) are
        always structurally tight due to short wrist offsets and don't
        indicate a poor configuration choice.

        Returns the smallest (segment_distance - combined_radii).
        Higher values indicate more spread-out configurations.
        """
        link_radii = [0.075, 0.065, 0.055, 0.045, 0.045, 0.040]
        joint_positions = compute_all_joint_positions(joint_angles)
        num_links = len(joint_positions) - 1
        min_clearance = float('inf')
        for i in range(num_links):
            for j in range(i + 3, num_links):
                p1 = np.array(joint_positions[i])
                p2 = np.array(joint_positions[i + 1])
                p3 = np.array(joint_positions[j])
                p4 = np.array(joint_positions[j + 1])
                dist = segment_distance(p1, p2, p3, p4)
                clearance = dist - (link_radii[i] + link_radii[j])
                if clearance < min_clearance:
                    min_clearance = clearance
        return min_clearance

    _collision_log_count = 0  # Class-level counter for verbose logging

    def _check_collision(self, joint_angles):
        """Check all collision criteria. Returns True if collision detected.

        Optimized: computes FK once and short-circuits on first collision.
        """
        # Compute FK once (previously computed 6-8 times across sub-checks)
        joint_positions = compute_all_joint_positions(joint_angles)
        T_ee = forward_kinematics(dh_params, joint_angles)

        # Short-circuit: check cheapest tests first, return on first collision
        # 1. Table collision (cheap: iterate positions, check Z)
        for pos in joint_positions:
            if pos[2] < TABLE_HEIGHT:
                return True

        # 2. EE below base (cheap: single Z check)
        ee_pos = joint_positions[-1]
        if ee_pos[2] < 0.1625:
            return True

        # 3. Compact configuration (cheap: XY distance)
        shoulder_pos = np.array(joint_positions[1])
        wrist2_pos = np.array(joint_positions[5])
        xy_dist = np.linalg.norm(wrist2_pos[:2] - shoulder_pos[:2])
        if xy_dist < 0.20:
            return True

        # 4. EE facing robot (+Y world direction)
        R_ee = T_ee[:3, :3]
        has_ee_facing, _ = self.check_ee_facing_robot(R_ee)
        if has_ee_facing:
            return True

        # 4b. Camera body extending toward -X world
        has_camera_bad, _ = self.check_camera_direction(R_ee)
        if has_camera_bad:
            return True

        # 5. Self collision (expensive: O(n^2) segment distances)
        if check_self_collision(joint_angles):
            return True

        # 6. Extended gripper collision (moderate: segment distance)
        if self.check_extended_gripper_collision(joint_angles):
            return True

        return False

    def compute_ik_with_current_seed(self, target_position, target_quat, max_tries=5, dx=0.001, try_yaw_flip=True):
        from primitives.shared.ik import IKSolverConfig, IKSolver

        ReorientForAssembly._collision_log_count = 0  # Reset for each IK attempt

        target_rot = R.from_quat(target_quat).as_matrix()
        target_pose = np.eye(4)
        # Convert gripper center target to flange target when rotating about gripper center
        if self.rotate_about_gripper_center:
            target_pose[:3, 3] = target_position - target_rot @ GRIPPER_CENTER_TOOL_OFFSET
        else:
            target_pose[:3, 3] = target_position
        target_pose[:3, :3] = target_rot

        # For roll≈±90° targets (gripper sideways), yaw and yaw+180° are equivalent.
        # Generate the 180°-flipped variant to try both.
        yaw_flipped_quat = None
        if try_yaw_flip:
            target_rpy = R.from_quat(target_quat).as_euler('xyz')
            target_roll_deg = np.degrees(target_rpy[0])
            if 60 < abs(target_roll_deg) < 120:  # Roll close to ±90°
                flipped_yaw = target_rpy[2] + np.pi if target_rpy[2] < 0 else target_rpy[2] - np.pi
                yaw_flipped_quat = R.from_euler('xyz', [target_rpy[0], target_rpy[1], flipped_yaw]).as_quat()

        if self.current_joint_angles is None:
            return None

        q_guess = self.current_joint_angles.copy()

        joint_bounds = [
            (-np.pi, np.pi),     # shoulder_pan: full range
            (-np.pi, 0),         # shoulder_lift: negative only (reaching forward/down)
            (0, np.pi),          # elbow: positive only (elbow down)
            (-np.pi, np.pi),     # wrist_1: full range
            (-np.pi, np.pi),     # wrist_2: full range
            (-2*np.pi, 2*np.pi)  # wrist_3: extended range to avoid wrapping
        ]
        solver = IKSolver(IKSolverConfig(early_termination=True, joint_bounds=joint_bounds))
        collision_checker = self._check_collision

        # Collect valid solutions across phases, then pick the best one
        all_solutions = []  # list of (joints, cost)

        def _dedupe_record(solutions):
            """Deduplicate solutions: skip if wrist_2 is within 3deg of existing."""
            for joints, cost in solutions:
                w2 = joints[4]
                duplicate = False
                for existing_joints, _ in all_solutions:
                    if abs(existing_joints[4] - w2) < np.radians(3):
                        duplicate = True
                        break
                if not duplicate:
                    all_solutions.append((joints.copy(), cost))

        # Compute current EE orientation from FK
        T_current = forward_kinematics(dh_params, self.current_joint_angles)
        R_current = T_current[:3, :3]

        # Compute orientation difference (rotation from current to target)
        R_diff = target_rot @ R_current.T
        angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))

        # Check if this is primarily a yaw-only rotation (wrist_2 near +/-90deg)
        wrist_2_near_singular = abs(abs(self.current_joint_angles[4]) - np.pi/2) < np.radians(15)

        # For small orientation changes or yaw-only rotations, try wrist_3-adjusted seed first
        if wrist_2_near_singular and angle_diff < np.radians(90):
            current_yaw = np.arctan2(R_current[1, 0], R_current[0, 0])
            target_yaw = np.arctan2(target_rot[1, 0], target_rot[0, 0])
            yaw_diff = target_yaw - current_yaw
            while yaw_diff > np.pi:
                yaw_diff -= 2 * np.pi
            while yaw_diff < -np.pi:
                yaw_diff += 2 * np.pi

            yaw_adjusted_seed = self.current_joint_angles.copy()
            if self.current_joint_angles[4] < 0:
                yaw_adjusted_seed[5] -= yaw_diff
            else:
                yaw_adjusted_seed[5] += yaw_diff

            # Preserve raw yaw-adjusted seed: evaluate directly without L-BFGS-B.
            # L-BFGS-B may drift to a wrist-flip; this ensures the wrist_3-only
            # solution is always available for _select_best to prefer.
            raw_cost = ik_objective_quaternion(yaw_adjusted_seed, target_pose)
            # Tightened from 0.1 → 0.025 (2026-05-04): the old loose threshold
            # accepted raw seeds with up to ~6° angle error as "solutions".
            # When current EE was off-cardinal by 3-5°, _select_best preferred
            # this raw-seed (= current joints with tilt) and rotate_object
            # produced a no-op trajectory leaving the EE tilted. 0.025 → max
            # 2.5mm position error or ~1.4° angle error.
            if raw_cost < 0.025 and (collision_checker is None or not collision_checker(yaw_adjusted_seed)):
                all_solutions.append((yaw_adjusted_seed.copy(), raw_cost))

            yaw_results = solver.solve_collect(
                seeds=[yaw_adjusted_seed],
                target_pose=target_pose,
                collision_checker=collision_checker,
                perturbations=1,
                dx=dx,
            )
            _dedupe_record(yaw_results)

        # Phase 1: Try current seed with perturbations
        phase1_results = solver.solve_collect(
            seeds=[q_guess],
            target_pose=target_pose,
            collision_checker=collision_checker,
            perturbations=max_tries,
            dx=dx,
        )
        _dedupe_record(phase1_results)

        # Phase 2: Wrist-only fallback seed
        # Data from 45 test runs shows this single seed (w1=-30, w2=90) wins
        # 100% of Phase 2a victories. It keeps the upper arm fixed and only
        # moves wrist joints — essential for face-change transitions (down->forward)
        # where Phase 1 perturbations can't reach the target.
        yaw_rad = np.arctan2(2.0 * (target_quat[3] * target_quat[2] + target_quat[0] * target_quat[1]),
                            1.0 - 2.0 * (target_quat[1]**2 + target_quat[2]**2))
        yaw_deg = np.degrees(yaw_rad)

        curr = self.current_joint_angles if self.current_joint_angles is not None else np.zeros(6)
        curr_deg = np.degrees(curr)

        seeds = [
            np.radians([curr_deg[0], curr_deg[1], curr_deg[2], -30, 90, yaw_deg]),
            np.radians([curr_deg[0], curr_deg[1], curr_deg[2], -30, -90, yaw_deg]),
        ]

        probe_results = solver.solve_collect(
            seeds=seeds,
            target_pose=target_pose,
            collision_checker=collision_checker,
            perturbations=1,
            dx=dx,
        )
        _dedupe_record(probe_results)

        # === Select best solution ===
        MAX_JOINT_CHANGE = np.radians(143)

        def _select_best(solutions):
            """Score and sort solutions: wrist_1 quality, then link clearance, then normal arm.

            Optimization: only compute expensive _min_link_clearance for top 5 candidates
            after initial sorting by cheap metrics (wrist_1 quality, max_change).

            Cost gate (added 2026-05-04): pre-filter to solutions with cost
            < SELECTION_COST_THRESHOLD so we never pick a "solution" that
            doesn't actually match the target pose. Cost = 10*pos_err_m +
            angle_err_rad; threshold 0.025 ≈ max 2.5mm position OR ~1.4° angle.
            Without this, L-BFGS-B converged-but-high-cost results (e.g., the
            seed itself if it's at a local min) would compete with truly-
            accurate solutions, and the wrist_1-quality sort below would
            pick whichever was closest to current joints — yielding a no-op
            trajectory when current EE was off-cardinal.
            """
            if not solutions:
                return None

            SELECTION_COST_THRESHOLD = 0.025
            accurate = [(j, c) for j, c in solutions if c < SELECTION_COST_THRESHOLD]
            if accurate:
                solutions = accurate
            # else: fall through with all solutions; caller may notice the
            # high alignment_error_deg and warn. Avoids hard-failing edge cases.

            # Phase 1: Score by cheap metrics only
            prescored = []
            for joints, cost in solutions:
                w1_quality = self._wrist1_quality(joints)
                max_change = self._max_joint_change(joints)
                normal_arm = max_change < MAX_JOINT_CHANGE
                prescored.append((joints, cost, w1_quality, max_change, normal_arm))

            # Sort by wrist_1 quality and normal_arm (cheap metrics)
            prescored.sort(key=lambda s: (s[2], not s[4]))

            # Phase 2: Only compute clearance for top 5 candidates
            TOP_N_FOR_CLEARANCE = 5
            scored = []
            for i, (joints, cost, w1_quality, max_change, normal_arm) in enumerate(prescored[:TOP_N_FOR_CLEARANCE]):
                min_clearance = self._min_link_clearance(joints)
                scored.append((joints, cost, w1_quality, max_change, normal_arm, min_clearance))

            # Add remaining candidates with default clearance (won't win on clearance tiebreak)
            for joints, cost, w1_quality, max_change, normal_arm in prescored[TOP_N_FOR_CLEARANCE:]:
                scored.append((joints, cost, w1_quality, max_change, normal_arm, 0.0))

            # Final sort: wrist_1 quality, then clearance, then normal arm
            scored.sort(key=lambda s: (s[2], -s[5], not s[4]))
            return scored[0]

        best = _select_best(all_solutions)

        if best is not None:
            w1_deg = np.degrees(best[0][3])
            w1_qual_deg = np.degrees(best[2])
            max_change_deg = np.degrees(best[3])
            clearance_mm = best[5] * 1000

            if len(all_solutions) > 1:
                self.get_logger().info(
                    f"  → IK: {len(all_solutions)} solutions found, selected wrist_1={w1_deg:.1f}° "
                    f"(quality={w1_qual_deg:.0f}° from current, max joint change={max_change_deg:.0f}°, "
                    f"normal_arm={best[4]}, clearance={clearance_mm:.0f}mm)"
                )

            return best[0]

        # If roll≈±90° target failed, try the 180°-flipped yaw variant (equivalent orientation)
        if yaw_flipped_quat is not None:
            flipped_rpy = R.from_quat(yaw_flipped_quat).as_euler('xyz', degrees=True)
            self.get_logger().info(f"  → Trying yaw-flipped target: EE RPY=[{flipped_rpy[0]:.1f}, {flipped_rpy[1]:.1f}, {flipped_rpy[2]:.1f}]")
            # Recursively try with flipped target (disable further flipping to avoid infinite recursion)
            return self.compute_ik_with_current_seed(target_position, yaw_flipped_quat.tolist(), max_tries, dx, try_yaw_flip=False)

        if self.mode == 'sim':
            self.get_logger().error("Motion planning failed: no collision-free joint configuration exists for the reorientation target")
        else:
            self.get_logger().error("Motion planning failed: reorientation target is unreachable with current joint constraints")
        return None

    def compute_ik_with_y_search(self, target_position, target_quat, dy=0.01, y_steps=5):
        """
        Try IK with Y-axis position search (moving forward, away from robot).

        This is used as a fallback when IK fails at the original position,
        particularly for roll≈±90° orientations that may cause self-collision.

        Args:
            target_position: Target EE position [x, y, z]
            target_quat: Target EE orientation as quaternion
            dy: Y-axis search step size (meters), applied as negative (forward)
            y_steps: Number of Y-axis steps to try (1 to y_steps, skips 0)

        Returns:
            (joint_angles, y_offset) if solution found, (None, 0) otherwise
        """
        base_position = np.array(target_position)

        for y_step in range(1, y_steps + 1):
            y_offset = -y_step * dy  # Negative = forward, away from robot base
            offset_position = base_position.copy()
            offset_position[1] += y_offset

            self.get_logger().info(f"  → Trying Y offset: {y_offset*1000:.0f}mm (forward)")

            joint_angles = self.compute_ik_with_current_seed(offset_position.tolist(), target_quat)
            if joint_angles is not None:
                self.get_logger().info(f"  → IK: solution found at Y offset={y_offset*1000:.0f}mm")
                return joint_angles, y_offset

        return None, 0

    def compute_cardinal_to_cardinal_adjustment(self, R_object_current, R_object_target_world,
                                                 R_EE_current, R_grasp, fold_data):
        """
        Disabled optimization - always falls back to find_best_cardinal_for_assembly
        which has proper facing-away preference logic with ee_position.
        """
        return (False, None, None, None, float('inf'))
    
    def find_best_cardinal_for_assembly(self, R_object_target_world, R_grasp, fold_data, R_object_current=None, R_EE_current=None, ee_position=None, R_base=None):
        """
        Find the cardinal EE orientation that places the OBJECT closest
        to a valid assembly pose (considering fold symmetry).

        Algorithm:
        1. Generate all equivalent target orientations using fold symmetry
        2. For each relevant cardinal EE orientation:
           - Calculate resulting object orientation: R_object = R_EE × R_grasp
           - Find minimum distance to ANY equivalent target
           - Calculate rotation distance from current EE to this cardinal
           - Check if EE would face the robot (penalize heavily)
        3. Return sorted candidates list for IK to try, with best alignment first

        Args:
            R_base: If provided, used to transform cardinals to world frame for facing-robot check
        """
        # Generate all symmetry-equivalent target orientations
        equivalent_targets = equivalent_orientations(R_object_target_world, fold_data)

        self.get_logger().info(f"  -> Generated {len(equivalent_targets)} equivalent targets from fold symmetry")
        
        cardinals = ExtendedCardinalOrientations.get_relevant_cardinals(
            equivalent_targets, R_grasp, top_k=3
        )
        self.get_logger().info(
            f"  -> Filtered to {len(cardinals)} relevant cardinals (from 24) based on fold symmetry"
        )

        best_cardinal_name = None
        best_cardinal_quat = None
        best_resulting_object_R = None
        best_matched_target_R = None
        best_object_error = float('inf')

        # Collect all candidates for IK to try (ensures fallback options if first choice fails)
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

            # Check world-frame direction constraints
            facing_robot, _ = self.check_ee_facing_robot(R_EE_cardinal)
            camera_bad, _ = self.check_camera_direction(R_EE_cardinal)

            candidates.append((
                card_name, card_quat, R_object_result,
                best_target_for_cardinal, min_error_for_cardinal, ee_rotation_distance,
                facing_robot, camera_bad
            ))
            
            if min_error_for_cardinal < best_object_error:
                best_object_error = min_error_for_cardinal
                best_cardinal_name = card_name
                best_cardinal_quat = card_quat
                best_resulting_object_R = R_object_result
                best_matched_target_R = best_target_for_cardinal
        
        # Sort candidates: primarily by object error, then prefer facing-away as tiebreaker
        # Facing-away is only a preference among cardinals with SIMILAR object error
        # (within same 5° bucket), not a reason to pick high-error cardinals
        error_tolerance = 5.0  # degrees - bucket size for "similar" errors

        # Detect when facing-away preference should be skipped in favor of
        # minimizing EE rotation. Two cases:
        # 1. Yaw-only rotation (axis aligned with Z) - avoids gimbal lock issues
        # 2. Small rotation (< 10°) on any axis - the current face direction can
        #    always accommodate it (cardinal error threshold is 45°), so changing
        #    face direction would be an unnecessary large movement.
        prefer_minimal_ee_rotation = False
        if R_object_current is not None:
            # Find the closest equivalent target to current object
            closest_target = None
            min_dist = float('inf')
            for R_target_equiv in equivalent_targets:
                dist = ExtendedCardinalOrientations.rotation_matrix_distance(
                    R_object_current, R_target_equiv
                )
                if dist < min_dist:
                    min_dist = dist
                    closest_target = R_target_equiv

            if closest_target is not None:
                # Compute rotation from current to target: R_diff = R_target @ R_current^T
                R_diff = closest_target @ R_object_current.T

                # Extract rotation axis and angle using scipy
                rot_diff = R.from_matrix(R_diff)
                rotvec = rot_diff.as_rotvec()  # axis * angle (radians)
                angle_rad = np.linalg.norm(rotvec)

                if angle_rad > 1e-6:  # Non-trivial rotation
                    axis = rotvec / angle_rad  # Normalized rotation axis

                    # Check if rotation axis is close to Z-axis [0, 0, ±1]
                    z_alignment = abs(axis[2])

                    YAW_AXIS_THRESHOLD = 0.966  # cos(15°)
                    SMALL_ROTATION_THRESHOLD = np.radians(10)
                    if z_alignment > YAW_AXIS_THRESHOLD or angle_rad < SMALL_ROTATION_THRESHOLD:
                        prefer_minimal_ee_rotation = True
                else:
                    # Very small rotation needed - no face change needed
                    prefer_minimal_ee_rotation = True

        def sort_key(candidate):
            obj_error = candidate[4]
            ee_rotation = candidate[5]
            facing_robot = candidate[6] if len(candidate) > 6 else False
            camera_bad = candidate[7] if len(candidate) > 7 else False

            # Reject cardinals with bad world-frame directions (sort to end)
            if facing_robot or camera_bad:
                return (999, 0, ee_rotation)

            # PRIMARY: Object error bucket (low error is best)
            # Round to 1 decimal to avoid floating point boundary issues (e.g., 44.999° vs 45.001°)
            error_bucket = int(round(obj_error, 1) / error_tolerance)

            # SECONDARY: Minimize EE rotation (prefer smallest joint motion)
            return (error_bucket, 0, ee_rotation)

        candidates.sort(key=sort_key)

        # Update best selection if we have a better candidate (same error but smaller rotation)
        if len(candidates) > 0:
            best_candidate = candidates[0]
            best_object_error = best_candidate[4]
            best_cardinal_name = best_candidate[0]
            best_cardinal_quat = best_candidate[1]
            best_resulting_object_R = best_candidate[2]
            best_matched_target_R = best_candidate[3]

            # Note: The sort_key already handles facing-away preference, object error,
            # and EE rotation in the correct priority order. No need for post-sort override.
        
        return (best_cardinal_name, best_cardinal_quat, best_resulting_object_R, 
                best_matched_target_R, best_object_error, candidates)
    
    def check_grasp(self, object_name, radius=0.06):
        """Check if object is within grasp radius of gripper center (sim mode only).

        Returns:
            (True, distance) if grasped, (False, distance) if not
        """
        if object_name not in self.current_poses:
            return False, float('inf')

        transform = self.current_poses[object_name].transform
        object_pos = np.array([transform.translation.x, transform.translation.y, transform.translation.z])

        tcp_pos = np.array([self.current_ee_pose.pose.position.x,
                            self.current_ee_pose.pose.position.y,
                            self.current_ee_pose.pose.position.z])
        tcp_quat = np.array([self.current_ee_pose.pose.orientation.x,
                             self.current_ee_pose.pose.orientation.y,
                             self.current_ee_pose.pose.orientation.z,
                             self.current_ee_pose.pose.orientation.w])
        gripper_center = tcp_pos + R.from_quat(tcp_quat).as_matrix() @ GRIPPER_CENTER_TOOL_OFFSET
        distance = np.linalg.norm(object_pos - gripper_center)
        return distance <= radius, distance

    def reorient_for_target(self, object_name, base_name,
                            current_object_orientation=None):
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

        # Compute IK target position: gripper center or flange depending on config
        if self.rotate_about_gripper_center:
            gc_offset = R_EE_current @ GRIPPER_CENTER_TOOL_OFFSET
            ik_target_position = ee_position + gc_offset
        else:
            ik_target_position = ee_position

        # Store initial EE orientation for JSON output
        initial_ee_quat = R.from_matrix(R_EE_current).as_quat()
        self.initial_ee_orientation_quat = initial_ee_quat
        initial_ee_rpy = R.from_quat(initial_ee_quat).as_euler('xyz', degrees=True)
        self.initial_ee_orientation_rpy_deg = self.canonicalize_euler(initial_ee_rpy)

        # === Validate quaternion inputs ===
        if current_object_orientation is not None:
            err = self.validate_quaternion(current_object_orientation, "current_object_orientation")
            if err:
                self.error_message = err
                self.get_logger().error(self.error_message)
                return False

        # === Get current object orientation ===
        if current_object_orientation is not None:
            R_object_current = self.get_rotation_from_quat(current_object_orientation)
        else:
            if object_name not in self.current_poses:
                self.error_message = f"Object {object_name} not found"
                self.get_logger().error(self.error_message)
                return False
            R_object_current = self.get_rotation_from_transform(self.current_poses[object_name].transform)

        # Store initial object orientation for JSON output
        initial_quat = R.from_matrix(R_object_current).as_quat()
        self.initial_object_orientation_quat = initial_quat
        self.initial_object_orientation_rpy_deg = R.from_quat(initial_quat).as_euler('xyz', degrees=True)

        # === Get base orientation ===
        # Real mode: always use default [0, 0, 0, 1]
        # Sim mode: read from ROS topic
        if self.mode == 'real':
            R_base = self.get_rotation_from_quat(DEFAULT_BASE_ORIENTATION)
            self.get_logger().info(f"Using default base orientation: {DEFAULT_BASE_ORIENTATION}")
        else:
            if base_name not in self.current_poses:
                self.error_message = f"Base {base_name} not found"
                self.get_logger().error(self.error_message)
                return False
            R_base = self.get_rotation_from_transform(self.current_poses[base_name].transform)

        # === Get target orientation from JSON (relative to base, quaternion) ===
        target_quat = self.get_object_target_orientation(object_name)
        if target_quat is None:
            self.error_message = f"No target orientation for {object_name} in assembly config"
            self.get_logger().error(self.error_message)
            return False
        
        # === Load fold symmetry ===
        fold_data = load_symmetry_data(object_name, self.symmetry_dir)
        
        # === Calculate grasp rotation ===
        # R_grasp = R_EE^T × R_object (object orientation relative to EE frame)
        #
        # 2026-05-04 PM bugfix: previously used `R_EE_current` (FK from current
        # joints). That drifts across loop iterations because force-mode
        # compliance + retract leaves the EE at a slightly different orientation
        # than the prior rotate's commanded target. The drift compounds:
        # iter N's R_grasp is recomputed from a stale R_object_current input
        # combined with a drifted R_EE_current, the IK selects a slightly
        # different cardinal, the new resulting_object_R is fed forward as next
        # iter's R_object_current, and so on. After ~10 iterations the IK can
        # snap to a non-canonical fold-equivalent (gimbal-lock case observed
        # 2026-05-04: EE RPY [135,-90,0] instead of [180,0,180]).
        #
        # Fix: use the CANONICAL face-down EE rotation as the reference for
        # R_grasp computation. R_grasp is a geometric constant (where the
        # gripper grabs the part) and must not depend on the noisy FK-derived
        # current EE pose. The canonical face-down quaternion is [0,-1,0,0]
        # which is the operator-verified TCP orientation at start of every
        # ACTIVE phase (see CSV `tcp_qx,tcp_qy,tcp_qz,tcp_qw` columns).
        # IK seed and trajectory generation continue to use the actual
        # R_EE_current; only R_grasp computation is canonicalized.
        R_EE_canonical_face_down = R.from_quat([0.0, -1.0, 0.0, 0.0]).as_matrix()
        R_grasp = R_EE_canonical_face_down.T @ R_object_current
        
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
                    R_EE_cand_base_rel = R.from_quat(card_quat_base_rel).as_matrix()
                    R_EE_cand_world = R_base @ R_EE_cand_base_rel
                    card_quat_world = R.from_matrix(R_EE_cand_world).as_quat()
                    card_obj_R_world = R_base @ card_obj_R_base_rel
                    card_target_R_world = R_base @ card_target_R_base_rel
                    # Recompute direction checks in world frame
                    facing_robot, _ = self.check_ee_facing_robot(R_EE_cand_world)
                    camera_bad, _ = self.check_camera_direction(R_EE_cand_world)
                    transformed_candidates.append((card_name, card_quat_world, card_obj_R_world, card_target_R_world, card_error, ee_rot_dist, facing_robot, camera_bad))
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
            equivalent_targets = equivalent_orientations(R_object_target_world, fold_data)
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

        # === Batched IK search with Y-offset fallback ===
        # Strategy: Process cardinals in batches. For each batch:
        #   1. Try all cardinals in batch at Y=0
        #   2. If all fail, try all cardinals in batch with Y offsets
        #   3. If still fail, move to next batch
        BATCH_SIZE = 3
        y_offset_used = 0.0

        # Ensure we have the full candidate list
        if candidates is None:
            (_, best_quat_cardinal_base_relative, resulting_object_R_base_relative,
             matched_target_R_base_relative, _, candidates) = self.find_best_cardinal_for_assembly(
                R_object_target_base_relative, R_grasp, fold_data, R_object_current_base_relative,
                R_EE_current_base_relative, ee_position=ee_position
            )
            # Transform candidates back to world frame
            if candidates is not None:
                transformed_candidates = []
                for cand in candidates:
                    card_name = cand[0]
                    card_quat_base_rel = cand[1]
                    card_obj_R_base_rel = cand[2]
                    card_target_R_base_rel = cand[3]
                    card_error = cand[4]
                    ee_rot_dist = cand[5]
                    R_EE_cand_base_rel = R.from_quat(card_quat_base_rel).as_matrix()
                    R_EE_cand_world = R_base @ R_EE_cand_base_rel
                    card_quat_world = R.from_matrix(R_EE_cand_world).as_quat()
                    card_obj_R_world = R_base @ card_obj_R_base_rel
                    card_target_R_world = R_base @ card_target_R_base_rel
                    facing_robot, _ = self.check_ee_facing_robot(R_EE_cand_world)
                    camera_bad, _ = self.check_camera_direction(R_EE_cand_world)
                    transformed_candidates.append((card_name, card_quat_world, card_obj_R_world, card_target_R_world, card_error, ee_rot_dist, facing_robot, camera_bad))
                candidates = transformed_candidates

        # Helper to prepare a candidate for IK (snap to exact equivalent target)
        def prepare_candidate(cand):
            card_name, card_quat, card_object_R, card_target_R, card_error = cand[0], cand[1], cand[2], cand[3], cand[4]
            # In real mode, use the clean cardinal quaternion directly.
            # The prepare_candidate exact-snap (card_target_R @ R_grasp.T) would
            # re-introduce noise from the detected object orientation into the EE target.
            if self.mode == 'real':
                return card_name, card_quat, card_object_R, card_target_R, card_error
            R_EE_card_exact = card_target_R @ R_grasp.T
            card_quat_exact = R.from_matrix(R_EE_card_exact).as_quat()
            R_object_card_exact = R_EE_card_exact @ R_grasp
            card_exact_error = ExtendedCardinalOrientations.rotation_matrix_distance(
                R_object_card_exact, card_target_R
            )
            if card_exact_error < 1.0:
                return card_name, card_quat_exact, R_object_card_exact, card_target_R, card_exact_error
            return card_name, card_quat, card_object_R, card_target_R, card_error

        joint_angles = None

        if candidates is not None:
            # Process candidates in batches
            num_candidates = len(candidates)
            batch_start = 0

            while joint_angles is None and batch_start < num_candidates:
                batch_end = min(batch_start + BATCH_SIZE, num_candidates)
                batch = candidates[batch_start:batch_end]
                batch_num = (batch_start // BATCH_SIZE) + 1

                if batch_start > 0:
                    self.get_logger().info(f"  → Trying cardinal batch {batch_num} ({batch_start+1}-{batch_end})")

                # Phase 1: Try all cardinals in this batch at Y=0
                for cand in batch:
                    card_name, card_quat, card_object_R, card_target_R, card_error = prepare_candidate(cand)

                    if batch_start == 0 and cand == batch[0]:
                        # Log the first (best) cardinal
                        target_rpy = R.from_quat(card_quat).as_euler('xyz', degrees=True)
                        self.get_logger().info(f"  → IK target for {card_name}: EE RPY=[{target_rpy[0]:.1f}, {target_rpy[1]:.1f}, {target_rpy[2]:.1f}]")

                    joint_angles = self.compute_ik_with_current_seed(ik_target_position.tolist(), card_quat.tolist())

                    if joint_angles is not None:
                        best_cardinal = card_name
                        best_quat = card_quat
                        resulting_object_R = card_object_R
                        matched_target_R = card_target_R
                        object_error = card_error
                        break

                # Phase 2: If all in batch failed at Y=0, try with Y offsets
                if joint_angles is None:
                    self.get_logger().info(f"  → Batch {batch_num} failed at Y=0, trying with Y offsets")
                    for cand in batch:
                        card_name, card_quat, card_object_R, card_target_R, card_error = prepare_candidate(cand)

                        joint_angles, y_offset_used = self.compute_ik_with_y_search(ik_target_position.tolist(), card_quat.tolist())

                        if joint_angles is not None:
                            best_cardinal = card_name
                            best_quat = card_quat
                            resulting_object_R = card_object_R
                            matched_target_R = card_target_R
                            object_error = card_error
                            break

                # Move to next batch
                batch_start = batch_end

        # If IK succeeded but link clearance is poor, try alternative equivalent
        # targets for a more spread-out configuration.
        CLEARANCE_SEARCH_THRESHOLD = 0.04  # 40mm
        CLEARANCE_HYSTERESIS = 0.002  # 2mm hysteresis to avoid sub-mm triggering
        if joint_angles is not None and self.mode == 'sim':
            primary_clearance = self._min_link_clearance(joint_angles)
            if primary_clearance < (CLEARANCE_SEARCH_THRESHOLD - CLEARANCE_HYSTERESIS):
                self.get_logger().info(
                    f"  → IK clearance={primary_clearance*1000:.0f}mm < "
                    f"{CLEARANCE_SEARCH_THRESHOLD*1000:.0f}mm, searching for spread-out alternative"
                )
                # Generate candidates if we don't have them (optimization path was used)
                if candidates is None:
                    (_, _, _, _, _, candidates) = self.find_best_cardinal_for_assembly(
                        R_object_target_base_relative, R_grasp, fold_data,
                        R_object_current_base_relative, R_EE_current_base_relative,
                        ee_position=ee_position
                    )
                    if candidates is not None:
                        transformed = []
                        for cand in candidates:
                            cn, cq_br, co_br, ct_br, ce = cand[0], cand[1], cand[2], cand[3], cand[4]
                            ed = cand[5]
                            R_cand_w = R_base @ R.from_quat(cq_br).as_matrix()
                            fr, _ = self.check_ee_facing_robot(R_cand_w)
                            cb, _ = self.check_camera_direction(R_cand_w)
                            transformed.append((
                                cn, R.from_matrix(R_cand_w).as_quat(),
                                R_base @ co_br, R_base @ ct_br, ce, ed, fr, cb
                            ))
                        candidates = transformed

                # Try all candidates, pick the one with best clearance
                # Only consider candidates with safe world-frame directions
                if candidates is not None:
                    best_clearance = primary_clearance
                    for cand in candidates:
                        card_name = cand[0]
                        card_quat = cand[1]
                        card_object_R = cand[2]
                        card_target_R = cand[3]
                        card_error = cand[4]
                        card_facing_robot = cand[6] if len(cand) > 6 else False
                        card_camera_bad = cand[7] if len(cand) > 7 else False

                        # Skip candidates with bad world-frame directions
                        if card_facing_robot or card_camera_bad:
                            continue
                        # Snap to exact equivalent target
                        R_EE_card_exact = card_target_R @ R_grasp.T
                        card_quat_exact = R.from_matrix(R_EE_card_exact).as_quat()
                        R_object_card_exact = R_EE_card_exact @ R_grasp
                        card_exact_error = ExtendedCardinalOrientations.rotation_matrix_distance(
                            R_object_card_exact, card_target_R
                        )
                        if card_exact_error < 1.0:
                            card_quat = card_quat_exact
                            card_object_R = R_object_card_exact
                            card_error = card_exact_error

                        alt_joints = self.compute_ik_with_current_seed(
                            ik_target_position.tolist(), card_quat.tolist()
                        )
                        if alt_joints is not None:
                            alt_clearance = self._min_link_clearance(alt_joints)
                            # Only switch if improvement is meaningful (> 5mm) to avoid sub-mm noise
                            CLEARANCE_MIN_IMPROVEMENT = 0.005  # 5mm
                            if alt_clearance > best_clearance + CLEARANCE_MIN_IMPROVEMENT:
                                self.get_logger().info(
                                    f"  → Switching to {card_name}: clearance "
                                    f"{alt_clearance*1000:.0f}mm > {best_clearance*1000:.0f}mm (+{(alt_clearance-best_clearance)*1000:.0f}mm)"
                                )
                                best_clearance = alt_clearance
                                joint_angles = alt_joints
                                best_cardinal = card_name
                                best_quat = card_quat
                                resulting_object_R = card_object_R
                                matched_target_R = card_target_R
                                object_error = card_error

        if joint_angles is None:
            self.error_message = "Motion planning failed: no valid end-effector orientation for the reorientation could be solved"
            self.get_logger().error(self.error_message)
            return False

        # Log final EE orientation (from FK of actual joint angles, not from target quat)
        T_final = forward_kinematics(dh_params, joint_angles)
        R_final_ee = T_final[:3, :3]
        final_ee_quat = R.from_matrix(R_final_ee).as_quat()
        final_ee_rpy = R.from_quat(final_ee_quat).as_euler('xyz', degrees=True)
        self.get_logger().info(f"Final EE orientation (RPY, degrees): [{final_ee_rpy[0]:.1f}, {final_ee_rpy[1]:.1f}, {final_ee_rpy[2]:.1f}]")

        # Also update the expected object orientation based on actual EE pose
        resulting_object_R = R_final_ee @ R_grasp
        best_quat = final_ee_quat  # Update for later use

        # === Execute with single target point (no joint wrapping — let controller choose path) ===
        joint_dist = float(np.max(np.abs(np.array(joint_angles) - np.array(self.current_joint_angles))))
        duration = compute_duration(joint_distance=joint_dist, profile='s_curve')
        self.get_logger().info(f"Duration: {duration:.2f}s (joint={joint_dist:.2f}rad)")
        positions, velocities, t = single_point(joint_angles, duration)[0]
        trajectory = {"traj1": [{
            "positions": positions,
            "velocities": velocities,
            "time_from_start": Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
        }]}
        
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

            _traj_t0 = time.time()
            while rclpy.ok() and not self.trajectory_completed:
                if time.time() - _traj_t0 > 60:
                    self.error_message = "Trajectory execution timed out after 60s"
                    self.get_logger().error(self.error_message)
                    return False
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
            if self.initial_object_orientation_quat is not None and self.initial_object_orientation_rpy_deg is not None:
                result["initial_object_orientation"] = {
                    "quat": {
                        "x": round(float(self.initial_object_orientation_quat[0]), QUAT_DECIMAL_PLACES),
                        "y": round(float(self.initial_object_orientation_quat[1]), QUAT_DECIMAL_PLACES),
                        "z": round(float(self.initial_object_orientation_quat[2]), QUAT_DECIMAL_PLACES),
                        "w": round(float(self.initial_object_orientation_quat[3]), QUAT_DECIMAL_PLACES)
                    },
                }
            if self.final_object_orientation_quat is not None and self.final_object_orientation_rpy_deg is not None:
                result["final_object_orientation"] = {
                    "quat": {
                        "x": round(float(self.final_object_orientation_quat[0]), QUAT_DECIMAL_PLACES),
                        "y": round(float(self.final_object_orientation_quat[1]), QUAT_DECIMAL_PLACES),
                        "z": round(float(self.final_object_orientation_quat[2]), QUAT_DECIMAL_PLACES),
                        "w": round(float(self.final_object_orientation_quat[3]), QUAT_DECIMAL_PLACES)
                    },
                }
            # if self.initial_ee_orientation_quat is not None and self.initial_ee_orientation_rpy_deg is not None:
            #     result["initial_end_effector_orientation"] = {
            #         "quat": {
            #             "x": round(float(self.initial_ee_orientation_quat[0]), 6),
            #             "y": round(float(self.initial_ee_orientation_quat[1]), 6),
            #             "z": round(float(self.initial_ee_orientation_quat[2]), 6),
            #             "w": round(float(self.initial_ee_orientation_quat[3]), 6)
            #         },
            #         "rpy": {
            #             "roll": round(float(self.initial_ee_orientation_rpy_deg[0]), 4),
            #             "pitch": round(float(self.initial_ee_orientation_rpy_deg[1]), 4),
            #             "yaw": round(float(self.initial_ee_orientation_rpy_deg[2]), 4)
            #         }
            #     }
            # if self.final_ee_orientation_quat is not None and self.final_ee_orientation_rpy_deg is not None:
            #     result["final_end_effector_orientation"] = {
            #         "quat": {
            #             "x": round(float(self.final_ee_orientation_quat[0]), 6),
            #             "y": round(float(self.final_ee_orientation_quat[1]), 6),
            #             "z": round(float(self.final_ee_orientation_quat[2]), 6),
            #             "w": round(float(self.final_ee_orientation_quat[3]), 6)
            #         },
            #         "rpy": {
            #             "roll": round(float(self.final_ee_orientation_rpy_deg[0]), 4),
            #             "pitch": round(float(self.final_ee_orientation_rpy_deg[1]), 4),
            #             "yaw": round(float(self.final_ee_orientation_rpy_deg[2]), 4)
            #         }
            #     }
        else:
            # Add error message on failure
            if self.error_message:
                result["error"] = self.error_message

        output_result(result)


def run_verify_grasp(object_name, mode):
    """Run verify_grasp.py as a subprocess to check if object is grasped.

    Sim mode: checks object-to-gripper distance.
    Real mode: checks gripper width is valid for the grasp.

    Returns True if grasp is verified, False otherwise.
    """
    import subprocess
    import threading

    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'queries', 'verify_grasp.py')

    cmd = [sys.executable, script_path,
           '--object-name', object_name,
           '--mode', mode]

    if mode == 'real':
        # Use width-only check (just verify gripper is holding something)
        cmd.append('--width-only')

    print(f"[INFO] Verifying grasp for {object_name} ({mode} mode)...")

    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root

    output_lines = []

    def stream_output(pipe):
        for line in iter(pipe.readline, ''):
            line = line.rstrip()
            if line:
                output_lines.append(line)
                print(f"[verify_grasp] {line}")
        pipe.close()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    output_thread = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True)
    output_thread.start()

    try:
        returncode = process.wait(timeout=15)
        output_thread.join(timeout=1.0)
    except subprocess.TimeoutExpired:
        print("[ERROR] verify_grasp timed out")
        process.kill()
        return False

    if returncode != 0:
        print("[ERROR] Grasp verification FAILED - object may not be grasped")
        return False

    print("[INFO] Grasp verification passed")
    return True


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
    parser.add_argument('--skip-verify-grasp', action='store_true',
                       help='Skip internal grasp verification (caller already verified)')
    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == 'real':
        if args.current_object_orientation is None:
            parser.error("--current-object-orientation is required in real mode")
    
    # Ensure finally blocks run on SIGTERM (from timeout command) so output is always produced
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))

    # === Verify grasp before starting (unless caller already did it) ===
    if not args.skip_verify_grasp:
        grasp_verified = run_verify_grasp(
            object_name=args.object_name,
            mode=args.mode
        )
        if not grasp_verified:
            output_result({
                "result": "failure",
                "object_name": args.object_name,
                "base_name": args.base_name,
                "mode": args.mode,
                "movement_type": "rotate_object",
                "error": "Grasp verification failed - object may not be grasped"
            })
            sys.exit(1)

    node = None
    success = False
    try:
        rclpy.init()
        node = ReorientForAssembly(mode=args.mode)
        if not node.action_client.wait_for_server(timeout_sec=15):
            raise RuntimeError("Action server not available after 15s")
        _t0 = time.time()
        while node.current_ee_pose is None:
            if time.time() - _t0 > 10:
                raise RuntimeError("EE pose not received after 10s")
            rclpy.spin_once(node, timeout_sec=0.1)

        # In sim mode, wait for poses from topic if not provided via arguments
        if args.mode == 'sim' and args.current_object_orientation is None:
            _t0 = time.time()
            while not node.current_poses:
                if time.time() - _t0 > 10:
                    raise RuntimeError("Object poses not received after 10s")
                rclpy.spin_once(node, timeout_sec=0.1)

        success = node.reorient_for_target(
            args.object_name, args.base_name,
            args.current_object_orientation
        )

        if success:
            node.get_logger().info("Movement completed successfully")
        else:
            node.get_logger().error("Reorientation failed")

    except KeyboardInterrupt:
        error = "Operation interrupted by user"
    except SystemExit:
        error = "Process terminated (SIGTERM)"
    except Exception as e:
        error = f"Unexpected error: {e}"
    else:
        error = None
    finally:
        # Build result from local vars — always emits JSON regardless of node state
        if node is not None:
            if error and not node.error_message:
                node.error_message = error
            node.output_result_json(success)
            try:
                if node.object_sub is not None:
                    node.destroy_subscription(node.object_sub)
                node.destroy_subscription(node.ee_sub)
                node.destroy_subscription(node.joint_state_sub)
                node.action_client.destroy()
                node.destroy_node()
            except Exception:
                pass
        else:
            result = {
                "result": "success" if success else "failure",
                "object_name": args.object_name,
                "base_name": args.base_name,
                "mode": args.mode,
                "movement_type": "rotate_object",
            }
            if not success:
                result["error"] = error or "ROS connection failed before node initialization"
            output_result(result)

        try:
            rclpy.shutdown()
        except Exception:
            pass
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()