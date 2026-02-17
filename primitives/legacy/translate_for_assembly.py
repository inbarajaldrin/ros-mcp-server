#!/usr/bin/env python3
"""
Translate for Assembly - Step 1: Move to hover position (translation only, no rotation)

The algorithm:
1. Read current base pose and target position from JSON (or accept as arguments)
2. Calculate target EE position to place object at target location
3. Keep current EE orientation unchanged (translation only)
4. Move to hover height (0.25m) above target position

Note: This is step 1 only. Step 2 (moving down to final position+force feedback) is handled separately.
"""

import sys
import os

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped, TransformStamped
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

from primitives.shared.ik import forward_kinematics, dh_params, compute_cartesian_waypoints_ik
from primitives.shared.velocity_profiles import trapezoidal_profile
from utils.data_path_finder import get_assembly_data_dir, get_aruco_data_dir, get_symmetry_dir
from primitives.shared.fold_symmetry import load_symmetry_data, equivalent_orientations
from primitives.rotate_object import ExtendedCardinalOrientations
from primitives.shared.config import TABLE_HEIGHT, GRIPPER_CENTER_TOOL_OFFSET, DEFAULT_BASE_POSITION, DEFAULT_BASE_ORIENTATION
from primitives.shared.collision import compute_all_joint_positions, check_collision_with_table, segment_distance, check_self_collision, check_ee_below_base, check_compact_configuration

# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())
BASE_TOPIC = "/objects_poses_sim"
OBJECT_TOPIC = "/objects_poses_sim"
EE_TOPIC = "/tcp_pose_broadcaster/pose"
HOVER_HEIGHT = 0.15  # Height to hover above base before descending


def output_result(result):
    """Output JSON result with markers for MCP server parsing"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")

# DEFAULT_BASE_POSITION and DEFAULT_BASE_ORIENTATION imported from config


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
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)

            # Check if any component matches the base name
            components = config.get('components', [])
            for component in components:
                comp_name = component.get('name', '')
                if comp_name == base_name:
                    return json_file
        except (json.JSONDecodeError, IOError) as e:
            # Skip invalid JSON files
            if logger:
                logger.debug(f"Skipping invalid JSON file {json_file}: {e}")
            continue
    
    if logger:
        logger.warn(f"No assembly JSON found for base '{base_name}' in {data_dir}")
    return None


def load_grasp_point_position(object_name, grasp_id, logger=None):
    """
    Load grasp point position from grasp points JSON file.

    Args:
        object_name: Object name (e.g., 'fork_orange')
        grasp_id: Integer grasp point ID
        logger: Optional logger

    Returns:
        np.array([x, y, z]) position relative to object CAD center, or None if not found
    """
    data_dir = get_aruco_data_dir() / "grasp_points"
    json_path = data_dir / f"{object_name}_grasp_points.json"
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            for gp in data.get('grasp_points', []):
                if gp['id'] == grasp_id:
                    pos = gp['position']
                    if logger:
                        logger.info(f"Loaded grasp point {grasp_id} for {object_name}: [{pos['x']:.4f}, {pos['y']:.4f}, {pos['z']:.4f}]")
                    return np.array([pos['x'], pos['y'], pos['z']])
            if logger:
                logger.warn(f"Grasp ID {grasp_id} not found in {json_path.name}")
        except (json.JSONDecodeError, IOError, KeyError) as e:
            if logger:
                logger.error(f"Error reading {json_path}: {e}")
    if logger:
        logger.error(f"No grasp points file found for '{object_name}'")
    return None


class TranslateForAssembly(Node):
    def __init__(self, mode=None, base_topic=None, object_topic=None, ee_topic=EE_TOPIC):
        super().__init__('translate_for_assembly')
        
        # Mode must be explicitly specified
        if mode is None:
            raise ValueError("Mode must be explicitly specified. Use 'sim' or 'real'.")
        if mode not in ['sim', 'real']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'sim' or 'real'.")
        
        self.mode = mode  # 'sim' or 'real'

        # Error tracking for JSON output
        self.error_message = None

        # Store object and base names (set during translate call)
        self.object_name = None
        self.base_name = None

        # Load assembly configuration (will be loaded when base_name is available)
        self.assembly_config = {}
        self.assembly_json_file = None
        self.loaded_base_name = None
        
        # Subscribers for pose data
        # In sim mode, subscribe to topics; in real mode, no topic subscriptions needed
        if self.mode == 'sim':
            if base_topic is None:
                base_topic = BASE_TOPIC
            if object_topic is None:
                object_topic = OBJECT_TOPIC
            self.base_sub = self.create_subscription(TFMessage, base_topic, self.base_callback, 10)
            self.object_sub = self.create_subscription(TFMessage, object_topic, self.object_callback, 10)
        else:
            # Real mode: no topic subscriptions
            self.base_sub = None
            self.object_sub = None

        # Configure QoS to match the publisher (VOLATILE durability)
        ee_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,  # Match UR driver publisher QoS
            depth=10
        )

        self.ee_sub = self.create_subscription(PoseStamped, ee_topic, self.ee_callback, ee_qos_profile)
        
        # Subscriber for joint states to get current joint angles (use as IK seed)
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Store current poses
        self.current_poses = {}
        self.current_ee_pose = None
        
        # Current joint angles storage
        self.current_joint_angles = None
        self.joint_angles_received = False
        
        # Action client for trajectory execution
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.action_client = ActionClient(self, FollowJointTrajectory, '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        
        self.get_logger().info(f"Using {self.mode.upper()} mode")
        # self.get_logger().info(f"Assembly config loaded with {len(self.assembly_config.get('components', []))} components")
        # self.get_logger().info(f"Hover height set to: {HOVER_HEIGHT}m")
    
    def load_assembly_config(self, base_name=None):
        """
        Load the assembly configuration from JSON file.
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
        except FileNotFoundError:
            self.get_logger().error(f"Assembly file not found: {json_file}")
            return {}
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Error parsing assembly JSON: {e}")
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
    
    def joint_state_callback(self, msg: JointState):
        """Callback for joint state data"""
        # Extract joint angles in the correct order
        if len(msg.name) == 6 and len(msg.position) == 6:
            joint_dict = dict(zip(msg.name, msg.position))
            # Map joint names to positions in correct order
            ordered_positions = []
            for joint_name in self.joint_names:
                if joint_name in joint_dict:
                    ordered_positions.append(joint_dict[joint_name])
            
            if len(ordered_positions) == 6:
                self.current_joint_angles = np.array(ordered_positions)
                self.joint_angles_received = True
    
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
        """Get target position for object from assembly configuration"""
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
    
    def read_current_joint_angles(self):
        """Read current joint angles using ROS2 subscriber"""
        # self.get_logger().info("Reading current joint angles...")
        
        # Reset the flag
        self.joint_angles_received = False
        
        # Wait for joint angles to arrive (with timeout)
        timeout_count = 0
        max_timeout = 100  # 10 seconds (100 * 0.1s)
        
        while rclpy.ok() and not self.joint_angles_received and timeout_count < max_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout_count += 1
            
            # if timeout_count % 10 == 0:  # Log every second
            #     self.get_logger().info(f"Waiting for joint angles... ({timeout_count * 0.1:.1f}s)")
        
        if not self.joint_angles_received:
            self.get_logger().error("Timeout waiting for joint angles message")
            return None
        
        if self.current_joint_angles is None:
            self.get_logger().error("Joint angles data is None")
            return None
        
        # self.get_logger().info(f"Successfully read joint angles: {self.current_joint_angles}")
        return self.current_joint_angles.copy()

    def compute_ik_with_current_seed(self, target_position, target_quat, max_tries=5, dx=0.001):
        """
        Compute IK using current joint angles as seed.
        Uses unified IK solver with joint-distance tracking for smooth motion.

        Args:
            target_position: [x, y, z] target position
            target_quat: [x, y, z, w] target orientation quaternion
            max_tries: Number of position perturbations to try
            dx: Position perturbation step size

        Returns:
            Joint angles if successful, None otherwise
        """
        from primitives.shared.ik import IKSolverConfig, IKSolver

        target_rotation = R.from_quat(target_quat)
        target_rot_matrix = target_rotation.as_matrix()

        target_pose = np.eye(4)
        target_pose[:3, 3] = target_position
        target_pose[:3, :3] = target_rot_matrix

        if self.current_joint_angles is None:
            self.get_logger().error("Current joint angles not available! Cannot compute IK.")
            return None

        q_guess = self.current_joint_angles.copy()

        # Collision checker for sim mode
        def collision_checker(joint_angles):
            if self.mode != 'sim':
                return False
            return (check_collision_with_table(joint_angles)
                    or check_self_collision(joint_angles)
                    or check_ee_below_base(joint_angles)
                    or check_compact_configuration(joint_angles))

        joint_bounds = [
            (-np.pi, np.pi),     # shoulder_pan
            (-np.pi, np.pi),     # shoulder_lift
            (-np.pi, np.pi),     # elbow
            (-np.pi, np.pi),     # wrist_1
            (-np.pi, np.pi),     # wrist_2
            (-2*np.pi, 2*np.pi)  # wrist_3: extended range to avoid wrapping
        ]
        config = IKSolverConfig(
            joint_bounds=joint_bounds,
        )
        solver = IKSolver(config)

        result = solver.solve(
            seeds=[q_guess],
            target_pose=target_pose,
            collision_checker=collision_checker,
            perturbations=max_tries,
            dx=dx,
        )
        if result is not None:
            return result

        self.get_logger().error("IK failed: couldn't find solution for translate")
        return None
    
    def translate_for_target_sim(self, object_name, base_name, duration=20.0):
        """
        Sim mode: Calculate and execute EE translation to hover position (step 1 only).
        Uses topics to get base and object poses.
        """
        # Store names for JSON output
        self.object_name = object_name
        self.base_name = base_name

        # Load assembly config based on base_name if not already loaded for this base
        if self.loaded_base_name != base_name:
            self.assembly_config = self.load_assembly_config(base_name=base_name)
            if not self.assembly_config:
                self.error_message = f"Failed to load assembly config for base '{base_name}'"
                self.get_logger().error(self.error_message)
                return False

        # Wait for pose data
        if not self.current_poses or self.current_ee_pose is None:
            self.error_message = "No pose data available"
            self.get_logger().error(self.error_message)
            return False

        # Get current EE pose
        if self.current_ee_pose is None:
            self.error_message = "End-effector pose not available"
            self.get_logger().error(self.error_message)
            return False

        # Check if object exists
        if object_name not in self.current_poses:
            self.error_message = f"Object {object_name} not found"
            self.get_logger().error(self.error_message)
            return False

        # Check if base exists
        if base_name not in self.current_poses:
            self.error_message = f"Base {base_name} not found"
            self.get_logger().error(self.error_message)
            return False
        
        # Verify grasp before executing translation
        tcp_pos = np.array([self.current_ee_pose.pose.position.x,
                            self.current_ee_pose.pose.position.y,
                            self.current_ee_pose.pose.position.z])
        tcp_quat = np.array([self.current_ee_pose.pose.orientation.x,
                             self.current_ee_pose.pose.orientation.y,
                             self.current_ee_pose.pose.orientation.z,
                             self.current_ee_pose.pose.orientation.w])
        gripper_center = tcp_pos + R.from_quat(tcp_quat).as_matrix() @ GRIPPER_CENTER_TOOL_OFFSET
        obj_transform = self.current_poses[object_name].transform
        object_pos = np.array([obj_transform.translation.x, obj_transform.translation.y, obj_transform.translation.z])
        grasp_distance = np.linalg.norm(object_pos - gripper_center)
        if grasp_distance > 0.06:
            self.error_message = f"Grasp check failed: {object_name} is {grasp_distance*1000:.1f}mm from gripper center."
            self.get_logger().error(self.error_message)
            return False
        self.get_logger().info(f"Grasp verified: {object_name} is {grasp_distance*1000:.1f}mm from gripper center")

        # Convert poses to matrices
        T_EE_current = self.pose_to_matrix(self.current_ee_pose.pose)
        T_object_current = self.transform_to_matrix(self.current_poses[object_name].transform)
        T_base_current = self.transform_to_matrix(self.current_poses[base_name].transform)

        # Calculate grasp transformation
        T_grasp = np.linalg.inv(T_EE_current) @ T_object_current
        
        # Get current positions
        ee_current_position, ee_current_rpy = self.matrix_to_rpy(T_EE_current)
        base_current_position, base_current_rpy = self.matrix_to_rpy(T_base_current)
        
        # Get target object position from JSON (relative to base)
        target_position_relative = self.get_object_target_position(object_name)
        if target_position_relative is None:
            self.error_message = f"No target position found for {object_name} in JSON"
            self.get_logger().error(self.error_message)
            return False
        
        # Transform target position from base frame to world frame
        R_base_current = T_base_current[:3, :3]
        target_object_position_abs = base_current_position + R_base_current @ target_position_relative
        
        # Create target object transformation (keep current orientation)
        T_object_target = np.eye(4)
        T_object_target[:3, :3] = T_object_current[:3, :3]  # Keep current orientation
        T_object_target[:3, 3] = target_object_position_abs
        
        # Calculate required EE position to place object at target
        T_EE_target = T_object_target @ np.linalg.inv(T_grasp)
        
        # Extract target position and quaternion
        ee_target_position = T_EE_target[:3, 3]
        ee_target_rot_matrix = T_EE_target[:3, :3]
        ee_target_rotation = R.from_matrix(ee_target_rot_matrix)
        ee_target_quat = ee_target_rotation.as_quat()
        
        # Create hover position (same XY as target, but at HOVER_HEIGHT above base)
        # Compute hover for gripper center, then convert to flange position
        hover_gripper_center = ee_target_position.copy()
        hover_gripper_center[2] = base_current_position[2] + HOVER_HEIGHT
        tool_offset_world = ee_target_rot_matrix @ GRIPPER_CENTER_TOOL_OFFSET
        hover_position = hover_gripper_center - tool_offset_world
        self.get_logger().info(f"Hover gripper center Z: {hover_gripper_center[2]:.4f}, hover flange Z: {hover_position[2]:.4f} (offset: {tool_offset_world[2]:.4f})")
        
        # Log final object position
        self.get_logger().info(f"Final object position: [{target_object_position_abs[0]:.4f}, {target_object_position_abs[1]:.4f}, {target_object_position_abs[2]:.4f}]")

        # Read current joint angles before computing IK
        if self.current_joint_angles is None:
            joint_angles = self.read_current_joint_angles()
            if joint_angles is None:
                self.error_message = "Could not read current joint angles"
                self.get_logger().error(self.error_message)
                return False

        # Cartesian-interpolated waypoints with trapezoidal velocity profile
        total_duration = 5.0
        # Use FK flange position (not TCP from topic) so start and target are in the same frame
        T_fk_start = forward_kinematics(dh_params, self.current_joint_angles)
        start_pos = T_fk_start[:3, 3]
        target_pos = hover_position

        dist = np.linalg.norm(target_pos - start_pos)
        num_waypoints = max(2, int(dist / 0.02))  # one waypoint every 20mm

        all_joint_angles = [self.current_joint_angles.copy()]

        for i in range(1, num_waypoints + 1):
            alpha = i / num_waypoints
            waypoint_pos = start_pos + alpha * (target_pos - start_pos)

            waypoint_joint_angles = self.compute_ik_with_current_seed(
                waypoint_pos.tolist(),
                ee_target_quat.tolist(),
                max_tries=5,
                dx=0.001
            )

            if waypoint_joint_angles is None:
                self.error_message = f"IK failed at waypoint {i}/{num_waypoints}"
                self.get_logger().error(self.error_message)
                return False

            self.current_joint_angles = waypoint_joint_angles.copy()
            all_joint_angles.append(np.array([float(x) for x in waypoint_joint_angles]))

        # Trapezoidal velocity profile
        profile = trapezoidal_profile(all_joint_angles, total_duration)
        trajectory_points = []
        for positions, velocities, t_i in profile:
            trajectory_points.append({
                "positions": positions,
                "velocities": velocities,
                "time_from_start": Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
            })

        self.get_logger().info(f"Generated {len(trajectory_points)} Cartesian waypoints with trapezoidal velocity profile")

        success = self.execute_trajectory({"traj1": trajectory_points})
        if not success:
            return False

        return success

    def translate_for_target_real(self, object_name, base_name, duration=20.0,
                            final_base_pos=None, final_base_orientation=None,
                            use_default_base=False, grasp_id=None,
                            object_orientation=None):
        """
        Real mode: Calculate and execute EE translation to hover position (step 1 only).
        Uses provided base position/orientation (no topics).

        Args:
            object_name: Name of the object being held
            base_name: Name of the base object (e.g., 'base')
            duration: Duration for trajectory execution
            final_base_pos: [x, y, z] final base position (required unless use_default_base)
            final_base_orientation: [x, y, z, w] final base orientation quaternion (required unless use_default_base)
            use_default_base: Use default base position/orientation if True
            grasp_id: Grasp point ID (required)
            object_orientation: Current object orientation quaternion [x, y, z, w] (required)
        """
        # Store names for JSON output
        self.object_name = object_name
        self.base_name = base_name

        # Validate required parameters
        if grasp_id is None:
            self.error_message = "grasp_id is required for real mode"
            self.get_logger().error(self.error_message)
            return False
        if object_orientation is None:
            self.error_message = "object_orientation is required for real mode"
            self.get_logger().error(self.error_message)
            return False

        # Load assembly config based on base_name if not already loaded for this base
        if self.loaded_base_name != base_name:
            self.assembly_config = self.load_assembly_config(base_name=base_name)
            if not self.assembly_config:
                self.error_message = f"Failed to load assembly config for base '{base_name}'"
                self.get_logger().error(self.error_message)
                return None

        # Get current EE pose (always needed from topic)
        if self.current_ee_pose is None:
            self.error_message = "End-effector pose not available"
            self.get_logger().error(self.error_message)
            return None

        # Note: We don't need current object position - we calculate EE position directly from target object position

        # Use default base position and orientation only if explicitly requested
        if final_base_pos is None:
            if use_default_base:
                final_base_pos = DEFAULT_BASE_POSITION
                self.get_logger().info(f"Using default base position: {final_base_pos}")
            else:
                self.error_message = "Base position not provided. Use --final-base-pos or --use-default-base flag"
                self.get_logger().error(self.error_message)
                return None
        
        if final_base_orientation is None:
            if use_default_base:
                final_base_orientation = DEFAULT_BASE_ORIENTATION
                self.get_logger().info(f"Using default base orientation: {final_base_orientation}")
            else:
                # Orientation can default to identity if position is provided
                final_base_orientation = [0.0, 0.0, 0.0, 1.0]
                self.get_logger().info(f"Using identity base orientation (not provided)")
        
        # Create base pose from position and orientation
        base_pose = PoseStamped()
        base_pose.pose.position.x = final_base_pos[0]
        base_pose.pose.position.y = final_base_pos[1]
        base_pose.pose.position.z = final_base_pos[2]
        base_pose.pose.orientation.x = final_base_orientation[0]
        base_pose.pose.orientation.y = final_base_orientation[1]
        base_pose.pose.orientation.z = final_base_orientation[2]
        base_pose.pose.orientation.w = final_base_orientation[3]
        T_base_current = self.pose_to_matrix(base_pose.pose)
        self.get_logger().info(f"Using base position: {final_base_pos}, orientation: {final_base_orientation}")
        
        # Convert EE pose to matrix
        T_EE_current = self.pose_to_matrix(self.current_ee_pose.pose)
        
        # Get current EE position (needed for orientation)
        ee_current_position, ee_current_rpy = self.matrix_to_rpy(T_EE_current)
        base_current_position, base_current_rpy = self.matrix_to_rpy(T_base_current)
        
        # Get target object position and orientation from JSON (auto-calculated)
        # Get target object position from JSON (relative to base)
        target_position_relative = self.get_object_target_position(object_name)
        if target_position_relative is None:
            self.error_message = f"No target position found for {object_name} in JSON"
            self.get_logger().error(self.error_message)
            return None
        
        # Get target object orientation from JSON (relative to base)
        target_orientation_relative = self.get_object_target_orientation(object_name)
        if target_orientation_relative is None:
            self.get_logger().warn(f"No target orientation found for {object_name} in JSON, using identity")
            target_orientation_relative = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Transform target position and orientation from base frame to world frame
        R_base_current = T_base_current[:3, :3]
        target_object_position_abs = base_current_position + R_base_current @ target_position_relative
        
        # Transform target orientation from base frame to world frame
        R_target_relative = R.from_quat(target_orientation_relative).as_matrix()
        R_target_abs = R_base_current @ R_target_relative
        target_orientation_abs = R.from_matrix(R_target_abs).as_quat()
        
        self.get_logger().info(f"Target object position (world): {target_object_position_abs}")
        self.get_logger().info(f"Target object orientation (world): {target_orientation_abs}")
        
        self.get_logger().info("Keeping current EE orientation unchanged (from reorient step)")

        # Load grasp point offset
        grasp_offset = load_grasp_point_position(object_name, grasp_id, logger=self.get_logger())
        if grasp_offset is None:
            self.error_message = f"Could not load grasp point {grasp_id} for '{object_name}'"
            self.get_logger().error(self.error_message)
            return None

        # Validate quaternion before using
        quat_array = np.array(object_orientation)
        quat_norm_sq = np.sum(quat_array ** 2)
        if abs(quat_norm_sq - 1.0) > 0.1:
            self.get_logger().error(
                f"Invalid quaternion: norm² = {quat_norm_sq:.2f} (expected ~1.0). "
                f"Values: {quat_array}"
            )
            self.error_message = "Current object orientation quaternion is malformed or corrupted"
            return None

        # Use fold symmetry to snap current object orientation to closest equivalent
        R_object_current = R.from_quat(object_orientation).as_matrix()
        symmetry_dir = str(get_symmetry_dir())
        fold_data = load_symmetry_data(object_name, symmetry_dir)

        if fold_data is not None:
            equivalents = equivalent_orientations(R_target_abs, fold_data)
            best_pos_error = float('inf')
            best_orientation_error = float('inf')
            R_grasp_rotation = R_target_abs  # fallback
            for R_eq in equivalents:
                orientation_error = ExtendedCardinalOrientations.rotation_matrix_distance(R_object_current, R_eq)
                grasp_world_offset_candidate = R_eq @ grasp_offset
                pos_error = np.linalg.norm(grasp_world_offset_candidate - (R_object_current @ grasp_offset))
                if pos_error < best_pos_error or (pos_error == best_pos_error and orientation_error < best_orientation_error):
                    best_pos_error = pos_error
                    best_orientation_error = orientation_error
                    R_grasp_rotation = R_eq
            self.get_logger().info(f"Snapped object orientation to closest equivalent (angle error: {np.degrees(best_orientation_error):.1f}°, position error: {best_pos_error*1000:.2f}mm)")
        else:
            self.get_logger().info("No symmetry data found, using current object orientation directly")
            R_grasp_rotation = R_object_current

        grasp_world_offset = R_grasp_rotation @ grasp_offset
        self.get_logger().info(f"Grasp point {grasp_id} offset (CAD frame): {grasp_offset}")
        self.get_logger().info(f"Grasp point offset (world frame): {grasp_world_offset}")

        # Compute target gripper center position (no flange offset needed)
        target_gripper_center = target_object_position_abs + grasp_world_offset

        # Hover position: same XY as target gripper center, Z = base + HOVER_HEIGHT
        hover_gripper_center = target_gripper_center.copy()
        hover_gripper_center[2] = base_current_position[2] + HOVER_HEIGHT

        self.get_logger().info(f"Target gripper center: {target_gripper_center}")
        self.get_logger().info(f"Hover gripper center (with {HOVER_HEIGHT}m height offset): {hover_gripper_center}")

        # Read current joint angles before computing IK
        if self.current_joint_angles is None:
            joint_angles = self.read_current_joint_angles()
            if joint_angles is None:
                self.error_message = "Could not read current joint angles"
                self.get_logger().error(self.error_message)
                return False

        # Convert gripper center target to flange target using FK-derived rotation
        # (avoids mismatch between topic-reported orientation and FK)
        from primitives.shared.ik import forward_kinematics
        T_fk = forward_kinematics(dh_params, self.current_joint_angles)
        R_fk = T_fk[:3, :3]
        tool_offset_world = R_fk @ GRIPPER_CENTER_TOOL_OFFSET
        hover_flange = hover_gripper_center - tool_offset_world

        self.get_logger().info(f"Hover flange position (FK-derived): {hover_flange}")

        # Use Jacobian-based differential IK (same as move_to_safe_height)
        total_duration = 5.0
        num_waypoints = 60

        self.get_logger().info("Computing dense IK waypoints (Jacobian)...")
        waypoints = compute_cartesian_waypoints_ik(
            self.current_joint_angles,
            target_z=hover_flange[2],
            num_waypoints=num_waypoints,
            target_pos=hover_flange.tolist(),
        )
        if waypoints is None:
            self.error_message = "IK failed for Cartesian waypoints"
            self.get_logger().error(self.error_message)
            return False

        all_joint_angles = [self.current_joint_angles.copy()] + list(waypoints)

        # Trapezoidal velocity profile
        profile = trapezoidal_profile(all_joint_angles, total_duration)
        trajectory_points = []
        for positions, velocities, t_i in profile:
            trajectory_points.append({
                "positions": positions,
                "velocities": velocities,
                "time_from_start": Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
            })

        self.get_logger().info(f"Generated {len(trajectory_points)} Cartesian waypoints with trapezoidal velocity profile")

        success = self.execute_trajectory({"traj1": trajectory_points})
        if not success:
            self.get_logger().error("Failed to reach target position")
            return False

        # Closed-loop correction: re-read joints, check gripper center error, correct if needed
        CORRECTION_THRESHOLD = 0.00025  # 0.25mm
        MAX_CORRECTIONS = 3

        for correction_iter in range(MAX_CORRECTIONS):
            # Re-read current joint angles and EE pose from topics
            self.joint_angles_received = False
            self.current_ee_pose = None
            timeout = 0
            while rclpy.ok() and (not self.joint_angles_received or self.current_ee_pose is None) and timeout < 50:
                rclpy.spin_once(self, timeout_sec=0.1)
                timeout += 1

            if not self.joint_angles_received or self.current_ee_pose is None:
                self.get_logger().warn("Could not read pose data for correction")
                break

            # Get actual gripper center from ROS topic (ground truth)
            ee_pos_topic = np.array([self.current_ee_pose.pose.position.x,
                                     self.current_ee_pose.pose.position.y,
                                     self.current_ee_pose.pose.position.z])
            ee_quat_topic = np.array([self.current_ee_pose.pose.orientation.x,
                                      self.current_ee_pose.pose.orientation.y,
                                      self.current_ee_pose.pose.orientation.z,
                                      self.current_ee_pose.pose.orientation.w])
            R_ee_topic = R.from_quat(ee_quat_topic).as_matrix()
            actual_gripper_center = ee_pos_topic + R_ee_topic @ GRIPPER_CENTER_TOOL_OFFSET
            gripper_center_error = hover_gripper_center - actual_gripper_center
            pos_error = np.linalg.norm(gripper_center_error)

            self.get_logger().info(
                f"Correction check {correction_iter + 1}: gripper center error = {pos_error*1000:.2f}mm "
                f"(actual: [{actual_gripper_center[0]*1000:.1f}, {actual_gripper_center[1]*1000:.1f}, {actual_gripper_center[2]*1000:.1f}]mm)"
            )

            if pos_error <= CORRECTION_THRESHOLD:
                self.get_logger().info(f"Position accuracy OK ({pos_error*1000:.2f}mm <= {CORRECTION_THRESHOLD*1000:.2f}mm)")
                break

            # Apply the error as a delta to the current FK flange position
            # This bridges FK/topic mismatch: we measure error in topic space
            # but apply correction in FK space as a relative offset
            T_fk_current = forward_kinematics(dh_params, self.current_joint_angles)
            current_flange_fk = T_fk_current[:3, 3]
            corrected_flange = current_flange_fk + gripper_center_error

            self.get_logger().info(f"Applying correction move (error: {pos_error*1000:.2f}mm, delta: [{gripper_center_error[0]*1000:.2f}, {gripper_center_error[1]*1000:.2f}, {gripper_center_error[2]*1000:.2f}]mm)...")
            correction_waypoints = compute_cartesian_waypoints_ik(
                self.current_joint_angles,
                target_z=corrected_flange[2],
                num_waypoints=20,
                target_pos=corrected_flange.tolist(),
            )
            if correction_waypoints is None:
                self.get_logger().warn("Correction IK failed, skipping")
                break

            # Build quick correction trajectory (1s duration)
            corr_all = [self.current_joint_angles.copy()] + list(correction_waypoints)
            corr_n = len(corr_all)
            corr_duration = 1.0
            corr_points = []
            for i in range(corr_n):
                t_i = corr_duration * i / (corr_n - 1)
                if i == 0 or i == corr_n - 1:
                    vels = [0.0] * 6
                else:
                    delta = corr_all[min(i+1, corr_n-1)] - corr_all[max(i-1, 0)]
                    dn = np.linalg.norm(delta)
                    if dn > 1e-8:
                        vels = [float(delta[j] / dn * pos_error / corr_duration) for j in range(6)]
                    else:
                        vels = [0.0] * 6
                corr_points.append({
                    "positions": [float(x) for x in corr_all[i]],
                    "velocities": vels,
                    "time_from_start": Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
                })

            if not self.execute_trajectory({"traj1": corr_points}):
                self.get_logger().warn("Correction trajectory failed")
                break

        return True

    def execute_trajectory(self, trajectory):
        """Execute trajectory with multiple waypoints and wait for completion"""
        try:
            if 'traj1' not in trajectory or not trajectory['traj1']:
                return False

            points = trajectory['traj1']

            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names

            # Add all trajectory points
            for point in points:
                traj_point = JointTrajectoryPoint()
                traj_point.positions = point['positions']
                if 'velocities' in point:
                    traj_point.velocities = point['velocities']
                traj_point.time_from_start = point['time_from_start']
                traj_msg.points.append(traj_point)

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)

            future = self.action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()

            if not goal_handle.accepted:
                self.error_message = "External control program stopped or robot in protective stop"
                self.get_logger().error(self.error_message)
                return False

            self.get_logger().info(f"Trajectory with {len(points)} waypoints sent and accepted")
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result()

            if result.status == 4:  # SUCCEEDED
                self.get_logger().info("Movement completed successfully")
                return True
            else:
                result_msg = result.result
                if result_msg.error_code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
                    self.error_message = "Velocity or acceleration limits exceeded. The required velocity to reach the target exceeds joint velocity limits. Enable robot in URcap to fix this."
                else:
                    self.error_message = f"Trajectory failed with status code {result.status}"
                self.get_logger().error(self.error_message)
                return False
        except Exception as e:
            self.error_message = f"Trajectory execution error: {e}"
            self.get_logger().error(self.error_message)
            return False


def main(args=None):
    parser = argparse.ArgumentParser(description='Translate for Assembly - Move object to hover position')
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: sim (uses topics) or real (requires base position/orientation)')
    parser.add_argument('--object-name', type=str, required=True, help='Name of the object being held')
    parser.add_argument('--base-name', type=str, required=True, help='Name of the base object')
    
    # Real mode arguments
    parser.add_argument('--final-base-pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'), 
                       help='Final base position [x, y, z] in meters (required in real mode)')
    parser.add_argument('--final-base-orientation', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'),
                       help='Final base orientation quaternion [x, y, z, w] (required in real mode)')
    parser.add_argument('--use-default-base', action='store_true',
                       help=f'Use default base position ({DEFAULT_BASE_POSITION}) and orientation ({DEFAULT_BASE_ORIENTATION})')
    parser.add_argument('--grasp-id', type=int, default=None,
                       help='Grasp point ID to use for positioning (real mode only, offsets EE by grasp point position)')
    parser.add_argument('--current-object-orientation', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'),
                       help='Current object orientation quaternion [x, y, z, w] (real mode only, used with grasp-id for fold symmetry)')

    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'real':
        if not args.use_default_base and args.final_base_pos is None:
            parser.error("In real mode, either --final-base-pos or --use-default-base is required")
        if args.grasp_id is None:
            parser.error("In real mode, --grasp-id is required")
        if args.current_object_orientation is None:
            parser.error("In real mode, --current-object-orientation is required")
    
    rclpy.init()

    node = None
    success = False
    error = None

    try:
        node = TranslateForAssembly(mode=args.mode)
        node.action_client.wait_for_server()

        # Always need EE pose from topic
        while node.current_ee_pose is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.1)

        # In sim mode, wait for object and base poses from topics
        if args.mode == 'sim':
            while not node.current_poses:
                rclpy.spin_once(node, timeout_sec=0.1)
                time.sleep(0.1)

        # Default duration
        duration = 5.0

        # Execute translation (step 1 only: hover position)
        if args.mode == 'sim':
            success = node.translate_for_target_sim(
                args.object_name,
                args.base_name,
                duration=duration
            )
        else:  # real mode
            success = node.translate_for_target_real(
                args.object_name,
                args.base_name,
                duration=duration,
                final_base_pos=args.final_base_pos,
                final_base_orientation=args.final_base_orientation,
                use_default_base=args.use_default_base,
                grasp_id=args.grasp_id,
                object_orientation=args.current_object_orientation
            )

        if success:
            node.get_logger().info("Translation successful!")
        else:
            node.get_logger().error("Translation failed")
            error = node.error_message

    except KeyboardInterrupt:
        error = "Interrupted by user"
    except Exception as e:
        error = str(e)
    finally:
        # Build and output JSON result
        if success:
            result = {
                "result": "success",
                "mode": args.mode,
                "object_name": args.object_name,
                "base_name": args.base_name
            }
        else:
            result = {
                "result": "failure",
                "mode": args.mode,
                "object_name": args.object_name,
                "base_name": args.base_name,
                "error": error or (node.error_message if node else "Unknown error")
            }

        output_result(result)

        try:
            if node:
                node.action_client.destroy()
                node.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

