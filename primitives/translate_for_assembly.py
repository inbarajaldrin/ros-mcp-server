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
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
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

from primitives.utils.ik_solver import ik_objective_quaternion, forward_kinematics, dh_params
from primitives.utils.data_path_finder import get_assembly_data_dir

# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())
BASE_TOPIC = "/objects_poses_sim"
OBJECT_TOPIC = "/objects_poses_sim"
EE_TOPIC = "/tcp_pose_broadcaster/pose"
HOVER_HEIGHT = 0.25  # Height to hover above base before descending

# Default base position and orientation (used if not provided via command line)
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


class TranslateForAssembly(Node):
    def __init__(self, mode=None, base_topic=None, object_topic=None, ee_topic=EE_TOPIC):
        super().__init__('translate_for_assembly')
        
        # Mode must be explicitly specified
        if mode is None:
            raise ValueError("Mode must be explicitly specified. Use 'sim' or 'real'.")
        if mode not in ['sim', 'real']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'sim' or 'real'.")
        
        self.mode = mode  # 'sim' or 'real'
        
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
        
        self.ee_sub = self.create_subscription(PoseStamped, ee_topic, self.ee_callback, 10)
        
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
        
        self.get_logger().info(f"TranslateForAssembly node initialized (Mode: {self.mode})")
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
                self.get_logger().info(f"Loaded assembly config from: {json_file}")
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
            if component.get('name') == object_name or component.get('name') == f"{object_name}_scaled70":
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
        Compute IK using current joint angles as seed (similar to move_down.py)
        
        Args:
            target_position: [x, y, z] target position
            target_quat: [x, y, z, w] target orientation quaternion
            max_tries: Number of position perturbations to try
            dx: Position perturbation step size
            
        Returns:
            Joint angles if successful, None otherwise
        """
        # Convert quaternion to rotation matrix
        target_rotation = R.from_quat(target_quat)
        target_rot_matrix = target_rotation.as_matrix()
        
        # Create target pose
        target_pose = np.eye(4)
        target_pose[:3, 3] = target_position
        target_pose[:3, :3] = target_rot_matrix
        
        # Use current joint angles as seed
        if self.current_joint_angles is None:
            self.get_logger().error("Current joint angles not available! Cannot compute IK.")
            return None
        
        q_guess = self.current_joint_angles.copy()
        # self.get_logger().info(f"Using current joint angles from joint state as seed: {q_guess}")
        
        # Try IK with current joint angles and position perturbations
        solution_found = False
        best_result = None
        best_cost = float('inf')
        
        for i in range(max_tries):
            if solution_found:
                break
                
            # Try small x-shift each iteration (helps with workspace boundaries)
            perturbed_position = np.array(target_position).copy()
            perturbed_position[0] += i * dx
            
            perturbed_pose = target_pose.copy()
            perturbed_pose[:3, 3] = perturbed_position
            
            joint_bounds = [(-np.pi, np.pi)] * 6
            
            # Use quaternion-based objective
            result = minimize(ik_objective_quaternion, q_guess, args=(perturbed_pose,), 
                            method='L-BFGS-B', bounds=joint_bounds)
            
            if result.success:
                cost = ik_objective_quaternion(result.x, perturbed_pose)
                
                # Check if this is a good solution
                if cost < 0.01:
                    # self.get_logger().info(f"Quaternion-based IK succeeded with current joint angles seed (perturbation {i}), cost={cost:.6f}")
                    
                    # Verify orientation accuracy
                    # T_result = forward_kinematics(dh_params, result.x)
                    # orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
                    # self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
                    
                    return result.x
                
                # Keep track of best solution
                if cost < best_cost:
                    best_cost = cost
                    best_result = result.x
        
        # If we found any reasonable solution, use it
        if best_result is not None and best_cost < 0.1:
            # self.get_logger().info(f"Using best quaternion-based IK solution with cost={best_cost:.6f}")
            
            # Verify orientation accuracy
            # T_result = forward_kinematics(dh_params, best_result)
            # orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
            # self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
            
            return best_result
        
        # Fallback: Try multiple predefined seeds if current seed failed
        # self.get_logger().warn("IK failed with current joint angles as seed. Trying multiple predefined seeds...")
        
        # Convert target quaternion to RPY for seed generation
        target_rpy_deg = R.from_matrix(target_rot_matrix).as_euler('xyz', degrees=True)
        target_rpy_deg = target_rpy_deg.tolist()
        
        # Generate diverse seed configurations (similar to compute_ik_robust)
        seed_configs = [
            # Standard seeds
            np.radians([85, -80, 90, -90, -90, -(np.mod(target_rpy_deg[2] + 180, 360) - 180)]),
            np.radians([90, -90, 90, -90, -90, target_rpy_deg[2]]),
            np.radians([0, -90, 90, -90, -90, target_rpy_deg[2]]),
            np.radians([180, -90, 90, -90, -90, target_rpy_deg[2]]),
            # Elbow-up configurations
            np.radians([85, -100, 120, -110, -90, target_rpy_deg[2]]),
            np.radians([85, -60, 60, -90, -90, target_rpy_deg[2]]),
            # Wrist variations
            np.radians([85, -80, 90, -90, 0, target_rpy_deg[2]]),
            np.radians([85, -80, 90, -90, -180, target_rpy_deg[2]]),
            # Additional variations for pitch
            np.radians([85, -70, 80, -100, -90, target_rpy_deg[2]]),
            np.radians([85, -90, 100, -100, -90, target_rpy_deg[2]]),
        ]
        
        # self.get_logger().info(f"Trying {len(seed_configs)} alternative seed configurations...")
        
        best_result = None
        best_cost = float('inf')
        
        for seed_idx, q_guess_fallback in enumerate(seed_configs):
            for i in range(max_tries):
                # Try small x-shift each iteration
                perturbed_position = np.array(target_position).copy()
                perturbed_position[0] += i * dx
                
                perturbed_pose = target_pose.copy()
                perturbed_pose[:3, 3] = perturbed_position
                
                joint_bounds = [(-np.pi, np.pi)] * 6
                
                # Use quaternion-based objective
                result = minimize(ik_objective_quaternion, q_guess_fallback, args=(perturbed_pose,), 
                                method='L-BFGS-B', bounds=joint_bounds)
                
                if result.success:
                    cost = ik_objective_quaternion(result.x, perturbed_pose)
                    
                    # Check if this is a good solution
                    if cost < 0.01:
                        # self.get_logger().info(f"IK succeeded with fallback seed {seed_idx+1}/{len(seed_configs)} (perturbation {i}), cost={cost:.6f}")
                        
                        # Verify orientation accuracy
                        # T_result = forward_kinematics(dh_params, result.x)
                        # orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
                        # self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
                        
                        return result.x
                    
                    # Keep track of best solution
                    if cost < best_cost:
                        best_cost = cost
                        best_result = result.x
        
        # If we found any reasonable solution with fallback seeds, use it
        if best_result is not None and best_cost < 0.1:
            # self.get_logger().info(f"Using best fallback IK solution with cost={best_cost:.6f}")
            
            # Verify orientation accuracy
            # T_result = forward_kinematics(dh_params, best_result)
            # orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
            # self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
            
            return best_result
        
        self.get_logger().error("IK failed: couldn't find solution even with multiple seeds")
        return None
    
    def translate_for_target_sim(self, object_name, base_name, duration=20.0):
        """
        Sim mode: Calculate and execute EE translation to hover position (step 1 only).
        Uses topics to get base and object poses.
        """
        # Load assembly config based on base_name if not already loaded for this base
        if self.loaded_base_name != base_name:
            self.assembly_config = self.load_assembly_config(base_name=base_name)
            if not self.assembly_config:
                self.get_logger().error(f"Failed to load assembly config for base '{base_name}'")
                return False
        
        # Wait for pose data
        if not self.current_poses or self.current_ee_pose is None:
            self.get_logger().error("No pose data available")
            return False
        
        # Get current EE pose
        if self.current_ee_pose is None:
            self.get_logger().error("End-effector pose not available")
            return False
        
        # Check if object exists
        obj_key = object_name if object_name in self.current_poses else f"{object_name}_scaled70"
        if obj_key not in self.current_poses:
            self.get_logger().error(f"Object {object_name} not found")
            return False
        
        # Check if base exists
        base_key = base_name if base_name in self.current_poses else f"{base_name}_scaled70"
        if base_key not in self.current_poses:
            self.get_logger().error(f"Base {base_name} not found")
            return False
        
        # Convert poses to matrices
        T_EE_current = self.pose_to_matrix(self.current_ee_pose.pose)
        T_object_current = self.transform_to_matrix(self.current_poses[obj_key].transform)
        T_base_current = self.transform_to_matrix(self.current_poses[base_key].transform)
        
        # Calculate grasp transformation
        T_grasp = np.linalg.inv(T_EE_current) @ T_object_current
        
        # Get current positions
        ee_current_position, ee_current_rpy = self.matrix_to_rpy(T_EE_current)
        base_current_position, base_current_rpy = self.matrix_to_rpy(T_base_current)
        
        # Get target object position from JSON (relative to base)
        target_position_relative = self.get_object_target_position(object_name)
        if target_position_relative is None:
            self.get_logger().error(f"No target position found for {object_name} in JSON")
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
        hover_position = ee_target_position.copy()
        hover_position[2] = base_current_position[2] + HOVER_HEIGHT
        
        # Read current joint angles before computing IK
        if self.current_joint_angles is None:
            joint_angles = self.read_current_joint_angles()
            if joint_angles is None:
                self.get_logger().error("Could not read current joint angles")
                return False
        
        # Step 1: Move to hover position only (no step 2 in sim mode)
        self.get_logger().info(f"Moving to hover position: {hover_position} (height: {HOVER_HEIGHT}m above base)")
        
        # Compute IK for hover position
        hover_computed_joint_angles = self.compute_ik_with_current_seed(
            hover_position.tolist(),
            ee_target_quat.tolist(),
            max_tries=5,
            dx=0.001
        )
        
        if hover_computed_joint_angles is None:
            self.get_logger().error("Failed to compute IK for hover position")
            return False
        
        # Create hover trajectory
        hover_trajectory = [{
            "positions": [float(x) for x in hover_computed_joint_angles],
            "velocities": [0.0] * 6,
            "time_from_start": Duration(sec=int(duration))
        }]
        
        success = self.execute_trajectory({"traj1": hover_trajectory})
        if not success:
            self.get_logger().error("Failed to reach hover position")
            return False
        
        self.get_logger().info("Reached hover position")
        return success
    
    def translate_for_target_real(self, object_name, base_name, duration=20.0, 
                            final_base_pos=None, final_base_orientation=None,
                            use_default_base=False):
        """
        Real mode: Calculate and execute EE translation to hover position (step 1 only).
        Uses provided base position/orientation (no topics).
        
        Args:
            object_name: Name of the object being held
            base_name: Name of the base object (e.g., 'base')
            duration: Duration for trajectory execution
            final_base_pos: [x, y, z] final base position (required in real mode)
            final_base_orientation: [x, y, z, w] final base orientation quaternion (required in real mode)
            use_default_base: Use default base position/orientation if True
        """
        # Load assembly config based on base_name if not already loaded for this base
        if self.loaded_base_name != base_name:
            self.assembly_config = self.load_assembly_config(base_name=base_name)
            if not self.assembly_config:
                self.get_logger().error(f"Failed to load assembly config for base '{base_name}'")
                return None
        
        # Get current EE pose (always needed from topic)
        if self.current_ee_pose is None:
            self.get_logger().error("End-effector pose not available")
            return None
        
        # Note: We don't need current object position - we calculate EE position directly from target object position
        
        # Use default base position and orientation only if explicitly requested
        if final_base_pos is None:
            if use_default_base:
                final_base_pos = DEFAULT_BASE_POSITION
                self.get_logger().info(f"Using default base position: {final_base_pos}")
            else:
                self.get_logger().error("Base position not provided. Use --final-base-pos or --use-default-base flag")
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
            self.get_logger().error(f"No target position found for {object_name} in JSON")
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
        
        # Keep current EE orientation unchanged (assumed correct from reorient step)
        ee_target_quat = R.from_matrix(T_EE_current[:3, :3]).as_quat()
        self.get_logger().info("Keeping current EE orientation unchanged (from reorient step)")
        
        # Calculate EE position directly from target object position
        # Using same offsets as move_to_grasp: object_to_gripper_center_offset and tcp_to_gripper_center_offset
        # Since gripper is face-down, offsets are vertical in world Z-axis
        object_to_gripper_center_offset = 0.123  # 12.3cm - object is above gripper center
        tcp_to_gripper_center_offset = 0.24  # 24cm - TCP is above gripper center
        
        # Calculate gripper center position (below object)
        gripper_center_position = target_object_position_abs.copy()
        gripper_center_position[2] -= object_to_gripper_center_offset
        
        # Calculate TCP position (above gripper center)
        ee_target_position = gripper_center_position.copy()
        ee_target_position[2] += tcp_to_gripper_center_offset
        
        # Apply height offset - add HOVER_HEIGHT above base (step 1: hover position only)
        hover_position = ee_target_position.copy()
        hover_position[2] = base_current_position[2] + HOVER_HEIGHT
        
        self.get_logger().info(f"Calculated EE position from target object position:")
        self.get_logger().info(f"  Target object: {target_object_position_abs}")
        self.get_logger().info(f"  Gripper center: {gripper_center_position}")
        self.get_logger().info(f"  EE (TCP) position: {ee_target_position}")
        self.get_logger().info(f"  Hover position (with {HOVER_HEIGHT}m height offset): {hover_position}")
        
        # Read current joint angles before computing IK
        if self.current_joint_angles is None:
            joint_angles = self.read_current_joint_angles()
            if joint_angles is None:
                self.get_logger().error("Could not read current joint angles")
                return False
        
        # Step 1: Move to hover position only (no step 2 in real mode)
        self.get_logger().info(f"Moving to hover position: {hover_position} (height: {HOVER_HEIGHT}m above base)")
        
        # Compute IK for hover position using current joint angles as seed
        hover_computed_joint_angles = self.compute_ik_with_current_seed(
            hover_position.tolist(),
            ee_target_quat.tolist(),
            max_tries=5,
            dx=0.001
        )
        
        if hover_computed_joint_angles is None:
            self.get_logger().error("Failed to compute IK for hover position")
            return False
        
        # Create trajectory
        hover_trajectory = [{
            "positions": [float(x) for x in hover_computed_joint_angles],
            "velocities": [0.0] * 6,
            "time_from_start": Duration(sec=int(duration))
        }]
        
        success = self.execute_trajectory({"traj1": hover_trajectory})
        if not success:
            self.get_logger().error("Failed to reach hover position")
            return False
        
        self.get_logger().info("Reached hover position")
        return success
    
    def execute_trajectory(self, trajectory):
        """Execute trajectory and wait for completion"""
        try:
            if 'traj1' not in trajectory or not trajectory['traj1']:
                return False
            
            point = trajectory['traj1'][0]
            positions = point['positions']
            duration = point['time_from_start'].sec
            
            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names
            
            traj_point = JointTrajectoryPoint()
            traj_point.positions = positions
            traj_point.velocities = [0.0] * 6
            traj_point.time_from_start = Duration(sec=duration)
            traj_msg.points.append(traj_point)
            
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)
            
            future = self.action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            
            if not goal_handle.accepted:
                self.get_logger().error("Trajectory goal rejected")
                return False
            
            # self.get_logger().info("Trajectory goal accepted, waiting for completion...")
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result()
            
            if result.status == 4:  # SUCCEEDED
                # self.get_logger().info("Trajectory completed successfully")
                return True
            else:
                self.get_logger().error(f"Trajectory failed with status: {result.status}")
                return False
        except Exception as e:
            self.get_logger().error(f"Trajectory execution error: {e}")
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
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'real':
        if not args.use_default_base and args.final_base_pos is None:
            parser.error("In real mode, either --final-base-pos or --use-default-base is required")
    
    rclpy.init()
    node = TranslateForAssembly(mode=args.mode)
    
    node.action_client.wait_for_server()
    
    try:
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
                use_default_base=args.use_default_base
            )
        
        if success:
            node.get_logger().info("Translation successful!")
        else:
            node.get_logger().error("Translation failed")
        
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

