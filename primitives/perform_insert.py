#!/usr/bin/env python3
"""
Force Compliant Move Down Controller
=====================================
Moves down continuously with fixed orientation.

Search Phase (before contact):
- Only Z moves down (no X/Y adjustment)
- X and Y positions maintained at starting values

Alignment Phase (after contact detected via Z force threshold):
- Z moves slowly (10% speed)
- X and Y positions adjusted based on force compliance to minimize lateral forces
- Ideal for peg-in-hole insertion tasks

Usage:
    python3 perform_insert.py
    python3 perform_insert.py --speed 0.01 --z-threshold -10.0
"""

import sys
import os

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped
from builtin_interfaces.msg import Duration
from scipy.spatial.transform import Rotation as R

import numpy as np
import time
import argparse
from primitives.utils.ik_solver import compute_ik, ik_objective_quaternion
from scipy.optimize import minimize
from tf2_msgs.msg import TFMessage
import json
import os
import glob
from primitives.utils.data_path_finder import get_assembly_data_dir


# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())


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


class PerformInsertController(Node):
    """
    Performs insert operation with force compliance.
    Moves down continuously with fixed orientation.
    Adjusts X and Y positions based on force compliance.
    Z-direction moves down regardless of force.
    """
    
    def __init__(self, mode=None, speed=0.005, gain=1.67, deadband=1.0, max_vel=15.0, reverse=False, z_threshold=-10.0, xy_force_threshold=1.0, object_name=None, base_name=None):
        super().__init__('perform_insert_controller')
        
        # Mode must be explicitly specified
        if mode is None:
            raise ValueError("Mode must be explicitly specified. Use 'sim' or 'real'.")
        if mode not in ['sim', 'real']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'sim' or 'real'.")
        
        self.mode = mode  # 'sim' or 'real'
        
        # For sim mode, we need object and base names
        if self.mode == 'sim':
            if object_name is None or base_name is None:
                raise ValueError("In sim mode, object_name and base_name are required")
            self.object_name = object_name
            self.base_name = base_name
            # Find and load assembly config for sim mode based on base_name
            assembly_json_file = find_assembly_json_by_base_name(base_name, ASSEMBLY_DATA_DIR, self.get_logger())
            if assembly_json_file:
                self.assembly_config = self.load_assembly_config(assembly_json_file)
            else:
                self.get_logger().error(f"Could not find assembly JSON for base '{base_name}'")
                self.assembly_config = {}
            # Subscribe to object poses topic
            self.current_poses = {}
            self.object_sub = self.create_subscription(TFMessage, "/objects_poses_sim", self.object_callback, 10)
        else:
            self.object_name = None
            self.base_name = None
            self.assembly_config = None
            self.current_poses = {}
            self.object_sub = None
        
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        self.speed = speed          # m/s downward speed
        self.gain = gain            # mm/s per Newton for X/Y compliance
        self.deadband = deadband    # N deadband for X/Y forces
        self.max_vel = max_vel      # mm/s max velocity for X/Y compliance
        self.reverse = reverse      # If True, reverse X/Y force response directions
        self.z_threshold = z_threshold  # Z force threshold in N (negative = upward force/resistance)
        self.xy_force_threshold = xy_force_threshold  # Minimum X/Y force magnitude required to enter alignment (N)
        
        # Current state
        self.pos = None             # [x, y, z] in meters
        self.rpy = None             # [r, p, y] in degrees
        self.joints = None
        self.force = None           # [fx, fy, fz] in N
        
        # Fixed orientation (captured at start)
        self.fixed_rpy = None       # [r, p, y] in degrees
        
        # Baseline force
        self.baseline = np.array([0.0, 0.0, 0.0])
        
        # Force smoothing for alignment mode (to prevent oscillation)
        self.smoothed_force_xy = np.array([0.0, 0.0])  # Smoothed X/Y forces
        self.force_smoothing_alpha = 0.3  # EMA smoothing factor (0-1, lower = more smoothing)
        
        # Starting position (for maintaining X/Y during search phase)
        self.start_x = None
        self.start_y = None
        
        # Contact detection and alignment mode
        self.contact_detected = False  # Whether Z force threshold has been crossed
        self.alignment_mode = False    # Whether we're in alignment mode (minimizing X/Y forces)
        self.just_detected_contact = False  # Flag to trigger trajectory cancellation
        self.current_goal_handle = None  # Current trajectory goal handle (for cancellation)
        self.alignment_start_time = None  # Time when alignment mode started
        self.alignment_stop_timeout = 20.0  # Stop alignment after this many seconds
        self.alignment_min_duration = 1.0  # Minimum time in alignment mode before checking for completion
        self.low_force_start_time = None  # Time when X/Y forces first went low (after min duration)
        self.low_force_threshold = deadband  # X/Y force magnitude threshold to consider "low" (use deadband value)
        self.low_force_required_duration = 2.0  # How long forces must be low before stopping
        
        # Safety limits to prevent excessive forces
        self.max_z_force = -5.0  # Maximum Z force (N) - stop if exceeded
        self.max_xy_force = 10.0  # Maximum X/Y force (N) - stop if exceeded
        
        # Setup
        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self._pose_cb, 10)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        
        # Only subscribe to force sensor in real mode
        if self.mode == 'real':
            self.create_subscription(WrenchStamped, '/force_torque_sensor_broadcaster/wrench', self._wrench_cb, 10)
        else:
            self.force = np.array([0.0, 0.0, 0.0])  # Dummy force for sim mode
        
        # Wait for data
        self.get_logger().info(f"Waiting for robot data (mode: {self.mode})...")
        while self.pos is None or self.joints is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # In real mode, also wait for force data
        if self.mode == 'real':
            while self.force is None:
                rclpy.spin_once(self, timeout_sec=0.1)
        
        # Capture and store fixed orientation
        if self.rpy is not None:
            # Set fixed orientation to [0, 180, 0.01] - face down with yaw = 0.01°
            self.fixed_rpy = np.array([0.0, 180.0, 0.01])
            self.get_logger().info(f"Fixed orientation: R={self.fixed_rpy[0]:.1f}°, P={self.fixed_rpy[1]:.1f}°, Y={self.fixed_rpy[2]:.2f}°")
        else:
            # Default orientation: face down with yaw = 0.01°
            self.fixed_rpy = np.array([0.0, 180.0, 0.01])
            self.get_logger().info("Using default orientation: R=0.0°, P=180.0°, Y=0.01°")
        
        # Store starting X and Y position (will be maintained during search phase)
        if self.pos is not None:
            self.start_x = self.pos[0]
            self.start_y = self.pos[1]
            self.get_logger().info(f"Starting position: X={self.start_x*1000:.2f} mm, Y={self.start_y*1000:.2f} mm (will be maintained until contact)")
        
        # Get baseline force (average over 2 seconds) - only in real mode
        if self.mode == 'real':
            self.get_logger().info("Calibrating force sensor (2 sec)...")
            samples = []
            start = time.time()
            while time.time() - start < 2.0:
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.force is not None:
                    samples.append(self.force.copy())
                time.sleep(0.05)
            self.baseline = np.mean(samples, axis=0)
            self.get_logger().info(f"Baseline: [{self.baseline[0]:.2f}, {self.baseline[1]:.2f}, {self.baseline[2]:.2f}] N")
        else:
            self.baseline = np.array([0.0, 0.0, 0.0])  # No baseline needed for sim mode
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("FORCE COMPLIANT MOVE DOWN CONTROLLER INITIALIZED")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Downward speed: {self.speed * 1000:.1f} mm/s")
        self.get_logger().info(f"X/Y compliance gain: {gain} mm/s per N")
        self.get_logger().info(f"X/Y force deadband: {deadband} N")
        self.get_logger().info(f"X/Y max velocity: {max_vel} mm/s")
        self.get_logger().info(f"Force direction reversed: {'YES' if self.reverse else 'NO'}")
        self.get_logger().info(f"Z force threshold: {z_threshold:.1f} N (contact detection, negative = upward resistance)")
        self.get_logger().info(f"X/Y force threshold: {self.xy_force_threshold:.1f} N (minimum required for alignment entry)")
        self.get_logger().info(f"Fixed orientation: R={self.fixed_rpy[0]:.1f}°, P={self.fixed_rpy[1]:.1f}°, Y={self.fixed_rpy[2]:.1f}°")
        self.get_logger().info("=" * 60)
    
    def _canonicalize_euler(self, orientation):
        """
        Always canonicalize Euler angles to [0, 180, yaw] format.
        This is the same format used by move_down_compliant and works better with IK solver.
        """
        roll, pitch, yaw = orientation
        
        # Convert current RPY to quaternion
        current_rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        
        # Create base rotation [0, 180, 0] (face down)
        base_rot = R.from_euler('xyz', [0, 180, 0], degrees=True)
        
        # Find the rotation needed to go from base to current orientation
        relative_rot = current_rot * base_rot.inv()
        
        # Extract yaw from the relative rotation
        relative_rpy = relative_rot.as_euler('xyz', degrees=True)
        canonical_yaw = relative_rpy[2]
        
        # Normalize yaw to [-180, 180] range
        while canonical_yaw > 180:
            canonical_yaw -= 360
        while canonical_yaw < -180:
            canonical_yaw += 360
        
        return np.array([0.0, 180.0, canonical_yaw])
    
    def _pose_cb(self, msg):
        self.pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        raw_rpy = np.degrees(R.from_quat(q).as_euler('xyz'))
        # Canonicalize to [0, 180, yaw] format
        self.rpy = self._canonicalize_euler(raw_rpy)
    
    def _joint_cb(self, msg):
        # Extract joint angles in the correct order (must match DH parameters)
        if len(msg.name) >= 6 and len(msg.position) >= 6:
            joint_dict = dict(zip(msg.name, msg.position))
            # Map joint names to positions in correct order
            ordered_positions = []
            for joint_name in self.joint_names:
                if joint_name in joint_dict:
                    ordered_positions.append(joint_dict[joint_name])
            
            if len(ordered_positions) == 6:
                self.joints = np.array(ordered_positions)
    
    def object_callback(self, msg):
        """Callback for object poses (sim mode only)"""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            self.current_poses[frame_id] = transform
    
    def load_assembly_config(self, json_file):
        """Load assembly configuration from JSON file"""
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.get_logger().error(f"Assembly file not found: {json_file}")
            return {}
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Error parsing assembly JSON: {e}")
            return {}
    
    def get_object_target_position(self, object_name):
        """Get target position for object from assembly configuration"""
        for component in self.assembly_config.get('components', []):
            if component.get('name') == object_name or component.get('name') == f"{object_name}_scaled70":
                position = component.get('position', {})
                return np.array([position.get('x', 0), position.get('y', 0), position.get('z', 0)])
        return None
    
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
    
    def matrix_to_rpy(self, T):
        """Convert 4x4 transformation matrix to position and RPY (degrees)"""
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        r = R.from_matrix(rotation_matrix)
        rpy_rad = r.as_euler('xyz')
        rpy_deg = np.degrees(rpy_rad)
        return position, rpy_deg
    
    def read_current_joint_angles(self):
        """Read current joint angles using ROS2 subscriber"""
        timeout_count = 0
        max_timeout = 100
        while rclpy.ok() and timeout_count < max_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout_count += 1
        return self.joints.copy() if self.joints is not None else None
    
    def compute_ik_with_current_seed(self, target_position, target_quat, max_tries=5, dx=0.001):
        """
        Compute IK using current joint angles as seed (exactly like translate_for_assembly_old step 2)
        
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
        if self.joints is None:
            self.get_logger().error("Current joint angles not available! Cannot compute IK.")
            return None
        
        q_guess = self.joints.copy()
        
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
                    return result.x
                
                # Keep track of best solution
                if cost < best_cost:
                    best_cost = cost
                    best_result = result.x
        
        # If we found any reasonable solution, use it
        if best_result is not None and best_cost < 0.1:
            return best_result
        
        # Fallback: Try multiple predefined seeds if current seed failed
        # Convert target quaternion to RPY for seed generation
        target_rpy_deg = R.from_matrix(target_rot_matrix).as_euler('xyz', degrees=True)
        target_rpy_deg = target_rpy_deg.tolist()
        
        # Generate diverse seed configurations (same as translate_for_assembly_old)
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
                        return result.x
                    
                    # Keep track of best solution
                    if cost < best_cost:
                        best_cost = cost
                        best_result = result.x
        
        # If we found any reasonable solution with fallback seeds, use it
        if best_result is not None and best_cost < 0.1:
            return best_result
        
        self.get_logger().error("IK failed: couldn't find solution even with multiple seeds")
        return None
    
    def execute_trajectory_sim(self, trajectory):
        """Execute trajectory and wait for completion (sim mode)"""
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
            
            future = self.traj_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            
            if not goal_handle.accepted:
                self.get_logger().error("Trajectory goal rejected")
                return False
            
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result()
            
            if result.status == 4:  # SUCCEEDED
                return True
            else:
                self.get_logger().error(f"Trajectory failed with status: {result.status}")
                return False
        except Exception as e:
            self.get_logger().error(f"Trajectory execution error: {e}")
            return False
    
    def move_down_sim(self, duration=10.0):
        """Sim mode: Move down to final position (step 2 from translate_for_assembly)"""
        # Wait for pose data
        if not self.current_poses:
            self.get_logger().error("No pose data available")
            return False
        
        # Get current EE pose
        if self.pos is None:
            self.get_logger().error("End-effector pose not available")
            return False
        
        # Check if object exists
        obj_key = self.object_name if self.object_name in self.current_poses else f"{self.object_name}_scaled70"
        if obj_key not in self.current_poses:
            self.get_logger().error(f"Object {self.object_name} not found")
            return False
        
        # Check if base exists
        base_key = self.base_name if self.base_name in self.current_poses else f"{self.base_name}_scaled70"
        if base_key not in self.current_poses:
            self.get_logger().error(f"Base {self.base_name} not found")
            return False
        
        # Get current EE pose as PoseStamped-like structure
        # Use actual current EE orientation from topic (maintain current orientation like old code)
        if self.rpy is None:
            self.get_logger().error("EE RPY not available from topic")
            return False
        
        current_ee_pose_msg = PoseStamped()
        current_ee_pose_msg.pose.position.x = self.pos[0]
        current_ee_pose_msg.pose.position.y = self.pos[1]
        current_ee_pose_msg.pose.position.z = self.pos[2]
        
        # Use current EE orientation from topic (self.rpy), not fixed orientation
        quat = R.from_euler('xyz', self.rpy, degrees=True).as_quat()
        current_ee_pose_msg.pose.orientation.x = quat[0]
        current_ee_pose_msg.pose.orientation.y = quat[1]
        current_ee_pose_msg.pose.orientation.z = quat[2]
        current_ee_pose_msg.pose.orientation.w = quat[3]
        
        # Convert poses to matrices
        T_EE_current = self.pose_to_matrix(current_ee_pose_msg.pose)
        T_object_current = self.transform_to_matrix(self.current_poses[obj_key].transform)
        T_base_current = self.transform_to_matrix(self.current_poses[base_key].transform)
        
        # Calculate grasp transformation
        T_grasp = np.linalg.inv(T_EE_current) @ T_object_current
        
        # Get current positions
        ee_current_position, ee_current_rpy = self.matrix_to_rpy(T_EE_current)
        base_current_position, base_current_rpy = self.matrix_to_rpy(T_base_current)
        
        # Get target object position from JSON (relative to base)
        target_position_relative = self.get_object_target_position(self.object_name)
        if target_position_relative is None:
            self.get_logger().error(f"No target position found for {self.object_name} in JSON")
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
        
        self.get_logger().info(f"Step 2: Moving down to final position: {ee_target_position}")
        
        # Read current joint angles before computing IK
        if self.joints is None:
            self.read_current_joint_angles()
            if self.joints is None:
                self.get_logger().error("Could not read current joint angles")
                return False
        
        # Compute IK for final position
        final_computed_joint_angles = self.compute_ik_with_current_seed(
            ee_target_position.tolist(),
            ee_target_quat.tolist(),
            max_tries=5,
            dx=0.001
        )
        
        if final_computed_joint_angles is None:
            self.get_logger().error("Failed to compute IK for final position")
            return False
        
        # Create final trajectory
        final_trajectory = [{
            "positions": [float(x) for x in final_computed_joint_angles],
            "velocities": [0.0] * 6,
            "time_from_start": Duration(sec=int(duration))
        }]
        
        success = self.execute_trajectory_sim({"traj1": final_trajectory})
        
        if success:
            # Wait for poses to update (same as old code)
            time.sleep(0.5)
            self.get_logger().info("Reached final position")
        else:
            self.get_logger().error("Failed to reach final position")
        
        return success
    
    def _wrench_cb(self, msg):
        self.force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
    
    def get_force_xy(self, use_deadband=True):
        """
        Get X and Y force components with baseline removed.
        Optionally apply deadband.
        
        Args:
            use_deadband: If True, apply deadband. If False, return raw forces (for alignment mode).
        """
        if self.force is None:
            return np.array([0.0, 0.0])
        f = self.force - self.baseline
        # Only use X and Y components
        f_xy = np.array([f[0], f[1]])
        if use_deadband:
            mag = np.linalg.norm(f_xy)
            if mag < self.deadband:
                return np.array([0.0, 0.0])
        return f_xy
    
    def get_force_z(self):
        """
        Get Z force component with baseline removed.
        """
        if self.force is None:
            return 0.0
        f = self.force - self.baseline
        return f[2]
    
    def get_next_waypoint(self) -> tuple:
        """
        Get the next waypoint.
        - Search mode (before contact): Only Z moves down, X/Y maintained at starting position
        - Alignment mode (after contact): Z moves slowly, X/Y adjusted based on force compliance
        
        Returns:
            (x, y, z) tuple in meters
        """
        if self.pos is None:
            return None
        
        current_pos = self.pos.copy()
        
        # Check Z force and X/Y forces to detect contact with misalignment
        f_z = self.get_force_z()
        f_xy = self.get_force_xy(use_deadband=False)  # Get raw X/Y forces (no deadband)
        f_xy_mag = np.linalg.norm(f_xy)
        
        # Check if contact detected: Z force reaches threshold (negative = upward resistance)
        # AND X/Y forces are present (indicating misalignment that needs fixing)
        if not self.contact_detected:
            z_contact = f_z <= self.z_threshold  # Negative Z = upward resistance
            xy_misalignment = f_xy_mag >= self.xy_force_threshold  # X/Y forces present
            
            if z_contact and xy_misalignment:
                self.contact_detected = True
                self.just_detected_contact = True  # Flag to trigger cancellation in run loop
                self.get_logger().info("=" * 60)
                self.get_logger().info(f"CONTACT DETECTED: Z force = {f_z:.2f} N (threshold: {self.z_threshold:.1f} N)")
                self.get_logger().info(f"MISALIGNMENT DETECTED: X/Y force magnitude = {f_xy_mag:.2f} N (threshold: {self.xy_force_threshold:.1f} N)")
                self.get_logger().info("Will stop current trajectory and enter ALIGNMENT MODE")
                self.get_logger().info("Will move down very slowly while adjusting X/Y to minimize forces")
                self.get_logger().info("=" * 60)
        
        # TODO: Fix move down speed till first contact - adjust search phase speed if needed
        # Z-direction movement
        dt = 0.6  # Trajectory duration (matches time_from_start in trajectory)
        
        if self.alignment_mode:
            # In alignment mode: very slow downward movement (1% of normal speed)
            # Always moves down regardless of X/Y forces (like old code)
            dz = -self.speed * dt * 0.01  # 1% of normal speed for very slow insertion
        else:
            # Normal mode: move down at full speed
            dz = -self.speed * dt  # Negative Z is down
        
        next_z = current_pos[2] + dz
        
        # X and Y: Only adjust in alignment mode (after contact detected)
        # Before contact: maintain starting X/Y position (no force compliance)
        if self.alignment_mode:
            # In alignment mode: adjust X/Y based on force compliance
            # Don't use deadband to be more sensitive to small forces (like old code)
            f_xy_raw = self.get_force_xy(use_deadband=False)
            
            # Apply exponential moving average smoothing to reduce oscillation
            # This filters out high-frequency noise that causes back-and-forth movement
            self.smoothed_force_xy = (self.force_smoothing_alpha * f_xy_raw + 
                                     (1.0 - self.force_smoothing_alpha) * self.smoothed_force_xy)
            f_xy = self.smoothed_force_xy
            
            # Use base gain value
            effective_gain = self.gain
            
            # Compute displacement for X and Y based on force
            # Move WITH the force (yield to it) to minimize lateral forces
            # Default behavior:
            #   X axis is inverted: if force is +X, move -X (inverted)
            #   Y axis: same direction as force
            # If reverse=True, the directions are flipped
            displacement_xy = np.array([0.0, 0.0])
            
            # In alignment mode, always try to minimize forces (even very small ones)
            f_xy_mag = np.linalg.norm(f_xy)
            if f_xy_mag > 0.2:  # Small threshold to avoid noise (increased from 0.1 to reduce oscillation)
                # Determine sign based on reverse flag
                x_sign = 1.0 if self.reverse else -1.0  # Normal: inverted, Reverse: same direction
                y_sign = -1.0 if self.reverse else 1.0   # Normal: same direction, Reverse: inverted
                
                # X axis displacement
                dx = x_sign * effective_gain * f_xy[0] * dt / 1000.0  # meters
                # Limit maximum displacement per step
                max_disp = self.max_vel * dt / 1000.0  # meters per timestep
                if abs(dx) > max_disp:
                    dx = np.sign(dx) * max_disp
                displacement_xy[0] = dx
                
                # Y axis displacement
                dy = y_sign * effective_gain * f_xy[1] * dt / 1000.0  # meters
                if abs(dy) > max_disp:
                    dy = np.sign(dy) * max_disp
                displacement_xy[1] = dy
            
            next_x = current_pos[0] + displacement_xy[0]
            next_y = current_pos[1] + displacement_xy[1]
        else:
            # Search mode: maintain starting X/Y position (no force compliance)
            # Only Z moves down
            if self.start_x is not None and self.start_y is not None:
                next_x = self.start_x
                next_y = self.start_y
            else:
                # Fallback to current position if start not stored
                next_x = current_pos[0]
                next_y = current_pos[1]
        
        return (next_x, next_y, next_z)
    
    def run(self):
        """Main run method - different behavior for sim vs real mode"""
        if self.mode == 'sim':
            # Sim mode: execute step 2 (move down to final position)
            # Wait for object and base poses from topics
            self.get_logger().info("Waiting for object and base poses from topics...")
            while not self.current_poses:
                rclpy.spin_once(self, timeout_sec=0.1)
                time.sleep(0.1)
            return self.move_down_sim(duration=10.0)
        else:
            # Real mode: original force-compliant movement
            return self.run_real()
    
    def run_real(self):
        self.get_logger().info("Starting force compliant move down. Press Ctrl+C to stop.")
        
        if not self.traj_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Trajectory server not available!")
            return
        
        # First, run the search phase with precomputed waypoints
        self.running = True
        search_complete = self.run_search_phase()
        
        if not self.running:
            self.get_logger().info("Stopped during search phase.")
            return
        
        # If contact detected with misalignment, run alignment phase
        if self.contact_detected and self.alignment_mode:
            self.run_alignment_phase()
        
        self.get_logger().info("Stopped.")
    
    def run_search_phase(self):
        """
        Search phase: Pre-compute all waypoints (Z moving down, X/Y fixed) and send as single trajectory.
        Monitors force during execution and cancels if contact is detected.
        Returns True if completed, False if interrupted.
        """
        self.get_logger().info("=" * 60)
        self.get_logger().info("SEARCH PHASE: Pre-computing all waypoints")
        self.get_logger().info("=" * 60)
        
        if self.pos is None or self.joints is None:
            self.get_logger().error("Position or joint angles not available!")
            return False
        
        # Calculate how far we want to move down
        start_z = self.pos[2]
        target_z = 0.0  # Workspace base (or until contact)
        distance_to_move = start_z - target_z
        
        if distance_to_move <= 0.001:
            self.get_logger().info(f"Already at or below target Z position ({start_z:.3f}m). Exiting.")
            return True
        
        # Calculate number of waypoints based on speed and trajectory duration
        dt = 0.6  # Duration per waypoint (matches time_from_start)
        dz_per_waypoint = self.speed * dt  # Distance moved per waypoint
        num_waypoints = max(10, int(distance_to_move / dz_per_waypoint))
        
        self.get_logger().info(f"Pre-computing {num_waypoints} waypoints")
        self.get_logger().info(f"Moving from Z={start_z:.3f}m to Z={target_z:.3f}m")
        self.get_logger().info(f"Maintaining X={self.start_x:.4f}, Y={self.start_y:.4f}")
        
        # Pre-compute all waypoints
        waypoints = []
        joint_trajectory = []
        
        current_z = start_z
        q_guess = self.joints.copy()  # Use current joint angles as seed
        
        for i in range(num_waypoints):
            # Calculate next Z position
            current_z = start_z - (i + 1) * dz_per_waypoint
            if current_z < target_z:
                current_z = target_z
            
            # X and Y stay fixed at starting position
            waypoint = [self.start_x, self.start_y, current_z]
            waypoints.append(waypoint)
            
            # Compute IK for this waypoint
            target_rpy = list(self.fixed_rpy)
            joint_angles = compute_ik(waypoint, target_rpy, q_guess=q_guess)
            
            if joint_angles is None:
                self.get_logger().warn(f"IK failed at waypoint {i}, truncating trajectory at waypoint {i-1}")
                break
            
            joint_trajectory.append(joint_angles)
            q_guess = joint_angles  # Use this solution as seed for next waypoint
            
            # Stop if we've reached target Z
            if current_z <= target_z + 0.001:
                break
        
        if len(joint_trajectory) == 0:
            self.get_logger().error("Failed to compute any waypoints! Aborting.")
            return False
        
        self.get_logger().info(f"Computed {len(joint_trajectory)} waypoints successfully")
        
        # Calculate total trajectory duration
        trajectory_duration = len(joint_trajectory) * dt
        self.get_logger().info(f"Trajectory duration: {trajectory_duration:.1f}s")
        
        # Create trajectory message with all waypoints
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        
        for i, joint_angles in enumerate(joint_trajectory):
            point = JointTrajectoryPoint()
            point.positions = [float(j) for j in joint_angles]
            point.velocities = [0.0] * 6
            
            # Distribute time evenly across waypoints
            t = (i + 1) * dt
            point.time_from_start = Duration(
                sec=int(t),
                nanosec=int((t - int(t)) * 1e9)
            )
            trajectory.points.append(point)
        
        # Send trajectory
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = Duration(sec=2)
        
        self.get_logger().info(f"Sending trajectory with {len(trajectory.points)} waypoints...")
        
        future = self.traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected!")
            return False
        
        self.current_goal_handle = goal_handle
        self.get_logger().info("Trajectory accepted. Monitoring force during execution...")
        
        # Monitor force during trajectory execution
        start_time = time.time()
        last_log_time = start_time
        
        while self.running and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)  # 50Hz monitoring
            
            # Check force for contact detection
            f_z = self.get_force_z()
            f_xy = self.get_force_xy(use_deadband=False)
            f_xy_mag = np.linalg.norm(f_xy)
            
            # Check for contact
            z_contact = f_z <= self.z_threshold
            xy_misalignment = f_xy_mag >= self.xy_force_threshold
            
            if z_contact:
                self.contact_detected = True
                
                # Cancel trajectory
                if self.current_goal_handle is not None:
                    try:
                        cancel_future = self.current_goal_handle.cancel_goal_async()
                        rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=0.5)
                        self.get_logger().info("Cancelled search trajectory due to contact")
                    except Exception as e:
                        self.get_logger().warn(f"Error cancelling trajectory: {e}")
                
                if xy_misalignment:
                    self.get_logger().info("=" * 60)
                    self.get_logger().info(f"CONTACT DETECTED: Z force = {f_z:.2f} N (threshold: {self.z_threshold:.1f} N)")
                    self.get_logger().info(f"MISALIGNMENT DETECTED: X/Y force = {f_xy_mag:.2f} N (threshold: {self.xy_force_threshold:.1f} N)")
                    self.get_logger().info("Entering ALIGNMENT MODE")
                    self.get_logger().info("=" * 60)
                    self.alignment_mode = True
                    self.alignment_start_time = time.time()
                    self.low_force_start_time = None  # Reset low force timer when entering alignment
                    self.smoothed_force_xy = np.array([0.0, 0.0])  # Reset smoothed force when entering alignment
                else:
                    self.get_logger().info("=" * 60)
                    self.get_logger().info(f"CONTACT DETECTED: Z force = {f_z:.2f} N (threshold: {self.z_threshold:.1f} N)")
                    self.get_logger().info(f"NO MISALIGNMENT: X/Y force = {f_xy_mag:.2f} N (below threshold)")
                    self.get_logger().info("Moving down until Z force reaches -5N to confirm no misalignment...")
                    self.get_logger().info("=" * 60)
                    
                    # Move down until Z force reaches -5N to confirm no misalignment
                    confirm_z_target = -5.0  # Target Z force in N
                    dt_confirm = 0.6  # Trajectory duration for confirmation steps
                    confirm_waypoint_count = 0
                    last_confirm_trajectory_time = 0.0
                    min_trajectory_interval = 0.1
                    
                    while self.running and rclpy.ok():
                        rclpy.spin_once(self, timeout_sec=0.1)
                        
                        if self.pos is None or self.fixed_rpy is None:
                            break
                        
                        # Check current Z force
                        f_z_current = self.get_force_z()
                        
                        # Stop if we've reached target Z force
                        if f_z_current <= confirm_z_target:
                            # Check forces at target Z force
                            f_xy_confirm = self.get_force_xy(use_deadband=False)
                            f_xy_mag_confirm = np.linalg.norm(f_xy_confirm)
                            
                            self.get_logger().info(f"Reached Z force = {f_z_current:.2f} N")
                            self.get_logger().info(f"X/Y force = {f_xy_mag_confirm:.2f} N")
                            
                            # Check if misalignment appeared
                            if f_xy_mag_confirm >= self.xy_force_threshold:
                                self.get_logger().info("=" * 60)
                                self.get_logger().info(f"MISALIGNMENT DETECTED: X/Y force = {f_xy_mag_confirm:.2f} N")
                                self.get_logger().info("Entering ALIGNMENT MODE")
                                self.get_logger().info("=" * 60)
                                self.alignment_mode = True
                                self.alignment_start_time = time.time()
                                self.low_force_start_time = None
                                self.smoothed_force_xy = np.array([0.0, 0.0])
                                # Continue to alignment phase
                            else:
                                self.get_logger().info("=" * 60)
                                self.get_logger().info("Contact confirmed - no misalignment at Z force = -5N")
                                self.get_logger().info("=" * 60)
                                self.running = False
                            break
                        
                        # Rate limit trajectory sends
                        current_time = time.time()
                        if current_time - last_confirm_trajectory_time < min_trajectory_interval:
                            time.sleep(0.05)
                            continue
                        last_confirm_trajectory_time = current_time
                        
                        # Move down slowly (same speed as alignment phase: 0.5% of search speed)
                        current_pos = self.pos.copy()
                        dz = -self.speed * dt_confirm * 0.005  # 0.5% of search speed
                        next_z = current_pos[2] + dz
                        target_rpy = list(self.fixed_rpy)
                        
                        # Compute IK
                        if self.joints is not None:
                            joint_angles = compute_ik([current_pos[0], current_pos[1], next_z], target_rpy)
                            
                            if joint_angles is not None:
                                # Create and send trajectory
                                trajectory = JointTrajectory()
                                trajectory.joint_names = self.joint_names
                                
                                point = JointTrajectoryPoint()
                                point.positions = [float(j) for j in joint_angles]
                                point.velocities = [0.0] * 6
                                point.time_from_start = Duration(sec=0, nanosec=600000000)
                                trajectory.points.append(point)
                                
                                goal = FollowJointTrajectory.Goal()
                                goal.trajectory = trajectory
                                goal.goal_time_tolerance = Duration(sec=1)
                                
                                future = self.traj_client.send_goal_async(goal)
                                rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
                                
                                goal_handle = future.result()
                                if goal_handle and goal_handle.accepted:
                                    time.sleep(0.2)
                                    confirm_waypoint_count += 1
                                else:
                                    self.get_logger().warn(f"Confirmation trajectory rejected at waypoint {confirm_waypoint_count}")
                            else:
                                self.get_logger().warn(f"IK failed for confirmation waypoint {confirm_waypoint_count}")
                                time.sleep(0.1)
                        
                        rclpy.spin_once(self, timeout_sec=0.1)
                    
                    # If we're not entering alignment mode, return True to exit search phase
                    if not self.alignment_mode:
                        return True
                
                return True
            
            # Check if trajectory completed (check position)
            if self.pos is not None and self.pos[2] <= target_z + 0.005:
                self.get_logger().info(f"Reached target Z position ({self.pos[2]:.3f}m) without contact.")
                return True
            
            # Periodic logging
            current_time = time.time()
            if current_time - last_log_time >= 2.0:
                if self.pos is not None:
                    self.get_logger().info(
                        f"Search: Z={self.pos[2]*1000:.1f}mm, "
                        f"Force Z={f_z:+.2f}N, X/Y={f_xy_mag:.2f}N"
                    )
                last_log_time = current_time
        
        return True
    
    def run_alignment_phase(self):
        """
        Alignment phase: Slow movement with X/Y force compliance.
        Uses one-by-one waypoints for responsive force-based adjustment.
        """
        self.get_logger().info("=" * 60)
        self.get_logger().info("ALIGNMENT PHASE: Slow movement with X/Y compliance")
        self.get_logger().info("=" * 60)
        
        self.low_force_start_time = None
        self.smoothed_force_xy = np.array([0.0, 0.0])
        
        dt = 0.6  # Trajectory duration for alignment steps
        last_log_time = time.time()
        waypoint_count = 0
        last_trajectory_time = 0.0
        min_trajectory_interval = 0.1
        
        while self.running and rclpy.ok():
            try:
                rclpy.spin_once(self, timeout_sec=0.1)
                
                # Check safety limits first (before any other checks)
                f_z_current = self.get_force_z()
                f_xy_raw = self.get_force_xy(use_deadband=False)  # Get raw forces for safety check
                f_xy_mag_raw = np.linalg.norm(f_xy_raw)
                
                # Check alignment completion
                if self.alignment_start_time is not None:
                    elapsed = time.time() - self.alignment_start_time
                    
                    # Check for low forces after minimum duration
                    if elapsed >= self.alignment_min_duration:
                        # Check if X/Y forces are low for sufficient duration
                        # Use smoothed forces to be consistent with movement calculations
                        f_xy_raw = self.get_force_xy(use_deadband=False)
                        # Update smoothed force (same as in get_next_waypoint)
                        self.smoothed_force_xy = (self.force_smoothing_alpha * f_xy_raw + 
                                                 (1.0 - self.force_smoothing_alpha) * self.smoothed_force_xy)
                        f_xy_mag = np.linalg.norm(self.smoothed_force_xy)
                        
                        if f_xy_mag < self.low_force_threshold:
                            if self.low_force_start_time is None:
                                self.low_force_start_time = time.time()
                            else:
                                low_force_duration = time.time() - self.low_force_start_time
                                if low_force_duration >= self.low_force_required_duration:
                                    self.get_logger().info("=" * 60)
                                    self.get_logger().info(f"ALIGNMENT COMPLETE: X/Y forces below {self.low_force_threshold:.1f}N for {low_force_duration:.1f}s")
                                    self.get_logger().info(f"Final X/Y force magnitude: {f_xy_mag:.2f} N")
                                    self.get_logger().info("=" * 60)
                                    break
                        else:
                            self.low_force_start_time = None  # Reset if forces increase
                
                # Get next waypoint
                waypoint = self.get_next_waypoint()
                
                if waypoint is None:
                    for _ in range(10):
                        if not self.running:
                            break
                        time.sleep(0.01)
                    continue
                
                x, y, z = waypoint
                
                # Use fixed orientation
                if self.fixed_rpy is None:
                    self.get_logger().warn("Fixed RPY not available, skipping waypoint")
                    time.sleep(0.1)
                    continue
                
                target_rpy = list(self.fixed_rpy)
                
                # Compute IK for this waypoint with fixed orientation
                joint_angles = compute_ik([x, y, z], target_rpy)
                
                if joint_angles is None:
                    self.get_logger().warn(
                        f"IK failed at waypoint {waypoint_count}: "
                        f"pos=[{x*1000:.1f}, {y*1000:.1f}, {z*1000:.1f}] mm"
                    )
                    for _ in range(10):
                        if not self.running:
                            break
                        time.sleep(0.01)
                    continue
                
                # Rate limit trajectory sends
                current_time = time.time()
                if current_time - last_trajectory_time < min_trajectory_interval:
                    time.sleep(0.05)
                    continue
                last_trajectory_time = current_time
                
                # Create trajectory with single waypoint
                trajectory = JointTrajectory()
                trajectory.joint_names = self.joint_names
                
                point = JointTrajectoryPoint()
                point.positions = [float(j) for j in joint_angles]
                point.velocities = [0.0] * 6
                point.time_from_start = Duration(sec=0, nanosec=600000000)  # 0.6 seconds
                trajectory.points.append(point)
                
                # Send trajectory
                goal = FollowJointTrajectory.Goal()
                goal.trajectory = trajectory
                goal.goal_time_tolerance = Duration(sec=1, nanosec=0)
                
                future = self.traj_client.send_goal_async(goal)
                rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
                
                if not self.running:
                    break
                
                goal_handle = future.result()
                if goal_handle and goal_handle.accepted:
                    self.current_goal_handle = goal_handle
                    for _ in range(20):
                        if not self.running:
                            break
                        time.sleep(0.01)
                else:
                    self.get_logger().warn(f"Trajectory goal rejected at waypoint {waypoint_count}")
                
                # Spin to update pose and wrench data
                rclpy.spin_once(self, timeout_sec=0.1)
                
            except KeyboardInterrupt:
                self.get_logger().info("KeyboardInterrupt detected in trajectory loop")
                self.running = False
                break
            
            # Periodically log status
            current_time = time.time()
            if current_time - last_log_time >= 2.0:
                if self.pos is not None:
                    f = self.force - self.baseline if self.force is not None else np.array([0.0, 0.0, 0.0])
                    # Log forces with deadband applied (same as used for compliance)
                    f_xy = self.get_force_xy()
                    f_xy_mag = np.linalg.norm(f_xy)
                    f_z = self.get_force_z()
                    self.get_logger().info(
                        f"Waypoint {waypoint_count} [ALIGNMENT]: "
                        f"Pos=[{self.pos[0]*1000:.1f}, {self.pos[1]*1000:.1f}, {self.pos[2]*1000:.1f}] mm, "
                        f"X/Y Force magnitude={f_xy_mag:.2f} N, "
                        f"X/Y Force=[{f_xy[0]:+.2f}, {f_xy[1]:+.2f}] N, "
                        f"Z Force={f_z:+.2f} N"
                    )
                last_log_time = current_time
            
            waypoint_count += 1


def main():
    parser = argparse.ArgumentParser(
        description='Force compliant move down - fixed orientation, moves down continuously, adjusts X/Y based on force',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real mode with defaults
  python3 perform_insert.py --mode real

  # Sim mode (step 2 from translate_for_assembly)
  python3 perform_insert.py --mode sim --object-name fork_orange --base-name base

  # Real mode with custom speed and compliance parameters
  python3 perform_insert.py --mode real --speed 0.01 --gain 5.0 --deadband 3.0

  # Real mode: Reverse force response directions
  python3 perform_insert.py --mode real --reverse

  # Real mode: Peg-in-hole insertion with custom Z threshold
  python3 perform_insert.py --mode real --z-threshold -8.0
        """
    )
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: sim (step 2 from translate_for_assembly) or real (force-compliant movement)')
    
    # Sim mode arguments
    parser.add_argument('--object-name', type=str, help='Name of the object being held (required in sim mode)')
    parser.add_argument('--base-name', type=str, help='Name of the base object (required in sim mode)')
    
    # Real mode arguments
    parser.add_argument('--speed', type=float, default=0.005,
                        help='Downward speed in m/s (default: 0.005 = 5mm/s, real mode only)')
    parser.add_argument('--gain', type=float, default=1.67,
                        help='X/Y compliance gain in mm/s per Newton (default: 1.67, real mode only)')
    parser.add_argument('--deadband', type=float, default=1.0,
                        help='X/Y force deadband in N (default: 1.0, real mode only)')
    parser.add_argument('--max-vel', type=float, default=15.0,
                        help='X/Y max compliance velocity in mm/s (default: 15.0, real mode only)')
    parser.add_argument('--reverse', action='store_true',
                        help='Reverse X/Y force response directions (default: False, real mode only)')
    parser.add_argument('--z-threshold', type=float, default=-10.0,
                        help='Z force threshold in N to detect contact (default: -10.0 N, negative = upward resistance, real mode only)')
    parser.add_argument('--xy-threshold', type=float, default=1.0,
                        help='Minimum X/Y force magnitude required to enter alignment mode (default: 1.0 N, real mode only)')
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'sim':
        if args.object_name is None or args.base_name is None:
            parser.error("In sim mode, --object-name and --base-name are required")
    
    rclpy.init()
    
    try:
        if args.mode == 'sim':
            node = PerformInsertController(
                mode=args.mode,
                object_name=args.object_name,
                base_name=args.base_name
            )
        else:
            node = PerformInsertController(
                mode=args.mode,
                speed=args.speed,
                gain=args.gain,
                deadband=args.deadband,
                max_vel=args.max_vel,
                reverse=args.reverse,
                z_threshold=args.z_threshold,
                xy_force_threshold=args.xy_threshold
            )
        
        # Give system time to stabilize
        time.sleep(0.5)
        
        # Execute trajectory
        try:
            if args.mode == 'sim':
                success = node.run()
                if success:
                    node.get_logger().info("Move down completed successfully!")
                else:
                    node.get_logger().error("Move down failed")
            else:
                node.run()
        except KeyboardInterrupt:
            # Set running flag to stop the loop
            if hasattr(node, 'running'):
                node.running = False
            node.get_logger().info("=" * 60)
            node.get_logger().info("TRAJECTORY INTERRUPTED BY USER")
            node.get_logger().info("=" * 60)
            if node.pos is not None:
                node.get_logger().info(f"Final position: X={node.pos[0]*1000:.2f}, Y={node.pos[1]*1000:.2f}, Z={node.pos[2]*1000:.2f} mm")
            if node.force is not None:
                f = node.force - node.baseline
                f_xy = node.get_force_xy()
                f_xy_mag = np.linalg.norm(f_xy)
                f_z = node.get_force_z()
                node.get_logger().info(f"Final X/Y force magnitude: {f_xy_mag:.2f} N")
                node.get_logger().info(f"Final Z force: {f_z:.2f} N")
                if node.contact_detected:
                    node.get_logger().info("Contact was detected during execution (alignment mode activated)")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
