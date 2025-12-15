#!/usr/bin/env python3
"""
Move Down Primitive for UR5e - ROS2 Version
Moves robot down in Z-axis with force monitoring on all three axes (X, Y, Z).
Stops when any axis force exceeds -10N (like in simulation).
Supports both simulation and real robot modes via --mode argument.
"""

import sys
import os

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import re
import json
import yaml
import argparse
import threading
import time

try:
    from primitives.utils.action_libraries import move_robust
except ImportError:
    # Fallback if action_libraries not in path
    move_robust = None

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================
GRIPPER_FORCE_THRESHOLD = 100.0  # Stop when gripper force exceeds this value (for sim mode)

# Real mode force thresholds:
# Use DELTA-based detection - stop when force/torque changes by threshold from baseline
# Monitors all 6 axes: Fx, Fy, Fz, Tx, Ty, Tz
# Z force uses directional threshold (negative = upward resistance)
FORCE_THRESHOLD = 20.0  # Stop when force/torque changes by this amount from baseline (Newtons)
Z_FORCE_THRESHOLD = -5.0  # Z force threshold in N (negative = upward force/resistance)

MOVEMENT_DURATION = 10.0  # Duration for smooth movement in seconds
FORCE_CHECK_INTERVAL = 0.02  # Check force every 20ms during movement
MOVE_DOWN_INCREMENT = 0.6  # Distance to move down per increment in meters
MIN_WORKSPACE_Z = 0.0  # Minimum Z position (workspace limit, below base is unreachable)
# =============================================================================

class MoveDown(Node):
    def __init__(self, target_height=None, mode='real'):
        super().__init__('move_down')
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        # Mode: 'sim' or 'real'
        self.mode = mode
        
        # Action client for trajectory control
        self.action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        # Target height (if None, will use current height - MOVE_DOWN_INCREMENT as default)
        self.target_height = target_height
        
        # Force monitoring - different for sim vs real
        if self.mode == 'sim':
            # Sim mode: gripper force (single value)
            self.current_gripper_force = 0.0
        else:
            # Real mode: force readings for all axes (forces and torques)
            self.current_force_x = 0.0
            self.current_force_y = 0.0
            self.current_force_z = 0.0
            self.current_torque_x = 0.0
            self.current_torque_y = 0.0
            self.current_torque_z = 0.0
            # Store baseline force values (calibrated over 2 seconds, like perform_insert)
            self.baseline_force_x = None
            self.baseline_force_y = None
            self.baseline_force_z = None
            self.baseline_torque_x = None
            self.baseline_torque_y = None
            self.baseline_torque_z = None
            # Grace period after movement starts before checking forces (avoid false positives from movement initiation)
            self.movement_start_time = None
            self.force_check_grace_period = 0.5  # Wait 0.5 seconds after movement starts
        self.force_threshold_reached = False
        
        # Current goal handle for cancellation
        self._current_goal_handle = None
        
        # Timer for force monitoring during movement
        self.force_monitor_timer = None
        
        # Timer for scheduling next movement (non-blocking)
        self.next_movement_timer = None
        
        # Movement state
        self.moving = False
        self.current_z_position = None  # Track current Z position for incremental movements
        
        # EE pose data storage
        self.ee_pose_received = False
        self.ee_position = None
        self.ee_quat = None
        
        # Current joint angles storage
        self.current_joint_angles = None
        self.joint_angles_received = False
        
        # Subscriber for force/torque data - use different topics and message types for sim vs real
        if self.mode == 'sim':
            # Simulation mode: use gripper force topic (Float64)
            self.gripper_force_sub = self.create_subscription(
                Float64,
                '/gripper_force',
                self.gripper_force_callback,
                10
            )
        else:
            # Real mode: use force/torque sensor topic (WrenchStamped)
            self.force_sub = self.create_subscription(
                WrenchStamped,
                '/force_torque_sensor_broadcaster/wrench',
                self.force_callback,
                10
            )
        
        # Subscriber for EE pose data
        # Use VOLATILE durability (default for most publishers) to avoid QoS incompatibility warnings
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,  # Changed from TRANSIENT_LOCAL to match most publishers
            depth=10
        )
        
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            qos_profile
        )
        
        # Subscriber for joint states to get current joint angles (use as IK seed)
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.action_client.wait_for_server()
        
        # Log mode
        self.get_logger().info(f"Using {self.mode.upper()} mode")
        
        # Execute movement
        self.move_down()
    
    def ee_pose_callback(self, msg: PoseStamped):
        """Callback for end-effector pose data"""
        self.ee_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.ee_quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        self.ee_pose_received = True
    
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

    def quaternion_to_rpy(self, x, y, z, w):
        """Convert quaternion to roll, pitch, yaw in degrees - same as other primitives"""
        import math
        
        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
        
        # Pitch
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.degrees(math.copysign(math.pi / 2, sinp))
        else:
            pitch = math.degrees(math.asin(sinp))
        
        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        
        return [roll, pitch, yaw]

    def read_current_ee_pose(self):
        """Read current end-effector pose and joint angles using ROS2 subscriber"""
        # Reset the flags
        self.ee_pose_received = False
        self.joint_angles_received = False
        
        # Wait for both pose and joint angles to arrive (with timeout)
        timeout_count = 0
        max_timeout = 100  # 10 seconds (100 * 0.1s)
        
        while rclpy.ok() and (not self.ee_pose_received or not self.joint_angles_received) and timeout_count < max_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout_count += 1
            
            if timeout_count % 10 == 0:  # Log every second
                status = []
                if not self.ee_pose_received:
                    status.append("EE pose")
                if not self.joint_angles_received:
                    status.append("joint angles")
                self.get_logger().debug(f"Waiting for {' and '.join(status)}... ({timeout_count * 0.1:.1f}s)")
        
        if not self.ee_pose_received:
            self.get_logger().error("Timeout waiting for EE pose message")
            return None
        
        if not self.joint_angles_received:
            self.get_logger().error("Timeout waiting for joint angles message")
            return None
        
        if self.ee_position is None or self.ee_quat is None:
            self.get_logger().error("EE pose data is None")
            return None
        
        if self.current_joint_angles is None:
            self.get_logger().error("Joint angles data is None")
            return None
        
        # Extract position and orientation
        position = self.ee_position.tolist()
        orientation = self.ee_quat.tolist()
        
        return {
            'position': position,
            'orientation': orientation
        }

    def gripper_force_callback(self, msg: Float64):
        """Callback for gripper force data (sim mode)"""
        self.current_gripper_force = msg.data
        
        # Check if force threshold is exceeded
        if self.current_gripper_force > GRIPPER_FORCE_THRESHOLD and not self.force_threshold_reached:
            self.get_logger().warn(f"Gripper force threshold reached! Force: {self.current_gripper_force:.2f} (threshold: {GRIPPER_FORCE_THRESHOLD})")
            self.force_threshold_reached = True
            self.emergency_stop()
    
    def force_callback(self, msg: WrenchStamped):
        """Callback for force/torque sensor data - monitors all axes (real mode)
        
        Uses ABSOLUTE force detection: stops when any force/torque magnitude exceeds threshold.
        Handles both positive and negative forces (absolute value comparison).
        """
        # Extract force and torque components for all axes
        self.current_force_x = msg.wrench.force.x
        self.current_force_y = msg.wrench.force.y
        self.current_force_z = msg.wrench.force.z
        self.current_torque_x = msg.wrench.torque.x
        self.current_torque_y = msg.wrench.torque.y
        self.current_torque_z = msg.wrench.torque.z

    def start_force_monitoring(self):
        """Start monitoring force during trajectory execution (non-blocking)"""
        import time
        if self.mode == 'real':
            # NON-BLOCKING calibration: Use brief sleep like old code to get instant baseline
            # This avoids blocking the ROS event loop which prevents goal_result from being called
            time.sleep(0.1)  # Brief wait to ensure we have current readings
            
            # Use current values as baseline (same as old code)
            self.baseline_force_x = self.current_force_x
            self.baseline_force_y = self.current_force_y
            self.baseline_force_z = self.current_force_z
            self.baseline_torque_x = self.current_torque_x
            self.baseline_torque_y = self.current_torque_y
            self.baseline_torque_z = self.current_torque_z
            
            # Record movement start time for grace period
            self.movement_start_time = time.time()
            
        if self.force_monitor_timer is None:
            self.force_monitor_timer = self.create_timer(FORCE_CHECK_INTERVAL, self.check_force_during_execution)

    def stop_force_monitoring(self):
        """Stop force monitoring"""
        if self.force_monitor_timer:
            self.force_monitor_timer.cancel()
            self.force_monitor_timer = None
    
    def schedule_next_movement(self):
        """Schedule the next movement using a ROS2 one-shot timer (non-blocking)"""
        if self.next_movement_timer:
            self.next_movement_timer.cancel()
            self.next_movement_timer = None
        
        def timer_callback():
            if self.next_movement_timer:
                self.next_movement_timer.cancel()
                self.next_movement_timer = None
            self.move_down_continue()
        
        self.next_movement_timer = self.create_timer(0.3, timer_callback)

    def check_force_during_execution(self):
        """Check force during trajectory execution and cancel if threshold exceeded"""
        if not self.moving:
            return
        
        # Grace period: don't check forces immediately after movement starts (avoid false positives from movement initiation)
        if self.mode == 'real' and self.movement_start_time is not None:
            import time
            elapsed_since_start = time.time() - self.movement_start_time
            if elapsed_since_start < self.force_check_grace_period:
                return  # Skip force check during grace period
        
        if self.mode == 'sim':
            # Sim mode: check gripper force
            if self.current_gripper_force > GRIPPER_FORCE_THRESHOLD:
                self.get_logger().error(f"EMERGENCY STOP: Gripper force threshold exceeded during movement! Force: {self.current_gripper_force:.2f}")
                self.force_threshold_reached = True
                self.emergency_stop()
            else:
                # Debug: log force values during movement
                self.get_logger().debug(f"Force monitoring: Gripper force = {self.current_gripper_force:.2f} (threshold: {GRIPPER_FORCE_THRESHOLD})")
        else:
            # Real mode: check forces using baseline-subtracted detection for ALL axes
            # Monitors all 6 axes: Fx, Fy, Fz, Tx, Ty, Tz
            # Subtracts baseline (calibrated over 2 seconds) before checking thresholds
            # Z force uses directional check (negative = upward resistance)
            # Other axes use magnitude checks
            
            # Check if we have baseline values set
            if (self.baseline_force_x is None or 
                self.baseline_force_y is None or 
                self.baseline_force_z is None):
                return  # Wait for baseline values to be set
            
            # Calculate forces after baseline removal (like perform_insert)
            f_x = self.current_force_x - self.baseline_force_x
            f_y = self.current_force_y - self.baseline_force_y
            f_z = self.current_force_z - self.baseline_force_z
            t_x = self.current_torque_x - self.baseline_torque_x
            t_y = self.current_torque_y - self.baseline_torque_y
            t_z = self.current_torque_z - self.baseline_torque_z
            
            # Check Z force with directional threshold (negative = upward resistance)
            if f_z <= Z_FORCE_THRESHOLD:
                self.get_logger().error(
                    f"EMERGENCY STOP: Z force threshold exceeded! "
                    f"Z force (after baseline): {f_z:.2f}N (threshold: {Z_FORCE_THRESHOLD}N, "
                    f"current: {self.current_force_z:.2f}N, baseline: {self.baseline_force_z:.2f}N)"
                )
                self.force_threshold_reached = True
                self.emergency_stop()
                return  # Exit immediately after emergency stop
            
            # Check other axes (Fx, Fy, Tx, Ty, Tz) with magnitude thresholds
            axes = [
                (abs(f_x), "Fx", f_x, self.current_force_x, self.baseline_force_x),
                (abs(f_y), "Fy", f_y, self.current_force_y, self.baseline_force_y),
                (abs(t_x), "Tx", t_x, self.current_torque_x, self.baseline_torque_x),
                (abs(t_y), "Ty", t_y, self.current_torque_y, self.baseline_torque_y),
                (abs(t_z), "Tz", t_z, self.current_torque_z, self.baseline_torque_z),
            ]
            
            # Check each axis
            for magnitude, axis_name, force_value, current_value, baseline_value in axes:
                if magnitude > FORCE_THRESHOLD:
                    self.get_logger().error(
                        f"EMERGENCY STOP: {axis_name} force/torque exceeded threshold! "
                        f"{axis_name} (after baseline): {force_value:.2f}N (magnitude: {magnitude:.2f}N, "
                        f"current: {current_value:.2f}N, baseline: {baseline_value:.2f}N, "
                        f"threshold: {FORCE_THRESHOLD}N)"
                    )
                    self.force_threshold_reached = True
                    self.emergency_stop()
                    return  # Exit immediately after emergency stop

    def emergency_stop(self):
        """Emergency stop - cancel current trajectory and stop immediately, then exit"""
        self.moving = False
        self.get_logger().error("EMERGENCY STOP: Force threshold exceeded. Exiting...")
        
        # Stop force monitoring
        self.stop_force_monitoring()
        
        # Cancel the current goal if it exists
        if self._current_goal_handle is not None:
            try:
                self._current_goal_handle.cancel_goal_async()
            except Exception as e:
                self.get_logger().error(f"Failed to cancel trajectory: {e}")
        
        # Force immediate exit - use a timer to shutdown after brief delay to allow logging
        def delayed_shutdown():
            try:
                rclpy.shutdown()
            except:
                pass
            # Force exit if rclpy.shutdown() doesn't work
            os._exit(0)
        
        # Start shutdown in a separate thread after 100ms to allow log messages to flush
        shutdown_timer = threading.Timer(0.1, delayed_shutdown)
        shutdown_timer.daemon = True
        shutdown_timer.start()

    def move_down_continue(self):
        """Continue moving down using stored position (non-blocking, called from timer)"""
        # Reset baselines for next movement segment (will be recalibrated in start_force_monitoring)
        if self.mode == 'real':
            self.baseline_force_x = None
            self.baseline_force_y = None
            self.baseline_force_z = None
            self.baseline_torque_x = None
            self.baseline_torque_y = None
            self.baseline_torque_z = None
        
        # Get orientation from callback data (needed for IK)
        if self.ee_quat is None:
            for _ in range(3):
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.ee_quat is not None:
                    break
            if self.ee_quat is None:
                self.get_logger().warn("No orientation data, using default")
                current_quat = [-0.001, 0.707, -0.707, 0.001]
            else:
                current_quat = self.ee_quat.tolist()
        else:
            current_quat = self.ee_quat.tolist()
        
        # Use stored Z position (updated from previous target), X/Y from callback if available
        if self.ee_position is not None:
            current_pos = [self.ee_position[0], self.ee_position[1], self.current_z_position]
        else:
            current_pos = [0, 0, self.current_z_position]
        
        self._execute_move_down(current_pos, current_quat)
    
    def move_down(self):
        """Move down while maintaining current position and orientation"""
        # Reset baseline forces for new movement sequence (will be recalibrated in start_force_monitoring)
        if self.mode == 'real':
            self.baseline_force_x = None
            self.baseline_force_y = None
            self.baseline_force_z = None
            self.baseline_torque_x = None
            self.baseline_torque_y = None
            self.baseline_torque_z = None
        
        # Initialize current Z position if not set
        if self.current_z_position is None:
            # Read current end-effector pose
            pose_data = self.read_current_ee_pose()
            
            if pose_data is None:
                self.get_logger().error("Could not read current end-effector pose")
                rclpy.shutdown()
                return
                
            current_pos = pose_data['position']
            current_quat = pose_data['orientation']
            self.current_z_position = current_pos[2]
        else:
            # Use stored position and read fresh pose for orientation
            pose_data = self.read_current_ee_pose()
            
            if pose_data is None:
                self.get_logger().error("Could not read current end-effector pose")
                rclpy.shutdown()
                return
                
            # Use actual current position from EE pose callback
            if self.ee_position is not None:
                current_pos = self.ee_position.tolist()
                self.current_z_position = current_pos[2]  # Update from actual position
            else:
                current_pos = [0, 0, self.current_z_position]
            current_quat = pose_data['orientation']
        
        # Continue with movement calculation
        self._execute_move_down(current_pos, current_quat)
    
    def _execute_move_down(self, current_pos, current_quat):
        """Execute the actual move down calculation and trajectory sending"""
        from scipy.spatial.transform import Rotation as Rot
        
        target_rotation = Rot.from_quat(current_quat)
        target_rot_matrix = target_rotation.as_matrix()

        target_position = list(current_pos)
        
        if self.target_height is not None:
            target_position[2] = self.target_height
        else:
            # Move to base workspace height (MIN_WORKSPACE_Z)
            target_position[2] = MIN_WORKSPACE_Z
            self.current_z_position = target_position[2]  # Update for next movement
        
        # Check if we're already at the target (or very close)
        current_z = current_pos[2]
        distance_to_target = abs(current_z - target_position[2])
        if distance_to_target < 0.001:  # Already at target (within 1mm)
            self.get_logger().info(f"Already at target Z position ({current_z:.3f}m). Exiting.")
            rclpy.shutdown()
            return
        
        # Check workspace limit (safety check)
        if target_position[2] < MIN_WORKSPACE_Z:
            self.get_logger().error(f"Target Z position ({target_position[2]:.3f}m) is below workspace limit ({MIN_WORKSPACE_Z}m). Cannot proceed.")
            rclpy.shutdown()
            return

        # Compute inverse kinematics
        try:
            from scipy.optimize import minimize
            from primitives.utils.ik_solver import ik_objective_quaternion, forward_kinematics, dh_params
            
            target_pose = np.eye(4)
            target_pose[:3, 3] = target_position
            target_pose[:3, :3] = target_rot_matrix
            
            if self.current_joint_angles is None:
                self.get_logger().error("Current joint angles not available! Cannot compute IK.")
                rclpy.shutdown()
                return
            
            q_guess = self.current_joint_angles.copy()
            
            joint_angles = None
            best_result = None
            best_cost = float('inf')
            max_tries = 5
            dx = 0.001
            
            for i in range(max_tries):
                perturbed_position = np.array(target_position).copy()
                perturbed_position[0] += i * dx
                
                perturbed_pose = target_pose.copy()
                perturbed_pose[:3, 3] = perturbed_position
                
                result = minimize(ik_objective_quaternion, q_guess, args=(perturbed_pose,), 
                                method='L-BFGS-B', bounds=[(-np.pi, np.pi)] * 6)
                
                if result.success:
                    cost = ik_objective_quaternion(result.x, perturbed_pose)
                    
                    if cost < 0.01:
                        joint_angles = result.x
                        break
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_result = result.x
            
            if joint_angles is None and best_result is not None and best_cost < 0.1:
                self.get_logger().info(f"Using best IK solution with cost={best_cost:.6f}")
                joint_angles = best_result
                T_result = forward_kinematics(dh_params, joint_angles)
                orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
                self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
            
            # If IK failed at target Z, try incrementally higher Z values until IK succeeds
            if joint_angles is None:
                self.get_logger().warn(f"IK failed at target Z={target_position[2]:.3f}m. Trying higher Z values...")
                z_increment = 0.05  # Try 5cm increments
                max_z_attempts = 10  # Try up to 0.5m above target
                
                for z_attempt in range(1, max_z_attempts + 1):
                    test_z = target_position[2] + z_attempt * z_increment
                    self.get_logger().info(f"Trying IK at Z={test_z:.3f}m (attempt {z_attempt}/{max_z_attempts})")
                    
                    test_position = np.array(target_position).copy()
                    test_position[2] = test_z
                    test_pose = target_pose.copy()
                    test_pose[:3, 3] = test_position
                    
                    # Try IK with current joint angles as seed
                    result = minimize(ik_objective_quaternion, q_guess, args=(test_pose,), 
                                    method='L-BFGS-B', bounds=[(-np.pi, np.pi)] * 6)
                    
                    if result.success:
                        cost = ik_objective_quaternion(result.x, test_pose)
                        if cost < 0.01:
                            self.get_logger().info(f"IK succeeded at Z={test_z:.3f}m (cost={cost:.6f})")
                            joint_angles = result.x
                            target_position[2] = test_z  # Update target to reachable Z
                            self.get_logger().info(f"Updated target Z to {test_z:.3f}m (lowest reachable height)")
                            break
                
                if joint_angles is None:
                    self.get_logger().error("IK failed: couldn't compute move down position even after trying higher Z values")
                    rclpy.shutdown()
                    return
                
            self.get_logger().info(f"Computed joint angles: {joint_angles}")
            
            # Create and send trajectory
            point = JointTrajectoryPoint(
                positions=[float(x) for x in joint_angles],
                velocities=[0.0] * 6,
                time_from_start=Duration(sec=int(MOVEMENT_DURATION))
            )
            
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            traj.points = [point]
            
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj
            goal.goal_time_tolerance = Duration(sec=1)
            
            self.moving = True
            
            # Clear old futures/callbacks before sending new goal
            self._current_goal_handle = None
            self._send_goal_future = None
            self._get_result_future = None
            
            self.get_logger().info("Trajectory sent and accepted")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)
            
        except Exception as e:
            self.get_logger().error(f"Failed to compute IK: {e}")
            rclpy.shutdown()

    def goal_response(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            self.stop_force_monitoring()
            self.moving = False
            rclpy.shutdown()
            return

        self._current_goal_handle = goal_handle
        
        # Start force monitoring AFTER goal is accepted
        self.start_force_monitoring()
        
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result)

    def goal_result(self, future):
        """Handle goal result"""
        try:
            result = future.result()
            self.stop_force_monitoring()
            self.moving = False
            self._current_goal_handle = None
            
            # Handle different trajectory completion statuses
            if result.status == 1:  # SUCCEEDED
                pass  # Continue to check force threshold
            elif result.status == 5:  # PREEMPTED
                pass  # Continue to check force threshold
            elif result.status == 4:  # ABORTED
                pass  # Continue to check force threshold
            else:
                self.get_logger().error(f"Trajectory failed with status: {result.status}")
                rclpy.shutdown()
                return
            
            # Check if force threshold was reached
            if self.force_threshold_reached:
                self.get_logger().info("Movement completed successfully: Force threshold reached. Stopping.")
                if self.ee_position is not None:
                    self.current_z_position = self.ee_position[2]
                rclpy.shutdown()
            else:
                # Check if we're already at the target before continuing
                if self.ee_position is not None:
                    current_z = self.ee_position[2]
                    target_z = MIN_WORKSPACE_Z if self.target_height is None else self.target_height
                    distance_to_target = abs(current_z - target_z)
                    
                    if distance_to_target < 0.001:  # Already at target (within 1mm)
                        self.get_logger().info("Movement completed successfully")
                        rclpy.shutdown()
                        return
                
                # Continue with next movement
                self.force_threshold_reached = False
                self.schedule_next_movement()
                
        except Exception as e:
            self.get_logger().error(f"Error in goal result callback: {e}")
            rclpy.shutdown()
        finally:
            self.stop_force_monitoring()
            self.moving = False

def main(args=None):
    parser = argparse.ArgumentParser(description='Move Down Primitive with force monitoring on all axes')
    parser.add_argument('--height', type=float, default=None,
                       help='Target height in meters (optional, defaults to moving down in increments until force threshold)')
    parser.add_argument('--mode', type=str, default='real', choices=['sim', 'real'],
                       help='Mode: "sim" for simulation, "real" for real robot (default: real)')
    
    # Parse known args to avoid conflicts with ROS2
    known_args, unknown_args = parser.parse_known_args()
    
    rclpy.init(args=args)
    node = MoveDown(known_args.height, known_args.mode)
    
    try:
        # Use regular spin() - rclpy.shutdown() will cause spin to exit
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Move down interrupted by user")
    except Exception as e:
        node.get_logger().error(f"Error in move down: {e}")
    finally:
        try:
            rclpy.shutdown()
        except RuntimeError:
            pass  # Context already shut down

if __name__ == '__main__':
    main()