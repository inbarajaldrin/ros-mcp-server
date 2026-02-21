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
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
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
import subprocess

from primitives.shared.config import TABLE_HEIGHT, TABLE_COLLISION_MARGIN_SIDEWAYS, TABLE_COLLISION_MARGIN_FACEDOWN, GRIPPER_CENTER_TOOL_OFFSET
from primitives.shared.ik import compute_cartesian_waypoints_ik
from primitives.shared.velocity_profiles import s_curve_profile, compute_duration
from primitives.shared.collision import compute_all_joint_positions, check_collision_with_table, segment_distance, check_self_collision

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================
# Sim mode: F/T sensor via /force_torque_sensor_broadcaster/wrench_sim (WrenchStamped)
# Uses same force detection logic and message type as real mode but with different thresholds
# Simulated forces may have different scaling/characteristics than real sensor

# Sim mode force thresholds (tuned for Isaac Sim get_measured_joint_forces()):
# Sim baseline is ~-14 N (gravity), contact causes POSITIVE delta (+40 to +500 N)
SIM_FORCE_THRESHOLD = 50.0  # Force change threshold for Fx, Fy, Tx, Ty, Tz (N/Nm)
SIM_Z_FORCE_THRESHOLD = 10.0  # Z force threshold (positive = contact pushback in sim)

# Real mode force thresholds:
# Use DELTA-based detection - stop when force/torque changes by threshold from baseline
# Monitors all 6 axes: Fx, Fy, Fz, Tx, Ty, Tz
# Z force uses directional threshold (negative = upward resistance)
FORCE_THRESHOLD = 20.0  # Stop when force/torque changes by this amount from baseline (Newtons)
Z_FORCE_THRESHOLD = -5.0  # Z force threshold in N (negative = upward force/resistance)

FORCE_CHECK_INTERVAL = 0.02  # Check force every 20ms during movement
MOVE_DOWN_INCREMENT = 0.6  # Distance to move down per increment in meters
MIN_WORKSPACE_Z = TABLE_HEIGHT  # Absolute minimum flange Z (workspace limit)
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
        
        # Force monitoring - both sim and real use WrenchStamped (6-DOF F/T sensor)
        # Sim: /force_torque_sensor_broadcaster/wrench_sim
        # Real: /force_torque_sensor_broadcaster/wrench
        self.current_force_x = 0.0
        self.current_force_y = 0.0
        self.current_force_z = 0.0
        self.current_torque_x = 0.0
        self.current_torque_y = 0.0
        self.current_torque_z = 0.0
        # Store baseline force values (calibrated at movement start)
        self.baseline_force_x = None
        self.baseline_force_y = None
        self.baseline_force_z = None
        self.baseline_torque_x = None
        self.baseline_torque_y = None
        self.baseline_torque_z = None
        # Grace period after movement starts before checking forces
        self.movement_start_time = None
        self.force_check_grace_period = 0.5  # Wait 0.5 seconds after movement starts
        self.force_threshold_reached = False
        self.error_message = None  # Track error for JSON output

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
        
        # Subscriber for force/torque data - different message types for sim vs real
        if self.mode == 'sim':
            # Simulation mode: full 6-DOF WrenchStamped from Isaac Sim extension
            self.force_sub = self.create_subscription(
                WrenchStamped,
                '/force_torque_sensor_broadcaster/wrench_sim',
                self.force_callback,
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

    def zero_force_sensor(self):
        """Zero the force/torque sensor via service call"""
        try:
            self.get_logger().info("Zeroing force sensor...")
            result = subprocess.run(
                ['ros2', 'service', 'call', '/io_and_status_controller/zero_ftsensor',
                 'std_srvs/srv/Trigger'],
                capture_output=True, text=True, timeout=10
            )
            if 'success=True' in result.stdout or 'success: true' in result.stdout.lower():
                self.get_logger().info("Force sensor zeroed successfully")
                return True
            else:
                self.get_logger().warn(f"Force sensor zero may have failed: {result.stdout.strip()}")
                return False
        except Exception as e:
            self.get_logger().warn(f"Could not zero force sensor: {e}")
            return False

    def read_current_ee_pose(self):
        """Read current end-effector pose and joint angles using ROS2 subscriber with retry"""
        max_retries = 5
        timeout_sec = 15.0

        for attempt in range(max_retries):
            # Reset the flags
            self.ee_pose_received = False
            self.joint_angles_received = False

            if attempt > 0:
                self.get_logger().info(f"Retrying EE pose read (attempt {attempt + 1}/{max_retries})...")
                # Brief delay before retry
                time.sleep(0.5)

            # Wait for both pose and joint angles to arrive (with timeout)
            timeout_count = 0
            max_timeout = int(timeout_sec / 0.1)  # Convert to count

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

            # Check if we got the data
            if self.ee_pose_received and self.joint_angles_received:
                if self.ee_position is not None and self.ee_quat is not None and self.current_joint_angles is not None:
                    # Success!
                    position = self.ee_position.tolist()
                    orientation = self.ee_quat.tolist()
                    return {
                        'position': position,
                        'orientation': orientation
                    }

            # Log what's missing
            missing = []
            if not self.ee_pose_received or self.ee_position is None or self.ee_quat is None:
                missing.append("EE pose")
            if not self.joint_angles_received or self.current_joint_angles is None:
                missing.append("joint angles")
            self.get_logger().warn(f"Timeout waiting for: {', '.join(missing)} (attempt {attempt + 1}/{max_retries})")

        # All retries exhausted
        self.get_logger().error(f"Failed to read EE pose after {max_retries} attempts")
        self.get_logger().error("SUGGESTION: Try running the same command again. The issue is often transient and succeeds on retry.")
        return None

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

        # NON-BLOCKING calibration: Use brief sleep to get instant baseline
        # This avoids blocking the ROS event loop which prevents goal_result from being called
        time.sleep(0.1)  # Brief wait to ensure we have current readings

        # Use current values as baseline (same approach for both sim and real)
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
        """Check force during trajectory execution and cancel if threshold exceeded.

        Both sim and real modes use the same WrenchStamped-based detection:
        - Sim: /force_torque_sensor_broadcaster/wrench_sim - uses SIM_* thresholds
        - Real: /force_torque_sensor_broadcaster/wrench - uses real thresholds
        """
        if not self.moving:
            return

        # Grace period: don't check forces immediately after movement starts
        if self.movement_start_time is not None:
            import time
            elapsed_since_start = time.time() - self.movement_start_time
            if elapsed_since_start < self.force_check_grace_period:
                return  # Skip force check during grace period

        # Check if we have baseline values set
        if (self.baseline_force_x is None or
            self.baseline_force_y is None or
            self.baseline_force_z is None):
            return  # Wait for baseline values to be set

        # Select thresholds based on mode
        if self.mode == 'sim':
            z_threshold = SIM_Z_FORCE_THRESHOLD
            force_threshold = SIM_FORCE_THRESHOLD
        else:
            z_threshold = Z_FORCE_THRESHOLD
            force_threshold = FORCE_THRESHOLD

        # Calculate forces after baseline removal
        f_x = self.current_force_x - self.baseline_force_x
        f_y = self.current_force_y - self.baseline_force_y
        f_z = self.current_force_z - self.baseline_force_z
        t_x = self.current_torque_x - self.baseline_torque_x
        t_y = self.current_torque_y - self.baseline_torque_y
        t_z = self.current_torque_z - self.baseline_torque_z

        # Check Z force with directional threshold
        # Real mode: negative delta (upward resistance from F/T sensor)
        # Sim mode: positive delta (contact pushback from joint forces)
        if self.mode == 'sim':
            z_triggered = f_z >= z_threshold  # Sim: positive delta on contact
        else:
            z_triggered = f_z <= z_threshold  # Real: negative delta on contact

        if z_triggered:
            self.get_logger().info(
                f"Contact detected: Z force threshold reached. "
                f"Z force (after baseline): {f_z:.2f}N (threshold: {z_threshold}N, "
                f"current: {self.current_force_z:.2f}N, baseline: {self.baseline_force_z:.2f}N)"
            )
            self.force_threshold_reached = True
            self.contact_detected_stop()
            return  # Exit immediately after contact detected

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
            if magnitude > force_threshold:
                self.get_logger().info(
                    f"Contact detected: {axis_name} force/torque threshold reached. "
                    f"{axis_name} (after baseline): {force_value:.2f}N (magnitude: {magnitude:.2f}N, "
                    f"current: {current_value:.2f}N, baseline: {baseline_value:.2f}N, "
                    f"threshold: {force_threshold}N)"
                )
                self.force_threshold_reached = True
                self.contact_detected_stop()
                return  # Exit immediately after contact detected

    def contact_detected_stop(self):
        """Graceful stop when contact is detected - this is SUCCESS, not failure.

        Contact detection is the goal of move_down, so reaching force threshold
        during trajectory execution is a successful outcome.
        """
        self.moving = False
        self.get_logger().info("Contact detected: Force threshold reached during movement. Stopping trajectory.")

        # Stop force monitoring
        self.stop_force_monitoring()

        # Cancel the current goal if it exists
        if self._current_goal_handle is not None:
            try:
                self._current_goal_handle.cancel_goal_async()
            except Exception as e:
                self.get_logger().warn(f"Could not cancel trajectory (may already be stopped): {e}")

        # Update current Z position from actual EE pose
        if self.ee_position is not None:
            self.current_z_position = self.ee_position[2]

        # Graceful shutdown - use a timer to shutdown after brief delay to allow logging
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

    def emergency_stop(self):
        """Emergency stop for actual failures (not contact detection).

        Note: Force threshold during move_down is SUCCESS (contact detected),
        so this method should only be used for actual error conditions.
        """
        self.moving = False
        if not self.error_message:
            self.error_message = "EMERGENCY STOP: Unexpected error"
        self.get_logger().error(f"EMERGENCY STOP: {self.error_message}")

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
                self.error_message = "Could not read current end-effector pose"
                self.get_logger().error(self.error_message)
                rclpy.shutdown()
                return

            current_pos = pose_data['position']
            current_quat = pose_data['orientation']
            self.current_z_position = current_pos[2]
        else:
            # Use stored position and read fresh pose for orientation
            pose_data = self.read_current_ee_pose()

            if pose_data is None:
                self.error_message = "Could not read current end-effector pose"
                self.get_logger().error(self.error_message)
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

        # Zero force sensor before movement starts (only on first movement)
        if not self.moving and self.baseline_force_z is None:
            self.zero_force_sensor()
            time.sleep(0.5)  # Wait for sensor to settle after zeroing

        target_rotation = Rot.from_quat(current_quat)
        target_rot_matrix = target_rotation.as_matrix()

        # Compute dynamic minimum flange Z based on current gripper orientation.
        # The gripper center is GRIPPER_CENTER_TOOL_OFFSET along the tool Z-axis from the flange.
        # The vertical component depends on how much the tool Z-axis projects onto world Z.
        #   face-down (z_tool_z=-1): TABLE_HEIGHT + MARGIN + 0.23
        #   sideways  (z_tool_z= 0): TABLE_HEIGHT + MARGIN
        #   face-up   (z_tool_z=+1): TABLE_HEIGHT + MARGIN - 0.23
        z_tool_z = target_rot_matrix[2, 2]  # Z-component of tool Z-axis in world frame
        margin = TABLE_COLLISION_MARGIN_FACEDOWN if z_tool_z < -0.5 else TABLE_COLLISION_MARGIN_SIDEWAYS
        min_flange_z = TABLE_HEIGHT + margin - GRIPPER_CENTER_TOOL_OFFSET[2] * z_tool_z
        self._min_flange_z = min_flange_z  # Store for convergence check

        target_position = list(current_pos)

        if self.target_height is not None:
            target_position[2] = self.target_height
        else:
            # Move to minimum flange Z (gripper center at table surface)
            target_position[2] = min_flange_z
            self.current_z_position = target_position[2]  # Update for next movement

        # Check if we're already at the target (or very close)
        current_z = current_pos[2]
        distance_to_target = abs(current_z - target_position[2])
        if distance_to_target < 0.001:  # Already at target (within 1mm)
            self.get_logger().info(f"Already at target Z position ({current_z:.3f}m). Exiting.")
            # Use delayed shutdown with os._exit() to ensure clean exit
            # (same pattern as contact_detected_stop)
            def delayed_shutdown():
                try:
                    rclpy.shutdown()
                except:
                    pass
                os._exit(0)

            shutdown_timer = threading.Timer(0.1, delayed_shutdown)
            shutdown_timer.daemon = True
            shutdown_timer.start()
            return

        # Check workspace limit (safety check)
        if target_position[2] < min_flange_z:
            self.error_message = f"Target Z position ({target_position[2]:.3f}m) is below workspace limit ({min_flange_z:.3f}m for current orientation). Cannot proceed."
            self.get_logger().error(self.error_message)
            rclpy.shutdown()
            return

        # Compute inverse kinematics
        try:
            if self.current_joint_angles is None:
                self.error_message = "Current joint angles not available! Cannot compute IK."
                self.get_logger().error(self.error_message)
                rclpy.shutdown()
                return

            num_waypoints = 60

            self.get_logger().info("Computing dense IK waypoints (Jacobian)...")
            waypoints = compute_cartesian_waypoints_ik(
                self.current_joint_angles, target_position[2],
                num_waypoints=num_waypoints
            )
            if waypoints is None:
                self.error_message = "Motion planning failed: no collision-free descent path to the target height could be computed"
                self.get_logger().error(self.error_message)
                rclpy.shutdown()
                return

            all_joint_angles = [self.current_joint_angles.copy()] + list(waypoints)

            joint_dist = float(np.max(np.abs(np.array(waypoints[-1]) - np.array(self.current_joint_angles))))
            total_duration = compute_duration(
                joint_distance=joint_dist, cartesian_distance=distance_to_target, profile='s_curve'
            )
            self.get_logger().info(f"Duration: {total_duration:.2f}s (cart={distance_to_target:.3f}m, joint={joint_dist:.2f}rad)")

            profile = s_curve_profile(all_joint_angles, total_duration)
            traj_points = []
            for positions, velocities, t_i in profile:
                point = JointTrajectoryPoint(
                    positions=positions,
                    velocities=velocities,
                    time_from_start=Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
                )
                traj_points.append(point)

            self.get_logger().info(f"Generated {len(traj_points)} dense waypoints with s-curve velocity profile")

            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            traj.points = traj_points

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
            self.error_message = f"Failed to compute IK: {e}"
            self.get_logger().error(self.error_message)
            rclpy.shutdown()

    def goal_response(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.error_message = "External control program stopped or robot in protective stop"
            self.get_logger().error(self.error_message)
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
                result_msg = result.result
                if result_msg.error_code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
                    self.error_message = "Velocity or acceleration limits exceeded. The required velocity to reach the target exceeds joint velocity limits. Enable robot in URcap to fix this."
                else:
                    self.error_message = f"Trajectory failed with status: {result.status}"
                self.get_logger().error(self.error_message)
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
                    target_z = getattr(self, '_min_flange_z', MIN_WORKSPACE_Z) if self.target_height is None else self.target_height
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

    def output_result_json(self, movement_type="move_down"):
        """Output result in JSON format for MCP server"""
        if self.error_message:
            # Failure format
            result = {
                "result": "failure",
                "mode": self.mode,
                "movement_type": movement_type,
                "error": self.error_message
            }
        else:
            # Success format - minimal output (no position/orientation)
            result = {
                "result": "success",
                "mode": self.mode,
                "movement_type": movement_type
            }

        output_result(result)


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


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
            node.output_result_json()
            node.action_client.destroy()
            node.destroy_node()
            rclpy.shutdown()
        except RuntimeError:
            pass  # Context already shut down

if __name__ == '__main__':
    main()