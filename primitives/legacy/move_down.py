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

    def compute_all_joint_positions(self, joint_angles):
        """
        Compute the 3D positions of all joints given joint angles.
        Returns a list of [x, y, z] positions for each joint.
        """
        # UR5e DH parameters (from ik_solver.py)
        dh_params = [
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
        for i, (theta, d, a, alpha) in enumerate(dh_params):
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

    def gripper_force_callback(self, msg: Float64):
        """Callback for gripper force data (sim mode)"""
        self.current_gripper_force = msg.data

        # Check if force threshold is exceeded (contact detected - this is SUCCESS)
        if self.current_gripper_force > GRIPPER_FORCE_THRESHOLD and not self.force_threshold_reached:
            self.get_logger().info(f"Contact detected: Gripper force {self.current_gripper_force:.2f} exceeded threshold {GRIPPER_FORCE_THRESHOLD}")
            self.force_threshold_reached = True
            self.contact_detected_stop()
    
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
                self.get_logger().info(f"Contact detected: Gripper force {self.current_gripper_force:.2f} exceeded threshold {GRIPPER_FORCE_THRESHOLD}")
                self.force_threshold_reached = True
                self.contact_detected_stop()
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
                self.get_logger().info(
                    f"Contact detected: Z force threshold reached. "
                    f"Z force (after baseline): {f_z:.2f}N (threshold: {Z_FORCE_THRESHOLD}N, "
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
                if magnitude > FORCE_THRESHOLD:
                    self.get_logger().info(
                        f"Contact detected: {axis_name} force/torque threshold reached. "
                        f"{axis_name} (after baseline): {force_value:.2f}N (magnitude: {magnitude:.2f}N, "
                        f"current: {current_value:.2f}N, baseline: {baseline_value:.2f}N, "
                        f"threshold: {FORCE_THRESHOLD}N)"
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
            self.error_message = f"Target Z position ({target_position[2]:.3f}m) is below workspace limit ({MIN_WORKSPACE_Z}m). Cannot proceed."
            self.get_logger().error(self.error_message)
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
                self.error_message = "Current joint angles not available! Cannot compute IK."
                self.get_logger().error(self.error_message)
                rclpy.shutdown()
                return
            
            # Normalize joint angles to [-pi, pi] to avoid issues with joints outside bounds
            q_guess = np.array([np.arctan2(np.sin(a), np.cos(a)) for a in self.current_joint_angles])
            
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

                    # Check for collisions (sim mode only)
                    has_collision = False
                    if self.mode == 'sim':
                        has_table_collision = self.check_collision_with_table(result.x, z_threshold=-0.01)
                        has_self_collision = self.check_self_collision(result.x)
                        has_collision = has_table_collision or has_self_collision

                    # Check if this is a good solution (low cost and no collision)
                    if cost < 0.01 and not has_collision:
                        joint_angles = result.x
                        break

                    # Keep track of best solution (only if no collision)
                    if not has_collision and cost < best_cost:
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

                        # Check for collisions (sim mode only)
                        has_collision = False
                        if self.mode == 'sim':
                            has_table_collision = self.check_collision_with_table(result.x, z_threshold=-0.01)
                            has_self_collision = self.check_self_collision(result.x)
                            has_collision = has_table_collision or has_self_collision

                        # Accept solution if cost is good and no collision
                        if cost < 0.01 and not has_collision:
                            self.get_logger().info(f"IK succeeded at Z={test_z:.3f}m (cost={cost:.6f})")
                            joint_angles = result.x
                            target_position[2] = test_z  # Update target to reachable Z
                            self.get_logger().info(f"Updated target Z to {test_z:.3f}m (lowest reachable height)")
                            break

                if joint_angles is None:
                    if self.mode == 'sim':
                        self.error_message = "IK solver failed: no valid solution to perform collision-free move down"
                    else:
                        self.error_message = "IK solver failed: no valid solution to perform move down"
                    self.get_logger().error(self.error_message)
                    rclpy.shutdown()
                    return
                
            self.get_logger().info(f"Computed joint angles: {joint_angles}")

            # Create joint-space interpolated trajectory (smoother, prevents swinging)
            num_waypoints = 10
            total_duration = MOVEMENT_DURATION

            # Use RAW current joint angles as start (where robot actually is)
            start_joints = self.current_joint_angles.copy()
            # Normalize target joints first
            target_joints = np.array([np.arctan2(np.sin(a), np.cos(a)) for a in joint_angles])

            # Wrap target to be within π of the RAW start (shortest path from actual position)
            for i in range(6):
                # Keep adding/subtracting 2π until target is within π of start
                while target_joints[i] - start_joints[i] > np.pi:
                    target_joints[i] -= 2 * np.pi
                while target_joints[i] - start_joints[i] < -np.pi:
                    target_joints[i] += 2 * np.pi

            self.get_logger().info(f"Creating joint-space trajectory with {num_waypoints} waypoints")

            trajectory_points = []

            # Add starting point at t=0 (zero velocity - starting from rest)
            trajectory_points.append(JointTrajectoryPoint(
                positions=[float(x) for x in start_joints],
                velocities=[0.0] * 6,
                time_from_start=Duration(sec=0, nanosec=0)
            ))

            # Add intermediate waypoints (no velocity specified - controller computes smooth velocities)
            for i in range(1, num_waypoints):
                alpha = i / num_waypoints
                interpolated_joints = start_joints + alpha * (target_joints - start_joints)
                time_from_start = (i / num_waypoints) * total_duration

                trajectory_points.append(JointTrajectoryPoint(
                    positions=[float(x) for x in interpolated_joints],
                    time_from_start=Duration(sec=int(time_from_start),
                                            nanosec=int((time_from_start % 1) * 1e9))
                ))

            # Add final point (zero velocity - stop at end)
            trajectory_points.append(JointTrajectoryPoint(
                positions=[float(x) for x in target_joints],
                velocities=[0.0] * 6,
                time_from_start=Duration(sec=int(total_duration),
                                        nanosec=int((total_duration % 1) * 1e9))
            ))

            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            traj.points = trajectory_points

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj
            goal.goal_time_tolerance = Duration(sec=1)
            
            self.moving = True
            
            # Clear old futures/callbacks before sending new goal
            self._current_goal_handle = None
            self._send_goal_future = None
            self._get_result_future = None
            
            self.get_logger().info(f"Joint-space trajectory with {len(trajectory_points)} waypoints sent")
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
            rclpy.shutdown()
        except RuntimeError:
            pass  # Context already shut down

if __name__ == '__main__':
    main()