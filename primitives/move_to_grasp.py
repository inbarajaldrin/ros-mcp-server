#!/usr/bin/env python3
"""
Direct Object Movement - Native ROS2 Node
Read object poses from TFMessage and perform single direct movement to specific object by name
Includes calibration offset correction for accurate positioning
Supports grasp point selection from /grasp_points_sim (sim mode) or /grasp_points_real (real mode) topics
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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from controller_manager_msgs.srv import ListControllers
import math
import argparse
import numpy as np
import subprocess
import json
from scipy.spatial.transform import Rotation as R

# Import from local action_libraries file
from primitives.utils.action_libraries import hover_over_grasp_quat
from primitives.utils.ik_solver import rpy_to_matrix, ik_objective_quaternion, dh_params, forward_kinematics

# Import quaternion controller for gimbal-lock-free gripper orientation
from primitives.utils.quaternion_orientation_controller import QuaternionOrientationController

# Import path finder for auto-discovering aruco-grasp-annotator data directory
from primitives.utils.data_path_finder import get_symmetry_dir
from primitives.utils.workspace_config import TABLE_HEIGHT, GRIPPER_CENTER_TOOL_OFFSET

# Import grasp points message type (using standard visualization_msgs MarkerArray)
from visualization_msgs.msg import MarkerArray, Marker


def compute_ik_wrist3_extended(position, rpy, current_joints=None, max_tries=2, dx=0.001, prefer_elbow_down=True):
    """IK solver with wrist_3 joint range extended to (-2pi, 2pi).

    Constrains the robot to maintain a consistent "elbow-down" picking configuration:
    - shoulder_lift: negative (arm reaching forward/down)
    - elbow: positive (elbow pointing down, not up)
    - wrist_2: negative (wrist-down, gripper facing consistently)

    Args:
        prefer_elbow_down: If True, constrain joints to the standard picking configuration
            to avoid unusual arm postures regardless of initial robot pose.
    """
    from primitives.utils.unified_ik import IKSolverConfig, IKSolver

    original_position = np.array(position)
    target_rot_matrix = R.from_euler('xyz', rpy, degrees=True).as_matrix()

    # Convert gripper center target to flange target
    flange_position = original_position - target_rot_matrix @ GRIPPER_CENTER_TOOL_OFFSET

    target_pose = np.eye(4)
    target_pose[:3, 3] = flange_position
    target_pose[:3, :3] = target_rot_matrix

    # Fallback seeds (minimal set - most diverse configurations)
    # These seeds have: shoulder_lift negative, elbow positive, wrist_2 = -90°
    seed_configs = [
        np.radians([85, -80, 90, -90, -90, -(np.mod(rpy[2] + 180, 360) - 180)]),
        np.radians([90, -90, 90, -90, -90, rpy[2]]),
        np.radians([0, -90, 90, -90, -90, rpy[2]]),
        np.radians([85, -100, 120, -110, -90, rpy[2]]),
    ]

    # Add current joints as additional seed only if already in valid configuration
    # This helps with smooth motion when already in a good pose
    if current_joints is not None:
        curr = np.array(current_joints)
        # Check if current joints are already in the desired configuration:
        # shoulder_lift < 0, elbow > 0, wrist_2 < 0
        is_valid_config = (curr[1] < 0 and curr[2] > 0 and curr[4] < 0)
        if is_valid_config:
            # Current config is good - use it as primary seed for smooth motion
            seed_configs.insert(0, curr)

    # Constrain joints to the standard picking configuration when prefer_elbow_down is True
    if prefer_elbow_down:
        joint_bounds = [
            (-np.pi, np.pi),     # shoulder_pan: full range
            (-np.pi, 0),         # shoulder_lift: negative only (reaching forward/down)
            (0, np.pi),          # elbow: positive only (elbow down)
            (-np.pi, np.pi),     # wrist_1: full range
            (-np.pi, 0),         # wrist_2: negative only (wrist-down)
            (-2*np.pi, 2*np.pi)  # wrist_3: extended range
        ]
    else:
        joint_bounds = [
            (-np.pi, np.pi),     # shoulder_pan
            (-np.pi, np.pi),     # shoulder_lift
            (-np.pi, np.pi),     # elbow
            (-np.pi, np.pi),     # wrist_1
            (-np.pi, np.pi),     # wrist_2
            (-2*np.pi, 2*np.pi)  # wrist_3
        ]

    solver = IKSolver(IKSolverConfig(joint_bounds=joint_bounds))
    return solver.solve(
        seeds=seed_configs,
        target_pose=target_pose,
        perturbations=max_tries,
        dx=dx,
    )


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


class DirectObjectMove(Node):
    def __init__(self, topic_name=None, object_name="blue_dot_0", height=None, movement_duration=5.0, target_xyz=None, target_xyzw=None, grasp_points_topic="/grasp_points", grasp_id=None, offset=None, mode=None):
        super().__init__('direct_object_move')

        # Mode must be explicitly specified - no default
        if mode is None:
            raise ValueError("Mode must be explicitly specified. Use 'sim' or 'real'.")
        if mode not in ['sim', 'real']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'sim' or 'real'.")

        self.mode = mode  # 'sim' or 'real'

        # Movement parameters (configurable)
        self.hover_height_offset = 0.075  # Hover height above grasp point in step 1 (meters)
        self.table_clearance = 0.01  # Minimum clearance above table for gripper fingers (meters)
        
        # Set default topic based on mode if not provided
        if topic_name is None:
            if self.mode == 'sim':
                self.topic_name = "/objects_poses_sim"
            else:
                self.topic_name = "/objects_poses_real"
        else:
            self.topic_name = topic_name
        
        # Set default grasp points topic based on mode if using default value
        if grasp_points_topic == "/grasp_points":  # Default value - override based on mode
            if self.mode == 'sim':
                self.grasp_points_topic = "/grasp_points_sim"
            else:
                self.grasp_points_topic = "/grasp_points_real"  # Real mode uses /grasp_points_real
        else:
            self.grasp_points_topic = grasp_points_topic  # Use explicitly provided topic
        
        self.object_name = object_name
        self.height = height  # None means use offset, otherwise use exact height
        self.movement_duration = movement_duration  # Duration for IK movement
        self.target_xyz = target_xyz  # Optional target position [x, y, z]
        self.target_xyzw = target_xyzw  # Optional target orientation [x, y, z, w]
        self.grasp_id = grasp_id  # Specific grasp point ID to use
        self.last_target_pose = None
        self.position_threshold = 0.005  # 5mm
        self.angle_threshold = 2.0       # 2 degrees
        
        # State tracking for two-step movement
        self.step1_completed = False  # Track if first step is done
        self.step1_z_position = None  # Store Z position from step 1
        
        # Minimum actual gripper center height above table surface (meters)
        # The gripper has material both above and below the center point.
        # For small objects near the table, the gripper center must be raised
        # to prevent the lower finger material from hitting the table during closure.
        self.min_gripper_center_z = TABLE_HEIGHT + self.table_clearance  # Ensures clearance for gripper fingers below center

        # Vertical offset below the grasp point for the gripper center (meters).
        # IK targets are converted from gripper center to flange using GRIPPER_CENTER_TOOL_OFFSET.
        # Default 0: gripper center placed at the grasp point.
        self.offset = offset if offset is not None else 0
        
        # Initialize Quaternion Orientation Controller for gimbal-lock-free gripper control
        # This ensures stable gripper orientation at pitch=180° (face down) for any yaw angle
        self.quat_controller = QuaternionOrientationController()
        self.get_logger().info("Quaternion orientation controller initialized (gimbal-lock-free mode)")
        
        # Fold symmetry directory for canonical pose matching (auto-discovered)
        self.symmetry_dir = str(get_symmetry_dir())
        
        # Store latest grasp points
        self.latest_grasp_points = None
        self.selected_grasp_point = None
        
        # Track final object pose for logging before exit
        self.final_object_pose = None  # Store final object pose (PoseStamped) before exit
        self.final_position = None  # [x, y, z]
        self.final_orientation_quat = None  # [x, y, z, w]
        self.final_orientation_rpy_deg = None  # [roll, pitch, yaw] in degrees

        # Store current end-effector pose
        self.current_ee_pose = None
        self.ee_pose_received = False

        # Store current joint angles (for IK seeding)
        self.current_joint_angles = None
        self.joint_angles_received = False
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # Subscribe to object poses topic based on mode
        # Both sim and real modes use TFMessage (simulation publishes TFMessage, not ObjectPoseArray)
        if self.mode == 'sim':
            # Sim mode: use TFMessage (for /objects_poses_sim topic which publishes TFMessage)
            self.pose_sub = self.create_subscription(
                TFMessage,
                self.topic_name,
                self.tf_message_callback,
                5  # Lower QoS to reduce update frequency
            )
            self.get_logger().info(f"Using SIM mode: subscribed to {self.topic_name} (TFMessage)")
        else:
            # Real mode: use TFMessage (for /objects_poses_real topic which publishes TFMessage)
            self.pose_sub = self.create_subscription(
                TFMessage,
                self.topic_name,
                self.tf_message_callback,
                5  # Lower QoS to reduce update frequency
            )
            self.get_logger().info(f"Using REAL mode: subscribed to {self.topic_name} (TFMessage)")
        
        # Subscribe to end-effector pose topic
        # Use VOLATILE to match publisher QoS (UR driver uses VOLATILE durability)
        qos_volatile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            qos_volatile
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, qos_volatile
        )

        # Subscribe to grasp points topic if grasp_id is provided
        if self.grasp_id is not None:
            self.grasp_points_sub = self.create_subscription(
                MarkerArray,
                self.grasp_points_topic,
                self.grasp_points_callback,
                5
            )
            self.get_logger().info(f"Grasp point mode: Looking for grasp_id {grasp_id} on topic {self.grasp_points_topic}")
        else:
            self.grasp_points_sub = None
        
        # Add timer to control update frequency (same for both modes)
        # Fast polling until data ready, then execute trajectory
        self.timer_period_step1 = 0.5  # Fast polling for step 1
        self.timer_period_step2 = 0.5  # Fast polling for step 2
        self.update_timer = self.create_timer(self.timer_period_step1, self.timer_callback)
        
        # Track last pose update time to ensure we use fresh data
        self.last_grasp_point_update_time = None
        self.last_pose_update_time = None
        self.latest_pose = None
        self.movement_completed = False  # Flag to track if movement has been completed
        self.should_exit = False  # Flag to control exit
        self.error_message = None  # Specific error message for failure modes
        self.trajectory_in_progress = False  # Flag to track if trajectory is executing
        
        # Visual servoing variables (for real mode continuous tracking)
        self.stable_count = 0  # Count consecutive stable readings
        self.stable_threshold = 3  # Exit after N consecutive stable readings
        self.current_goal_handle = None  # Store current goal handle for potential cancellation
        self.using_stale_data_step2 = False  # Track if we're using stale data in step 2
        self.cancelled_for_fresh_data = False  # Track if we cancelled trajectory to recompute with fresh data
        self.convergence_distance_threshold = 0.02  # 2cm - stop when within this distance of target
        self.convergence_stable_count = 0  # Count stable readings when within convergence distance
        self.convergence_stable_threshold = 2  # Need 2 stable readings within convergence distance
        
        # Tracking loss recovery variables (for real mode)
        self.tracking_lost_count = 0  # Count consecutive frames without detection
        self.max_tracking_lost = 3  # Max consecutive lost detections before recovery
        self.last_known_object_position = None  # Store last known good position
        self.last_known_object_quat = None  # Store last known good orientation (quaternion, no RPY)
        self.recovery_mode = False  # Flag to indicate we're in recovery mode
        self.recovery_backoff_distance = 0.05  # Move back 5cm when tracking lost
        self.recovery_slowdown_factor = 2.0  # Slow down movement by this factor during recovery
        self.waiting_at_last_known = False  # Flag to indicate we've moved to last known location and are waiting
        self.last_known_target_sent = False  # Flag to track if we've sent trajectory to last known location
        
        # Z position smoothing after recovery (to prevent height jumps)
        self.smoothed_object_z = None  # Smoothed Z position to prevent jumps after recovery
        self.z_smoothing_alpha = 0.3  # Smoothing factor (0.0 = no change, 1.0 = immediate update)
        self.recovery_z_update_count = 0  # Count updates after recovery
        self.recovery_z_smoothing_steps = 5  # Number of steps to smooth Z after recovery
        
        # Wait after tracking recovery (for step 2 downward movement)
        self.waiting_after_recovery = False  # Flag to indicate we're waiting after tracking recovery
        self.recovery_wait_start_time = None  # Timestamp when recovery wait started
        self.recovery_wait_duration = 2.0  # Wait duration in seconds after recovery
        
        # Grasp topic wait retry settings
        self.grasp_wait_attempts = 0
        self.max_grasp_wait_attempts = 3

        # Canonical pose threshold settings (used in _try_canonical_match_with_threshold)
        self.canonical_threshold_initial = 0.45  # Initial threshold (~90°)
        self.canonical_threshold_max = 0.9  # Maximum threshold to try (~100°)
        self.canonical_threshold_increment = 0.1  # Increment threshold by this amount each retry
        self.current_canonical_threshold = self.canonical_threshold_initial  # Current threshold being used
        self.best_canonical_match = None  # Store best match found so far
        self.best_canonical_distance = float('inf')  # Distance of best match
        self.fold_symmetry_error_logged = False  # Flag to log fold symmetry error only once

        # Action client for trajectory execution
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        mode_str = self.mode.upper()
        if self.grasp_id is not None:
            self.get_logger().info(f"Using {mode_str} mode: Moving to object '{object_name}' (grasp_id: {grasp_id})")
        else:
            self.get_logger().info(f"Using {mode_str} mode: Moving to object '{object_name}'")
        self.get_logger().info(f"Movement duration: {movement_duration}s")
        
    def quaternion_to_rpy(self, x, y, z, w):
        """Convert quaternion to roll, pitch, yaw in degrees"""
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
    
    def poses_are_similar(self, position, quaternion):
        """Check if pose is similar to last target (QUATERNION-BASED, no RPY)"""
        if self.last_target_pose is None:
            return False
            
        last_pos, last_quat = self.last_target_pose
        
        # Check position difference (only x, y)
        pos_diff = math.sqrt(
            (position[0] - last_pos[0])**2 +
            (position[1] - last_pos[1])**2
        )
        
        if pos_diff > self.position_threshold:
            return False
            
        # Check quaternion difference using dot product (quaternion similarity)
        # For unit quaternions, dot product gives cos(angle/2), so abs(dot) gives angle similarity
        # If dot product is close to 1 or -1, quaternions represent similar orientations
        quat_array = np.array(quaternion)
        last_quat_array = np.array(last_quat)
        
        # Normalize quaternions (should already be normalized, but ensure it)
        quat_array = quat_array / np.linalg.norm(quat_array)
        last_quat_array = last_quat_array / np.linalg.norm(last_quat_array)
        
        # Dot product to measure angular distance
        dot_product = abs(np.dot(quat_array, last_quat_array))
        
        # Convert to angle: angle = 2 * acos(dot_product)
        # For small angles: angle ≈ 2 * (1 - dot_product)
        angle_diff_radians = 2 * math.acos(np.clip(dot_product, -1.0, 1.0))
        angle_diff_degrees = math.degrees(angle_diff_radians)
        
        return angle_diff_degrees <= self.angle_threshold
    
    def tf_message_callback(self, msg):
        """Handle TFMessage and find target object by child_frame_id"""
        # Find the transform with matching child_frame_id (object name)
        target_transform = None
        for transform in msg.transforms:
            if transform.child_frame_id == self.object_name:
                target_transform = transform
                break
        
        if target_transform is not None:
            # Check if we were using stale data in step 2 and fresh data just arrived
            if self.step1_completed and self.using_stale_data_step2 and self.trajectory_in_progress:
                self.get_logger().debug("Fresh object pose data received during step 2! Cancelling current trajectory and recomputing...")
                # Cancel current trajectory
                if self.current_goal_handle is not None:
                    try:
                        cancel_future = self.current_goal_handle.cancel_goal_async()
                        self.get_logger().debug("Cancelling current trajectory...")
                    except Exception as e:
                        self.get_logger().debug(f"Could not cancel trajectory: {e}")
                # Reset trajectory flag so timer callback can recompute step 2
                self.trajectory_in_progress = False
                self.using_stale_data_step2 = False
            
            # Convert TransformStamped to PoseStamped
            pose_stamped = PoseStamped()
            pose_stamped.header = target_transform.header
            pose_stamped.pose.position.x = target_transform.transform.translation.x
            pose_stamped.pose.position.y = target_transform.transform.translation.y
            pose_stamped.pose.position.z = target_transform.transform.translation.z
            pose_stamped.pose.orientation.x = target_transform.transform.rotation.x
            pose_stamped.pose.orientation.y = target_transform.transform.rotation.y
            pose_stamped.pose.orientation.z = target_transform.transform.rotation.z
            pose_stamped.pose.orientation.w = target_transform.transform.rotation.w
            self.latest_pose = pose_stamped
            # Track final object pose for logging
            self.final_object_pose = pose_stamped
            self._update_final_pose_data(pose_stamped)
        else:
            # Object not found in this message
            self.latest_pose = None

    def pose_callback(self, msg):
        """Store latest pose message (fallback for PoseStamped)"""
        self.latest_pose = msg
        # Track final object pose for logging
        self.final_object_pose = msg
        self._update_final_pose_data(msg)
    
    def grasp_points_callback(self, msg):
        """Handle MarkerArray message and find target grasp point"""
        # Store all grasp points (markers)
        self.latest_grasp_points = msg
        
        # Find the marker with the specified ID and object name (ns)
        target_marker = None
        for marker in msg.markers:
            if (marker.id == self.grasp_id and 
                marker.ns == self.object_name):
                target_marker = marker
                break
        
        if target_marker is not None:
            # Check if we were using stale data in step 2 and fresh data just arrived
            if self.step1_completed and self.using_stale_data_step2 and self.trajectory_in_progress:
                self.get_logger().debug("Fresh grasp point data received during step 2! Cancelling current trajectory and recomputing...")
                # Cancel current trajectory
                if self.current_goal_handle is not None:
                    try:
                        # Set flag BEFORE cancelling to ensure it's set when goal_result is called
                        self.cancelled_for_fresh_data = True
                        cancel_future = self.current_goal_handle.cancel_goal_async()
                        self.get_logger().debug("Cancelling current trajectory...")
                    except Exception as e:
                        self.get_logger().debug(f"Could not cancel trajectory: {e}")
                        self.cancelled_for_fresh_data = False
                else:
                    self.get_logger().debug("No goal handle available to cancel")
                # Reset trajectory flag so timer callback can recompute step 2
                self.trajectory_in_progress = False
                self.using_stale_data_step2 = False
            
            # Update grasp point in real-time (like object poses)
            self.selected_grasp_point = target_marker
            # Track update time to ensure we use fresh data
            self.last_grasp_point_update_time = self.get_clock().now()
            # Don't unsubscribe - keep receiving updates in real-time
        else:
            # Grasp point not found in this message - keep previous one if available
            if self.selected_grasp_point is None:
                self.get_logger().debug(f"Grasp point {self.grasp_id} for object '{self.object_name}' not found in current message")
    
    def ee_pose_callback(self, msg: PoseStamped):
        """Callback for end-effector pose data"""
        self.current_ee_pose = msg
        self.ee_pose_received = True

    def joint_state_callback(self, msg):
        """Store current joint angles in standard order for IK seeding"""
        if len(msg.name) == 6 and len(msg.position) == 6:
            joint_dict = dict(zip(msg.name, msg.position))
            positions = [joint_dict.get(name, 0) for name in self.joint_names]
            if len(positions) == 6:
                self.current_joint_angles = np.array(positions)
                self.joint_angles_received = True

    def timer_callback(self):
        """Process pose and perform movement to object"""
        if self.movement_completed or self.should_exit:
            return
        
        # Handle canonical pose retry mode - adjust threshold instead of moving robot
        # Threshold adjustment happens in canonical match checking, so we just continue to normal processing
        
        # Wait for trajectory to complete before sending new one (same for both modes)
        if self.trajectory_in_progress:
            self.get_logger().debug("Trajectory already in progress, skipping...")
            return
        
        # Check if we're waiting after tracking recovery (step 2 only)
        if self.waiting_after_recovery and self.recovery_wait_start_time is not None:
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.recovery_wait_start_time).nanoseconds / 1e9  # Convert to seconds
            if elapsed_time < self.recovery_wait_duration:
                self.get_logger().debug(f"Waiting after tracking recovery: {elapsed_time:.2f}/{self.recovery_wait_duration:.2f} seconds...")
                return
            else:
                # Wait period completed, clear flag and continue
                self.waiting_after_recovery = False
                self.recovery_wait_start_time = None
                self.get_logger().info(f"Wait period completed. Continuing movement after tracking recovery.")
        
        # Wait for end-effector pose if not received yet
        if not self.ee_pose_received or self.current_ee_pose is None:
            self.get_logger().debug("Waiting for end-effector pose...")
            return
        
        # Get current end-effector position
        current_ee_position = np.array([
            self.current_ee_pose.pose.position.x,
            self.current_ee_pose.pose.position.y,
            self.current_ee_pose.pose.position.z
        ])
        
        # If current EE position is below 0.25, skip straight to step 2 (both sim and real modes)
        if not self.step1_completed:
            if current_ee_position[2] < 0.25:
                self.get_logger().info(f"Current EE Z position ({current_ee_position[2]:.3f}m) is below 0.25m. Skipping step 1 and going straight to step 2.")
                self.step1_completed = True
                # Switch to faster timer period for step 2
                self.update_timer.cancel()
                self.update_timer = self.create_timer(self.timer_period_step2, self.timer_callback)
        
        # Verify that at least one explicit mode is specified
        # Object detection mode is valid if object_name is provided (even if latest_pose is None - we'll wait for it)
        has_explicit_mode = (
            (self.target_xyz is not None and self.target_xyzw is not None) or
            (self.grasp_id is not None) or
            (self.object_name is not None and self.object_name != "")  # Object detection mode when object_name is provided
        )
        
        if not has_explicit_mode:
            self.error_message = "No explicit mode specified. Must provide one of: target_xyz/xyzw, grasp_id, or object detection."
            self.get_logger().error(self.error_message + " Exiting.")
            self.should_exit = True
            return
        
        # If in object detection mode but no pose received yet, wait
        if (self.object_name is not None and self.object_name != "" and 
            self.target_xyz is None and self.grasp_id is None and 
            self.latest_pose is None and 
            (self.mode != 'real' or self.last_known_object_position is None)):
            return
        
        # Check if we have optional target position/orientation
        if self.target_xyz is not None and self.target_xyzw is not None:
            # Use provided target position and orientation
            object_position = np.array(self.target_xyz[:3])  # Take first 3 elements
            
            
            # Use provided target quaternion and apply fold symmetry matching
            provided_quat = np.array(self.target_xyzw)
            
            # Try to find canonical match with increasing thresholds
            canonical_quat, canonical_match, match_distance = self._try_canonical_match_with_threshold(
                provided_quat, self.object_name
            )
            
            # Extract yaw from returned quaternion (canonical match or fallback)
            object_yaw = self.quat_controller.extract_yaw_from_quaternion(canonical_quat)
            
            target_quaternion = self.quat_controller.face_down_quaternion(object_yaw)
            
            step_msg = "Step 2: Fine positioning" if self.step1_completed else "Step 1: Moving to hover position"
            self.get_logger().info(step_msg)
            self.get_logger().info(f"Object at: ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
        elif self.grasp_id is not None:
            # Grasp point mode: must have selected_grasp_point, retry if not available yet
            if self.selected_grasp_point is None:
                self.grasp_wait_attempts += 1
                # Differentiate between: topic not publishing, object not found, or grasp point not found
                if self.latest_grasp_points is None:
                    msg = f"No data from grasp topic {self.grasp_points_topic}"
                else:
                    # Check if object exists at all in the grasp points data
                    object_exists = any(
                        marker.ns == self.object_name
                        for marker in self.latest_grasp_points.markers
                    )
                    if not object_exists:
                        if self.latest_pose is None:
                            msg = f"Object '{self.object_name}' not detected on pose topic {self.topic_name} (object may be out of view)"
                        else:
                            msg = f"Object '{self.object_name}' detected on pose topic but not in grasp topic. Make sure grasp publisher is running"
                    else:
                        # Object exists but grasp_id not found - list available IDs
                        available_ids = [
                            marker.id for marker in self.latest_grasp_points.markers
                            if marker.ns == self.object_name
                        ]
                        msg = f"Grasp point {self.grasp_id} not found for object '{self.object_name}'. Available grasp IDs: {sorted(available_ids)}"
                if self.grasp_wait_attempts < self.max_grasp_wait_attempts:
                    self.get_logger().warn(f"{msg} (attempt {self.grasp_wait_attempts}/{self.max_grasp_wait_attempts}), retrying...")
                    return
                self.error_message = msg
                self.get_logger().error(self.error_message + " Cannot proceed in grasp point mode. Exiting.")
                self.should_exit = True
                return
            
            # Check if we're using a stale grasp point (topic is empty or doesn't contain our grasp point)
            # Stale means: we're in step 2, have a selected_grasp_point, but current message is empty or doesn't have it
            using_stale_grasp_point = False
            if self.step1_completed:
                if self.latest_grasp_points is None:
                    # No message received yet - using stale data from step 1
                    using_stale_grasp_point = True
                elif len(self.latest_grasp_points.markers) == 0:
                    # Message is empty - using stale data
                    using_stale_grasp_point = True
                else:
                    # Check if our grasp point is in the current message
                    grasp_point_found = False
                    for marker in self.latest_grasp_points.markers:
                        if (marker.id == self.grasp_id and 
                            marker.ns == self.object_name):
                            grasp_point_found = True
                            break
                    # If not found in current message, we're using stale data
                    using_stale_grasp_point = not grasp_point_found
            
            # Use only grasp point position (ignore orientation)
            grasp_point_position = np.array([
                self.selected_grasp_point.pose.position.x,
                self.selected_grasp_point.pose.position.y,
                self.selected_grasp_point.pose.position.z
            ])
            
            # Track if we're using stale data in step 2
            self.using_stale_data_step2 = using_stale_grasp_point and self.step1_completed
            
            # Log the exact grasp point being used
            if using_stale_grasp_point:
                self.get_logger().debug(f"Using stale grasp point {self.grasp_id} (topic is empty)")
            
            
            # Set object position to grasp point position for distance calculation
            object_position = grasp_point_position
            
            # Store last known good position for recovery (real mode)
            if self.mode == 'real':
                self.last_known_object_position = object_position.copy()
                # Reset tracking lost count since we have a detection
                if self.tracking_lost_count > 0:
                    self.get_logger().info(f"Tracking recovered after {self.tracking_lost_count} lost frames")
                self.tracking_lost_count = 0
                self.recovery_mode = False
            
            # Use grasp point orientation if available, otherwise exit (grasp point must have orientation)
            # Check if grasp point has valid orientation (non-zero quaternion)
            grasp_point_has_orientation = (
                hasattr(self.selected_grasp_point, 'pose') and
                hasattr(self.selected_grasp_point.pose, 'orientation') and
                (abs(self.selected_grasp_point.pose.orientation.w) > 1e-6 or
                 abs(self.selected_grasp_point.pose.orientation.x) > 1e-6 or
                 abs(self.selected_grasp_point.pose.orientation.y) > 1e-6 or
                 abs(self.selected_grasp_point.pose.orientation.z) > 1e-6)
            )
            
            if not grasp_point_has_orientation:
                # Grasp point orientation is required - exit if not available
                self.error_message = f"Grasp point {self.grasp_id} does not have valid orientation."
                self.get_logger().error(self.error_message + " Cannot proceed. Exiting.")
                self.should_exit = True
                return
            
            if grasp_point_has_orientation:
                # Extract grasp point orientation and apply fold symmetry matching
                grasp_point_quat = np.array([
                    self.selected_grasp_point.pose.orientation.x,
                    self.selected_grasp_point.pose.orientation.y,
                    self.selected_grasp_point.pose.orientation.z,
                    self.selected_grasp_point.pose.orientation.w
                ])
                
                # Try to find canonical match with increasing thresholds
                canonical_quat, canonical_match, match_distance = self._try_canonical_match_with_threshold(
                    grasp_point_quat, self.object_name
                )
                
                # Extract yaw from returned quaternion (canonical match or fallback)
                grasp_point_yaw = self.quat_controller.extract_yaw_from_quaternion(canonical_quat)
                
                # Create face-down quaternion with grasp point yaw (QUATERNION-BASED, no gimbal lock)
                target_quaternion = self.quat_controller.face_down_quaternion(grasp_point_yaw)
                
                step_msg = "Step 2: Fine positioning" if self.step1_completed else "Step 1: Moving to hover position"
                self.get_logger().info(step_msg)
                self.get_logger().info(f"Object at: ({grasp_point_position[0]:.3f}, {grasp_point_position[1]:.3f}, {grasp_point_position[2]:.3f})")
        elif self.latest_pose is not None:
            # Use detected object pose
            # Check if we were using stale data in step 2 (using last_known_object_position)
            was_using_stale_data = False
            if self.step1_completed and self.using_stale_data_step2 and self.trajectory_in_progress:
                was_using_stale_data = True
                self.get_logger().debug("Fresh object pose data received during step 2! Cancelling current trajectory and recomputing...")
                # Cancel current trajectory
                if self.current_goal_handle is not None:
                    try:
                        self.cancelled_for_fresh_data = True  # Mark that we're cancelling for fresh data
                        cancel_future = self.current_goal_handle.cancel_goal_async()
                        self.get_logger().debug("Cancelling current trajectory...")
                    except Exception as e:
                        self.get_logger().debug(f"Could not cancel trajectory: {e}")
                        self.cancelled_for_fresh_data = False
                # Reset trajectory flag so timer callback can recompute step 2
                self.trajectory_in_progress = False
                self.using_stale_data_step2 = False
            
            # Reset tracking lost count since we have a detection
            was_tracking_lost = False
            if self.mode == 'real':
                was_tracking_lost = self.tracking_lost_count > 0 or self.waiting_at_last_known
                if was_tracking_lost:
                    self.get_logger().info(f"Tracking recovered after {self.tracking_lost_count} lost frames")
                    # Reset recovery flags
                    self.tracking_lost_count = 0
                    self.recovery_mode = False
                    self.waiting_at_last_known = False
                    self.last_known_target_sent = False
                    # Reset smoothing when tracking is recovered
                    self.recovery_z_update_count = 0
                    if self.smoothed_object_z is None and self.last_known_object_position is not None:
                        # Initialize smoothed Z with last known Z to prevent jump
                        self.smoothed_object_z = self.last_known_object_position[2]
                        self.get_logger().info(f"Initializing Z smoothing with last known Z: {self.smoothed_object_z:.3f}m")
                else:
                    self.tracking_lost_count = 0
            
            # If we're in step 2 (moving down) and pose wasn't available first (tracking lost or stale data), wait 2 seconds after recovery
            if self.step1_completed and (was_tracking_lost or was_using_stale_data):
                self.waiting_after_recovery = True
                self.recovery_wait_start_time = self.get_clock().now()
                self.get_logger().info(f"Object pose became available again after being unavailable. Waiting {self.recovery_wait_duration} seconds before continuing movement...")
            
            # Extract position directly from latest_pose (no filtering)
            object_position = np.array([
                self.latest_pose.pose.position.x,
                self.latest_pose.pose.position.y,
                self.latest_pose.pose.position.z
            ])
            
            # Extract quaternion directly from latest_pose
            # This ensures we work with pure quaternions, avoiding gimbal lock
            detected_object_quat = np.array([
                self.latest_pose.pose.orientation.x,
                self.latest_pose.pose.orientation.y,
                self.latest_pose.pose.orientation.z,
                self.latest_pose.pose.orientation.w
            ])
            
            # Try to find canonical match with increasing thresholds
            canonical_quat, canonical_match, match_distance = self._try_canonical_match_with_threshold(
                detected_object_quat, self.object_name
            )
            
            # Extract yaw from returned quaternion (canonical match or fallback)
            object_yaw = self.quat_controller.extract_yaw_from_quaternion(canonical_quat)
            
            
            # Smooth Z position after recovery to prevent height jumps
            if self.mode == 'real' and was_tracking_lost:
                if self.smoothed_object_z is not None:
                    # Gradually update Z position after recovery
                    detected_z = object_position[2]
                    # Use exponential smoothing to gradually transition to new Z
                    self.smoothed_object_z = (self.z_smoothing_alpha * detected_z + 
                                              (1.0 - self.z_smoothing_alpha) * self.smoothed_object_z)
                    object_position[2] = self.smoothed_object_z
                    self.recovery_z_update_count += 1
                    
                    if self.recovery_z_update_count < self.recovery_z_smoothing_steps:
                        self.get_logger().info(f"Smoothing Z after recovery: detected={detected_z:.3f}m, smoothed={self.smoothed_object_z:.3f}m "
                                              f"({self.recovery_z_update_count}/{self.recovery_z_smoothing_steps})")
                    else:
                        # Done smoothing, use detected Z directly
                        self.smoothed_object_z = None
                        self.get_logger().info(f"Z smoothing complete, using detected Z: {detected_z:.3f}m")
                else:
                    # First detection after recovery, initialize smoothed Z
                    self.smoothed_object_z = object_position[2]
            
            # Store last known good position and quaternion for recovery
            if self.mode == 'real':
                self.last_known_object_position = object_position.copy()
                self.last_known_object_quat = detected_object_quat.copy()
            
            # Align end-effector with object orientation using QUATERNION (no gimbal lock)
            # For top-down approach: use object's yaw to align, pitch=180 (face down), roll=0
            # This ensures the gripper aligns with the object's orientation while approaching from above
            target_quaternion = self.quat_controller.face_down_quaternion(object_yaw)
            
            # Log fold symmetry matching result
            match_status = "Canonical match" if canonical_match else "No canonical match (using detected)"
            
            self.get_logger().info(f"Detected object at ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
            self.get_logger().info(f"Object quaternion (detected): q=[{detected_object_quat[0]:.6f}, {detected_object_quat[1]:.6f}, "
                                 f"{detected_object_quat[2]:.6f}, {detected_object_quat[3]:.6f}]")
            if canonical_match:
                self.get_logger().info(f"Object quaternion (canonical match): q=[{canonical_quat[0]:.6f}, {canonical_quat[1]:.6f}, "
                                     f"{canonical_quat[2]:.6f}, {canonical_quat[3]:.6f}] - {match_status}")
            else:
                self.get_logger().info(f"{match_status}")
            self.get_logger().info(f"EE orientation (quaternion-based, no gimbal lock):\n"
                                 f"   q=[{target_quaternion[0]:.6f}, {target_quaternion[1]:.6f}, "
                                 f"{target_quaternion[2]:.6f}, {target_quaternion[3]:.6f}]\n"
                                 f"   Aligned with object yaw: {object_yaw:.1f}°")
        else:
            # No target provided and no object detected
            if self.mode == 'real' and self.last_known_object_position is not None:
                # Real mode: handle tracking loss
                self.tracking_lost_count += 1
                self.get_logger().warn(f"Tracking lost! (consecutive misses: {self.tracking_lost_count}/{self.max_tracking_lost})")
                
                if self.tracking_lost_count >= self.max_tracking_lost:
                    # Move to last known location and wait
                    if not self.waiting_at_last_known:
                        self.waiting_at_last_known = True
                        self.recovery_mode = True
                        self.get_logger().warn(f"Moving to last known location and waiting for tracking recovery...")
                    
                    # Use last known position (QUATERNION-BASED, no gimbal lock)
                    object_position = self.last_known_object_position.copy()
                    working_object_quat = self.last_known_object_quat.copy()
                    object_yaw = self.quat_controller.extract_yaw_from_quaternion(working_object_quat)
                    target_quaternion = self.quat_controller.face_down_quaternion(object_yaw)
                    
                    # Track if we're using stale data in step 2
                    if self.step1_completed:
                        self.using_stale_data_step2 = True
                    
                    # Only send trajectory once to last known location
                    if not self.last_known_target_sent:
                        self.get_logger().info(f"Moving to last known position: ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
                        # Continue to calculate and send trajectory (will be sent once)
                    else:
                        # Already sent trajectory to last known location, just wait
                        self.get_logger().info("Waiting at last known location for tracking recovery...")
                        return  # Don't send new trajectories, just wait
                else:
                    # Not enough consecutive misses yet, use last known position (QUATERNION-BASED)
                    if self.last_known_object_position is not None:
                        object_position = self.last_known_object_position.copy()
                        working_object_quat = self.last_known_object_quat.copy()
                        object_yaw = self.quat_controller.extract_yaw_from_quaternion(working_object_quat)
                        target_quaternion = self.quat_controller.face_down_quaternion(object_yaw)
                        
                        # Track if we're using stale data in step 2
                        if self.step1_completed:
                            self.using_stale_data_step2 = True
                        
                        self.get_logger().warn(f"Using last known position (miss {self.tracking_lost_count}/{self.max_tracking_lost})")
                    else:
                        self.error_message = "No target position provided, no object detected, and no last known position."
                        self.get_logger().error(self.error_message + " Cannot proceed. Exiting.")
                        self.should_exit = True
                        return
            else:
                # No target provided and no object detected
                # If in object detection mode (object_name provided), wait for pose
                if self.object_name is not None and self.object_name != "":
                    self.get_logger().debug("Waiting for object pose to be received...")
                    return
                # Otherwise, no explicit mode specified: exit
                self.get_logger().error("No explicit mode specified (no target_xyz/xyzw, no grasp_id) and no object detected. Cannot proceed. Exiting.")
                self.should_exit = True
                return
        
        # Verify that we have a valid target position and orientation (safety check)
        if 'object_position' not in locals() or 'target_quaternion' not in locals():
            self.error_message = "Failed to determine target position or orientation. Cannot proceed."
            self.get_logger().error(self.error_message + " Exiting.")
            self.should_exit = True
            return
        
        
        # Calculate target gripper center position.
        # IK converts this to flange position using GRIPPER_CENTER_TOOL_OFFSET.

        if self.height is not None:
            target_ee_position = np.array([object_position[0], object_position[1], self.height])
        else:
            target_ee_position = object_position.copy()
            if self.offset is not None and self.offset != 0:
                target_ee_position[2] -= self.offset

            # Enforce minimum gripper center height to prevent table collision
            if target_ee_position[2] < self.min_gripper_center_z:
                self.get_logger().info(
                    f"Raising gripper center from {target_ee_position[2]*1000:.1f}mm to "
                    f"{self.min_gripper_center_z*1000:.1f}mm (min clearance for table)")
                target_ee_position[2] = self.min_gripper_center_z

        # Step 1: add hover height offset (both sim and real modes)
        if not self.step1_completed:
            target_ee_position[2] += self.hover_height_offset  # Add hover offset for step 1

        self.get_logger().info(f"Target gripper center: ({target_ee_position[0]:.3f}, {target_ee_position[1]:.3f}, {target_ee_position[2]:.3f})")
        
        # Check convergence in step 2: if robot is close enough to target, count stable readings and exit
        # current_ee_position is the flange from the broadcaster; convert to gripper center
        if self.step1_completed:
            current_ee_quat = [
                self.current_ee_pose.pose.orientation.x,
                self.current_ee_pose.pose.orientation.y,
                self.current_ee_pose.pose.orientation.z,
                self.current_ee_pose.pose.orientation.w,
            ]
            R_ee = R.from_quat(current_ee_quat).as_matrix()
            current_gc_position = current_ee_position + R_ee @ GRIPPER_CENTER_TOOL_OFFSET
            distance_to_target = np.linalg.norm(current_gc_position - target_ee_position)
            
            if distance_to_target <= self.convergence_distance_threshold:
                self.convergence_stable_count += 1
                
                if self.convergence_stable_count >= self.convergence_stable_threshold:
                    self.get_logger().info(f"Converged! Within {self.convergence_distance_threshold*100:.2f}cm of target for {self.convergence_stable_threshold} consecutive readings.")
                    self.movement_completed = True
                    self.should_exit = True
                    self._print_final_object_pose()
                    return  # Exit early, don't send trajectory
            else:
                # Reset stable count if we're not within threshold
                self.convergence_stable_count = 0
        
        # If waiting at last known location, mark that we've sent the trajectory
        if self.waiting_at_last_known and not self.last_known_target_sent:
            self.last_known_target_sent = True
        
        # Create target pose with calculated position (PURE QUATERNION, no RPY conversion)
        target_position = target_ee_position.tolist()

        # Extract yaw from quaternion for IK (face-down orientation)
        q_x, q_y, q_z, q_w = target_quaternion
        yaw_radians = 2.0 * math.atan2(q_z, q_w)
        yaw_degrees = math.degrees(yaw_radians)
        yaw_degrees = ((yaw_degrees + 180) % 360) - 180
        target_rot = [0, 180, yaw_degrees]

        # Use specified movement duration (same for both modes)
        movement_duration = self.movement_duration

        # Solve IK for both yaw and yaw+180 (gripper is symmetric),
        # then pick the solution requiring least joint movement.
        from primitives.utils.action_libraries import make_point
        ik_position = target_position.copy()
        ik_position[2] = target_ee_position[2]

        yaw_alt = yaw_degrees + 180.0
        yaw_alt = ((yaw_alt + 180) % 360) - 180
        target_rot_alt = [0, 180, yaw_alt]

        joint_angles = None
        chosen_yaw = None

        if self.step1_completed:
            # Step 2: solve with current_joints + joint perturbations (no fallback seeds).
            # The arm only descends ~5cm from hover, so current_joints neighborhood always
            # contains a valid solution. Fallback seeds risk early-terminating with a
            # different wrist_3 branch, causing a 180° flip. Joint perturbations escape
            # local minima that cause ~2-3° orientation error from a single-seed solve.
            from primitives.utils.unified_ik import IKSolverConfig, IKSolver
            curr = np.array(self.current_joint_angles)
            joint_bounds = [
                (-np.pi, np.pi), (-np.pi, 0), (0, np.pi),
                (-np.pi, np.pi), (-np.pi, 0), (-2*np.pi, 2*np.pi),
            ]

            # Generate perturbed seeds around current joints
            n_perturbations = 30
            perturbation_sigma = 0.1  # radians (~5.7°)
            rng = np.random.default_rng()
            seeds = [curr]
            for _ in range(n_perturbations):
                perturbed = curr + rng.normal(0, perturbation_sigma, 6)
                perturbed = np.clip(perturbed, [b[0] for b in joint_bounds], [b[1] for b in joint_bounds])
                seeds.append(perturbed)

            solver = IKSolver(IKSolverConfig(
                cost_threshold=0.01, joint_bounds=joint_bounds,
            ))
            # Try both yaw variants with perturbed seeds
            # Convert gripper center targets to flange targets
            gc_pos = np.array(ik_position)
            target_rot_matrix_a = R.from_euler('xyz', target_rot, degrees=True).as_matrix()
            pose_a = np.eye(4)
            pose_a[:3, 3] = gc_pos - target_rot_matrix_a @ GRIPPER_CENTER_TOOL_OFFSET
            pose_a[:3, :3] = target_rot_matrix_a

            target_rot_matrix_b = R.from_euler('xyz', target_rot_alt, degrees=True).as_matrix()
            pose_b = np.eye(4)
            pose_b[:3, 3] = gc_pos - target_rot_matrix_b @ GRIPPER_CENTER_TOOL_OFFSET
            pose_b[:3, :3] = target_rot_matrix_b

            cj_sol_a = solver.solve(seeds=seeds, target_pose=pose_a, perturbations=1)
            solver._best_result = None  # Reset for second solve
            cj_sol_b = solver.solve(seeds=seeds, target_pose=pose_b, perturbations=1)

            candidates = []
            if cj_sol_a is not None:
                candidates.append((cj_sol_a, yaw_degrees))
            if cj_sol_b is not None:
                candidates.append((cj_sol_b, yaw_alt))
            if candidates:
                best = min(candidates, key=lambda s: np.linalg.norm(np.array(s[0]) - curr))
                joint_angles, chosen_yaw = best
                self.get_logger().info(
                    f"Step 2: current_joints solved (yaw={chosen_yaw:.1f}°, "
                    f"jdist={np.linalg.norm(np.array(joint_angles) - curr):.3f} rad)")
            else:
                self.get_logger().warn("Step 2: current_joints failed both yaw variants — IK failed")
        else:
            # Step 1: full multi-seed search
            sol_a = compute_ik_wrist3_extended(ik_position, target_rot, current_joints=self.current_joint_angles, max_tries=1)

            # Skip sol_b if sol_a is good enough (close to current joints)
            sol_b = None
            if self.current_joint_angles is not None and sol_a is not None:
                ref = np.array(self.current_joint_angles)
                joint_dist_a = np.linalg.norm(np.array(sol_a) - ref)
                if joint_dist_a > 1.5:
                    sol_b = compute_ik_wrist3_extended(ik_position, target_rot_alt, current_joints=self.current_joint_angles, max_tries=1)
            elif sol_a is None:
                sol_b = compute_ik_wrist3_extended(ik_position, target_rot_alt, current_joints=self.current_joint_angles, max_tries=1)

            # Pick solution closest to current joints
            if self.current_joint_angles is not None:
                ref = np.array(self.current_joint_angles)
                candidates = []
                if sol_a is not None:
                    candidates.append(sol_a)
                if sol_b is not None:
                    candidates.append(sol_b)
                if candidates:
                    joint_angles = min(candidates, key=lambda s: np.linalg.norm(np.array(s) - ref))
                    chosen_yaw = yaw_degrees if (sol_a is not None and np.array_equal(joint_angles, sol_a)) else yaw_alt
                    self.get_logger().info(f"Chose yaw={chosen_yaw:.1f}° (joint distance: {np.linalg.norm(np.array(joint_angles) - ref):.3f} rad)")
                else:
                    joint_angles = None
            else:
                joint_angles = sol_a if sol_a is not None else sol_b

        if joint_angles is not None:
            traj_point = make_point(joint_angles, movement_duration)
            trajectory = {"traj1": [traj_point]}
        else:
            trajectory = {"traj1": []}
        
        # For step 1: store Z position for step 2 (both sim and real modes)
        if not self.step1_completed:
            self.step1_z_position = target_ee_position[2]
        
        # Execute trajectory (same for both modes: mark as in progress and wait for completion)
        self.trajectory_in_progress = True
        self.execute_trajectory(trajectory)
        
        # Update last target pose for similarity checking (PURE QUATERNION, no RPY)
        self.last_target_pose = (target_ee_position.tolist(), target_quaternion)
        
        # Don't set movement_completed here - wait for trajectory completion callback
    
    def execute_trajectory(self, trajectory):
        """Execute trajectory using ROS2 action"""
        try:
            if 'traj1' not in trajectory or not trajectory['traj1']:
                self.get_logger().error("No trajectory found (IK solver failed to find a solution)")
                self.trajectory_in_progress = False
                self.error_message = "IK solver failed to find a solution for the target pose"
                self.movement_completed = True
                self.should_exit = True
                return
            
            point = trajectory['traj1'][0]
            positions = point['positions']
            duration = point['time_from_start'].sec
            
            # Create trajectory message
            traj_msg = JointTrajectory()
            traj_msg.joint_names = [
                'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
            ]
            
            traj_point = JointTrajectoryPoint()
            traj_point.positions = positions
            traj_point.velocities = [0.0] * 6
            traj_point.time_from_start = Duration(sec=duration)
            traj_msg.points.append(traj_point)
            
            # Create and send goal
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)
            
            # Send trajectory using callbacks to track completion (same for both modes)
            self.get_logger().info("Trajectory sent and accepted")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)
            
        except Exception as e:
            self.error_message = f"Trajectory execution error: {e}"
            self.get_logger().error(self.error_message)
            self.trajectory_in_progress = False  # Clear flag on error
            self.movement_completed = True
            self.should_exit = True

    def diagnose_rejection(self):
        """Query controller_manager to determine why the goal was rejected."""
        cli = self.create_client(ListControllers, '/controller_manager/list_controllers')
        if not cli.wait_for_service(timeout_sec=2.0):
            return "Trajectory goal rejected (controller_manager unavailable for diagnostics)"

        future = cli.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            return "Trajectory goal rejected (could not query controller state)"

        for c in future.result().controller:
            if c.name == 'scaled_joint_trajectory_controller':
                if c.state == 'inactive':
                    return "scaled_joint_trajectory_controller is not active"
                elif c.state == 'unconfigured':
                    return "scaled_joint_trajectory_controller is unconfigured"
                elif c.state == 'active':
                    return "External control program stopped or robot in protective stop"
                else:
                    return f"scaled_joint_trajectory_controller in unexpected state: {c.state}"

        return "scaled_joint_trajectory_controller not found"

    def goal_response(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.error_message = self.diagnose_rejection()
            self.get_logger().error(f"Trajectory goal rejected: {self.error_message}")
            # Set exit flags if goal is rejected
            self.trajectory_in_progress = False
            self.movement_completed = True
            self.should_exit = True
            return

        self.current_goal_handle = goal_handle  # Store goal handle for potential cancellation
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result)

    def _try_canonical_match_with_threshold(self, detected_quat, object_name):
        """
        Try to find canonical match with current threshold, tracking best match found.
        Uses parallel threshold checking to find matches faster.
        
        Returns:
            Tuple of (canonical_quat, match_found, distance)
        """
        from primitives.utils.quaternion_orientation_controller import QuaternionOrientationController
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Load fold symmetry data
        fold_data = QuaternionOrientationController.load_fold_symmetry_json(object_name, self.symmetry_dir)
        
        if fold_data is None:
            # No symmetry data - log error once and use fallback
            if not self.fold_symmetry_error_logged:
                self.get_logger().error(
                    f"Fold symmetry data not found for object '{object_name}' in {self.symmetry_dir}. "
                    f"Expected file: {object_name}_symmetry.json. Using detected quaternion as fallback."
                )
                self.fold_symmetry_error_logged = True
            detected_quat = np.array(detected_quat)
            detected_quat = detected_quat / np.linalg.norm(detected_quat)
            return detected_quat, False, float('inf')
        
        # Generate list of thresholds to try in parallel
        thresholds_to_try = []
        threshold = self.canonical_threshold_initial
        while threshold <= self.canonical_threshold_max:
            thresholds_to_try.append(threshold)
            threshold += self.canonical_threshold_increment
        
        # Also include a very high threshold to find absolute best match
        thresholds_to_try.append(1.0)
        
        # Function to try a single threshold
        def try_threshold(thresh):
            canonical_quat, symmetry_used, distance = \
                QuaternionOrientationController.find_closest_canonical_quaternion(
                    detected_quat, fold_data, thresh
                )
            return thresh, canonical_quat, distance
        
        # Try all thresholds in parallel
        best_match = None
        best_distance = float('inf')
        best_threshold = None
        
        with ThreadPoolExecutor(max_workers=min(len(thresholds_to_try), 8)) as executor:
            # Submit all threshold checks
            future_to_threshold = {
                executor.submit(try_threshold, thresh): thresh 
                for thresh in thresholds_to_try
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_threshold):
                thresh, canonical_quat, distance = future.result()
                
                # Track best match overall
                if canonical_quat is not None and distance < best_distance:
                    best_distance = distance
                    best_match = canonical_quat
                    best_threshold = thresh
                
                # If we found a match with current threshold or lower, use it immediately
                if canonical_quat is not None and thresh <= self.current_canonical_threshold:
                    # Update current threshold if we found a match with a lower threshold
                    if thresh < self.current_canonical_threshold:
                        self.current_canonical_threshold = thresh
                    return canonical_quat, True, distance
        
        # Track best match found so far
        if best_match is not None and best_distance < self.best_canonical_distance:
            self.best_canonical_distance = best_distance
            self.best_canonical_match = best_match.copy()
        
        # If we found a match with any threshold, use it
        if best_match is not None:
            # Update current threshold to the one that worked
            if best_threshold is not None:
                self.current_canonical_threshold = best_threshold
            return best_match, True, best_distance
        else:
            # No match found, return detected normalized
            detected_quat = np.array(detected_quat)
            detected_quat = detected_quat / np.linalg.norm(detected_quat)
            return detected_quat, False, float('inf')
    
    def _update_final_pose_data(self, pose_stamped):
        """Update final pose position, quaternion, and RPY when final_object_pose is set"""
        if pose_stamped is not None:
            pose = pose_stamped.pose
            # Store position
            self.final_position = [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z)
            ]
            # Store quaternion
            self.final_orientation_quat = [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w)
            ]
            # Calculate and store RPY (in degrees)
            quat = np.array([
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            ])
            rpy_rad = R.from_quat(quat).as_euler('xyz')
            self.final_orientation_rpy_deg = [
                float(np.degrees(rpy_rad[0])),
                float(np.degrees(rpy_rad[1])),
                float(np.degrees(rpy_rad[2]))
            ]

    def _print_final_object_pose(self):
        """Print the final object pose recorded before exit"""
        if self.final_object_pose is not None:
            # Print object detection pose (regardless of mode)
            pose = self.final_object_pose
            self.get_logger().info(f"Final object pose: position=({pose.pose.position.x:.6f}, {pose.pose.position.y:.6f}, {pose.pose.position.z:.6f}), quaternion=({pose.pose.orientation.x:.6f}, {pose.pose.orientation.y:.6f}, {pose.pose.orientation.z:.6f}, {pose.pose.orientation.w:.6f})")

    def output_result_json(self, movement_type="move_to_object"):
        """Output movement result as JSON"""
        if self.movement_completed and not self.error_message and self.final_position is not None and self.final_orientation_quat is not None:
            # Success
            result = {
                "result": "success",
                "object_name": self.object_name,
                "grasp_id": self.grasp_id,
                "mode": self.mode,
                "movement_type": movement_type,
                "current_object_position": {
                    "x": round(self.final_position[0], 4),
                    "y": round(self.final_position[1], 4),
                    "z": round(self.final_position[2], 4)
                },
                "current_object_orientation": {
                    "quat": {
                        "x": round(self.final_orientation_quat[0], 6),
                        "y": round(self.final_orientation_quat[1], 6),
                        "z": round(self.final_orientation_quat[2], 6),
                        "w": round(self.final_orientation_quat[3], 6)
                    },
                    "rpy": {
                        "roll": round(self.final_orientation_rpy_deg[0], 4),
                        "pitch": round(self.final_orientation_rpy_deg[1], 4),
                        "yaw": round(self.final_orientation_rpy_deg[2], 4)
                    }
                }
            }
        else:
            # Failure
            result = {
                "result": "failure",
                "object_name": self.object_name,
                "grasp_id": self.grasp_id,
                "mode": self.mode,
                "movement_type": movement_type,
                "error": self.error_message if self.error_message else "Movement did not complete successfully"
            }

        output_result(result)
    
    def goal_result(self, future):
        """Handle goal result (used for both sim and real modes)"""
        result = future.result()
        self.trajectory_in_progress = False  # Clear trajectory in progress flag
        self.current_goal_handle = None  # Clear goal handle
        
        # Check if this was a cancellation due to fresh data in step 2
        if result.status == 5:  # CANCELED
            if self.cancelled_for_fresh_data:
                # This was cancelled because fresh data arrived - don't exit, let timer recompute
                self.cancelled_for_fresh_data = False  # Reset flag
                return  # Don't exit, timer callback will recompute step 2
            elif self.step1_completed:
                # In step 2 and cancelled (but flag not set) - might be due to fresh data, don't exit
                # This handles race conditions where flag might not be set yet
                return  # Don't exit, timer callback will recompute step 2
            else:
                # Cancelled for other reasons (not in step 2)
                self.get_logger().error(f"Trajectory cancelled with status: {result.status}")
                self.movement_completed = True
                self.should_exit = True
                return
        
        if result.status == 4:  # SUCCEEDED
            # Check if step 1 completed, trigger step 2 (both sim and real modes)
            if not self.step1_completed:
                self.step1_completed = True
                # Switch to faster timer period for step 2 to get more frequent pose updates
                self.update_timer.cancel()
                self.update_timer = self.create_timer(self.timer_period_step2, self.timer_callback)
                self.get_logger().info("Step 1 completed. Starting Step 2: fine positioning")
                # Don't exit - let timer callback trigger step 2 with latest pose
                return
            # Step 2 succeeded
            self._print_final_object_pose()
            self.movement_completed = True
            self.should_exit = True
            self.get_logger().info("Movement completed successfully")
            return

        # Trajectory failed - map error codes to user-friendly messages
        result_msg = result.result
        error_messages = {
            FollowJointTrajectory.Result.INVALID_GOAL: "Trajectory rejected: invalid goal (may indicate velocity/acceleration limits exceeded or joint limits violated)",
            FollowJointTrajectory.Result.INVALID_JOINTS: "Invalid joints: joint names don't match",
            FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP: "Old header timestamp: trajectory is too old",
            FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED: "Velocity or acceleration limits exceeded. The required velocity to reach the target exceeds joint velocity limits. Enable robot in URcap to fix this.",
            FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED: "Goal tolerance violated: did not reach target position",
        }
        error_msg = error_messages.get(result_msg.error_code, None)
        if error_msg is None:
            if result.status == 6:  # ABORTED
                error_msg = "Trajectory ABORTED: likely protective stop or velocity/acceleration limits exceeded. Click 'Continue' in URSim/URcap to clear the error, then retry."
            else:
                error_msg = f"Trajectory failed with status code {result.status}"
        self.error_message = error_msg
        self.get_logger().error(self.error_message)

        self._print_final_object_pose()
        self.movement_completed = True
        self.should_exit = True
    


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Direct Object Movement Node')
    parser.add_argument('--topic', type=str, default=None, 
                       help='Topic name for object poses subscription (default: /objects_poses_sim for sim mode, /objects_poses_real for real mode)')
    parser.add_argument('--object-name', type=str, default="fork_orange_scaled70",
                       help='Name of the object to move to (e.g., blue_dot_0, red_dot_0)')
    parser.add_argument('--height', type=float, default=None,
                       help='Exact gripper center height in meters (if not specified, uses grasp point Z minus offset)')
    parser.add_argument('--movement-duration', type=float, default=5.0,
                       help='Duration for the movement in seconds (default: 5.0)')
    parser.add_argument('--target-xyz', type=float, nargs=3, default=None,
                       help='Optional target position [x, y, z] in meters')
    parser.add_argument('--target-xyzw', type=float, nargs=4, default=None,
                       help='Optional target orientation [x, y, z, w] quaternion')
    parser.add_argument('--grasp-points-topic', type=str, default="/grasp_points",
                       help='Topic name for grasp points subscription')
    parser.add_argument('--grasp-id', type=int, required=True,
                       help='Specific grasp point ID to use (required - will use grasp point instead of object center)')
    parser.add_argument('--offset', type=float, default=None,
                       help='Vertical offset below grasp point for gripper center in meters (default: 0 = gripper center at grasp point)')
    parser.add_argument('--mode', type=str, default=None, choices=['sim', 'real'], required=True,
                       help='Mode: "sim" for simulation (uses /objects_poses_sim with TFMessage), "real" for real robot (uses /objects_poses_real with TFMessage). REQUIRED - no default.')
    parser.add_argument('--move-to-object', action='store_true',
                       help='Move to object (default mode - must be specified)')
    parser.add_argument('--move-to-safe-height', action='store_true',
                       help='Only move to safe height (after closing gripper)')
    
    # Parse arguments from sys.argv if args is None
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    # Check that at least one mode flag is specified
    if not args.move_to_object and not args.move_to_safe_height:
        parser.error("Must specify either --move-to-object or --move-to-safe-height")
    
    # If move-to-safe-height flag is set, only call move_to_safe_height and exit
    if args.move_to_safe_height:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print("[INFO] Moving to safe height...")
        try:
            # Set PYTHONPATH to include project root for imports
            env = os.environ.copy()
            project_root = os.path.dirname(script_dir)
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = project_root
            
            cmd_parts = [
                f"cd {script_dir}",
                f"timeout 30 /usr/bin/python3 legacy/move_to_safe_height.py"
            ]
            cmd = "\n".join(cmd_parts)
            
            result = subprocess.run(
                cmd,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True,
                timeout=40,
                env=env
            )
            
            # Log output
            if result.stdout:
                print(f"[INFO] Move to safe height output: {result.stdout}")
            if result.stderr:
                print(f"[WARN] Move to safe height stderr: {result.stderr}")
            
            if result.returncode != 0:
                print(f"[ERROR] Move to safe height failed with return code: {result.returncode}")
                return
            else:
                print("[INFO] Successfully moved to safe height")
                return  # Exit after safe height movement
                
        except subprocess.TimeoutExpired:
            print("[ERROR] Move to safe height timed out")
            return
        except KeyboardInterrupt:
            print("\n[INFO] Move to safe height stopped by user")
            return
        except Exception as e:
            print(f"[ERROR] Failed to execute move to safe height: {e}")
            return
    
    # Only proceed with object movement if --move-to-object flag is set
    if not args.move_to_object:
        parser.error("Must specify --move-to-object to move to an object")
    
    rclpy.init(args=None)
    node = DirectObjectMove(topic_name=args.topic, object_name=args.object_name,
                      height=args.height, movement_duration=args.movement_duration,
                      target_xyz=args.target_xyz, target_xyzw=args.target_xyzw,
                      grasp_points_topic=args.grasp_points_topic, grasp_id=args.grasp_id,
                      offset=args.offset, mode=args.mode)

    # Wait for essential data before starting (like other primitives)
    # This eliminates the 10-second startup delay from timer-based polling
    while node.current_ee_pose is None and rclpy.ok() and not node.should_exit:
        rclpy.spin_once(node, timeout_sec=0.1)

    # Wait for grasp points if using grasp_id mode
    if args.grasp_id is not None:
        while node.latest_grasp_points is None and rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)

    # Wait for joint states
    while node.current_joint_angles is None and rclpy.ok() and not node.should_exit:
        rclpy.spin_once(node, timeout_sec=0.1)

    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Direct movement stopped by user")
    except Exception as e:
        error_str = str(e)
        if "Unable to convert function return value" in error_str or "context is invalid" in error_str:
            node.error_message = "Robot likely in protective stop. Click 'Continue' in URSim/URcap to clear the error, then retry."
        else:
            node.error_message = f"Direct movement error: {error_str}"
        node.get_logger().error(node.error_message)
    finally:
        try:
            # Output result as JSON
            node.output_result_json(movement_type="move_to_object")
            # Explicit cleanup for faster shutdown
            node.action_client.destroy()
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            # Ignore shutdown errors
            pass

if __name__ == '__main__':
    main()
