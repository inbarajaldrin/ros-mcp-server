#!/usr/bin/env python3
"""
Direct Object Movement - Native ROS2 Node
Read object poses from TFMessage and perform single direct movement to specific object by name
Includes calibration offset correction for accurate positioning
Supports grasp point selection from /grasp_points_sim or /grasp_points_real topics
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
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64, Float32
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
        self.hover_height_offset = 0.1  # Hover height above grasp point in step 1 (meters)
        self.table_clearance = 0.0  # Minimum clearance above table for gripper fingers (meters)
        
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
        
        # State tracking for three-step movement
        self.step1_completed = False  # Track if first step is done
        self.step1_z_position = None  # Store Z position from step 1
        self.step2_completed = False  # Track if second step is done
        self.step3_attempted = False  # Track if step 3 has been attempted
        self.step3_completed = False  # Track if step 3 is done
        
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

        # Subscribe to object poses topic (TFMessage)
        self.pose_sub = self.create_subscription(
            TFMessage,
            self.topic_name,
            self.tf_message_callback,
            5  # Lower QoS to reduce update frequency
        )
        self.get_logger().info(f"Subscribed to {self.topic_name} (TFMessage)")
        
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

        # Force/torque monitoring for step 2 collision detection (soft stop)
        # Thresholds match move_down.py
        self.force_monitor_timer = None
        self.force_check_interval = 0.02  # 20ms
        self.force_check_grace_period = 0.5  # seconds after movement starts
        self.force_movement_start_time = None
        self.force_threshold_reached = False
        self.current_force = np.zeros(3)
        self.current_torque = np.zeros(3)
        self.baseline_force = None
        self.baseline_torque = None
        wrench_topic = '/force_torque_sensor_broadcaster/wrench_sim' if self.mode == 'sim' else '/force_torque_sensor_broadcaster/wrench'
        self.force_sub = self.create_subscription(
            WrenchStamped, wrench_topic, self._force_callback, 10
        )
        self.force_threshold = 50.0  # N for Fx,Fy,Tx,Ty,Tz
        self.z_force_threshold = 10.0  # force change threshold for contact detection

        # Gripper width monitoring and control
        self.current_gripper_width = None
        self.initial_gripper_width = None
        self.GRIPPER_FULLY_OPEN_WIDTH = 90.0  # mm, considered fully open
        self.gripper_check_done = False  # Pre-grasp check completed
        self.gripper_open_check_done = False  # Tried opening gripper to restore visibility (once only)
        self.gripper_needs_restore = False  # Gripper was opened for visibility, needs restore before step 2
        gripper_topic = '/gripper_width_sim' if self.mode == 'sim' else '/gripper_width_offset'
        gripper_msg_type = Float64 if self.mode == 'sim' else Float32
        self.gripper_width_sub = self.create_subscription(
            gripper_msg_type, gripper_topic, self._gripper_width_callback, 10
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

        # Step 2 upward search when object not detected
        self.upward_search_in_progress = False  # Flag to track if we're doing upward search
        self.upward_search_done = False  # Flag to indicate we've already tried upward search once
        self.upward_search_distance = 0.05  # 5cm upward search from current position
        self.is_y_movement_trajectory = False  # Flag to distinguish Y movement from real step 2

        # Step 2 force sensor management
        self.force_sensor_zeroed_for_step2 = False  # Track if sensor already zeroed for current step 2

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

    def _force_callback(self, msg: WrenchStamped):
        """Force/torque callback (WrenchStamped for both sim and real)"""
        self.current_force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        self.current_torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

    def _gripper_width_callback(self, msg):
        """Gripper width callback (Float64 for sim, Float32 for real)"""
        self.current_gripper_width = msg.data
        if self.initial_gripper_width is None:
            self.initial_gripper_width = msg.data

    def _start_force_monitoring(self):
        """Start force monitoring for step 2 (baseline calibration + timer)"""
        import time as _time
        # Brief wait to ensure we have current force readings, then capture baseline
        _time.sleep(0.1)
        self.baseline_force = self.current_force.copy()
        self.baseline_torque = self.current_torque.copy()
        # Record when movement actually starts (for grace period)
        self.force_movement_start_time = _time.time()
        self.force_threshold_reached = False
        if self.force_monitor_timer is None:
            self.force_monitor_timer = self.create_timer(self.force_check_interval, self._check_force)
        self.get_logger().info(f"Force monitoring started with baseline: F={self.baseline_force}, T={self.baseline_torque}")

    def _stop_force_monitoring(self):
        """Stop force monitoring timer and reset baseline for next movement"""
        if self.force_monitor_timer is not None:
            self.force_monitor_timer.cancel()
            self.force_monitor_timer = None
        # Reset baseline forces to allow re-zeroing on next step 2
        self.baseline_force = None
        self.baseline_torque = None

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

    def _check_force(self):
        """Check Z force during step 2 trajectory — cancel on contact with object below"""
        if not self.trajectory_in_progress:
            return

        import time as _time
        if self.force_movement_start_time is not None:
            if _time.time() - self.force_movement_start_time < self.force_check_grace_period:
                return

        if self.baseline_force is None:
            return

        # Step 2 is top-down vertical descent - check Z force delta magnitude for contact
        df_z = self.current_force[2] - self.baseline_force[2]
        z_triggered = abs(df_z) >= self.z_force_threshold

        if z_triggered:
            self.get_logger().warn(f"Contact detected during step 2: Fz={df_z:.1f}N (threshold: {self.z_force_threshold}N). Soft stopping.")
            self.force_threshold_reached = True
            self.error_message = "Force stopped - contact with object detected. Move to safe height to retry with a better pose."
            self._stop_force_monitoring()
            # Cancel current trajectory
            if self.current_goal_handle is not None:
                try:
                    self.current_goal_handle.cancel_goal_async()
                except Exception as e:
                    self.get_logger().warn(f"Could not cancel trajectory: {e}")
            self.trajectory_in_progress = False
            self.movement_completed = True
            self.should_exit = True

    def timer_callback(self):
        """Process pose and perform movement to object"""
        if self.movement_completed or self.should_exit:
            return
        
        # Handle canonical pose retry mode - adjust threshold instead of moving robot
        # Threshold adjustment happens in canonical match checking, so we just continue to normal processing
        
        # Wait for trajectory to complete before sending new one (same for both modes)
        if self.trajectory_in_progress:
            # During retreat, cancel if grasp point is re-detected
            if self.is_y_movement_trajectory and self.latest_grasp_points is not None:
                for marker in self.latest_grasp_points.markers:
                    if marker.id == self.grasp_id and marker.ns == self.object_name:
                        self.get_logger().info("Grasp point re-detected during retreat — cancelling retreat")
                        if self.current_goal_handle is not None:
                            try:
                                self.current_goal_handle.cancel_goal_async()
                            except Exception as e:
                                self.get_logger().debug(f"Could not cancel retreat: {e}")
                        self.is_y_movement_trajectory = False
                        self.trajectory_in_progress = False
                        # Restore gripper to initial width before step 2
                        self._restore_gripper_if_needed()
                        # Pause before retrying step 2
                        self.waiting_after_recovery = True
                        self.recovery_wait_start_time = self.get_clock().now()
                        self.recovery_wait_duration = 1.0
                        return
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
        
        # Pre-grasp gripper check (once before step 1)
        if not self.gripper_check_done:
            if self.current_gripper_width is None:
                # Give a few seconds for topic to arrive
                if not hasattr(self, '_gripper_wait_start'):
                    self._gripper_wait_start = self.get_clock().now()
                elapsed = (self.get_clock().now() - self._gripper_wait_start).nanoseconds / 1e9
                if elapsed < 3.0:
                    self.get_logger().debug(f"Waiting for gripper width topic... ({elapsed:.1f}s)")
                    return
                else:
                    self.get_logger().warn("Gripper width topic not available. Proceeding without gripper check.")
                    self.gripper_check_done = True
            elif self.current_gripper_width < 1.0:
                self.error_message = f"Gripper is closed ({self.current_gripper_width:.1f}mm). Open gripper before running move_to_grasp."
                self.get_logger().error(self.error_message)
                self.should_exit = True
                return
            else:
                self.initial_gripper_width = self.current_gripper_width
                self.get_logger().info(f"Initial gripper width: {self.initial_gripper_width:.0f}mm")
                self.gripper_check_done = True

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

        # Get current EE quaternion (needed for trajectory computation and checks)
        current_ee_quat = [
            self.current_ee_pose.pose.orientation.x,
            self.current_ee_pose.pose.orientation.y,
            self.current_ee_pose.pose.orientation.z,
            self.current_ee_pose.pose.orientation.w,
        ]

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
            self.last_known_object_position is None):
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
                        # In step 2, if object not detected, try opening gripper or retreat (only during step 2, not after)
                        if self.step1_completed and not self.step2_completed and not self.upward_search_done:
                            self.upward_search_done = True
                            self._try_gripper_open_or_retreat(f"Object '{self.object_name}' not detected in step 2.")
                            return  # Wait for gripper check or Y movement to complete

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
                # In step 2, check if grasp point is still available
                grasp_point_available = False
                if self.latest_grasp_points is not None and len(self.latest_grasp_points.markers) > 0:
                    for marker in self.latest_grasp_points.markers:
                        if marker.id == self.grasp_id and marker.ns == self.object_name:
                            grasp_point_available = True
                            break

                # If grasp point lost in step 2, try opening gripper or retreat (only during step 2, not after)
                if not grasp_point_available and not self.step2_completed and not self.upward_search_done:
                    self.upward_search_done = True
                    self._try_gripper_open_or_retreat(f"Grasp point {self.grasp_id} lost in step 2.")
                    return  # Wait for gripper check or Y movement to complete

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
            
            # Store last known good position for recovery
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
            if was_tracking_lost:
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
            if self.last_known_object_position is not None:
                # Handle tracking loss
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

        # Step 1: always add hover height offset
        if not self.step1_completed:
            target_ee_position[2] += self.hover_height_offset  # Move to hover height

        self.get_logger().info(f"Target gripper center: ({target_ee_position[0]:.3f}, {target_ee_position[1]:.3f}, {target_ee_position[2]:.3f})")
        
        # Step 3: Attempt final fine positioning if step 2 is done and step 3 not yet attempted (real mode only)
        if self.step2_completed and not self.step3_attempted and self.grasp_id is not None and self.mode == 'real':
            self.step3_attempted = True
            self.get_logger().info("Step 3: Attempting final fine positioning with fresh grasp pose data")

            # Check if grasp point is available
            grasp_point_available = False
            if self.latest_grasp_points is not None and len(self.latest_grasp_points.markers) > 0:
                for marker in self.latest_grasp_points.markers:
                    if marker.id == self.grasp_id and marker.ns == self.object_name:
                        # Update selected_grasp_point with fresh data
                        self.selected_grasp_point = marker
                        grasp_point_available = True
                        break

            if not grasp_point_available:
                self.get_logger().info("Grasp point not available for step 3, exiting with current position")
                self.movement_completed = True
                self.should_exit = True
                self._print_final_object_pose()
                return

            # Get fresh grasp point position
            grasp_point_position = np.array([
                self.selected_grasp_point.pose.position.x,
                self.selected_grasp_point.pose.position.y,
                self.selected_grasp_point.pose.position.z
            ])

            # Use same orientation computation as step 2
            grasp_point_quat = np.array([
                self.selected_grasp_point.pose.orientation.x,
                self.selected_grasp_point.pose.orientation.y,
                self.selected_grasp_point.pose.orientation.z,
                self.selected_grasp_point.pose.orientation.w
            ])

            # Try to find canonical match
            canonical_quat, canonical_match, match_distance = self._try_canonical_match_with_threshold(
                grasp_point_quat, self.object_name
            )

            # Extract yaw from returned quaternion
            grasp_point_yaw = self.quat_controller.extract_yaw_from_quaternion(canonical_quat)
            step3_target_quaternion = self.quat_controller.face_down_quaternion(grasp_point_yaw)

            # Compute step 3 target position (same as step 2 target, but recomputed for final positioning)
            step3_target_ee = grasp_point_position.copy()
            if self.offset is not None and self.offset != 0:
                step3_target_ee[2] -= self.offset

            # Enforce minimum gripper center height
            if step3_target_ee[2] < self.min_gripper_center_z:
                step3_target_ee[2] = self.min_gripper_center_z

            # Store these for step 3 trajectory computation (will recompute trajectory with these targets)
            self.step3_target_position = step3_target_ee
            self.step3_target_quaternion = step3_target_quaternion
            self.get_logger().info(f"Step 3 target: ({step3_target_ee[0]:.3f}, {step3_target_ee[1]:.3f}, {step3_target_ee[2]:.3f})")
            # Continue to compute and send step 3 trajectory below (will use step3_target_position/quaternion if set)

        # Exit if step 3 was attempted but grasp point was unavailable
        if self.step2_completed and self.step3_attempted and not hasattr(self, 'step3_target_position'):
            return

        # Check convergence in step 2: if robot is close enough to target, count stable readings and exit
        # Only check convergence during step 2 (not step 3 or after step 2 completes)
        if self.step1_completed and not self.step2_completed:
            R_ee = R.from_quat(current_ee_quat).as_matrix()
            current_gc_position = current_ee_position + R_ee @ GRIPPER_CENTER_TOOL_OFFSET
            distance_to_target = np.linalg.norm(current_gc_position - target_ee_position)

            if distance_to_target <= self.convergence_distance_threshold:
                self.convergence_stable_count += 1

                if self.convergence_stable_count >= self.convergence_stable_threshold:
                    self.get_logger().info(f"Converged! Within {self.convergence_distance_threshold*100:.2f}cm of target for {self.convergence_stable_threshold} consecutive readings.")
                    # Mark step 2 as completed to trigger step 3 attempt
                    self.step2_completed = True
                    return  # Exit, goal_result will transition to step 3
            else:
                # Reset stable count if we're not within threshold
                self.convergence_stable_count = 0
        
        # If waiting at last known location, mark that we've sent the trajectory
        if self.waiting_at_last_known and not self.last_known_target_sent:
            self.last_known_target_sent = True

        # Use step 3 target if available
        step3_mode = False
        if self.step2_completed and self.step3_attempted and hasattr(self, 'step3_target_position'):
            step3_mode = True
            target_ee_position = self.step3_target_position
            target_quaternion = self.step3_target_quaternion
            # Clean up step 3 attributes to avoid reuse
            delattr(self, 'step3_target_position')
            delattr(self, 'step3_target_quaternion')

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
            # Step 2: use Cartesian waypoints for straight-line descent (no curved joint-space interpolation)
            from primitives.utils.ik_solver import compute_cartesian_waypoints_ik

            gc_target = np.array(ik_position)

            # Use current EE orientation for straight-line descent
            # Step 1 already aligned the gripper. No Euler angles needed — avoids gimbal lock.
            T_current = forward_kinematics(dh_params, self.current_joint_angles)
            current_rot_matrix = T_current[:3, :3]
            current_flange_fk = T_current[:3, 3]

            # Compute flange target using topic-measured gripper center delta
            # (avoids FK/topic mismatch that causes ~1-2mm error)
            R_ee_topic = R.from_quat(current_ee_quat).as_matrix()
            current_gc_topic = current_ee_position + R_ee_topic @ GRIPPER_CENTER_TOOL_OFFSET
            gc_error = gc_target - current_gc_topic
            flange_target = current_flange_fk + gc_error

            num_waypoints = 60
            step2_duration = max(movement_duration, 5.0)  # At least 5s for smooth descent
            step_label = "step 3" if step3_mode else "step 2"
            self.get_logger().info(f"Computing {num_waypoints} Cartesian waypoints for {step_label} ({step2_duration:.1f}s)...")

            waypoints = compute_cartesian_waypoints_ik(
                self.current_joint_angles, target_z=flange_target[2],
                num_waypoints=num_waypoints, target_pos=flange_target,
                target_orientation=current_rot_matrix
            )

            if waypoints is None:
                step_label = "step 3" if step3_mode else "step 2"
                self.error_message = f"Failed to compute Cartesian waypoints for {step_label}"
                self.get_logger().error(self.error_message)
                self.trajectory_in_progress = True
                self.execute_trajectory({"traj1": []})
                return

            # Build trajectory with trapezoidal velocity profile (same as move_to_safe_height)
            all_joint_angles = [self.current_joint_angles.copy()] + list(waypoints)
            total_duration = step2_duration
            n_total = len(all_joint_angles)

            segment_dists = []
            for i in range(1, n_total):
                dist = np.linalg.norm(all_joint_angles[i] - all_joint_angles[i - 1])
                segment_dists.append(max(dist, 1e-6))
            cumulative_s = [0.0]
            for d in segment_dists:
                cumulative_s.append(cumulative_s[-1] + d)
            total_s = cumulative_s[-1]

            accel_frac = 0.2
            decel_frac = 0.2
            t_accel = accel_frac * total_duration
            t_decel = decel_frac * total_duration
            t_cruise = total_duration - t_accel - t_decel
            v_max = total_s / (0.5 * t_accel + t_cruise + 0.5 * t_decel)
            a_accel = v_max / t_accel
            a_decel = v_max / t_decel

            def trapez_s_and_v(t_query):
                if t_query <= t_accel:
                    s = 0.5 * a_accel * t_query ** 2
                    v = a_accel * t_query
                elif t_query <= t_accel + t_cruise:
                    s_accel = 0.5 * v_max * t_accel
                    s = s_accel + v_max * (t_query - t_accel)
                    v = v_max
                else:
                    s_accel = 0.5 * v_max * t_accel
                    s_cruise = v_max * t_cruise
                    t_in_decel = t_query - t_accel - t_cruise
                    s = s_accel + s_cruise + v_max * t_in_decel - 0.5 * a_decel * t_in_decel ** 2
                    v = v_max - a_decel * t_in_decel
                return s, max(v, 0.0)

            def find_time_for_s(target_s):
                lo, hi = 0.0, total_duration
                for _ in range(50):
                    mid = (lo + hi) / 2
                    s_mid, _ = trapez_s_and_v(mid)
                    if s_mid < target_s:
                        lo = mid
                    else:
                        hi = mid
                return (lo + hi) / 2

            waypoint_times = [find_time_for_s(s) for s in cumulative_s]
            waypoint_times[0] = 0.0
            waypoint_times[-1] = total_duration

            traj_points = []
            for i in range(n_total):
                t_i = waypoint_times[i]
                _, speed_scalar = trapez_s_and_v(t_i)

                if i == 0 or i == n_total - 1:
                    velocities = [0.0] * 6
                else:
                    delta = all_joint_angles[i + 1] - all_joint_angles[i - 1]
                    delta_norm = np.linalg.norm(delta)
                    if delta_norm > 1e-8:
                        direction = delta / delta_norm
                        velocities = [float(speed_scalar * direction[j]) for j in range(6)]
                    else:
                        velocities = [0.0] * 6

                point = JointTrajectoryPoint(
                    positions=[float(x) for x in all_joint_angles[i]],
                    velocities=velocities,
                    time_from_start=Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
                )
                traj_points.append(point)

            self.get_logger().info(f"Generated {len(traj_points)} Cartesian waypoints with trapezoidal velocity profile")

            # Send trajectory directly (bypass execute_trajectory dict format)
            traj_msg = JointTrajectory()
            traj_msg.joint_names = [
                'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
            ]
            traj_msg.points = traj_points

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)

            self.trajectory_in_progress = True
            step_label = "Step 3" if step3_mode else "Step 2"
            self.get_logger().info(f"{step_label} trajectory sent and accepted")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)

            self.last_target_pose = (target_ee_position.tolist(), target_quaternion)
            return
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

        # Zero force sensor only in step 2, right before first trajectory execution
        # This ensures sensor is zeroed only after object is found, not during retries
        if self.step1_completed and self.baseline_force is None and not self.force_sensor_zeroed_for_step2:
            self.zero_force_sensor()
            self.force_sensor_zeroed_for_step2 = True
            import time as _time
            _time.sleep(0.5)  # Wait for sensor to settle after zeroing

        # Execute trajectory (same for both modes: mark as in progress and wait for completion)
        self.trajectory_in_progress = True
        self.execute_trajectory(trajectory)
        
        # Update last target pose for similarity checking (PURE QUATERNION, no RPY)
        self.last_target_pose = (target_ee_position.tolist(), target_quaternion)

        # Don't set movement_completed here - wait for trajectory completion callback

    def _try_gripper_open_or_retreat(self, reason: str):
        """Try opening gripper to restore visibility before falling back to retreat.

        If gripper is partially closed (< 90mm) and we haven't tried this yet,
        opens it to check if the object becomes visible. Only attempts once —
        if it didn't help, go straight to retreat on subsequent calls.
        """
        if (not self.gripper_open_check_done and
                self.current_gripper_width is not None and
                self.current_gripper_width < self.GRIPPER_FULLY_OPEN_WIDTH):
            self.gripper_open_check_done = True
            self.get_logger().info(f"{reason} Gripper at {self.current_gripper_width:.0f}mm — trying open to restore visibility...")
            self._open_gripper_and_check()
        else:
            self.get_logger().info(f"{reason} Retreating...")
            self.perform_retreat_from_object()

    def _run_control_gripper(self, command):
        """Run control_gripper.py as subprocess. Returns True on success."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = ['python3', os.path.join(script_dir, 'control_gripper.py'), str(command), '--mode', self.mode]
        self.get_logger().info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                self.get_logger().info(f"control_gripper succeeded: {command}")
                return True
            else:
                self.get_logger().warn(f"control_gripper failed (rc={result.returncode}): {result.stdout.strip()}")
                return False
        except subprocess.TimeoutExpired:
            self.get_logger().warn(f"control_gripper timed out for command: {command}")
            return False
        except Exception as e:
            self.get_logger().warn(f"control_gripper error: {e}")
            return False

    def _is_grasp_visible(self):
        """Check if the target grasp point is in the latest grasp points message."""
        if self.latest_grasp_points is not None:
            for marker in self.latest_grasp_points.markers:
                if marker.id == self.grasp_id and marker.ns == self.object_name:
                    return True
        return False

    def _restore_gripper_if_needed(self):
        """Restore gripper to initial width if it was opened for visibility check."""
        if self.gripper_needs_restore and self.initial_gripper_width is not None:
            self.get_logger().info(f"Restoring gripper to initial width ({self.initial_gripper_width:.0f}mm)...")
            self._run_control_gripper(int(self.initial_gripper_width))
            self.gripper_needs_restore = False

    def _open_gripper_and_check(self):
        """Open gripper via control_gripper.py, check detection, then restore or retreat.

        control_gripper blocks until gripper is fully open and verified, by which
        time the camera has had time to update. Check visibility immediately after.
        If visible: restore gripper to initial width and proceed to step 2.
        If not visible: retreat (gripper stays open), restore happens when grasp
        is re-detected after retreat.
        """
        self._run_control_gripper("open")
        self.gripper_needs_restore = True

        if self._is_grasp_visible():
            self.get_logger().info("Grasp point visible after opening gripper! Restoring to initial width.")
            self._restore_gripper_if_needed()
            self.upward_search_done = False
        else:
            self.get_logger().info("Grasp point not visible after opening gripper. Retreating...")
            self.perform_retreat_from_object()

    def perform_retreat_from_object(self):
        """Move 0.05m along the EE's +Y axis in XY, maintaining Z height.

        Uses the EE's current orientation (from FK) to get the local +Y direction
        projected onto the world XY plane, then moves 0.05m in that direction.
        Uses trapezoidal velocity profile for smooth motion (same as step 2).
        """
        if self.current_ee_pose is None or self.current_joint_angles is None:
            self.get_logger().warn("Cannot perform retreat: current EE pose or joint angles not available")
            self.trajectory_in_progress = False
            return

        from primitives.utils.ik_solver import compute_cartesian_waypoints_ik

        T_current = forward_kinematics(dh_params, self.current_joint_angles)
        current_rot_matrix = T_current[:3, :3]
        fk_pos = T_current[:3, 3]  # Flange position from FK

        retreat_distance = 0.05  # 5cm

        # Use EE's +Y axis projected onto world XY as the retreat direction
        ee_y_world = current_rot_matrix[:2, 1]  # Second column (Y axis), XY components
        ee_y_norm = np.linalg.norm(ee_y_world)
        if ee_y_norm > 1e-6:
            ee_y_world = ee_y_world / ee_y_norm
        else:
            ee_y_world = np.array([0.0, 1.0])

        # Ensure we move toward +Y in world frame (flip if EE +Y points toward -Y)
        if ee_y_world[1] < 0:
            ee_y_world = -ee_y_world

        target_flange = fk_pos.copy()
        target_flange[0] += ee_y_world[0] * retreat_distance
        target_flange[1] += ee_y_world[1] * retreat_distance
        target_flange[2] += 0.05  # Also move up 5cm in world Z

        # Safety check: reject if target is too close to robot base in XY
        min_base_distance_xy = 0.20  # 20cm minimum distance from base in XY plane
        target_xy_dist = np.linalg.norm(target_flange[:2])
        if target_xy_dist < min_base_distance_xy:
            self.get_logger().warn(
                f"Object too close to robot base frame. Collision-free IK not possible. "
                f"Target XY distance from base: {target_xy_dist*1000:.0f}mm (min: {min_base_distance_xy*1000:.0f}mm)"
            )
            self.trajectory_in_progress = False
            return

        self.get_logger().info(
            f"Retreating along EE +Y toward world +Y: ({fk_pos[0]:.3f}, {fk_pos[1]:.3f}, {fk_pos[2]:.3f}) -> "
            f"({target_flange[0]:.3f}, {target_flange[1]:.3f}, {target_flange[2]:.3f}), "
            f"dir=({ee_y_world[0]:.2f}, {ee_y_world[1]:.2f})"
        )

        try:

            num_waypoints = 30
            total_duration = 3.0
            waypoints = compute_cartesian_waypoints_ik(
                self.current_joint_angles, target_z=target_flange[2],
                num_waypoints=num_waypoints, target_pos=target_flange,
                target_orientation=current_rot_matrix
            )

            if waypoints is None:
                self.get_logger().warn("Failed to compute retreat waypoints")
                self.trajectory_in_progress = False
                return

            # Build trajectory with trapezoidal velocity profile (same as step 2)
            all_joint_angles = [self.current_joint_angles.copy()] + list(waypoints)
            n_total = len(all_joint_angles)

            segment_dists = []
            for i in range(1, n_total):
                dist = np.linalg.norm(all_joint_angles[i] - all_joint_angles[i - 1])
                segment_dists.append(max(dist, 1e-6))
            cumulative_s = [0.0]
            for d in segment_dists:
                cumulative_s.append(cumulative_s[-1] + d)
            total_s = cumulative_s[-1]

            accel_frac = 0.2
            decel_frac = 0.2
            t_accel = accel_frac * total_duration
            t_decel = decel_frac * total_duration
            t_cruise = total_duration - t_accel - t_decel
            v_max = total_s / (0.5 * t_accel + t_cruise + 0.5 * t_decel)
            a_accel = v_max / t_accel
            a_decel = v_max / t_decel

            def trapez_s_and_v(t_query):
                if t_query <= t_accel:
                    s = 0.5 * a_accel * t_query ** 2
                    v = a_accel * t_query
                elif t_query <= t_accel + t_cruise:
                    s_accel = 0.5 * v_max * t_accel
                    s = s_accel + v_max * (t_query - t_accel)
                    v = v_max
                else:
                    s_accel = 0.5 * v_max * t_accel
                    s_cruise = v_max * t_cruise
                    t_in_decel = t_query - t_accel - t_cruise
                    s = s_accel + s_cruise + v_max * t_in_decel - 0.5 * a_decel * t_in_decel ** 2
                    v = v_max - a_decel * t_in_decel
                return s, max(v, 0.0)

            def find_time_for_s(target_s):
                lo, hi = 0.0, total_duration
                for _ in range(50):
                    mid = (lo + hi) / 2
                    s_mid, _ = trapez_s_and_v(mid)
                    if s_mid < target_s:
                        lo = mid
                    else:
                        hi = mid
                return (lo + hi) / 2

            waypoint_times = [find_time_for_s(s) for s in cumulative_s]
            waypoint_times[0] = 0.0
            waypoint_times[-1] = total_duration

            traj_points = []
            for i in range(n_total):
                t_i = waypoint_times[i]
                _, speed_scalar = trapez_s_and_v(t_i)

                if i == 0 or i == n_total - 1:
                    velocities = [0.0] * 6
                else:
                    delta = all_joint_angles[i + 1] - all_joint_angles[i - 1]
                    delta_norm = np.linalg.norm(delta)
                    if delta_norm > 1e-8:
                        direction = delta / delta_norm
                        velocities = [float(speed_scalar * direction[j]) for j in range(6)]
                    else:
                        velocities = [0.0] * 6

                point = JointTrajectoryPoint(
                    positions=[float(x) for x in all_joint_angles[i]],
                    velocities=velocities,
                    time_from_start=Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
                )
                traj_points.append(point)

            # Send trajectory directly (bypass execute_trajectory which only uses first point)
            traj_msg = JointTrajectory()
            traj_msg.joint_names = [
                'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
            ]
            traj_msg.points = traj_points

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)

            self.trajectory_in_progress = True
            self.is_y_movement_trajectory = True
            self.get_logger().info(f"Retreat trajectory sent: {n_total} waypoints over {total_duration:.1f}s")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)

        except Exception as e:
            self.get_logger().warn(f"Retreat movement error: {e}")
            self.trajectory_in_progress = False

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

        # Start force monitoring during step 2 descent (real mode only, not for Y movements)
        if self.mode == 'real' and self.step1_completed and not self.is_y_movement_trajectory:
            self._start_force_monitoring()

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
        self._stop_force_monitoring()
        result = future.result()
        self.trajectory_in_progress = False  # Clear trajectory in progress flag
        self.current_goal_handle = None  # Clear goal handle

        # If force threshold caused the cancellation, report as failure with guidance
        if self.force_threshold_reached:
            self.get_logger().warn("Step 2 stopped due to contact detection (soft stop)")
            self.error_message = "Force stopped - collision with object detected. Move to safe height to retry with a better pose."
            self._print_final_object_pose()
            self.movement_completed = True
            self.should_exit = True
            return

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
            # Check if this was a Y movement
            if self.is_y_movement_trajectory:
                self.is_y_movement_trajectory = False
                self.upward_search_in_progress = False
                self.get_logger().info("Retreat completed. Waiting 0.5s before retrying step 2...")
                # Pause 0.5s before retrying to let detection settle
                self.waiting_after_recovery = True
                self.recovery_wait_start_time = self.get_clock().now()
                self.recovery_wait_duration = 1.0
                return

            # Check if this was an upward search movement
            if self.upward_search_in_progress:
                self.upward_search_in_progress = False
                self.get_logger().info("Upward search completed. Retrying object detection...")
                # Don't exit - let timer callback retry detection from new height
                return

            # Check if step 1 completed, trigger step 2 (both sim and real modes)
            if not self.step1_completed:
                self.step1_completed = True
                self.upward_search_done = False  # Reset search flag for step 2
                self.force_sensor_zeroed_for_step2 = False  # Reset force sensor flag for step 2
                # Switch to faster timer period for step 2 to get more frequent pose updates
                self.update_timer.cancel()
                self.update_timer = self.create_timer(self.timer_period_step2, self.timer_callback)
                self.get_logger().info("Step 1 completed. Starting Step 2: fine positioning")
                # Don't exit - let timer callback trigger step 2 with latest pose
                return

            # Check if step 2 completed
            if self.step1_completed and not self.step2_completed:
                self.step2_completed = True
                self.get_logger().info("Step 2 completed. Attempting Step 3: final fine positioning")
                # Don't exit yet - let timer callback attempt step 3
                return

            # Step 3 completed (or step 3 was not attempted)
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
                       help='Duration for the movement in seconds (default: 3.0)')
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
