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
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# Import from local action_libraries file
from primitives.utils.action_libraries import hover_over_grasp_quat

# Import quaternion controller for gimbal-lock-free gripper orientation
from primitives.utils.quaternion_orientation_controller import QuaternionOrientationController

# Import path finder for auto-discovering aruco-grasp-annotator data directory
from primitives.utils.data_path_finder import get_symmetry_dir

# Import the new message types
try:
    from max_camera_msgs.msg import ObjectPoseArray
except ImportError:
    # Fallback if the message type is not available
    print("Warning: max_camera_msgs not found. Using geometry_msgs.PoseStamped as fallback.")
    ObjectPoseArray = None

# Import grasp points message type (using standard visualization_msgs MarkerArray)
from visualization_msgs.msg import MarkerArray, Marker

class PoseKalmanFilter:
    """Kalman filter for pose estimation and smoothing"""
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        # State vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw]
        self.state_dim = 12
        self.measurement_dim = 6  # [x, y, z, roll, pitch, yaw]
        
        # State vector
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 10  # Initial covariance
        
        # Process noise
        self.Q = np.eye(self.state_dim) * process_noise
        
        # Measurement noise
        self.R = np.eye(self.measurement_dim) * measurement_noise
        
        # Measurement matrix (we only measure position and orientation)
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[:6, :6] = np.eye(6)
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(self.state_dim)
        dt = 1.0  # Time step (will be updated dynamically)
        self.F[:6, 6:] = np.eye(6) * dt
        
        self.initialized = False
        
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
    
    def update(self, pose_msg, dt=1.0):
        """Update Kalman filter with new pose measurement"""
        # Extract measurement
        position = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
        rpy = self.quaternion_to_rpy(
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w
        )
        
        measurement = np.array(position + rpy)
        
        # Update state transition matrix with current dt
        self.F[:6, 6:] = np.eye(6) * dt
        
        if not self.initialized:
            # Initialize state
            self.x[:6] = measurement
            self.initialized = True
            return self.x[:6], self.x[6:12]  # Return position and velocity
        
        # Predict step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update step
        y = measurement - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
        
        return self.x[:6], self.x[6:12]  # Return filtered position and velocity
    
    def get_filtered_pose(self):
        """Get current filtered pose"""
        if not self.initialized:
            return None, None
        return self.x[:6], self.x[6:12]

class DirectObjectMove(Node):
    def __init__(self, topic_name=None, object_name="blue_dot_0", height=None, movement_duration=5.0, target_xyz=None, target_xyzw=None, grasp_points_topic="/grasp_points", grasp_id=None, offset=None, mode=None):
        super().__init__('direct_object_move')
        
        # Mode must be explicitly specified - no default
        if mode is None:
            raise ValueError("Mode must be explicitly specified. Use 'sim' or 'real'.")
        if mode not in ['sim', 'real']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'sim' or 'real'.")
        
        self.mode = mode  # 'sim' or 'real'
        
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
        
        # Calibration offset to correct systematic detection bias (only for real mode)
        if self.mode == 'real':
            # First step offsets (initial movement)
            self.calibration_offset_x = 0.0  # X-axis correction
            self.calibration_offset_y = +0.0  # Y-axis correction
            self.calibration_offset_z = 0.05  # Z-axis correction (height)
            # Second step offsets (fine adjustment)
            self.fine_offset_x = 0.00  # Fine X-axis correction
            self.fine_offset_y = -0.00  # Fine Y-axis correction
            self.fine_offset_z = -0.048  # Fine Z-axis correction (height)
            # State tracking for two-step movement
            self.step1_completed = False  # Track if first step is done
            self.step1_z_position = None  # Store Z position from step 1
        else:
            self.calibration_offset_x = 0.000
            self.calibration_offset_y = 0.000
            self.calibration_offset_z = 0.000
            self.fine_offset_x = 0.000
            self.fine_offset_y = 0.000
            self.fine_offset_z = 0.000
            self.step1_completed = False
            self.step1_z_position = None
        
        # TCP to gripper center offset distance (from TCP to gripper center along gripper Z-axis)
        # This implements a spherical flexure joint concept (same as URSim TCP control):
        # - The offset point (gripper center) acts as a fixed point in space
        # - When rotating the gripper, TCP moves to keep the offset point fixed
        # - offset_point = tcp_position + tcp_to_gripper_center_offset * z_axis_gripper
        # - tcp_position = offset_point - tcp_to_gripper_center_offset * z_axis_gripper
        self.tcp_to_gripper_center_offset = 0.24  # 0.24m = 24cm (distance from TCP to gripper center)
        
        # Offset from target object to gripper center (grasp candidate position to gripper center)
        # This is the distance from object/grasp point to gripper center
        # Gripper center = object_position - offset (below object)
        # TCP = gripper center + 0.24 (above gripper center)
        # So: TCP = object - offset + 0.24 = object + (0.24 - offset)
        # Example: offset=0.123 -> gripper_center = object - 0.123, TCP = object - 0.123 + 0.24 = object + 0.117
        # When offset increases, gripper center moves further down, TCP moves down
        self.object_to_gripper_center_offset = offset if offset is not None else 0.123  # Default: 0.123m = 12.3cm
        
        # Initialize Kalman filter
        self.kalman_filter = PoseKalmanFilter(process_noise=0.005, measurement_noise=0.05)
        self.last_update_time = None
        
        # Initialize Quaternion Orientation Controller for gimbal-lock-free gripper control
        # This ensures stable gripper orientation at pitch=180° (face down) for any yaw angle
        self.quat_controller = QuaternionOrientationController()
        self.get_logger().info("✅ Quaternion orientation controller initialized (gimbal-lock-free mode)")
        
        # Fold symmetry directory for canonical pose matching (auto-discovered)
        self.symmetry_dir = str(get_symmetry_dir())
        
        # Store latest grasp points
        self.latest_grasp_points = None
        self.selected_grasp_point = None
        
        # Track final object pose for logging before exit
        self.final_object_pose = None  # Store final object pose (PoseStamped) before exit
        
        # Store current end-effector pose
        self.current_ee_pose = None
        self.ee_pose_received = False
        
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

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE, 
            depth=10
        )
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            qos_profile
        )
        
        # Subscribe to grasp points topic if grasp_id is provided
        if self.grasp_id is not None:
            self.grasp_points_sub = self.create_subscription(
                MarkerArray,
                self.grasp_points_topic,
                self.grasp_points_callback,
                5
            )
            self.get_logger().info(f"🎯 Grasp point mode: Looking for grasp_id {grasp_id} on topic {self.grasp_points_topic}")
        else:
            self.grasp_points_sub = None
        
        # Add timer to control update frequency (same for both modes)
        # Use shorter period for step 2 to get more frequent updates as robot gets closer
        self.timer_period_step1 = 5.0
        self.timer_period_step2 = 1.0  # More frequent updates for step 2
        self.update_timer = self.create_timer(self.timer_period_step1, self.timer_callback)
        
        # Track last pose update time to ensure we use fresh data
        self.last_grasp_point_update_time = None
        self.last_pose_update_time = None
        self.latest_pose = None
        self.movement_completed = False  # Flag to track if movement has been completed
        self.should_exit = False  # Flag to control exit
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
        
        # Canonical pose retry mechanism - adjust threshold instead of moving robot
        self.canonical_retry_mode = False  # Flag to indicate we're retrying for canonical pose
        self.canonical_retry_count = 0  # Number of retry attempts
        self.max_canonical_retries = 10  # Maximum number of retries before giving up
        self.canonical_threshold_initial = 0.45  # Initial threshold (~90°)
        self.canonical_threshold_max = 0.9  # Maximum threshold to try (~100°)
        self.canonical_threshold_increment = 0.1  # Increment threshold by this amount each retry
        self.current_canonical_threshold = self.canonical_threshold_initial  # Current threshold being used
        self.best_canonical_match = None  # Store best match found so far
        self.best_canonical_distance = float('inf')  # Distance of best match
        
        # Action client for trajectory execution
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        self.get_logger().info(f"🤖 Direct object movement started for object '{object_name}' on topic {topic_name}")
        if height is not None:
            self.get_logger().info(f"📏 Target height: {height}m (offset will be ignored)")
        else:
            self.get_logger().info(f"📏 Using {self.tcp_to_gripper_center_offset*100:.1f}cm TCP to gripper center offset (along gripper Z-axis)")
            self.get_logger().info(f"📏 Using {self.object_to_gripper_center_offset*100:.1f}cm object to gripper center offset")
        self.get_logger().info(f"⏱️ Movement duration: {movement_duration}s")
        if self.grasp_id is not None:
            self.get_logger().info(f"🎯 Grasp point mode: Using grasp_id {grasp_id} from topic {self.grasp_points_topic}")
        else:
            self.get_logger().info(f"🎯 Object center mode: Moving to object center")
        
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
    
    def compute_offset_point(self, tcp_position, quaternion):
        """Compute the offset point from TCP position using spherical flexure joint concept
        (Same as URSim TCP control)
        
        The offset vector is defined in the tool frame (gripper frame) and then
        transformed to world frame using the tool orientation quaternion.
        
        Args:
            tcp_position: TCP position in world frame [x, y, z]
            quaternion: TCP/tool orientation quaternion [x, y, z, w] (tool frame to world frame)
        
        Returns:
            offset_point: Position of the offset point (gripper center) in world frame [x, y, z]
        """
        # Offset vector in tool frame (gripper frame): [0, 0, offset_distance]
        # In tool frame, Z-axis points from TCP to gripper center (downward)
        offset_vector_tool_frame = np.array([0.0, 0.0, self.tcp_to_gripper_center_offset])
        
        # Transform offset vector from tool frame to world frame using quaternion
        # The quaternion represents the rotation from tool frame to world frame
        r = R.from_quat(quaternion)
        offset_vector_world = r.apply(offset_vector_tool_frame)
        
        # Compute offset point: TCP + offset_vector_world
        # (going forward from TCP to gripper center along the tool Z-axis)
        offset_point = np.array(tcp_position) + offset_vector_world
        
        return offset_point.tolist()
    
    def compute_tcp_from_offset_point(self, offset_point, quaternion):
        """Compute TCP position from offset point using the gripper orientation
        
        The offset is computed along the gripper Z-axis in world frame.
        The gripper Z-axis is obtained from the quaternion orientation.
        
        Args:
            offset_point: Position of the gripper center (offset point) in world frame [x, y, z]
            quaternion: Gripper orientation quaternion [x, y, z, w] (gripper frame to world frame)
                       The quaternion's Z-axis points from TCP to gripper center.
        
        Returns:
            tcp_position: TCP position in world frame [x, y, z]
        """
        # Get gripper Z-axis direction in world frame
        r = R.from_quat(quaternion)
        gripper_z_axis = r.apply(np.array([0.0, 0.0, 1.0]))  # Gripper Z-axis in world frame
        gripper_z_axis = gripper_z_axis / np.linalg.norm(gripper_z_axis)  # Normalize
        
        # Compute offset vector in world frame
        # The offset goes from gripper center to TCP, opposite to gripper Z-axis
        offset_vector_world = -self.tcp_to_gripper_center_offset * gripper_z_axis
        
        # Compute TCP position: offset_point + offset_vector_world
        # (going from gripper center towards TCP, opposite to gripper Z-axis)
        tcp_position = np.array(offset_point) + offset_vector_world
        
        return tcp_position.tolist()
    
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
    
    def objects_poses_callback(self, msg):
        """Handle ObjectPoseArray message and find target object"""
        if ObjectPoseArray is None:
            return
            
        # Find the object with the specified name
        target_object = None
        for obj in msg.objects:
            if obj.object_name == self.object_name:
                target_object = obj
                break
        
        if target_object is not None:
            # Convert ObjectPose to PoseStamped for compatibility
            pose_stamped = PoseStamped()
            pose_stamped.header = target_object.header
            pose_stamped.pose = target_object.pose
            self.latest_pose = pose_stamped
        else:
            # Object not found in this message
            self.get_logger().warn(f"Object '{self.object_name}' not found in current message")
            self.latest_pose = None
    
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
                self.get_logger().info(f"🔄 Fresh object pose data received during step 2! Cancelling current trajectory and recomputing...")
                # Cancel current trajectory
                if self.current_goal_handle is not None:
                    try:
                        cancel_future = self.current_goal_handle.cancel_goal_async()
                        self.get_logger().info("Cancelling current trajectory...")
                    except Exception as e:
                        self.get_logger().warn(f"Could not cancel trajectory: {e}")
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
        else:
            # Object not found in this message
            self.latest_pose = None
    
    def pose_callback(self, msg):
        """Store latest pose message (fallback for PoseStamped)"""
        self.latest_pose = msg
        # Track final object pose for logging
        self.final_object_pose = msg
    
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
                self.get_logger().info(f"🔄 Fresh grasp point data received during step 2! Cancelling current trajectory and recomputing...")
                # Cancel current trajectory
                if self.current_goal_handle is not None:
                    try:
                        # Set flag BEFORE cancelling to ensure it's set when goal_result is called
                        self.cancelled_for_fresh_data = True
                        self.get_logger().info(f"Setting cancelled_for_fresh_data=True before cancelling")
                        cancel_future = self.current_goal_handle.cancel_goal_async()
                        self.get_logger().info("Cancelling current trajectory...")
                    except Exception as e:
                        self.get_logger().warn(f"Could not cancel trajectory: {e}")
                        self.cancelled_for_fresh_data = False
                else:
                    self.get_logger().warn("No goal handle available to cancel")
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
    
    def timer_callback(self):
        """Process pose and perform movement to object"""
        if self.movement_completed:
            return
        
        # Handle canonical pose retry mode - adjust threshold instead of moving robot
        # Threshold adjustment happens in canonical match checking, so we just continue to normal processing
        
        # Wait for trajectory to complete before sending new one (same for both modes)
        if self.trajectory_in_progress:
            self.get_logger().debug("Trajectory already in progress, skipping...")
            return
        
        # Wait for end-effector pose if not received yet
        if not self.ee_pose_received or self.current_ee_pose is None:
            self.get_logger().warn("Waiting for end-effector pose...")
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
                self.get_logger().info(f"📌 {self.mode.upper()} mode: Current EE Z position ({current_ee_position[2]:.3f}m) is below 0.25m. Skipping step 1 and going straight to step 2.")
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
            self.get_logger().error("❌ No explicit mode specified. Must provide one of: target_xyz/xyzw, grasp_id, or object detection. Exiting.")
            self.should_exit = True
            return
        
        # If in object detection mode but no pose received yet, wait
        if (self.object_name is not None and self.object_name != "" and 
            self.target_xyz is None and self.grasp_id is None and 
            self.latest_pose is None and 
            (self.mode != 'real' or self.last_known_object_position is None)):
            self.get_logger().debug("Waiting for object pose to be received...")
            return
        
        # Check if we have optional target position/orientation
        if self.target_xyz is not None and self.target_xyzw is not None:
            # Use provided target position and orientation
            object_position = np.array(self.target_xyz[:3])  # Take first 3 elements
            
            # Apply calibration offset to correct systematic detection bias (only for real mode)
            if self.mode == 'real':
                if not self.step1_completed:
                    # Step 1: Apply first calibration offsets
                    object_position[0] += self.calibration_offset_x  # Correct X offset
                    object_position[1] += self.calibration_offset_y  # Correct Y offset
                    object_position[2] += self.calibration_offset_z  # Correct Z offset
                else:
                    # Step 2: Apply ONLY fine offsets to X, Y, and Z (not first offsets)
                    object_position[0] += self.fine_offset_x  # Fine X offset only
                    object_position[1] += self.fine_offset_y  # Fine Y offset only
                    object_position[2] += self.fine_offset_z  # Fine Z offset only
            
            # Use provided target quaternion and apply fold symmetry matching
            provided_quat = np.array(self.target_xyzw)
            
            # Try to find canonical match with increasing thresholds
            canonical_quat, canonical_match, match_distance = self._try_canonical_match_with_threshold(
                provided_quat, self.object_name
            )
            
            # Extract yaw from canonical quaternion if match found, otherwise from provided quaternion
            if canonical_match:
                object_yaw = self.quat_controller.extract_yaw_from_quaternion(canonical_quat)
                match_status = f"✅ Canonical match (threshold: {self.current_canonical_threshold:.3f}, distance: {match_distance:.4f})"
                yaw_source = "canonical quaternion (fold symmetry normalized)"
                # Exit retry mode if we were in it, but keep the threshold that worked
                if self.canonical_retry_mode:
                    self.get_logger().info(f"✅ Canonical match found with threshold {self.current_canonical_threshold:.3f}! Keeping threshold for future detections.")
                    self.canonical_retry_mode = False
                    self.canonical_retry_count = 0
                    # Keep current_canonical_threshold - don't reset it!
                    self.best_canonical_match = None
                    self.best_canonical_distance = float('inf')
            else:
                # No canonical match - trigger retry mode or continue retrying
                if not self.canonical_retry_mode:
                    self.get_logger().warn("⚠️ No canonical match found. Starting threshold adjustment...")
                    self.canonical_retry_mode = True
                    self.canonical_retry_count = 0
                    self.current_canonical_threshold = self.canonical_threshold_initial
                    self.best_canonical_match = None
                    self.best_canonical_distance = float('inf')
                    return  # Exit early to trigger retry
                
                # Already in retry mode - increment threshold and try again
                self.canonical_retry_count += 1
                if self.current_canonical_threshold < self.canonical_threshold_max:
                    self.current_canonical_threshold = min(
                        self.current_canonical_threshold + self.canonical_threshold_increment,
                        self.canonical_threshold_max
                    )
                    self.get_logger().info(f"🔄 Retry {self.canonical_retry_count}/{self.max_canonical_retries}: Adjusting threshold to {self.current_canonical_threshold:.3f}...")
                    return  # Try again with new threshold
                else:
                    # Max threshold reached - use best match found or fallback
                    if self.best_canonical_match is not None:
                        self.get_logger().warn(f"⚠️ Using best match found (distance: {self.best_canonical_distance:.4f}) as canonical pose.")
                        canonical_quat = self.best_canonical_match
                        object_yaw = self.quat_controller.extract_yaw_from_quaternion(canonical_quat)
                        match_status = f"⚠️ Using best canonical match (distance: {self.best_canonical_distance:.4f})"
                        yaw_source = "best canonical quaternion found"
                        self.canonical_retry_mode = False
                        self.canonical_retry_count = 0
                        # Keep the threshold that gave us the best match (don't reset to initial)
                        self.best_canonical_match = None
                        self.best_canonical_distance = float('inf')
                    else:
                        self.get_logger().error(f"❌ Max threshold ({self.canonical_threshold_max:.3f}) reached. Using provided quaternion as fallback.")
                        self.canonical_retry_mode = False
                        object_yaw = self.quat_controller.extract_yaw_from_quaternion(provided_quat)
                        match_status = "⚠️ No canonical match (using provided - max threshold exceeded)"
                        yaw_source = "provided quaternion"
                        # Reset threshold only when we completely fail
                        self.current_canonical_threshold = self.canonical_threshold_initial
                        self.best_canonical_match = None
                        self.best_canonical_distance = float('inf')
            
            target_quaternion = self.quat_controller.face_down_quaternion(object_yaw)
            
            self.get_logger().info(f"🎯 Using provided target position: {object_position}")
            self.get_logger().info(f"🎯 Provided quaternion: q=[{provided_quat[0]:.6f}, {provided_quat[1]:.6f}, "
                                 f"{provided_quat[2]:.6f}, {provided_quat[3]:.6f}]")
            if canonical_match:
                self.get_logger().info(f"🎯 Provided quaternion (canonical match): q=[{canonical_quat[0]:.6f}, {canonical_quat[1]:.6f}, "
                                     f"{canonical_quat[2]:.6f}, {canonical_quat[3]:.6f}] - {match_status}")
            else:
                self.get_logger().info(f"🎯 {match_status}")
            self.get_logger().info(f"🎯 Target gripper quaternion: q=[{target_quaternion[0]:.6f}, {target_quaternion[1]:.6f}, "
                                 f"{target_quaternion[2]:.6f}, {target_quaternion[3]:.6f}] (yaw: {object_yaw:.1f}° extracted from {yaw_source})")
        elif self.grasp_id is not None:
            # Grasp point mode: must have selected_grasp_point, exit if not available
            if self.selected_grasp_point is None:
                self.get_logger().error(f"❌ Grasp point {self.grasp_id} not found. Cannot proceed in grasp point mode. Exiting.")
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
                self.get_logger().warn(f"⚠️ Using stale grasp point {self.grasp_id} (topic is empty): "
                                      f"x={self.selected_grasp_point.pose.position.x:.6f}, "
                                      f"y={self.selected_grasp_point.pose.position.y:.6f}, "
                                      f"z={self.selected_grasp_point.pose.position.z:.6f}")
            else:
                self.get_logger().info(f"📍 Using grasp point {self.grasp_id} from message: "
                                      f"x={self.selected_grasp_point.pose.position.x:.6f}, "
                                      f"y={self.selected_grasp_point.pose.position.y:.6f}, "
                                      f"z={self.selected_grasp_point.pose.position.z:.6f}")
            
            # Apply calibration offset to correct systematic detection bias (only for real mode)
            if self.mode == 'real':
                if not self.step1_completed:
                    # Step 1: Apply first calibration offsets
                    grasp_point_position[0] += self.calibration_offset_x  # Correct X offset
                    grasp_point_position[1] += self.calibration_offset_y  # Correct Y offset
                    grasp_point_position[2] += self.calibration_offset_z  # Correct Z offset
                else:
                    # Step 2: Apply ONLY fine offsets to X, Y, and Z (not first offsets)
                    # Use regular fine offsets (same for both stale and fresh grasp points)
                    grasp_point_position[0] += self.fine_offset_x  # Fine X offset only
                    grasp_point_position[1] += self.fine_offset_y  # Fine Y offset only
                    grasp_point_position[2] += self.fine_offset_z  # Fine Z offset only
            
            # Set object position to grasp point position for distance calculation
            object_position = grasp_point_position
            
            # Store last known good position for recovery (real mode)
            if self.mode == 'real':
                self.last_known_object_position = object_position.copy()
                # Reset tracking lost count since we have a detection
                if self.tracking_lost_count > 0:
                    self.get_logger().info(f"✅ Tracking recovered after {self.tracking_lost_count} lost frames")
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
                self.get_logger().error(f"❌ Grasp point {self.grasp_id} does not have valid orientation. Cannot proceed. Exiting.")
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
                
                # Extract yaw from canonical quaternion if match found, otherwise from detected quaternion
                if canonical_match:
                    grasp_point_yaw = self.quat_controller.extract_yaw_from_quaternion(canonical_quat)
                    match_status = f"✅ Canonical match (threshold: {self.current_canonical_threshold:.3f}, distance: {match_distance:.4f})"
                    yaw_source = "canonical quaternion (fold symmetry normalized)"
                    # Exit retry mode if we were in it, but keep the threshold that worked
                    if self.canonical_retry_mode:
                        self.get_logger().info(f"✅ Canonical match found with threshold {self.current_canonical_threshold:.3f}! Keeping threshold for future detections.")
                        self.canonical_retry_mode = False
                        self.canonical_retry_count = 0
                        # Keep current_canonical_threshold - don't reset it!
                        self.best_canonical_match = None
                        self.best_canonical_distance = float('inf')
                else:
                    # No canonical match - trigger retry mode or continue retrying
                    if not self.canonical_retry_mode:
                        self.get_logger().warn(f"⚠️ No canonical match found with current threshold {self.current_canonical_threshold:.3f}. Starting threshold adjustment...")
                        self.canonical_retry_mode = True
                        self.canonical_retry_count = 0
                        # Start from current threshold (which might be higher than initial if we had a previous match)
                        # Only reset to initial if we're starting fresh
                        if self.current_canonical_threshold == self.canonical_threshold_initial:
                            # Already at initial, start incrementing
                            pass
                        else:
                            # We had a higher threshold that worked before, but now it doesn't work
                            # Try incrementing from current threshold
                            self.get_logger().info(f"📊 Previous threshold {self.current_canonical_threshold:.3f} no longer works, incrementing...")
                        self.best_canonical_match = None
                        self.best_canonical_distance = float('inf')
                        return  # Exit early to trigger retry
                    
                    # Already in retry mode - increment threshold and try again
                    self.canonical_retry_count += 1
                    if self.current_canonical_threshold < self.canonical_threshold_max:
                        self.current_canonical_threshold = min(
                            self.current_canonical_threshold + self.canonical_threshold_increment,
                            self.canonical_threshold_max
                        )
                        self.get_logger().info(f"🔄 Retry {self.canonical_retry_count}/{self.max_canonical_retries}: Adjusting threshold to {self.current_canonical_threshold:.3f}...")
                        return  # Try again with new threshold
                    else:
                        # Max threshold reached - use best match found or fallback
                        if self.best_canonical_match is not None:
                            self.get_logger().warn(f"⚠️ Using best match found (distance: {self.best_canonical_distance:.4f}) as canonical pose.")
                            canonical_quat = self.best_canonical_match
                            grasp_point_yaw = self.quat_controller.extract_yaw_from_quaternion(canonical_quat)
                            match_status = f"⚠️ Using best canonical match (distance: {self.best_canonical_distance:.4f})"
                            yaw_source = "best canonical quaternion found"
                            self.canonical_retry_mode = False
                            self.canonical_retry_count = 0
                            # Keep the threshold that gave us the best match (don't reset to initial)
                            self.best_canonical_match = None
                            self.best_canonical_distance = float('inf')
                        else:
                            self.get_logger().error(f"❌ Max threshold ({self.canonical_threshold_max:.3f}) reached. Using grasp point quaternion as fallback.")
                            self.canonical_retry_mode = False
                            grasp_point_yaw = self.quat_controller.extract_yaw_from_quaternion(grasp_point_quat)
                            match_status = "⚠️ No canonical match (using grasp point - max threshold exceeded)"
                            yaw_source = "detected quaternion"
                            # Reset threshold only when we completely fail
                            self.current_canonical_threshold = self.canonical_threshold_initial
                            self.best_canonical_match = None
                            self.best_canonical_distance = float('inf')
                
                # Create face-down quaternion with grasp point yaw (QUATERNION-BASED, no gimbal lock)
                target_quaternion = self.quat_controller.face_down_quaternion(grasp_point_yaw)
                
                self.get_logger().info(f"🎯 Using grasp point {self.grasp_id} position: {grasp_point_position}")
                self.get_logger().info(f"🎯 Grasp point quaternion (detected): q=[{grasp_point_quat[0]:.6f}, {grasp_point_quat[1]:.6f}, "
                                     f"{grasp_point_quat[2]:.6f}, {grasp_point_quat[3]:.6f}]")
                if canonical_match:
                    self.get_logger().info(f"🎯 Grasp point quaternion (canonical match): q=[{canonical_quat[0]:.6f}, {canonical_quat[1]:.6f}, "
                                         f"{canonical_quat[2]:.6f}, {canonical_quat[3]:.6f}] - {match_status}")
                else:
                    self.get_logger().info(f"🎯 {match_status}")
                self.get_logger().info(f"🎯 Gripper orientation (quaternion-based, no gimbal lock):\n"
                                     f"   q=[{target_quaternion[0]:.6f}, {target_quaternion[1]:.6f}, "
                                     f"{target_quaternion[2]:.6f}, {target_quaternion[3]:.6f}]\n"
                                     f"   Aligned with grasp point yaw: {grasp_point_yaw:.1f}° (extracted from {yaw_source})")
        elif self.latest_pose is not None:
            # Use detected object pose
            # Check if we were using stale data in step 2 (using last_known_object_position)
            if self.step1_completed and self.using_stale_data_step2 and self.trajectory_in_progress:
                self.get_logger().info(f"🔄 Fresh object pose data received during step 2! Cancelling current trajectory and recomputing...")
                # Cancel current trajectory
                if self.current_goal_handle is not None:
                    try:
                        self.cancelled_for_fresh_data = True  # Mark that we're cancelling for fresh data
                        cancel_future = self.current_goal_handle.cancel_goal_async()
                        self.get_logger().info("Cancelling current trajectory...")
                    except Exception as e:
                        self.get_logger().warn(f"Could not cancel trajectory: {e}")
                        self.cancelled_for_fresh_data = False
                # Reset trajectory flag so timer callback can recompute step 2
                self.trajectory_in_progress = False
                self.using_stale_data_step2 = False
            
            # Reset tracking lost count since we have a detection
            was_tracking_lost = False
            if self.mode == 'real':
                was_tracking_lost = self.tracking_lost_count > 0 or self.waiting_at_last_known
                if was_tracking_lost:
                    self.get_logger().info(f"✅ Tracking recovered after {self.tracking_lost_count} lost frames")
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
                        self.get_logger().info(f"🔄 Initializing Z smoothing with last known Z: {self.smoothed_object_z:.3f}m")
                else:
                    self.tracking_lost_count = 0
            
            # Calculate time delta for Kalman filter
            current_time = self.get_clock().now().nanoseconds / 1e9
            if self.last_update_time is not None:
                dt = current_time - self.last_update_time
            else:
                dt = 1.0  # Default time step
            self.last_update_time = current_time
            
            # Update Kalman filter with new measurement
            filtered_pose, velocity = self.kalman_filter.update(self.latest_pose, dt)
            
            if filtered_pose is None:
                return
                
            # Extract filtered position (Kalman filter still used for position smoothing)
            object_position = np.array(filtered_pose[:3])
            
            # Extract quaternion directly from latest_pose (bypass Kalman filter for orientation)
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
            
            # Extract yaw from canonical quaternion if match found, otherwise from detected quaternion
            if canonical_match:
                object_yaw = self.quat_controller.extract_yaw_from_quaternion(canonical_quat)
                yaw_source = "canonical quaternion (fold symmetry normalized)"
                # Exit retry mode if we were in it, but keep the threshold that worked
                if self.canonical_retry_mode:
                    self.get_logger().info(f"✅ Canonical match found with threshold {self.current_canonical_threshold:.3f}! Keeping threshold for future detections.")
                    self.canonical_retry_mode = False
                    self.canonical_retry_count = 0
                    # Keep current_canonical_threshold - don't reset it!
                    self.best_canonical_match = None
                    self.best_canonical_distance = float('inf')
            else:
                # No canonical match - trigger retry mode or continue retrying
                if not self.canonical_retry_mode:
                    self.get_logger().warn(f"⚠️ No canonical match found with current threshold {self.current_canonical_threshold:.3f}. Starting threshold adjustment...")
                    self.canonical_retry_mode = True
                    self.canonical_retry_count = 0
                    # Start from current threshold (which might be higher than initial if we had a previous match)
                    # Only reset to initial if we're starting fresh
                    if self.current_canonical_threshold == self.canonical_threshold_initial:
                        # Already at initial, start incrementing
                        pass
                    else:
                        # We had a higher threshold that worked before, but now it doesn't work
                        # Try incrementing from current threshold
                        self.get_logger().info(f"📊 Previous threshold {self.current_canonical_threshold:.3f} no longer works, incrementing...")
                    self.best_canonical_match = None
                    self.best_canonical_distance = float('inf')
                    return  # Exit early to trigger retry
                
                # Already in retry mode - increment threshold and try again
                self.canonical_retry_count += 1
                if self.current_canonical_threshold < self.canonical_threshold_max:
                    self.current_canonical_threshold = min(
                        self.current_canonical_threshold + self.canonical_threshold_increment,
                        self.canonical_threshold_max
                    )
                    self.get_logger().info(f"🔄 Retry {self.canonical_retry_count}/{self.max_canonical_retries}: Adjusting threshold to {self.current_canonical_threshold:.3f}...")
                    return  # Try again with new threshold
                else:
                    # Max threshold reached - use best match found or fallback
                    if self.best_canonical_match is not None:
                        self.get_logger().warn(f"⚠️ Using best match found (distance: {self.best_canonical_distance:.4f}) as canonical pose.")
                        canonical_quat = self.best_canonical_match
                        object_yaw = self.quat_controller.extract_yaw_from_quaternion(canonical_quat)
                        yaw_source = "best canonical quaternion found"
                        self.canonical_retry_mode = False
                        self.canonical_retry_count = 0
                        # Keep the threshold that gave us the best match (don't reset to initial)
                        self.best_canonical_match = None
                        self.best_canonical_distance = float('inf')
                    else:
                        self.get_logger().error(f"❌ Max threshold ({self.canonical_threshold_max:.3f}) reached. Using detected quaternion as fallback.")
                        self.canonical_retry_mode = False
                        object_yaw = self.quat_controller.extract_yaw_from_quaternion(detected_object_quat)
                        yaw_source = "detected quaternion"
                        # Reset threshold only when we completely fail
                        self.current_canonical_threshold = self.canonical_threshold_initial
                        self.best_canonical_match = None
                        self.best_canonical_distance = float('inf')
            
            # Apply calibration offset to correct systematic detection bias (only for real mode)
            if self.mode == 'real':
                if not self.step1_completed:
                    # Step 1: Apply first calibration offsets
                    object_position[0] += self.calibration_offset_x  # Correct X offset
                    object_position[1] += self.calibration_offset_y  # Correct Y offset
                    object_position[2] += self.calibration_offset_z  # Correct Z offset
                else:
                    # Step 2: Apply ONLY fine offsets to X, Y, and Z (not first offsets)
                    object_position[0] += self.fine_offset_x  # Fine X offset only
                    object_position[1] += self.fine_offset_y  # Fine Y offset only
                    object_position[2] += self.fine_offset_z  # Fine Z offset only
            
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
                        self.get_logger().info(f"🔄 Smoothing Z after recovery: detected={detected_z:.3f}m, smoothed={self.smoothed_object_z:.3f}m "
                                              f"({self.recovery_z_update_count}/{self.recovery_z_smoothing_steps})")
                    else:
                        # Done smoothing, use detected Z directly
                        self.smoothed_object_z = None
                        self.get_logger().info(f"✅ Z smoothing complete, using detected Z: {detected_z:.3f}m")
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
            match_status = "✅ Canonical match" if canonical_match else "⚠️ No canonical match (using detected)"
            
            self.get_logger().info(f"🎯 Detected object at ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
            self.get_logger().info(f"🎯 Object quaternion (detected): q=[{detected_object_quat[0]:.6f}, {detected_object_quat[1]:.6f}, "
                                 f"{detected_object_quat[2]:.6f}, {detected_object_quat[3]:.6f}]")
            if canonical_match:
                self.get_logger().info(f"🎯 Object quaternion (canonical match): q=[{canonical_quat[0]:.6f}, {canonical_quat[1]:.6f}, "
                                     f"{canonical_quat[2]:.6f}, {canonical_quat[3]:.6f}] - {match_status}")
            else:
                self.get_logger().info(f"🎯 {match_status}")
            self.get_logger().info(f"🎯 EE orientation (quaternion-based, no gimbal lock):\n"
                                 f"   q=[{target_quaternion[0]:.6f}, {target_quaternion[1]:.6f}, "
                                 f"{target_quaternion[2]:.6f}, {target_quaternion[3]:.6f}]\n"
                                 f"   Aligned with object yaw: {object_yaw:.1f}° (extracted from {yaw_source})")
        else:
            # No target provided and no object detected
            if self.mode == 'real' and self.last_known_object_position is not None:
                # Real mode: handle tracking loss
                self.tracking_lost_count += 1
                self.get_logger().warn(f"⚠️ Tracking lost! (consecutive misses: {self.tracking_lost_count}/{self.max_tracking_lost})")
                
                if self.tracking_lost_count >= self.max_tracking_lost:
                    # Move to last known location and wait
                    if not self.waiting_at_last_known:
                        self.waiting_at_last_known = True
                        self.recovery_mode = True
                        self.get_logger().warn(f"🔄 Moving to last known location and waiting for tracking recovery...")
                    
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
                        self.get_logger().info(f"📍 Moving to last known position: ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
                        # Continue to calculate and send trajectory (will be sent once)
                    else:
                        # Already sent trajectory to last known location, just wait
                        self.get_logger().info("⏸️ Waiting at last known location for tracking recovery...")
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
                        
                        self.get_logger().warn(f"⚠️ Using last known position (miss {self.tracking_lost_count}/{self.max_tracking_lost})")
                    else:
                        self.get_logger().error("❌ No target position provided, no object detected, and no last known position. Cannot proceed. Exiting.")
                        self.should_exit = True
                        return
            else:
                # No target provided and no object detected
                # If in object detection mode (object_name provided), wait for pose
                if self.object_name is not None and self.object_name != "":
                    self.get_logger().debug("Waiting for object pose to be received...")
                    return
                # Otherwise, no explicit mode specified: exit
                self.get_logger().error("❌ No explicit mode specified (no target_xyz/xyzw, no grasp_id) and no object detected. Cannot proceed. Exiting.")
                self.should_exit = True
                return
        
        # Verify that we have a valid target position and orientation (safety check)
        if 'object_position' not in locals() or 'target_quaternion' not in locals():
            self.get_logger().error("❌ Failed to determine target position or orientation. Cannot proceed. Exiting.")
            self.should_exit = True
            return
        
        # Calculate direction vector from object to current end-effector
        direction_vector = current_ee_position - object_position
        current_distance = np.linalg.norm(direction_vector)
        
        self.get_logger().info(f"📏 Current distance between object and EE: {current_distance*100:.2f} cm")
        self.get_logger().info(f"📍 Current EE position: ({current_ee_position[0]:.3f}, {current_ee_position[1]:.3f}, {current_ee_position[2]:.3f})")
        self.get_logger().info(f"📍 Object position: ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
        
        # Calculate target end-effector position using simple vertical offsets
        # Since gripper is always face-down (pitch=180°), all offsets are vertical in world Z-axis
        
        if self.height is not None:
            # If height is explicitly specified, use that exact height (ignore offset)
            target_ee_position = np.array([
                object_position[0],
                object_position[1],
                self.height
            ])
            self.get_logger().info(f"📏 Using specified height={self.height:.3f}m (offset ignored)")
        else:
            # Calculate positions: Object -> Gripper Center -> TCP
            # 1. Subtract offset from object Z to get gripper center (offset point) - gripper center is BELOW object
            # 2. Add TCP offset to gripper center to get TCP position (vertical offset, since gripper is face-down)
            # Formula: gripper_center = object - object_to_gripper_center_offset
            #          TCP = gripper_center + tcp_to_gripper_center_offset = object - object_to_gripper_center_offset + tcp_to_gripper_center_offset
            
            # First calculate gripper center (offset point) - gripper center is BELOW the object
            offset_point = object_position.copy()
            offset_point[2] -= self.object_to_gripper_center_offset  # Gripper center is offset below object
            
            # Then calculate TCP position from offset point (simple vertical offset since gripper is face-down)
            target_ee_position = offset_point.copy()
            target_ee_position[2] += self.tcp_to_gripper_center_offset  # TCP is above gripper center
            
            self.get_logger().info(f"📏 Offset point (gripper center): ({offset_point[0]:.3f}, {offset_point[1]:.3f}, {offset_point[2]:.3f})")
            self.get_logger().info(f"📏 TCP to gripper center offset: {self.tcp_to_gripper_center_offset*100:.1f}cm (vertical, world Z-axis)")
            self.get_logger().info(f"🎯 Target TCP position: ({target_ee_position[0]:.3f}, {target_ee_position[1]:.3f}, {target_ee_position[2]:.3f})")
        
        # For sim mode step 1: add z offset of 0.05
        if self.mode == 'sim' and not self.step1_completed:
            target_ee_position[2] += 0.05  # Add 0.05m z offset for step 1
            self.get_logger().info(f"📌 Sim mode Step 1: Adding 0.05m z offset. Target Z: {target_ee_position[2]:.3f}m")
        
        # Verify the target distance (from TCP to offset point)
        if self.height is None:
            # Simple vertical offset verification (no quaternion needed)
            calculated_distance = abs(target_ee_position[2] - offset_point[2])
            self.get_logger().info(f"✅ Calculated distance from TCP to offset point: {calculated_distance*100:.2f} cm")
            
            # Verify the offset matches expected value
            # In Step 1 (sim mode), account for the 0.05m z offset that was added
            expected_distance = self.tcp_to_gripper_center_offset
            if self.mode == 'sim' and not self.step1_completed:
                expected_distance += 0.05  # Account for Step 1 offset
            
            distance_error = abs(calculated_distance - expected_distance)
            if distance_error > 0.001:  # 1mm tolerance
                self.get_logger().warn(f"⚠️ TCP offset verification error: {distance_error*1000:.2f}mm (expected {expected_distance*100:.1f}cm, got {calculated_distance*100:.1f}cm)")
            else:
                self.get_logger().info(f"✅ TCP offset verification passed: {calculated_distance*100:.1f}cm (error = {distance_error*1000:.3f}mm)")
        else:
            # For explicit height, just show distance to object
            calculated_distance = np.linalg.norm(target_ee_position - object_position)
            self.get_logger().info(f"✅ Calculated target distance: {calculated_distance*100:.2f} cm")
        
        self.get_logger().info(f"🎯 Final target EE position: ({target_ee_position[0]:.3f}, {target_ee_position[1]:.3f}, {target_ee_position[2]:.3f})")
        
        # Check convergence in step 2: if robot is close enough to target, count stable readings and exit
        if self.step1_completed:
            distance_to_target = np.linalg.norm(current_ee_position - target_ee_position)
            self.get_logger().info(f"📏 Distance to target: {distance_to_target*100:.2f} cm (threshold: {self.convergence_distance_threshold*100:.2f} cm)")
            
            if distance_to_target <= self.convergence_distance_threshold:
                self.convergence_stable_count += 1
                self.get_logger().info(f"✅ Within convergence threshold! Stable count: {self.convergence_stable_count}/{self.convergence_stable_threshold}")
                
                if self.convergence_stable_count >= self.convergence_stable_threshold:
                    self.get_logger().info(f"✅ Converged! Robot is within {self.convergence_distance_threshold*100:.2f}cm of target for {self.convergence_stable_threshold} consecutive readings. Exiting step 2.")
                    self.movement_completed = True
                    self.should_exit = True
                    self._print_final_object_pose()
                    return  # Exit early, don't send trajectory
            else:
                # Reset stable count if we're not within threshold
                if self.convergence_stable_count > 0:
                    self.get_logger().info(f"⚠️ Moved outside convergence threshold. Resetting stable count.")
                    self.convergence_stable_count = 0
        
        # If waiting at last known location, mark that we've sent the trajectory
        if self.waiting_at_last_known and not self.last_known_target_sent:
            self.last_known_target_sent = True
        
        # Create target pose with calculated position (PURE QUATERNION, no RPY conversion)
        target_position = target_ee_position.tolist()
        
        # Keep quaternion representation throughout - NO RPY conversion
        # hover_over_grasp_quat extracts yaw directly from quaternion without RPY
        target_pose_quat = (target_position, target_quaternion)
        
        # Use specified movement duration (same for both modes)
        movement_duration = self.movement_duration
        
        self.get_logger().info(f"🎯 Final gripper orientation (PURE QUATERNION, no gimbal lock):")
        self.get_logger().info(f"   q=[{target_quaternion[0]:.6f}, {target_quaternion[1]:.6f}, "
                             f"{target_quaternion[2]:.6f}, {target_quaternion[3]:.6f}]")
        
        trajectory = hover_over_grasp_quat(target_pose_quat, target_ee_position[2], movement_duration)
        
        # For step 1: store Z position for step 2 (both sim and real modes)
        if not self.step1_completed:
            self.step1_z_position = target_ee_position[2]
            self.get_logger().info(f"📌 Step 1: Storing Z position {self.step1_z_position:.3f}m for step 2")
        
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
                self.get_logger().error("No trajectory found")
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
            self.get_logger().info(f"Sending trajectory ({self.mode} mode)...")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)
            
        except Exception as e:
            self.get_logger().error(f"❌ Trajectory execution error: {e}")
            self.trajectory_in_progress = False  # Clear flag on error
            self.movement_completed = True
            self.should_exit = True

    def goal_response(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            # Set exit flags if goal is rejected
            self.trajectory_in_progress = False
            self.movement_completed = True
            self.should_exit = True
            return

        self.get_logger().info("Trajectory goal accepted")
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
            # No symmetry data, return detected as-is
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
    
    def _print_final_object_pose(self):
        """Print the final object pose recorded before exit"""
        self.get_logger().info("=" * 80)
        self.get_logger().info("📊 FINAL OBJECT POSE RECORDED:")
        
        if self.final_object_pose is not None:
            # Print object detection pose (regardless of mode)
            pose = self.final_object_pose
            self.get_logger().info(f"   Object: {self.object_name}")
            self.get_logger().info(f"   Position: x={pose.pose.position.x:.6f}, y={pose.pose.position.y:.6f}, z={pose.pose.position.z:.6f}")
            self.get_logger().info(f"   Orientation (quaternion): x={pose.pose.orientation.x:.6f}, y={pose.pose.orientation.y:.6f}, "
                                 f"z={pose.pose.orientation.z:.6f}, w={pose.pose.orientation.w:.6f}")
        else:
            self.get_logger().warn("   ⚠️ No final object pose recorded (pose data not available)")
        
        self.get_logger().info("=" * 80)
    
    def goal_result(self, future):
        """Handle goal result (used for both sim and real modes)"""
        result = future.result()
        self.trajectory_in_progress = False  # Clear trajectory in progress flag
        self.current_goal_handle = None  # Clear goal handle
        
        # Check if this was a cancellation due to fresh data in step 2
        if result.status == 5:  # CANCELED
            self.get_logger().info(f"Trajectory cancelled (status 5). cancelled_for_fresh_data={self.cancelled_for_fresh_data}, step1_completed={self.step1_completed}")
            if self.cancelled_for_fresh_data:
                # This was cancelled because fresh data arrived - don't exit, let timer recompute
                self.get_logger().info("✅ Trajectory cancelled due to fresh data arrival. Recomputing step 2...")
                self.cancelled_for_fresh_data = False  # Reset flag
                return  # Don't exit, timer callback will recompute step 2
            elif self.step1_completed:
                # In step 2 and cancelled (but flag not set) - might be due to fresh data, don't exit
                # This handles race conditions where flag might not be set yet
                self.get_logger().info("✅ Trajectory cancelled during step 2. Assuming fresh data arrival. Recomputing step 2...")
                return  # Don't exit, timer callback will recompute step 2
            else:
                # Cancelled for other reasons (not in step 2)
                self.get_logger().error(f"Trajectory cancelled with status: {result.status}")
                self.movement_completed = True
                self.should_exit = True
                return
        
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info("✅ Trajectory completed successfully")
            
            # Check if step 1 completed, trigger step 2 (both sim and real modes)
            if not self.step1_completed:
                self.step1_completed = True
                # Switch to faster timer period for step 2 to get more frequent pose updates
                self.update_timer.cancel()
                self.update_timer = self.create_timer(self.timer_period_step2, self.timer_callback)
                if self.mode == 'real':
                    self.get_logger().info(f"📌 Step 1 completed. Starting step 2: fixing Z at {self.step1_z_position:.3f}m, applying fine offsets (X: {self.fine_offset_x:.3f}m, Y: {self.fine_offset_y:.3f}m)")
                    self.get_logger().info(f"📌 Step 2: Monitoring latest object pose (timer period: {self.timer_period_step2}s)")
                else:  # sim mode
                    self.get_logger().info(f"📌 Sim mode Step 1 completed. Starting step 2: moving to final position (removing 0.05m z offset)")
                # Don't exit - let timer callback trigger step 2 with latest pose
                return
        else:
            self.get_logger().error(f"Trajectory failed with status: {result.status}")
        
        # Print final object pose before exiting
        self._print_final_object_pose()
        
        # Set exit flags after trajectory completes (or if step 2 completed)
        self.movement_completed = True
        self.should_exit = True
        self.get_logger().info("✅ Direct movement completed. Exiting.")
    


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Direct Object Movement Node')
    parser.add_argument('--topic', type=str, default=None, 
                       help='Topic name for object poses subscription (default: /objects_poses_sim for sim mode, /objects_poses_real for real mode)')
    parser.add_argument('--object-name', type=str, default="fork_orange_scaled70",
                       help='Name of the object to move to (e.g., blue_dot_0, red_dot_0)')
    parser.add_argument('--height', type=float, default=None,
                       help='Hover height in meters (if not specified, will use 5.5cm offset from object/grasp point)')
    parser.add_argument('--movement-duration', type=float, default=5.0,
                       help='Duration for the movement in seconds (default: 5.0)')
    parser.add_argument('--target-xyz', type=float, nargs=3, default=None,
                       help='Optional target position [x, y, z] in meters')
    parser.add_argument('--target-xyzw', type=float, nargs=4, default=None,
                       help='Optional target orientation [x, y, z, w] quaternion')
    parser.add_argument('--grasp-points-topic', type=str, default="/grasp_points",
                       help='Topic name for grasp points subscription')
    parser.add_argument('--grasp-id', type=int, default=None,
                       help='Specific grasp point ID to use (if provided, will use grasp point instead of object center)')
    parser.add_argument('--offset', type=float, default=None,
                       help='Distance offset from object/grasp point in meters (default: 0.123m = 12.3cm)')
    parser.add_argument('--mode', type=str, default=None, choices=['sim', 'real'], required=True,
                       help='Mode: "sim" for simulation (uses /objects_poses_sim with TFMessage), "real" for real robot (uses /objects_poses_real with TFMessage). REQUIRED - no default.')
    
    # Parse arguments from sys.argv if args is None
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    rclpy.init(args=None)
    node = DirectObjectMove(topic_name=args.topic, object_name=args.object_name, 
                      height=args.height, movement_duration=args.movement_duration,
                      target_xyz=args.target_xyz, target_xyzw=args.target_xyzw,
                      grasp_points_topic=args.grasp_points_topic, grasp_id=args.grasp_id,
                      offset=args.offset, mode=args.mode)
    
    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Direct movement stopped by user")
    except Exception as e:
        node.get_logger().error(f"Direct movement error: {e}")
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            # Ignore shutdown errors
            pass

if __name__ == '__main__':
    main()
