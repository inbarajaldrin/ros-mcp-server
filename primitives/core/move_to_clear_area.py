import sys
import os
import json
import subprocess
import time

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
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import numpy as np

from primitives.shared.config import SAFE_HEIGHT, GRIPPER_CENTER_TOOL_OFFSET
import argparse

from primitives.shared.velocity_profiles import single_point, compute_duration
from primitives.shared.config import TABLE_HEIGHT, TABLE_COLLISION_MARGIN_SIDEWAYS, TABLE_COLLISION_MARGIN_FACEDOWN
from primitives.shared.collision import compute_all_joint_positions, check_collision_with_table, segment_distance, check_self_collision, check_trajectory_collision

GRASP_DISTANCE_THRESHOLD = 0.06  # 60mm — same as translate_object.py

class MoveToClearArea(Node):
    def __init__(self, mode='move', object_name=None, robot_mode='real', target_xy=None):
        super().__init__('move_to_clear_area')
        self.mode = mode  # 'move' or 'fix_orientation'
        self.object_name = object_name  # If set, verify grasp before moving
        self.robot_mode = robot_mode  # 'sim' or 'real'
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # Action client for trajectory control
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        # Target gripper center position — converted to flange for IK via: flange = gc - R @ offset
        self.target_gripper_center_position = [-0.320, -0.5, SAFE_HEIGHT]

        # Apply target_xy override (before sim collision avoidance logic)
        if target_xy is not None:
            self.target_gripper_center_position[0] = target_xy[0]
            self.target_gripper_center_position[1] = target_xy[1]

        # EE pose data storage
        self.ee_pose_received = False
        self.ee_position = None
        self.ee_quat = None

        # Current joint angles storage
        self.current_joint_angles = None
        self.joint_angles_received = False

        # Object pose data (for grasp verification)
        self.object_poses = {}

        # Robot safety monitoring (detect protective stop during execution)
        self.safety_mode = None  # Last received safety mode
        self.robot_program_running = None  # Last received program running state
        self.trajectory_sent = False  # Only check health after trajectory is dispatched

        # Success/failure tracking
        self.operation_success = False
        self.operation_complete = False
        self.error_message = None  # Track error for JSON output

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

        # Monitor robot safety mode and program running state to detect protective stop
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        try:
            from ur_dashboard_msgs.msg import SafetyMode
            self.safety_mode_sub = self.create_subscription(
                SafetyMode,
                '/io_and_status_controller/safety_mode',
                self._safety_mode_cb,
                latched_qos
            )
        except ImportError:
            self.safety_mode_sub = None
        self.program_running_sub = self.create_subscription(
            Bool,
            '/io_and_status_controller/robot_program_running',
            self._program_running_cb,
            latched_qos
        )

        # Subscribe to sim object poses for grasp verification or sim placement collision check
        if self.object_name or self.robot_mode == 'sim':
            from tf2_msgs.msg import TFMessage
            self.object_pose_sub = self.create_subscription(
                TFMessage, '/objects_poses_sim', self._object_pose_cb, 10
            )

        self.action_client.wait_for_server()

        # Log mode
        self.get_logger().info(f"Using {self.mode.upper()} mode")

        # Verify grasp if object_name provided
        if self.object_name and not self._verify_grasp():
            return

        # Execute movement
        self.move_to_clear_space()
    
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

    def _object_pose_cb(self, msg):
        """Callback for sim object poses."""
        for transform in msg.transforms:
            self.object_poses[transform.child_frame_id] = transform

    def _verify_grasp(self):
        """Verify the robot is holding the object (gripper center within threshold of object)."""
        from scipy.spatial.transform import Rotation as Rot

        self.get_logger().info(f"Verifying grasp on '{self.object_name}'...")

        # Wait for EE pose and object pose
        timeout_count = 0
        while rclpy.ok() and timeout_count < 50:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout_count += 1
            if self.ee_pose_received and self.object_name in self.object_poses:
                break

        if not self.ee_pose_received:
            self.error_message = "Grasp verification failed: no EE pose received"
            self.get_logger().error(self.error_message)
            self.operation_success = False
            self.operation_complete = True
            rclpy.shutdown()
            return False

        if self.object_name not in self.object_poses:
            self.error_message = f"Grasp verification failed: object '{self.object_name}' not found in sim poses"
            self.get_logger().error(self.error_message)
            self.operation_success = False
            self.operation_complete = True
            rclpy.shutdown()
            return False

        # Compute gripper center position
        ee_rot = Rot.from_quat(self.ee_quat).as_matrix()
        gripper_center = self.ee_position + ee_rot @ GRIPPER_CENTER_TOOL_OFFSET

        # Get object position
        obj_t = self.object_poses[self.object_name].transform.translation
        object_pos = np.array([obj_t.x, obj_t.y, obj_t.z])

        grasp_distance = np.linalg.norm(object_pos - gripper_center)
        if grasp_distance > GRASP_DISTANCE_THRESHOLD:
            self.error_message = (
                f"Grasp check failed: {self.object_name} is {grasp_distance * 1000:.1f}mm "
                f"from gripper center (threshold: {GRASP_DISTANCE_THRESHOLD * 1000:.0f}mm)."
            )
            self.get_logger().error(self.error_message)
            self.operation_success = False
            self.operation_complete = True
            rclpy.shutdown()
            return False

        self.get_logger().info(
            f"Grasp verified: {self.object_name} is {grasp_distance * 1000:.1f}mm from gripper center"
        )
        return True

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
        
        # Wait for both pose and joint angles to arrive (no timeout - wait indefinitely)
        while rclpy.ok() and (not self.ee_pose_received or not self.joint_angles_received):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if not self.ee_pose_received:
            self.get_logger().error("EE pose message not received")
            return None
        
        if not self.joint_angles_received:
            self.get_logger().error("Joint angles message not received")
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

    def move_to_clear_space(self):
        """Move to clear space position while maintaining current end-effector orientation"""
        pose_data = self.read_current_ee_pose()
        
        if pose_data is None:
            self.error_message = "Could not read current end-effector pose"
            self.get_logger().error(self.error_message)
            self.operation_success = False
            self.operation_complete = True
            rclpy.shutdown()
            return
            
        current_pos = pose_data['position']
        current_quat = pose_data['orientation']

        # In sim mode, check if hardcoded clear space is occupied; if so, step +Y until clear
        # Skip collision avoidance in fix_orientation mode — goal is to reorient in place, not find a clear spot
        if self.robot_mode == 'sim' and self.mode != 'fix_orientation':
            # Wait for sim object poses to arrive
            self.get_logger().info("Waiting for sim object poses...")
            wait_start = time.time()
            while not self.object_poses and (time.time() - wait_start) < 10.0:
                rclpy.spin_once(self, timeout_sec=0.1)
            if self.object_poses:
                self.get_logger().info(f"Received {len(self.object_poses)} object poses ({time.time() - wait_start:.1f}s)")
            else:
                self.get_logger().warning("No object poses received after 10s — collision check will be skipped")

            # Auto-detect held object if not provided — find object closest to gripper center
            if not self.object_name and self.object_poses:
                from scipy.spatial.transform import Rotation as Rot
                ee_rot = Rot.from_quat(np.array(current_quat)).as_matrix()
                gripper_center = np.array(current_pos) + ee_rot @ GRIPPER_CENTER_TOOL_OFFSET
                best_name, best_dist = None, GRASP_DISTANCE_THRESHOLD
                for name, tf in self.object_poses.items():
                    obj_pos = np.array([tf.transform.translation.x,
                                        tf.transform.translation.y,
                                        tf.transform.translation.z])
                    d = np.linalg.norm(obj_pos - gripper_center)
                    if d < best_dist:
                        best_name, best_dist = name, d
                if best_name:
                    self.object_name = best_name
                    self.get_logger().info(
                        f"Auto-detected held object: '{best_name}' ({best_dist*1000:.1f}mm from gripper)")

            occupy_threshold = 0.085  # m — if any object XY is within this radius, spot is occupied
            step = 0.085  # m — Y increment per attempt
            x_target = self.target_gripper_center_position[0]
            y_candidate = self.target_gripper_center_position[1]

            for attempt in range(20):
                blocker_y = None
                for name, tf in self.object_poses.items():
                    # Skip the held object — its sim pose follows the gripper
                    if self.object_name and name == self.object_name:
                        continue
                    ox = tf.transform.translation.x
                    oy = tf.transform.translation.y
                    dist_xy = np.sqrt((x_target - ox) ** 2 + (y_candidate - oy) ** 2)
                    if dist_xy < occupy_threshold:
                        blocker_y = oy
                        break
                if blocker_y is None:
                    break
                # Step from the blocking object's actual Y, not from the candidate
                y_candidate = blocker_y + step

            self.target_gripper_center_position = [x_target, y_candidate, SAFE_HEIGHT]
            self.get_logger().info(
                f"Sim mode: target=[{x_target:.3f}, {y_candidate:.3f}, {SAFE_HEIGHT:.3f}] "
                f"({len(self.object_poses)} objects checked)"
            )

        # Convert quaternion directly to rotation matrix to avoid precision loss from RPY conversion
        from scipy.spatial.transform import Rotation as Rot
        from scipy.optimize import minimize
        
        # Determine target orientation based on mode
        if self.mode == 'fix_orientation':
            from scipy.spatial.transform import Rotation as Rot
            from primitives.shared.ik import IKSolverConfig, IKSolver

            if self.current_joint_angles is None:
                self.error_message = "Current joint angles not available for hover IK"
                self.get_logger().error(self.error_message)
                self.operation_success = False
                self.operation_complete = True
                rclpy.shutdown()
                return

            trajectory_points = []
            current_height = current_pos[2]

            # Waypoint 1: Lift to intermediate height (if needed)
            if current_height < 0.15:
                self.get_logger().info(f"Computing lift to 0.15m from {current_height:.3f}m...")
                target_rot_matrix = Rot.from_quat(current_quat).as_matrix()
                lift_target = current_pos.copy()
                lift_target[2] = 0.15

                lift_pose = np.eye(4)
                lift_pose[:3, 3] = lift_target
                lift_pose[:3, :3] = target_rot_matrix

                lift_bounds = [
                    (-np.pi, np.pi),     # shoulder_pan
                    (-np.pi, np.pi),     # shoulder_lift
                    (-np.pi, np.pi),     # elbow
                    (-np.pi, np.pi),     # wrist_1
                    (-np.pi, np.pi),     # wrist_2
                    (-2*np.pi, 2*np.pi)  # wrist_3
                ]
                lift_solver = IKSolver(IKSolverConfig(joint_bounds=lift_bounds))
                lift_joints = lift_solver.solve(
                    seeds=[self.current_joint_angles.copy()],
                    target_pose=lift_pose,
                    perturbations=5,
                    dx=0.001,
                )

                if lift_joints is None:
                    self.error_message = "Motion planning failed: intermediate lift position is unreachable"
                    self.get_logger().error(self.error_message)
                    self.operation_success = False
                    self.operation_complete = True
                    rclpy.shutdown()
                    return

                self.get_logger().info(f"Lift IK solved: cost={lift_solver._best_result.cost:.6f}")
                trajectory_points.append(JointTrajectoryPoint(
                    positions=[float(x) for x in lift_joints],
                    velocities=[0.0] * 6,
                    time_from_start=Duration(sec=3)
                ))
            else:
                self.get_logger().info(f"EE already at height {current_height:.3f}m (>= 0.15m), skipping lift")

            # Waypoint 2: Hover position (face-down at clear area)
            self.get_logger().info("Computing hover position IK...")
            hover_rot = Rot.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()

            # Convert gripper center XY to flange XY; Z is already flange height
            offset_3d = hover_rot @ GRIPPER_CENTER_TOOL_OFFSET
            hover_flange = np.array(self.target_gripper_center_position)
            hover_flange[0] -= offset_3d[0]
            hover_flange[1] -= offset_3d[1]

            hover_pose = np.eye(4)
            hover_pose[:3, 3] = hover_flange
            hover_pose[:3, :3] = hover_rot

            hover_bounds = [
                (-np.pi, np.pi),     # shoulder_pan
                (-np.pi, 0),         # shoulder_lift: negative only (reaching forward/down)
                (0, np.pi),          # elbow: positive only (elbow down)
                (-np.pi, np.pi),     # wrist_1
                (-np.pi, 0),         # wrist_2: negative only (wrist-down)
                (-2*np.pi, 2*np.pi)  # wrist_3: extended range
            ]
            hover_solver = IKSolver(IKSolverConfig(joint_bounds=hover_bounds))
            hover_joints = hover_solver.solve(
                seeds=[
                    np.radians([85, -80, 90, -90, -90, 0]),
                    np.radians([90, -90, 90, -90, -90, 0]),
                    np.radians([0, -90, 90, -90, -90, 0]),
                ],
                target_pose=hover_pose,
                perturbations=5,
                dx=0.001,
            )

            if hover_joints is None:
                self.error_message = "Motion planning failed: hover position above the target is unreachable"
                self.get_logger().error(self.error_message)
                self.operation_success = False
                self.operation_complete = True
                rclpy.shutdown()
                return

            self.get_logger().info(f"Hover IK solved: cost={hover_solver._best_result.cost:.6f}")
            lift_duration = 3 if trajectory_points else 0
            trajectory_points.append(JointTrajectoryPoint(
                positions=[float(x) for x in hover_joints],
                velocities=[0.0] * 6,
                time_from_start=Duration(sec=lift_duration + 5)
            ))

            # Send single merged trajectory
            goal = FollowJointTrajectory.Goal()
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            traj.points = trajectory_points

            goal.trajectory = traj
            goal.goal_time_tolerance = Duration(sec=1)

            num_waypoints = len(trajectory_points)
            self.get_logger().info(f"Sending hover trajectory ({num_waypoints} waypoint{'s' if num_waypoints > 1 else ''})")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)
            self.trajectory_sent = True
            return  # Exit early for hover mode
        else:
            # Move mode: Check EE face direction and fix if needed
            current_rotation = Rot.from_quat(current_quat)
            current_rot_matrix = current_rotation.as_matrix()
            tool_z = current_rot_matrix[:, 2]  # Tool Z-axis in world frame

            # Snap to cardinal face direction while preserving roll.
            # 1. Determine target face (face_right for horizontal, face_down for downward)
            # 2. Choose the roll variant that best matches current roll
            current_rpy = current_rotation.as_euler('xyz', degrees=True)
            current_roll = current_rpy[0]

            face_threshold = 0.707  # cos(45°)
            is_face_down = tool_z[2] < -face_threshold

            if is_face_down:
                R_face = Rot.from_euler('yz', [180, 0], degrees=True)
                face_name = 'face_down'
            else:
                R_face = Rot.from_euler('yz', [90, -90], degrees=True)
                face_name = 'face_right'

            # Find the roll variant that best preserves current roll
            # Skip variants with pitch near ±90° (gimbal lock)
            best_roll_variant = 0
            best_roll_diff = float('inf')
            best_target_rot = None
            for roll in [0, 90, 180, 270]:
                R_roll = Rot.from_euler('z', roll, degrees=True)
                R_candidate = R_face * R_roll
                candidate_rpy = R_candidate.as_euler('xyz', degrees=True)
                candidate_roll = candidate_rpy[0]
                candidate_pitch = candidate_rpy[1]

                # Skip gimbal lock orientations (pitch near ±90°)
                if abs(abs(candidate_pitch) - 90) < 5:
                    continue

                # Compare rolls (handle wrap-around)
                roll_diff = abs(candidate_roll - current_roll)
                if roll_diff > 180:
                    roll_diff = 360 - roll_diff
                if roll_diff < best_roll_diff:
                    best_roll_diff = roll_diff
                    best_roll_variant = roll
                    best_target_rot = R_candidate

            # Fallback if all variants are gimbal lock (shouldn't happen for face_right)
            if best_target_rot is None:
                self.get_logger().warn("All roll variants have gimbal lock, using roll90")
                best_roll_variant = 90
                R_roll = Rot.from_euler('z', 90, degrees=True)
                best_target_rot = R_face * R_roll
                best_roll_diff = 0

            target_rot_matrix = best_target_rot.as_matrix()
            target_rpy = best_target_rot.as_euler('xyz', degrees=True)

            self.get_logger().info(
                f"Snapping to {face_name}_roll{best_roll_variant} "
                f"(current roll: {current_roll:.1f}° → target roll: {target_rpy[0]:.1f}°, diff: {best_roll_diff:.1f}°)"
            )

            # Only one target orientation now (the best roll-preserving one)
            target_orientations = [(target_rpy[2], target_rot_matrix)]

            # Convert gripper center XY to flange XY; Z is already flange height
            offset_3d = target_rot_matrix @ GRIPPER_CENTER_TOOL_OFFSET
            flange_position = np.array(self.target_gripper_center_position)
            flange_position[0] -= offset_3d[0]
            flange_position[1] -= offset_3d[1]

            # Compute inverse kinematics for target pose
            # Use quaternion directly converted to rotation matrix for more accurate orientation preservation
            try:
                # Create target pose with quaternion-derived rotation matrix
                target_pose = np.eye(4)
                target_pose[:3, 3] = flange_position
                target_pose[:3, :3] = target_rot_matrix
                
                # Use quaternion-based IK directly - no RPY conversion at all!
                max_tries = 5  # Reduced from 10 for efficiency
                dx = 0.001

                # Primary seed: use current joint angles from joint state subscription
                if self.current_joint_angles is None:
                    self.error_message = "Current joint angles not available! Cannot compute IK."
                    self.get_logger().error(self.error_message)
                    self.operation_success = False
                    self.operation_complete = True
                    rclpy.shutdown()
                    return

                from primitives.shared.ik import IKSolverConfig, IKSolver

                q_guess = self.current_joint_angles.copy()
                start_joints = self.current_joint_angles.copy()

                def collision_checker(joint_angles):
                    return (check_collision_with_table(joint_angles)
                            or check_self_collision(joint_angles))

                joint_bounds_base = [
                    (-np.pi, np.pi),     # shoulder_pan
                    (-np.pi, 0),         # shoulder_lift: negative only (reaching forward/down)
                    (0, np.pi),          # elbow: positive only (elbow down)
                    (-np.pi, 0),         # wrist_1: negative only
                    (-np.pi, np.pi),     # wrist_2
                    (-2*np.pi, 2*np.pi)  # wrist_3: extended range
                ]

                # Cardinal wrist configurations differ by face direction:
                #   face_right: pan ≈ w2, w3 ≈ 0 (freeze w3, couple pan=w2)
                #   face_down:  w2 ≈ -π/2, lift+elbow+w1 ≈ -π/2 (freeze w2, seed kinematic constraint)
                w3_val = np.clip(q_guess[5], -np.pi + 0.05, np.pi - 0.05)
                w2_val = q_guess[4]
                joint_bounds_cardinal = list(joint_bounds_base)

                def make_seeds_free(joint_bounds):
                    seeds = [q_guess.copy()]
                    for sp in [q_guess[0], 0, np.pi/2, -np.pi/2, np.pi]:
                        s = q_guess.copy()
                        s[0] = sp
                        seeds.append(s)
                    return seeds

                if is_face_down:
                    # face_down cardinal: w2 ≈ -π/2, lift+elbow+w1 ≈ -π/2
                    joint_bounds_cardinal[4] = (-np.pi/2, -np.pi/2)  # freeze w2 at -π/2
                    seeds_cardinal = [q_guess.copy()]
                    # Generate seeds satisfying lift+elbow+w1 ≈ -π/2 constraint
                    for sp in [q_guess[0], 0, np.pi/2, -np.pi/2, np.pi]:
                        for lift in [-1.9, -1.5, -1.3, -1.0]:
                            for elbow in [1.2, 1.5, 1.8, 2.0]:
                                w1 = -np.pi/2 - lift - elbow  # satisfy constraint
                                if -np.pi <= w1 <= 0:  # within bounds
                                    s = q_guess.copy()
                                    s[0] = sp
                                    s[1] = lift
                                    s[2] = elbow
                                    s[3] = w1
                                    s[4] = -np.pi/2
                                    seeds_cardinal.append(s)
                else:
                    # face_right cardinal: pan ≈ w2, w3 ≈ 0
                    joint_bounds_cardinal[5] = (w3_val, w3_val)  # freeze wrist_3
                    seeds_cardinal = [q_guess.copy()]
                    for sp in [q_guess[0], 0, np.pi/2, -np.pi/2, np.pi]:
                        s = q_guess.copy()
                        s[0] = sp
                        s[4] = sp  # w2 = pan
                        seeds_cardinal.append(s)

                candidate_solutions = []

                # First pass: face-specific cardinal wrist constraints
                solver = IKSolver(IKSolverConfig(joint_bounds=joint_bounds_cardinal, cost_threshold=0.01))
                for yaw, rot_matrix in target_orientations:
                    off = rot_matrix @ GRIPPER_CENTER_TOOL_OFFSET
                    fp = np.array(self.target_gripper_center_position)
                    fp[0] -= off[0]; fp[1] -= off[1]
                    pose = np.eye(4)
                    pose[:3, 3] = fp
                    pose[:3, :3] = rot_matrix

                    joint_angles = solver.solve(
                        seeds=seeds_cardinal,
                        target_pose=pose,
                        collision_checker=collision_checker,
                        perturbations=max_tries,
                        dx=dx,
                    )
                    if joint_angles is not None:
                        candidate_solutions.append((solver._best_result.cost, joint_angles, yaw))

                # Fallback: free wrist constraints if cardinal solve failed
                if not candidate_solutions:
                    self.get_logger().info("Motion planning failed for cardinal wrist orientation, falling back to free wrist bounds")
                    solver = IKSolver(IKSolverConfig(joint_bounds=joint_bounds_base, cost_threshold=0.01))
                    seeds_free = make_seeds_free(joint_bounds_base)
                    for yaw, rot_matrix in target_orientations:
                        off = rot_matrix @ GRIPPER_CENTER_TOOL_OFFSET
                        fp = np.array(self.target_gripper_center_position)
                        fp[0] -= off[0]; fp[1] -= off[1]
                        pose = np.eye(4)
                        pose[:3, 3] = fp
                        pose[:3, :3] = rot_matrix

                        joint_angles = solver.solve(
                            seeds=seeds_free,
                            target_pose=pose,
                            collision_checker=collision_checker,
                            perturbations=max_tries,
                            dx=dx,
                        )
                        if joint_angles is not None:
                            candidate_solutions.append((solver._best_result.cost, joint_angles, yaw))

                if not candidate_solutions:
                    self.error_message = "Motion planning failed: all candidate paths to the clear area result in collision"
                    self.get_logger().error(self.error_message)
                    self.operation_success = False
                    self.operation_complete = True
                    rclpy.shutdown()
                    return

                # Sort by cost (best first)
                candidate_solutions.sort(key=lambda x: x[0])
                self.get_logger().info(f"Found {len(candidate_solutions)} candidate IK solutions")

                # Try each candidate solution with different joint wrapping options
                # Collect ALL collision-free trajectories, then pick shortest path
                num_waypoints = 10
                collision_free_trajectories = []  # List of (travel_distance, target_joints, ik_cost, yaw)

                for cost, candidate_joints, yaw in candidate_solutions:
                    target_joints = np.array(candidate_joints).copy()
                    joints_str = ' '.join(f'{v:.6f}' for v in target_joints)
                    self.get_logger().info(f"Candidate IK (cost={cost:.4f}): {joints_str}")

                    # Generate wrapping variants (±2π) for joints 1-5 only (skip wrist_3/joint 6)
                    wrapping_variants = [target_joints.copy()]
                    for joint_idx in range(5):  # joints 0-4 only
                        diff = target_joints[joint_idx] - start_joints[joint_idx]
                        new_variants = []
                        for variant in wrapping_variants:
                            new_variants.append(variant.copy())
                            if diff < -np.pi/2:
                                v = variant.copy()
                                v[joint_idx] += 2 * np.pi
                                new_variants.append(v)
                            if diff > np.pi/2:
                                v = variant.copy()
                                v[joint_idx] -= 2 * np.pi
                                new_variants.append(v)
                        wrapping_variants = new_variants

                    for variant in wrapping_variants:
                        if check_collision_with_table(variant):
                            continue
                        if check_self_collision(variant):
                            continue

                        traj_margin = TABLE_COLLISION_MARGIN_FACEDOWN if is_face_down else TABLE_COLLISION_MARGIN_SIDEWAYS
                        if not check_trajectory_collision(start_joints, variant, z_threshold=TABLE_HEIGHT - traj_margin, num_samples=20, logger=self.get_logger()):
                            travel_distance = np.sum(np.abs(variant - start_joints))
                            collision_free_trajectories.append((travel_distance, variant.copy(), cost, yaw))

                if not collision_free_trajectories:
                    self.error_message = "Motion planning failed: all candidate trajectories to the target were rejected due to collision"
                    self.get_logger().error(self.error_message)
                    self.operation_success = False
                    self.operation_complete = True
                    rclpy.shutdown()
                    return

                # Sort by travel distance (shortest path first)
                collision_free_trajectories.sort(key=lambda x: x[0])

                shortest_travel, target_joints, best_cost, best_yaw = collision_free_trajectories[0]
                self.get_logger().info(f"Found {len(collision_free_trajectories)} collision-free trajectories")
                self.get_logger().info(f"Using shortest path: travel={np.degrees(shortest_travel):.1f}deg, IK cost={best_cost:.4f}, yaw={best_yaw:.1f}°")

                joint_dist = float(np.max(np.abs(target_joints - start_joints)))
                total_duration = compute_duration(joint_distance=joint_dist, profile='s_curve')
                self.get_logger().info(f"Duration: {total_duration:.2f}s (joint={joint_dist:.2f}rad)")

                # Single endpoint trajectory — let the UR controller handle smooth acceleration/deceleration
                positions, velocities, t = single_point(target_joints, total_duration)[0]
                trajectory_points = [JointTrajectoryPoint(
                    positions=positions,
                    velocities=velocities,
                    time_from_start=Duration(sec=int(t),
                                            nanosec=int((t % 1) * 1e9))
                )]

                # Create and send trajectory
                goal = FollowJointTrajectory.Goal()
                traj = JointTrajectory()
                traj.joint_names = self.joint_names
                traj.points = trajectory_points

                goal.trajectory = traj
                goal.goal_time_tolerance = Duration(sec=1)

                self.get_logger().info("Trajectory sent and accepted")
                self._send_goal_future = self.action_client.send_goal_async(
                    goal,
                    feedback_callback=self.feedback_callback
                )
                self._send_goal_future.add_done_callback(self.goal_response)
                self.trajectory_sent = True

            except Exception as e:
                self.error_message = f"Failed to compute IK: {e}"
                self.get_logger().error(self.error_message)
                self.operation_success = False
                self.operation_complete = True
                rclpy.shutdown()

    def _safety_mode_cb(self, msg):
        self.safety_mode = msg.mode

    def _program_running_cb(self, msg):
        self.robot_program_running = msg.data

    SAFETY_MODE_NAMES = {
        1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY",
        5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP",
        8: "VIOLATION", 9: "FAULT",
    }

    def check_robot_health(self):
        """Check safety mode and program running state. Returns error string or None."""
        if self.safety_mode is not None and self.safety_mode != 1:  # Not NORMAL
            name = self.SAFETY_MODE_NAMES.get(self.safety_mode, f"UNKNOWN({self.safety_mode})")
            return (f"Robot entered {name} (safety_mode={self.safety_mode}) — "
                    "trajectory likely hit joint limits. "
                    "Fix via ursim_cli: unlock -> play")
        if self.robot_program_running is not None and not self.robot_program_running:
            return ("External control program stopped (robot_program_running=False). "
                    "Fix via ursim_cli: stop -> close_popup -> play")
        return None

    def feedback_callback(self, feedback_msg):
        """Handle trajectory execution feedback"""
        feedback = feedback_msg.feedback
        # Log progress - feedback contains actual vs desired positions
        if hasattr(feedback, 'error') and feedback.error:
            # Check for position errors that might indicate problems
            if hasattr(feedback.error, 'positions') and feedback.error.positions:
                max_error = max(abs(e) for e in feedback.error.positions)
                if max_error > 0.1:  # More than 0.1 rad error
                    self.get_logger().warn(f"Large trajectory tracking error: {np.degrees(max_error):.1f}deg")

    def goal_response(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.error_message = "External control program stopped or robot in protective stop"
            self.get_logger().error(self.error_message)
            self.operation_success = False
            self.operation_complete = True
            try:
                self.destroy_node()
            except:
                pass
            rclpy.shutdown()
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result)

    def goal_result(self, future):
        """Handle goal result"""
        try:
            result = future.result()
            if result.status == 4:  # SUCCEEDED
                self.get_logger().info("Movement completed successfully")
                self.operation_success = True
            else:
                try:
                    result_msg = result.result
                    error_code = result_msg.error_code
                    if error_code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
                        self.error_message = "PATH_TOLERANCE_VIOLATED: Velocity or acceleration limits exceeded. The required velocity to reach the target exceeds joint velocity limits."
                    elif error_code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED:
                        self.error_message = "GOAL_TOLERANCE_VIOLATED: Final position tolerance exceeded."
                    elif error_code == FollowJointTrajectory.Result.INVALID_GOAL:
                        self.error_message = "INVALID_GOAL: The trajectory goal is invalid."
                    elif error_code == FollowJointTrajectory.Result.INVALID_JOINTS:
                        self.error_message = "INVALID_JOINTS: Invalid joint names in trajectory."
                    elif error_code == FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP:
                        self.error_message = "OLD_HEADER_TIMESTAMP: Trajectory header timestamp is too old."
                    else:
                        status_names = {1: "ACCEPTED", 2: "EXECUTING", 3: "CANCELING", 4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}
                        status_name = status_names.get(result.status, f"UNKNOWN({result.status})")
                        self.error_message = f"Trajectory failed: status={status_name}, error_code={error_code}"
                except Exception as e:
                    status_names = {1: "ACCEPTED", 2: "EXECUTING", 3: "CANCELING", 4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}
                    status_name = status_names.get(result.status, f"UNKNOWN({result.status})")
                    self.error_message = f"Trajectory failed: status={status_name} ({e})"
                self.get_logger().error(self.error_message)
                self.operation_success = False
        except Exception as e:
            self.error_message = f"Goal result callback error: {e}"
            self.get_logger().error(self.error_message)
            self.operation_success = False
        self.operation_complete = True
        try:
            self.destroy_node()
        except:
            pass
        rclpy.shutdown()

    def output_result_json(self, movement_type="move_to_clear_area"):
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
    parser = argparse.ArgumentParser(description='Move to clear space position')
    parser.add_argument('--move', action='store_true',
                       help='Move to target position keeping current EE orientation (default)')
    parser.add_argument('--fix-orientation', action='store_true',
                       help='Move to target position with top-down (face-down) EE orientation')
    parser.add_argument('--target-xy', type=float, nargs=2, metavar=('X', 'Y'),
                       help='Override target XY position (gripper center coords)')
    parser.add_argument('--object-name', type=str, default=None,
                       help='Object name for grasp verification (checks gripper is holding object)')
    parser.add_argument('--mode', type=str, default='real', choices=['sim', 'real'],
                       help='Robot mode: sim varies placement Y to avoid stacking (default: real)')

    args = parser.parse_args()

    # Determine mode
    if args.fix_orientation:
        mode = 'fix_orientation'
    else:
        mode = 'move'  # Default mode

    rclpy.init()
    node = MoveToClearArea(mode=mode, object_name=args.object_name,
                           robot_mode=args.mode, target_xy=args.target_xy)
    try:
        # Spin until operation is complete with timeout
        import time
        start_time = time.time()
        timeout_sec = 30.0  # 30 second timeout for trajectory execution

        while rclpy.ok() and not node.operation_complete:
            rclpy.spin_once(node, timeout_sec=0.1)

            # Check for robot safety issues (protective stop, program stopped)
            health_err = node.trajectory_sent and node.check_robot_health()
            if health_err:
                node.error_message = health_err
                node.get_logger().error(node.error_message)
                node.operation_success = False
                node.operation_complete = True
                break

            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                node.error_message = f"Trajectory execution timed out after {timeout_sec:.0f}s - robot may have hit velocity limits or stalled"
                node.get_logger().error(node.error_message)
                node.operation_success = False
                node.operation_complete = True
                break

    except KeyboardInterrupt:
        node.error_message = "Operation cancelled by user (Ctrl+C)"
        node.get_logger().warn(node.error_message)
        node.operation_success = False
        node.operation_complete = True
    except Exception as e:
        node.error_message = f"Error during spin: {e}"
        node.get_logger().error(node.error_message)
        node.operation_success = False
        node.operation_complete = True
    finally:
        try:
            node.output_result_json()
            node.action_client.destroy()
            node.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass

    # Exit with appropriate code
    sys.exit(0 if node.operation_success else 1)

if __name__ == '__main__':
    main()

