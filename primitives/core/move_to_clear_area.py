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
from scipy.spatial.transform import Rotation as R

from primitives.shared.config import SAFE_HEIGHT, GRIPPER_CENTER_TOOL_OFFSET
import argparse

from primitives.shared.velocity_profiles import single_point, compute_duration
from primitives.shared.config import TABLE_HEIGHT, TABLE_COLLISION_MARGIN_SIDEWAYS, TABLE_COLLISION_MARGIN_FACEDOWN
from primitives.shared.collision import compute_all_joint_positions, check_collision_with_table, segment_distance, check_self_collision, check_trajectory_collision
from primitives.shared.ik import dh_params, forward_kinematics

# Watchdog (seconds).  Execution counts from goal acceptance, never from planning.
PLANNING_TIMEOUT_S = 180.0
GOAL_ACCEPT_TIMEOUT_S = 10.0
EXECUTION_TIMEOUT_MIN_S = 30.0          # the former whole-run timeout, now execution only
EXECUTION_TIMEOUT_MARGIN_S = 10.0
CANCEL_CONFIRM_TIMEOUT_S = 5.0
GOAL_STATUS_SUCCEEDED, GOAL_STATUS_CANCELED, GOAL_STATUS_ABORTED = 4, 5, 6   # action_msgs/GoalStatus
from primitives.shared.robot_contract import (
    ARM_ACTION, BARE_ARM_JOINTS, DROIDSTACK_SIM, JOINT_STATES_TOPIC,
    arm_positions, public_arm_joints)

GRASP_DISTANCE_THRESHOLD = 0.06  # 60mm — same as translate_object.py

class MoveToClearArea(Node):
    def __init__(self, mode='move', object_name=None, robot_mode='real', target_xy=None,
                 held_quat=None, grasp_id=None):
        super().__init__('move_to_clear_area')
        self.mode = mode  # 'move' or 'fix_orientation'
        self.object_name = object_name  # If set, verify grasp before moving
        # Held-part route check (primitives/shared/held_part_collision.py): the chained held
        # orientation (world xyzw) and grasp id place the part's CAD on the flange.
        self.held_quat = held_quat
        self.grasp_id = grasp_id
        self.robot_mode = robot_mode  # 'sim' or 'real'
        self.joint_names = BARE_ARM_JOINTS

        # Action client for trajectory control
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            ARM_ACTION
        )

        # Target gripper center position — converted to flange for IK via: flange = gc - R @ offset
        from primitives.shared.config import CLEAR_AREA_XY
        self.target_gripper_center_position = [CLEAR_AREA_XY[0], CLEAR_AREA_XY[1], SAFE_HEIGHT]

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
        # Execution watchdog: the clock starts when the controller ACCEPTS the goal (planning is not
        # execution); on timeout the goal is cancelled and the cancel confirmed, never left running.
        self.trajectory_duration_s = None
        self.goal_sent_at = None
        self.goal_accepted_at = None
        self._goal_handle = None
        self._send_goal_future = None
        self._get_result_future = None
        self._cancel_on_accept = False
        self._cancelling = False
        self._cancel_result_status = None

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

        self.ee_pose_sub = (None if DROIDSTACK_SIM else self.create_subscription(
            PoseStamped, '/tcp_pose_broadcaster/pose', self.ee_pose_callback, qos_profile))

        # Subscriber for joint states to get current joint angles (use as IK seed)
        self.joint_state_sub = self.create_subscription(
            JointState,
            JOINT_STATES_TOPIC,
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

        # Subscribe to object poses for grasp verification (real or sim) or sim placement collision check
        if self.object_name or self.robot_mode == 'sim':
            from tf2_msgs.msg import TFMessage
            poses_topic = '/objects_poses_real' if self.robot_mode == 'real' else '/objects_poses_sim'
            self.object_pose_sub = self.create_subscription(
                TFMessage, poses_topic, self._object_pose_cb, 10
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
        ordered_positions = arm_positions(msg)
        if ordered_positions is not None:
            self.current_joint_angles = np.array(ordered_positions)
            self.joint_angles_received = True
            if DROIDSTACK_SIM:
                transform = forward_kinematics(dh_params, ordered_positions)
                self.ee_position = transform[:3, 3].copy()
                self.ee_quat = R.from_matrix(transform[:3, :3]).as_quat()
                self.ee_pose_received = True

    def _held_part_model(self, start_joints):
        """The part this carry holds, for the route check; None for an empty gripper.

        sim: measured from the simulator's ground-truth part pose and the measured joints.
        real: the cell's chained held orientation + the CAD grasp point at the executed grasp
        depth; with no orientation passed, the orientation-free CAD envelope (fail closed)."""
        if not self.object_name:
            return None
        from primitives.shared import held_part_collision as hpc
        from primitives.shared.config import held_tool_offset_for
        T_f = forward_kinematics(dh_params, start_joints)
        if self.robot_mode == 'sim' and self.object_name in self.object_poses:
            tf = self.object_poses[self.object_name].transform
            T_part = np.eye(4)
            T_part[:3, :3] = R.from_quat([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]).as_matrix()
            T_part[:3, 3] = [tf.translation.x, tf.translation.y, tf.translation.z]
            return hpc.measured_held_part(self.object_name, np.linalg.inv(T_f) @ T_part)
        grasp_id = self.grasp_id
        if grasp_id is None:
            from primitives.shared.config import default_grasp_id
            grasp_id = default_grasp_id(self.object_name)
        offset = float(held_tool_offset_for(self.object_name)[2])
        if self.held_quat is not None:
            return hpc.chained_held_part(self.object_name, grasp_id, self.held_quat, T_f[:3, 3],
                                         R.from_matrix(T_f[:3, :3]).as_quat(), offset)
        return hpc.envelope_held_part(self.object_name, grasp_id, offset)

    def _object_pose_cb(self, msg):
        """Callback for sim object poses."""
        for transform in msg.transforms:
            self.object_poses[transform.child_frame_id] = transform

    def _verify_grasp(self):
        """Verify the robot is holding the object (gripper center within threshold of object).

        In real mode the object's AprilTag is occluded by the gripper once grasped,
        so /objects_poses_real drops it. The upstream caller (translate_object)
        already runs a width-only verify; skip the pose-lookup here for real mode.
        """
        from scipy.spatial.transform import Rotation as Rot

        if self.robot_mode == 'real':
            self.get_logger().info(
                f"Skipping pose-based grasp verification for '{self.object_name}' (real mode — tag occluded by gripper)"
            )
            return True

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
                    (-np.pi, np.pi),     # shoulder_pan: full range
                    (-np.pi, 0),         # shoulder_lift: negative only (reaching forward/down)
                    (0, np.pi),          # elbow: positive only (elbow down)
                    (-np.pi, np.pi),     # wrist_1: full range
                    (-np.pi, np.pi),     # wrist_2: full range
                    (-2*np.pi, 2*np.pi)  # wrist_3: extended range
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
                (-np.pi, np.pi),     # shoulder_pan: full range
                (-np.pi, 0),         # shoulder_lift: negative only (reaching forward/down)
                (0, np.pi),          # elbow: positive only (elbow down)
                (-np.pi, np.pi),     # wrist_1: full range
                (-np.pi, np.pi),     # wrist_2: full range
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
            traj.joint_names = public_arm_joints()
            traj.points = trajectory_points

            goal.trajectory = traj
            if not DROIDSTACK_SIM:
                goal.goal_time_tolerance = Duration(sec=1)

            num_waypoints = len(trajectory_points)
            self.get_logger().info(f"Sending hover trajectory ({num_waypoints} waypoint{'s' if num_waypoints > 1 else ''})")
            self._dispatch(goal)
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
                    (-np.pi, np.pi),     # shoulder_pan: full range
                    (-np.pi, 0),         # shoulder_lift: negative only (reaching forward/down)
                    (0, np.pi),          # elbow: positive only (elbow down)
                    (-np.pi, np.pi),     # wrist_1: full range
                    (-np.pi, np.pi),     # wrist_2: full range
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
                # the same pose with wrist_1 / wrist_2 a full turn nearer where they are now (within +-2 pi): 10-25 run
                # 109b the rotate left wrist_2 at -4.195 rad, the only candidate had +0.126, and the 4.3 rad swing
                # self-collided (direct and via home) - the -2 pi twin (-6.158) is 1.96 rad away. wrist_3 untouched
                # (cable: +-4 rad). Tried first; the original candidates stay as they were.
                twins = []
                for cost_, cj_, yaw_ in candidate_solutions:
                    tj_ = np.array(cj_, dtype=float).copy()
                    for j_ in (3, 4):
                        opts = [tj_[j_] + k_ * 2 * np.pi for k_ in (-1, 0, 1) if abs(tj_[j_] + k_ * 2 * np.pi) <= 2 * np.pi]
                        tj_[j_] = min(opts, key=lambda v_: abs(v_ - start_joints[j_]))
                    if np.max(np.abs(tj_ - np.array(cj_, dtype=float))) > 1e-6:
                        twins.append((cost_, tj_, yaw_))
                if twins:
                    self.get_logger().info(f"{len(twins)} wrist twin(s) nearer the current wrist angles")
                candidate_solutions = twins + candidate_solutions
                self.get_logger().info(f"Found {len(candidate_solutions)} candidate IK solutions")

                # Check each candidate for collision-free trajectory (no joint wrapping)
                collision_free_trajectories = []  # List of (travel_distance, target_joints, ik_cost, yaw)
                from primitives.shared.held_part_collision import held_part_route_refusal
                from primitives.shared.camera_up import camera_route_refusal
                held_part = self._held_part_model(start_joints)
                held_refusals = []
                if held_part is not None:
                    self.get_logger().info(
                        f"Held-part route check: {held_part.name} ({held_part.source} model)")

                for cost, candidate_joints, yaw in candidate_solutions:
                    target_joints = np.array(candidate_joints).copy()
                    joints_str = ' '.join(f'{v:.6f}' for v in target_joints)
                    self.get_logger().info(f"Candidate IK (cost={cost:.4f}): {joints_str}")

                    if check_collision_with_table(target_joints):
                        continue
                    if check_self_collision(target_joints):
                        continue

                    traj_margin = TABLE_COLLISION_MARGIN_FACEDOWN if is_face_down else TABLE_COLLISION_MARGIN_SIDEWAYS
                    if not check_trajectory_collision(start_joints, target_joints, z_threshold=TABLE_HEIGHT - traj_margin, num_samples=20, logger=self.get_logger()):
                        # Hard rule (lead, 2026-11-19): the place-down stance is carried to the table, so its camera
                        # side may not face the table; the mount/D415 keeps its clearance along the route.
                        refusal = (camera_route_refusal(start_joints, target_joints, near_table_end=True)
                                   or held_part_route_refusal(start_joints, target_joints, held_part))
                        if refusal:
                            held_refusals.append(refusal)
                            self.get_logger().warn(f"Held-part route refused: {refusal}")
                            continue
                        travel_distance = np.sum(np.abs(target_joints - start_joints))
                        collision_free_trajectories.append((travel_distance, target_joints.copy(), cost, yaw,
                                                            [target_joints.copy()]))

                via = None
                if not collision_free_trajectories:
                    # the direct joint path from the staging side swings the arm across itself (10-24 run 80: u_orange's
                    # only candidate self-collided at alpha 0.30; the same set-down passed 10-23 from other starts) -
                    # through the home joints, when both legs are clear
                    from primitives.shared.config import HOME_JOINTS
                    home = np.array(HOME_JOINTS, dtype=float)
                    margin = TABLE_COLLISION_MARGIN_FACEDOWN if is_face_down else TABLE_COLLISION_MARGIN_SIDEWAYS
                    for cost, candidate_joints, yaw in candidate_solutions:
                        tj = np.array(candidate_joints).copy()
                        if check_collision_with_table(tj) or check_self_collision(tj):
                            continue
                        if (not check_trajectory_collision(start_joints, home, z_threshold=TABLE_HEIGHT - margin, num_samples=20, logger=self.get_logger())
                                and not check_trajectory_collision(home, tj, z_threshold=TABLE_HEIGHT - margin, num_samples=20, logger=self.get_logger())):
                            refusal = (camera_route_refusal(start_joints, home)
                                       or camera_route_refusal(home, tj, near_table_end=True)
                                       or held_part_route_refusal(start_joints, home, held_part)
                                       or held_part_route_refusal(home, tj, held_part))
                            if refusal:
                                held_refusals.append(f"via home: {refusal}")
                                self.get_logger().warn(f"Held-part route via home refused: {refusal}")
                                continue
                            collision_free_trajectories.append((np.sum(np.abs(home - start_joints)) + np.sum(np.abs(tj - home)),
                                                                tj, cost, yaw, [home.copy(), tj.copy()]))
                    if collision_free_trajectories:
                        via = home
                        self.get_logger().info("Direct path collides - going through the home joints")
                if not collision_free_trajectories:
                    # Split by joint group (route_split; lead approval 2026-11-19): every segment a straight joint
                    # line from where the last one ended, checked by the arm, camera and held-part rules alike.
                    from primitives.shared.route_split import admissible_route
                    from primitives.shared.config import HOME_JOINTS
                    margin = TABLE_COLLISION_MARGIN_FACEDOWN if is_face_down else TABLE_COLLISION_MARGIN_SIDEWAYS

                    def segment_refusal(a, b, is_last):
                        if check_trajectory_collision(a, b, z_threshold=TABLE_HEIGHT - margin, num_samples=20):
                            return "ARM_TABLE_OR_SELF_COLLISION"
                        return (camera_route_refusal(a, b, near_table_end=is_last)
                                or held_part_route_refusal(a, b, held_part))

                    for cost, candidate_joints, yaw in candidate_solutions:
                        tj = np.array(candidate_joints).copy()
                        if check_collision_with_table(tj) or check_self_collision(tj):
                            continue
                        route, why = admissible_route(start_joints, tj, segment_refusal,
                                                      vias=(np.array(HOME_JOINTS, dtype=float),))
                        if route is None:
                            held_refusals.append(f"split: {'; '.join(sorted(set(why))[:2])}")
                            continue
                        collision_free_trajectories.append(
                            (float(sum(np.sum(np.abs(b - a)) for a, b in zip(route[:-1], route[1:]))),
                             tj, cost, yaw, [np.asarray(w, float) for w in route[1:]]))
                    if collision_free_trajectories:
                        self.get_logger().info("Direct and via-home routes refused - joint-group split route")
                if not collision_free_trajectories:
                    self.error_message = "Motion planning failed: all candidate trajectories to the target were rejected due to collision"
                    if held_refusals:
                        self.error_message = ("HELD_PART_ROUTE_REFUSED: every route to the clear area was refused for "
                                              f"the held part or the camera ({'; '.join(held_refusals[:3])})")
                    self.get_logger().error(self.error_message)
                    self.operation_success = False
                    self.operation_complete = True
                    rclpy.shutdown()
                    return

                # Sort by travel distance (shortest path first)
                collision_free_trajectories.sort(key=lambda x: x[0])

                shortest_travel, target_joints, best_cost, best_yaw, route_legs = collision_free_trajectories[0]
                self.get_logger().info(f"Found {len(collision_free_trajectories)} collision-free trajectories")
                self.get_logger().info(f"Using shortest path: travel={np.degrees(shortest_travel):.1f}deg, IK cost={best_cost:.4f}, yaw={best_yaw:.1f}°")

                legs = route_legs
                trajectory_points = []
                t, prev = 0.0, start_joints
                for leg in legs:
                    joint_dist = float(np.max(np.abs(leg - prev)))
                    leg_s = compute_duration(joint_distance=joint_dist, profile='s_curve')
                    self.get_logger().info(f"Duration: {leg_s:.2f}s (joint={joint_dist:.2f}rad)")
                    t += leg_s
                    # each waypoint at rest (velocity 0): the UR controller interpolates each leg smoothly
                    positions, velocities, _ = single_point(leg, t)[0]
                    trajectory_points.append(JointTrajectoryPoint(
                        positions=positions,
                        velocities=velocities,
                        time_from_start=Duration(sec=int(t),
                                                nanosec=int((t % 1) * 1e9))
                    ))
                    prev = leg

                # Create and send trajectory
                goal = FollowJointTrajectory.Goal()
                traj = JointTrajectory()
                traj.joint_names = public_arm_joints()
                traj.points = trajectory_points

                goal.trajectory = traj
                if not DROIDSTACK_SIM:
                    goal.goal_time_tolerance = Duration(sec=1)

                self.get_logger().info("Sending trajectory")
                self._dispatch(goal, feedback_callback=self.feedback_callback)

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

    def _dispatch(self, goal, feedback_callback=None):
        """Send the trajectory goal; the watchdog's execution clock starts at acceptance."""
        last = goal.trajectory.points[-1].time_from_start
        self.trajectory_duration_s = float(last.sec) + float(last.nanosec) * 1e-9
        if feedback_callback is None:
            self._send_goal_future = self.action_client.send_goal_async(goal)
        else:
            self._send_goal_future = self.action_client.send_goal_async(goal, feedback_callback=feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response)
        self.goal_sent_at = time.monotonic()
        self.trajectory_sent = True

    def watchdog_refusal(self, now, started_at):
        """A typed timeout, or None.  Planning (no goal sent) is bounded by PLANNING_TIMEOUT_S and moves
        nothing; a sent goal must be accepted within GOAL_ACCEPT_TIMEOUT_S; an accepted goal must finish
        within its own duration plus EXECUTION_TIMEOUT_MARGIN_S, and at least EXECUTION_TIMEOUT_MIN_S."""
        if self.goal_accepted_at is not None:
            limit = max(EXECUTION_TIMEOUT_MIN_S, (self.trajectory_duration_s or 0.0) + EXECUTION_TIMEOUT_MARGIN_S)
            if now - self.goal_accepted_at > limit:
                return (f"EXECUTION_TIMEOUT: the trajectory ({self.trajectory_duration_s:.1f}s) did not finish "
                        f"{limit:.0f}s after the controller accepted it")
            return None
        if self.goal_sent_at is not None:
            if now - self.goal_sent_at > GOAL_ACCEPT_TIMEOUT_S:
                return f"GOAL_ACCEPT_TIMEOUT: no goal response {GOAL_ACCEPT_TIMEOUT_S:.0f}s after sending"
            return None
        if now - started_at > PLANNING_TIMEOUT_S:
            return f"PLANNING_TIMEOUT: no trajectory planned within {PLANNING_TIMEOUT_S:.0f}s (nothing was sent)"
        return None

    def _spin_until(self, future, timeout_s):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        return future.done()

    def cancel_active_goal(self, wait_s=CANCEL_CONFIRM_TIMEOUT_S):
        """Cancel the accepted goal and wait for the controller to confirm it stopped.  Returns None when
        the goal ended (canceled, or finished meanwhile), else a typed CANCEL_UNCONFIRMED.  A goal still
        awaiting its response is cancelled as soon as it is accepted."""
        if self._goal_handle is None:
            if self.goal_sent_at is None:
                return None
            self._cancel_on_accept = True
            if not self._spin_until(self._send_goal_future, wait_s):
                return "CANCEL_UNCONFIRMED: the goal response never came; the arm may be moving - stop it"
            if self._goal_handle is None:
                return None                                   # rejected: nothing runs
        self._cancelling = True
        cancel_future = self._goal_handle.cancel_goal_async()
        if not self._spin_until(cancel_future, wait_s):
            return "CANCEL_UNCONFIRMED: no cancel response; the arm may be moving - stop it"
        if self._cancel_result_status is None and self._get_result_future is not None:
            self._spin_until(self._get_result_future, wait_s)
        if self._cancel_result_status in (GOAL_STATUS_CANCELED, GOAL_STATUS_SUCCEEDED, GOAL_STATUS_ABORTED):
            return None
        return (f"CANCEL_UNCONFIRMED: goal status {self._cancel_result_status} after the cancel; "
                "the arm may be moving - stop it")

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

        self._goal_handle = goal_handle
        self.goal_accepted_at = time.monotonic()
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result)
        if self._cancel_on_accept:
            self.get_logger().error("Goal accepted after the watchdog fired: cancelling it")

    def goal_result(self, future):
        """Handle goal result"""
        if self._cancelling:
            # the watchdog cancelled this goal: record how it ended; main() reports and shuts down
            try:
                self._cancel_result_status = future.result().status
            except Exception:
                self._cancel_result_status = None
            return
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
    parser.add_argument('--held-quat', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'), default=None,
                       help="Held part's chained world orientation (xyzw) for the held-part route check")
    parser.add_argument('--grasp-id', type=int, default=None,
                       help="Grasp id the part is held by (default: the assembly's grasp id for it)")

    args = parser.parse_args()

    # Determine mode
    if args.fix_orientation:
        mode = 'fix_orientation'
    else:
        mode = 'move'  # Default mode

    rclpy.init()
    node = MoveToClearArea(mode=mode, object_name=args.object_name,
                           robot_mode=args.mode, target_xy=args.target_xy,
                           held_quat=args.held_quat, grasp_id=args.grasp_id)
    try:
        # Spin until operation is complete with timeout
        import time
        start_time = time.monotonic()

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

            timeout = node.watchdog_refusal(time.monotonic(), start_time)
            if timeout:
                unconfirmed = node.cancel_active_goal()
                node.error_message = timeout + (f"; {unconfirmed}" if unconfirmed else "; goal cancelled")
                node.get_logger().error(node.error_message)
                node.operation_success = False
                node.operation_complete = True
                break

    except KeyboardInterrupt:
        node.error_message = "Operation cancelled by user (Ctrl+C)"
        unconfirmed = node.cancel_active_goal() if node.trajectory_sent and not node.operation_complete else None
        if unconfirmed:
            node.error_message += f"; {unconfirmed}"
        node.get_logger().warn(node.error_message)
        node.operation_success = False
        node.operation_complete = True
    except Exception as e:
        node.error_message = f"Error during spin: {e}"
        try:
            unconfirmed = node.cancel_active_goal() if node.trajectory_sent and not node.operation_complete else None
        except Exception as cancel_error:
            unconfirmed = f"CANCEL_UNCONFIRMED: {cancel_error}; the arm may be moving - stop it"
        if unconfirmed:
            node.error_message += f"; {unconfirmed}"
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

