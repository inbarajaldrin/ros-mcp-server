import sys
import os
import json
import subprocess

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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import argparse

from primitives.utils.ik_solver import ik_objective_quaternion, forward_kinematics, dh_params

class MoveToClearArea(Node):
    def __init__(self, mode='move'):
        super().__init__('move_to_clear_area')
        self.mode = mode  # 'move' or 'hover'
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

        # Target position for clear space (gripper center position, not TCP)
        self.target_gripper_center_position = [-0.320, -0.5, 0.30]  # [x, y, z] - gripper center position (matches safe_height)
        # TCP to gripper center offset (24cm along gripper Z-axis, from TCP to gripper center)
        self.tcp_to_gripper_center_offset = 0.24  # 0.24m = 24cm

        # EE pose data storage
        self.ee_pose_received = False
        self.ee_position = None
        self.ee_quat = None

        # Current joint angles storage
        self.current_joint_angles = None
        self.joint_angles_received = False

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
        
        self.action_client.wait_for_server()
        
        # Log mode
        self.get_logger().info(f"Using {self.mode.upper()} mode")
        
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
        Compute the 3D positions of all joints in the robot arm using forward kinematics.

        Args:
            joint_angles: Array of 6 joint angles in radians

        Returns:
            List of 7 positions (base + 6 joints) as numpy arrays [x, y, z]
        """
        # UR5e DH parameters (same as in ik_solver.py)
        dh_params_local = [
            (0,  0.1625,  0,     np.pi/2),   # Joint 1
            (0,  0,      -0.425,  0),         # Joint 2
            (0,  0,      -0.3922, 0),         # Joint 3
            (0,  0.1333,  0,     np.pi/2),   # Joint 4
            (0,  0.0997,  0,    -np.pi/2),   # Joint 5
            (0,  0.0996,  0,     0)           # Joint 6
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
        for i, (theta, d, a, alpha) in enumerate(dh_params_local):
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
        Compute the minimum distance between two line segments.

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

    def check_trajectory_collision(self, start_joints, target_joints, num_samples=20, z_threshold=-0.01):
        """
        Check if any point along the interpolated trajectory has a collision.

        Args:
            start_joints: Starting joint configuration
            target_joints: Target joint configuration
            num_samples: Number of samples along the trajectory to check
            z_threshold: Minimum allowed Z position (meters)

        Returns:
            True if collision detected along trajectory, False otherwise
        """
        for i in range(num_samples + 1):
            alpha = i / num_samples
            interpolated_joints = start_joints + alpha * (target_joints - start_joints)
            if self.check_collision_with_table(interpolated_joints, z_threshold=z_threshold):
                self.get_logger().warn(f"Trajectory table collision at alpha={alpha:.2f}")
                return True
            if self.check_self_collision(interpolated_joints):
                self.get_logger().warn(f"Trajectory self-collision at alpha={alpha:.2f}")
                return True
        return False

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
        
        # Convert quaternion directly to rotation matrix to avoid precision loss from RPY conversion
        from scipy.spatial.transform import Rotation as Rot
        from scipy.optimize import minimize
        
        # Determine target orientation based on mode
        if self.mode == 'hover':
            # Check if EE is already above 0.15m - if so, skip step 1
            current_height = current_pos[2]
            if current_height >= 0.15:
                self.get_logger().info(f"EE already at height {current_height:.3f}m (>= 0.15m), skipping step 1")
            else:
                # Step 1: First move to intermediate height (0.15m) to normalize joint positions
                # This prevents joint limit issues when transitioning to hover position
                self.get_logger().info(f"Step 1: Moving to intermediate height (0.15m) from {current_height:.3f}m...")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                safe_height_cmd = f"timeout 30 /usr/bin/python3 {script_dir}/move_to_safe_height.py --height 0.15"

                step1_success = False
                step1_error = None

                try:
                    result = subprocess.run(
                        safe_height_cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        cwd=script_dir
                    )

                    # Parse JSON result to check actual success/failure
                    if result.stdout and "__RESULT_JSON__" in result.stdout and "__END_RESULT_JSON__" in result.stdout:
                        start_idx = result.stdout.rfind("__RESULT_JSON__") + len("__RESULT_JSON__")
                        end_idx = result.stdout.rfind("__END_RESULT_JSON__")
                        json_str = result.stdout[start_idx:end_idx].strip()
                        try:
                            import json
                            step1_result = json.loads(json_str)
                            if step1_result.get("result") == "success":
                                step1_success = True
                            else:
                                step1_error = step1_result.get("error", "unknown error")
                        except json.JSONDecodeError:
                            step1_error = "Failed to parse JSON result"

                    # Check return code
                    if result.returncode == 124:
                        step1_error = "Timed out after 30s"
                    elif result.returncode != 0 and not step1_error:
                        step1_error = f"Return code {result.returncode}"

                except subprocess.TimeoutExpired:
                    step1_error = "Subprocess timed out"
                except Exception as e:
                    step1_error = str(e)

                if step1_success:
                    self.get_logger().info("Step 1 completed successfully")
                else:
                    self.error_message = f"Step 1 failed: {step1_error or 'move_to_safe_height did not complete'}"
                    self.get_logger().error(self.error_message)
                    self.operation_success = False
                    self.operation_complete = True
                    rclpy.shutdown()
                    return

            # Step 2: Now do the actual hover movement
            self.get_logger().info("Step 2: Moving to hover position...")

            # Hover mode: Use fixed joint angles for top-down hover position
            # Pre-computed joint angles for RPY [0, 180, 0]
            joint_angles = [
                0.775002,   # shoulder_pan_joint
                -1.272476,  # shoulder_lift_joint
                1.718332,   # elbow_joint
                -2.016652,  # wrist_1_joint
                -1.570796,  # wrist_2_joint
                -0.795794,  # wrist_3_joint
            ]

            # Create trajectory point - same duration as move_home (5 seconds)
            point = JointTrajectoryPoint(
                positions=[float(x) for x in joint_angles],
                velocities=[0.0] * 6,
                time_from_start=Duration(sec=5)  # 5 seconds movement (same as HOME_MOVEMENT_DURATION)
            )

            # Create and send trajectory
            goal = FollowJointTrajectory.Goal()
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            traj.points = [point]

            goal.trajectory = traj
            goal.goal_time_tolerance = Duration(sec=1)

            self.get_logger().info("Trajectory sent and accepted")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)
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

            # Calculate TCP position from gripper center position
            # The gripper Z-axis points from TCP to gripper center
            # Apply offset only to X and Y, keep Z constant
            gripper_z_axis = target_rot_matrix[:, 2]  # Z-axis of gripper frame in world frame
            offset_vector = -self.tcp_to_gripper_center_offset * gripper_z_axis
            tcp_position = np.array(self.target_gripper_center_position) + offset_vector
            tcp_position[2] = self.target_gripper_center_position[2]  # Keep Z constant
            
            # Compute inverse kinematics for target pose
            # Use quaternion directly converted to rotation matrix for more accurate orientation preservation
            try:
                # Create target pose with quaternion-derived rotation matrix
                target_pose = np.eye(4)
                target_pose[:3, 3] = tcp_position
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

                from primitives.utils.unified_ik import IKSolverConfig, IKSolver

                q_guess = self.current_joint_angles.copy()
                start_joints = self.current_joint_angles.copy()

                def collision_checker(joint_angles):
                    return (self.check_collision_with_table(joint_angles, z_threshold=-0.01)
                            or self.check_self_collision(joint_angles))

                extended_bounds = [(-2*np.pi, 2*np.pi)] * 6
                config = IKSolverConfig(
                    early_termination=False,
                    joint_bounds=extended_bounds,
                )

                # Strategy: Try all yaw orientations with position perturbations and seed variations
                solver = IKSolver(config)
                candidate_solutions = []

                seed_perturbation_offsets = [
                    [0, 0, 0, 0, 0, 0],  # No perturbation first
                    [0.1, 0, 0, 0, 0, 0],
                    [-0.1, 0, 0, 0, 0, 0],
                    [0.5, 0, 0, 0, 0, 0],
                    [-0.5, 0, 0, 0, 0, 0],
                    [np.pi, 0, 0, 0, 0, 0],
                ]

                for yaw, rot_matrix in target_orientations:
                    # Build pose for this yaw orientation
                    # Recalculate TCP position for this orientation (Z offset may differ slightly)
                    gripper_z = rot_matrix[:, 2]
                    offset_vec = -self.tcp_to_gripper_center_offset * gripper_z
                    tcp_pos = np.array(self.target_gripper_center_position) + offset_vec
                    tcp_pos[2] = self.target_gripper_center_position[2]  # Keep Z constant

                    pose = np.eye(4)
                    pose[:3, 3] = tcp_pos
                    pose[:3, :3] = rot_matrix

                    # Try with position perturbations
                    for i in range(max_tries):
                        perturbation_values = [i * dx, -i * dx] if i > 0 else [0]
                        for perturbation in perturbation_values:
                            perturbed_pos = tcp_pos.copy()
                            perturbed_pos[0] += perturbation
                            if i > max_tries // 2:
                                perturbed_pos[1] += perturbation * 0.5
                            perturbed_pose = pose.copy()
                            perturbed_pose[:3, 3] = perturbed_pos

                            ik_result = solver._solve_single(q_guess, perturbed_pose, collision_checker, None)
                            if ik_result is not None and not ik_result.has_collision and ik_result.cost < config.acceptable_cost:
                                candidate_solutions.append((ik_result.cost, ik_result.joint_angles, yaw))

                    # Try with different seeds
                    for pert in seed_perturbation_offsets:
                        seed = q_guess + np.array(pert)
                        ik_result = solver._solve_single(seed, pose, collision_checker, None)
                        if ik_result is not None and not ik_result.has_collision and ik_result.cost < config.acceptable_cost:
                            candidate_solutions.append((ik_result.cost, ik_result.joint_angles, yaw))

                if not candidate_solutions:
                    self.error_message = "IK solver failed: no collision-free solution found for clear space move"
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
                total_duration = 5.0
                collision_free_trajectories = []  # List of (travel_distance, target_joints, ik_cost, yaw)

                for cost, candidate_joints, yaw in candidate_solutions:
                    target_joints = np.array(candidate_joints).copy()

                    # Generate multiple joint wrapping variants for this solution
                    # Try different combinations of ±2π on joints that have large differences
                    wrapping_variants = [target_joints.copy()]

                    for joint_idx in range(6):
                        diff = target_joints[joint_idx] - start_joints[joint_idx]
                        new_variants = []

                        for variant in wrapping_variants:
                            # Keep original
                            new_variants.append(variant.copy())

                            # Try +2π if diff is negative and large
                            if diff < -np.pi/2:
                                v = variant.copy()
                                v[joint_idx] += 2 * np.pi
                                new_variants.append(v)

                            # Try -2π if diff is positive and large
                            if diff > np.pi/2:
                                v = variant.copy()
                                v[joint_idx] -= 2 * np.pi
                                new_variants.append(v)

                        wrapping_variants = new_variants

                    # Test each wrapping variant for collision-free trajectory
                    for variant in wrapping_variants:
                        # Also check that wrapped solution is still collision-free
                        if self.check_collision_with_table(variant, z_threshold=-0.01):
                            continue
                        if self.check_self_collision(variant):
                            continue

                        # Check trajectory collision
                        if not self.check_trajectory_collision(start_joints, variant, num_samples=20, z_threshold=-0.01):
                            # Calculate total joint travel distance (sum of absolute angle changes)
                            travel_distance = np.sum(np.abs(variant - start_joints))
                            collision_free_trajectories.append((travel_distance, variant.copy(), cost, yaw))

                if not collision_free_trajectories:
                    self.error_message = "IK solution rejected: all candidate trajectories would cause collision"
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

                self.get_logger().info(f"Creating joint-space trajectory with {num_waypoints} waypoints")

                trajectory_points = []

                # Add starting point at t=0
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

                # Create and send trajectory
                goal = FollowJointTrajectory.Goal()
                traj = JointTrajectory()
                traj.joint_names = self.joint_names
                traj.points = trajectory_points

                goal.trajectory = traj
                goal.goal_time_tolerance = Duration(sec=1)

                self.get_logger().info(f"Joint-space trajectory with {len(trajectory_points)} waypoints sent")
                self._send_goal_future = self.action_client.send_goal_async(
                    goal,
                    feedback_callback=self.feedback_callback
                )
                self._send_goal_future.add_done_callback(self.goal_response)

            except Exception as e:
                self.error_message = f"Failed to compute IK: {e}"
                self.get_logger().error(self.error_message)
                self.operation_success = False
                self.operation_complete = True
                rclpy.shutdown()

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
        result = future.result()
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info("Movement completed successfully")
            self.operation_success = True
        else:
            result_msg = result.result
            # Handle various error codes
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
                # Status codes: 1=ACCEPTED, 2=EXECUTING, 3=CANCELING, 4=SUCCEEDED, 5=CANCELED, 6=ABORTED
                status_names = {1: "ACCEPTED", 2: "EXECUTING", 3: "CANCELING", 4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}
                status_name = status_names.get(result.status, f"UNKNOWN({result.status})")
                self.error_message = f"Trajectory failed: status={status_name}, error_code={error_code}"
            self.get_logger().error(self.error_message)
            self.operation_success = False
        self.operation_complete = True
        # Destroy node before shutdown to ensure clean exit
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
    parser.add_argument('--hover', action='store_true',
                       help='Move to target position with top-down (face-down) EE orientation')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.hover:
        mode = 'hover'
    else:
        mode = 'move'  # Default mode
    
    rclpy.init()
    node = MoveToClearArea(mode=mode)
    try:
        # Spin until operation is complete with timeout
        import time
        start_time = time.time()
        timeout_sec = 30.0  # 30 second timeout for trajectory execution

        while rclpy.ok() and not node.operation_complete:
            rclpy.spin_once(node, timeout_sec=0.1)

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

