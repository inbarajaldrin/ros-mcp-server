#!/usr/bin/env python3
"""
Translate Object Primitive - Move held object to assembly position

Modes:
- --move-to-base: Move to hover position above base (sim: topics, real: provided args)
- --perform-insert: Sim = move to final position, Real = subprocess to prismatic_peg_insertion.py
- --move-to-safe-height: Subprocess to move_to_safe_height.py
- --move-away-from-base: Subprocess to move_to_clear_area.py

Note: --move-to-base, --perform-insert, --move-to-safe-height, and --move-away-from-base are mutually exclusive.

Usage:
    # Sim mode - move to hover above base
    python3 translate_object.py --mode sim --object-name fork_orange --base-name base --move-to-base

    # Sim mode - move to final position (insert)
    python3 translate_object.py --mode sim --object-name fork_orange --base-name base --perform-insert

    # Real mode - move to hover above base
    python3 translate_object.py --mode real --base-name base --move-to-base --final-base-pos 0.5 -0.37 0.1882 --final-base-orientation 0.0 0.0 0.0 1.0

    # Real mode - perform insert (peg-in-hole force control)
    python3 translate_object.py --mode real --perform-insert

    # Move to safe height
    python3 translate_object.py --mode sim --move-to-safe-height

    # Move away from base
    python3 translate_object.py --mode real --move-away-from-base
"""

import sys
import os

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import subprocess
import argparse
import threading
import json
import time
import glob
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.action import ActionClient
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from primitives.shared.ik import (
    forward_kinematics, dh_params, compute_cartesian_waypoints_ik,
    IKSolverConfig, IKSolver,
)
from primitives.shared.velocity_profiles import s_curve_profile, compute_duration
from primitives.shared.config import (
    TABLE_HEIGHT, GRIPPER_CENTER_TOOL_OFFSET,
    DEFAULT_BASE_POSITION, DEFAULT_BASE_ORIENTATION,
)
from primitives.shared.collision import (
    check_collision_with_table, check_self_collision,
    check_ee_below_base, check_compact_configuration,
)
from primitives.shared.fold_symmetry import load_symmetry_data, equivalent_orientations
from primitives.rotate_object import ExtendedCardinalOrientations
from utils.data_path_finder import get_assembly_data_dir, get_aruco_data_dir, get_symmetry_dir

# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())
BASE_TOPIC = "/objects_poses_sim"
OBJECT_TOPIC = "/objects_poses_sim"
EE_TOPIC = "/tcp_pose_broadcaster/pose"
HOVER_HEIGHT = 0.15  # Height to hover above base before descending

# Set up Python logging for non-ROS contexts (subprocess helpers)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('translate_object')


def output_result(result):
    """Output JSON result with markers for MCP server parsing"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


def find_assembly_json_by_base_name(base_name, data_dir=ASSEMBLY_DATA_DIR, logger=None):
    """Find the assembly JSON file that contains the given base name."""
    if not os.path.exists(data_dir):
        if logger:
            logger.error(f"Data directory not found: {data_dir}")
        return None

    json_files = glob.glob(os.path.join(data_dir, "*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)
            components = config.get('components', [])
            for component in components:
                if component.get('name', '') == base_name:
                    return json_file
        except (json.JSONDecodeError, IOError):
            continue

    if logger:
        logger.warn(f"No assembly JSON found for base '{base_name}' in {data_dir}")
    return None


def load_grasp_point_position(object_name, grasp_id, logger=None):
    """Load grasp point position from grasp points JSON file."""
    data_dir = get_aruco_data_dir() / "grasp_points"
    json_path = data_dir / f"{object_name}_grasp_points.json"
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            for gp in data.get('grasp_points', []):
                if gp['id'] == grasp_id:
                    pos = gp['position']
                    if logger:
                        logger.info(f"Loaded grasp point {grasp_id} for {object_name}: [{pos['x']:.4f}, {pos['y']:.4f}, {pos['z']:.4f}]")
                    return np.array([pos['x'], pos['y'], pos['z']])
            if logger:
                logger.warn(f"Grasp ID {grasp_id} not found in {json_path.name}")
        except (json.JSONDecodeError, IOError, KeyError) as e:
            if logger:
                logger.error(f"Error reading {json_path}: {e}")
    if logger:
        logger.error(f"No grasp points file found for '{object_name}'")
    return None


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class TranslateObject(Node):
    def __init__(self, mode):
        super().__init__('translate_object')

        if mode not in ('sim', 'real'):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'sim' or 'real'.")
        self.mode = mode

        # Error tracking for JSON output
        self.error_message = None
        self.object_name = None
        self.base_name = None

        # Assembly config (lazy-loaded per base)
        self.assembly_config = {}
        self.assembly_json_file = None
        self.loaded_base_name = None

        # Pose subscribers — topics only needed in sim mode
        if self.mode == 'sim':
            self.base_sub = self.create_subscription(TFMessage, BASE_TOPIC, self._base_cb, 10)
            self.object_sub = self.create_subscription(TFMessage, OBJECT_TOPIC, self._object_cb, 10)
        else:
            self.base_sub = None
            self.object_sub = None

        ee_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.ee_sub = self.create_subscription(PoseStamped, EE_TOPIC, self._ee_cb, ee_qos)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)

        self.current_poses = {}
        self.current_ee_pose = None
        self.current_joint_angles = None
        self.joint_angles_received = False

        # Action client
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]
        self.action_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
        )

        self.get_logger().info(f"Using {self.mode.upper()} mode")

    # --- Callbacks ---

    def _base_cb(self, msg):
        for transform in msg.transforms:
            self.current_poses[transform.child_frame_id] = transform

    def _object_cb(self, msg):
        for transform in msg.transforms:
            self.current_poses[transform.child_frame_id] = transform

    def _ee_cb(self, msg):
        self.current_ee_pose = msg

    def _joint_state_cb(self, msg: JointState):
        if len(msg.name) == 6 and len(msg.position) == 6:
            joint_dict = dict(zip(msg.name, msg.position))
            ordered = [joint_dict[n] for n in self.joint_names if n in joint_dict]
            if len(ordered) == 6:
                self.current_joint_angles = np.array(ordered)
                self.joint_angles_received = True

    # --- Helpers ---

    def transform_to_matrix(self, transform):
        """Convert ROS Transform to 4x4 matrix."""
        t = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
        q = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
        T = np.eye(4)
        T[:3, :3] = R.from_quat(q).as_matrix()
        T[:3, 3] = t
        return T

    def pose_to_matrix(self, pose):
        """Convert ROS Pose to 4x4 matrix."""
        t = np.array([pose.position.x, pose.position.y, pose.position.z])
        q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        T = np.eye(4)
        T[:3, :3] = R.from_quat(q).as_matrix()
        T[:3, 3] = t
        return T

    def matrix_to_rpy(self, T):
        """Convert 4x4 matrix to (position, rpy_degrees)."""
        position = T[:3, 3]
        rpy_deg = np.degrees(R.from_matrix(T[:3, :3]).as_euler('xyz'))
        # Canonicalize
        roll, pitch, yaw = rpy_deg
        if abs(pitch) < 5 and abs(abs(roll) - 180) < 5:
            rpy_deg = np.array([0.0, 180.0, (yaw % 360) - 180])
        return position, rpy_deg

    def load_assembly_config(self, base_name):
        """Load assembly JSON for the given base."""
        json_file = find_assembly_json_by_base_name(base_name, ASSEMBLY_DATA_DIR, self.get_logger())
        if json_file:
            self.assembly_json_file = json_file
            self.loaded_base_name = base_name
        else:
            self.get_logger().error(f"Could not find assembly JSON for base '{base_name}'")
            return {}
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.get_logger().error(f"Error loading assembly JSON: {e}")
            return {}

    def get_object_target_position(self, object_name):
        """Get target position for object from assembly config (relative to base)."""
        for comp in self.assembly_config.get('components', []):
            if comp.get('name') == object_name:
                p = comp.get('position', {})
                return np.array([p.get('x', 0), p.get('y', 0), p.get('z', 0)])
        return None

    def get_object_target_orientation(self, object_name):
        """Get target orientation quaternion for object from assembly config (relative to base)."""
        for comp in self.assembly_config.get('components', []):
            if comp.get('name') == object_name:
                q = comp.get('rotation', {}).get('quaternion', {})
                return np.array([q.get('x', 0.0), q.get('y', 0.0), q.get('z', 0.0), q.get('w', 1.0)])
        return None

    def read_current_joint_angles(self):
        """Block until joint angles arrive (10s timeout)."""
        self.joint_angles_received = False
        timeout_count = 0
        while rclpy.ok() and not self.joint_angles_received and timeout_count < 100:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout_count += 1
        if not self.joint_angles_received or self.current_joint_angles is None:
            self.get_logger().error("Timeout waiting for joint angles")
            return None
        return self.current_joint_angles.copy()

    def compute_ik_with_current_seed(self, target_position, target_quat, max_tries=5, dx=0.001):
        """Compute IK using current joint angles as seed."""
        target_rotation = R.from_quat(target_quat)
        target_pose = np.eye(4)
        target_pose[:3, 3] = target_position
        target_pose[:3, :3] = target_rotation.as_matrix()

        if self.current_joint_angles is None:
            self.get_logger().error("Current joint angles not available!")
            return None

        def collision_checker(joint_angles):
            if self.mode != 'sim':
                return False
            return (check_collision_with_table(joint_angles)
                    or check_self_collision(joint_angles)
                    or check_ee_below_base(joint_angles)
                    or check_compact_configuration(joint_angles))

        joint_bounds = [
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-2 * np.pi, 2 * np.pi),
        ]
        solver = IKSolver(IKSolverConfig(joint_bounds=joint_bounds))
        result = solver.solve(
            seeds=[self.current_joint_angles.copy()],
            target_pose=target_pose,
            collision_checker=collision_checker,
            perturbations=max_tries,
            dx=dx,
        )
        if result is not None:
            return result
        self.get_logger().error("IK failed: couldn't find solution")
        return None

    def execute_trajectory(self, trajectory):
        """Execute trajectory via FollowJointTrajectory action (blocking)."""
        try:
            if 'traj1' not in trajectory or not trajectory['traj1']:
                return False

            points = trajectory['traj1']
            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names

            for point in points:
                tp = JointTrajectoryPoint()
                tp.positions = point['positions']
                if 'velocities' in point:
                    tp.velocities = point['velocities']
                tp.time_from_start = point['time_from_start']
                traj_msg.points.append(tp)

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)

            future = self.action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()

            if not goal_handle.accepted:
                self.error_message = "External control program stopped or robot in protective stop"
                self.get_logger().error(self.error_message)
                return False

            self.get_logger().info(f"Trajectory with {len(points)} waypoints sent and accepted")
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result()

            if result.status == 4:  # SUCCEEDED
                self.get_logger().info("Movement completed successfully")
                return True
            else:
                result_msg = result.result
                if result_msg.error_code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
                    self.error_message = "Velocity or acceleration limits exceeded. Enable robot in URcap to fix this."
                else:
                    self.error_message = f"Trajectory failed with status code {result.status}"
                self.get_logger().error(self.error_message)
                return False
        except Exception as e:
            self.error_message = f"Trajectory execution error: {e}"
            self.get_logger().error(self.error_message)
            return False

    # ------------------------------------------------------------------
    # Sim mode: translate to target (hover or final position)
    # ------------------------------------------------------------------

    def translate_for_target_sim(self, object_name, base_name, hover=True):
        """
        Sim mode: Compute and execute EE translation.

        Args:
            object_name: Name of the held object
            base_name: Name of the base object
            hover: If True, target is HOVER_HEIGHT above base (move-to-base).
                   If False, target is final object position from JSON (perform-insert sim).
        """
        self.object_name = object_name
        self.base_name = base_name

        # Load assembly config
        if self.loaded_base_name != base_name:
            self.assembly_config = self.load_assembly_config(base_name=base_name)
            if not self.assembly_config:
                self.error_message = f"Failed to load assembly config for base '{base_name}'"
                self.get_logger().error(self.error_message)
                return False

        # Wait for pose data
        if not self.current_poses or self.current_ee_pose is None:
            self.error_message = "No pose data available"
            self.get_logger().error(self.error_message)
            return False

        if object_name not in self.current_poses:
            self.error_message = f"Object {object_name} not found"
            self.get_logger().error(self.error_message)
            return False

        if base_name not in self.current_poses:
            self.error_message = f"Base {base_name} not found"
            self.get_logger().error(self.error_message)
            return False

        # Verify grasp
        tcp_pos = np.array([self.current_ee_pose.pose.position.x,
                            self.current_ee_pose.pose.position.y,
                            self.current_ee_pose.pose.position.z])
        tcp_quat = np.array([self.current_ee_pose.pose.orientation.x,
                             self.current_ee_pose.pose.orientation.y,
                             self.current_ee_pose.pose.orientation.z,
                             self.current_ee_pose.pose.orientation.w])
        gripper_center = tcp_pos + R.from_quat(tcp_quat).as_matrix() @ GRIPPER_CENTER_TOOL_OFFSET
        obj_transform = self.current_poses[object_name].transform
        object_pos = np.array([obj_transform.translation.x, obj_transform.translation.y, obj_transform.translation.z])
        grasp_distance = np.linalg.norm(object_pos - gripper_center)
        if grasp_distance > 0.06:
            self.error_message = f"Grasp check failed: {object_name} is {grasp_distance * 1000:.1f}mm from gripper center."
            self.get_logger().error(self.error_message)
            return False
        self.get_logger().info(f"Grasp verified: {object_name} is {grasp_distance * 1000:.1f}mm from gripper center")

        # Convert poses to matrices
        T_EE_current = self.pose_to_matrix(self.current_ee_pose.pose)
        T_object_current = self.transform_to_matrix(self.current_poses[object_name].transform)
        T_base_current = self.transform_to_matrix(self.current_poses[base_name].transform)

        # Calculate grasp transformation
        T_grasp = np.linalg.inv(T_EE_current) @ T_object_current

        ee_current_position, _ = self.matrix_to_rpy(T_EE_current)
        base_current_position, _ = self.matrix_to_rpy(T_base_current)

        # Get target object position from JSON (relative to base)
        target_position_relative = self.get_object_target_position(object_name)
        if target_position_relative is None:
            self.error_message = f"No target position found for {object_name} in JSON"
            self.get_logger().error(self.error_message)
            return False

        # Transform to world frame
        R_base_current = T_base_current[:3, :3]
        target_object_position_abs = base_current_position + R_base_current @ target_position_relative

        # Create target object transformation (keep current orientation)
        T_object_target = np.eye(4)
        T_object_target[:3, :3] = T_object_current[:3, :3]
        T_object_target[:3, 3] = target_object_position_abs

        # Required EE position to place object at target
        T_EE_target = T_object_target @ np.linalg.inv(T_grasp)

        ee_target_position = T_EE_target[:3, 3]
        ee_target_rot_matrix = T_EE_target[:3, :3]

        if hover:
            # Hover: same XY as target, Z = base + HOVER_HEIGHT
            hover_gripper_center = ee_target_position.copy()
            hover_gripper_center[2] = base_current_position[2] + HOVER_HEIGHT
            tool_offset_world = ee_target_rot_matrix @ GRIPPER_CENTER_TOOL_OFFSET
            target_flange = hover_gripper_center - tool_offset_world
            self.get_logger().info(
                f"Hover gripper center Z: {hover_gripper_center[2]:.4f}, "
                f"hover flange Z: {target_flange[2]:.4f} (offset: {tool_offset_world[2]:.4f})"
            )
        else:
            # Final position: use EE target directly
            target_flange = ee_target_position
            self.get_logger().info(f"Target flange position: {target_flange}")

        self.get_logger().info(
            f"Final object position: [{target_object_position_abs[0]:.4f}, "
            f"{target_object_position_abs[1]:.4f}, {target_object_position_abs[2]:.4f}]"
        )

        # Read current joint angles
        if self.current_joint_angles is None:
            if self.read_current_joint_angles() is None:
                self.error_message = "Could not read current joint angles"
                self.get_logger().error(self.error_message)
                return False

        # Jacobian-based differential IK
        num_waypoints = 60
        self.get_logger().info("Computing dense IK waypoints (Jacobian)...")
        waypoints = compute_cartesian_waypoints_ik(
            self.current_joint_angles,
            target_z=target_flange[2],
            num_waypoints=num_waypoints,
            target_pos=target_flange.tolist() if hasattr(target_flange, 'tolist') else list(target_flange),
            target_orientation=ee_target_rot_matrix if not hover else None,
        )
        if waypoints is None:
            self.error_message = "IK failed for Cartesian waypoints"
            self.get_logger().error(self.error_message)
            return False

        # Post-hoc collision check
        for i, wp_joints in enumerate(waypoints):
            if (check_collision_with_table(wp_joints)
                    or check_self_collision(wp_joints)
                    or check_ee_below_base(wp_joints)
                    or check_compact_configuration(wp_joints)):
                self.error_message = f"Collision detected at waypoint {i + 1}/{num_waypoints}"
                self.get_logger().error(self.error_message)
                return False

        all_joint_angles = [self.current_joint_angles.copy()] + list(waypoints)

        joint_dist = float(np.max(np.abs(np.array(waypoints[-1]) - np.array(self.current_joint_angles))))
        total_duration = compute_duration(joint_distance=joint_dist, profile='s_curve')
        self.get_logger().info(f"Duration: {total_duration:.2f}s (joint={joint_dist:.2f}rad)")

        profile = s_curve_profile(all_joint_angles, total_duration)
        trajectory_points = []
        for positions, velocities, t_i in profile:
            trajectory_points.append({
                "positions": positions,
                "velocities": velocities,
                "time_from_start": Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
            })

        self.get_logger().info(
            f"Generated {len(trajectory_points)} Cartesian waypoints with s-curve velocity profile"
        )

        success = self.execute_trajectory({"traj1": trajectory_points})
        if not success:
            return False

        if not hover:
            # For insert: just wait and return
            time.sleep(0.5)
            self.get_logger().info("Insert movement completed successfully")

        return success

    # ------------------------------------------------------------------
    # Real mode: translate to hover with grasp offsets + fold symmetry
    # ------------------------------------------------------------------

    def translate_for_target_real(self, object_name, base_name,
                                  final_base_pos=None, final_base_orientation=None,
                                  use_default_base=False, grasp_id=None,
                                  object_orientation=None):
        """
        Real mode: Calculate and execute EE translation to hover position.
        Uses provided base position/orientation (no sim topics).
        """
        self.object_name = object_name
        self.base_name = base_name

        if grasp_id is None:
            self.error_message = "grasp_id is required for real mode"
            self.get_logger().error(self.error_message)
            return False
        if object_orientation is None:
            self.error_message = "object_orientation is required for real mode"
            self.get_logger().error(self.error_message)
            return False

        # Load assembly config
        if self.loaded_base_name != base_name:
            self.assembly_config = self.load_assembly_config(base_name=base_name)
            if not self.assembly_config:
                self.error_message = f"Failed to load assembly config for base '{base_name}'"
                self.get_logger().error(self.error_message)
                return False

        if self.current_ee_pose is None:
            self.error_message = "End-effector pose not available"
            self.get_logger().error(self.error_message)
            return False

        # Resolve base position/orientation
        if final_base_pos is None:
            if use_default_base:
                final_base_pos = DEFAULT_BASE_POSITION
                self.get_logger().info(f"Using default base position: {final_base_pos}")
            else:
                self.error_message = "Base position not provided. Use --final-base-pos or --use-default-base-position"
                self.get_logger().error(self.error_message)
                return False

        if final_base_orientation is None:
            if use_default_base:
                final_base_orientation = DEFAULT_BASE_ORIENTATION
                self.get_logger().info(f"Using default base orientation: {final_base_orientation}")
            else:
                final_base_orientation = [0.0, 0.0, 0.0, 1.0]
                self.get_logger().info("Using identity base orientation (not provided)")

        # Create base pose matrix from args
        base_pose = PoseStamped()
        base_pose.pose.position.x = final_base_pos[0]
        base_pose.pose.position.y = final_base_pos[1]
        base_pose.pose.position.z = final_base_pos[2]
        base_pose.pose.orientation.x = final_base_orientation[0]
        base_pose.pose.orientation.y = final_base_orientation[1]
        base_pose.pose.orientation.z = final_base_orientation[2]
        base_pose.pose.orientation.w = final_base_orientation[3]
        T_base_current = self.pose_to_matrix(base_pose.pose)
        self.get_logger().info(f"Using base position: {final_base_pos}, orientation: {final_base_orientation}")

        T_EE_current = self.pose_to_matrix(self.current_ee_pose.pose)
        ee_current_position, _ = self.matrix_to_rpy(T_EE_current)
        base_current_position, _ = self.matrix_to_rpy(T_base_current)

        # Target object position from JSON
        target_position_relative = self.get_object_target_position(object_name)
        if target_position_relative is None:
            self.error_message = f"No target position found for {object_name} in JSON"
            self.get_logger().error(self.error_message)
            return False

        target_orientation_relative = self.get_object_target_orientation(object_name)
        if target_orientation_relative is None:
            self.get_logger().warn(f"No target orientation found for {object_name}, using identity")
            target_orientation_relative = np.array([0.0, 0.0, 0.0, 1.0])

        # Transform to world frame
        R_base_current = T_base_current[:3, :3]
        target_object_position_abs = base_current_position + R_base_current @ target_position_relative
        R_target_relative = R.from_quat(target_orientation_relative).as_matrix()
        R_target_abs = R_base_current @ R_target_relative
        target_orientation_abs = R.from_matrix(R_target_abs).as_quat()

        self.get_logger().info(f"Target object position (world): {target_object_position_abs}")
        self.get_logger().info(f"Target object orientation (world): {target_orientation_abs}")
        self.get_logger().info("Keeping current EE orientation unchanged (from reorient step)")

        # Load grasp point offset
        grasp_offset = load_grasp_point_position(object_name, grasp_id, logger=self.get_logger())
        if grasp_offset is None:
            self.error_message = f"Could not load grasp point {grasp_id} for '{object_name}'"
            self.get_logger().error(self.error_message)
            return False

        # Validate quaternion
        quat_array = np.array(object_orientation)
        quat_norm_sq = np.sum(quat_array ** 2)
        if abs(quat_norm_sq - 1.0) > 0.1:
            self.get_logger().error(f"Invalid quaternion: norm² = {quat_norm_sq:.2f}")
            self.error_message = "Current object orientation quaternion is malformed"
            return False

        # Fold symmetry: snap to closest equivalent orientation
        R_object_current = R.from_quat(object_orientation).as_matrix()
        symmetry_dir = str(get_symmetry_dir())
        fold_data = load_symmetry_data(object_name, symmetry_dir)

        if fold_data is not None:
            equivalents = equivalent_orientations(R_target_abs, fold_data)
            best_pos_error = float('inf')
            best_orientation_error = float('inf')
            R_grasp_rotation = R_target_abs
            for R_eq in equivalents:
                orientation_error = ExtendedCardinalOrientations.rotation_matrix_distance(R_object_current, R_eq)
                grasp_world_offset_candidate = R_eq @ grasp_offset
                pos_error = np.linalg.norm(grasp_world_offset_candidate - (R_object_current @ grasp_offset))
                if pos_error < best_pos_error or (pos_error == best_pos_error and orientation_error < best_orientation_error):
                    best_pos_error = pos_error
                    best_orientation_error = orientation_error
                    R_grasp_rotation = R_eq
            self.get_logger().info(
                f"Snapped orientation to closest equivalent "
                f"(angle error: {np.degrees(best_orientation_error):.1f}°, "
                f"position error: {best_pos_error * 1000:.2f}mm)"
            )
        else:
            self.get_logger().info("No symmetry data, using current object orientation")
            R_grasp_rotation = R_object_current

        grasp_world_offset = R_grasp_rotation @ grasp_offset
        self.get_logger().info(f"Grasp point {grasp_id} offset (CAD frame): {grasp_offset}")
        self.get_logger().info(f"Grasp point offset (world frame): {grasp_world_offset}")

        # Target gripper center
        target_gripper_center = target_object_position_abs + grasp_world_offset
        hover_gripper_center = target_gripper_center.copy()
        hover_gripper_center[2] = base_current_position[2] + HOVER_HEIGHT

        self.get_logger().info(f"Target gripper center: {target_gripper_center}")
        self.get_logger().info(f"Hover gripper center (with {HOVER_HEIGHT}m offset): {hover_gripper_center}")

        # Read joints
        if self.current_joint_angles is None:
            if self.read_current_joint_angles() is None:
                self.error_message = "Could not read current joint angles"
                self.get_logger().error(self.error_message)
                return False

        # Convert gripper center to flange using FK-derived rotation
        T_fk = forward_kinematics(dh_params, self.current_joint_angles)
        R_fk = T_fk[:3, :3]
        tool_offset_world = R_fk @ GRIPPER_CENTER_TOOL_OFFSET
        hover_flange = hover_gripper_center - tool_offset_world
        self.get_logger().info(f"Hover flange position (FK-derived): {hover_flange}")

        # Jacobian IK
        num_waypoints = 60
        self.get_logger().info("Computing dense IK waypoints (Jacobian)...")
        waypoints = compute_cartesian_waypoints_ik(
            self.current_joint_angles,
            target_z=hover_flange[2],
            num_waypoints=num_waypoints,
            target_pos=hover_flange.tolist(),
        )
        if waypoints is None:
            self.error_message = "IK failed for Cartesian waypoints"
            self.get_logger().error(self.error_message)
            return False

        all_joint_angles = [self.current_joint_angles.copy()] + list(waypoints)

        joint_dist = float(np.max(np.abs(np.array(waypoints[-1]) - np.array(self.current_joint_angles))))
        total_duration = compute_duration(joint_distance=joint_dist, profile='s_curve')
        self.get_logger().info(f"Duration: {total_duration:.2f}s (joint={joint_dist:.2f}rad)")

        profile = s_curve_profile(all_joint_angles, total_duration)
        trajectory_points = []
        for positions, velocities, t_i in profile:
            trajectory_points.append({
                "positions": positions,
                "velocities": velocities,
                "time_from_start": Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
            })

        self.get_logger().info(f"Generated {len(trajectory_points)} Cartesian waypoints with s-curve velocity profile")

        success = self.execute_trajectory({"traj1": trajectory_points})
        if not success:
            self.get_logger().error("Failed to reach target position")
            return False

        # Closed-loop correction
        CORRECTION_THRESHOLD = 0.00025  # 0.25mm
        MAX_CORRECTIONS = 3

        for correction_iter in range(MAX_CORRECTIONS):
            self.joint_angles_received = False
            self.current_ee_pose = None
            timeout = 0
            while rclpy.ok() and (not self.joint_angles_received or self.current_ee_pose is None) and timeout < 50:
                rclpy.spin_once(self, timeout_sec=0.1)
                timeout += 1

            if not self.joint_angles_received or self.current_ee_pose is None:
                self.get_logger().warn("Could not read pose data for correction")
                break

            ee_pos_topic = np.array([self.current_ee_pose.pose.position.x,
                                     self.current_ee_pose.pose.position.y,
                                     self.current_ee_pose.pose.position.z])
            ee_quat_topic = np.array([self.current_ee_pose.pose.orientation.x,
                                      self.current_ee_pose.pose.orientation.y,
                                      self.current_ee_pose.pose.orientation.z,
                                      self.current_ee_pose.pose.orientation.w])
            R_ee_topic = R.from_quat(ee_quat_topic).as_matrix()
            actual_gripper_center = ee_pos_topic + R_ee_topic @ GRIPPER_CENTER_TOOL_OFFSET
            gripper_center_error = hover_gripper_center - actual_gripper_center
            pos_error = np.linalg.norm(gripper_center_error)

            self.get_logger().info(
                f"Correction check {correction_iter + 1}: gripper center error = {pos_error * 1000:.2f}mm "
                f"(actual: [{actual_gripper_center[0] * 1000:.1f}, {actual_gripper_center[1] * 1000:.1f}, {actual_gripper_center[2] * 1000:.1f}]mm)"
            )

            if pos_error <= CORRECTION_THRESHOLD:
                self.get_logger().info(f"Position accuracy OK ({pos_error * 1000:.2f}mm <= {CORRECTION_THRESHOLD * 1000:.2f}mm)")
                break

            # Apply correction in FK space
            T_fk_current = forward_kinematics(dh_params, self.current_joint_angles)
            current_flange_fk = T_fk_current[:3, 3]
            corrected_flange = current_flange_fk + gripper_center_error

            self.get_logger().info(f"Applying correction move (error: {pos_error * 1000:.2f}mm)...")
            correction_waypoints = compute_cartesian_waypoints_ik(
                self.current_joint_angles,
                target_z=corrected_flange[2],
                num_waypoints=20,
                target_pos=corrected_flange.tolist(),
            )
            if correction_waypoints is None:
                self.get_logger().warn("Correction IK failed, skipping")
                break

            corr_all = [self.current_joint_angles.copy()] + list(correction_waypoints)
            corr_n = len(corr_all)
            corr_duration = 1.0
            corr_points = []
            for i in range(corr_n):
                t_i = corr_duration * i / (corr_n - 1)
                if i == 0 or i == corr_n - 1:
                    vels = [0.0] * 6
                else:
                    delta = corr_all[min(i + 1, corr_n - 1)] - corr_all[max(i - 1, 0)]
                    dn = np.linalg.norm(delta)
                    if dn > 1e-8:
                        vels = [float(delta[j] / dn * pos_error / corr_duration) for j in range(6)]
                    else:
                        vels = [0.0] * 6
                corr_points.append({
                    "positions": [float(x) for x in corr_all[i]],
                    "velocities": vels,
                    "time_from_start": Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
                })

            if not self.execute_trajectory({"traj1": corr_points}):
                self.get_logger().warn("Correction trajectory failed")
                break

        return True


# ---------------------------------------------------------------------------
# Subprocess helpers (for actions that remain as subprocesses)
# ---------------------------------------------------------------------------

def extract_json_from_output(output_text):
    """Extract JSON result from subprocess output."""
    if "__RESULT_JSON__" in output_text and "__END_RESULT_JSON__" in output_text:
        start = output_text.find("__RESULT_JSON__") + len("__RESULT_JSON__")
        end = output_text.find("__END_RESULT_JSON__")
        json_str = output_text[start:end].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None


def stream_output(pipe, output_lines, prefix=""):
    """Stream subprocess output line by line."""
    for line in iter(pipe.readline, ''):
        if line:
            line = line.rstrip()
            if line:
                output_lines.append(line)
                logger.info(f"{prefix}{line}")
    pipe.close()


def _make_env():
    """Create environment with PYTHONPATH for subprocess."""
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root
    return env


def run_subprocess(script_path, cmd_args=None, timeout=None):
    """Run a subprocess script and return (success, output_text)."""
    cmd = [sys.executable, script_path] + (cmd_args or [])
    env = _make_env()
    output_lines = []

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    output_thread = threading.Thread(
        target=stream_output, args=(process.stdout, output_lines), daemon=True,
    )
    output_thread.start()

    try:
        returncode = process.wait(timeout=timeout)
        output_thread.join(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        output_thread.join(timeout=1.0)
        return False, '\n'.join(output_lines)

    output_text = '\n'.join(output_lines)
    return returncode == 0, output_text


def run_perform_insert_real(args):
    """Run real-mode insertion subprocess."""
    insertion_type = getattr(args, 'insertion_type', 'prismatic')

    if insertion_type == 'prismatic':
        script_path = os.path.join(os.path.dirname(__file__), 'prismatic_peg_insertion.py')
        logger.info("Using prismatic peg insertion")
    elif insertion_type == 'legacy':
        script_path = os.path.join(os.path.dirname(__file__), '_real_mode_stash', 'legacy', 'peg_in_hole_insert.py')
        logger.info("Using legacy peg_in_hole_insert")
    else:
        return False, f"Unknown insertion type: {insertion_type}"

    cmd_args = []
    if args.object_name:
        cmd_args.extend(['--object-name', args.object_name])
    if args.base_name:
        cmd_args.extend(['--base-name', args.base_name])
    if args.grasp_id is not None:
        cmd_args.extend(['--grasp-id', str(args.grasp_id)])
    if args.final_base_pos:
        cmd_args.extend(['--final-base-pos'] + [str(x) for x in args.final_base_pos])
    if args.final_base_orientation:
        cmd_args.extend(['--final-base-orientation'] + [str(x) for x in args.final_base_orientation])
    if args.use_default_base_position:
        cmd_args.append('--use-default-base-position')
    if args.current_object_orientation is not None:
        cmd_args.extend(['--current-object-orientation'] + [str(x) for x in args.current_object_orientation])

    logger.info("Moving down with passive compliance")
    return run_subprocess(script_path, cmd_args)


def run_move_to_safe_height():
    """Run move_to_safe_height subprocess."""
    script_path = os.path.join(os.path.dirname(__file__), 'core', 'move_to_safe_height.py')
    logger.info("Moving to safe height...")
    return run_subprocess(script_path, timeout=40)


def run_move_to_clear_area():
    """Run move_to_clear_area subprocess."""
    script_path = os.path.join(os.path.dirname(__file__), 'core', 'move_to_clear_area.py')
    logger.info("Moving object away from base to clear area")
    return run_subprocess(script_path, ['--move'])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Translate Object - Move held object to assembly position',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'])
    parser.add_argument('--object-name', type=str)
    parser.add_argument('--base-name', type=str)

    # Movement flags (mutually exclusive)
    parser.add_argument('--move-to-base', action='store_true')
    parser.add_argument('--perform-insert', action='store_true', dest='perform_insert')
    parser.add_argument('--move-to-safe-height', action='store_true')
    parser.add_argument('--move-away-from-base', action='store_true')

    # Real mode arguments
    parser.add_argument('--final-base-pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--final-base-orientation', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'))
    parser.add_argument('--use-default-base-position', action='store_true', dest='use_default_base_position')
    parser.add_argument('--grasp-id', type=int, default=None)
    parser.add_argument('--current-object-orientation', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'))
    parser.add_argument('--insertion-type', type=str, default='prismatic', choices=['prismatic', 'legacy'])

    args = parser.parse_args()

    # Validate flags
    flags_set = sum([args.move_to_base, args.perform_insert, args.move_to_safe_height, args.move_away_from_base])
    if flags_set == 0:
        parser.error("Specify one of --move-to-base, --perform-insert, --move-to-safe-height, --move-away-from-base")
    if flags_set > 1:
        parser.error("Cannot use multiple movement flags together")

    # Validate sim mode requirements
    if args.mode == 'sim' and not args.move_to_safe_height and not args.move_away_from_base:
        if args.object_name is None:
            parser.error("--object-name is required in sim mode")
        if args.base_name is None:
            parser.error("--base-name is required in sim mode")

    # Validate real mode requirements
    if args.mode == 'real' and args.move_to_base:
        if args.base_name is None:
            parser.error("--base-name is required for --move-to-base in real mode")
        if not args.use_default_base_position and args.final_base_pos is None:
            parser.error("--final-base-pos or --use-default-base-position required in real mode")

    # --- Subprocess-only paths (no ROS node needed) ---

    if args.move_to_safe_height:
        success, output_text = run_move_to_safe_height()
        subprocess_json = extract_json_from_output(output_text)
        if subprocess_json:
            subprocess_json["movement_type"] = "move_to_safe_height"
            output_result(subprocess_json)
        else:
            output_result({
                "result": "success" if success else "failure",
                "mode": args.mode,
                "movement_type": "move_to_safe_height",
                **({"error": "move_to_safe_height failed"} if not success else {}),
            })
        sys.exit(0 if success else 1)

    if args.move_away_from_base:
        success, output_text = run_move_to_clear_area()
        subprocess_json = extract_json_from_output(output_text)
        if subprocess_json:
            subprocess_json["movement_type"] = "move_away_from_base"
            output_result(subprocess_json)
        else:
            output_result({
                "result": "success" if success else "failure",
                "mode": args.mode,
                "movement_type": "move_away_from_base",
                **({"error": "move_to_clear_area failed"} if not success else {}),
            })
        sys.exit(0 if success else 1)

    if args.perform_insert and args.mode == 'real':
        success, output_text = run_perform_insert_real(args)
        subprocess_json = extract_json_from_output(output_text)
        if subprocess_json:
            subprocess_json["movement_type"] = "perform_insert"
            output_result(subprocess_json)
        else:
            result = {
                "result": "success" if success else "failure",
                "mode": "real",
                "movement_type": "perform_insert",
            }
            if not success:
                result["error"] = "perform_insert failed"
            output_result(result)
        sys.exit(0 if success else 1)

    # --- ROS node paths (sim move-to-base, sim perform-insert, real move-to-base) ---

    rclpy.init()
    node = None
    success = False
    error = None

    try:
        node = TranslateObject(mode=args.mode)
        node.action_client.wait_for_server()

        # Wait for EE pose (always needed)
        while node.current_ee_pose is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.1)

        # In sim mode, wait for object and base poses
        if args.mode == 'sim':
            while not node.current_poses:
                rclpy.spin_once(node, timeout_sec=0.1)
                time.sleep(0.1)

        if args.move_to_base:
            if args.mode == 'sim':
                success = node.translate_for_target_sim(args.object_name, args.base_name, hover=True)
            else:
                success = node.translate_for_target_real(
                    args.object_name, args.base_name,
                    final_base_pos=args.final_base_pos,
                    final_base_orientation=args.final_base_orientation,
                    use_default_base=args.use_default_base_position,
                    grasp_id=args.grasp_id,
                    object_orientation=args.current_object_orientation,
                )
        elif args.perform_insert:
            # Sim mode only (real mode handled above as subprocess)
            success = node.translate_for_target_sim(args.object_name, args.base_name, hover=False)

        if success:
            node.get_logger().info("Operation completed successfully!")
        else:
            error = node.error_message

    except KeyboardInterrupt:
        error = "Interrupted by user"
    except Exception as e:
        error = str(e)
    finally:
        # Determine movement type for JSON output
        if args.move_to_base:
            movement_type = "move_to_base"
        elif args.perform_insert:
            movement_type = "perform_insert"
        else:
            movement_type = "unknown"

        result = {
            "result": "success" if success else "failure",
            "mode": args.mode,
            "movement_type": movement_type,
        }
        if args.object_name:
            result["object_name"] = args.object_name
        if args.base_name:
            result["base_name"] = args.base_name
        if not success:
            result["error"] = error or (node.error_message if node else "Unknown error")

        output_result(result)

        try:
            if node:
                node.action_client.destroy()
                node.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
