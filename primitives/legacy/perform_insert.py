#!/usr/bin/env python3
"""
Force Compliant Move Down Controller
=====================================
Moves down continuously with fixed orientation.

Search Phase (before contact):
- Only Z moves down (no X/Y adjustment)
- X and Y positions maintained at starting values

Alignment Phase (after contact detected via Z force threshold):
- Z moves slowly (10% speed)
- X and Y positions adjusted based on force compliance to minimize lateral forces
- Ideal for peg-in-hole insertion tasks

Usage:
    python3 perform_insert.py
    python3 perform_insert.py --speed 0.01 --z-threshold -10.0
"""

import sys
import os

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Wrench
from geometry_msgs.msg import WrenchStamped
from builtin_interfaces.msg import Duration
from scipy.spatial.transform import Rotation as R

import numpy as np
import time
import argparse
from primitives.utils.ik_solver import compute_cartesian_waypoints_ik
from tf2_msgs.msg import TFMessage
import json
import os
import glob
from primitives.utils.data_path_finder import get_assembly_data_dir, get_aruco_data_dir, get_symmetry_dir
from primitives.utils.workspace_config import TABLE_HEIGHT, GRIPPER_CENTER_TOOL_OFFSET, DEFAULT_BASE_POSITION, DEFAULT_BASE_ORIENTATION


# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())


def output_result(result):
    """Output JSON result with markers for MCP server parsing"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


def find_assembly_json_by_base_name(base_name, data_dir=ASSEMBLY_DATA_DIR, logger=None):
    """
    Find the assembly JSON file that contains the given base name.
    
    Args:
        base_name: Name of the base object to search for
        data_dir: Directory to search for JSON files
        logger: Optional logger for debug output
        
    Returns:
        Path to the matching JSON file, or None if not found
    """
    if not os.path.exists(data_dir):
        if logger:
            logger.error(f"Data directory not found: {data_dir}")
        return None
    
    # Search for all JSON files in the data directory
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    # Try exact match first, then with _scaled70 suffix
    base_name_variants = [base_name, f"{base_name}_scaled70"]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)
            
            # Check if any component matches the base name
            components = config.get('components', [])
            for component in components:
                comp_name = component.get('name', '')
                if comp_name in base_name_variants:
                    if logger:
                        logger.info(f"Found assembly JSON for base '{base_name}': {json_file}")
                    return json_file
        except (json.JSONDecodeError, IOError) as e:
            # Skip invalid JSON files
            if logger:
                logger.debug(f"Skipping invalid JSON file {json_file}: {e}")
            continue
    
    if logger:
        logger.warn(f"No assembly JSON found for base '{base_name}' in {data_dir}")
    return None


def load_grasp_point_position(object_name, grasp_id, logger=None):
    """Load grasp point position from grasp points JSON file.

    Returns:
        np.array([x, y, z]) position relative to object CAD center, or None if not found
    """
    data_dir = get_aruco_data_dir() / "grasp_points"
    for name_variant in [object_name, f"{object_name}_scaled70"]:
        json_path = data_dir / f"{name_variant}_grasp_points.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                for gp in data.get('grasp_points', []):
                    if gp['id'] == grasp_id:
                        pos = gp['position']
                        if logger:
                            logger.info(f"Loaded grasp point {grasp_id} for {name_variant}: [{pos['x']:.4f}, {pos['y']:.4f}, {pos['z']:.4f}]")
                        return np.array([pos['x'], pos['y'], pos['z']])
                if logger:
                    logger.warn(f"Grasp ID {grasp_id} not found in {json_path.name}")
            except (json.JSONDecodeError, IOError, KeyError) as e:
                if logger:
                    logger.error(f"Error reading {json_path}: {e}")
    if logger:
        logger.error(f"No grasp points file found for '{object_name}'")
    return None


class PerformInsertController(Node):
    """
    Performs insert operation with force compliance.
    Moves down continuously with fixed orientation.
    Adjusts X and Y positions based on force compliance.
    Z-direction moves down regardless of force.
    """
    
    def __init__(self, mode=None, object_name=None, base_name=None, grasp_id=None,
                 yaw_compliant=True, max_retries=3,
                 final_base_pos=None, final_base_orientation=None,
                 use_default_base=False, object_orientation=None):
        super().__init__('perform_insert_controller')

        # Mode must be explicitly specified
        if mode is None:
            raise ValueError("Mode must be explicitly specified. Use 'sim' or 'real'.")
        if mode not in ['sim', 'real']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'sim' or 'real'.")

        self.mode = mode  # 'sim' or 'real'

        # Error tracking for JSON output
        self.error_message = None

        # Store object and base names
        self.object_name = object_name
        self.base_name = base_name
        self.grasp_id = grasp_id

        # Real mode: grasp-adjusted target params
        self.final_base_pos = final_base_pos
        self.final_base_orientation = final_base_orientation
        self.use_default_base = use_default_base
        self.object_orientation = object_orientation

        # For sim mode, we need object and base names
        if self.mode == 'sim':
            if object_name is None or base_name is None:
                raise ValueError("In sim mode, object_name and base_name are required")
            # Find and load assembly config for sim mode based on base_name
            assembly_json_file = find_assembly_json_by_base_name(base_name, ASSEMBLY_DATA_DIR, self.get_logger())
            if assembly_json_file:
                self.assembly_config = self.load_assembly_config(assembly_json_file)
            else:
                self.get_logger().error(f"Could not find assembly JSON for base '{base_name}'")
                self.assembly_config = {}
            # Subscribe to object poses topic
            self.current_poses = {}
            self.object_sub = self.create_subscription(TFMessage, "/objects_poses_sim", self.object_callback, 10)
        else:
            self.assembly_config = None
            self.current_poses = {}
            self.object_sub = None
        
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        # Real mode: insert_compliance params
        self.yaw_compliant = yaw_compliant
        self.max_retries = max_retries

        # Current state
        self.pos = None             # [x, y, z] in meters
        self.rpy = None             # [r, p, y] in degrees
        self.quat = None            # [x, y, z, w] quaternion (raw from topic, for sim mode)
        self.joints = None
        self.force = None           # [fx, fy, fz] in N
        self.wrench = None          # Wrench message (force + torque)
        
        # Setup
        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        # Configure QoS to match the publisher (VOLATILE durability)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,  # Match UR driver publisher QoS
            depth=10
        )

        # Store QoS profile for potential resubscription
        self.pose_qos_profile = qos_profile
        self.pose_subscription = None
        self.joint_subscription = None
        self.force_subscription = None

        # Create initial subscriptions
        self._create_subscriptions()

        # Wait for data — only needed in sim mode (real mode subprocesses to insert_compliance)
        if self.mode == 'sim':
            max_retries = 5
            timeout_sec = 15.0

            for attempt in range(max_retries):
                self.get_logger().info(f"Waiting for initial data (attempt {attempt + 1}/{max_retries})...")
                start_time = time.time()

                while rclpy.ok():
                    elapsed = time.time() - start_time

                    have_all_data = (self.pos is not None and
                                   self.joints is not None and
                                   self.quat is not None)

                    if have_all_data:
                        self.get_logger().info(f"Received all initial data after {elapsed:.1f}s")
                        break

                    if elapsed > timeout_sec:
                        self.get_logger().warn(f"Timeout waiting for data after {timeout_sec}s")
                        missing = []
                        if self.pos is None:
                            missing.append("pose")
                        if self.joints is None:
                            missing.append("joints")
                        if self.quat is None:
                            missing.append("quaternion")
                        self.get_logger().warn(f"Missing: {', '.join(missing)}")

                        if attempt < max_retries - 1:
                            self.get_logger().info("Recreating subscriptions...")
                            self._destroy_subscriptions()
                            time.sleep(0.5)
                            self._create_subscriptions()
                        break

                    rclpy.spin_once(self, timeout_sec=0.1)

                if self.pos is not None and self.joints is not None and self.quat is not None:
                    break
            else:
                self.get_logger().error(f"Failed to receive initial data after {max_retries} attempts")
                raise RuntimeError(f"Failed to receive initial data after {max_retries} attempts.")
        
        # Store grasp_id
        self.grasp_id = grasp_id
        
        self.get_logger().info(f"Using {self.mode.upper()} mode")
    
    def _canonicalize_euler(self, orientation):
        """
        Always canonicalize Euler angles to [0, 180, yaw] format.
        This is the same format used by move_down_compliant and works better with IK solver.
        """
        roll, pitch, yaw = orientation
        
        # Convert current RPY to quaternion
        current_rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        
        # Create base rotation [0, 180, 0] (face down)
        base_rot = R.from_euler('xyz', [0, 180, 0], degrees=True)
        
        # Find the rotation needed to go from base to current orientation
        relative_rot = current_rot * base_rot.inv()
        
        # Extract yaw from the relative rotation
        relative_rpy = relative_rot.as_euler('xyz', degrees=True)
        canonical_yaw = relative_rpy[2]
        
        # Normalize yaw to [-180, 180] range
        while canonical_yaw > 180:
            canonical_yaw -= 360
        while canonical_yaw < -180:
            canonical_yaw += 360
        
        return np.array([0.0, 180.0, canonical_yaw])
    
    def _pose_cb(self, msg):
        self.pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        # Store raw quaternion (used in sim mode to maintain exact orientation)
        self.quat = np.array(q)
        raw_rpy = np.degrees(R.from_quat(q).as_euler('xyz'))
        # Canonicalize to [0, 180, yaw] format
        self.rpy = self._canonicalize_euler(raw_rpy)
    
    def _joint_cb(self, msg):
        # Extract joint angles in the correct order (must match DH parameters)
        if len(msg.name) >= 6 and len(msg.position) >= 6:
            joint_dict = dict(zip(msg.name, msg.position))
            # Map joint names to positions in correct order
            ordered_positions = []
            for joint_name in self.joint_names:
                if joint_name in joint_dict:
                    ordered_positions.append(joint_dict[joint_name])

            if len(ordered_positions) == 6:
                self.joints = np.array(ordered_positions)

    def _wrench_cb(self, msg):
        """Callback for force/torque sensor data"""
        self.wrench = msg.wrench

    def object_callback(self, msg):
        """Callback for object poses (sim mode only)"""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            self.current_poses[frame_id] = transform
    
    def load_assembly_config(self, json_file):
        """Load assembly configuration from JSON file"""
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.get_logger().error(f"Assembly file not found: {json_file}")
            return {}
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Error parsing assembly JSON: {e}")
            return {}
    
    def get_object_target_position(self, object_name, assembly_config=None):
        """Get target position for object from assembly configuration"""
        config = assembly_config if assembly_config is not None else self.assembly_config
        for component in config.get('components', []):
            if component.get('name') == object_name or component.get('name') == f"{object_name}_scaled70":
                position = component.get('position', {})
                return np.array([position.get('x', 0), position.get('y', 0), position.get('z', 0)])
        return None

    def get_object_target_orientation(self, object_name, assembly_config=None):
        """Get target orientation for object from assembly configuration"""
        config = assembly_config if assembly_config is not None else self.assembly_config
        for component in config.get('components', []):
            comp_name = component.get('name', '')
            if comp_name == object_name or comp_name == f"{object_name}_scaled70":
                rotation = component.get('rotation', {})
                quat = rotation.get('quaternion', {})
                return np.array([
                    quat.get('x', 0.0),
                    quat.get('y', 0.0),
                    quat.get('z', 0.0),
                    quat.get('w', 1.0),
                ])
        return None

    def transform_to_matrix(self, transform):
        """Convert ROS Transform to 4x4 transformation matrix"""
        t = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
        q = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
        r = R.from_quat(q)
        rotation_matrix = r.as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = t
        return T
    
    def pose_to_matrix(self, pose):
        """Convert ROS Pose to 4x4 transformation matrix"""
        t = np.array([pose.position.x, pose.position.y, pose.position.z])
        q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        r = R.from_quat(q)
        rotation_matrix = r.as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = t
        return T
    
    def matrix_to_rpy(self, T):
        """Convert 4x4 transformation matrix to position and RPY (degrees)"""
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        r = R.from_matrix(rotation_matrix)
        rpy_rad = r.as_euler('xyz')
        rpy_deg = np.degrees(rpy_rad)
        return position, rpy_deg

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

    def check_collision_with_table(self, joint_angles, z_threshold=TABLE_HEIGHT, verbose=False):
        """
        Check if any part of the robot (all joints) goes below the table.

        Args:
            joint_angles: Array of 6 joint angles
            z_threshold: Minimum allowed Z position (meters).
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

    def check_ee_below_base(self, joint_angles, z_threshold=0.1625, verbose=False):
        """
        Check if the end effector goes below the robot base height.

        The UR5e base is at Z=0, and the shoulder (joint 1) is at Z=162.5mm.
        We don't want the EE to go below this height as it could collide with
        the robot's own base or mounting structure.

        Args:
            joint_angles: Array of 6 joint angles
            z_threshold: Minimum allowed Z position for EE (meters).
                        Default 0.1625m (162.5mm) = robot base height.
            verbose: If True, log details

        Returns:
            True if EE is below threshold (violation), False otherwise
        """
        joint_positions = self.compute_all_joint_positions(joint_angles)

        # EE position is the last joint position
        ee_pos = joint_positions[-1]
        ee_z = ee_pos[2]

        if ee_z < z_threshold:
            if verbose:
                self.get_logger().warn(
                    f"EE below robot base: EE Z={ee_z*1000:.1f}mm "
                    f"(threshold: {z_threshold*1000:.1f}mm)"
                )
            return True  # Violation detected

        return False  # No violation

    def check_compact_configuration(self, joint_angles, min_wrist_shoulder_xy=0.20, verbose=False):
        """
        Check if the robot configuration is too compact (wrist too close to shoulder).

        Args:
            joint_angles: Array of 6 joint angles
            min_wrist_shoulder_xy: Minimum allowed XY distance (meters) between
                                   wrist2 and shoulder.
            verbose: If True, log details

        Returns:
            True if configuration is too compact (should be rejected), False otherwise
        """
        joint_positions = self.compute_all_joint_positions(joint_angles)

        shoulder_pos = np.array(joint_positions[1])
        wrist2_pos = np.array(joint_positions[5])

        xy_dist = np.linalg.norm(wrist2_pos[:2] - shoulder_pos[:2])

        if xy_dist < min_wrist_shoulder_xy:
            if verbose:
                self.get_logger().warn(
                    f"Compact configuration detected: wrist-shoulder XY distance="
                    f"{xy_dist*1000:.1f}mm < threshold={min_wrist_shoulder_xy*1000:.1f}mm"
                )
            return True  # Too compact

        return False  # Configuration OK

    def read_current_joint_angles(self):
        """Read current joint angles using ROS2 subscriber"""
        timeout_count = 0
        max_timeout = 100
        while rclpy.ok() and timeout_count < max_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout_count += 1
        return self.joints.copy() if self.joints is not None else None
    
    def execute_trajectory_sim(self, trajectory):
        """Execute trajectory and wait for completion (sim mode)"""
        try:
            if 'traj1' not in trajectory or not trajectory['traj1']:
                return False

            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names

            for point in trajectory['traj1']:
                traj_point = JointTrajectoryPoint()
                traj_point.positions = point['positions']
                if 'velocities' in point:
                    traj_point.velocities = point['velocities']
                traj_point.time_from_start = point['time_from_start']
                traj_msg.points.append(traj_point)
            
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)
            
            future = self.traj_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            
            if not goal_handle.accepted:
                self.error_message = "External control program stopped or robot in protective stop"
                self.get_logger().error(self.error_message)
                return False

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result()

            if result.status == 4:  # SUCCEEDED
                return True
            else:
                self.error_message = f"Trajectory failed with status code {result.status}"
                self.get_logger().error(self.error_message)
                return False
        except Exception as e:
            self.get_logger().error(f"Trajectory execution error: {e}")
            return False
    
    def check_grasp(self, object_name, radius=0.06):
        """Check if object is within grasp radius of gripper center.

        Returns:
            (True, distance) if grasped, (False, distance) if not
        """
        obj_key = object_name if object_name in self.current_poses else f"{object_name}_scaled70"
        if obj_key not in self.current_poses:
            return False, float('inf')

        transform = self.current_poses[obj_key].transform
        object_pos = np.array([transform.translation.x, transform.translation.y, transform.translation.z])

        gripper_center = self.pos + R.from_quat(self.quat).as_matrix() @ GRIPPER_CENTER_TOOL_OFFSET
        distance = np.linalg.norm(object_pos - gripper_center)
        return distance <= radius, distance

    def move_down_sim(self, duration=10.0):
        """Sim mode: Move down to final position (step 2 from translate_for_assembly)"""
        # Wait for pose data
        if not self.current_poses:
            self.get_logger().error("No pose data available")
            return False
        
        # Get current EE pose
        if self.pos is None:
            self.get_logger().error("End-effector pose not available")
            return False
        
        # Check if object exists
        obj_key = self.object_name if self.object_name in self.current_poses else f"{self.object_name}_scaled70"
        if obj_key not in self.current_poses:
            self.error_message = f"Object {self.object_name} not found"
            self.get_logger().error(self.error_message)
            return False

        # Check if base exists
        base_key = self.base_name if self.base_name in self.current_poses else f"{self.base_name}_scaled70"
        if base_key not in self.current_poses:
            self.error_message = f"Base {self.base_name} not found"
            self.get_logger().error(self.error_message)
            return False
        
        # Verify grasp before executing insertion
        grasped, dist = self.check_grasp(self.object_name)
        if not grasped:
            self.error_message = f"Grasp check failed: {self.object_name} is {dist*1000:.1f}mm from gripper center."
            self.get_logger().error(self.error_message)
            return False
        self.get_logger().info(f"Grasp verified: {self.object_name} is {dist*1000:.1f}mm from gripper center")

        # Get current EE pose as PoseStamped-like structure
        # Use actual current EE orientation from topic (maintain current orientation like old code)
        if self.quat is None:
            self.get_logger().error("EE quaternion not available from topic")
            return False

        current_ee_pose_msg = PoseStamped()
        current_ee_pose_msg.pose.position.x = self.pos[0]
        current_ee_pose_msg.pose.position.y = self.pos[1]
        current_ee_pose_msg.pose.position.z = self.pos[2]

        # Use raw quaternion from topic to maintain exact orientation (not canonicalized RPY)
        current_ee_pose_msg.pose.orientation.x = self.quat[0]
        current_ee_pose_msg.pose.orientation.y = self.quat[1]
        current_ee_pose_msg.pose.orientation.z = self.quat[2]
        current_ee_pose_msg.pose.orientation.w = self.quat[3]
        
        # Convert poses to matrices
        T_EE_current = self.pose_to_matrix(current_ee_pose_msg.pose)
        T_object_current = self.transform_to_matrix(self.current_poses[obj_key].transform)
        T_base_current = self.transform_to_matrix(self.current_poses[base_key].transform)
        
        # Calculate grasp transformation
        T_grasp = np.linalg.inv(T_EE_current) @ T_object_current
        
        # Get current positions
        ee_current_position, ee_current_rpy = self.matrix_to_rpy(T_EE_current)
        base_current_position, base_current_rpy = self.matrix_to_rpy(T_base_current)
        
        # Get target object position from JSON (relative to base)
        target_position_relative = self.get_object_target_position(self.object_name)
        if target_position_relative is None:
            self.error_message = f"No target position found for {self.object_name} in JSON"
            self.get_logger().error(self.error_message)
            return False
        
        # Transform target position from base frame to world frame
        R_base_current = T_base_current[:3, :3]
        target_object_position_abs = base_current_position + R_base_current @ target_position_relative
        
        # Create target object transformation (keep current orientation)
        T_object_target = np.eye(4)
        T_object_target[:3, :3] = T_object_current[:3, :3]  # Keep current orientation
        T_object_target[:3, 3] = target_object_position_abs
        
        # Calculate required EE position to place object at target
        T_EE_target = T_object_target @ np.linalg.inv(T_grasp)
        
        # Extract target position and quaternion
        ee_target_position = T_EE_target[:3, 3]
        ee_target_rot_matrix = T_EE_target[:3, :3]
        ee_target_rotation = R.from_matrix(ee_target_rot_matrix)
        ee_target_quat = ee_target_rotation.as_quat()
        
        
        # Read current joint angles before computing IK
        if self.joints is None:
            self.read_current_joint_angles()
            if self.joints is None:
                self.get_logger().error("Could not read current joint angles")
                return False

        # Compute IK waypoints using Jacobian-based differential IK
        num_waypoints = 60
        total_duration = 5.0

        self.get_logger().info("Computing Cartesian IK waypoints...")
        ik_waypoints = compute_cartesian_waypoints_ik(
            np.array(self.joints, dtype=float),
            target_z=None,
            num_waypoints=num_waypoints,
            target_pos=ee_target_position,
            target_orientation=ee_target_rot_matrix,
        )
        if ik_waypoints is None:
            self.error_message = "IK failed for Cartesian waypoints"
            self.get_logger().error(self.error_message)
            return False

        # Prepend current joints as waypoint 0 for smooth ramp-up
        start_joints = np.array([float(x) for x in self.joints])
        all_joint_angles = [start_joints] + [np.array([float(x) for x in w]) for w in ik_waypoints]

        # Trapezoidal velocity profile (arc-length parameterized)
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

        trajectory_points = []
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

            point = {
                "positions": [float(x) for x in all_joint_angles[i]],
                "velocities": velocities,
                "time_from_start": Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
            }
            trajectory_points.append(point)

        self.get_logger().info(
            f"Trajectory created with {len(trajectory_points)} waypoints "
            f"({total_duration:.1f}s, trapezoidal velocity profile)"
        )

        success = self.execute_trajectory_sim({"traj1": trajectory_points})

        if success:
            time.sleep(0.5)
            self.get_logger().info("Movement completed successfully")
        else:
            self.get_logger().error("Failed to reach final position")

        return success
    
    def _create_subscriptions(self):
        """Create all subscriptions"""
        self.pose_subscription = self.create_subscription(
            PoseStamped, '/tcp_pose_broadcaster/pose', self._pose_cb, self.pose_qos_profile
        )
        self.joint_subscription = self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 10
        )
        self.force_subscription = self.create_subscription(
            WrenchStamped, '/force_torque_sensor_broadcaster/wrench', self._wrench_cb, 10
        )

        self.force = np.array([0.0, 0.0, 0.0])  # Dummy force (real mode uses insert_compliance subprocess)

    def _destroy_subscriptions(self):
        """Destroy all subscriptions to allow recreation"""
        if self.pose_subscription is not None:
            self.destroy_subscription(self.pose_subscription)
            self.pose_subscription = None
        if self.joint_subscription is not None:
            self.destroy_subscription(self.joint_subscription)
            self.joint_subscription = None
        if self.force_subscription is not None:
            self.destroy_subscription(self.force_subscription)
            self.force_subscription = None

    def wait_for_poses_with_retry(self, timeout_per_attempt=5.0, max_retries=3):
        """
        Wait for object poses from /objects_poses_sim topic with retry logic.

        Args:
            timeout_per_attempt: Timeout in seconds for each attempt
            max_retries: Maximum number of retry attempts

        Returns:
            True if poses received, False if all retries exhausted
        """
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                self.get_logger().info(f"Retry attempt {attempt}/{max_retries} - waiting for object poses...")
                time.sleep(0.5)  # Small delay between retries

            start_time = time.time()
            while rclpy.ok() and (time.time() - start_time) < timeout_per_attempt:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.current_poses:
                    self.get_logger().info(f"Object poses received after {attempt} attempt(s)")
                    return True
                time.sleep(0.1)

            if attempt < max_retries:
                self.get_logger().warn(
                    f"Timeout waiting for object poses on /objects_poses_sim topic "
                    f"(attempt {attempt}/{max_retries}, waited {timeout_per_attempt:.1f}s)"
                )

        # All retries exhausted
        self.error_message = (
            f"Failed to receive object poses from /objects_poses_sim topic after {max_retries} attempts "
            f"(timeout: {timeout_per_attempt:.1f}s per attempt). "
            f"Ensure the simulation is running and publishing object poses."
        )
        self.get_logger().error(self.error_message)
        return False

    def run(self):
        """Main run method - different behavior for sim vs real mode"""
        if self.mode == 'sim':
            # Sim mode: execute step 2 (move down to final position)
            # Wait for object and base poses from topics with retry logic
            if not self.wait_for_poses_with_retry(timeout_per_attempt=5.0, max_retries=3):
                return False
            return self.move_down_sim(duration=10.0)
        else:
            # Real mode: original force-compliant movement
            return self.run_real()
    
    def _compute_grasp_adjusted_target_z(self):
        """Compute grasp-adjusted target Z using assembly config and grasp points.

        Returns the target Z for insert_compliance, or None on error.
        """
        from primitives.rotate_object import FoldSymmetry, ExtendedCardinalOrientations

        if self.grasp_id is None:
            self.error_message = "grasp_id is required for real mode insertion"
            self.get_logger().error(self.error_message)
            return None
        if self.object_name is None:
            self.error_message = "object_name is required for real mode insertion"
            self.get_logger().error(self.error_message)
            return None
        if self.base_name is None:
            self.error_message = "base_name is required for real mode insertion"
            self.get_logger().error(self.error_message)
            return None
        if self.object_orientation is None:
            self.error_message = "current_object_orientation is required for real mode insertion"
            self.get_logger().error(self.error_message)
            return None

        # Load assembly config
        assembly_json = find_assembly_json_by_base_name(self.base_name, ASSEMBLY_DATA_DIR, self.get_logger())
        if not assembly_json:
            self.error_message = f"No assembly JSON found for base '{self.base_name}'"
            self.get_logger().error(self.error_message)
            return None

        assembly_config = self.load_assembly_config(assembly_json)
        target_pos_relative = self.get_object_target_position(self.object_name, assembly_config)
        if target_pos_relative is None:
            self.error_message = f"No target position found for '{self.object_name}' in assembly JSON"
            self.get_logger().error(self.error_message)
            return None

        target_ori_relative = self.get_object_target_orientation(self.object_name, assembly_config)

        # Determine base position and orientation
        if self.use_default_base:
            base_pos = np.array(DEFAULT_BASE_POSITION)
            base_ori_quat = np.array(DEFAULT_BASE_ORIENTATION)
        elif self.final_base_pos is not None and self.final_base_orientation is not None:
            base_pos = np.array(self.final_base_pos)
            base_ori_quat = np.array(self.final_base_orientation)
        else:
            self.error_message = "No base position provided (use --use-default-base or --final-base-pos/--final-base-orientation)"
            self.get_logger().error(self.error_message)
            return None

        # Transform target position to world frame
        R_base = R.from_quat(base_ori_quat).as_matrix()
        target_object_position_abs = base_pos + R_base @ target_pos_relative

        # Transform target orientation to world frame
        if target_ori_relative is not None:
            R_target_abs = R_base @ R.from_quat(target_ori_relative).as_matrix()
        else:
            R_target_abs = R_base

        # Load grasp offset
        grasp_offset = load_grasp_point_position(self.object_name, self.grasp_id, self.get_logger())
        if grasp_offset is None:
            self.error_message = f"Could not load grasp point {self.grasp_id} for '{self.object_name}'"
            self.get_logger().error(self.error_message)
            return None

        # Determine rotation to apply to grasp offset using fold symmetry
        R_object_current = R.from_quat(self.object_orientation).as_matrix()
        symmetry_dir = str(get_symmetry_dir())
        fold_data = FoldSymmetry.load_symmetry_data(self.object_name, symmetry_dir)

        if fold_data is not None:
            equivalents = FoldSymmetry.generate_equivalent_target_orientations(
                R_target_abs, fold_data, logger=self.get_logger()
            )
            best_dist = float('inf')
            R_grasp_rotation = R_target_abs
            for R_eq in equivalents:
                dist = ExtendedCardinalOrientations.rotation_matrix_distance(R_object_current, R_eq)
                if dist < best_dist:
                    best_dist = dist
                    R_grasp_rotation = R_eq
            self.get_logger().info(f"Snapped object orientation to closest equivalent (error: {np.degrees(best_dist):.1f}°)")
        else:
            self.get_logger().info("No symmetry data found, using current object orientation directly")
            R_grasp_rotation = R_object_current

        # Compute EE target position
        grasp_world_offset = R_grasp_rotation @ grasp_offset
        ee_target_position = target_object_position_abs + grasp_world_offset

        # Wait for EE pose to get current orientation
        timeout = 3.0
        start = time.time()
        while self.pos is None and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.quat is not None:
            R_ee = R.from_quat(self.quat).as_matrix()
            ee_target_position -= R_ee @ GRIPPER_CENTER_TOOL_OFFSET
        else:
            self.error_message = "Could not read EE pose (timeout waiting for /tcp_pose_broadcaster/pose)"
            self.get_logger().error(self.error_message)
            return None

        # Extract full target position (x, y, z) for insertion
        target_z = ee_target_position[2]
        self.get_logger().info(f"Computed grasp-adjusted target position: {target_object_position_abs}")
        self.get_logger().info(f"  Target Z for insertion: {target_z:.4f}m")
        self.get_logger().info(f"  Grasp offset (world): {grasp_world_offset}")
        # Return (target_x, target_y, target_z) so insertion moves to correct hole location
        return (target_object_position_abs[0], target_object_position_abs[1], target_z)

    def _perform_xy_correction(self):
        """Apply Rx/Ry compliance to correct residual tilt after insertion.

        If high_torque_detected during insertion, the object may still have
        some tilt. This function keeps Rx/Ry compliance active and monitors
        Tx/Ty to verify the object is settling into alignment.

        Returns:
            True if torques stabilize, False on timeout/error
        """
        try:
            self.get_logger().info("Applying Rx/Ry compliance to correct residual tilt...")

            # Enable XYZ+Rx/Ry compliance to let object self-correct
            from ur_msgs.srv import SetForceMode
            from geometry_msgs.msg import Wrench, Twist
            import subprocess
            import re

            # Step 1: Reactivate controllers (insert_compliance may have deactivated them)
            def get_active_controllers():
                """Return set of active controller names."""
                result = subprocess.run(
                    ['ros2', 'control', 'list_controllers'],
                    capture_output=True, text=True, timeout=10
                )
                active = set()
                ansi_re = re.compile(r'\x1b\[[0-9;]*m')
                for line in result.stdout.splitlines():
                    clean = ansi_re.sub('', line)
                    parts = clean.split()
                    if len(parts) >= 3 and 'active' in parts:
                        active.add(parts[0])
                return active

            def switch_controllers(activate_list, deactivate_list):
                """Switch controllers."""
                active = get_active_controllers()
                to_activate = [c for c in activate_list if c not in active]
                to_deactivate = [c for c in deactivate_list if c in active]
                if not to_activate and not to_deactivate:
                    return True
                cmd = ['ros2', 'control', 'switch_controllers']
                if to_activate:
                    cmd += ['--activate'] + to_activate
                if to_deactivate:
                    cmd += ['--deactivate'] + to_deactivate
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                return result.returncode == 0

            # Activate passthrough + force mode, deactivate scaled
            switch_controllers(
                ['passthrough_trajectory_controller', 'force_mode_controller'],
                ['scaled_joint_trajectory_controller']
            )
            time.sleep(0.5)

            # Setup force mode client
            force_mode_client = self.create_client(SetForceMode, '/force_mode_controller/start_force_mode')
            if not force_mode_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn("Force mode service unavailable, using passive settling only")
                time.sleep(2.0)
                return True

            # Start XYZ+Rx/Ry compliance with downward force
            req = SetForceMode.Request()
            req.task_frame.header.frame_id = 'base_link'
            req.task_frame.pose.orientation.w = 1.0
            req.selection_vector_x = True
            req.selection_vector_y = True
            req.selection_vector_z = True
            req.selection_vector_rx = True
            req.selection_vector_ry = True
            req.selection_vector_rz = False
            req.wrench = Wrench()
            req.wrench.force.z = -5.0  # maintain contact force
            req.type = 2  # NO_TRANSFORM
            req.speed_limits = Twist()
            req.speed_limits.linear.x = 0.005
            req.speed_limits.linear.y = 0.005
            req.speed_limits.linear.z = 0.03
            req.speed_limits.angular.x = 0.1
            req.speed_limits.angular.y = 0.1
            req.speed_limits.angular.z = 0.5
            req.damping_factor = 0.0
            req.gain_scaling = 1.0

            future = force_mode_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            resp = future.result()
            if resp is None or not resp.success:
                self.get_logger().warn("Failed to enable compliance, using passive settling")
                time.sleep(2.0)
                return True

            self.get_logger().info("XYZ+Rx/Ry compliance active, monitoring Tx/Ty for settling...")

            # Monitor Tx/Ty for stabilization
            start_time = time.time()
            SETTLING_TIMEOUT = 5.0
            LOW_TORQUE_THRESHOLD = 0.1  # Nm
            SUSTAIN_TIME = 1.0  # seconds
            low_torque_start = None
            last_log = 0.0

            while (time.time() - start_time) < SETTLING_TIMEOUT:
                rclpy.spin_once(self, timeout_sec=0.1)

                if self.wrench is not None:
                    tx = abs(self.wrench.torque.x)
                    ty = abs(self.wrench.torque.y)
                    elapsed = time.time() - start_time

                    # Log every 1 second
                    if elapsed - last_log >= 1.0:
                        last_log = elapsed
                        self.get_logger().info(f"Settling: Tx={tx:.4f} Ty={ty:.4f} ({elapsed:.1f}s)")

                    # Check if torques are low
                    if tx < LOW_TORQUE_THRESHOLD and ty < LOW_TORQUE_THRESHOLD:
                        if low_torque_start is None:
                            low_torque_start = time.time()
                        elif (time.time() - low_torque_start) >= SUSTAIN_TIME:
                            self.get_logger().info(
                                f"Tilt corrected! Tx={tx:.4f} Ty={ty:.4f} stable for {SUSTAIN_TIME:.1f}s"
                            )
                            return True
                    else:
                        low_torque_start = None

            self.get_logger().warn(f"Settling timeout after {SETTLING_TIMEOUT}s")
            return True  # Still consider it success (object is in hole)

        except Exception as e:
            self.get_logger().error(f"Tilt correction failed: {e}")
            return False

    def run_real(self):
        """Run real-mode insertion via insert_compliance.py subprocess."""
        import subprocess as sp

        result = self._compute_grasp_adjusted_target_z()
        if result is None:
            return False

        # Result is (target_x, target_y, target_z) computed from object pose and grasp point
        if isinstance(result, tuple) and len(result) == 3:
            target_x, target_y, target_z = result
            self.get_logger().info(f"Will insert at target position: [{target_x:.4f}, {target_y:.4f}, {target_z:.4f}]")
        else:
            # Fallback for old code that returns just Z
            target_z = result
            target_x = None
            target_y = None
            self.get_logger().info(f"Inserting at current X/Y. Target Z (grasp-adjusted): {target_z:.4f}")

        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'insert_compliance.py')
        cmd = [
            sys.executable, script_path,
            '--target-z', str(target_z),
        ]
        if target_x is not None:
            cmd.extend(['--target-x', str(target_x)])
        if target_y is not None:
            cmd.extend(['--target-y', str(target_y)])

        self.get_logger().info(f"Running: {' '.join(cmd)}")

        env = os.environ.copy()
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"

        output_lines = []
        process = sp.Popen(
            cmd,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                stripped = line.rstrip()
                output_lines.append(stripped)
                self.get_logger().info(stripped)
        process.wait()

        output_text = '\n'.join(output_lines)

        # Parse JSON result
        if '__RESULT_JSON__' in output_text and '__END_RESULT_JSON__' in output_text:
            json_str = output_text.split('__RESULT_JSON__')[1].split('__END_RESULT_JSON__')[0].strip()
            try:
                result_data = json.loads(json_str)
                if result_data.get('result') == 'success':
                    self.get_logger().info(f"Insertion success: {result_data}")

                    # Post-insertion XY correction if high torque was detected
                    if result_data.get('high_torque_detected', False):
                        self.get_logger().info("High torque was detected during insertion. Performing XY adjustment...")
                        if self._perform_xy_correction():
                            self.get_logger().info("XY correction completed successfully")
                        else:
                            self.get_logger().warn("XY correction failed or not available")

                    return True
                else:
                    self.error_message = result_data.get('error', 'Unknown insertion failure')
                    self.get_logger().error(f"Insertion failed: {self.error_message}")
                    return False
            except json.JSONDecodeError as e:
                self.error_message = f"Failed to parse insertion result: {e}"
                self.get_logger().error(self.error_message)
                return False

        self.error_message = f"No JSON result from insert_compliance.py (exit code {process.returncode})"
        self.get_logger().error(self.error_message)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Perform insertion — sim mode (Cartesian IK) or real mode (peg-in-hole force control)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real mode (peg-in-hole with force mode)
  python3 perform_insert.py --mode real

  # Sim mode (step 2 from translate_for_assembly)
  python3 perform_insert.py --mode sim --object-name fork_orange --base-name base

  # Real mode with yaw compliance disabled
  python3 perform_insert.py --mode real --no-yaw-compliant
        """
    )
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: sim (Cartesian IK move down) or real (peg-in-hole force control)')

    # Sim mode arguments
    parser.add_argument('--object-name', type=str, help='Name of the object being held (required in sim mode)')
    parser.add_argument('--base-name', type=str, help='Name of the base object (required in sim mode)')

    # Real mode arguments
    parser.add_argument('--no-yaw-compliant', action='store_true',
                        help='Disable yaw compliance during insertion (default: enabled)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Max retries on drift (default: 3, real mode only)')
    parser.add_argument('--grasp-id', type=int, default=None,
                        help='Grasp point ID for target Z computation (real mode)')
    parser.add_argument('--final-base-pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Base position [x, y, z] in meters (real mode)')
    parser.add_argument('--final-base-orientation', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'),
                        help='Base orientation quaternion [x, y, z, w] (real mode)')
    parser.add_argument('--use-default-base', action='store_true',
                        help='Use default base position and orientation (real mode)')
    parser.add_argument('--current-object-orientation', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'),
                        help='Current object orientation quaternion [x, y, z, w] (real mode, for fold symmetry)')
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'sim':
        if args.object_name is None or args.base_name is None:
            parser.error("In sim mode, --object-name and --base-name are required")
    
    rclpy.init()

    node = None
    success = False
    error = None

    try:
        if args.mode == 'sim':
            node = PerformInsertController(
                mode=args.mode,
                object_name=args.object_name,
                base_name=args.base_name
            )
        else:
            node = PerformInsertController(
                mode=args.mode,
                object_name=args.object_name,
                base_name=args.base_name,
                grasp_id=args.grasp_id,
                yaw_compliant=not args.no_yaw_compliant,
                max_retries=args.max_retries,
                final_base_pos=args.final_base_pos,
                final_base_orientation=args.final_base_orientation,
                use_default_base=args.use_default_base,
                object_orientation=args.current_object_orientation,
            )

        # Give system time to stabilize
        time.sleep(0.5)

        # Execute trajectory
        try:
            success = node.run()
            if success:
                node.get_logger().info("Insert completed successfully!")
            else:
                node.get_logger().error("Insert failed")
                error = node.error_message
        except KeyboardInterrupt:
            # Set running flag to stop the loop
            if hasattr(node, 'running'):
                node.running = False
            node.get_logger().info("TRAJECTORY INTERRUPTED BY USER")
            error = "Interrupted by user"

    except KeyboardInterrupt:
        error = "Interrupted by user"
    except Exception as e:
        error = str(e)
    finally:
        # Build and output JSON result
        if args.mode == 'sim':
            if success:
                result = {
                    "result": "success",
                    "mode": "sim",
                    "object_name": args.object_name,
                    "base_name": args.base_name
                }
            else:
                result = {
                    "result": "failure",
                    "mode": "sim",
                    "object_name": args.object_name,
                    "base_name": args.base_name,
                    "error": error or (node.error_message if node else "Unknown error")
                }
        else:  # real mode
            if success:
                result = {
                    "result": "success",
                    "mode": "real"
                }
            else:
                result = {
                    "result": "failure",
                    "mode": "real",
                    "error": error or (node.error_message if node else "Unknown error")
                }

        output_result(result)

        try:
            if node:
                node.traj_client.destroy()
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