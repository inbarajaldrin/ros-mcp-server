#!/usr/bin/env python3
"""
Reorient Object - Standalone single object reorientation

This primitive reorients an object held by the end-effector to a target orientation.
It does NOT require assembly JSON files or base object information.

Usage:
    # Using quaternion target (world frame)
    python3 reorient_object.py --mode sim --object-name fork \\
        --target-orientation 0.0 0.0 0.0 1.0

    # Providing current object orientation explicitly (real mode)
    python3 reorient_object.py --mode real --object-name fork \\
        --current-orientation 0.1 0.2 0.3 0.9 \\
        --target-orientation 0.0 0.0 0.7071 0.7071
"""

import sys
import os

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import argparse
import time
import glob

from primitives.utils.ik_solver import ik_objective_quaternion, forward_kinematics, dh_params
from primitives.utils.data_path_finder import get_symmetry_dir

# Configuration
SYMMETRY_DIR = str(get_symmetry_dir())
DEFAULT_OBJECT_TOPIC = "/objects_poses_sim"
DEFAULT_EE_TOPIC = "/tcp_pose_broadcaster/pose"


class ExtendedCardinalOrientations:
    """24 extended cardinal orientations with intermediary angles"""

    @staticmethod
    def get_all_extended_cardinals():
        cardinals = {}

        # Primary face directions (cardinal)
        # NOTE: 'up' (0, 0) is intentionally excluded - this orientation is blocked
        # for safety/mechanical reasons (quaternion [0, 0, 0, 1] or RPY [0, 0, 0])
        primary_directions = {
            'down': (180, 0),
            'forward': (90, 0),
            'backward': (90, 180),
            'right': (90, -90),
        }

        # Intermediary face directions (45° increments)
        intermediary_directions = {
            'forward_right': (90, -45),
            'forward_left': (90, 45),
            'backward_right': (90, -135),
            'backward_left': (90, 135),
        }

        # Roll variations for primary directions (0°, 90°, 180°, 270°)
        roll_angles = [0, 90, 180, 270]

        # Add primary cardinal directions with roll variations (4 × 4 = 16)
        for face_name, (pitch, yaw) in primary_directions.items():
            for roll in roll_angles:
                name = f"face_{face_name}_roll{roll}"
                q = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat()
                cardinals[name] = q

        # Add intermediary directions with 2 roll variations each (4 × 2 = 8)
        intermediary_rolls = [0, 180]
        for face_name, (pitch, yaw) in intermediary_directions.items():
            for roll in intermediary_rolls:
                name = f"face_{face_name}_roll{roll}"
                q = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat()
                cardinals[name] = q

        return cardinals

    @staticmethod
    def rotation_matrix_distance(R1, R2):
        """Angular distance between two rotation matrices in degrees."""
        R_diff = R1.T @ R2
        trace = np.trace(R_diff)
        cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    @staticmethod
    def find_closest_cardinal(R_orientation, threshold_deg=10.0):
        """Find the closest cardinal orientation to the given rotation matrix."""
        cardinals = ExtendedCardinalOrientations.get_all_extended_cardinals()

        best_name = None
        best_quat = None
        best_distance = float('inf')

        for card_name, card_quat in cardinals.items():
            R_cardinal = R.from_quat(card_quat).as_matrix()
            distance = ExtendedCardinalOrientations.rotation_matrix_distance(
                R_orientation, R_cardinal
            )

            if distance < best_distance:
                best_distance = distance
                best_name = card_name
                best_quat = card_quat

        if best_distance <= threshold_deg:
            return (best_name, best_quat, best_distance)
        else:
            return (None, None, best_distance)


class FoldSymmetry:
    """Fold symmetry handling for objects with rotational symmetry."""

    @staticmethod
    def load_symmetry_data(object_name, symmetry_dir):
        """Load fold symmetry JSON"""
        patterns = [
            os.path.join(symmetry_dir, f"{object_name}_symmetry.json"),
            os.path.join(symmetry_dir, f"{object_name}*_symmetry.json"),
            os.path.join(symmetry_dir, f"{object_name.replace('_scaled70', '')}*_symmetry.json"),
        ]

        for pattern in patterns:
            if '*' in pattern:
                matches = glob.glob(pattern)
                if matches:
                    with open(matches[0], 'r') as f:
                        return json.load(f)
            elif os.path.exists(pattern):
                with open(pattern, 'r') as f:
                    return json.load(f)
        return None

    @staticmethod
    def get_symmetry_rotations_as_matrices(fold_data):
        """Extract symmetry rotations as rotation matrices."""
        if fold_data is None:
            return [np.eye(3)]

        symmetry_matrices = []
        seen = set()

        # Always include identity
        symmetry_matrices.append(np.eye(3))
        seen.add(tuple(np.eye(3).flatten().round(6)))

        for axis in ['x', 'y', 'z']:
            if axis not in fold_data.get('fold_axes', {}):
                continue

            axis_data = fold_data['fold_axes'][axis]
            for q_data in axis_data.get('quaternions', []):
                q = np.array([
                    q_data['quaternion']['x'],
                    q_data['quaternion']['y'],
                    q_data['quaternion']['z'],
                    q_data['quaternion']['w']
                ])
                q = q / np.linalg.norm(q)

                R_sym = R.from_quat(q).as_matrix()

                key = tuple(R_sym.flatten().round(6))
                if key not in seen:
                    seen.add(key)
                    symmetry_matrices.append(R_sym)

        return symmetry_matrices

    @staticmethod
    def generate_equivalent_target_orientations(R_target_world, fold_data, logger=None):
        """Generate all symmetry-equivalent target orientations."""
        symmetry_rotations = FoldSymmetry.get_symmetry_rotations_as_matrices(fold_data)

        equivalent_targets = []
        for i, R_sym in enumerate(symmetry_rotations):
            R_equivalent = R_target_world @ R_sym
            equivalent_targets.append(R_equivalent)

        return equivalent_targets


class ReorientObject(Node):
    def __init__(self, mode=None, object_topic=None, ee_topic=DEFAULT_EE_TOPIC):
        super().__init__('reorient_object')

        if mode is None:
            raise ValueError("Mode must be explicitly specified. Use 'sim' or 'real'.")
        if mode not in ['sim', 'real']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'sim' or 'real'.")

        self.mode = mode
        self.symmetry_dir = SYMMETRY_DIR

        # Set default object topic based on mode
        if object_topic is None:
            if self.mode == 'sim':
                object_topic = DEFAULT_OBJECT_TOPIC
            else:
                object_topic = None

        # Subscribe to topics
        if object_topic is not None:
            self.object_sub = self.create_subscription(TFMessage, object_topic,
                                                      self.object_callback, 10)
        else:
            self.object_sub = None

        self.ee_sub = self.create_subscription(PoseStamped, ee_topic, self.ee_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states',
                                                       self.joint_state_callback, 10)

        self.current_poses = {}
        self.current_ee_pose = None
        self.current_joint_angles = None
        self.joint_angles_received = False
        self.trajectory_success = False
        self.trajectory_completed = False

        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                           "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.action_client = ActionClient(self, FollowJointTrajectory,
                                         '/scaled_joint_trajectory_controller/follow_joint_trajectory')

        self.get_logger().info(f"Using {self.mode.upper()} mode")

    def object_callback(self, msg):
        for transform in msg.transforms:
            self.current_poses[transform.child_frame_id] = transform

    def ee_callback(self, msg):
        self.current_ee_pose = msg

    def joint_state_callback(self, msg):
        if len(msg.name) == 6 and len(msg.position) == 6:
            joint_dict = dict(zip(msg.name, msg.position))
            positions = [joint_dict.get(name, 0) for name in self.joint_names]
            if len(positions) == 6:
                self.current_joint_angles = np.array(positions)
                self.joint_angles_received = True

    def get_rotation_from_transform(self, transform):
        q = np.array([transform.rotation.x, transform.rotation.y,
                      transform.rotation.z, transform.rotation.w])
        return R.from_quat(q).as_matrix()

    def get_rotation_from_quat(self, quat):
        return R.from_quat(quat).as_matrix()

    def get_pose_from_msg(self, pose_msg):
        position = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y,
                            pose_msg.pose.position.z])
        q = np.array([pose_msg.pose.orientation.x, pose_msg.pose.orientation.y,
                      pose_msg.pose.orientation.z, pose_msg.pose.orientation.w])
        return position, R.from_quat(q).as_matrix()

    def read_current_joint_angles(self):
        self.joint_angles_received = False
        timeout = 0
        while rclpy.ok() and not self.joint_angles_received and timeout < 100:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout += 1
        return self.current_joint_angles.copy() if self.joint_angles_received else None

    def compute_ik_with_current_seed(self, target_position, target_quat, max_tries=5, dx=0.001):
        target_rot = R.from_quat(target_quat).as_matrix()
        target_pose = np.eye(4)
        target_pose[:3, 3] = target_position
        target_pose[:3, :3] = target_rot

        if self.current_joint_angles is None:
            return None

        q_guess = self.current_joint_angles.copy()
        best_result, best_cost = None, float('inf')
        joint_bounds = [(-np.pi, np.pi)] * 6

        for i in range(max_tries):
            perturbed = target_pose.copy()
            perturbed[0, 3] += i * dx

            result = minimize(ik_objective_quaternion, q_guess, args=(perturbed,),
                            method='L-BFGS-B', bounds=joint_bounds)
            if result.success:
                cost = ik_objective_quaternion(result.x, perturbed)
                if cost < 0.01:
                    return result.x
                if cost < best_cost:
                    best_cost, best_result = cost, result.x

        if best_result is not None and best_cost < 0.1:
            return best_result

        # Fallback seeds
        yaw_rad = np.arctan2(2.0 * (target_quat[3] * target_quat[2] + target_quat[0] * target_quat[1]),
                            1.0 - 2.0 * (target_quat[1]**2 + target_quat[2]**2))
        yaw_deg = np.degrees(yaw_rad)

        seeds = [
            np.radians([85, -80, 90, -90, -90, yaw_deg]),
            np.radians([90, -90, 90, -90, -90, yaw_deg]),
            np.radians([0, -90, 90, -90, -90, yaw_deg]),
            np.radians([180, -90, 90, -90, -90, yaw_deg]),
        ]

        for seed in seeds:
            for i in range(max_tries):
                perturbed = target_pose.copy()
                perturbed[0, 3] += i * dx
                result = minimize(ik_objective_quaternion, seed, args=(perturbed,),
                                method='L-BFGS-B', bounds=joint_bounds)
                if result.success:
                    cost = ik_objective_quaternion(result.x, perturbed)
                    if cost < 0.01:
                        return result.x
                    if cost < best_cost:
                        best_cost, best_result = cost, result.x

        return best_result if best_cost < 0.1 else None

    def find_best_cardinal_for_target(self, R_object_target, R_grasp, fold_data,
                                     R_object_current=None, R_EE_current=None):
        """
        Find the cardinal EE orientation that places the object closest to target orientation.
        """
        # Generate all symmetry-equivalent target orientations
        equivalent_targets = FoldSymmetry.generate_equivalent_target_orientations(
            R_object_target, fold_data, self.get_logger()
        )

        cardinals = ExtendedCardinalOrientations.get_all_extended_cardinals()

        candidates = []

        for card_name, card_quat in cardinals.items():
            # Block "up" orientation (quaternion [0, 0, 0, 1] or RPY [0, 0, 0])
            # This orientation is not allowed for safety/mechanical reasons
            R_EE_cardinal = R.from_quat(card_quat).as_matrix()
            ee_rpy = R.from_matrix(R_EE_cardinal).as_euler('xyz', degrees=True)
            rpy_tolerance = 5.0  # degrees tolerance
            if (abs(ee_rpy[0]) < rpy_tolerance and 
                abs(ee_rpy[1]) < rpy_tolerance and 
                abs(ee_rpy[2]) < rpy_tolerance):
                continue  # Skip this "up" orientation
            
            # What object orientation results from this cardinal EE?
            R_object_result = R_EE_cardinal @ R_grasp

            # Find closest equivalent target
            min_error = float('inf')
            best_target = None
            for R_target_equiv in equivalent_targets:
                error = ExtendedCardinalOrientations.rotation_matrix_distance(
                    R_object_result, R_target_equiv
                )
                if error < min_error:
                    min_error = error
                    best_target = R_target_equiv

            # Calculate rotation distance from current EE
            ee_rotation_distance = 0.0
            if R_EE_current is not None:
                ee_rotation_distance = ExtendedCardinalOrientations.rotation_matrix_distance(
                    R_EE_current, R_EE_cardinal
                )

            candidates.append((
                card_name, card_quat, R_object_result,
                best_target, min_error, ee_rotation_distance
            ))

        # Sort candidates: first by object error, then by EE rotation distance
        error_tolerance = 5.0  # degrees

        def sort_key(candidate):
            obj_error = candidate[4]
            ee_rotation = candidate[5]
            error_bucket = round(obj_error / error_tolerance) * error_tolerance
            return (error_bucket, ee_rotation)

        candidates.sort(key=sort_key)

        if len(candidates) > 0:
            best = candidates[0]
            return (best[0], best[1], best[2], best[3], best[4], candidates)

        return (None, None, None, None, float('inf'), [])

    def reorient_to_target(self, object_name, target_orientation,
                          current_object_orientation=None, duration=5.0):
        """
        Reorient object to target orientation.

        Args:
            object_name: Name of the object to reorient
            target_orientation: Target orientation quaternion [x,y,z,w] in world frame
            current_object_orientation: Current object orientation [x,y,z,w] (optional)
            duration: Trajectory duration in seconds
        """
        self.get_logger().info(f"Reorienting {object_name} to target orientation")

        # Get current EE pose
        if self.current_ee_pose is None:
            self.get_logger().error("EE pose not available")
            return False
        ee_position, R_EE_current = self.get_pose_from_msg(self.current_ee_pose)

        # Get current object orientation
        if current_object_orientation is not None:
            R_object_current = self.get_rotation_from_quat(current_object_orientation)
        else:
            obj_key = object_name if object_name in self.current_poses else f"{object_name}_scaled70"
            if obj_key not in self.current_poses:
                self.get_logger().error(f"Object {object_name} not found in poses")
                return False
            R_object_current = self.get_rotation_from_transform(self.current_poses[obj_key].transform)

        # Get target orientation (world frame)
        R_object_target_world = self.get_rotation_from_quat(target_orientation)

        # Load fold symmetry
        fold_data = FoldSymmetry.load_symmetry_data(object_name, self.symmetry_dir)
        if fold_data is None:
            fold_data = FoldSymmetry.load_symmetry_data(f"{object_name}_scaled70", self.symmetry_dir)

        # Log initial orientation
        initial_rpy = R.from_matrix(R_object_current).as_euler('xyz', degrees=True)
        self.get_logger().info(f"Initial object orientation (RPY): [{initial_rpy[0]:.1f}, {initial_rpy[1]:.1f}, {initial_rpy[2]:.1f}]")

        target_rpy = R.from_matrix(R_object_target_world).as_euler('xyz', degrees=True)
        self.get_logger().info(f"Target object orientation (RPY): [{target_rpy[0]:.1f}, {target_rpy[1]:.1f}, {target_rpy[2]:.1f}]")

        # === Early check: Is current orientation already at target (considering symmetry)? ===
        # Generate all symmetry-equivalent target orientations
        equivalent_targets = FoldSymmetry.generate_equivalent_target_orientations(
            R_object_target_world, fold_data, self.get_logger()
        )
        
        # Check if current object orientation matches any equivalent target
        min_error = float('inf')
        matched_target_idx = 0
        for idx, R_equiv_target in enumerate(equivalent_targets):
            error = ExtendedCardinalOrientations.rotation_matrix_distance(
                R_object_current, R_equiv_target
            )
            if error < min_error:
                min_error = error
                matched_target_idx = idx
        
        # If already at target (within tolerance), skip reorientation
        orientation_tolerance = 5.0  # degrees
        if min_error <= orientation_tolerance:
            # Check if symmetry was used (index 0 is the original target, >0 means symmetry was used)
            symmetry_used = matched_target_idx > 0
            num_equivalent_targets = len(equivalent_targets)
            
            if symmetry_used:
                self.get_logger().info(
                    f"Object is already at target orientation (error: {min_error:.2f}°). "
                    f"Match found via fold symmetry (equivalent target {matched_target_idx + 1}/{num_equivalent_targets}). "
                    f"No reorientation needed."
                )
            else:
                self.get_logger().info(
                    f"Object is already at target orientation (error: {min_error:.2f}°). "
                    f"Direct match (no symmetry needed). No reorientation needed."
                )
            return True

        # Calculate grasp rotation
        R_grasp = R_EE_current.T @ R_object_current

        # === KEY: Use target orientation as reference frame for cardinal selection ===
        # This is analogous to using base orientation in assembly mode
        # Cardinals are selected in target-relative frame, then transformed back to world

        # Transform to target-relative frame
        R_object_current_target_relative = R_object_target_world.T @ R_object_current
        R_EE_current_target_relative = R_object_target_world.T @ R_EE_current

        # Target in its own frame is identity
        R_object_target_target_relative = np.eye(3)

        # Find best cardinal orientation in target-relative frame
        (best_cardinal, best_quat_target_relative, resulting_object_R_target_relative,
         matched_target_R_target_relative, object_error, candidates) = self.find_best_cardinal_for_target(
            R_object_target_target_relative, R_grasp, fold_data,
            R_object_current_target_relative, R_EE_current_target_relative
        )

        if best_cardinal is None:
            self.get_logger().error("Failed to find suitable cardinal orientation")
            return False

        # Transform result back to world frame
        R_EE_result_target_relative = R.from_quat(best_quat_target_relative).as_matrix()
        R_EE_result_world = R_object_target_world @ R_EE_result_target_relative
        best_quat = R.from_matrix(R_EE_result_world).as_quat()

        resulting_object_R = R_object_target_world @ resulting_object_R_target_relative
        matched_target_R = R_object_target_world @ matched_target_R_target_relative

        # Transform candidates back to world frame
        if candidates is not None:
            transformed_candidates = []
            for card_name, card_quat_target_rel, card_obj_R_target_rel, card_target_R_target_rel, card_error, ee_rot_dist in candidates:
                R_EE_cand_target_rel = R.from_quat(card_quat_target_rel).as_matrix()
                R_EE_cand_world = R_object_target_world @ R_EE_cand_target_rel
                card_quat_world = R.from_matrix(R_EE_cand_world).as_quat()
                card_obj_R_world = R_object_target_world @ card_obj_R_target_rel
                card_target_R_world = R_object_target_world @ card_target_R_target_rel
                transformed_candidates.append((card_name, card_quat_world, card_obj_R_world,
                                              card_target_R_world, card_error, ee_rot_dist))
            candidates = transformed_candidates

        self.get_logger().info(f"Selected cardinal: {best_cardinal}")
        self.get_logger().info(f"Object alignment error: {object_error:.1f}°")

        # Check if error is acceptable
        if object_error > 30.0:
            self.get_logger().warn(f"High alignment error ({object_error:.1f}°) - result may not be ideal")

        # Compute IK
        if self.current_joint_angles is None:
            if self.read_current_joint_angles() is None:
                self.get_logger().error("Could not read joint angles")
                return False

        # Try IK with best solution
        joint_angles = self.compute_ik_with_current_seed(ee_position.tolist(), best_quat.tolist())

        # Try alternatives if IK fails
        if joint_angles is None and candidates is not None:
            for i, (card_name, card_quat, card_object_R, card_target_R, card_error, _) in enumerate(candidates[1:6], 1):
                self.get_logger().info(f"Trying alternative cardinal {i}: {card_name}")
                joint_angles = self.compute_ik_with_current_seed(ee_position.tolist(), card_quat.tolist())

                if joint_angles is not None:
                    best_cardinal = card_name
                    best_quat = card_quat
                    resulting_object_R = card_object_R
                    matched_target_R = card_target_R
                    object_error = card_error
                    self.get_logger().info(f"Using alternative: {card_name} (error: {card_error:.1f}°)")
                    break

        if joint_angles is None:
            self.get_logger().error("IK failed for all attempted cardinals")
            return False

        # === Block "up" orientation (quaternion [0, 0, 0, 1] or RPY [0, 0, 0]) ===
        # This is a safety check to prevent the EE from reaching the forbidden "up" orientation
        # If detected, reject and return False (do not redirect)
        EE_rpy = R.from_matrix(R.from_quat(best_quat).as_matrix()).as_euler('xyz', degrees=True)
        rpy_tolerance = 5.0  # degrees tolerance
        if (abs(EE_rpy[0]) < rpy_tolerance and 
            abs(EE_rpy[1]) < rpy_tolerance and 
            abs(EE_rpy[2]) < rpy_tolerance):
            self.get_logger().error(f"Rejected 'up' orientation (RPY: [{EE_rpy[0]:.1f}, {EE_rpy[1]:.1f}, {EE_rpy[2]:.1f}]). This orientation is forbidden.")
            return False

        # Execute trajectory
        trajectory = {"traj1": [{
            "positions": [float(x) for x in joint_angles],
            "velocities": [0.0] * 6,
            "time_from_start": Duration(sec=int(duration))
        }]}

        success = self.execute_trajectory(trajectory)

        if success:
            final_rpy = R.from_matrix(resulting_object_R).as_euler('xyz', degrees=True)
            self.get_logger().info(f"Final object orientation (RPY): [{final_rpy[0]:.1f}, {final_rpy[1]:.1f}, {final_rpy[2]:.1f}]")

        return success

    def execute_trajectory(self, trajectory):
        try:
            point = trajectory['traj1'][0]

            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names

            traj_point = JointTrajectoryPoint()
            traj_point.positions = point['positions']
            traj_point.velocities = [0.0] * 6
            traj_point.time_from_start = point['time_from_start']
            traj_msg.points.append(traj_point)

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)

            self.trajectory_completed = False
            self.trajectory_success = False

            self.get_logger().info("Trajectory sent and accepted")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response_callback)

            while rclpy.ok() and not self.trajectory_completed:
                rclpy.spin_once(self, timeout_sec=0.1)

            if self.trajectory_success:
                self.get_logger().info("Movement completed successfully")
            else:
                self.get_logger().error("Trajectory failed")

            return self.trajectory_success
        except Exception as e:
            self.get_logger().error(f"Trajectory error: {e}")
            return False

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            self.trajectory_completed = True
            self.trajectory_success = False
            return
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        result = future.result()
        self.trajectory_success = (result.status == 4)
        self.trajectory_completed = True


def main(args=None):
    parser = argparse.ArgumentParser(description='Reorient Object to Target Orientation')
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: sim (reads from topic) or real (requires current orientation)')
    parser.add_argument('--object-name', type=str, required=True,
                       help='Name of the object to reorient')
    parser.add_argument('--target-orientation', type=float, nargs=4, required=True,
                       metavar=('X','Y','Z','W'),
                       help='Target object orientation quaternion [x, y, z, w] in world frame')
    parser.add_argument('--current-orientation', type=float, nargs=4,
                       metavar=('X','Y','Z','W'),
                       help='Current object orientation quaternion [x, y, z, w] (required in real mode)')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Trajectory duration in seconds (default: 5.0)')

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == 'real' and args.current_orientation is None:
        parser.error("--current-orientation is required in real mode")

    rclpy.init()
    node = ReorientObject(mode=args.mode)
    node.action_client.wait_for_server()

    try:
        # Wait for EE pose
        while node.current_ee_pose is None:
            rclpy.spin_once(node, timeout_sec=0.1)

        # In sim mode, wait for object poses if current orientation not provided
        if args.mode == 'sim' and args.current_orientation is None:
            while not node.current_poses:
                rclpy.spin_once(node, timeout_sec=0.1)

        success = node.reorient_to_target(
            args.object_name,
            args.target_orientation,
            args.current_orientation,
            args.duration
        )

        if success:
            node.get_logger().info("Reorientation completed successfully")
        else:
            node.get_logger().error("Reorientation failed")

        time.sleep(0.5)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()
