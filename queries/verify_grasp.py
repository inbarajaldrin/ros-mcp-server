#!/usr/bin/env python3
"""
Verify Grasp - Checks if object is within a configurable radius from gripper center

The algorithm:
1. Get current object pose from ROS topic
2. Get current gripper center pose from TCP pose
3. Calculate Euclidean distance between object position and gripper center position
4. Check if distance is within the specified radius tolerance
5. Return success if within radius, failure if not
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import time
import sys
import os
import json

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from primitives.shared.config import GRIPPER_CENTER_TOOL_OFFSET
from utils.data_path_finder import get_aruco_data_dir
from std_msgs.msg import Float32, Float64

# Gripper specifications
GRIPPER_HALF_OPEN_WIDTH_MM = 35.0
GRIPPER_MAX_WIDTH_MM = 100.0
GRIPPER_WIDTH_MATCH_TOLERANCE_MM = 2.0  # If width is within this of approach width, gripper hasn't closed


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")

# Configuration
OBJECT_TOPIC_SIM = "/objects_poses_sim"
OBJECT_TOPIC_REAL = "/objects_poses_real"
EE_TOPIC = "/tcp_pose_broadcaster/pose"

# Default radius tolerance (in meters)
DEFAULT_RADIUS = 0.06  # 6cm default radius


# ============================================================================
# REAL MODE HELPER FUNCTIONS
# ============================================================================

def load_grasp_point_and_validity(object_name, grasp_id=0, logger=None):
    """Load grasp point and validity from JSON"""
    try:
        data_dir = get_aruco_data_dir() / "grasp_points"
    except Exception:
        if logger:
            logger.error("Could not find aruco data directory")
        return None, None

    json_path = data_dir / f"{object_name}_grasp_points.json"
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            for gp in data.get('grasp_points', []):
                if gp['id'] == grasp_id:
                    pos = gp['position']
                    grasp_point = np.array([pos['x'], pos['y'], pos['z']])
                    grasp_validity = gp.get('grasp_validity', {})
                    if logger:
                        logger.info(f"Loaded grasp point {grasp_id} for {object_name}")
                    return grasp_point, grasp_validity
        except (json.JSONDecodeError, IOError, KeyError):
            pass

    if logger:
        logger.error(f"No grasp points file found for '{object_name}'")
    return None, None


from primitives.shared.fold_symmetry import (
    load_symmetry_data as _load_sym,
    get_symmetry_matrices as _get_sym_matrices,
)


def load_fold_symmetry(object_name, logger=None):
    """Load fold symmetry data for an object. Returns None if not found."""
    try:
        symmetry_dir = str(get_aruco_data_dir() / "symmetry")
    except Exception:
        return None
    data = _load_sym(object_name, symmetry_dir)
    if data and logger:
        logger.info(f"Loaded fold symmetry for {object_name}")
    return data


def get_fold_symmetry_matrices(fold_data):
    """Generate all combinations of fold symmetry rotations as 3x3 matrices.
    Always includes identity."""
    return _get_sym_matrices(fold_data)


def determine_grip_axis(grasp_point_world):
    """Determine grip axis from grasp point direction"""
    abs_vals = np.abs(grasp_point_world)
    max_idx = np.argmax(abs_vals)
    return ["x_axis", "y_axis", "z_axis"][max_idx]


def check_gripper_width_valid(gripper_width_mm, valid_modes):
    """Check if gripper width matches valid modes"""
    if gripper_width_mm <= 0:
        return False, f"Gripper fully closed (width: {gripper_width_mm:.2f}mm) - nothing grasped"

    if gripper_width_mm >= GRIPPER_MAX_WIDTH_MM:
        return False, f"Gripper fully open (width: {gripper_width_mm:.2f}mm) - no grip"

    if not valid_modes:
        return False, "No valid gripper modes for this axis"

    has_half_open = "half-open" in valid_modes
    has_open = "open" in valid_modes

    if has_half_open:
        if gripper_width_mm < GRIPPER_HALF_OPEN_WIDTH_MM:
            return True, f"Valid: width {gripper_width_mm:.2f}mm < half-open {GRIPPER_HALF_OPEN_WIDTH_MM:.1f}mm"
        else:
            return False, f"Invalid: width {gripper_width_mm:.2f}mm >= half-open {GRIPPER_HALF_OPEN_WIDTH_MM:.1f}mm"
    elif has_open:
        if GRIPPER_HALF_OPEN_WIDTH_MM <= gripper_width_mm < GRIPPER_MAX_WIDTH_MM:
            return True, f"Valid: width {gripper_width_mm:.2f}mm in open range [{GRIPPER_HALF_OPEN_WIDTH_MM:.1f}, {GRIPPER_MAX_WIDTH_MM:.1f})"
        else:
            return False, f"Invalid: width {gripper_width_mm:.2f}mm not in open range [{GRIPPER_HALF_OPEN_WIDTH_MM:.1f}, {GRIPPER_MAX_WIDTH_MM:.1f})"

    return False, "Unknown valid modes"


class VerifyGrasp(Node):
    def __init__(self, object_name, mode='sim', radius=DEFAULT_RADIUS, grasp_id=0, object_orientation=None):
        super().__init__('verify_grasp')

        self.object_name = object_name
        self.mode = mode
        self.radius = radius
        self.grasp_id = grasp_id
        self.object_orientation = object_orientation

        # TCP to gripper center offset distance (from TCP to gripper center along gripper Z-axis)
        self.tcp_to_gripper_center_offset = GRIPPER_CENTER_TOOL_OFFSET

        # Determine topic name based on mode
        if mode == 'sim':
            object_topic = OBJECT_TOPIC_SIM
        else:  # real mode
            object_topic = OBJECT_TOPIC_REAL

        # Subscribers for pose data
        self.object_sub = self.create_subscription(TFMessage, object_topic, self.object_callback, 10)

        # Configure QoS to match the publisher (VOLATILE durability - default for most publishers)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.ee_sub = self.create_subscription(
            PoseStamped,
            EE_TOPIC,
            self.ee_callback,
            qos_profile
        )

        # Gripper width (both modes, different topics)
        self.gripper_width = None
        self.gripper_width_received = False
        self.gripper_asymmetry = 0.0
        self.gripper_asymmetry_received = False
        if mode == 'real':
            self.width_sub = self.create_subscription(
                Float32, '/gripper_width_offset', self.width_callback, 10)
        else:
            self.width_sub = self.create_subscription(
                Float64, '/gripper_width_sim', self.width_callback_sim, 10)
            self.asymmetry_sub = self.create_subscription(
                Float64, '/gripper_asymmetry_sim', self.asymmetry_callback_sim, 10)

        # Store current poses
        self.current_poses = {}
        self.current_ee_pose = None
        self.object_pose_received = False
        self.ee_pose_received = False
    
    def object_callback(self, msg):
        """Callback for object poses"""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            self.current_poses[frame_id] = transform
            if frame_id == self.object_name:
                self.object_pose_received = True
    
    def ee_callback(self, msg):
        """Callback for end-effector pose"""
        self.current_ee_pose = msg
        self.ee_pose_received = True

    def width_callback(self, msg):
        """Callback for gripper width (real mode, Float32 in mm)"""
        self.gripper_width = msg.data
        self.gripper_width_received = True

    def width_callback_sim(self, msg):
        """Callback for gripper width (sim mode, Float64 in mm)"""
        self.gripper_width = msg.data
        self.gripper_width_received = True

    def asymmetry_callback_sim(self, msg):
        """Callback for gripper asymmetry (sim mode, Float64 in mm)"""
        self.gripper_asymmetry = msg.data
        self.gripper_asymmetry_received = True

    def compute_gripper_center_from_tcp(self, tcp_position, tcp_quaternion):
        """Compute the gripper center pose from TCP pose using the same logic as move_to_grasp.
        
        The offset vector is defined in the tool frame (gripper frame) and then
        transformed to world frame using the tool orientation quaternion.
        
        Args:
            tcp_position: TCP position in world frame [x, y, z]
            tcp_quaternion: TCP/tool orientation quaternion [x, y, z, w] (tool frame to world frame)
        
        Returns:
            gripper_center_position: Position of the gripper center in world frame [x, y, z]
        """
        # Offset vector in tool frame (gripper frame): [0, 0, offset_distance]
        # In tool frame, Z-axis points from TCP to gripper center (downward)
        offset_vector_tool_frame = self.tcp_to_gripper_center_offset
        
        # Transform offset vector from tool frame to world frame using quaternion
        # The quaternion represents the rotation from tool frame to world frame
        r = R.from_quat(tcp_quaternion)
        offset_vector_world = r.apply(offset_vector_tool_frame)
        
        # Compute gripper center: TCP + offset_vector_world
        # (going forward from TCP to gripper center along the tool Z-axis)
        gripper_center_position = np.array(tcp_position) + offset_vector_world

        return gripper_center_position
    
    def verify_grasp(self):
        """
        Verify if object is within the specified radius from gripper center
        
        Algorithm:
        1. Get current object pose
        2. Get current gripper center pose from TCP
        3. Calculate Euclidean distance between object position and gripper center position
        4. Check if distance is within the specified radius
        
        Returns:
            tuple: (success: bool, distance: float, object_position: np.array, gripper_center_position: np.array)
        """
        # Wait for pose data
        if not self.object_pose_received:
            self.get_logger().error(f"Object pose for '{self.object_name}' not received")
            return False, None, None, None
        
        if not self.ee_pose_received:
            self.get_logger().error("End-effector pose not received")
            return False, None, None, None
        
        # Check if object exists
        if self.object_name not in self.current_poses:
            self.get_logger().error(f"Object {self.object_name} not found in poses")
            return False, None, None, None
        
        object_key = self.object_name

        # Get object position
        transform = self.current_poses[object_key].transform
        object_position = np.array([
            transform.translation.x,
            transform.translation.y,
            transform.translation.z
        ])
        
        # Get TCP pose and compute gripper center position
        tcp_position = np.array([
            self.current_ee_pose.pose.position.x,
            self.current_ee_pose.pose.position.y,
            self.current_ee_pose.pose.position.z
        ])
        tcp_quat = np.array([
            self.current_ee_pose.pose.orientation.x,
            self.current_ee_pose.pose.orientation.y,
            self.current_ee_pose.pose.orientation.z,
            self.current_ee_pose.pose.orientation.w
        ])
        
        gripper_center_position = self.compute_gripper_center_from_tcp(tcp_position, tcp_quat)
        
        # Calculate Euclidean distance between object position and gripper center position
        distance = np.linalg.norm(object_position - gripper_center_position)
        
        # Check if within radius
        success = distance <= self.radius
        
        if success:
            self.get_logger().info(f"Grasp verification successful: Object is within radius ({distance*1000:.2f}mm <= {self.radius*1000:.2f}mm)")
        else:
            self.get_logger().error(f"Grasp verification failed: Object is outside radius ({distance*1000:.2f}mm > {self.radius*1000:.2f}mm)")
        
        # Log detailed information
        self.get_logger().info(f"Object position: [{object_position[0]*1000:.2f}, {object_position[1]*1000:.2f}, {object_position[2]*1000:.2f}] mm")
        self.get_logger().info(f"Gripper center position: [{gripper_center_position[0]*1000:.2f}, {gripper_center_position[1]*1000:.2f}, {gripper_center_position[2]*1000:.2f}] mm")
        self.get_logger().info(f"Distance: {distance*1000:.2f} mm, Radius threshold: {self.radius*1000:.2f} mm")
        
        return success, distance, object_position, gripper_center_position

    def verify_grasp_real(self):
        """
        Real mode: Verify grasp using grasp points and gripper width.

        Returns:
            tuple: (success: bool, result_dict: dict)
        """
        result = {
            'result': 'success',
            'object_name': self.object_name,
            'mode': 'real',
        }

        # Check data
        if not self.gripper_width_received:
            result['result'] = 'failure'
            result['error'] = "Gripper width not received"
            self.get_logger().error(result['error'])
            return False, result

        if self.object_orientation is None:
            result['result'] = 'failure'
            result['error'] = "Object orientation not provided"
            self.get_logger().error(result['error'])
            return False, result

        # Load grasp point and validity
        grasp_point_cad, grasp_validity = load_grasp_point_and_validity(
            self.object_name, self.grasp_id, logger=self.get_logger()
        )

        if grasp_point_cad is None:
            result['result'] = 'failure'
            result['error'] = f"Could not load grasp point {self.grasp_id}"
            self.get_logger().error(result['error'])
            return False, result

        # Load fold symmetry to try all equivalent orientations
        fold_data = load_fold_symmetry(self.object_name, logger=self.get_logger())
        symmetry_matrices = get_fold_symmetry_matrices(fold_data)
        self.get_logger().info(f"Fold symmetry: {len(symmetry_matrices)} equivalent orientations")

        R_object = R.from_quat(self.object_orientation).as_matrix()

        # Try all fold-equivalent orientations, accept if any yields a valid grip axis
        is_valid = False
        reason = ""
        best_axis = None
        for i, R_sym in enumerate(symmetry_matrices):
            R_equiv = R_object @ R_sym
            grasp_point_world = R_equiv @ grasp_point_cad
            grip_axis = determine_grip_axis(grasp_point_world)
            valid_modes = grasp_validity.get(grip_axis, [])

            valid_i, reason_i = check_gripper_width_valid(self.gripper_width, valid_modes)
            if i == 0:
                self.get_logger().info(f"Grasp point in world: [{grasp_point_world[0]:.6f}, {grasp_point_world[1]:.6f}, {grasp_point_world[2]:.6f}]")
                self.get_logger().info(f"Grip axis: {grip_axis}, valid modes: {valid_modes}")

            if valid_i:
                is_valid = True
                reason = reason_i
                best_axis = grip_axis
                if i > 0:
                    self.get_logger().info(f"Valid via fold symmetry #{i}: axis={grip_axis}, modes={valid_modes}")
                break
            elif i == 0:
                reason = reason_i  # Keep first orientation's reason as fallback

        if best_axis:
            self.get_logger().info(f"Grip axis (accepted): {best_axis}")

        self.get_logger().info(f"Gripper width: {self.gripper_width:.2f}mm")
        self.get_logger().info(f"Width check: {reason}")

        if is_valid:
            self.get_logger().info("✓ Real mode grasp verification PASSED")
        else:
            result['result'] = 'failure'
            result['error'] = reason
            self.get_logger().error("✗ Real mode grasp verification FAILED")

        return is_valid, result

    def verify_grasp_width_only(self):
        """Simple check that gripper is holding something (width > 0 and < max). Works for both sim and real.

        Returns:
            tuple: (success: bool, result_dict: dict)
        """
        result = {
            'result': 'success',
            'object_name': self.object_name,
            'mode': self.mode,
        }

        if not self.gripper_width_received:
            result['result'] = 'failure'
            result['error'] = "Gripper width not received"
            self.get_logger().error(result['error'])
            return False, result

        self.get_logger().info(f"Gripper width: {self.gripper_width:.2f}mm")

        if self.gripper_width <= 0:
            result['result'] = 'failure'
            result['error'] = f"Gripper fully closed (width: {self.gripper_width:.2f}mm) - nothing grasped"
            self.get_logger().error(result['error'])
            return False, result

        if self.gripper_width >= GRIPPER_MAX_WIDTH_MM:
            result['result'] = 'failure'
            result['error'] = f"Gripper fully open (width: {self.gripper_width:.2f}mm) - no grip"
            self.get_logger().error(result['error'])
            return False, result

        self.get_logger().info(f"Width-only grasp check PASSED (width: {self.gripper_width:.2f}mm)")
        return True, result

    def verify_grasp_dispatch(self, width_only=False):
        """Dispatch to sim or real mode verification"""
        # Width-only check works for both sim and real
        if width_only:
            return self.verify_grasp_width_only()

        if self.mode == 'real':
            return self.verify_grasp_real()
        elif self.mode == 'sim':
            # Sim mode: use gripper width as the primary grasp signal.
            # The distance-based check is unreliable for large objects whose
            # center can be far from gripper center even when properly grasped.
            if self.gripper_width_received:
                self.get_logger().info(f"Gripper width: {self.gripper_width:.1f}mm")

                # Asymmetric gripper = jammed, not a valid grasp
                ASYMMETRY_THRESHOLD_MM = 15.0
                if self.gripper_asymmetry_received and self.gripper_asymmetry > ASYMMETRY_THRESHOLD_MM:
                    result = {
                        'result': 'failure',
                        'object_name': self.object_name,
                        'mode': 'sim',
                        'error': "Grasp verification failed - gripper is asymmetric (jammed)",
                    }
                    self.get_logger().info(f"Gripper asymmetry: {self.gripper_asymmetry:.1f}mm (threshold: {ASYMMETRY_THRESHOLD_MM}mm) — jammed")
                    return False, result

                # Fully open gripper = definitely not grasping
                if self.gripper_width >= GRIPPER_MAX_WIDTH_MM - GRIPPER_WIDTH_MATCH_TOLERANCE_MM:
                    result = {
                        'result': 'failure',
                        'object_name': self.object_name,
                        'mode': 'sim',
                        'error': "Grasp verification failed - object is not grasped",
                    }
                    self.get_logger().info(f"Gripper fully open ({self.gripper_width:.1f}mm) — not grasping")
                    return False, result

                # Gripper fully closed (~0mm) = missed the object
                if self.gripper_width <= GRIPPER_WIDTH_MATCH_TOLERANCE_MM:
                    result = {
                        'result': 'failure',
                        'object_name': self.object_name,
                        'mode': 'sim',
                        'error': "Grasp verification failed - object is not grasped",
                    }
                    self.get_logger().info(f"Gripper fully closed ({self.gripper_width:.1f}mm) — nothing between fingers")
                    return False, result

                # Check approach widths — if gripper is still at approach width,
                # it hasn't closed on anything (pre-grasp or re-opened).
                approach_widths = []
                if self.grasp_id is not None:
                    # Known grasp_id: check its specific approach width
                    _, grasp_validity = load_grasp_point_and_validity(
                        self.object_name, self.grasp_id, logger=self.get_logger()
                    )
                    if grasp_validity:
                        aw = grasp_validity.get('x_axis_gripper_width_mm')
                        if aw is not None:
                            approach_widths = [aw]
                else:
                    # No grasp_id: check all grasp points' approach widths
                    try:
                        data_dir = get_aruco_data_dir() / "grasp_points"
                        json_path = data_dir / f"{self.object_name}_grasp_points.json"
                        if json_path.exists():
                            with open(json_path, 'r') as f:
                                data = json.load(f)
                            for gp in data.get('grasp_points', []):
                                gv = gp.get('grasp_validity', {})
                                aw = gv.get('x_axis_gripper_width_mm')
                                if aw is not None and aw not in approach_widths:
                                    approach_widths.append(aw)
                            if approach_widths:
                                self.get_logger().info(f"No grasp_id — checking all approach widths: {approach_widths}")
                    except Exception:
                        pass
                for aw in approach_widths:
                    if abs(self.gripper_width - aw) <= GRIPPER_WIDTH_MATCH_TOLERANCE_MM:
                        result = {
                            'result': 'failure',
                            'object_name': self.object_name,
                            'mode': 'sim',
                            'error': "Grasp verification failed - object is not grasped",
                        }
                        self.get_logger().info(f"Gripper at approach width ({self.gripper_width:.1f}mm ≈ {aw:.1f}mm) — not grasping")
                        return False, result

                # Gripper is between 0 and approach/max width → holding something
                self.get_logger().info(f"Gripper holding object (width: {self.gripper_width:.1f}mm)")
                result = {
                    'result': 'success',
                    'object_name': self.object_name,
                    'mode': 'sim',
                }
                return True, result

            # Fallback: no gripper width data, use distance-based check
            self.get_logger().warning("No gripper width data — falling back to distance check")
            success, distance, object_pos, ref_pos = self.verify_grasp()
            result = {
                'result': 'success' if success else 'failure',
                'object_name': self.object_name,
                'mode': 'sim',
            }
            if not success:
                if distance is None:
                    result['error'] = f"Object '{self.object_name}' not found"
                else:
                    result['error'] = "Object is outside grasp radius"
            return success, result
        else:
            result = {
                'result': 'failure',
                'object_name': self.object_name,
                'error': f"Invalid mode '{self.mode}'. Must be 'sim' or 'real'."
            }
            return False, result


def main(args=None):
    parser = argparse.ArgumentParser(description='Verify Grasp - Sim or Real mode')
    parser.add_argument('--object-name', type=str, required=True, help='Object name')
    parser.add_argument('--mode', type=str, choices=['sim', 'real'], default='sim', help='Sim or real mode')
    parser.add_argument('--radius', type=float, default=DEFAULT_RADIUS,
                       help=f'Radius tolerance in meters (sim mode)')
    parser.add_argument('--grasp-id', type=int, default=None, help='Grasp point ID (required for real mode)')
    parser.add_argument('--current-object-orientation', type=float, nargs=4,
                       help='Object orientation quaternion [x, y, z, w] (real mode)')
    parser.add_argument('--width-only', action='store_true',
                       help='Real mode: only check gripper width > 0 (skip grasp point/axis validation)')

    parsed_args = parser.parse_args()

    # Validate arguments
    if parsed_args.mode == 'real' and not parsed_args.width_only:
        if parsed_args.current_object_orientation is None:
            result = {'result': 'failure', 'object_name': parsed_args.object_name, 'mode': 'real',
                     'error': 'Object orientation required for real mode (--current-object-orientation x y z w)'}
            output_result(result)
            sys.exit(1)
        if parsed_args.grasp_id is None or parsed_args.grasp_id < 0:
            result = {'result': 'failure', 'object_name': parsed_args.object_name, 'mode': 'real',
                     'error': 'Grasp ID required for real mode (--grasp-id <int>)'}
            output_result(result)
            sys.exit(1)

    success = False
    result = {}
    node = None

    try:
        rclpy.init()

        # Real mode requires object orientation
        object_quat = None
        if parsed_args.mode == 'real':
            object_quat = np.array(parsed_args.current_object_orientation)

        node = VerifyGrasp(
            object_name=parsed_args.object_name,
            mode=parsed_args.mode,
            radius=parsed_args.radius,
            grasp_id=parsed_args.grasp_id,
            object_orientation=object_quat
        )

        # Wait for required data
        if parsed_args.width_only:
            # Width-only mode: just need gripper width (works for both sim and real)
            node.get_logger().info("Waiting for gripper width data...")
            start_time = time.time()
            last_log_time = start_time

            while not node.gripper_width_received:
                rclpy.spin_once(node, timeout_sec=0.1)
                time.sleep(0.1)

                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    elapsed = current_time - start_time
                    node.get_logger().info(f"Waiting for gripper width... ({elapsed:.1f}s)")
                    last_log_time = current_time

            elapsed = time.time() - start_time
            node.get_logger().info(f"Received gripper width (waited {elapsed:.1f}s)")

        elif parsed_args.mode == 'sim':
            node.get_logger().info(f"Waiting for pose data for object: {parsed_args.object_name}")
            start_time = time.time()
            last_log_time = start_time

            # NOTE: /gripper_asymmetry_sim was REMOVED by the joint-actuator migration (the twin's
            # pure joint-actuator seam no longer publishes effort/asymmetry — see control_gripper.py),
            # so DON'T block on gripper_asymmetry_received here or the wait times out forever. It is
            # already guarded everywhere it's USED (only applied if received), so degrading the WAIT
            # to width-only is correct — matches control_gripper's width-only degrade.
            while not (node.object_pose_received and node.ee_pose_received and node.gripper_width_received):
                rclpy.spin_once(node, timeout_sec=0.1)
                time.sleep(0.1)

                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    elapsed = current_time - start_time
                    missing = []
                    if not node.object_pose_received:
                        missing.append("object")
                    if not node.ee_pose_received:
                        missing.append("end-effector")
                    if not node.gripper_width_received:
                        missing.append("gripper_width")
                    node.get_logger().info(f"Waiting for ({', '.join(missing)})... ({elapsed:.1f}s)")
                    last_log_time = current_time

            elapsed = time.time() - start_time
            node.get_logger().info(f"Received pose data (waited {elapsed:.1f}s)")

        else:  # real mode (full verification)
            node.get_logger().info("Waiting for gripper width data...")
            start_time = time.time()
            last_log_time = start_time

            while not node.gripper_width_received:
                rclpy.spin_once(node, timeout_sec=0.1)
                time.sleep(0.1)

                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    elapsed = current_time - start_time
                    node.get_logger().info(f"Waiting for gripper width... ({elapsed:.1f}s)")
                    last_log_time = current_time

            elapsed = time.time() - start_time
            node.get_logger().info(f"Received gripper width (waited {elapsed:.1f}s)")

        # Verify grasp
        success, result = node.verify_grasp_dispatch(width_only=parsed_args.width_only)

    except KeyboardInterrupt:
        result = {'result': 'failure', 'object_name': parsed_args.object_name, 'mode': parsed_args.mode, 'error': 'Interrupted'}
    except Exception as e:
        result = {'result': 'failure', 'object_name': parsed_args.object_name, 'mode': parsed_args.mode, 'error': str(e)}
    finally:
        try:
            if node is not None:
                node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

    output_result(result)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

