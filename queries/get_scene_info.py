#!/usr/bin/env python3
"""
Get Scene Info (per-object)

Consolidated query that returns everything needed to manipulate one object:
1. Grasp points with live z_heights from ROS MarkerArray topic
2. Current pose from ROS TFMessage topic
3. Target pose with all fold-symmetric valid orientations

Replaces the old get_scene_info + get_current_object_pose + get_target_object_pose trio.

Output: Flat JSON for the requested object (not keyed by name — caller already knows it)
"""

import sys
import os

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R
import time

from utils.data_path_finder import get_aruco_data_dir, get_symmetry_dir, load_assembly_config
from primitives.shared.fold_symmetry import load_symmetry_data, equivalent_orientations
from primitives.shared.config import DEFAULT_BASE_POSITION, DEFAULT_BASE_ORIENTATION


# =============================================================================
# GRASP DIRECTION CONFIGURATION
# =============================================================================
# Must match the configuration in grasp_points_publisher.py.
# Only gripper widths from enabled axes are reported.
# =============================================================================
ENABLE_X_AXIS_GRASPS = True   # Include gripper width for X-axis approach
ENABLE_Y_AXIS_GRASPS = False  # Include gripper width for Y-axis approach


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


def load_grasp_validity_data(object_name):
    """Load grasp validity (gripper_width_mm) for a single object from grasp points JSON.

    Returns:
        Dict mapping grasp_id -> gripper_width_mm, or empty dict
    """
    try:
        data_dir = get_aruco_data_dir() / "grasp_points"
    except FileNotFoundError:
        return {}

    if not data_dir.exists():
        return {}

    # Try exact match first, then glob
    grasp_file = data_dir / f"{object_name}_grasp_points.json"
    if not grasp_file.exists():
        matches = list(data_dir.glob(f"{object_name}*_grasp_points.json"))
        if matches:
            grasp_file = matches[0]
        else:
            return {}

    try:
        with open(grasp_file, 'r') as f:
            data = json.load(f)

        id_to_width = {}
        for gp in data.get('grasp_points', []):
            gp_id = gp.get('id', 0)
            grasp_validity = gp.get('grasp_validity', {})
            widths = []
            if ENABLE_X_AXIS_GRASPS:
                w = grasp_validity.get('x_axis_gripper_width_mm')
                if w is not None:
                    widths.append(w)
            if ENABLE_Y_AXIS_GRASPS:
                w = grasp_validity.get('y_axis_gripper_width_mm')
                if w is not None:
                    widths.append(w)
            id_to_width[gp_id] = min(widths) if widths else None

        return id_to_width
    except Exception:
        return {}


def compute_target_pose(assembly_config, object_name, base_position, base_quaternion):
    """Compute target position and orientation in world frame.

    Returns:
        (target_position, target_quaternion) as numpy arrays, or (None, None)
    """
    for component in assembly_config.get('components', []):
        if component.get('name') != object_name:
            continue

        # Position relative to base
        pos = component.get('position', {})
        target_rel = np.array([pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0)])

        # Orientation relative to base
        rot = component.get('rotation', {})
        quat = rot.get('quaternion', {})
        target_quat_rel = np.array([
            quat.get('x', 0.0), quat.get('y', 0.0),
            quat.get('z', 0.0), quat.get('w', 1.0)
        ])

        # Transform to world frame
        R_base = R.from_quat(base_quaternion).as_matrix()
        target_pos_abs = base_position + R_base @ target_rel

        R_target_rel = R.from_quat(target_quat_rel).as_matrix()
        R_target_abs = R_base @ R_target_rel
        target_quat_abs = R.from_matrix(R_target_abs).as_quat()

        return target_pos_abs, target_quat_abs

    return None, None


def compute_valid_orientations(target_quaternion, object_name):
    """Generate all fold-symmetric equivalent orientations as quaternion dicts.

    Returns:
        List of {"x": ..., "y": ..., "z": ..., "w": ...} dicts
    """
    symmetry_dir = str(get_symmetry_dir())
    fold_data = load_symmetry_data(object_name, symmetry_dir)

    R_target = R.from_quat(target_quaternion).as_matrix()
    equiv_matrices = equivalent_orientations(R_target, fold_data)

    orientations = []
    for R_equiv in equiv_matrices:
        quat = R.from_matrix(R_equiv).as_quat()
        orientations.append({
            "x": round(float(quat[0]), 6),
            "y": round(float(quat[1]), 6),
            "z": round(float(quat[2]), 6),
            "w": round(float(quat[3]), 6),
        })

    return orientations


class SceneInfoReader(Node):
    """ROS2 node that reads grasp points, current pose, and base pose in one spin."""

    def __init__(self, object_name, base_name, objects_topic, grasp_points_topic):
        super().__init__('scene_info_reader')

        self.object_name = object_name
        self.base_name = base_name

        # Grasp data (pre-loaded from JSON for gripper widths)
        self.validity_data = load_grasp_validity_data(object_name)
        self.grasp_points = {}  # grasp_id -> {id, gripper_width_mm, z_height}
        self.grasps_received = False

        # Current pose
        self.current_position = None
        self.current_quaternion = None
        self.pose_received = False

        # Base pose
        self.base_position = None
        self.base_quaternion = None
        self.base_received = False

        # Subscribe to objects topic (TFMessage) for current pose + base pose
        self.objects_sub = self.create_subscription(
            TFMessage, objects_topic, self.objects_callback, 10)

        # Subscribe to grasp points topic (MarkerArray) for live z_heights
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.grasp_sub = self.create_subscription(
            MarkerArray, grasp_points_topic, self.grasp_points_callback, qos_profile)

    def objects_callback(self, msg):
        """Extract current pose for object_name and base pose for base_name."""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            t = transform.transform

            if frame_id == self.object_name:
                self.current_position = np.array([
                    t.translation.x, t.translation.y, t.translation.z])
                self.current_quaternion = np.array([
                    t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])
                self.pose_received = True

            if frame_id == self.base_name:
                self.base_position = np.array([
                    t.translation.x, t.translation.y, t.translation.z])
                self.base_quaternion = np.array([
                    t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])
                self.base_received = True

    def grasp_points_callback(self, msg):
        """Accumulate grasp points for object_name only."""
        for marker in msg.markers:
            if marker.ns != self.object_name:
                continue

            grasp_id = marker.id
            if grasp_id in self.grasp_points:
                # Update full pose (live)
                self.grasp_points[grasp_id]["position"]["x"] = round(marker.pose.position.x, 4)
                self.grasp_points[grasp_id]["position"]["y"] = round(marker.pose.position.y, 4)
                self.grasp_points[grasp_id]["position"]["z"] = round(marker.pose.position.z, 4)
                self.grasp_points[grasp_id]["orientation"]["x"] = round(marker.pose.orientation.x, 4)
                self.grasp_points[grasp_id]["orientation"]["y"] = round(marker.pose.orientation.y, 4)
                self.grasp_points[grasp_id]["orientation"]["z"] = round(marker.pose.orientation.z, 4)
                self.grasp_points[grasp_id]["orientation"]["w"] = round(marker.pose.orientation.w, 4)
                continue

            gripper_width = self.validity_data.get(grasp_id)
            if gripper_width is None:
                continue  # Skip grasps with no valid width for enabled axes

            self.grasp_points[grasp_id] = {
                "id": grasp_id,
                "position": {
                    "x": round(marker.pose.position.x, 4),
                    "y": round(marker.pose.position.y, 4),
                    "z": round(marker.pose.position.z, 4),
                },
                "orientation": {
                    "x": round(marker.pose.orientation.x, 4),
                    "y": round(marker.pose.orientation.y, 4),
                    "z": round(marker.pose.orientation.z, 4),
                    "w": round(marker.pose.orientation.w, 4),
                },
            }

        if self.grasp_points:
            self.grasps_received = True

    def spin_until_ready(self, timeout=5.0):
        """Spin until we have grasps, current pose, and base pose (or timeout)."""
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.grasps_received and self.pose_received and self.base_received:
                break


def main(args=None):
    parser = argparse.ArgumentParser(description='Get Scene Info (per-object)')
    parser.add_argument('--object-name', type=str, required=True,
                        help='Name of the object to query')
    parser.add_argument('--base-name', type=str, required=True,
                        help='Name of the base/board object')
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                        help='Mode: sim or real')

    args = parser.parse_args()

    # Determine topic names
    if args.mode == 'sim':
        objects_topic = "/objects_poses_sim"
        grasp_points_topic = "/grasp_candidates_sim"
    else:
        objects_topic = "/objects_poses_real"
        grasp_points_topic = "/grasp_candidates_real"

    # --- Read live data from ROS ---
    reader = None
    try:
        rclpy.init()
        reader = SceneInfoReader(args.object_name, args.base_name,
                                 objects_topic, grasp_points_topic)
        reader.spin_until_ready(timeout=5.0)

        if not reader.pose_received:
            print(f"Error: Could not read pose for '{args.object_name}' from {objects_topic}.")
            sys.exit(1)
        if not reader.base_received:
            print(f"Error: Could not read base pose for '{args.base_name}' from {objects_topic}.")
            sys.exit(1)

        # Extract data before destroying node
        grasps = sorted(reader.grasp_points.values(), key=lambda x: x["id"])
        current_position = reader.current_position
        current_quaternion = reader.current_quaternion
        base_position = reader.base_position
        base_quaternion = reader.base_quaternion

    except Exception as e:
        print(f"Error: Could not read scene info ({e}).")
        sys.exit(1)
    finally:
        try:
            if reader is not None:
                reader.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

    # --- Current pose ---
    current_pose = {
        "position": {
            "x": round(float(current_position[0]), 4),
            "y": round(float(current_position[1]), 4),
            "z": round(float(current_position[2]), 4)
        },
        "orientation": {
            "x": round(float(current_quaternion[0]), 6),
            "y": round(float(current_quaternion[1]), 6),
            "z": round(float(current_quaternion[2]), 6),
            "w": round(float(current_quaternion[3]), 6),
        }
    }

    # --- Target pose with valid orientations ---
    assembly_config = load_assembly_config(args.base_name)
    target_pose = None

    if assembly_config:
        target_pos, target_quat = compute_target_pose(
            assembly_config, args.object_name, base_position, base_quaternion)

        if target_pos is not None:
            valid_orientations = compute_valid_orientations(target_quat, args.object_name)
            target_pose = {
                "position": {
                    "x": round(float(target_pos[0]), 4),
                    "y": round(float(target_pos[1]), 4),
                    "z": round(float(target_pos[2]), 4)
                },
                "valid_orientations": valid_orientations
            }

    # --- Build result ---
    result = {
        "grasps": grasps,
        "current_pose": current_pose,
    }
    if target_pose is not None:
        result["target_pose"] = target_pose

    output_result(result)
    sys.exit(0)


if __name__ == '__main__':
    main()
