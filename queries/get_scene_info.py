#!/usr/bin/env python3
"""
Get Scene Info

Combines object detection and grasp point information into a single query by:
1. Subscribing to /objects_poses_sim or /objects_poses_real (TFMessage) for object names
2. Subscribing to /grasp_points_sim or /grasp_points_real (MarkerArray) for grasp IDs
3. Loading grasp validity data from source JSON files (gripper_width_mm per grasp ID)
4. Returning combined scene information with all objects and their available grasps

Output: JSON containing all objects with their grasp information
        Objects without grasp points will have an empty grasps array
"""

import sys
import os

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import argparse
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time
from utils.data_path_finder import get_aruco_data_dir, load_assembly_order_map, load_disassembly_order_map


# =============================================================================
# GRASP DIRECTION CONFIGURATION
# =============================================================================
# Must match the configuration in grasp_points_publisher.py.
# Only gripper widths from enabled axes are reported.
# =============================================================================
ENABLE_X_AXIS_GRASPS = True   # Include gripper width for X-axis approach
ENABLE_Y_AXIS_GRASPS = False  # Include gripper width for Y-axis approach


def output_result(result, pretty=False):
    """Output JSON result with markers"""
    if pretty:
        # Compact pretty print: one line per grasp, grouped by object
        print("{")
        objects = list(result.items())
        for i, (obj_name, obj_data) in enumerate(objects):
            comma = "," if i < len(objects) - 1 else ""
            grasps = obj_data.get("grasps", [])
            # Detect whichever order key is present
            order_key = "disassembly_order" if "disassembly_order" in obj_data else "assembly_order"
            order = obj_data.get(order_key)
            order_str = f', "{order_key}": {order}' if order is not None else ""
            if not grasps:
                print(f'  "{obj_name}": {{"grasps": []{order_str}}}{comma}')
            else:
                print(f'  "{obj_name}": {{"grasps": [')
                for j, g in enumerate(grasps):
                    g_comma = "," if j < len(grasps) - 1 else ""
                    width = g["gripper_width_mm"]
                    z_val = g.get("z_height", "N/A")
                    print(f'    {{"id": {g["id"]}, "gripper_width_mm": {width}, "z_height": {z_val}}}{g_comma}')
                print(f'  ]{order_str}}}{comma}')
        print("}")
    else:
        print("__RESULT_JSON__")
        print(json.dumps(result))
        print("__END_RESULT_JSON__")


def load_grasp_validity_data():
    """Load grasp validity (gripper_width_mm) from grasp points JSON files.

    Reads numeric gripper widths from enabled grasp axes (controlled by
    ENABLE_X_AXIS_GRASPS, ENABLE_Y_AXIS_GRASPS). When multiple axes are
    enabled, picks the minimum valid width (tightest fit).

    Returns:
        Dict mapping object_name -> {grasp_id -> gripper_width_mm or None}
    """
    validity_data = {}
    try:
        data_dir = get_aruco_data_dir() / "grasp_points"
    except FileNotFoundError:
        return validity_data

    if not data_dir.exists():
        return validity_data

    for grasp_file in data_dir.glob("*_grasp_points.json"):
        try:
            with open(grasp_file, 'r') as f:
                data = json.load(f)
                object_name_json = data.get('object_name', '')
                topic_name = object_name_json

                grasp_points = data.get('grasp_points', [])
                id_to_width = {}
                for gp in grasp_points:
                    gp_id = gp.get('id', 0)
                    grasp_validity = gp.get('grasp_validity', {})
                    # Collect widths from enabled axes
                    widths = []
                    if ENABLE_X_AXIS_GRASPS:
                        w = grasp_validity.get('x_axis_gripper_width_mm')
                        if w is not None:
                            widths.append(w)
                    if ENABLE_Y_AXIS_GRASPS:
                        w = grasp_validity.get('y_axis_gripper_width_mm')
                        if w is not None:
                            widths.append(w)

                    # Use minimum width (tightest clearance) if multiple axes valid
                    id_to_width[gp_id] = min(widths) if widths else None

                validity_data[topic_name] = id_to_width
                validity_data[object_name_json] = id_to_width
        except Exception:
            continue

    return validity_data


class SceneInfoReader(Node):
    """ROS2 node to read scene information from multiple topics"""

    def __init__(self, objects_topic, grasp_points_topic, task_type='assembly'):
        super().__init__('scene_info_reader')

        # Object tracking
        self.object_names = set()
        self.objects_received = False

        # Grasp tracking
        self.grasp_ids_by_object = {}
        self.grasps_received = False

        # Pre-load gripper state data from source JSON files
        self.validity_data = load_grasp_validity_data()

        # Pre-load order data based on task type
        self.task_type = task_type
        if task_type == 'disassembly':
            self.order_map = load_disassembly_order_map()
            self.order_key = 'disassembly_order'
        else:
            self.order_map = load_assembly_order_map()
            self.order_key = 'assembly_order'

        # Subscribe to objects topic
        self.objects_sub = self.create_subscription(
            TFMessage,
            objects_topic,
            self.objects_callback,
            10
        )

        # Subscribe to grasp points topic with appropriate QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.grasp_sub = self.create_subscription(
            MarkerArray,
            grasp_points_topic,
            self.grasp_points_callback,
            qos_profile
        )

    def objects_callback(self, msg):
        """Callback for TFMessage - extract object names from child_frame_id"""
        for transform in msg.transforms:
            object_name = transform.child_frame_id
            self.object_names.add(object_name)

        if len(self.object_names) > 0:
            self.objects_received = True

    def grasp_points_callback(self, msg):
        """Callback for MarkerArray message - accumulate grasp IDs across messages.

        The publisher may not have poses for all objects in every publish cycle,
        so we merge new grasps into the running dict rather than replacing it.
        """
        for marker in msg.markers:
            object_name = marker.ns
            grasp_id = marker.id

            if object_name not in self.grasp_ids_by_object:
                self.grasp_ids_by_object[object_name] = {}

            if grasp_id not in self.grasp_ids_by_object[object_name]:
                obj_validity = self.validity_data.get(object_name, {})
                gripper_width = obj_validity.get(grasp_id)
                if gripper_width is None:
                    continue  # Skip grasps with no valid width for enabled axes
                self.grasp_ids_by_object[object_name][grasp_id] = {
                    "id": grasp_id,
                    "gripper_width_mm": gripper_width,
                    "z_height": round(marker.pose.position.z, 4)
                }

        self.grasps_received = True

    def get_scene_info(self, timeout=5.0):
        """Get combined scene information with timeout"""
        start_time = time.time()

        while rclpy.ok() and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.objects_received and self.grasps_received:
                break

        if not self.objects_received or len(self.object_names) == 0:
            return None

        # Include all objects, empty array for those without grasp points
        scene = {}
        for object_name in sorted(self.object_names):
            grasp_dict = self.grasp_ids_by_object.get(object_name, {})
            grasps = sorted(grasp_dict.values(), key=lambda x: x["id"])

            entry = {"grasps": grasps}
            if object_name in self.order_map:
                entry[self.order_key] = self.order_map[object_name]
            scene[object_name] = entry

        return scene


def main(args=None):
    parser = argparse.ArgumentParser(description='Get Scene Info')
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: sim or real')
    parser.add_argument('--task-type', type=str, default='assembly', choices=['assembly', 'disassembly'],
                       help='Task type: assembly (show assembly_order) or disassembly (show disassembly_order)')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty print output for terminal readability')

    args = parser.parse_args()

    # Determine topic names based on mode
    if args.mode == 'sim':
        objects_topic = "/objects_poses_sim"
        grasp_points_topic = "/grasp_points_sim"
    else:
        objects_topic = "/objects_poses_real"
        grasp_points_topic = "/grasp_points_real"

    reader = None
    try:
        rclpy.init()
        reader = SceneInfoReader(objects_topic, grasp_points_topic, task_type=args.task_type)
        scene_info = reader.get_scene_info(timeout=5.0)

        if scene_info is None:
            print(f"Error: Could not read scene info from ROS topics.")
            sys.exit(1)
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

    output_result(scene_info, pretty=args.pretty)
    sys.exit(0)


if __name__ == '__main__':
    main()
