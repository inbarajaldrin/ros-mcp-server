#!/usr/bin/env python3
"""
Get Current Grasp Points Pose

Reads grasp point positions from the grasp points topic by:
1. Subscribing to /grasp_points_sim (sim mode) or /grasp_points_real (real mode)
2. Reading MarkerArray messages
3. Extracting object names, grasp IDs, and x,y,z positions from markers
4. Returning a dictionary mapping object names to lists of grasp point poses

Output: JSON containing grasp point poses (id, x, y, z) per object
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
from visualization_msgs.msg import MarkerArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


class GraspPointsPoseReader(Node):
    """ROS2 node to read grasp point positions from topic"""
    def __init__(self, topic_name):
        super().__init__('grasp_points_pose_reader')
        self.topic_name = topic_name
        self.grasp_points_received = False
        self.grasp_points_poses = {}  # Dict mapping object_name -> list of grasp point poses

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.subscription = self.create_subscription(
            MarkerArray,
            topic_name,
            self.grasp_points_callback,
            qos_profile
        )

    def grasp_points_callback(self, msg):
        """Callback for MarkerArray message"""
        # Extract object names, grasp IDs, and positions from markers
        grasp_poses_by_object = {}
        seen_ids = {}  # Track seen IDs per object to avoid duplicates

        for marker in msg.markers:
            object_name = marker.ns
            grasp_id = marker.id

            if object_name not in grasp_poses_by_object:
                grasp_poses_by_object[object_name] = []
                seen_ids[object_name] = set()

            if grasp_id not in seen_ids[object_name]:
                seen_ids[object_name].add(grasp_id)
                grasp_pose = {
                    "id": grasp_id,
                    "position": {
                        "x": round(marker.pose.position.x, 4),
                        "y": round(marker.pose.position.y, 4),
                        "z": round(marker.pose.position.z, 4)
                    }
                }
                grasp_poses_by_object[object_name].append(grasp_pose)

        # Sort grasp poses by z (highest first) so topmost is first
        for object_name in grasp_poses_by_object:
            grasp_poses_by_object[object_name].sort(key=lambda g: g["position"]["z"], reverse=True)

        self.grasp_points_poses = grasp_poses_by_object
        self.grasp_points_received = True

    def get_grasp_points_poses(self, timeout=5.0):
        """Get grasp point positions from topic with timeout"""
        start_time = time.time()
        while rclpy.ok() and not self.grasp_points_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self.grasp_points_received:
            return None

        return self.grasp_points_poses


def main(args=None):
    parser = argparse.ArgumentParser(description='Get Current Grasp Points Pose')
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: sim (reads from /grasp_points_sim) or real (reads from /grasp_points_real)')

    args = parser.parse_args()

    # Determine topic name based on mode
    if args.mode == 'sim':
        topic_name = "/grasp_points_sim"
    else:  # real mode
        topic_name = "/grasp_points_real"

    # Read grasp point poses from ROS topic
    try:
        rclpy.init()
        reader = GraspPointsPoseReader(topic_name)
        grasp_points_poses = reader.get_grasp_points_poses(timeout=5.0)
        reader.destroy_node()
        rclpy.shutdown()

        if grasp_points_poses is None:
            print(f"Error: Could not read grasp point poses from ROS topic {topic_name}.")
            sys.exit(1)
    except Exception as e:
        print(f"Error: Could not read grasp point poses from ROS topic ({e}).")
        sys.exit(1)

    # Output grasp point poses as JSON
    result = {
        "grasp_points_poses": grasp_points_poses
    }

    output_result(result)
    sys.exit(0)


if __name__ == '__main__':
    main()
