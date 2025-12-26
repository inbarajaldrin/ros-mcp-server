#!/usr/bin/env python3
"""
Get Available Objects

Reads available object names from ROS topic by:
1. Subscribing to /objects_poses_sim (sim mode) or /objects_poses_real (real mode)
2. Reading TFMessage messages
3. Extracting object names from child_frame_id
4. Returning a list of unique object names

Output: JSON containing list of available object names
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
import time


class ObjectsReader(Node):
    """ROS2 node to read available object names from topic"""
    def __init__(self, topic_name):
        super().__init__('objects_reader')
        self.topic_name = topic_name
        self.object_names = set()  # Use set to avoid duplicates
        self.objects_received = False
        
        self.subscription = self.create_subscription(
            TFMessage,
            topic_name,
            self.objects_callback,
            10
        )
    
    def objects_callback(self, msg):
        """Callback for TFMessage - extract object names from child_frame_id"""
        for transform in msg.transforms:
            object_name = transform.child_frame_id
            self.object_names.add(object_name)
        
        # Mark as received once we have data
        if len(self.object_names) > 0:
            self.objects_received = True
    
    def get_available_objects(self, timeout=5.0):
        """Get available object names from topic with timeout"""
        start_time = time.time()
        while rclpy.ok() and not self.objects_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if not self.objects_received or len(self.object_names) == 0:
            return None
        
        # Convert to sorted list for consistent output
        return sorted(list(self.object_names))


def main(args=None):
    parser = argparse.ArgumentParser(description='Get Available Objects')
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: sim (reads from /objects_poses_sim) or real (reads from /objects_poses_real)')

    args = parser.parse_args()
    
    # Determine topic name based on mode
    if args.mode == 'sim':
        topic_name = "/objects_poses_sim"
    else:  # real mode
        topic_name = "/objects_poses_real"
    
    # Read object names from ROS topic
    try:
        rclpy.init()
        reader = ObjectsReader(topic_name)
        available_objects = reader.get_available_objects(timeout=5.0)
        reader.destroy_node()
        rclpy.shutdown()

        if available_objects is None:
            print(f"Error: Could not read object names from ROS topic {topic_name}.")
            sys.exit(1)
    except Exception as e:
        print(f"Error: Could not read object names from ROS topic ({e}).")
        sys.exit(1)

    # Output available object names as JSON
    result = {
        "available_objects": available_objects
    }

    print(json.dumps(result, indent=2))
    sys.exit(0)


if __name__ == '__main__':
    main()

