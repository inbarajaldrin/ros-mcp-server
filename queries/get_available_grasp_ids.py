#!/usr/bin/env python3
"""
Get Available Grasp IDs

Reads available grasp IDs from the grasp points topic by:
1. Subscribing to /grasp_points_sim (sim mode) or /grasp_points_real (real mode)
2. Reading MarkerArray messages
3. Extracting object names (from namespace) and grasp IDs (from marker id)
4. Returning a dictionary mapping object names to lists of available grasp IDs

Output: JSON containing available grasp IDs per object
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


class GraspIdsReader(Node):
    """ROS2 node to read available grasp IDs from topic"""
    def __init__(self, topic_name):
        super().__init__('grasp_ids_reader')
        self.topic_name = topic_name
        self.grasp_ids_received = False
        self.available_grasp_ids = {}  # Dict mapping object_name -> list of grasp IDs
        
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
        # Extract object names and grasp IDs from markers
        grasp_ids_by_object = {}
        
        for marker in msg.markers:
            object_name = marker.ns
            grasp_id = marker.id
            
            if object_name not in grasp_ids_by_object:
                grasp_ids_by_object[object_name] = []
            
            if grasp_id not in grasp_ids_by_object[object_name]:
                grasp_ids_by_object[object_name].append(grasp_id)
        
        # Sort grasp IDs for each object
        for object_name in grasp_ids_by_object:
            grasp_ids_by_object[object_name].sort()
        
        self.available_grasp_ids = grasp_ids_by_object
        self.grasp_ids_received = True
    
    def get_available_grasp_ids(self, timeout=5.0):
        """Get available grasp IDs from topic with timeout"""
        start_time = time.time()
        while rclpy.ok() and not self.grasp_ids_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if not self.grasp_ids_received:
            return None
        
        return self.available_grasp_ids


def main(args=None):
    parser = argparse.ArgumentParser(description='Get Available Grasp IDs')
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: sim (reads from /grasp_points_sim) or real (reads from /grasp_points_real)')

    args = parser.parse_args()
    
    # Determine topic name based on mode
    if args.mode == 'sim':
        topic_name = "/grasp_points_sim"
    else:  # real mode
        topic_name = "/grasp_points_real"
    
    # Read grasp IDs from ROS topic
    try:
        rclpy.init()
        reader = GraspIdsReader(topic_name)
        available_grasp_ids = reader.get_available_grasp_ids(timeout=5.0)
        reader.destroy_node()
        rclpy.shutdown()

        if available_grasp_ids is None:
            print(f"Error: Could not read grasp IDs from ROS topic {topic_name}.")
            sys.exit(1)
    except Exception as e:
        print(f"Error: Could not read grasp IDs from ROS topic ({e}).")
        sys.exit(1)

    # Output available grasp IDs as JSON
    result = {
        "available_grasp_ids": available_grasp_ids
    }

    print(json.dumps(result))
    sys.exit(0)


if __name__ == '__main__':
    main()

