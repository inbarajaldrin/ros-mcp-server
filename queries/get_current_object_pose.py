#!/usr/bin/env python3
"""
Get Current Object Pose

Reads the current object pose(s) from ROS topic by:
1. Subscribing to /objects_poses_sim (sim mode) or /objects_poses_real (real mode)
2. Reading TFMessage messages
3. Finding the transform for the specified object (or all objects if --all is used)
4. Extracting position and orientation

Output: JSON containing the current object pose(s) (position and orientation)
"""

import sys
import os

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import numpy as np
import argparse
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R
import time


def output_result(result, pretty=False):
    """Output JSON result with markers"""
    if pretty:
        # Check if this is multi-object (--all) or single object
        if "object_poses" in result:
            # Multi-object: compact format with one object per block
            print("{")
            objects = list(result["object_poses"].items())
            for i, (obj_name, pose) in enumerate(objects):
                comma = "," if i < len(objects) - 1 else ""
                pos = pose["position"]
                ori = pose["orientation"]
                print(f'  "{obj_name}": {{')
                print(f'    "position": {{"x": {pos["x"]}, "y": {pos["y"]}, "z": {pos["z"]}}},')
                print(f'    "orientation": {{')
                print(f'      "quat": {{"x": {ori["quat"]["x"]}, "y": {ori["quat"]["y"]}, "z": {ori["quat"]["z"]}, "w": {ori["quat"]["w"]}}},')
                print(f'      "rpy": {{"roll": {ori["rpy"]["roll"]}, "pitch": {ori["rpy"]["pitch"]}, "yaw": {ori["rpy"]["yaw"]}}}')
                print(f'    }}')
                print(f'  }}{comma}')
            print("}")
        else:
            # Single object
            print(json.dumps(result, indent=2))
    else:
        print("__RESULT_JSON__")
        print(json.dumps(result))
        print("__END_RESULT_JSON__")


class ObjectPoseReader(Node):
    """ROS2 node to read object pose from topic"""
    def __init__(self, object_name, topic_name, get_all=False):
        super().__init__('object_pose_reader')
        self.object_name = object_name
        self.get_all = get_all
        self.current_poses = {}
        self.pose_received = False
        
        self.subscription = self.create_subscription(
            TFMessage,
            topic_name,
            self.pose_callback,
            10
        )
    
    def pose_callback(self, msg):
        """Callback for pose data"""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            self.current_poses[frame_id] = transform
            # If getting all, mark as received once we have any data
            if self.get_all:
                self.pose_received = True
            elif frame_id == self.object_name:
                self.pose_received = True
    
    def get_object_pose(self, timeout=5.0):
        """Get object pose from topic with timeout"""
        start_time = time.time()
        while rclpy.ok() and not self.pose_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.get_all:
            # Return all poses
            all_poses = {}
            for frame_id, transform in self.current_poses.items():
                all_poses[frame_id] = {
                    'position': np.array([
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z
                    ]),
                    'orientation': np.array([
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w
                    ])
                }
            return all_poses
        
        if self.object_name not in self.current_poses:
            return None, None
        
        transform = self.current_poses[self.object_name].transform
        position = np.array([
            transform.translation.x,
            transform.translation.y,
            transform.translation.z
        ])
        quaternion = np.array([
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w
        ])
        
        return position, quaternion


def main(args=None):
    parser = argparse.ArgumentParser(description='Get Current Object Pose')
    parser.add_argument('--object-name', type=str, default=None, help='Name of the object (required if --all is not used)')
    parser.add_argument('--all', action='store_true', help='Get poses for all objects (mutually exclusive with --object-name)')
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: sim (reads from /objects_poses_sim) or real (reads from /objects_poses_real)')
    parser.add_argument('--pretty', action='store_true', help='Pretty print output for terminal readability')

    args = parser.parse_args()
    
    # Validate that either --object-name or --all is provided
    if not args.all and args.object_name is None:
        parser.error("Either --object-name or --all must be provided")
    
    if args.all and args.object_name is not None:
        parser.error("--all and --object-name are mutually exclusive")
    
    # Determine topic name based on mode
    if args.mode == 'sim':
        topic_name = "/objects_poses_sim"
    else:  # real mode
        topic_name = "/objects_poses_real"
    
    # Read object pose(s) from ROS topic
    try:
        rclpy.init()
        reader = ObjectPoseReader(args.object_name, topic_name, get_all=args.all)
        
        if args.all:
            all_poses = reader.get_object_pose(timeout=5.0)
            reader.destroy_node()
            rclpy.shutdown()

            if all_poses is None or len(all_poses) == 0:
                print(f"Error: Could not read object poses from ROS topic {topic_name}.")
                sys.exit(1)
            
            # Format all poses as JSON
            result = {
                "object_poses": {}
            }
            for object_name, pose_data in all_poses.items():
                # Convert quaternion to RPY (roll, pitch, yaw) in degrees
                rpy = R.from_quat(pose_data['orientation']).as_euler('xyz', degrees=True)
                result["object_poses"][object_name] = {
                    "position": {
                        "x": round(float(pose_data['position'][0]), 4),
                        "y": round(float(pose_data['position'][1]), 4),
                        "z": round(float(pose_data['position'][2]), 4)
                    },
                    "orientation": {
                        "quat": {
                            "x": round(float(pose_data['orientation'][0]), 6),
                            "y": round(float(pose_data['orientation'][1]), 6),
                            "z": round(float(pose_data['orientation'][2]), 6),
                            "w": round(float(pose_data['orientation'][3]), 6)
                        },
                    }
                }
        else:
            object_position, object_quaternion = reader.get_object_pose(timeout=5.0)
            reader.destroy_node()
            rclpy.shutdown()

            if object_position is None or object_quaternion is None:
                print(f"Error: Could not read object pose for '{args.object_name}' from ROS topic {topic_name}.")
                sys.exit(1)
            
            # Convert quaternion to RPY (roll, pitch, yaw) in degrees
            rpy = R.from_quat(object_quaternion).as_euler('xyz', degrees=True)

            # Output single object pose as JSON
            result = {
                "object_pose": {
                    "position": {
                        "x": round(float(object_position[0]), 4),
                        "y": round(float(object_position[1]), 4),
                        "z": round(float(object_position[2]), 4)
                    },
                    "orientation": {
                        "quat": {
                            "x": round(float(object_quaternion[0]), 6),
                            "y": round(float(object_quaternion[1]), 6),
                            "z": round(float(object_quaternion[2]), 6),
                            "w": round(float(object_quaternion[3]), 6)
                        },
                    }
                }
            }
    except Exception as e:
        print(f"Error: Could not read object pose from ROS topic ({e}).")
        sys.exit(1)

    output_result(result, pretty=args.pretty)
    sys.exit(0)


if __name__ == '__main__':
    main()

