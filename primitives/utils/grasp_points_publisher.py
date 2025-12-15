#!/usr/bin/env python3
"""
Grasp Points Publisher
Reads object poses from topic and publishes grasp points to topic.
Uses grasp points data from JSON files and transforms them using object poses.

Supports three modes:
- sim: Uses /objects_poses_sim and /grasp_points_sim topics
- real: Uses /objects_poses_real and /grasp_points_real topics
- default/auto: Publishes to both sim and real topics

Usage:
    python3 grasp_points_publisher.py [--mode sim|real|default]
"""

import sys

# Check Python version - ROS2 Humble requires Python 3.10
if sys.version_info[:2] != (3, 10):
    print("Error: ROS2 Humble requires Python 3.10")
    print(f"Current Python version: {sys.version}")
    print("\nSolutions:")
    print("1. Deactivate conda environment: conda deactivate")
    print("2. Use python3.10 directly: python3.10 src/grasp_candidates/grasp_points_publisher.py")
    print("3. Source ROS2 setup.bash which should set the correct Python")
    sys.exit(1)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
import json
import math
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
from .data_path_finder import get_aruco_data_dir


class GraspPointsPublisher(Node):
    """ROS2 node that publishes grasp points based on object poses"""
    
    
    def __init__(self, objects_poses_topic=None, 
                 grasp_points_topic=None,
                 data_dir=None,
                 mode='default'):
        super().__init__('grasp_points_publisher')
        
        self.mode = mode  # 'sim', 'real', or 'default'
        
        # Set up data directory
        if data_dir is None:
            # Auto-discover aruco-grasp-annotator data directory
            aruco_data_dir = get_aruco_data_dir()
            self.data_dir = aruco_data_dir / "grasp"
        else:
            self.data_dir = Path(data_dir)
        
        # Load all grasp points JSON files
        self.grasp_data: Dict[str, dict] = {}
        # Map from topic object names (e.g., "fork_yellow") to JSON object names (e.g., "fork_yellow_scaled70")
        self.object_name_map: Dict[str, str] = {}
        self.load_grasp_data()
        
        # Store latest object poses - separate for sim and real in default mode
        self.object_poses: Dict[str, dict] = {}
        self.object_poses_sim: Dict[str, dict] = {}
        self.object_poses_real: Dict[str, dict] = {}
        
        # QoS profile for subscriptions and publishers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Handle different modes
        if self.mode == 'default' or self.mode == 'auto':
            # Default mode: subscribe to both sim and real, publish to both
            self.objects_poses_topic_sim = "/objects_poses_sim"
            self.objects_poses_topic_real = "/objects_poses_real"
            self.grasp_points_topic_sim = "/grasp_points_sim"
            self.grasp_points_topic_real = "/grasp_points_real"
            
            # Create subscriptions for both sim and real
            self.pose_sub_sim = self.create_subscription(
                TFMessage,
                self.objects_poses_topic_sim,
                lambda msg: self.objects_poses_callback(msg, 'sim'),
                qos_profile
            )
            self.pose_sub_real = self.create_subscription(
                TFMessage,
                self.objects_poses_topic_real,
                lambda msg: self.objects_poses_callback(msg, 'real'),
                qos_profile
            )
            
            # Create publishers for both sim and real
            self.grasp_pub_sim = self.create_publisher(
                MarkerArray,
                self.grasp_points_topic_sim,
                qos_profile
            )
            self.grasp_pub_real = self.create_publisher(
                MarkerArray,
                self.grasp_points_topic_real,
                qos_profile
            )
            
            self.get_logger().info(f"Grasp Points Publisher started (DEFAULT/AUTO mode)")
            self.get_logger().info(f"Subscribing to: {self.objects_poses_topic_sim} and {self.objects_poses_topic_real}")
            self.get_logger().info(f"Publishing to: {self.grasp_points_topic_sim} and {self.grasp_points_topic_real}")
        else:
            # Single mode: sim or real
            if objects_poses_topic is None:
                if self.mode == 'sim':
                    self.objects_poses_topic = "/objects_poses_sim"
                else:
                    self.objects_poses_topic = "/objects_poses_real"
            else:
                self.objects_poses_topic = objects_poses_topic
            
            if grasp_points_topic is None:
                if self.mode == 'sim':
                    self.grasp_points_topic = "/grasp_points_sim"
                else:
                    self.grasp_points_topic = "/grasp_points_real"
            else:
                self.grasp_points_topic = grasp_points_topic
            
            # Create single subscription
            self.pose_sub = self.create_subscription(
                TFMessage,
                self.objects_poses_topic,
                self.objects_poses_callback,
                qos_profile
            )
            
            # Create single publisher
            self.grasp_pub = self.create_publisher(
                MarkerArray,
                self.grasp_points_topic,
                qos_profile
            )
            
            self.get_logger().info(f"Grasp Points Publisher started")
            self.get_logger().info(f"Subscribing to: {self.objects_poses_topic}")
            self.get_logger().info(f"Publishing to: {self.grasp_points_topic}")
            self.get_logger().info(f"Mode: {self.mode.upper()}")
        
        # Timer to publish grasp points periodically
        # Lower frequency to reduce race conditions
        self.publish_timer = self.create_timer(0.2, self.publish_grasp_points)  # 5 Hz
        
        self.get_logger().info(f"Data directory: {self.data_dir}")
        self.get_logger().info(f"Loaded grasp data for {len(self.grasp_data)} objects")
        self.get_logger().info(f"Using standard ROS2 visualization_msgs/MarkerArray")
    
    def load_grasp_data(self):
        """Load all grasp points JSON files from data directory"""
        if not self.data_dir.exists():
            self.get_logger().warn(f"Data directory does not exist: {self.data_dir}")
            return
        
        # Find all grasp points JSON files
        pattern = "*_grasp_points_all_markers.json"
        for grasp_file in self.data_dir.glob(pattern):
            try:
                with open(grasp_file, 'r') as f:
                    data = json.load(f)
                    object_name_json = data.get('object_name')
                    if object_name_json:
                        # Store with full name
                        self.grasp_data[object_name_json] = data
                        
                        # Create mapping: topic name (without _scaled70) -> JSON name (with _scaled70)
                        # Also try direct match and without _scaled70
                        topic_name = object_name_json.replace('_scaled70', '')
                        self.object_name_map[topic_name] = object_name_json
                        # Also allow direct match
                        self.object_name_map[object_name_json] = object_name_json
                        
                        self.get_logger().info(f"  Loaded: {object_name_json} ({data.get('total_grasp_points', 0)} grasp points)")
                        self.get_logger().debug(f"    Mapped topic name '{topic_name}' -> JSON name '{object_name_json}'")
            except Exception as e:
                self.get_logger().error(f"Error loading {grasp_file}: {e}")
    
    def objects_poses_callback(self, msg: TFMessage, source_mode=None):
        """Handle incoming object poses from TFMessage - update stored poses
        
        Args:
            msg: TFMessage containing object poses
            source_mode: 'sim' or 'real' (only used in default mode)
        """
        # Determine which pose storage to use
        if self.mode == 'default' or self.mode == 'auto':
            # In default mode, store poses separately for sim and real
            if source_mode == 'sim':
                target_poses = self.object_poses_sim
            elif source_mode == 'real':
                target_poses = self.object_poses_real
            else:
                # Fallback (shouldn't happen)
                target_poses = self.object_poses
        else:
            # Single mode: use main storage
            target_poses = self.object_poses
        
        # Clear all existing poses for this source first
        target_poses.clear()
        
        # If message is empty, we're done (poses already cleared)
        if not msg.transforms:
            return
        
        # Store poses for all objects in the message
        for transform in msg.transforms:
            object_name = transform.child_frame_id
            trans = transform.transform.translation
            rot = transform.transform.rotation
            
            target_poses[object_name] = {
                'translation': np.array([trans.x, trans.y, trans.z]),
                'quaternion': np.array([rot.x, rot.y, rot.z, rot.w]),
                'header': transform.header
            }

    def transform_grasp_point(self, grasp_point_local, object_pose):
        """
        Transform grasp point from CAD center frame to base frame using object pose.
        Grasp point inherits the orientation of the object.

        Args:
            grasp_point_local: Dict with position (x, y, z) relative to CAD center
            object_pose: Dict with translation and quaternion of object in base frame
        
        Returns:
            Transformed position and orientation in base frame
        """
        # Local position (relative to CAD center)
        pos_local = np.array([
            grasp_point_local['position']['x'],
            grasp_point_local['position']['y'],
            grasp_point_local['position']['z']
        ])
        
        # Object pose in base frame
        obj_translation = object_pose['translation']
        obj_quaternion = object_pose['quaternion']
        
        # Create rotation matrix from quaternion
        r_object_world = R.from_quat(obj_quaternion)
        rot_matrix = r_object_world.as_matrix()

        # Transform position to world frame
        pos_base = obj_translation + rot_matrix @ pos_local

        # Grasp point inherits object's orientation directly - no conversion needed
        # 
        # Note: The move_to_grasp code ensures the gripper is always pointing down
        # (face-down, pitch=180°), so we can use the object's orientation as-is.
        # The approach vector [0,0,1] defined in the JSON file only affects position
        # transformation, not orientation. If grasp point orientation differs from
        # object orientation in the future, the code will need to be updated to handle
        # that transformation.
        quat_base = obj_quaternion

        return pos_base, quat_base

    def _create_grasp_array_from_poses(self, object_poses_dict):
        """Create a MarkerArray message from a dictionary of object poses"""
        marker_array = MarkerArray()
        now = self.get_clock().now()
        
        # If no object poses or no grasp data, return empty array
        if not object_poses_dict or not self.grasp_data:
            return marker_array
        
        # Process each object with a pose
        for object_name_topic, object_pose in object_poses_dict.items():
            # Find matching grasp data
            object_name_json = self.object_name_map.get(object_name_topic, object_name_topic)
            if object_name_json not in self.grasp_data:
                continue
            
            # Get grasp points for this object
            grasp_points_local = self.grasp_data[object_name_json].get('grasp_points', [])
            
            # Transform and add each grasp point
            for gp_local in grasp_points_local:
                try:
                    pos_base, quat_base = self.transform_grasp_point(gp_local, object_pose)
                    
                    marker = Marker()
                    marker.header.stamp = now.to_msg()
                    marker.header.frame_id = "base"
                    
                    # Store object name in namespace
                    marker.ns = object_name_topic
                    
                    # Store grasp ID
                    marker.id = gp_local.get('id', 0)
                    
                    # Marker visualization settings
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    
                    # Position
                    marker.pose.position.x = float(pos_base[0])
                    marker.pose.position.y = float(pos_base[1])
                    marker.pose.position.z = float(pos_base[2])
                    
                    # Orientation (from object pose)
                    marker.pose.orientation.x = float(quat_base[0])
                    marker.pose.orientation.y = float(quat_base[1])
                    marker.pose.orientation.z = float(quat_base[2])
                    marker.pose.orientation.w = float(quat_base[3])
                    
                    # Visualization settings - small green spheres
                    marker.scale.x = 0.02  # 2cm diameter
                    marker.scale.y = 0.02
                    marker.scale.z = 0.02
                    
                    # Color: Green with full opacity
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                    
                    # Lifetime (0 = forever)
                    marker.lifetime.sec = 0
                    marker.lifetime.nanosec = 0
                    
                    marker_array.markers.append(marker)
                    
                except Exception as e:
                    self.get_logger().error(f"Error transforming grasp point {gp_local.get('id')} for {object_name_topic}: {e}")
        
        return marker_array
    
    def publish_grasp_points(self):
        """Publish grasp points for all objects with known poses"""
        if self.mode == 'default' or self.mode == 'auto':
            # Default mode: publish to both sim and real topics
            marker_array_sim = self._create_grasp_array_from_poses(self.object_poses_sim)
            marker_array_real = self._create_grasp_array_from_poses(self.object_poses_real)
            
            # Always publish to both (even if empty arrays)
            self.grasp_pub_sim.publish(marker_array_sim)
            self.grasp_pub_real.publish(marker_array_real)
        else:
            # Single mode: publish to single topic
            marker_array = self._create_grasp_array_from_poses(self.object_poses)
            # Always publish (even if empty array)
            self.grasp_pub.publish(marker_array)


def main(args=None):
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Grasp Points Publisher Node')
    parser.add_argument('--objects-poses-topic', type=str, default=None,
                       help='Topic name for object poses subscription (default: based on mode)')
    parser.add_argument('--grasp-points-topic', type=str, default=None,
                       help='Topic name for grasp points publication (default: based on mode)')
    parser.add_argument('--mode', type=str, default='default', choices=['sim', 'real', 'default', 'auto'],
                       help='Mode: "sim" for simulation only, "real" for real robot only, "default"/"auto" to automatically publish to both based on topic availability. Default: default')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing grasp points JSON files (default: data/grasp relative to project root)')
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    rclpy.init(args=None)
    
    node = GraspPointsPublisher(
        objects_poses_topic=args.objects_poses_topic,
        grasp_points_topic=args.grasp_points_topic,
        data_dir=args.data_dir,
        mode=args.mode
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()