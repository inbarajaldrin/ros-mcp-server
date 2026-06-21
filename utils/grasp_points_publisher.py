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
from pathlib import Path

# Add project root to Python path if running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

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
from utils.data_path_finder import get_aruco_data_dir


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
            # Auto-discover aruco-grasp-annotator data directory.
            # Candidate-native (schema_version 2): grasp_candidates/, NOT the legacy grasp_points/.
            aruco_data_dir = get_aruco_data_dir()
            self.data_dir = aruco_data_dir / "grasp_candidates"
        else:
            self.data_dir = Path(data_dir)
        
        # Load all grasp points JSON files
        self.grasp_data: Dict[str, dict] = {}
        self.object_name_map: Dict[str, str] = {}
        self.load_grasp_data()
        
        # Store latest object poses - separate for sim and real in default mode
        self.object_poses: Dict[str, dict] = {}
        self.object_poses_sim: Dict[str, dict] = {}
        self.object_poses_real: Dict[str, dict] = {}
        
        # Track last update time for staleness detection
        self.last_pose_update_time = None  # For single mode
        self.last_pose_update_time_sim = None  # For default mode
        self.last_pose_update_time_real = None  # For default mode
        self.pose_staleness_threshold = 1.0  # seconds - poses older than this are considered stale
        
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
        """Load all grasp CANDIDATE JSON files (schema_version 2) from data directory."""
        if not self.data_dir.exists():
            self.get_logger().warn(f"Data directory does not exist: {self.data_dir}")
            return

        # Candidate-native hard-switch: read *_grasp_candidates.json (schema_version 2), not the
        # legacy *_grasp_points.json. Each file's grasp_candidates[] carries per-candidate
        # width_mm / standoff_m / approach_quaternion (the true TCP orientation in object frame).
        pattern = "*_grasp_candidates.json"
        for grasp_file in self.data_dir.glob(pattern):
            try:
                with open(grasp_file, 'r') as f:
                    data = json.load(f)
                    object_name_json = data.get('object_name')
                    if object_name_json:
                        self.grasp_data[object_name_json] = data
                        self.object_name_map[object_name_json] = object_name_json

                        self.get_logger().info(f"  Loaded: {object_name_json} ({data.get('total_grasp_candidates', 0)} grasp candidates)")
            except Exception as e:
                self.get_logger().error(f"Error loading {grasp_file}: {e}")
    
    def objects_poses_callback(self, msg: TFMessage, source_mode=None):
        """Handle incoming object poses from TFMessage - update stored poses
        
        Args:
            msg: TFMessage containing object poses
            source_mode: 'sim' or 'real' (only used in default mode)
        """
        # Get current time for staleness tracking
        current_time = self.get_clock().now()
        
        # Determine which pose storage to use
        if self.mode == 'default' or self.mode == 'auto':
            # In default mode, store poses separately for sim and real
            if source_mode == 'sim':
                target_poses = self.object_poses_sim
                self.last_pose_update_time_sim = current_time
            elif source_mode == 'real':
                target_poses = self.object_poses_real
                self.last_pose_update_time_real = current_time
            else:
                # Fallback (shouldn't happen)
                target_poses = self.object_poses
                self.last_pose_update_time = current_time
        else:
            # Single mode: use main storage
            target_poses = self.object_poses
            self.last_pose_update_time = current_time
        
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

    def transform_candidate(self, candidate, object_pose):
        """
        Transform a grasp CANDIDATE (schema_version 2) from CAD-center frame to world (base) frame.

        Position: p_W = p_WO + R_WO @ grasp_candidate_position.
        Orientation: the candidate carries `approach_quaternion` = R_OT, the TCP/grasp frame
        orientation expressed IN THE OBJECT frame. The world TCP frame is therefore
            R_WT = R_WO @ R_OT  ->  q_WT = q_WO (X) q_approach
        so the published marker carries the TRUE world TCP frame (face-down + the candidate's
        in-plane yaw baked in), NOT the bare object orientation. move_to_grasp's top-down gate then
        recovers the approach axis and yaw directly from this quaternion — no face-down assumption
        here, no canonical/fold-symmetry matching downstream.

        Args:
            candidate: one entry of grasp_candidates[] (has grasp_candidate_position + approach_quaternion)
            object_pose: Dict with translation and quaternion of object in base frame

        Returns:
            (pos_world [3], quat_world [x,y,z,w])
        """
        # Local position (relative to CAD center)
        gp = candidate['grasp_candidate_position']
        pos_local = np.array([gp['x'], gp['y'], gp['z']])

        # Object pose in base frame
        obj_translation = object_pose['translation']
        obj_quaternion = object_pose['quaternion']  # [x, y, z, w]

        r_object_world = R.from_quat(obj_quaternion)

        # Transform position to world frame
        pos_base = obj_translation + r_object_world.as_matrix() @ pos_local

        # Orientation (UN-BAKED redesign): derive the face-down TCP at the LIVE pose from the
        # candidate's closing axis, instead of reading a frozen baked R_OT.
        #   R_WT = R_facedown(world-down, R_WO @ closing_axis)
        # +Z = world-down (approach), +X = horizontal projection of the clamp line, +Y = Z x X.
        # Azimuth-invariant; move_to_grasp's top-down gate + yaw-recovery handle the rest. Falls back
        # to the legacy baked approach_quaternion only if there is no usable (horizontal) clamp axis.
        ca = candidate.get('closing_axis')
        axis_local = {'x': [1.0, 0.0, 0.0], 'y': [0.0, 1.0, 0.0], 'z': [0.0, 0.0, 1.0]}.get(ca)
        quat_base = None
        if axis_local is not None:
            clamp_w = r_object_world.as_matrix() @ np.array(axis_local)
            horiz = np.array([clamp_w[0], clamp_w[1], 0.0])
            n = np.linalg.norm(horiz)
            if n >= 1e-6:                       # clamp line is graspable top-down at this pose
                xa = horiz / n
                za = np.array([0.0, 0.0, -1.0])
                ya = np.cross(za, xa)
                quat_base = R.from_matrix(np.column_stack([xa, ya, za])).as_quat()
        if quat_base is None:                   # legacy / non-top-down fallback
            aq = candidate.get('approach_quaternion')
            if aq is not None:
                quat_base = (r_object_world * R.from_quat(
                    np.array([aq['x'], aq['y'], aq['z'], aq['w']]))).as_quat()
            else:
                quat_base = r_object_world.as_quat()  # last resort (will not pass the top-down gate)

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
            
            # Get grasp candidates for this object
            grasp_candidates_local = self.grasp_data[object_name_json].get('grasp_candidates', [])

            # Transform and add each candidate. One marker per candidate; the composite id
            # (grasp_point_id*100 + direction_id) lets a caller address e.g. 101 (gp1/x-close)
            # vs 102 (gp1/y-close). All candidates are published; gating is done by the consumer.
            for cand in grasp_candidates_local:
                try:
                    pos_base, quat_base = self.transform_candidate(cand, object_pose)

                    marker = Marker()
                    marker.header.stamp = now.to_msg()
                    marker.header.frame_id = "base"

                    # Store object name in namespace
                    marker.ns = object_name_topic

                    # Composite candidate id = grasp_point_id*100 + direction_id
                    marker.id = cand.get('grasp_point_id', 0) * 100 + cand.get('direction_id', 0)

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
                    cid = cand.get('grasp_point_id', 0) * 100 + cand.get('direction_id', 0)
                    self.get_logger().error(f"Error transforming grasp candidate {cid} for {object_name_topic}: {e}")
        
        return marker_array
    
    def _are_poses_stale(self, last_update_time):
        """Check if poses are stale based on last update time"""
        if last_update_time is None:
            return True  # Never received poses, consider stale
        
        current_time = self.get_clock().now()
        time_since_update = (current_time - last_update_time).nanoseconds / 1e9  # Convert to seconds
        
        return time_since_update > self.pose_staleness_threshold
    
    def publish_grasp_points(self):
        """Publish grasp points for all objects with known poses"""
        if self.mode == 'default' or self.mode == 'auto':
            # Default mode: publish to both sim and real topics
            # Check if sim poses are stale
            if self._are_poses_stale(self.last_pose_update_time_sim):
                if self.object_poses_sim:
                    self.get_logger().warn(f"Sim poses are stale (last update: {self.last_pose_update_time_sim}), clearing poses")
                    self.object_poses_sim.clear()
            
            # Check if real poses are stale
            if self._are_poses_stale(self.last_pose_update_time_real):
                if self.object_poses_real:
                    self.get_logger().warn(f"Real poses are stale (last update: {self.last_pose_update_time_real}), clearing poses")
                    self.object_poses_real.clear()
            
            marker_array_sim = self._create_grasp_array_from_poses(self.object_poses_sim)
            marker_array_real = self._create_grasp_array_from_poses(self.object_poses_real)
            
            # Always publish to both (even if empty arrays)
            self.grasp_pub_sim.publish(marker_array_sim)
            self.grasp_pub_real.publish(marker_array_real)
        else:
            # Single mode: publish to single topic
            # Check if poses are stale
            if self._are_poses_stale(self.last_pose_update_time):
                if self.object_poses:
                    self.get_logger().warn(f"Poses are stale (last update: {self.last_pose_update_time}), clearing poses")
                    self.object_poses.clear()
            
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
                       help='Directory containing grasp points JSON files (default: data/grasp_points relative to project root)')
    
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