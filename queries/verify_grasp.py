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


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Configuration
OBJECT_TOPIC_SIM = "/objects_poses_sim"
OBJECT_TOPIC_REAL = "/objects_poses_real"
EE_TOPIC = "/tcp_pose_broadcaster/pose"

# Default radius tolerance (in meters)
DEFAULT_RADIUS = 0.06  # 6cm default radius


class VerifyGrasp(Node):
    def __init__(self, object_name, mode='sim', radius=DEFAULT_RADIUS):
        super().__init__('verify_grasp')
        
        self.object_name = object_name
        self.mode = mode
        self.radius = radius
        
        # TCP to gripper center offset distance (from TCP to gripper center along gripper Z-axis)
        # This matches the offset used in move_to_grasp.py and get_current_gripper_center_pose.py
        self.tcp_to_gripper_center_offset = 0.24  # 0.24m = 24cm (distance from TCP to gripper center)
        
        # Z-offset to match object height (calibration offset)
        # Object z: 150.43 mm, TCP z: 27.50 mm, difference: 122.93 mm = 0.1229 m
        self.gripper_center_z_offset = 0.1229  # Offset to match object height (in meters)
        
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
            # Check for exact match or with _scaled70 suffix
            if frame_id == self.object_name or frame_id == f"{self.object_name}_scaled70":
                self.object_pose_received = True
    
    def ee_callback(self, msg):
        """Callback for end-effector pose"""
        self.current_ee_pose = msg
        self.ee_pose_received = True
    
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
        offset_vector_tool_frame = np.array([0.0, 0.0, self.tcp_to_gripper_center_offset])
        
        # Transform offset vector from tool frame to world frame using quaternion
        # The quaternion represents the rotation from tool frame to world frame
        r = R.from_quat(tcp_quaternion)
        offset_vector_world = r.apply(offset_vector_tool_frame)
        
        # Compute gripper center: TCP + offset_vector_world
        # (going forward from TCP to gripper center along the tool Z-axis)
        gripper_center_position = np.array(tcp_position) + offset_vector_world
        
        # Apply z-offset to match object height
        gripper_center_position[2] += self.gripper_center_z_offset
        
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
        original_object_name = self.object_name
        object_key = None
        if self.object_name in self.current_poses:
            object_key = self.object_name
        elif f"{self.object_name}_scaled70" in self.current_poses:
            object_key = f"{self.object_name}_scaled70"
        
        if object_key is None:
            self.get_logger().error(f"Object {original_object_name} not found in poses")
            return False, None, None, None
        
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


def main(args=None):
    parser = argparse.ArgumentParser(description='Verify Grasp - Check if object is within radius from gripper center')
    parser.add_argument('--object-name', type=str, required=True, help='Name of the object to verify')
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: sim (reads from /objects_poses_sim) or real (reads from /objects_poses_real)')
    parser.add_argument('--radius', type=float, default=DEFAULT_RADIUS,
                       help=f'Radius tolerance in meters (default: {DEFAULT_RADIUS}m = {DEFAULT_RADIUS*1000:.0f}mm)')

    parsed_args = parser.parse_args()

    success = False
    error = None
    node = None

    try:
        rclpy.init()
        node = VerifyGrasp(object_name=parsed_args.object_name, mode=parsed_args.mode, radius=parsed_args.radius)

        # Wait for pose data (wait indefinitely until received)
        node.get_logger().info(f"Waiting for pose data for object: {parsed_args.object_name} and end-effector")
        start_time = time.time()
        last_log_time = start_time

        while not (node.object_pose_received and node.ee_pose_received):
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.1)

            # Log every 5 seconds to show we're still waiting
            current_time = time.time()
            if current_time - last_log_time >= 5.0:
                elapsed = current_time - start_time
                missing = []
                if not node.object_pose_received:
                    missing.append("object")
                if not node.ee_pose_received:
                    missing.append("end-effector")
                node.get_logger().info(f"Still waiting for pose data ({', '.join(missing)})... ({elapsed:.1f}s elapsed)")
                last_log_time = current_time

        elapsed = time.time() - start_time
        node.get_logger().info(f"Received pose data (waited {elapsed:.1f}s)")

        # Verify grasp (already logs the result)
        success, distance, object_pos, gripper_center_pos = node.verify_grasp()

        if not success:
            if distance is None:
                error = f"Object '{parsed_args.object_name}' not found"
            else:
                error = "Object is outside grasp radius"

    except KeyboardInterrupt:
        error = "Interrupted by user"
    except Exception as e:
        error = str(e)
    finally:
        try:
            if node is not None:
                node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

    # Build and output result
    result = {
        "result": "success" if success else "failure",
        "object_name": parsed_args.object_name,
        "mode": parsed_args.mode
    }
    if not success and error:
        result["error"] = error

    output_result(result)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

