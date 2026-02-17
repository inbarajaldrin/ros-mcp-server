#!/usr/bin/env python3
"""
Get Current Gripper Center Pose

Reads the current TCP pose from ROS topic /tcp_pose_broadcaster/pose and calculates
the gripper center pose using the same logic as move_to_grasp.
Outputs position (xyz in mm) and orientation (rpy in degrees, quaternion) of the gripper center.
"""

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np
import json
import sys
import os

# Add project root so primitives can be imported when run as python3 queries/get_current_gripper_center_pose.py
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from primitives.shared.config import GRIPPER_CENTER_TOOL_OFFSET


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")

class GripperCenterPoseReader:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node('gripper_center_pose_reader')
        
        # TCP to gripper center offset distance (from TCP to gripper center along gripper Z-axis)
        self.tcp_to_gripper_center_offset = GRIPPER_CENTER_TOOL_OFFSET
        
        # Default values (gripper center pose)
        self.gripper_center_position = np.array([-0.144, -0.435, 0.202])
        self.gripper_center_quat = np.array([0.0, 1.0, 0.0, 0.0])
        self.received_message = False
        
        # Configure QoS to match the publisher (VOLATILE durability - default for most publishers)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,  # Changed from TRANSIENT_LOCAL to match most publishers
            depth=10
        )
        
        # Subscribe to the same topic as the localizer
        self.subscription = self.node.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.gripper_center_pose_callback,
            qos_profile
        )
    
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
        
    def gripper_center_pose_callback(self, msg: PoseStamped):
        # Get TCP pose from message
        tcp_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        tcp_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                            msg.pose.orientation.z, msg.pose.orientation.w])
        
        # Calculate gripper center pose from TCP pose
        self.gripper_center_position = self.compute_gripper_center_from_tcp(tcp_position, tcp_quat)
        # Gripper center orientation is the same as TCP orientation
        self.gripper_center_quat = tcp_quat.copy()
        self.received_message = True
    
    def canonicalize_euler(self, orientation):
        roll, pitch, yaw = orientation
        if abs(pitch) < 1 and abs(abs(roll) - 180) < 1:
            return (0.0, 180.0, (yaw % 360)-180)
        else:
            return orientation
    
    def print_gripper_center_pose(self):
        # Convert quaternion to Euler angles
        gripper_center_euler = R.from_quat(self.gripper_center_quat).as_euler('xyz', degrees=True)
        gripper_center_euler = self.canonicalize_euler(gripper_center_euler)
        
        print(f"Gripper Center xyz: ({1000*self.gripper_center_position[0]:.1f}, {1000*self.gripper_center_position[1]:.1f}, {1000*self.gripper_center_position[2]:.1f}) mm")
        print(f"Gripper Center rpy: ({gripper_center_euler[0]: 5.1f}, {gripper_center_euler[1]: 5.1f}, {gripper_center_euler[2]: 5.1f}) deg")
        print(f"Gripper Center quat: ({self.gripper_center_quat[0]: 8.6f}, {self.gripper_center_quat[1]: 8.6f}, {self.gripper_center_quat[2]: 8.6f}, {self.gripper_center_quat[3]: 8.6f}) [x,y,z,w]")
    
    def run(self):
        try:
            # Wait for the first message to arrive
            while rclpy.ok() and not self.received_message:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            
            # Print the pose once and exit
            if self.received_message:
                self.print_gripper_center_pose()
            else:
                print("No message received.")
        except KeyboardInterrupt:
            pass
        finally:
            self.node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    reader = GripperCenterPoseReader()
    reader.run()