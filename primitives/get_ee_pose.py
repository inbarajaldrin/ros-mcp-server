#!/usr/bin/env python3

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np

class EEPoseReader:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node('ee_pose_reader')
        
        # Default values 
        self.ee_position = np.array([-0.144, -0.435, 0.202])
        self.ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
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
            self.ee_pose_callback,
            qos_profile
        )
        
    def ee_pose_callback(self, msg: PoseStamped):
        self.ee_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.ee_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                               msg.pose.orientation.z, msg.pose.orientation.w])
        self.received_message = True
    
    def canonicalize_euler(self, orientation):
        roll, pitch, yaw = orientation
        if abs(pitch) < 1 and abs(abs(roll) - 180) < 1:
            return (0.0, 180.0, (yaw % 360)-180)
        else:
            return orientation
    
    def print_ee_pose(self):
        # Convert quaternion to Euler angles
        ee_euler = R.from_quat(self.ee_quat).as_euler('xyz', degrees=True)
        ee_euler = self.canonicalize_euler(ee_euler)
        
        print(f"EE xyz: ({1000*self.ee_position[0]:.1f}, {1000*self.ee_position[1]:.1f}, {1000*self.ee_position[2]:.1f}) mm")
        print(f"EE rpy: ({ee_euler[0]: 5.1f}, {ee_euler[1]: 5.1f}, {ee_euler[2]: 5.1f}) deg")
        print(f"EE quat: ({self.ee_quat[0]: 8.6f}, {self.ee_quat[1]: 8.6f}, {self.ee_quat[2]: 8.6f}, {self.ee_quat[3]: 8.6f}) [x,y,z,w]")
    
    def run(self):
        try:
            # Wait for the first message to arrive
            while rclpy.ok() and not self.received_message:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            
            # Print the pose once and exit
            if self.received_message:
                self.print_ee_pose()
            else:
                print("No message received.")
        except KeyboardInterrupt:
            pass
        finally:
            self.node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    reader = EEPoseReader()
    reader.run()