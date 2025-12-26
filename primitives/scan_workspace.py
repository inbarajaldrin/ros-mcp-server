#!/usr/bin/env python3
"""
Scan Workspace - Scans workspace at fixed height to locate object

The algorithm:
1. Only works in real mode (raises error if sim mode is used)
2. Defines a predefined scanning path across x,y at a fixed z height
3. Subscribes to /objects_poses_real to detect object
4. Moves along the path, checking for the object at each waypoint
5. Stops as soon as the object is detected
6. Exits
"""

import sys
import os

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_msgs.msg import TFMessage
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import argparse
import time
from primitives.utils.ik_solver import compute_ik

# Configuration
OBJECT_TOPIC_REAL = "/objects_poses_real"
ACTION_SERVER = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

# Scanning parameters
SCAN_HEIGHT = 0.35  # Fixed Z height for scanning (in meters)
SCAN_X_MIN = -0.4   # Minimum X position
SCAN_X_MAX = 0.1    # Maximum X position
SCAN_Y_MIN = -0.5   # Minimum Y position
SCAN_Y_MAX = -0.3   # Maximum Y position
SCAN_STEP = 0.1     # Step size for scanning grid (in meters)
MOVEMENT_DURATION = 3.0  # Duration for each movement (in seconds)

# Fixed orientation (face down gripper)
FIXED_ROLL = 0.0
FIXED_PITCH = 180.0
FIXED_YAW = 0.0


def generate_scan_path():
    """
    Generate a predefined scanning path across x,y at fixed z.
    Uses a raster/zigzag pattern.
    
    Returns:
        List of [x, y, z] positions to scan
    """
    path = []
    x_positions = np.arange(SCAN_X_MIN, SCAN_X_MAX + SCAN_STEP, SCAN_STEP)
    y_positions = np.arange(SCAN_Y_MIN, SCAN_Y_MAX + SCAN_STEP, SCAN_STEP)
    
    # Zigzag pattern: alternate direction for each row
    for i, y in enumerate(y_positions):
        if i % 2 == 0:
            # Left to right
            for x in x_positions:
                path.append([x, y, SCAN_HEIGHT])
        else:
            # Right to left
            for x in reversed(x_positions):
                path.append([x, y, SCAN_HEIGHT])
    
    return path


class ScanWorkspace(Node):
    def __init__(self, object_name):
        super().__init__('scan_workspace')
        
        self.object_name = object_name
        self.object_found = False
        self.current_poses = {}
        self.scan_path = generate_scan_path()
        self.current_waypoint_index = 0
        self.trajectory_completed = False
        self.trajectory_success = False
        
        # Action client for trajectory execution
        self.action_client = ActionClient(self, FollowJointTrajectory, ACTION_SERVER)
        
        # Subscribe to object poses (real mode only)
        self.object_sub = self.create_subscription(
            TFMessage,
            OBJECT_TOPIC_REAL,
            self.object_callback,
            10
        )
        
        # Timer to check for object and move to next waypoint
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.movement_in_progress = False
        
        self.get_logger().info(f"Scanning workspace for object: {object_name}")
        self.get_logger().info(f"Scan path contains {len(self.scan_path)} waypoints")
        self.get_logger().info(f"Scanning at fixed height: {SCAN_HEIGHT}m")
        
        # Start with first waypoint
        self.trajectory_completed = True  # Allow first movement to start
    
    def object_callback(self, msg):
        """Callback for object poses - check if target object is detected"""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            self.current_poses[frame_id] = transform
            
            # Check for exact match or with _scaled70 suffix
            if frame_id == self.object_name or frame_id == f"{self.object_name}_scaled70":
                if not self.object_found:
                    self.object_found = True
                    self.get_logger().info(f"Object '{self.object_name}' detected at waypoint {self.current_waypoint_index + 1}/{len(self.scan_path)}")
                    # Stop moving - cancel any ongoing trajectory
                    self.cancel_current_movement()
    
    def cancel_current_movement(self):
        """Cancel any ongoing trajectory movement"""
        # Note: ActionClient doesn't have a direct cancel method in rclpy
        # The movement will complete naturally, but we won't move to next waypoint
        pass
    
    def execute_trajectory(self, position):
        """
        Execute a trajectory to move to the specified position.
        
        Args:
            position: [x, y, z] target position
            
        Returns:
            True if trajectory was sent successfully, False otherwise
        """
        # Compute IK for target position
        rpy = [FIXED_ROLL, FIXED_PITCH, FIXED_YAW]
        joint_angles = compute_ik(position, rpy)
        
        if joint_angles is None:
            self.get_logger().warn(f"IK failed for position {position}, skipping waypoint")
            return False
        
        # Create trajectory point
        traj_point = JointTrajectoryPoint()
        traj_point.positions = [float(x) for x in joint_angles]
        traj_point.velocities = [0.0] * 6
        traj_point.time_from_start = Duration(sec=int(MOVEMENT_DURATION))
        
        # Create trajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = JOINT_NAMES
        traj_msg.points = [traj_point]
        
        # Create goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj_msg
        goal.goal_time_tolerance = Duration(sec=1)
        
        # Reset flags
        self.trajectory_completed = False
        self.trajectory_success = False
        self.movement_in_progress = True
        
        # Send goal
        self.get_logger().info(f"Moving to waypoint {self.current_waypoint_index + 1}/{len(self.scan_path)}: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        send_goal_future = self.action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)
        
        return True
    
    def goal_response_callback(self, future):
        """Callback when goal is accepted/rejected"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            self.trajectory_completed = True
            self.trajectory_success = False
            return
        
        self.get_logger().debug("Trajectory goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)
    
    def goal_result_callback(self, future):
        """Callback when trajectory execution completes"""
        result = future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.trajectory_success = True
            self.get_logger().debug("Trajectory completed successfully")
        else:
            self.trajectory_success = False
            self.get_logger().warn(f"Trajectory completed with error code: {result.error_code}")
        
        self.trajectory_completed = True
    
    def timer_callback(self):
        """Timer callback to check for object and move to next waypoint"""
        # If object found, exit
        if self.object_found:
            self.get_logger().info("Object found! Stopping scan and exiting.")
            self.timer.cancel()
            rclpy.shutdown()
            return
        
        # If currently moving, wait for trajectory to complete
        if self.movement_in_progress:
            if not self.trajectory_completed:
                return
            # Movement just completed
            self.movement_in_progress = False
        
        # Check if we've completed all waypoints
        if self.current_waypoint_index >= len(self.scan_path):
            self.get_logger().warn(f"Scan complete: Object '{self.object_name}' not found after scanning {len(self.scan_path)} waypoints")
            self.timer.cancel()
            rclpy.shutdown()
            return
        
        # Move to next waypoint
        waypoint = self.scan_path[self.current_waypoint_index]
        if self.execute_trajectory(waypoint):
            self.movement_in_progress = True
            self.current_waypoint_index += 1
        else:
            # If IK failed, skip this waypoint and try next
            self.get_logger().warn(f"Skipping waypoint {self.current_waypoint_index + 1} due to IK failure")
            self.current_waypoint_index += 1
            # Don't set movement_in_progress, allow next waypoint to be tried immediately
    
    def run(self):
        """Run the scanning process"""
        # Wait for action server
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Action server not available. Exiting.")
            return False
        
        # Start scanning
        self.get_logger().info("Starting workspace scan...")
        # The timer will handle movement to waypoints
        
        return True


def main(args=None):
    parser = argparse.ArgumentParser(description='Scan Workspace - Locate object by scanning at fixed height')
    parser.add_argument('--object-name', type=str, required=True, help='Name of the object to locate')
    parser.add_argument('--mode', type=str, default='real', choices=['real'],
                       help='Mode: only "real" is supported (default: "real")')
    
    args = parser.parse_args()
    
    # Enforce real mode only
    if args.mode != 'real':
        print("Error: scan_workspace only works in real mode. Use --mode real")
        sys.exit(1)
    
    rclpy.init()
    node = ScanWorkspace(object_name=args.object_name)
    
    try:
        if not node.run():
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)
        
        # Spin until object found or scan complete
        rclpy.spin(node)
        
        # Check result
        if node.object_found:
            node.get_logger().info("Scan workspace: SUCCESS - Object located")
            sys.exit(0)
        else:
            node.get_logger().error("Scan workspace: FAILED - Object not found")
            sys.exit(1)
            
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()

