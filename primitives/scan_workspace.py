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
import json

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
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
SCAN_X_MAX = 0.4    # Maximum X position
SCAN_Y_MIN = -0.5   # Minimum Y position
SCAN_Y_MAX = -0.3   # Maximum Y position
SCAN_STEP = 0.1     # Step size for scanning grid (in meters)
MOVEMENT_DURATION = 3.0  # Duration for each movement (in seconds)

# Fixed orientation (face down gripper)
FIXED_ROLL = 0.0
FIXED_PITCH = 180.0
FIXED_YAW = 0.0


def generate_scan_path(current_x, current_y):
    """
    Generate a Boustrophedon scanning path starting from current position.
    Scans with fixed X first (moves in Y direction), then moves to next X column.
    Uses zigzag pattern for Y movement.
    
    Args:
        current_x: Current X position of the robot
        current_y: Current Y position of the robot
    
    Returns:
        List of [x, y, z] positions to scan
    """
    path = []
    # np.arange is exclusive of stop, so add tiny epsilon to include the max value
    epsilon = 1e-9
    x_positions = np.arange(SCAN_X_MIN, SCAN_X_MAX + epsilon, SCAN_STEP)
    y_positions = np.arange(SCAN_Y_MIN, SCAN_Y_MAX + epsilon, SCAN_STEP)
    
    # Find the closest X position to start from
    start_x_idx = np.argmin(np.abs(x_positions - current_x))
    start_x = x_positions[start_x_idx]
    
    # Find the closest Y position to start from
    start_y_idx = np.argmin(np.abs(y_positions - current_y))
    start_y = y_positions[start_y_idx]
    
    # Reorder X positions to start from closest to current position
    x_positions_ordered = list(x_positions[start_x_idx:]) + list(x_positions[:start_x_idx])
    
    # Boustrophedon pattern: for each X (column), scan in Y direction (zigzag)
    for i, x in enumerate(x_positions_ordered):
        # Determine Y direction for this column (zigzag: alternate up/down)
        if i % 2 == 0:
            # Even columns: scan from Y_MIN to Y_MAX
            y_ordered = list(y_positions)
        else:
            # Odd columns: scan from Y_MAX to Y_MIN
            y_ordered = list(reversed(y_positions))
        
        # For the first column (starting column), start from current Y position
        if x == start_x:
            # Find where current_y is in the ordered list
            try:
                current_y_idx = y_ordered.index(start_y)
                # Reorder to start from current Y
                y_ordered = y_ordered[current_y_idx:] + y_ordered[:current_y_idx]
            except ValueError:
                # If current_y not exactly in list, use closest
                closest_y_idx = np.argmin(np.abs(np.array(y_ordered) - current_y))
                y_ordered = y_ordered[closest_y_idx:] + y_ordered[:closest_y_idx]
        
        # Add all Y positions for this X column
        for y in y_ordered:
            path.append([x, y, SCAN_HEIGHT])
    
    return path


class ScanWorkspace(Node):
    def __init__(self, object_name):
        super().__init__('scan_workspace')

        self.object_name = object_name
        self.object_found = False
        self.current_poses = {}
        self.current_ee_position = None
        self.ee_pose_received = False
        self.scan_path = []
        self.current_waypoint_index = 0
        self.trajectory_completed = False
        self.trajectory_success = False
        self.should_exit = False  # Flag to signal main loop to exit
        self.error_message = None  # Track error for JSON output
        
        # Action client for trajectory execution
        self.action_client = ActionClient(self, FollowJointTrajectory, ACTION_SERVER)
        
        # Subscribe to current EE pose to get starting position
        ee_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            ee_qos
        )
        
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
        self.get_logger().info(f"Waiting for current EE pose to generate scan path...")
        
        # Start with first waypoint
        self.trajectory_completed = True  # Allow first movement to start
    
    def ee_pose_callback(self, msg):
        """Callback for current end-effector pose"""
        if not self.ee_pose_received:
            self.current_ee_position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
            self.ee_pose_received = True
            # Generate scan path based on current position
            self.scan_path = generate_scan_path(
                self.current_ee_position[0],
                self.current_ee_position[1]
            )
            self.get_logger().info(f"Current EE position: [{self.current_ee_position[0]:.3f}, {self.current_ee_position[1]:.3f}, {self.current_ee_position[2]:.3f}]")
            self.get_logger().info(f"Scan path contains {len(self.scan_path)} waypoints")
            self.get_logger().info(f"Scanning at fixed height: {SCAN_HEIGHT}m")
            self.get_logger().info(f"Using Boustrophedon pattern: fixed X first, zigzag in Y direction")
    
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
        # Wait for path to be generated
        if not self.ee_pose_received or len(self.scan_path) == 0:
            return
        
        # If object found, exit
        if self.object_found:
            self.get_logger().info("Object found! Stopping scan and exiting.")
            self.timer.cancel()
            self.should_exit = True
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
            self.should_exit = True
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
            self.error_message = "UR robot driver isn't running"
            self.get_logger().error("Action server not available. Exiting.")
            return False

        # Start scanning
        self.get_logger().info("Starting workspace scan...")
        # The timer will handle movement to waypoints

        return True

    def output_result_json(self, movement_type="scan_workspace"):
        """Output result in JSON format for MCP server"""
        if self.error_message:
            # Failure format
            result = {
                "result": "failure",
                "mode": "real",
                "object_name": self.object_name,
                "error": self.error_message
            }
        else:
            # Success format
            result = {
                "result": "success",
                "mode": "real",
                "object_name": self.object_name
            }

        output_result(result)


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


def main(args=None):
    parser = argparse.ArgumentParser(description='Scan Workspace - Locate object by scanning at fixed height')
    parser.add_argument('--object-name', type=str, required=True, help='Name of the object to locate')
    parser.add_argument('--mode', type=str, default='real', choices=['real'],
                       help='Mode: only "real" is supported (default: "real")')

    args = parser.parse_args()

    # Enforce real mode only
    if args.mode != 'real':
        # Create minimal node for JSON output before exiting
        result = {
            "result": "failure",
            "mode": "real",
            "object_name": args.object_name,
            "error": "Only real mode is supported for workspace scanning"
        }
        output_result(result)
        sys.exit(1)

    rclpy.init()
    node = ScanWorkspace(object_name=args.object_name)

    try:
        if not node.run():
            output_result_json_from_node(node)
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)

        # Spin until object found or scan complete
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)

        # Check result
        if node.object_found:
            node.get_logger().info("Scan workspace: SUCCESS - Object located")
            output_result_json_from_node(node)
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(0)
        else:
            node.error_message = "Object not found after scanning workspace"
            node.get_logger().error("Scan workspace: FAILED - Object not found")
            output_result_json_from_node(node)
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)

    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        sys.exit(1)
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
        node.error_message = f"Unexpected error: {e}"
        import traceback
        traceback.print_exc()
        try:
            output_result_json_from_node(node)
        except:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        sys.exit(1)
    finally:
        if rclpy.ok():
            try:
                node.destroy_node()
                rclpy.shutdown()
            except:
                pass


def output_result_json_from_node(node):
    """Helper function to call output_result_json on the node"""
    try:
        node.output_result_json()
    except:
        pass


if __name__ == '__main__':
    main()

