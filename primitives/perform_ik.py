#!/usr/bin/env python3

import sys
import os
import argparse

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time
from primitives.utils.ik_solver import compute_ik
from threading import Timer

ACTION_SERVER = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

class PerformIKRunner(Node):
    def __init__(self, target_position, target_rpy, duration):
        super().__init__('perform_ik_runner')
        self.target_position = target_position
        self.target_rpy = target_rpy
        self.duration = duration
        self.client = ActionClient(self, FollowJointTrajectory, ACTION_SERVER)
        self.shutdown_called = False
        self.retry_count = 0
        self.goal_accepted = False
        self.goal_rejected = False
        self.acceptance_timer = None
        self.trajectory_completed = False
        self.trajectory_success = False
        
        if self.client.wait_for_server(timeout_sec=10.0):
            self.send_trajectory()
        else:
            self.get_logger().error("Action server not available. Exiting.")
            self.shutdown()
    
    def shutdown(self):
        if not self.shutdown_called:
            self.shutdown_called = True
            rclpy.shutdown()

    def send_trajectory(self):
        # Solve IK
        joint_angles = compute_ik(position=self.target_position, rpy=self.target_rpy)
        
        if joint_angles is None:
            self.get_logger().error("IK failed: couldn't find solution for target pose.")
            self.shutdown()
            return
        
        # Create trajectory point
        traj_point = JointTrajectoryPoint()
        traj_point.positions = joint_angles.tolist()
        traj_point.velocities = [0.0] * 6
        traj_point.time_from_start = Duration(sec=int(self.duration), nanosec=int((self.duration % 1) * 1e9))
        
        # Create goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(
            joint_names=JOINT_NAMES,
            points=[traj_point]
        )
        goal.goal_time_tolerance = Duration(sec=1)

        self.goal_accepted = False
        self.goal_rejected = False
        if self.acceptance_timer:
            self.acceptance_timer.cancel()
        self.acceptance_timer = Timer(5.0, self.check_goal_acceptance)
        self.acceptance_timer.start()
        self.client.send_goal_async(goal).add_done_callback(self.goal_response)
    
    def check_goal_acceptance(self):
        # Don't retry if goal was explicitly rejected
        if self.goal_rejected:
            return
        
        if not self.goal_accepted:
            self.retry_count += 1
            if self.retry_count <= 5:
                self.get_logger().debug(f"Goal not accepted within 5s (attempt {self.retry_count}/5). Retrying...")
                time.sleep(0.5)
                self.send_trajectory()
            else:
                self.get_logger().error("Goal not accepted after max retries. Exiting.")
                self.shutdown()

    def goal_response(self, future):
        goal_handle = future.result()
        if self.acceptance_timer:
            self.acceptance_timer.cancel()
            self.acceptance_timer = None
        
        if not goal_handle.accepted:
            self.goal_rejected = True
            self.get_logger().error("Trajectory goal rejected")
            self.shutdown()
            return
        
        self.goal_accepted = True
        self.retry_count = 0
        self.get_logger().info("Trajectory sent and accepted")
        goal_handle.get_result_async().add_done_callback(self.goal_result)

    def goal_result(self, future):
        result = future.result()
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info("Movement completed successfully")
            self.trajectory_success = True
        else:
            self.get_logger().error(f"Trajectory failed with status: {result.status}")
            self.trajectory_success = False
        self.trajectory_completed = True
        self.shutdown()

def main(args=None):
    parser = argparse.ArgumentParser(description='Perform IK and execute trajectory')
    parser.add_argument('--target-position', type=float, nargs=3, required=True,
                        metavar=('X', 'Y', 'Z'),
                        help='Target position [x, y, z] in meters')
    parser.add_argument('--target-rpy', type=float, nargs=3, required=True,
                        metavar=('ROLL', 'PITCH', 'YAW'),
                        help='Target orientation [roll, pitch, yaw] in degrees')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Time to complete the movement in seconds (default: 5.0)')
    
    args = parser.parse_args()
    
    target_position = args.target_position
    target_rpy = args.target_rpy
    duration = args.duration
    
    rclpy.init(args=None)
    node = PerformIKRunner(target_position, target_rpy, duration)
    
    try:
        # Spin until trajectory completes or timeout
        timeout = duration + 10  # Add buffer for communication overhead
        start_time = time.time()
        while rclpy.ok() and not node.trajectory_completed and (time.time() - start_time) < timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
        
        if not node.trajectory_completed:
            node.get_logger().error(f"Trajectory execution timed out after {timeout} seconds")
            sys.exit(1)
        elif not node.trajectory_success:
            node.get_logger().error("Trajectory execution failed")
            sys.exit(1)
    except KeyboardInterrupt:
        node.get_logger().info("Trajectory execution interrupted by user")
        sys.exit(1)
    finally:
        node.destroy_node()
        if not node.shutdown_called:
            rclpy.shutdown()

if __name__ == '__main__':
    main()

