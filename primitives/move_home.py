import sys
import os

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
from primitives.utils.action_libraries import home
from threading import Timer

ACTION_SERVER = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
HOME_MOVEMENT_DURATION = 5.0

class HomeRunner(Node):
    def __init__(self):
        super().__init__('home_runner')
        self.client = ActionClient(self, FollowJointTrajectory, ACTION_SERVER)
        self.shutdown_called = False
        self.retry_count = 0
        self.goal_accepted = False
        self.goal_rejected = False
        self.acceptance_timer = None
        
        if self.client.wait_for_server(timeout_sec=10.0):
            self.get_logger().info("Action server available. Sending home trajectory...")
            self.send_home_trajectory()
        else:
            self.get_logger().error("Action server not available. Exiting.")
    
    def shutdown(self):
        if not self.shutdown_called:
            self.shutdown_called = True
            rclpy.shutdown()

    def send_home_trajectory(self):
        points = home()
        if not points:
            self.get_logger().error("IK failed: couldn't compute home position.")
            self.shutdown()
            return

        for pt in points:
            pt["time_from_start"] = Duration(sec=int(HOME_MOVEMENT_DURATION))

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(
            joint_names=JOINT_NAMES,
            points=[JointTrajectoryPoint(positions=pt["positions"], velocities=pt["velocities"], time_from_start=pt["time_from_start"]) for pt in points]
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
                self.get_logger().warn(f"Goal not accepted within 5s (attempt {self.retry_count}/5). Retrying...")
                time.sleep(0.5)
                self.send_home_trajectory()
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
        self.get_logger().info("Home trajectory accepted.")
        goal_handle.get_result_async().add_done_callback(self.goal_result)

    def goal_result(self, future):
        self.get_logger().info("Home movement complete.")
        self.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = HomeRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Home movement interrupted by user")
    finally:
        node.destroy_node()
        if not node.shutdown_called:
            rclpy.shutdown()

if __name__ == '__main__':
    main()
