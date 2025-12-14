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

from primitives.utils.action_libraries import pick

class PickRunner(Node):
    def __init__(self):
        super().__init__('pick_runner')
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.client = ActionClient(self, FollowJointTrajectory, '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        self.get_logger().info("Waiting for action server...")
        self.client.wait_for_server()
        self.send_pick_trajectory()

    def send_pick_trajectory(self):
        points = pick()
        if not points:
            self.get_logger().error("IK failed: couldn't compute pick position.")
            rclpy.shutdown()
            return

        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        traj.points = [JointTrajectoryPoint(
            positions=pt["positions"],
            velocities=pt["velocities"],
            time_from_start=pt["time_from_start"]
        ) for pt in points]

        goal.trajectory = traj
        goal.goal_time_tolerance = Duration(sec=1)

        self._send_goal_future = self.client.send_goal_async(goal)
        self._send_goal_future.add_done_callback(self.goal_response)

    def goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected.")
            rclpy.shutdown()
            return

        self.get_logger().info("Pick trajectory accepted.")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result)

    def goal_result(self, future):
        self.get_logger().info("Pick movement complete.")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = PickRunner()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
