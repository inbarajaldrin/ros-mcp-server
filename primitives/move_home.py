import sys
import os
import json

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


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")

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
        self.success = False
        self.error = None

        if self.client.wait_for_server(timeout_sec=10.0):
            self.send_home_trajectory()
        else:
            self.error = "UR robot driver isn't running"
            self.get_logger().error("Action server not available. Exiting.")
            self.shutdown()
    
    def shutdown(self):
        if not self.shutdown_called:
            self.shutdown_called = True
            rclpy.shutdown()

    def send_home_trajectory(self):
        points = home()

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
                self.get_logger().debug(f"Goal not accepted within 5s (attempt {self.retry_count}/5). Retrying...")
                time.sleep(0.5)
                self.send_home_trajectory()
            else:
                self.error = "Goal not accepted after max retries"
                self.get_logger().error("Goal not accepted after max retries. Exiting.")
                self.shutdown()

    def goal_response(self, future):
        goal_handle = future.result()
        if self.acceptance_timer:
            self.acceptance_timer.cancel()
            self.acceptance_timer = None

        if not goal_handle.accepted:
            self.goal_rejected = True
            self.error = "External control program stopped or robot in protective stop"
            self.get_logger().error("Trajectory goal rejected")
            self.shutdown()
            return

        self.goal_accepted = True
        self.retry_count = 0
        self.get_logger().info("Trajectory sent and accepted")
        goal_handle.get_result_async().add_done_callback(self.goal_result)

    def goal_result(self, future):
        result = future.result()
        result_msg = result.result

        # Check if trajectory execution was successful
        if result_msg.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.success = True
            self.get_logger().info("Movement completed successfully")
        else:
            self.success = False
            # Map error codes to user-friendly messages
            error_messages = {
                FollowJointTrajectory.Result.INVALID_GOAL: "Trajectory rejected: invalid goal (may indicate velocity/acceleration limits exceeded or joint limits violated)",
                FollowJointTrajectory.Result.INVALID_JOINTS: "Invalid joints: joint names don't match",
                FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP: "Old header timestamp: trajectory is too old",
                FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED: "Velocity or acceleration limits exceeded. The required velocity to reach the target exceeds joint velocity limits. Enable robot in URcap to fix this.",
                FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED: "Goal tolerance violated: did not reach target position",
            }

            # Get error message or use default
            error_msg = error_messages.get(result_msg.error_code, None)

            # If no specific error code match, check for status 6 (ABORTED) which often indicates velocity limits
            if error_msg is None:
                if result.status == 6:  # ABORTED
                    error_msg = "Trajectory ABORTED: likely velocity/acceleration limits exceeded. The required velocity to reach the target exceeds joint velocity limits. Click 'Continue' in URSim/URcap to clear the error, then retry."
                else:
                    error_msg = f"Trajectory execution failed with error code: {result_msg.error_code}"

            self.error = error_msg

            # Log detailed error information for debugging
            self.get_logger().error(f"{self.error} (error_code: {result_msg.error_code}, status: {result.status})")

        self.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = HomeRunner()
    success = False
    error = None

    try:
        # Only spin if not already shutdown (e.g., if wait_for_server failed)
        if not node.shutdown_called:
            rclpy.spin(node)
        success = node.success
        error = node.error
    except KeyboardInterrupt:
        node.get_logger().info("Home movement interrupted by user")
        error = "Interrupted by user"
    except Exception as e:
        error = str(e)
    finally:
        # Get result before cleanup
        if not success and not error:
            error = node.error
        success = node.success

        try:
            node.action_client.destroy()
            node.destroy_node()
        except Exception:
            pass
        if not node.shutdown_called:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    # Build and output result
    result = {"result": "success" if success else "failure"}
    if not success and error:
        result["error"] = error

    output_result(result)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
