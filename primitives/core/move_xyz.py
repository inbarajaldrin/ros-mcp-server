#!/usr/bin/env python3

import sys
import os
import json
import argparse
import time
import numpy as np
# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from threading import Timer

from primitives.shared.ik import IKSolver, IKSolverConfig, rpy_to_matrix
from primitives.shared.velocity_profiles import single_point, compute_duration
from primitives.shared.config import GRIPPER_CENTER_TOOL_OFFSET

ACTION_SERVER = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


def resolve_ee_position(target_position, target_rpy, reference_frame):
    """Convert target position to EE frame if needed.

    Args:
        target_position: [x, y, z] target position
        target_rpy: [roll, pitch, yaw] in degrees
        reference_frame: 'ee' or 'gripper_center'

    Returns:
        [x, y, z] position in EE frame
    """
    if reference_frame == 'ee':
        return target_position

    # Transform gripper_center target to EE target
    # The offset is in the tool frame, so we need to rotate it to world frame
    from scipy.spatial.transform import Rotation as R
    rot = R.from_euler('xyz', target_rpy, degrees=True)
    world_offset = rot.apply(GRIPPER_CENTER_TOOL_OFFSET)
    return [target_position[i] - world_offset[i] for i in range(3)]


class PerformIKRunner(Node):
    def __init__(self, target_position, target_rpy, duration, reference_frame='ee'):
        super().__init__('move_xyz_runner')

        # Resolve position to EE frame
        self.target_position = resolve_ee_position(target_position, target_rpy, reference_frame)
        self.target_rpy = target_rpy
        self.duration = duration if duration is not None else 5.0  # fallback if auto-compute fails
        self.client = ActionClient(self, FollowJointTrajectory, ACTION_SERVER)
        self.shutdown_called = False
        self.retry_count = 0
        self.goal_accepted = False
        self.goal_rejected = False
        self.acceptance_timer = None
        self.trajectory_completed = False
        self.success = False
        self.error = None
        self.current_joint_angles = None
        self.joint_angles_received = False
        self._user_duration = duration  # None means auto-compute

        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10
        )

        if self.client.wait_for_server(timeout_sec=10.0):
            self._read_joint_angles()
            self.send_trajectory()
        else:
            self.error = "UR robot driver isn't running"
            self.get_logger().error("Action server not available. Exiting.")
            self.shutdown()

    def _joint_state_callback(self, msg):
        if len(msg.name) >= 6 and len(msg.position) >= 6:
            joint_dict = dict(zip(msg.name, msg.position))
            positions = []
            for name in JOINT_NAMES:
                if name in joint_dict:
                    positions.append(joint_dict[name])
                else:
                    return
            self.current_joint_angles = np.array(positions)
            self.joint_angles_received = True

    def _read_joint_angles(self):
        self.joint_angles_received = False
        timeout = 0
        while rclpy.ok() and not self.joint_angles_received and timeout < 100:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout += 1

    def shutdown(self):
        if not self.shutdown_called:
            self.shutdown_called = True
            rclpy.shutdown()

    def send_trajectory(self):
        target_pose = np.eye(4)
        target_pose[:3, 3] = np.array(self.target_position)
        target_pose[:3, :3] = rpy_to_matrix(self.target_rpy)
        rpy = self.target_rpy
        q6 = -(np.mod(rpy[2] + 180, 360) - 180)
        q_guess = np.radians([85, -80, 90, -90, -90, q6])
        solver = IKSolver(IKSolverConfig())
        joint_angles = solver.solve(seeds=[q_guess], target_pose=target_pose, perturbations=5, dx=0.001)

        if joint_angles is None:
            self.error = "Motion planning failed: target pose is unreachable with current joint constraints"
            self.get_logger().error(self.error)
            self.shutdown()
            return

        # Compute duration dynamically if not explicitly set by caller
        if self.current_joint_angles is not None and self._user_duration is None:
            joint_dist = float(np.max(np.abs(np.array(joint_angles) - self.current_joint_angles)))
            self.duration = compute_duration(joint_distance=joint_dist, profile='s_curve')
            self.get_logger().info(f"Duration: {self.duration:.2f}s (joint={joint_dist:.2f}rad)")

        profile = single_point(joint_angles, self.duration)
        positions, velocities, t = profile[0]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(
            joint_names=JOINT_NAMES,
            points=[JointTrajectoryPoint(
                positions=positions,
                velocities=velocities,
                time_from_start=Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            )]
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
        if self.goal_rejected:
            return

        if not self.goal_accepted:
            self.retry_count += 1
            if self.retry_count <= 5:
                self.get_logger().debug(
                    f"Goal not accepted within 5s (attempt {self.retry_count}/5). Retrying...")
                time.sleep(0.5)
                self.send_trajectory()
            else:
                self.error = "Goal not accepted after max retries"
                self.get_logger().error(self.error)
                self.shutdown()

    def diagnose_rejection(self):
        """Query controller_manager to determine why the goal was rejected."""
        cli = self.create_client(ListControllers, '/controller_manager/list_controllers')
        if not cli.wait_for_service(timeout_sec=2.0):
            return "Trajectory goal rejected (controller_manager unavailable for diagnostics)"

        future = cli.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            return "Trajectory goal rejected (could not query controller state)"

        for c in future.result().controller:
            if c.name == 'scaled_joint_trajectory_controller':
                if c.state == 'inactive':
                    return "scaled_joint_trajectory_controller is not active"
                elif c.state == 'unconfigured':
                    return "scaled_joint_trajectory_controller is unconfigured"
                elif c.state == 'active':
                    return "External control program stopped or robot in protective stop"
                else:
                    return f"scaled_joint_trajectory_controller in unexpected state: {c.state}"

        return "scaled_joint_trajectory_controller not found"

    def goal_response(self, future):
        goal_handle = future.result()
        if self.acceptance_timer:
            self.acceptance_timer.cancel()
            self.acceptance_timer = None

        if not goal_handle.accepted:
            self.goal_rejected = True
            self.error = self.diagnose_rejection()
            self.get_logger().error(f"Trajectory goal rejected: {self.error}")
            self.shutdown()
            return

        self.goal_accepted = True
        self.retry_count = 0
        self.get_logger().info("Trajectory sent and accepted")
        goal_handle.get_result_async().add_done_callback(self.goal_result)

    def goal_result(self, future):
        result = future.result()
        result_msg = result.result

        if result_msg.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.success = True
            self.get_logger().info("Movement completed successfully")
        else:
            self.success = False
            error_messages = {
                FollowJointTrajectory.Result.INVALID_GOAL:
                    "Trajectory rejected: invalid goal (may indicate velocity/acceleration limits exceeded or joint limits violated)",
                FollowJointTrajectory.Result.INVALID_JOINTS:
                    "Invalid joints: joint names don't match",
                FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP:
                    "Old header timestamp: trajectory is too old",
                FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
                    "Velocity or acceleration limits exceeded. Enable robot in URcap to fix this.",
                FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED:
                    "Goal tolerance violated: did not reach target position",
            }

            error_msg = error_messages.get(result_msg.error_code, None)

            if error_msg is None:
                if result.status == 6:  # ABORTED
                    error_msg = ("Trajectory ABORTED: likely velocity/acceleration limits exceeded. "
                                 "Click 'Continue' in URSim/URcap to clear the error, then retry.")
                else:
                    error_msg = f"Trajectory execution failed with error code: {result_msg.error_code}"

            self.error = error_msg
            self.get_logger().error(
                f"{self.error} (error_code: {result_msg.error_code}, status: {result.status})")

        self.trajectory_completed = True
        self.shutdown()


def run_ik(target_position, target_rpy, duration=None, reference_frame='ee'):
    """Run IK and execute trajectory. Returns (success, error)."""
    rclpy.init(args=None)
    node = PerformIKRunner(target_position, target_rpy, duration, reference_frame)
    success = False
    error = None

    try:
        if not node.shutdown_called:
            timeout = (duration if duration is not None else node.duration) + 10
            start_time = time.time()
            while (rclpy.ok() and not node.trajectory_completed
                   and (time.time() - start_time) < timeout):
                rclpy.spin_once(node, timeout_sec=0.1)

            if not node.trajectory_completed:
                error = f"Trajectory execution timed out after {timeout} seconds"
            else:
                success = node.success
                error = node.error
        else:
            success = node.success
            error = node.error
    except KeyboardInterrupt:
        error = "Interrupted by user"
    finally:
        try:
            node.client.destroy()
            node.destroy_node()
        except Exception:
            pass
        if not node.shutdown_called:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    return success, error


def main(args=None):
    parser = argparse.ArgumentParser(description='Perform IK and execute trajectory')
    parser.add_argument('--target-position', type=float, nargs=3, required=True,
                        metavar=('X', 'Y', 'Z'),
                        help='Target position [x, y, z] in meters')
    parser.add_argument('--target-rpy', type=float, nargs=3, required=True,
                        metavar=('ROLL', 'PITCH', 'YAW'),
                        help='Target orientation [roll, pitch, yaw] in degrees')
    parser.add_argument('--duration', type=float, default=None,
                        help='Time to complete the movement in seconds (default: auto-computed)')
    parser.add_argument('--reference-frame', choices=['ee', 'gripper_center'], default='ee',
                        help='Reference frame for target position (default: ee)')

    parsed = parser.parse_args()

    success, error = run_ik(
        parsed.target_position, parsed.target_rpy,
        parsed.duration, parsed.reference_frame
    )

    result = {"result": "success" if success else "failure"}
    if not success and error:
        result["error"] = error

    output_result(result)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
