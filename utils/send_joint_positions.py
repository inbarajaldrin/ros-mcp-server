#!/usr/bin/env python3
"""
Script to send joint positions to the robot.

Usage examples:
    # Send specific joint positions (in radians)
    python send_joint_positions.py --positions 0.0 -1.57 1.57 -1.57 -1.57 0.0

    # Send positions with custom duration
    python send_joint_positions.py --positions 0.0 -1.57 1.57 -1.57 -1.57 0.0 --duration 10

    # Send positions in degrees (will be converted to radians)
    python send_joint_positions.py --positions-deg 0 -90 90 -90 -90 0

    # Use preset positions
    python send_joint_positions.py --preset home
    python send_joint_positions.py --preset zero

"""

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
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import argparse
import numpy as np
import json
import math

ACTION_SERVER = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]
DEFAULT_DURATION = 5.0

# Preset positions (in radians)
PRESETS = {
    "home": [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],  # Standard home position
    "zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                  # All joints at zero
    "up": [0.0, -3.14159, 0.0, -1.5708, 0.0, 0.0],           # Arm pointing up
    "forward": [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0],       # Arm extended forward
}


class JointPositionSender(Node):
    def __init__(self, positions, duration=DEFAULT_DURATION):
        super().__init__('joint_position_sender')
        self.positions = positions
        self.duration = duration
        self.shutdown_called = False
        self.success = False
        self.error = None
        self.trajectory_completed = False
        self.goal_accepted = False

        # Subscribe to joint states to read current position
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.current_joint_angles = None
        self.joint_angles_received = False

        self.client = ActionClient(self, FollowJointTrajectory, ACTION_SERVER)
        self.get_logger().info("Waiting for action server...")
        if not self.client.wait_for_server(timeout_sec=10.0):
            self.error = "UR robot driver isn't running - action server not available"
            self.get_logger().error(self.error)
            self.trajectory_completed = True
            return

        self.get_logger().info("Reading current joint states...")
        self.read_current_joint_angles()

        if self.current_joint_angles is None:
            self.error = "Failed to read current joint angles"
            self.get_logger().error(self.error)
            self.trajectory_completed = True
            return

        self.send_trajectory()

    def joint_state_callback(self, msg):
        """Callback to receive current joint states"""
        if len(msg.name) >= 6 and len(msg.position) >= 6:
            # Map joint names to positions (joint_states may be in different order)
            joint_dict = dict(zip(msg.name, msg.position))
            positions = []
            for name in JOINT_NAMES:
                if name in joint_dict:
                    positions.append(joint_dict[name])
                else:
                    return  # Wait for complete state
            self.current_joint_angles = np.array(positions)
            self.joint_angles_received = True

    def read_current_joint_angles(self):
        """Wait for joint state message"""
        self.joint_angles_received = False
        timeout = 0
        while rclpy.ok() and not self.joint_angles_received and timeout < 100:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout += 1
        if self.joint_angles_received:
            self.get_logger().info(f"Current joint angles (rad): {[f'{x:.4f}' for x in self.current_joint_angles]}")
            self.get_logger().info(f"Current joint angles (deg): {[f'{math.degrees(x):.2f}' for x in self.current_joint_angles]}")
        return self.current_joint_angles.copy() if self.joint_angles_received else None

    def shutdown(self):
        if not self.shutdown_called:
            self.shutdown_called = True
            rclpy.shutdown()

    def send_trajectory(self):
        """Send trajectory with interpolated waypoints from current to target positions"""
        # Create interpolated trajectory with multiple waypoints
        num_waypoints = 10
        trajectory_points = []

        # Get current and target joint angles
        start_joints = self.current_joint_angles.copy()
        target_joints = np.array(self.positions)

        # Handle joint wrapping for shortest path
        for i in range(6):
            diff = target_joints[i] - start_joints[i]
            if diff > np.pi:
                target_joints[i] -= 2 * np.pi
            elif diff < -np.pi:
                target_joints[i] += 2 * np.pi

        self.get_logger().info(f"Target joint angles (rad): {[f'{x:.4f}' for x in target_joints]}")
        self.get_logger().info(f"Target joint angles (deg): {[f'{math.degrees(x):.2f}' for x in target_joints]}")

        # Add starting point at t=0
        trajectory_points.append(JointTrajectoryPoint(
            positions=[float(x) for x in start_joints],
            velocities=[0.0] * 6,
            time_from_start=Duration(sec=0, nanosec=0)
        ))

        # Add intermediate waypoints
        for i in range(1, num_waypoints + 1):
            alpha = i / num_waypoints
            interpolated_joints = start_joints + alpha * (target_joints - start_joints)
            time_from_start = (i / num_waypoints) * self.duration
            trajectory_points.append(JointTrajectoryPoint(
                positions=[float(x) for x in interpolated_joints],
                velocities=[0.0] * 6,
                time_from_start=Duration(
                    sec=int(time_from_start),
                    nanosec=int((time_from_start % 1) * 1e9)
                )
            ))

        # Create trajectory message
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(
            joint_names=JOINT_NAMES,
            points=trajectory_points
        )
        goal.goal_time_tolerance = Duration(sec=1)

        self.get_logger().info(f"Sending trajectory with {len(trajectory_points)} waypoints, duration: {self.duration}s")

        # Send goal
        try:
            self._send_goal_future = self.client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response_callback)
        except Exception as e:
            self.error = f"Failed to send goal: {e}"
            self.get_logger().error(self.error)
            self.trajectory_completed = True

    def goal_response_callback(self, future):
        """Callback when goal is accepted/rejected"""
        try:
            goal_handle = future.result()

            if not goal_handle.accepted:
                self.error = "Goal rejected - external control program stopped or robot in protective stop"
                self.get_logger().error(self.error)
                self.trajectory_completed = True
                self.shutdown()
                return

            self.goal_accepted = True
            self.get_logger().info("Goal accepted, executing trajectory...")
            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.goal_result_callback)
        except Exception as e:
            self.error = f"Error in goal response: {e}"
            self.get_logger().error(self.error)
            self.trajectory_completed = True

    def goal_result_callback(self, future):
        """Callback when trajectory execution completes"""
        try:
            result = future.result()
            result_msg = result.result

            if result.status == 4:  # SUCCEEDED
                self.success = True
                self.get_logger().info("Movement completed successfully")
            else:
                self.success = False
                error_messages = {
                    FollowJointTrajectory.Result.INVALID_GOAL: "INVALID_GOAL: Trajectory rejected",
                    FollowJointTrajectory.Result.INVALID_JOINTS: "INVALID_JOINTS: Joint names don't match",
                    FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP: "OLD_HEADER_TIMESTAMP: Trajectory too old",
                    FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED: "PATH_TOLERANCE_VIOLATED: Velocity limits exceeded",
                    FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED: "GOAL_TOLERANCE_VIOLATED: Did not reach target",
                }

                error_msg = error_messages.get(result_msg.error_code)
                if error_msg is None:
                    if result.status == 6:  # ABORTED
                        error_msg = "ABORTED: Trajectory aborted, possibly velocity limits exceeded"
                    else:
                        error_msg = f"Failed with error code {result_msg.error_code}, status {result.status}"

                self.error = error_msg
                self.get_logger().error(self.error)

            self.trajectory_completed = True
            self.shutdown()
        except Exception as e:
            self.error = f"Error in result callback: {e}"
            self.get_logger().error(self.error)
            self.trajectory_completed = True
            self.shutdown()


def output_result(result):
    """Output result as JSON to stdout"""
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description='Send joint positions to the robot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Joint order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3

Presets available:
  home    - Standard home position [0, -90, 90, -90, -90, 0] degrees
  zero    - All joints at zero
  up      - Arm pointing up
  forward - Arm extended forward

Examples:
  python send_joint_positions.py --positions 0.0 -1.57 1.57 -1.57 -1.57 0.0
  python send_joint_positions.py --positions-deg 0 -90 90 -90 -90 0
  python send_joint_positions.py --preset home --duration 10
"""
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--positions', '-p',
        type=float,
        nargs=6,
        metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'),
        help='Joint positions in radians (6 values)'
    )
    group.add_argument(
        '--positions-deg', '--deg',
        type=float,
        nargs=6,
        metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'),
        help='Joint positions in degrees (will be converted to radians)'
    )
    group.add_argument(
        '--preset',
        type=str,
        choices=list(PRESETS.keys()),
        help=f'Use a preset position: {", ".join(PRESETS.keys())}'
    )

    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=DEFAULT_DURATION,
        help=f'Trajectory duration in seconds (default: {DEFAULT_DURATION})'
    )

    args = parser.parse_args()

    # Determine positions
    positions = None

    if args.positions:
        positions = args.positions
    elif args.positions_deg:
        positions = [math.radians(x) for x in args.positions_deg]
    elif args.preset:
        positions = PRESETS[args.preset]
    else:
        parser.print_help()
        print("\nError: Must specify --positions, --positions-deg, or --preset")
        sys.exit(1)

    # Initialize ROS
    rclpy.init()
    node = JointPositionSender(positions=positions, duration=args.duration)

    try:
        if not node.shutdown_called:
            rclpy.spin(node)
    except KeyboardInterrupt:
        node.error = "Interrupted by user"
    except Exception as e:
        node.error = str(e)
    finally:
        success = node.success
        error = node.error
        current_angles = node.current_joint_angles

        # Prepare result
        result = {
            "success": success,
            "current_positions_rad": [float(x) for x in current_angles] if current_angles is not None else None,
            "current_positions_deg": [float(math.degrees(x)) for x in current_angles] if current_angles is not None else None,
            "target_positions_rad": positions,
            "target_positions_deg": [math.degrees(x) for x in positions],
            "duration_seconds": args.duration
        }

        if error:
            result["error"] = error

        output_result(result)

        try:
            node.destroy_node()
        except Exception:
            pass
        if not node.shutdown_called:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
