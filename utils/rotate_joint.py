#!/usr/bin/env python3
"""
Unified script to rotate any robot joint by a specified angle and direction.

Usage examples:
    # Rotate wrist_3 by 90 degrees clockwise
    python rotate_joint.py --joint wrist_3 --angle 90 --direction cw

    # Rotate wrist_1 by 45 degrees counter-clockwise
    python rotate_joint.py --joint wrist_1 --angle 45 --direction ccw

    # Rotate shoulder_pan by 30 degrees (positive direction)
    python rotate_joint.py --joint shoulder_pan --angle 30

    # Rotate elbow joint by 60 degrees with 5 second duration
    python rotate_joint.py --joint elbow --angle 60 --duration 5

    # Use joint index instead of name (0-5)
    python rotate_joint.py --joint 5 --angle 180 --direction cw
"""

import sys
import os

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import argparse
import math
import json

JOINT_NAMES = [
    "shoulder_pan_joint",   # 0
    "shoulder_lift_joint",  # 1
    "elbow_joint",          # 2
    "wrist_1_joint",        # 3
    "wrist_2_joint",        # 4
    "wrist_3_joint"         # 5
]

# Short aliases for convenience
JOINT_ALIASES = {
    "shoulder_pan": 0, "pan": 0, "sp": 0,
    "shoulder_lift": 1, "lift": 1, "sl": 1,
    "elbow": 2, "el": 2,
    "wrist_1": 3, "wrist1": 3, "w1": 3,
    "wrist_2": 4, "wrist2": 4, "w2": 4,
    "wrist_3": 5, "wrist3": 5, "w3": 5,
}

ACTION_SERVER = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
DEFAULT_DURATION = 3.0


def resolve_joint_index(joint_arg):
    """
    Resolve joint argument to index (0-5).

    Args:
        joint_arg: Can be an integer index (0-5), joint name, or alias

    Returns:
        int: Joint index (0-5)

    Raises:
        ValueError: If joint cannot be resolved
    """
    # Try as integer index
    try:
        idx = int(joint_arg)
        if 0 <= idx <= 5:
            return idx
        raise ValueError(f"Joint index must be 0-5, got {idx}")
    except ValueError:
        pass

    # Try as joint name or alias
    joint_lower = joint_arg.lower().strip()

    # Check aliases
    if joint_lower in JOINT_ALIASES:
        return JOINT_ALIASES[joint_lower]

    # Check full joint names
    for i, name in enumerate(JOINT_NAMES):
        if joint_lower == name.lower() or joint_lower == name.replace("_joint", "").lower():
            return i

    raise ValueError(
        f"Unknown joint '{joint_arg}'. Valid options: "
        f"0-5, {', '.join(JOINT_ALIASES.keys())}, or full joint names"
    )


class RotateJoint(Node):
    def __init__(self, joint_index, angle_degrees, duration=DEFAULT_DURATION):
        super().__init__('rotate_joint')
        self.joint_index = joint_index
        self.joint_name = JOINT_NAMES[joint_index]
        self.angle_degrees = angle_degrees
        self.angle_radians = math.radians(angle_degrees)
        self.duration = duration
        self.current_joint_angles = None
        self.joint_angles_received = False
        self.trajectory_completed = False
        self.trajectory_success = False
        self.error_message = None

        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Action client for trajectory
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            ACTION_SERVER
        )

        # Wait for action server
        self.get_logger().info("Waiting for action server...")
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.error_message = "Action server not available - is the robot driver running?"
            self.get_logger().error(self.error_message)
            self.trajectory_completed = True
            return

        # Read current joint angles
        self.get_logger().info("Reading current joint angles...")
        self.read_current_joint_angles()

        if self.current_joint_angles is None:
            self.error_message = "Failed to read current joint angles"
            self.get_logger().error(self.error_message)
            self.trajectory_completed = True
            return

        # Execute rotation
        self.execute_rotation()

    def joint_state_callback(self, msg):
        """Callback to receive joint state messages"""
        if len(msg.position) >= 6:
            # Map joint names to positions
            joint_dict = dict(zip(msg.name, msg.position))
            positions = []
            for joint_name in JOINT_NAMES:
                if joint_name in joint_dict:
                    positions.append(joint_dict[joint_name])
                else:
                    return  # Wait for complete joint state

            self.current_joint_angles = np.array(positions)
            self.joint_angles_received = True

    def read_current_joint_angles(self):
        """Read current joint angles from topic"""
        self.joint_angles_received = False
        timeout = 0
        while rclpy.ok() and not self.joint_angles_received and timeout < 100:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout += 1
        return self.current_joint_angles.copy() if self.joint_angles_received else None

    def execute_rotation(self):
        """Execute rotation of specified joint by the given angle"""
        # Get current angles
        start_angles = self.current_joint_angles.copy()
        target_angles = self.current_joint_angles.copy()

        # Add rotation to specified joint
        target_angles[self.joint_index] += self.angle_radians

        # Normalize to [-pi, pi] for display purposes
        normalized_target = np.arctan2(
            np.sin(target_angles[self.joint_index]),
            np.cos(target_angles[self.joint_index])
        )

        direction_str = "+" if self.angle_degrees >= 0 else ""
        self.get_logger().info(
            f"Rotating {self.joint_name} by {direction_str}{self.angle_degrees} degrees "
            f"(from {math.degrees(start_angles[self.joint_index]):.2f} deg "
            f"to {math.degrees(normalized_target):.2f} deg)"
        )

        # Create interpolated trajectory for smoother motion
        num_waypoints = 10
        trajectory_points = []

        for i in range(num_waypoints + 1):
            alpha = i / num_waypoints
            interpolated = start_angles + alpha * (target_angles - start_angles)
            time_from_start = (i / num_waypoints) * self.duration

            trajectory_points.append(JointTrajectoryPoint(
                positions=[float(x) for x in interpolated],
                velocities=[0.0] * 6,
                time_from_start=Duration(
                    sec=int(time_from_start),
                    nanosec=int((time_from_start % 1) * 1e9)
                )
            ))

        # Create trajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = JOINT_NAMES
        traj_msg.points = trajectory_points

        # Create goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj_msg
        goal.goal_time_tolerance = Duration(sec=1)

        # Send goal
        self.trajectory_completed = False
        self.trajectory_success = False

        self.get_logger().info(f"Sending trajectory ({num_waypoints + 1} waypoints, {self.duration}s duration)...")
        self._send_goal_future = self.action_client.send_goal_async(goal)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

        # Wait for completion
        while rclpy.ok() and not self.trajectory_completed:
            rclpy.spin_once(self, timeout_sec=0.1)

    def goal_response_callback(self, future):
        """Callback when goal is accepted/rejected"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.error_message = "Goal rejected by action server"
            self.get_logger().error(self.error_message)
            self.trajectory_completed = True
            self.trajectory_success = False
            return

        self.get_logger().info("Goal accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        """Callback when goal result is received"""
        result = future.result()
        self.trajectory_success = (result.status == 4)  # SUCCEEDED = 4

        if self.trajectory_success:
            self.get_logger().info("Rotation completed successfully")
        else:
            result_msg = result.result
            if result_msg.error_code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
                self.error_message = "PATH_TOLERANCE_VIOLATED: Velocity or acceleration limits exceeded"
            elif result_msg.error_code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED:
                self.error_message = "GOAL_TOLERANCE_VIOLATED: Did not reach target position"
            elif result_msg.error_code == FollowJointTrajectory.Result.INVALID_GOAL:
                self.error_message = "INVALID_GOAL: Trajectory goal is invalid"
            else:
                self.error_message = f"Trajectory failed with status {result.status}, error code {result_msg.error_code}"
            self.get_logger().error(self.error_message)

        self.trajectory_completed = True


def output_result(result):
    """Output result as JSON to stdout"""
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description='Rotate a robot joint by a specified angle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Joint options:
  By index:  0-5 (shoulder_pan=0, shoulder_lift=1, elbow=2, wrist_1=3, wrist_2=4, wrist_3=5)
  By name:   shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
  By alias:  pan, lift, el, w1, w2, w3

Direction:
  cw   = clockwise (positive angle)
  ccw  = counter-clockwise (negative angle)
  If not specified, the sign of --angle determines direction.

Examples:
  python rotate_joint.py --joint wrist_3 --angle 90 --direction cw
  python rotate_joint.py --joint w1 --angle 45 --direction ccw
  python rotate_joint.py --joint 2 --angle -30
  python rotate_joint.py --joint elbow --angle 60 --duration 5
"""
    )

    parser.add_argument(
        '--joint', '-j',
        type=str,
        required=True,
        help='Joint to rotate (index 0-5, name, or alias)'
    )
    parser.add_argument(
        '--angle', '-a',
        type=float,
        required=True,
        help='Angle to rotate in degrees (positive or negative)'
    )
    parser.add_argument(
        '--direction', '-d',
        type=str,
        choices=['cw', 'ccw', 'clockwise', 'counter-clockwise', 'counterclockwise'],
        default=None,
        help='Rotation direction: cw (clockwise) or ccw (counter-clockwise). Overrides angle sign.'
    )
    parser.add_argument(
        '--duration', '-t',
        type=float,
        default=DEFAULT_DURATION,
        help=f'Movement duration in seconds (default: {DEFAULT_DURATION})'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sim', 'real'],
        default='sim',
        help='Robot mode (default: sim)'
    )

    args = parser.parse_args()

    # Resolve joint
    try:
        joint_index = resolve_joint_index(args.joint)
    except ValueError as e:
        output_result({
            "success": False,
            "error": str(e)
        })
        sys.exit(1)

    # Determine angle with direction
    angle = abs(args.angle)
    if args.direction:
        if args.direction in ['ccw', 'counter-clockwise', 'counterclockwise']:
            angle = -angle
        # cw/clockwise keeps positive
    else:
        # Use sign from angle argument
        angle = args.angle

    # Initialize ROS
    rclpy.init()
    node = RotateJoint(joint_index, angle, duration=args.duration)

    # Spin until operation completes
    while rclpy.ok() and not node.trajectory_completed:
        rclpy.spin_once(node, timeout_sec=0.1)

    # Prepare result
    result = {
        "success": node.trajectory_success,
        "joint": JOINT_NAMES[joint_index],
        "joint_index": joint_index,
        "angle_degrees": angle,
        "duration_seconds": args.duration
    }
    if node.error_message:
        result["error"] = node.error_message

    output_result(result)

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()

    sys.exit(0 if node.trajectory_success else 1)


if __name__ == '__main__':
    main()
