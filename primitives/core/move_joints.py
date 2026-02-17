#!/usr/bin/env python3
"""
Unified script for joint-space robot movements.

Supports two modes:
  1. Send absolute joint positions (radians or degrees)
  2. Rotate a single joint by a relative angle

Usage examples:
    # Send absolute joint positions (radians)
    python move_joints.py send --positions 0.0 -1.57 1.57 -1.57 -1.57 0.0

    # Send absolute joint positions (degrees)
    python move_joints.py send --positions-deg 0 -90 90 -90 -90 0

    # Use a preset position
    python move_joints.py send --preset home

    # Rotate wrist_3 by 90 degrees clockwise
    python move_joints.py rotate --joint wrist_3 --angle 90 --direction cw

    # Rotate elbow by 60 degrees with 5 second duration
    python move_joints.py rotate --joint elbow --angle 60 --duration 5

    # Use joint index (0-5)
    python move_joints.py rotate --joint 5 --angle 180 --direction cw
"""

import sys
import os

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from primitives.shared.velocity_profiles import single_point, compute_duration

JOINT_NAMES = [
    "shoulder_pan_joint",   # 0
    "shoulder_lift_joint",  # 1
    "elbow_joint",          # 2
    "wrist_1_joint",        # 3
    "wrist_2_joint",        # 4
    "wrist_3_joint"         # 5
]

JOINT_ALIASES = {
    "shoulder_pan": 0, "pan": 0, "sp": 0,
    "shoulder_lift": 1, "lift": 1, "sl": 1,
    "elbow": 2, "el": 2,
    "wrist_1": 3, "wrist1": 3, "w1": 3,
    "wrist_2": 4, "wrist2": 4, "w2": 4,
    "wrist_3": 5, "wrist3": 5, "w3": 5,
}

PRESETS = {
    "home": [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
    "zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "up": [0.0, -3.14159, 0.0, -1.5708, 0.0, 0.0],
    "forward": [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0],
}

ACTION_SERVER = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
DEFAULT_DURATION = 5.0


def resolve_joint_index(joint_arg):
    """Resolve joint argument to index (0-5). Accepts index, name, or alias."""
    try:
        idx = int(joint_arg)
        if 0 <= idx <= 5:
            return idx
        raise ValueError(f"Joint index must be 0-5, got {idx}")
    except ValueError:
        pass

    joint_lower = joint_arg.lower().strip()
    if joint_lower in JOINT_ALIASES:
        return JOINT_ALIASES[joint_lower]
    for i, name in enumerate(JOINT_NAMES):
        if joint_lower == name.lower() or joint_lower == name.replace("_joint", "").lower():
            return i

    raise ValueError(
        f"Unknown joint '{joint_arg}'. Valid: 0-5, {', '.join(JOINT_ALIASES.keys())}"
    )


def output_result(result):
    print(json.dumps(result, indent=2))


class MoveJoints(Node):
    """ROS node that moves robot joints to target positions with interpolated trajectory."""

    def __init__(self, target_positions, duration=DEFAULT_DURATION):
        super().__init__('move_joints')
        self.target_positions = np.array(target_positions)
        self.duration = duration
        self.shutdown_called = False
        self.success = False
        self.error = None
        self.trajectory_completed = False
        self.current_joint_angles = None
        self.joint_angles_received = False

        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.client = ActionClient(self, FollowJointTrajectory, ACTION_SERVER)

        self.get_logger().info("Waiting for action server...")
        if not self.client.wait_for_server(timeout_sec=10.0):
            self.error = "UR robot driver isn't running - action server not available"
            self.get_logger().error(self.error)
            self.trajectory_completed = True
            return

        self.get_logger().info("Reading current joint states...")
        self._read_joint_angles()

        if self.current_joint_angles is None:
            self.error = "Failed to read current joint angles"
            self.get_logger().error(self.error)
            self.trajectory_completed = True
            return

        self._send_trajectory()

    def joint_state_callback(self, msg):
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

    def _send_trajectory(self):
        start_joints = self.current_joint_angles.copy()
        target_joints = self.target_positions.copy()

        # Shortest path wrapping
        for i in range(6):
            diff = target_joints[i] - start_joints[i]
            if diff > np.pi:
                target_joints[i] -= 2 * np.pi
            elif diff < -np.pi:
                target_joints[i] += 2 * np.pi

        self.get_logger().info(
            f"Current (deg): {[f'{math.degrees(x):.2f}' for x in start_joints]}")
        self.get_logger().info(
            f"Target  (deg): {[f'{math.degrees(x):.2f}' for x in target_joints]}")

        # Compute duration dynamically unless user explicitly set it via CLI
        if self.duration == DEFAULT_DURATION:
            joint_dist = float(np.max(np.abs(target_joints - start_joints)))
            self.duration = compute_duration(joint_distance=joint_dist, profile='s_curve')
            self.get_logger().info(f"Duration: {self.duration:.2f}s (joint={joint_dist:.2f}rad)")

        # Single target point — the UR controller handles smooth interpolation
        profile = single_point(target_joints, self.duration)
        trajectory_points = []
        for positions, velocities, t in profile:
            trajectory_points.append(JointTrajectoryPoint(
                positions=positions,
                velocities=velocities,
                time_from_start=Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            ))

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(
            joint_names=JOINT_NAMES,
            points=trajectory_points
        )
        goal.goal_time_tolerance = Duration(sec=1)

        self.get_logger().info(
            f"Sending trajectory ({len(trajectory_points)} waypoints, {self.duration}s)")

        try:
            future = self.client.send_goal_async(goal)
            future.add_done_callback(self._goal_response)
        except Exception as e:
            self.error = f"Failed to send goal: {e}"
            self.get_logger().error(self.error)
            self.trajectory_completed = True

    def _goal_response(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.error = "Goal rejected - external control program stopped or robot in protective stop"
                self.get_logger().error(self.error)
                self.trajectory_completed = True
                self.shutdown()
                return

            self.get_logger().info("Goal accepted, executing trajectory...")
            goal_handle.get_result_async().add_done_callback(self._goal_result)
        except Exception as e:
            self.error = f"Error in goal response: {e}"
            self.get_logger().error(self.error)
            self.trajectory_completed = True

    def _goal_result(self, future):
        try:
            result = future.result()
            result_msg = result.result

            if result.status == 4:  # SUCCEEDED
                self.success = True
                self.get_logger().info("Movement completed successfully")
            else:
                self.success = False
                error_messages = {
                    FollowJointTrajectory.Result.INVALID_GOAL:
                        "Trajectory rejected: invalid goal",
                    FollowJointTrajectory.Result.INVALID_JOINTS:
                        "Invalid joints: joint names don't match",
                    FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP:
                        "Old header timestamp: trajectory too old",
                    FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
                        "Velocity or acceleration limits exceeded. Enable robot in URcap to fix this.",
                    FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED:
                        "Goal tolerance violated: did not reach target position",
                }

                error_msg = error_messages.get(result_msg.error_code)
                if error_msg is None:
                    if result.status == 6:  # ABORTED
                        error_msg = ("Trajectory ABORTED: likely velocity/acceleration limits exceeded. "
                                     "Click 'Continue' in URSim/URcap to clear the error, then retry.")
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


def run_move_joints(target_positions, duration=DEFAULT_DURATION):
    """Run joint movement. Returns (success, error, current_angles)."""
    rclpy.init()
    node = MoveJoints(target_positions, duration)
    success = False
    error = None
    current_angles = None

    try:
        if not node.shutdown_called:
            rclpy.spin(node)
        success = node.success
        error = node.error
        current_angles = node.current_joint_angles
    except KeyboardInterrupt:
        error = "Interrupted by user"
    except Exception as e:
        error = str(e)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if not node.shutdown_called:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    return success, error, current_angles


def run_rotate_joint(joint_index, angle_degrees, duration=DEFAULT_DURATION):
    """Rotate a single joint by a relative angle. Returns (success, error)."""
    rclpy.init()

    # We need to read current angles first, then compute target
    node = Node('rotate_joint_reader')
    joint_sub_data = {'angles': None, 'received': False}

    def cb(msg):
        if len(msg.name) >= 6 and len(msg.position) >= 6:
            joint_dict = dict(zip(msg.name, msg.position))
            positions = []
            for name in JOINT_NAMES:
                if name in joint_dict:
                    positions.append(joint_dict[name])
                else:
                    return
            joint_sub_data['angles'] = np.array(positions)
            joint_sub_data['received'] = True

    node.create_subscription(JointState, '/joint_states', cb, 10)

    timeout = 0
    while rclpy.ok() and not joint_sub_data['received'] and timeout < 100:
        rclpy.spin_once(node, timeout_sec=0.1)
        timeout += 1

    node.destroy_node()

    if not joint_sub_data['received']:
        rclpy.shutdown()
        return False, "Failed to read current joint angles"

    current = joint_sub_data['angles']
    target = current.copy()
    target[joint_index] += math.radians(angle_degrees)

    # Now use MoveJoints (rclpy already initialized)
    move_node = MoveJoints(target.tolist(), duration)
    success = False
    error = None

    try:
        if not move_node.shutdown_called:
            rclpy.spin(move_node)
        success = move_node.success
        error = move_node.error
    except KeyboardInterrupt:
        error = "Interrupted by user"
    except Exception as e:
        error = str(e)
    finally:
        try:
            move_node.destroy_node()
        except Exception:
            pass
        if not move_node.shutdown_called:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    return success, error


def main():
    parser = argparse.ArgumentParser(
        description='Joint-space robot movements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Movement mode')

    # --- send subcommand ---
    send_parser = subparsers.add_parser('send', help='Send absolute joint positions',
        epilog="""
Presets: home, zero, up, forward

Examples:
  python move_joints.py send --positions 0.0 -1.57 1.57 -1.57 -1.57 0.0
  python move_joints.py send --positions-deg 0 -90 90 -90 -90 0
  python move_joints.py send --preset home --duration 10
""", formatter_class=argparse.RawDescriptionHelpFormatter)

    pos_group = send_parser.add_mutually_exclusive_group()
    pos_group.add_argument('--positions', '-p', type=float, nargs=6,
                           metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'),
                           help='Joint positions in radians')
    pos_group.add_argument('--positions-deg', '--deg', type=float, nargs=6,
                           metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'),
                           help='Joint positions in degrees')
    pos_group.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                           help='Use a preset position')
    send_parser.add_argument('--duration', '-d', type=float, default=DEFAULT_DURATION,
                             help=f'Duration in seconds (default: {DEFAULT_DURATION})')

    # --- rotate subcommand ---
    rot_parser = subparsers.add_parser('rotate', help='Rotate a single joint by relative angle',
        epilog="""
Joints: 0-5, or names (shoulder_pan, elbow, wrist_3, etc.), or aliases (sp, el, w3)

Examples:
  python move_joints.py rotate --joint wrist_3 --angle 90 --direction cw
  python move_joints.py rotate --joint w1 --angle 45 --direction ccw
  python move_joints.py rotate --joint 2 --angle -30
""", formatter_class=argparse.RawDescriptionHelpFormatter)

    rot_parser.add_argument('--joint', '-j', type=str, required=True,
                            help='Joint to rotate (index, name, or alias)')
    rot_parser.add_argument('--angle', '-a', type=float, required=True,
                            help='Angle in degrees')
    rot_parser.add_argument('--direction', type=str, default=None,
                            choices=['cw', 'ccw', 'clockwise', 'counter-clockwise', 'counterclockwise'],
                            help='Direction (overrides angle sign)')
    rot_parser.add_argument('--duration', '-d', type=float, default=DEFAULT_DURATION,
                            help=f'Duration in seconds (default: {DEFAULT_DURATION})')

    args = parser.parse_args()

    if args.command == 'send':
        positions = None
        if args.positions:
            positions = args.positions
        elif args.positions_deg:
            positions = [math.radians(x) for x in args.positions_deg]
        elif args.preset:
            positions = PRESETS[args.preset]
        else:
            send_parser.print_help()
            sys.exit(1)

        success, error, current_angles = run_move_joints(positions, args.duration)

        result = {
            "success": success,
            "target_positions_deg": [math.degrees(x) for x in positions],
            "duration_seconds": args.duration,
        }
        if current_angles is not None:
            result["current_positions_deg"] = [float(math.degrees(x)) for x in current_angles]
        if error:
            result["error"] = error

        output_result(result)
        sys.exit(0 if success else 1)

    elif args.command == 'rotate':
        try:
            joint_index = resolve_joint_index(args.joint)
        except ValueError as e:
            output_result({"success": False, "error": str(e)})
            sys.exit(1)

        angle = abs(args.angle)
        if args.direction:
            if args.direction in ['ccw', 'counter-clockwise', 'counterclockwise']:
                angle = -angle
        else:
            angle = args.angle

        success, error = run_rotate_joint(joint_index, angle, args.duration)

        result = {
            "success": success,
            "joint": JOINT_NAMES[joint_index],
            "angle_degrees": angle,
            "duration_seconds": args.duration,
        }
        if error:
            result["error"] = error

        output_result(result)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
