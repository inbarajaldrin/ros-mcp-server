import sys
import os
import time

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import re

from primitives.utils.workspace_config import SAFE_HEIGHT
import json
import yaml

from primitives.utils.ik_solver import compute_ik, compute_ik_robust

class MoveToSafeHeight(Node):
    def __init__(self, height=None):
        super().__init__('move_to_safe_height')
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # Action client for trajectory control
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        # Safe height target (default 0.3, can be overridden)
        self.safe_height = height if height is not None else SAFE_HEIGHT
        
        # EE pose data storage
        self.ee_pose_received = False
        self.ee_position = None
        self.ee_quat = None
        
        # Current joint angles storage
        self.current_joint_angles = None
        self.joint_angles_received = False

        # Error tracking for JSON output
        self.error_message = None
        self.current_position = None
        self.current_orientation = None

        # Subscriber for EE pose data
        # Use VOLATILE durability (default for most publishers) to avoid QoS incompatibility warnings
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,  # Changed from TRANSIENT_LOCAL to match most publishers
            depth=10
        )
        
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            qos_profile
        )
        
        # Subscriber for joint states to get current joint angles (use as IK seed)
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.action_client.wait_for_server()
        
        # Execute movement
        self.move_to_safe_height()
    
    def ee_pose_callback(self, msg: PoseStamped):
        """Callback for end-effector pose data"""
        self.ee_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.ee_quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        self.ee_pose_received = True
    
    def joint_state_callback(self, msg: JointState):
        """Callback for joint state data"""
        # Extract joint angles in the correct order
        if len(msg.name) == 6 and len(msg.position) == 6:
            joint_dict = dict(zip(msg.name, msg.position))
            # Map joint names to positions in correct order
            ordered_positions = []
            for joint_name in self.joint_names:
                if joint_name in joint_dict:
                    ordered_positions.append(joint_dict[joint_name])
            
            if len(ordered_positions) == 6:
                self.current_joint_angles = np.array(ordered_positions)
                self.joint_angles_received = True

    def quaternion_to_rpy(self, x, y, z, w):
        """Convert quaternion to roll, pitch, yaw in degrees - same as other primitives"""
        import math
        
        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
        
        # Pitch
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.degrees(math.copysign(math.pi / 2, sinp))
        else:
            pitch = math.degrees(math.asin(sinp))
        
        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        
        return [roll, pitch, yaw]

    def read_current_ee_pose(self):
        """Read current end-effector pose and joint angles using ROS2 subscriber with retry"""
        max_retries = 5
        timeout_sec = 15.0

        for attempt in range(max_retries):
            # Reset the flags
            self.ee_pose_received = False
            self.joint_angles_received = False

            if attempt > 0:
                self.get_logger().info(f"Retrying EE pose read (attempt {attempt + 1}/{max_retries})...")
                # Brief delay before retry
                time.sleep(0.5)

            # Wait for both pose and joint angles to arrive (with timeout)
            timeout_count = 0
            max_timeout = int(timeout_sec / 0.1)  # Convert to count

            while rclpy.ok() and (not self.ee_pose_received or not self.joint_angles_received) and timeout_count < max_timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
                timeout_count += 1

                if timeout_count % 10 == 0:  # Log every second
                    status = []
                    if not self.ee_pose_received:
                        status.append("EE pose")
                    if not self.joint_angles_received:
                        status.append("joint angles")
                    self.get_logger().debug(f"Waiting for {' and '.join(status)}... ({timeout_count * 0.1:.1f}s)")

            # Check if we got the data
            if self.ee_pose_received and self.joint_angles_received:
                if self.ee_position is not None and self.ee_quat is not None and self.current_joint_angles is not None:
                    # Success!
                    position = self.ee_position.tolist()
                    orientation = self.ee_quat.tolist()
                    return {
                        'position': position,
                        'orientation': orientation
                    }

            # Log what's missing
            missing = []
            if not self.ee_pose_received or self.ee_position is None or self.ee_quat is None:
                missing.append("EE pose")
            if not self.joint_angles_received or self.current_joint_angles is None:
                missing.append("joint angles")
            self.get_logger().warn(f"Timeout waiting for: {', '.join(missing)} (attempt {attempt + 1}/{max_retries})")

        # All retries exhausted
        self.error_message = f"Failed to read EE pose after {max_retries} attempts"
        self.get_logger().error(self.error_message)
        self.get_logger().error("SUGGESTION: Try running the same command again. The issue is often transient and succeeds on retry.")
        return None

    def move_to_safe_height(self):
        """Move to safe height while maintaining current position and orientation"""
        pose_data = self.read_current_ee_pose()

        if pose_data is None:
            # error_message already set by read_current_ee_pose()
            return
            
        current_pos = pose_data['position']
        current_quat = pose_data['orientation']
        
        # Convert quaternion directly to rotation matrix to avoid precision loss from RPY conversion
        from scipy.spatial.transform import Rotation as Rot
        from primitives.utils.unified_ik import IKSolverConfig, IKSolver

        # Keep the current orientation (don't change it, just move to safe height)
        target_rotation = Rot.from_quat(current_quat)
        target_rot_matrix = target_rotation.as_matrix()

        # Create target position with safe height (same x,y but z=safe_height)
        target_position = current_pos.copy()
        target_position[2] = self.safe_height

        try:
            if self.current_joint_angles is None:
                self.error_message = "Current joint angles not available! Cannot compute IK."
                self.get_logger().error(self.error_message)
                return

            joint_bounds = [
                (-np.pi, np.pi),     # shoulder_pan
                (-np.pi, np.pi),     # shoulder_lift
                (-np.pi, np.pi),     # elbow
                (-np.pi, np.pi),     # wrist_1
                (-np.pi, np.pi),     # wrist_2
                (-2*np.pi, 2*np.pi)  # wrist_3: extended range to avoid wrapping
            ]

            # Cartesian-interpolated waypoints to prevent x,y drift
            dz = abs(target_position[2] - current_pos[2])
            num_waypoints = max(2, int(dz / 0.02))  # one waypoint every 20mm
            total_duration = 5.0

            prev_joints = self.current_joint_angles.copy()
            all_joint_angles = [prev_joints.copy()]

            for i in range(1, num_waypoints + 1):
                alpha = i / num_waypoints
                waypoint_pos = current_pos.copy()
                waypoint_pos[2] = current_pos[2] + alpha * (target_position[2] - current_pos[2])

                waypoint_pose = np.eye(4)
                waypoint_pose[:3, 3] = waypoint_pos
                waypoint_pose[:3, :3] = target_rot_matrix

                solver = IKSolver(IKSolverConfig(joint_bounds=joint_bounds))
                joint_angles = solver.solve(
                    seeds=[prev_joints.copy()],
                    target_pose=waypoint_pose,
                    perturbations=5,
                    dx=0.001,
                )

                if joint_angles is None:
                    self.error_message = f"IK failed at waypoint {i}/{num_waypoints} (z={waypoint_pos[2]*1000:.0f}mm)"
                    self.get_logger().error(self.error_message)
                    return

                all_joint_angles.append(np.array([float(x) for x in joint_angles]))
                prev_joints = np.array(joint_angles)

            # Trapezoidal velocity profile (same as perform_insert)
            n_total = len(all_joint_angles)
            segment_dists = []
            for i in range(1, n_total):
                dist = np.linalg.norm(all_joint_angles[i] - all_joint_angles[i - 1])
                segment_dists.append(max(dist, 1e-6))
            cumulative_s = [0.0]
            for d in segment_dists:
                cumulative_s.append(cumulative_s[-1] + d)
            total_s = cumulative_s[-1]

            accel_frac = 0.2
            decel_frac = 0.2
            t_accel = accel_frac * total_duration
            t_decel = decel_frac * total_duration
            t_cruise = total_duration - t_accel - t_decel
            v_max = total_s / (0.5 * t_accel + t_cruise + 0.5 * t_decel)
            a_accel = v_max / t_accel
            a_decel = v_max / t_decel

            def trapez_s_and_v(t_query):
                if t_query <= t_accel:
                    s = 0.5 * a_accel * t_query ** 2
                    v = a_accel * t_query
                elif t_query <= t_accel + t_cruise:
                    s_accel = 0.5 * v_max * t_accel
                    s = s_accel + v_max * (t_query - t_accel)
                    v = v_max
                else:
                    s_accel = 0.5 * v_max * t_accel
                    s_cruise = v_max * t_cruise
                    t_in_decel = t_query - t_accel - t_cruise
                    s = s_accel + s_cruise + v_max * t_in_decel - 0.5 * a_decel * t_in_decel ** 2
                    v = v_max - a_decel * t_in_decel
                return s, max(v, 0.0)

            def find_time_for_s(target_s):
                lo, hi = 0.0, total_duration
                for _ in range(50):
                    mid = (lo + hi) / 2
                    s_mid, _ = trapez_s_and_v(mid)
                    if s_mid < target_s:
                        lo = mid
                    else:
                        hi = mid
                return (lo + hi) / 2

            waypoint_times = [find_time_for_s(s) for s in cumulative_s]
            waypoint_times[0] = 0.0
            waypoint_times[-1] = total_duration

            traj_points = []
            for i in range(n_total):
                t_i = waypoint_times[i]
                _, speed_scalar = trapez_s_and_v(t_i)

                if i == 0 or i == n_total - 1:
                    velocities = [0.0] * 6
                else:
                    delta = all_joint_angles[i + 1] - all_joint_angles[i - 1]
                    delta_norm = np.linalg.norm(delta)
                    if delta_norm > 1e-8:
                        direction = delta / delta_norm
                        velocities = [float(speed_scalar * direction[j]) for j in range(6)]
                    else:
                        velocities = [0.0] * 6

                point = JointTrajectoryPoint(
                    positions=[float(x) for x in all_joint_angles[i]],
                    velocities=velocities,
                    time_from_start=Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
                )
                traj_points.append(point)

            self.get_logger().info(f"Generated {len(traj_points)} Cartesian waypoints with trapezoidal velocity profile")

            # Create and send trajectory
            goal = FollowJointTrajectory.Goal()
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            traj.points = traj_points
            
            goal.trajectory = traj
            goal.goal_time_tolerance = Duration(sec=1)
            
            self.get_logger().info("Trajectory sent and accepted")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)

        except Exception as e:
            self.error_message = f"Failed to compute IK: {e}"
            self.get_logger().error(self.error_message)

    def goal_response(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.error_message = "External control program stopped or robot in protective stop"
            self.get_logger().error("Trajectory goal rejected")
            rclpy.shutdown()
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result)

    def goal_result(self, future):
        """Handle goal result"""
        result = future.result()
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info("Movement completed successfully")
        else:
            result_msg = result.result
            if result_msg.error_code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
                self.error_message = "Velocity or acceleration limits exceeded. The required velocity to reach the target exceeds joint velocity limits. Enable robot in URcap to fix this."
            else:
                self.error_message = f"Trajectory failed with status code {result.status}"
            self.get_logger().error(self.error_message)
        rclpy.shutdown()

    def output_result_json(self, movement_type="move_to_safe_height"):
        """Output result in JSON format for MCP server"""
        if self.error_message:
            # Failure format
            result = {
                "result": "failure",
                "mode": "sim",
                "movement_type": movement_type,
                "error": self.error_message
            }
        else:
            # Success format - minimal output (no position/orientation)
            result = {
                "result": "success",
                "mode": "sim",
                "movement_type": movement_type
            }

        output_result(result)


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Move robot to safe height')
    parser.add_argument('--height', type=float, default=None,
                       help='Custom height in meters (default: 0.3)')

    known_args, remaining = parser.parse_known_args()

    rclpy.init(args=remaining)

    # Pass custom height to constructor (None uses default 0.3)
    node = MoveToSafeHeight(height=known_args.height)

    if known_args.height is not None:
        node.get_logger().info(f"Using custom height: {known_args.height} meters")

    try:
        if node.error_message:
            # Error occurred during __init__ (IK failure, missing data, etc.)
            # Don't spin — just output the result and clean up.
            pass
        else:
            rclpy.spin(node)
    finally:
        try:
            node.output_result_json(movement_type="move_to_safe_height")
            node.action_client.destroy()
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
