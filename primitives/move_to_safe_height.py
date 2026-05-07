import sys
import os
import time

# Add project root to path so primitives package can be imported when running directly.
# File is at <repo>/primitives/move_to_safe_height.py, so dirname()×2 = <repo>.
# Three dirnames was a typo that bumped us to the parent of the repo
# (/home/aaugus11/Documents) where there's no `primitives/` package, so
# script-path invocation broke with ModuleNotFoundError.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import numpy as np
import re

from primitives.shared.config import SAFE_HEIGHT, GRIPPER_CENTER_TOOL_OFFSET
import json
import yaml

from primitives.shared.ik import compute_cartesian_waypoints_ik, forward_kinematics, dh_params
from primitives.shared.velocity_profiles import s_curve_profile, compute_duration

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

        # Robot safety monitoring (detect protective stop during execution)
        self.safety_mode = None
        self.robot_program_running = None
        self.trajectory_sent = False

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
        
        # Monitor robot safety mode and program running state to detect protective stop
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        try:
            from ur_dashboard_msgs.msg import SafetyMode
            self.safety_mode_sub = self.create_subscription(
                SafetyMode,
                '/io_and_status_controller/safety_mode',
                self._safety_mode_cb,
                latched_qos
            )
        except ImportError:
            self.safety_mode_sub = None
        self.program_running_sub = self.create_subscription(
            Bool,
            '/io_and_status_controller/robot_program_running',
            self._program_running_cb,
            latched_qos
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
        
        try:
            if self.current_joint_angles is None:
                self.error_message = "Current joint angles not available for motion planning"
                self.get_logger().error(self.error_message)
                return

            num_waypoints = 60

            # SAFE_HEIGHT is calibrated for face-down EE, where the gripper
            # offset projects fully into -z. Derive the intended gripper center
            # safe height, then compute the flange z for the current orientation.
            T_fk = forward_kinematics(dh_params, self.current_joint_angles)
            R_fk = T_fk[:3, :3]
            tool_offset_world = R_fk @ GRIPPER_CENTER_TOOL_OFFSET
            facedown_offset_z = -GRIPPER_CENTER_TOOL_OFFSET[2]  # -0.2286
            gripper_center_safe_z = self.safe_height + facedown_offset_z
            target_flange_z = gripper_center_safe_z - tool_offset_world[2]
            self.get_logger().info(
                f"Gripper center safe Z: {gripper_center_safe_z:.4f}, "
                f"target flange Z: {target_flange_z:.4f} (offset: {tool_offset_world[2]:.4f})"
            )

            # Use fast Jacobian-based IK for dense waypoints
            self.get_logger().info("Computing dense IK waypoints (Jacobian)...")
            waypoints = compute_cartesian_waypoints_ik(
                self.current_joint_angles, target_flange_z,
                num_waypoints=num_waypoints
            )
            if waypoints is None:
                self.error_message = "Motion planning failed: no collision-free path to the safe height could be computed"
                self.get_logger().error(self.error_message)
                return

            all_joint_angles = [self.current_joint_angles.copy()] + list(waypoints)

            cart_dist = abs(target_flange_z - current_pos[2])
            joint_dist = float(np.max(np.abs(np.array(waypoints[-1]) - np.array(self.current_joint_angles))))
            total_duration = compute_duration(
                joint_distance=joint_dist, cartesian_distance=cart_dist, profile='s_curve'
            )
            self.get_logger().info(f"Duration: {total_duration:.2f}s (cart={cart_dist:.3f}m, joint={joint_dist:.2f}rad)")

            profile = s_curve_profile(all_joint_angles, total_duration)
            traj_points = []
            for positions, velocities, t_i in profile:
                point = JointTrajectoryPoint(
                    positions=positions,
                    velocities=velocities,
                    time_from_start=Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
                )
                traj_points.append(point)

            self.get_logger().info(f"Generated {len(traj_points)} Cartesian waypoints with s-curve velocity profile")

            # Create and send trajectory
            goal = FollowJointTrajectory.Goal()
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            traj.points = traj_points
            
            goal.trajectory = traj
            goal.goal_time_tolerance = Duration(sec=1)
            
            self.get_logger().info("Trajectory sent and accepted")
            self.trajectory_sent = True
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)

        except Exception as e:
            self.error_message = f"Motion planning failed: {e}"
            self.get_logger().error(self.error_message)

    def _safety_mode_cb(self, msg):
        self.safety_mode = msg.mode

    def _program_running_cb(self, msg):
        self.robot_program_running = msg.data

    SAFETY_MODE_NAMES = {
        1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY",
        5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP",
        8: "VIOLATION", 9: "FAULT",
    }

    def check_robot_health(self):
        """Check safety mode and program running state. Returns error string or None."""
        if self.safety_mode is not None and self.safety_mode != 1:  # Not NORMAL
            name = self.SAFETY_MODE_NAMES.get(self.safety_mode, f"UNKNOWN({self.safety_mode})")
            return (f"Robot entered {name} (safety_mode={self.safety_mode}) — "
                    "trajectory likely hit joint limits. "
                    "Fix via ursim_cli: unlock -> play")
        if self.robot_program_running is not None and not self.robot_program_running:
            return ("External control program stopped (robot_program_running=False). "
                    "Fix via ursim_cli: stop -> close_popup -> play")
        return None

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
        try:
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
        except Exception as e:
            self.error_message = f"Goal result callback error: {e}"
            self.get_logger().error(self.error_message)
        rclpy.shutdown()

    def output_result_json(self):
        """Output result in JSON format for MCP server"""
        if self.error_message:
            result = {
                "result": "failure",
                "error": self.error_message
            }
        else:
            result = {
                "result": "success",
            }

        output_result(result)


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


def main(args=None):
    rclpy.init(args=args)

    node = MoveToSafeHeight()

    try:
        if node.error_message:
            # Error occurred during __init__ (IK failure, missing data, etc.)
            # Don't spin — just output the result and clean up.
            pass
        else:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)

                # Check for robot safety issues (protective stop, program stopped)
                health_err = node.trajectory_sent and node.check_robot_health()
                if health_err:
                    node.error_message = health_err
                    node.get_logger().error(node.error_message)
                    break
    finally:
        try:
            node.output_result_json()
            node.action_client.destroy()
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
