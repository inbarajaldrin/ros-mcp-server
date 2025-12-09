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
import sys
import os
import argparse

# Add custom libraries to Python path
custom_lib_path = "/home/aaugus11/Desktop/ros2_ws/src/ur_asu-main/ur_asu/custom_libraries"
if custom_lib_path not in sys.path:
    sys.path.append(custom_lib_path)

try:
    from ik_solver import ik_objective_quaternion, forward_kinematics, dh_params
except ImportError as e:
    print(f"Failed to import IK solver: {e}")
    sys.exit(1)

class MoveToClearSpace(Node):
    def __init__(self, mode='move'):
        super().__init__('move_to_clear_space')
        self.mode = mode  # 'move' or 'hover'
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
        
        # Target position for clear space (gripper center position, not TCP)
        self.target_gripper_center_position = [-0.320, -0.5, 0.25]  # [x, y, z] - gripper center position
        # TCP to gripper center offset (24cm along gripper Z-axis, from TCP to gripper center)
        self.tcp_to_gripper_center_offset = 0.24  # 0.24m = 24cm
        
        # EE pose data storage
        self.ee_pose_received = False
        self.ee_position = None
        self.ee_quat = None
        
        # Current joint angles storage
        self.current_joint_angles = None
        self.joint_angles_received = False
        
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
        
        self.get_logger().info("Waiting for action server...")
        self.action_client.wait_for_server()
        
        # Execute movement
        self.move_to_clear_space()
    
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
        """Read current end-effector pose and joint angles using ROS2 subscriber"""
        self.get_logger().info("Reading current end-effector pose and joint angles...")
        
        # Reset the flags
        self.ee_pose_received = False
        self.joint_angles_received = False
        
        # Wait for both pose and joint angles to arrive (with timeout)
        timeout_count = 0
        max_timeout = 100  # 10 seconds (100 * 0.1s)
        
        while rclpy.ok() and (not self.ee_pose_received or not self.joint_angles_received) and timeout_count < max_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout_count += 1
            
            if timeout_count % 10 == 0:  # Log every second
                status = []
                if not self.ee_pose_received:
                    status.append("EE pose")
                if not self.joint_angles_received:
                    status.append("joint angles")
                self.get_logger().info(f"Waiting for {' and '.join(status)}... ({timeout_count * 0.1:.1f}s)")
        
        if not self.ee_pose_received:
            self.get_logger().error("Timeout waiting for EE pose message")
            return None
        
        if not self.joint_angles_received:
            self.get_logger().error("Timeout waiting for joint angles message")
            return None
        
        if self.ee_position is None or self.ee_quat is None:
            self.get_logger().error("EE pose data is None")
            return None
        
        if self.current_joint_angles is None:
            self.get_logger().error("Joint angles data is None")
            return None
        
        # Extract position and orientation
        position = self.ee_position.tolist()
        orientation = self.ee_quat.tolist()
        
        self.get_logger().info(f"Successfully read pose: position={position}, orientation={orientation}")
        self.get_logger().info(f"Successfully read joint angles: {self.current_joint_angles}")
        
        return {
            'position': position,
            'orientation': orientation
        }

    def move_to_clear_space(self):
        """Move to clear space position while maintaining current end-effector orientation"""
        # Read current end-effector pose
        self.get_logger().info("Reading current end-effector pose...")
        pose_data = self.read_current_ee_pose()
        
        if pose_data is None:
            self.get_logger().error("Could not read current end-effector pose")
            rclpy.shutdown()
            return
            
        current_pos = pose_data['position']
        current_quat = pose_data['orientation']
        
        # Convert quaternion directly to rotation matrix to avoid precision loss from RPY conversion
        from scipy.spatial.transform import Rotation as Rot
        from scipy.optimize import minimize
        
        # Also compute RPY for logging purposes
        current_rpy = self.quaternion_to_rpy(
            current_quat[0], current_quat[1], 
            current_quat[2], current_quat[3]
        )
        
        self.get_logger().info(f"Current EE position (TCP): {current_pos}")
        self.get_logger().info(f"Current EE quaternion: {current_quat}")
        self.get_logger().info(f"Current EE RPY (deg): {current_rpy}")
        self.get_logger().info(f"Target gripper center position: {self.target_gripper_center_position}")
        self.get_logger().info(f"Mode: {self.mode}")
        
        # Determine target orientation based on mode
        if self.mode == 'hover':
            # Hover mode: Use same approach as move_home - use compute_ik with RPY [0, 180, 0]
            # Import compute_ik for hover mode
            from ik_solver import compute_ik, rpy_to_matrix
            
            # Calculate TCP position from gripper center position using offset calculation
            # Same approach as move mode - use rotation matrix to get gripper Z-axis
            # For RPY [0, 180, 0], create rotation matrix to calculate offset properly
            target_rpy = [0, 180, 0]
            target_rot_matrix = rpy_to_matrix(target_rpy)
            
            # Calculate TCP position from gripper center position
            # The gripper Z-axis points from TCP to gripper center
            # Apply offset using the same method as move mode
            gripper_z_axis = target_rot_matrix[:, 2]  # Z-axis of gripper frame in world frame
            offset_vector = -self.tcp_to_gripper_center_offset * gripper_z_axis
            tcp_position = np.array(self.target_gripper_center_position) + offset_vector
            tcp_position[2] = self.target_gripper_center_position[2]  # Keep Z constant
            
            self.get_logger().info(f"Hover mode: Using compute_ik (same as move_home)")
            self.get_logger().info(f"Target gripper center position: {self.target_gripper_center_position}")
            self.get_logger().info(f"Calculated TCP position: {tcp_position}")
            self.get_logger().info(f"TCP to gripper center offset: {self.tcp_to_gripper_center_offset*100:.1f}cm along gripper Z-axis")
            self.get_logger().info(f"Target RPY: [0, 180, 0]")
            
            # Use compute_ik exactly like move_home does
            # compute_ik(position, rpy, q_guess=None, max_tries=5, dx=0.001)
            joint_angles = compute_ik(tcp_position, [0, 180, 0])
            
            if joint_angles is None:
                self.get_logger().error("IK failed: couldn't compute clear space position")
                rclpy.shutdown()
                return
                
            self.get_logger().info(f"Computed joint angles: {joint_angles}")
            
            # Create trajectory point - same duration as move_home (5 seconds)
            point = JointTrajectoryPoint(
                positions=[float(x) for x in joint_angles],
                velocities=[0.0] * 6,
                time_from_start=Duration(sec=5)  # 5 seconds movement (same as HOME_MOVEMENT_DURATION)
            )
            
            # Create and send trajectory
            goal = FollowJointTrajectory.Goal()
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            traj.points = [point]
            
            goal.trajectory = traj
            goal.goal_time_tolerance = Duration(sec=1)
            
            self.get_logger().info("Sending trajectory to clear space position (hover mode)...")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)
            return  # Exit early for hover mode
        else:
            # Move mode: Keep the current orientation (don't change it, just move to target position)
            target_rotation = Rot.from_quat(current_quat)
            target_quat = current_quat
            target_rot_matrix = target_rotation.as_matrix()
            
            self.get_logger().info(f"Move mode: Keeping current orientation")
            
            # Calculate TCP position from gripper center position
            # The gripper Z-axis points from TCP to gripper center
            # Apply offset only to X and Y, keep Z constant
            gripper_z_axis = target_rot_matrix[:, 2]  # Z-axis of gripper frame in world frame
            offset_vector = -self.tcp_to_gripper_center_offset * gripper_z_axis
            tcp_position = np.array(self.target_gripper_center_position) + offset_vector
            tcp_position[2] = self.target_gripper_center_position[2]  # Keep Z constant
            
            self.get_logger().info(f"Calculated TCP position: {tcp_position}")
            self.get_logger().info(f"TCP to gripper center offset: {self.tcp_to_gripper_center_offset*100:.1f}cm along gripper Z-axis")
            self.get_logger().info(f"Target orientation: keeping current quaternion (no RPY conversion)")
            
            # Compute inverse kinematics for target pose
            # Use quaternion directly converted to rotation matrix for more accurate orientation preservation
            try:
                # Create target pose with quaternion-derived rotation matrix
                target_pose = np.eye(4)
                target_pose[:3, 3] = tcp_position
                target_pose[:3, :3] = target_rot_matrix
                
                self.get_logger().info(f"Computing IK for TCP position: {tcp_position}")
                self.get_logger().info(f"Using quaternion-derived rotation matrix directly (NO RPY conversion)")
                
                # Use quaternion-based IK directly - no RPY conversion at all!
                joint_angles = None
                best_result = None
                best_cost = float('inf')
                max_tries = 5
                dx = 0.001
                
                # Primary seed: use current joint angles from joint state subscription
                if self.current_joint_angles is None:
                    self.get_logger().error("Current joint angles not available! Cannot compute IK.")
                    rclpy.shutdown()
                    return
                
                q_guess = self.current_joint_angles.copy()
                self.get_logger().info(f"Using current joint angles from joint state as seed: {q_guess}")
                
                # Try IK with current joint angles and position perturbations
                solution_found = False
                for i in range(max_tries):
                    if solution_found:
                        break
                        
                    # Try small x-shift each iteration (helps with workspace boundaries)
                    perturbed_position = np.array(tcp_position).copy()
                    perturbed_position[0] += i * dx
                    
                    perturbed_pose = target_pose.copy()
                    perturbed_pose[:3, 3] = perturbed_position
                    
                    joint_bounds = [(-np.pi, np.pi)] * 6
                    
                    # Use quaternion-based objective directly - NO RPY conversion!
                    result = minimize(ik_objective_quaternion, q_guess, args=(perturbed_pose,), 
                                    method='L-BFGS-B', bounds=joint_bounds)
                    
                    if result.success:
                        cost = ik_objective_quaternion(result.x, perturbed_pose)
                        
                        # Check if this is a good solution
                        if cost < 0.01:
                            self.get_logger().info(f"Quaternion-based IK succeeded with current joint angles seed (perturbation {i}), cost={cost:.6f}")
                            joint_angles = result.x
                            solution_found = True
                            
                            # Verify orientation accuracy
                            T_result = forward_kinematics(dh_params, joint_angles)
                            orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
                            self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
                            break
                        
                        # Keep track of best solution
                        if cost < best_cost:
                            best_cost = cost
                            best_result = result.x
                
                # If we found any reasonable solution, use it
                if joint_angles is None and best_result is not None and best_cost < 0.1:
                    self.get_logger().info(f"Using best quaternion-based IK solution with cost={best_cost:.6f}")
                    joint_angles = best_result
                    
                    # Verify orientation accuracy
                    T_result = forward_kinematics(dh_params, joint_angles)
                    orientation_error = np.linalg.norm(T_result[:3, :3] - target_rot_matrix)
                    self.get_logger().info(f"Orientation error: {orientation_error:.6f}")
                
                if joint_angles is None:
                    self.get_logger().error("IK failed: couldn't compute clear space position")
                    rclpy.shutdown()
                    return
                    
                self.get_logger().info(f"Computed joint angles: {joint_angles}")
                
                # Create trajectory point
                point = JointTrajectoryPoint(
                    positions=[float(x) for x in joint_angles],
                    velocities=[0.0] * 6,
                    time_from_start=Duration(sec=5)  # 5 seconds movement
                )
                
                # Create and send trajectory
                goal = FollowJointTrajectory.Goal()
                traj = JointTrajectory()
                traj.joint_names = self.joint_names
                traj.points = [point]
                
                goal.trajectory = traj
                goal.goal_time_tolerance = Duration(sec=1)
                
                self.get_logger().info("Sending trajectory to clear space position...")
                self._send_goal_future = self.action_client.send_goal_async(goal)
                self._send_goal_future.add_done_callback(self.goal_response)
                
            except Exception as e:
                self.get_logger().error(f"Failed to compute IK: {e}")
                rclpy.shutdown()

    def goal_response(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            rclpy.shutdown()
            return

        self.get_logger().info("Clear space trajectory accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result)

    def goal_result(self, future):
        """Handle goal result"""
        result = future.result()
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info("Successfully moved to clear space position")
        else:
            self.get_logger().error(f"Trajectory failed with status: {result.status}")
        rclpy.shutdown()

def main(args=None):
    parser = argparse.ArgumentParser(description='Move to clear space position')
    parser.add_argument('--move', action='store_true', 
                       help='Move to target position keeping current EE orientation (default)')
    parser.add_argument('--hover', action='store_true',
                       help='Move to target position with top-down (face-down) EE orientation')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.hover:
        mode = 'hover'
    else:
        mode = 'move'  # Default mode
    
    rclpy.init()
    node = MoveToClearSpace(mode=mode)
    rclpy.spin(node)

if __name__ == '__main__':
    main()

