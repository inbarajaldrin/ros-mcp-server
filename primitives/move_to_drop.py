#!/usr/bin/env python3
"""
Move to Drop Primitive
Moves the JETANK arm to drop position using the drop_poses topic.

Usage:
    python3 primitives/move_to_drop.py --drop_id 1 --mode real
    python3 primitives/move_to_drop.py --drop_id 2 --mode sim
"""

import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_msgs.msg import TFMessage
import time
import threading
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.perform_ik import compute_ik, forward_kinematics, verify_solution

# Trajectory duration in seconds
TRAJECTORY_DURATION = 1.0

class MoveToDropController(Node):
    """ROS2 node for moving to drop positions"""
    
    def __init__(self, mode=None):
        if mode is None:
            mode = get_default_mode()
        super().__init__('move_to_drop_controller')
        
        self.mode = mode
        self.use_real_hardware = (mode == 'real')
        
        # Drop poses data storage
        self.drop_poses_data = {}
        self.drop_poses_lock = threading.Lock()
        
        # Create subscriber for drop poses
        self.drop_poses_sub = self.create_subscription(
            TFMessage,
            '/drop_poses',
            self.drop_poses_callback,
            10
        )
        
        # Create publisher for joint commands (used for both real and sim)
        self.joint_command_pub = self.create_publisher(
            JointState, 
            'joint_commands', 
            10
        )
        
        # Create publisher for trajectory (simulation mode)
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            'arm_trajectory',
            10
        )
        
        # Arm joint names (matching GUI order)
        self.arm_joint_names = [
            'revolute_BEARING',           # 0
            'Revolute_SERVO_LOWER',        # 1
            'Revolute_SERVO_UPPER'         # 2
        ]
        
        # Current joint state
        self.current_joint_state = None
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )
        
        self.get_logger().info(f'Move to drop controller initialized in {mode} mode')
    
    def drop_poses_callback(self, msg):
        """Callback for drop_poses topic (TFMessage format)"""
        with self.drop_poses_lock:
            # Store the latest drop poses data
            self.drop_poses_data = {}
            for transform in msg.transforms:
                # Use child_frame_id as drop name (should be in format drop_1, drop_2, etc.)
                drop_name = transform.child_frame_id
                
                # Extract position from transform (in meters)
                pos = [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ]
                
                # Apply z-offset for drop_poses (+0.05m = +50mm, matching GUI behavior)
                z_position = pos[2] + 0.05
                
                # Convert to mm and store
                self.drop_poses_data[drop_name] = {
                    'position': [pos[0] * 1000, pos[1] * 1000, z_position * 1000],  # Convert to mm with z-offset
                    'header': transform.header
                }
    
    def joint_state_callback(self, msg):
        """Callback for joint states"""
        self.current_joint_state = msg
    
    def get_drop_position(self, drop_name):
        """Get position of a specific drop by name (returns in mm)"""
        with self.drop_poses_lock:
            if drop_name in self.drop_poses_data:
                return self.drop_poses_data[drop_name]['position']
            return None
    
    def list_available_drops(self):
        """Get list of available drop names"""
        with self.drop_poses_lock:
            return list(self.drop_poses_data.keys())
    
    def get_current_arm_joints(self):
        """Get current arm joint positions"""
        if self.current_joint_state is None:
            return None
        
        try:
            # Find indices of arm joints
            bearing_idx = self.current_joint_state.name.index('revolute_BEARING')
            servo_lower_idx = self.current_joint_state.name.index('Revolute_SERVO_LOWER')
            servo_upper_idx = self.current_joint_state.name.index('Revolute_SERVO_UPPER')
            
            return [
                self.current_joint_state.position[bearing_idx],
                self.current_joint_state.position[servo_lower_idx],
                self.current_joint_state.position[servo_upper_idx]
            ]
        except (ValueError, IndexError):
            return None
    
    def send_arm_command_real(self, theta0, theta1, theta3):
        """Send arm command to real hardware (sends each joint separately, matching GUI behavior)"""
        # Send each joint separately (same as GUI behavior)
        self.send_real_hardware_command('base_joint', theta0)
        self.send_real_hardware_command('shoulder_joint', theta1)
        self.send_real_hardware_command('elbow_joint', theta3)
    
    def send_real_hardware_command(self, joint_name, position, velocity=None):
        """Send joint command to real hardware (matching GUI behavior)"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [joint_name]
        msg.position = [position]
        if velocity is not None:
            msg.velocity = [velocity]
        
        self.joint_command_pub.publish(msg)
    
    def send_trajectory_sim(self, start_joints, target_joints, duration):
        """Send trajectory to simulation"""
        # Calculate number of steps (50 steps per second for smooth motion)
        steps = max(10, int(duration * 50))
        
        # Create trajectory points
        trajectory_points = []
        
        for i in range(steps + 1):
            # Calculate interpolation factor (0 to 1)
            t = i / steps
            
            # Smooth interpolation using cubic easing
            t_smooth = 3 * t**2 - 2 * t**3  # Smooth start and end
            
            # Interpolate joint positions
            current_positions = []
            for j in range(3):
                pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                current_positions.append(pos)
            
            # Create trajectory point
            point = JointTrajectoryPoint()
            point.positions = current_positions
            point.time_from_start = Duration(sec=int(t * duration), nanosec=int(((t * duration) % 1) * 1e9))
            
            trajectory_points.append(point)
        
        # Create and publish trajectory
        traj = JointTrajectory()
        traj.joint_names = self.arm_joint_names
        traj.points = trajectory_points
        
        # Publish trajectory
        self.trajectory_pub.publish(traj)
    
    def send_trajectory_real(self, start_joints, target_joints, duration):
        """Send trajectory to real hardware (continuous commands at 50Hz)"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            t = min(1.0, elapsed / duration)
            
            # Smooth interpolation using cubic easing
            t_smooth = 3 * t**2 - 2 * t**3
            
            # Interpolate joint positions
            current_positions = []
            for j in range(3):
                pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                current_positions.append(pos)
            
            # Send command
            self.send_arm_command_real(current_positions[0], current_positions[1], current_positions[2])
            time.sleep(0.02)  # 50Hz update rate
        
        # Ensure final position is set
        self.send_arm_command_real(target_joints[0], target_joints[1], target_joints[2])
    
    def move_to_drop(self, drop_id, duration=TRAJECTORY_DURATION):
        """Move arm to drop position"""
        # Construct drop name in format drop_(id)
        drop_name = f"drop_{drop_id}"
        
        # Wait for drop pose data
        position = None
        for _ in range(50):  # Wait up to 5 seconds
            rclpy.spin_once(self, timeout_sec=0.1)
            position = self.get_drop_position(drop_name)
            if position is not None:
                break
            time.sleep(0.1)
        
        if position is None:
            available = self.list_available_drops()
            error_msg = f"Drop position '{drop_name}' not found"
            if available:
                error_msg += f". Available drops: {', '.join(available)}"
            else:
                error_msg += ". No drop poses detected. Make sure the drop pose system is running."
            self.get_logger().error(error_msg)
            return False, error_msg
        
        x, y, z = position
        
        # Note: z already includes +50mm offset applied in callback
        self.get_logger().info(f"Moving to drop position '{drop_name}' at position: X={x:.1f}mm, Y={y:.1f}mm, Z={z:.1f}mm (with +50mm z-offset)")
        
        # Compute inverse kinematics
        joint_angles = compute_ik(x, y, z, max_tries=5, position_tolerance=2.0)
        
        if joint_angles is None:
            error_msg = f"IK failed: No solution found for target position ({x:.1f}, {y:.1f}, {z:.1f})mm"
            self.get_logger().error(error_msg)
            return False, error_msg
        
        theta0, theta1, theta3 = joint_angles
        
        # Get current joint positions
        current_joints = self.get_current_arm_joints()
        if current_joints is None:
            # Use default if current position unknown
            current_joints = [0.0, 0.785, -1.57]
            self.get_logger().warn("Current joint positions unknown, using default")
        
        target_joints = [theta0, theta1, theta3]
        
        # Send trajectory
        if self.use_real_hardware:
            self.send_trajectory_real(current_joints, target_joints, duration)
        else:
            self.send_trajectory_sim(current_joints, target_joints, duration)
        
        # Wait for trajectory to complete
        time.sleep(duration + 0.2)
        
        # Verify solution
        T, actual_pos = forward_kinematics(theta0, theta1, theta3)
        if actual_pos is not None:
            pos_error = math.sqrt((actual_pos[0] - x)**2 + (actual_pos[1] - y)**2 + (actual_pos[2] - z)**2)
            self.get_logger().info(f"Trajectory completed. Position error: {pos_error:.2f}mm")
            return True, f"Successfully moved to drop position. Position error: {pos_error:.2f}mm"
        else:
            self.get_logger().warn("Could not verify final position")
            return True, "Trajectory completed (unverified)"


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Move JETANK arm to drop position',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 primitives/move_to_drop.py --drop_id 1 --mode real
  python3 primitives/move_to_drop.py --drop_id 2 --mode sim
  python3 primitives/move_to_drop.py --drop_id 3 --mode real --duration 2.0
        """
    )
    
    parser.add_argument(
        '--drop_id',
        type=int,
        required=True,
        help='ID of the drop position (e.g., 1, 2, 3). Will look for drop_1, drop_2, etc. in drop_poses topic.'
    )
    
    default_mode = get_default_mode()
    parser.add_argument(
        '--mode',
        choices=['real', 'sim'],
        default=default_mode,
        help=f'Hardware mode: real for real hardware, sim for simulation (default: {default_mode} from config)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=TRAJECTORY_DURATION,
        help=f'Trajectory duration in seconds (default: {TRAJECTORY_DURATION})'
    )
    
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create controller node
        controller = MoveToDropController(mode=args.mode)
        
        # Give subscribers time to establish connection (DDS discovery)
        for _ in range(20):  # Wait up to 2 seconds for discovery
            rclpy.spin_once(controller, timeout_sec=0.1)
        time.sleep(0.5)  # Additional buffer for DDS discovery
        
        # Move to drop position
        success, message = controller.move_to_drop(args.drop_id, duration=args.duration)
        
        # Spin a few times to process callbacks
        for _ in range(10):
            rclpy.spin_once(controller, timeout_sec=0.1)
        
        # Clean shutdown
        controller.destroy_node()
        rclpy.shutdown()
        
        if success:
            print(f"Success: {message}")
            return 0
        else:
            print(f"Error: {message}")
            return 1
        
    except KeyboardInterrupt:
        print('\nInterrupted by user')
        try:
            controller.destroy_node()
            rclpy.shutdown()
        except:
            pass
        return 1
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        try:
            controller.destroy_node()
            rclpy.shutdown()
        except:
            pass
        return 1


if __name__ == '__main__':
    exit(main())
