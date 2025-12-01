#!/usr/bin/env python3
"""
Reset Joints Primitive
Resets the JETANK arm joints to home position (0, 0, 0).

Usage:
    python3 primitives/reset_joints.py --mode real
    python3 primitives/reset_joints.py --mode sim --duration 2.0
"""

import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time

# Trajectory duration in seconds
TRAJECTORY_DURATION = 1.0


class ResetJointsController(Node):
    """ROS2 node for resetting arm joints to home position"""
    
    def __init__(self, mode='sim'):
        super().__init__('reset_joints_controller')
        
        self.mode = mode
        self.use_real_hardware = (mode == 'real')
        
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
        
        self.get_logger().info(f'Reset joints controller initialized in {mode} mode')
    
    def joint_state_callback(self, msg):
        """Callback for joint states"""
        self.current_joint_state = msg
    
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
        iteration_count = 0
        
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
            
            # Send to real hardware at lower rate (every 5th iteration = 10Hz, matching GUI)
            if iteration_count % 5 == 0:
                self.send_arm_command_real(current_positions[0], current_positions[1], current_positions[2])
            
            iteration_count += 1
            time.sleep(0.02)  # 50Hz update rate
        
        # Ensure final position is set
        self.send_arm_command_real(target_joints[0], target_joints[1], target_joints[2])
    
    def reset_joints(self, duration=TRAJECTORY_DURATION):
        """Reset arm joints to home position (0, 0, 0)"""
        # Get current joint positions
        current_joints = self.get_current_arm_joints()
        if current_joints is None:
            # Use default if current position unknown
            current_joints = [0.0, 0.785, -1.57]
            self.get_logger().warn("Current joint positions unknown, using default")
        
        # Target joint positions (home position)
        target_joints = [0.0, 0.0, 0.0]
        
        self.get_logger().info(f"Resetting arm joints from [{current_joints[0]:.3f}, {current_joints[1]:.3f}, {current_joints[2]:.3f}] to [0.000, 0.000, 0.000]")
        
        # Send trajectory
        if self.use_real_hardware:
            self.send_trajectory_real(current_joints, target_joints, duration)
        else:
            self.send_trajectory_sim(current_joints, target_joints, duration)
        
        # Wait for trajectory to complete
        time.sleep(duration + 0.2)
        
        self.get_logger().info("Arm joints reset to home position (camera unchanged)")
        return True, "Successfully reset arm joints to home position"


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Reset JETANK arm joints to home position',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 primitives/reset_joints.py --mode real
  python3 primitives/reset_joints.py --mode sim --duration 2.0
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['real', 'sim'],
        default='sim',
        help='Hardware mode: real for real hardware, sim for simulation (default: sim)'
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
        controller = ResetJointsController(mode=args.mode)
        
        # Give subscribers time to establish connection (DDS discovery)
        for _ in range(20):  # Wait up to 2 seconds for discovery
            rclpy.spin_once(controller, timeout_sec=0.1)
        time.sleep(0.5)  # Additional buffer for DDS discovery
        
        # Reset joints
        success, message = controller.reset_joints(duration=args.duration)
        
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

