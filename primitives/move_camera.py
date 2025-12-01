#!/usr/bin/env python3
"""
Move Camera Primitive
Moves the JETANK camera to a specified angle.

Usage:
    python3 primitives/move_camera.py --angle -45 --mode real
    python3 primitives/move_camera.py --angle 0 --mode sim
    python3 primitives/move_camera.py down --mode real
    python3 primitives/move_camera.py reset --mode sim
"""

import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time
import math

# Trajectory duration in seconds
CAMERA_TRAJECTORY_DURATION = 1.0


class MoveCameraController(Node):
    """ROS2 node for moving camera"""
    
    def __init__(self, mode='sim'):
        super().__init__('move_camera_controller')
        
        self.mode = mode
        self.use_real_hardware = (mode == 'real')
        
        # Create publisher for joint commands (used for both real and sim)
        self.joint_command_pub = self.create_publisher(
            JointState, 
            'joint_commands', 
            10
        )
        
        # Current joint state
        self.current_joint_state = None
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )
        
        self.get_logger().info(f'Move camera controller initialized in {mode} mode')
    
    def joint_state_callback(self, msg):
        """Callback for joint states"""
        self.current_joint_state = msg
    
    def get_current_camera_angle(self):
        """Get current camera tilt angle"""
        if self.current_joint_state is None:
            return None
        
        try:
            camera_idx = self.current_joint_state.name.index('revolute_CAMERA_HOLDER_ARM_LOWER')
            return self.current_joint_state.position[camera_idx]
        except (ValueError, IndexError):
            return None
    
    def send_real_hardware_command(self, joint_name, position, velocity=None):
        """Send joint command to real hardware (matching GUI behavior)"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [joint_name]
        msg.position = [position]
        if velocity is not None:
            msg.velocity = [velocity]
        
        self.joint_command_pub.publish(msg)
    
    def send_camera_trajectory_real(self, start_angle, target_angle, duration):
        """Send camera trajectory to real hardware (continuous commands at 50Hz)"""
        start_time = time.time()
        iteration_count = 0
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            t = min(1.0, elapsed / duration)
            
            # Smooth interpolation using cubic easing
            t_smooth = 3 * t**2 - 2 * t**3
            
            # Calculate current camera angle
            current_angle = start_angle + (target_angle - start_angle) * t_smooth
            
            # Send to real hardware at lower rate (every 5th iteration = 10Hz, matching GUI)
            if iteration_count % 5 == 0:
                self.send_real_hardware_command('camera_joint', current_angle)
            
            iteration_count += 1
            time.sleep(0.02)  # 50Hz update rate
        
        # Ensure final position is set
        self.send_real_hardware_command('camera_joint', target_angle)
    
    def move_camera(self, target_angle, duration=CAMERA_TRAJECTORY_DURATION):
        """Move camera to target angle (in radians)"""
        # Get current camera angle
        current_angle = self.get_current_camera_angle()
        if current_angle is None:
            current_angle = 0.0
            self.get_logger().warn("Current camera angle unknown, assuming 0.0")
        
        target_degrees = math.degrees(target_angle)
        current_degrees = math.degrees(current_angle)
        
        self.get_logger().info(f"Moving camera from {current_degrees:.1f}° to {target_degrees:.1f}°")
        
        # Send trajectory (always use trajectory for camera, matching GUI)
        if self.use_real_hardware:
            self.send_camera_trajectory_real(current_angle, target_angle, duration)
        else:
            # For simulation, send commands continuously at 50Hz
            start_time = time.time()
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                t = min(1.0, elapsed / duration)
                
                # Smooth interpolation using cubic easing
                t_smooth = 3 * t**2 - 2 * t**3
                
                # Calculate current camera angle
                current_angle_interp = current_angle + (target_angle - current_angle) * t_smooth
                
                # Send command
                self.send_real_hardware_command('camera_joint', current_angle_interp)
                time.sleep(0.02)  # 50Hz update rate
            
            # Ensure final position is set
            self.send_real_hardware_command('camera_joint', target_angle)
        
        # Wait for trajectory to complete
        time.sleep(duration + 0.2)
        
        self.get_logger().info(f"Camera moved to {target_degrees:.1f}°")
        return True, f"Successfully moved camera to {target_degrees:.1f}°"


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Move JETANK camera to specified angle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 primitives/move_camera.py --angle -45 --mode real
  python3 primitives/move_camera.py --angle 0 --mode sim
  python3 primitives/move_camera.py down --mode real
  python3 primitives/move_camera.py reset --mode sim
        """
    )
    
    parser.add_argument(
        'action',
        nargs='?',
        choices=['down', 'reset'],
        help='Quick action: down (move to -45°) or reset (move to 0°)'
    )
    
    parser.add_argument(
        '--angle',
        type=float,
        default=None,
        help='Target camera angle in degrees (e.g., -45, 0, 30). Ignored if action is specified.'
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
        default=CAMERA_TRAJECTORY_DURATION,
        help=f'Trajectory duration in seconds (default: {CAMERA_TRAJECTORY_DURATION})'
    )
    
    args = parser.parse_args()
    
    # Determine target angle
    if args.action == 'down':
        target_angle = math.radians(-45.0)
    elif args.action == 'reset':
        target_angle = 0.0
    elif args.angle is not None:
        target_angle = math.radians(args.angle)
    else:
        parser.error("Either --angle or action (down/reset) must be specified")
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create controller node
        controller = MoveCameraController(mode=args.mode)
        
        # Give subscribers time to establish connection (DDS discovery)
        for _ in range(20):  # Wait up to 2 seconds for discovery
            rclpy.spin_once(controller, timeout_sec=0.1)
        time.sleep(0.5)  # Additional buffer for DDS discovery
        
        # Move camera
        success, message = controller.move_camera(target_angle, duration=args.duration)
        
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

