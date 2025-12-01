#!/usr/bin/env python3
"""
Wheel Control Script
Controls the JETANK wheels in both simulation and real hardware modes.

Usage:
    python3 primitives/control_wheels.py --mode real --linear 0.5 --duration 2.0
    python3 primitives/control_wheels.py --mode sim --linear -0.3 --duration 1.5
    python3 primitives/control_wheels.py --mode real --angular 0.4 --duration 1.0
    python3 primitives/control_wheels.py --mode sim --linear 0.5 --angular 0.2 --duration 2.0
    python3 primitives/control_wheels.py --mode real --linear 0.0 --angular 0.0
"""

import argparse
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import time

# Wheel inversion compensation (set to True if wheels are inverted, False otherwise)
WHEELS_INVERTED = True


class WheelController(Node):
    """ROS2 node for controlling the wheels"""
    
    def __init__(self, mode='sim'):
        super().__init__('wheel_controller')
        
        self.mode = mode
        self.use_real_hardware = (mode == 'real')
        
        # Create velocity controller publisher (for simulation)
        self.velocity_pub = self.create_publisher(
            Float64MultiArray, 
            '/forward_velocity_controller/commands', 
            10
        )
        
        # Create cmd_vel publisher (for real hardware motor driver)
        self.cmd_vel_pub = self.create_publisher(
            Twist, 
            'cmd_vel', 
            10
        )
        
        self.get_logger().info(f'Wheel controller initialized in {mode} mode')
    
    def send_command_real(self, linear_x, angular_z):
        """Send command to real hardware using Twist message"""
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        
        if WHEELS_INVERTED:
            msg.linear.x = -msg.linear.x
            msg.angular.z = -msg.angular.z
        
        self.cmd_vel_pub.publish(msg)
        self.get_logger().info(f'Real hardware command: linear.x={msg.linear.x}, angular.z={msg.angular.z}')
    
    def send_command_sim(self, linear_speed, angular_speed):
        """Send command to simulation using Float64MultiArray message"""
        # For forward/backward: [left_front, right_front, left_back, right_back]
        # Pattern: forward = [speed, -speed, speed, -speed]
        # For turning: [speed, speed, speed, speed] (left) or [-speed, -speed, -speed, -speed] (right)
        # Scale: linear/angular speed (0.0-1.0) -> sim speed (0.0-10.0)
        
        # Scale speeds to match GUI pattern (0.5 linear -> 5.0 sim speed)
        sim_linear = linear_speed * 10.0
        sim_angular = angular_speed * 10.0
        
        msg = Float64MultiArray()
        
        if abs(angular_speed) < 0.01:  # Pure linear motion
            # Forward/backward motion: [left, -right, left, -right]
            msg.data = [sim_linear, -sim_linear, sim_linear, -sim_linear]
        elif abs(linear_speed) < 0.01:  # Pure rotation
            # Pure turning: all wheels same direction
            msg.data = [sim_angular, sim_angular, sim_angular, sim_angular]
        else:  # Combined motion (arc)
            # Combine linear and angular: differential drive
            # Left wheels: linear + angular, Right wheels: linear - angular
            # But in sim format: [left_front, right_front, left_back, right_back]
            # Right wheels need to be negated for forward/backward pattern
            left_speed = sim_linear + sim_angular
            right_speed = sim_linear - sim_angular
            msg.data = [left_speed, -right_speed, left_speed, -right_speed]
        
        # Apply wheel inversion compensation if needed
        if WHEELS_INVERTED:
            msg.data = [-x for x in msg.data]
        
        self.velocity_pub.publish(msg)
        self.get_logger().info(f'Simulation command: {msg.data}')
    
    def move_custom(self, linear=0.0, angular=0.0, duration=None):
        """Move with custom linear and angular velocities"""
        if self.use_real_hardware:
            self.send_command_real(linear, angular)
        else:
            self.send_command_sim(linear, angular)
        
        if duration:
            time.sleep(duration)
            self.stop()
    
    def stop(self):
        """Stop all motion"""
        if self.use_real_hardware:
            self.send_command_real(0.0, 0.0)
        else:
            msg = Float64MultiArray()
            msg.data = [0.0, 0.0, 0.0, 0.0]
            self.velocity_pub.publish(msg)
        
        self.get_logger().info('Stop command sent')


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Control JETANK wheels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 primitives/control_wheels.py --mode real --linear 0.5 --duration 2.0
  python3 primitives/control_wheels.py --mode sim --linear -0.3 --duration 1.5
  python3 primitives/control_wheels.py --mode real --angular 0.4 --duration 1.0
  python3 primitives/control_wheels.py --mode sim --linear 0.5 --angular 0.2 --duration 2.0
  python3 primitives/control_wheels.py --mode real --linear 0.0 --angular 0.0
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['real', 'sim'],
        default='sim',
        help='Hardware mode: real for real hardware, sim for simulation (default: sim)'
    )
    
    parser.add_argument(
        '--linear',
        type=float,
        default=0.0,
        help='Linear velocity (positive = forward, negative = backward, default: 0.0)'
    )
    
    parser.add_argument(
        '--angular',
        type=float,
        default=0.0,
        help='Angular velocity (positive = left turn, negative = right turn, default: 0.0)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Duration in seconds. If not specified, command is sent once and robot continues until stopped.'
    )
    
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create wheel controller node
        controller = WheelController(mode=args.mode)
        
        # Give publisher time to establish connection (DDS discovery)
        for _ in range(20):  # Wait up to 2 seconds for discovery
            rclpy.spin_once(controller, timeout_sec=0.1)
        time.sleep(0.5)  # Additional buffer for DDS discovery
        
        # Execute the requested action
        if args.linear == 0.0 and args.angular == 0.0:
            controller.stop()
        else:
            controller.move_custom(linear=args.linear, angular=args.angular, duration=args.duration)
        
        # If duration was specified, we already stopped. Otherwise, keep publishing
        if args.duration is None and not (args.linear == 0.0 and args.angular == 0.0):
            # Keep the command active by publishing periodically
            # This allows the agent to control duration externally
            print(f"Command sent. Robot will continue until stopped. Use 'stop' command to halt.")
            # Spin a few times to ensure message delivery
            for _ in range(10):
                rclpy.spin_once(controller, timeout_sec=0.1)
        else:
            # Spin a few times to process callbacks and ensure message delivery
            for _ in range(10):
                rclpy.spin_once(controller, timeout_sec=0.1)
        
        # Clean shutdown
        controller.destroy_node()
        rclpy.shutdown()
        
        return 0
        
    except KeyboardInterrupt:
        print('\nInterrupted by user')
        # Make sure to stop the robot
        try:
            controller.stop()
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

