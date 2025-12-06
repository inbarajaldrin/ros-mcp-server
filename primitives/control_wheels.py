#!/usr/bin/env python3
"""
Wheel Control Script
Controls the JETANK wheels in both simulation and real hardware modes via websocket.

Usage:
    python3 primitives/control_wheels.py --mode real --linear 0.5 --duration 2.0
    python3 primitives/control_wheels.py --mode sim --linear -0.3 --duration 1.5
    python3 primitives/control_wheels.py --mode real --angular 0.4 --duration 1.0
    python3 primitives/control_wheels.py --mode sim --linear 0.5 --angular 0.2 --duration 2.0
    python3 primitives/control_wheels.py --mode real --linear 0.0 --angular 0.0
    python3 primitives/control_wheels.py --mode real --target-yaw 0.0
    python3 primitives/control_wheels.py --mode real --target-yaw 90.0 --angular-speed 0.4
"""

import argparse
import time
import sys
import os
import json
import math
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_utils import get_default_mode
from utils.websocket_manager import WebSocketManager, parse_json

# Wheel inversion compensation (set to True if wheels are inverted, False otherwise)
WHEELS_INVERTED = True

# Default websocket connection (can be overridden via environment or config)
ROSBRIDGE_IP = os.getenv("ROSBRIDGE_IP", "localhost")
ROSBRIDGE_PORT = int(os.getenv("ROSBRIDGE_PORT", "9090"))

# Control constants
CONTROL_RATE = 30.0  # Control loop frequency (Hz)
MAX_ANGULAR_VELOCITY = 0.4  # Maximum angular velocity (rad/s)


def quaternion_to_yaw(qx, qy, qz, qw):
    """Convert quaternion to yaw angle (rotation around z-axis) in radians"""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return yaw


def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class PIDController:
    """PID Controller for angle control"""
    
    def __init__(self, kp=1.0, ki=0.1, kd=0.5, integral_limit=2.0):
        """
        Initialize PID controller
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            integral_limit: Maximum absolute value for integral term (windup protection)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        
        self.integral = 0.0
        self.previous_error = 0.0
    
    def reset(self):
        """Reset PID controller state"""
        self.integral = 0.0
        self.previous_error = 0.0
    
    def compute(self, error, dt):
        """
        Compute PID control output
        
        Args:
            error: Current error (target - current)
            dt: Time step since last call
        
        Returns:
            Control output
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with windup protection
        self.integral += error * dt
        # Clamp integral to prevent windup
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        i_term = self.ki * self.integral
        
        # Derivative term (using error derivative)
        if dt > 0:
            d_term = self.kd * (error - self.previous_error) / dt
        else:
            d_term = 0.0
        
        # Store current error for next iteration
        self.previous_error = error
        
        # Total output
        output = p_term + i_term + d_term
        
        return output
    
    def reset_integral_on_sign_change(self, error):
        """Reset integral term when error sign changes to prevent overshoot"""
        if (self.previous_error > 0 and error < 0) or (self.previous_error < 0 and error > 0):
            self.integral = 0.0


def publish_once_websocket(
    ws_manager: WebSocketManager,
    topic: str,
    msg_type: str,
    msg: Dict[str, Any]
) -> Dict[str, Any]:
    """Publish a message to a ROS topic via websocket"""
    # Advertise the topic
    advertise_msg = {"op": "advertise", "topic": topic, "type": msg_type}
    send_error = ws_manager.send(advertise_msg)
    if send_error:
        return {"error": f"Failed to advertise topic: {send_error}"}
    
    # Check for advertise response/errors
    response = ws_manager.receive(timeout=1.0)
    if response:
        try:
            msg_data = json.loads(response)
            if msg_data.get("op") == "status" and msg_data.get("level") == "error":
                ws_manager.send({"op": "unadvertise", "topic": topic})
                return {"error": f"Advertise failed: {msg_data.get('msg', 'Unknown error')}"}
        except json.JSONDecodeError:
            pass  # Non-JSON response is usually fine for advertise
    
    # Publish the message
    publish_msg = {"op": "publish", "topic": topic, "msg": msg}
    send_error = ws_manager.send(publish_msg)
    if send_error:
        ws_manager.send({"op": "unadvertise", "topic": topic})
        return {"error": f"Failed to publish message: {send_error}"}
    
    # Check for publish response/errors
    response = ws_manager.receive(timeout=1.0)
    if response:
        try:
            msg_data = json.loads(response)
            if msg_data.get("op") == "status" and msg_data.get("level") == "error":
                ws_manager.send({"op": "unadvertise", "topic": topic})
                return {"error": f"Publish failed: {msg_data.get('msg', 'Unknown error')}"}
        except json.JSONDecodeError:
            pass  # Non-JSON response is usually fine for publish
    
    # Unadvertise the topic
    ws_manager.send({"op": "unadvertise", "topic": topic})
    
    return {"success": True}


class WheelController:
    """Controller for controlling wheels using websocket"""
    
    def __init__(self, mode=None, ws_manager=None):
        if mode is None:
            mode = get_default_mode()
        
        self.mode = mode
        self.use_real_hardware = (mode == 'real')
        self.ws_manager = ws_manager or WebSocketManager(ROSBRIDGE_IP, ROSBRIDGE_PORT, default_timeout=5.0)
        
        print(f'Wheel controller initialized in {mode} mode')
    
    def send_command_real(self, linear_x, angular_z, log=True):
        """Send command to real hardware using Twist message"""
        # Apply wheel inversion if needed
        if WHEELS_INVERTED:
            linear_x = -linear_x
            angular_z = -angular_z
        
        twist_msg = {
            "linear": {"x": float(linear_x), "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": float(angular_z)}
        }
        
        result = publish_once_websocket(
            self.ws_manager,
            "cmd_vel",
            "geometry_msgs/msg/Twist",
            twist_msg
        )
        
        if log:
            if "error" in result:
                print(f'Error sending real hardware command: {result.get("error")}')
            else:
                print(f'Real hardware command: linear.x={linear_x}, angular.z={angular_z}')
        
        return result
    
    def send_command_sim(self, linear_speed, angular_speed, log=True):
        """Send command to simulation using Float64MultiArray message"""
        # For forward/backward: [left_front, right_front, left_back, right_back]
        # Pattern: forward = [speed, -speed, speed, -speed]
        # For turning: [speed, speed, speed, speed] (left) or [-speed, -speed, -speed, -speed] (right)
        # Scale: linear/angular speed (0.0-1.0) -> sim speed (0.0-10.0)
        
        # Scale speeds to match GUI pattern (0.5 linear -> 5.0 sim speed)
        sim_linear = linear_speed * 10.0
        sim_angular = angular_speed * 10.0
        
        msg_data = []
        
        if abs(angular_speed) < 0.01:  # Pure linear motion
            # Forward/backward motion: [left, -right, left, -right]
            msg_data = [sim_linear, -sim_linear, sim_linear, -sim_linear]
        elif abs(linear_speed) < 0.01:  # Pure rotation
            # Pure turning: all wheels same direction
            msg_data = [sim_angular, sim_angular, sim_angular, sim_angular]
        else:  # Combined motion (arc)
            # Combine linear and angular: differential drive
            # Left wheels: linear + angular, Right wheels: linear - angular
            # But in sim format: [left_front, right_front, left_back, right_back]
            # Right wheels need to be negated for forward/backward pattern
            left_speed = sim_linear + sim_angular
            right_speed = sim_linear - sim_angular
            msg_data = [left_speed, -right_speed, left_speed, -right_speed]
        
        # Apply wheel inversion compensation if needed
        if WHEELS_INVERTED:
            msg_data = [-x for x in msg_data]
        
        msg = {"data": msg_data}
        
        result = publish_once_websocket(
            self.ws_manager,
            "/forward_velocity_controller/commands",
            "std_msgs/msg/Float64MultiArray",
            msg
        )
        
        if log:
            if "error" in result:
                print(f'Error sending simulation command: {result.get("error")}')
            else:
                print(f'Simulation command: {msg_data}')
        
        return result
    
    def move_custom(self, linear=0.0, angular=0.0, duration=0.0):
        """Move with custom linear and angular velocities"""
        # Send command once
        if self.use_real_hardware:
            result = self.send_command_real(linear, angular)
        else:
            result = self.send_command_sim(linear, angular)
        
        if "error" in result:
            return result
        
        # If duration > 0, wait for that duration then stop
        # If duration is 0.0, command is sent and robot continues until stopped externally
        if duration and duration > 0.0:
            time.sleep(duration)
            return self.stop()
        
        return {"success": True}
    
    def move_to_angle(self, target_yaw: float = 0.0, angular_speed: float = 0.4, 
                      tolerance: float = 0.05, max_iterations: int = 1000,
                      kp: float = 0.35, ki: float = 0.01, kd: float = 1.8) -> Dict[str, Any]:
        """
        Move wheels to a target yaw angle using IMU feedback with PID control.
        Follows the pattern from move_to_lego.py for subscription and control.
        
        Args:
            target_yaw: Target yaw angle in radians (default: 0.0)
            angular_speed: Maximum angular velocity limit (default: 0.4)
            tolerance: Angular tolerance in radians to consider target reached (default: 0.05 ~ 3 degrees)
            max_iterations: Maximum number of control loop iterations (default: 1000)
            kp: Proportional gain (default: 1.0)
            ki: Integral gain (default: 0.1)
            kd: Derivative gain (default: 0.5)
        
        Returns:
            Dict with success status and final angle
        """
        control_period = 1.0 / CONTROL_RATE
        iteration = 0
        
        # Initialize PID controller
        pid = PIDController(kp=kp, ki=ki, kd=kd)
        pid.reset()
        
        # Smoothing parameters
        smoothing_factor = 0.5  # Exponential smoothing factor (0-1, lower = smoother)
        max_velocity_change = 0.06  # Maximum change in velocity per control period (rad/s) - reduced for smoother motion
        dead_zone = 0.03  # Dead zone in radians (~1.7 degrees) to prevent jitter - increased
        previous_angular_velocity = 0.0
        previous_yaw = None
        
        print(f"Moving to target yaw: {math.degrees(target_yaw):.2f} degrees")
        print(f"PID gains: Kp={kp}, Ki={ki}, Kd={kd}")
        
        # Keep WebSocket connection open for the entire control loop (like move_to_lego.py)
        with self.ws_manager:
            # Subscribe to IMU topic
            subscribe_imu_msg = {
                "op": "subscribe",
                "topic": "/imu/data",
                "type": "sensor_msgs/msg/Imu",
            }
            
            self.ws_manager.send(subscribe_imu_msg)
            time.sleep(0.1)  # Give time for subscription to register
            
            current_yaw = None
            last_imu_update = 0
            last_log_time = time.time()
            
            try:
                while iteration < max_iterations:
                    # Receive messages from subscribed topic (check multiple times per iteration)
                    # Process multiple messages to catch all updates (like move_to_lego.py)
                    messages_processed = 0
                    while messages_processed < 5:  # Process up to 5 messages per iteration
                        response = self.ws_manager.receive(timeout=0.01)  # Shorter timeout for faster checking
                        if response:
                            msg_data = parse_json(response)
                            if msg_data:
                                # Handle IMU messages
                                if msg_data.get("op") == "publish" and msg_data.get("topic") == "/imu/data":
                                    msg = msg_data.get("msg", {})
                                    orientation = msg.get("orientation", {})
                                    qx = orientation.get("x", 0.0)
                                    qy = orientation.get("y", 0.0)
                                    qz = orientation.get("z", 0.0)
                                    qw = orientation.get("w", 1.0)
                                    
                                    current_yaw = quaternion_to_yaw(qx, qy, qz, qw)
                                    last_imu_update = time.time()
                                    messages_processed += 1
                                    if iteration == 0:
                                        print(f"IMU connected. Initial yaw: {math.degrees(current_yaw):.2f}°")
                        else:
                            break  # No more messages available
                    
                    # Check if we have recent data (within 0.5 seconds) - like move_to_lego.py
                    current_time = time.time()
                    if current_yaw is None or (current_time - last_imu_update) > 0.5:
                        if iteration % 50 == 0:
                            print(f"Warning: No recent IMU data")
                        time.sleep(control_period)
                        iteration += 1
                        continue
                    
                    # Calculate angle difference
                    angle_error = normalize_angle(target_yaw - current_yaw)
                    angle_error_deg = math.degrees(angle_error)
                    
                    # Log progress every 0.5 seconds
                    if time.time() - last_log_time >= 0.5:
                        print(f"Current yaw: {math.degrees(current_yaw):.2f}°, "
                              f"Target: {math.degrees(target_yaw):.2f}°, "
                              f"Error: {angle_error_deg:.2f}°")
                        last_log_time = time.time()
                    
                    # Check if we're close enough to target
                    if abs(angle_error) < tolerance:
                        self.stop()
                        print(f"Target reached! Final yaw: {math.degrees(current_yaw):.2f}°")
                        # Unsubscribe before returning
                        self.ws_manager.send({"op": "unsubscribe", "topic": "/imu/data"})
                        return {"success": True, "final_yaw": current_yaw, "target_yaw": target_yaw}
                    
                    # Calculate current angular velocity from yaw change (for predictive stopping)
                    current_angular_velocity_estimate = 0.0
                    if previous_yaw is not None:
                        yaw_change = normalize_angle(current_yaw - previous_yaw)
                        current_angular_velocity_estimate = yaw_change / control_period
                    previous_yaw = current_yaw
                    
                    # Dead zone: stop if error is very small to prevent jitter
                    if abs(angle_error) < dead_zone:
                        angular_velocity = 0.0
                    else:
                        # Calculate PID control output
                        pid.reset_integral_on_sign_change(angle_error)
                        pid_output = pid.compute(angle_error, control_period)
                        
                        # Clamp output to angular_speed limit
                        angular_velocity = max(-angular_speed, min(angular_speed, pid_output))
                        
                        # Predictive stopping: estimate stopping distance based on current velocity
                        # Using simplified physics: distance = v^2 / (2 * decel)
                        # Assume deceleration of ~0.3 rad/s^2
                        estimated_decel = 0.3
                        if abs(current_angular_velocity_estimate) > 0.01:
                            stopping_distance = (current_angular_velocity_estimate ** 2) / (2 * estimated_decel)
                            # If we're within stopping distance, start aggressive reduction
                            if abs(angle_error) < stopping_distance * 1.2:  # Add 20% safety margin
                                reduction_factor = max(0.1, abs(angle_error) / (stopping_distance * 1.2))
                                angular_velocity *= reduction_factor * reduction_factor
                        
                        # Reduce velocity when close to target - start much earlier
                        abs_error = abs(angle_error)
                        if abs_error < 0.2:  # Within ~11.5 degrees - very close
                            # Very aggressive cubic reduction
                            reduction_factor = abs_error / 0.2
                            angular_velocity *= reduction_factor * reduction_factor * reduction_factor
                        elif abs_error < 0.4:  # Within ~23 degrees
                            # Aggressive quadratic reduction
                            reduction_factor = (abs_error - 0.2) / 0.2  # 0 to 1 over this range
                            angular_velocity *= (0.3 + 0.7 * reduction_factor * reduction_factor)
                        elif abs_error < 0.7:  # Within ~40 degrees - start reducing earlier
                            # Moderate reduction
                            reduction_factor = (abs_error - 0.4) / 0.3  # 0 to 1 over this range
                            angular_velocity *= (0.7 + 0.3 * (1.0 - reduction_factor))
                        elif abs_error < 1.0:  # Within ~57 degrees
                            # Light reduction
                            reduction_factor = (abs_error - 0.7) / 0.3  # 0 to 1 over this range
                            angular_velocity *= (0.85 + 0.15 * (1.0 - reduction_factor))
                        
                        # Apply minimum speed to ensure robot can overcome friction
                        # Always apply 0.2 rad/s minimum for IMU control
                        min_speed = 0.2  # Minimum angular velocity (rad/s)
                        if abs(angle_error) > dead_zone and abs(angular_velocity) > 0.0:
                            # Ensure minimum speed in the correct direction
                            if abs(angular_velocity) < min_speed:
                                angular_velocity = min_speed * math.copysign(1.0, angular_velocity)
                    
                    # Rate limiting: limit the change in velocity per control period
                    velocity_change = angular_velocity - previous_angular_velocity
                    if abs(velocity_change) > max_velocity_change:
                        velocity_change = max_velocity_change * math.copysign(1.0, velocity_change)
                        angular_velocity = previous_angular_velocity + velocity_change
                    
                    # Exponential smoothing: blend current command with previous command
                    angular_velocity = (smoothing_factor * angular_velocity + 
                                      (1.0 - smoothing_factor) * previous_angular_velocity)
                    
                    # Store for next iteration
                    previous_angular_velocity = angular_velocity
                    
                    # Send wheel command (like move_to_lego.py - publish doesn't need subscription)
                    if self.use_real_hardware:
                        result = self.send_command_real(0.0, angular_velocity, log=(iteration % 50 == 0))
                    else:
                        result = self.send_command_sim(0.0, angular_velocity, log=(iteration % 50 == 0))
                    
                    if "error" in result:
                        print(f"Error sending wheel command: {result.get('error')}")
                        self.stop()
                        self.ws_manager.send({"op": "unsubscribe", "topic": "/imu/data"})
                        return result
                    
                    time.sleep(control_period)
                    iteration += 1
                
                # Timeout reached
                self.stop()
                if current_yaw is not None:
                    print(f"Timeout reached. Final yaw: {math.degrees(current_yaw):.2f}°")
                else:
                    print("Timeout reached. Could not get final yaw.")
                
                # Unsubscribe before returning
                self.ws_manager.send({"op": "unsubscribe", "topic": "/imu/data"})
                return {
                    "success": False, 
                    "error": "Timeout reached before reaching target angle",
                    "final_yaw": current_yaw,
                    "target_yaw": target_yaw
                }
            
            except KeyboardInterrupt:
                self.stop()
                self.ws_manager.send({"op": "unsubscribe", "topic": "/imu/data"})
                raise
    
    def stop(self):
        """Stop all motion - send stop command multiple times to ensure it's received"""
        # Send stop command multiple times to ensure it's received
        for i in range(5):
            if self.use_real_hardware:
                result = self.send_command_real(0.0, 0.0, log=(i == 0))  # Only log first time
            else:
                msg = {"data": [0.0, 0.0, 0.0, 0.0]}
                result = publish_once_websocket(
                    self.ws_manager,
                    "/forward_velocity_controller/commands",
                    "std_msgs/msg/Float64MultiArray",
                    msg
                )
                if i == 0:
                    print('Stop command sent')
            
            if "error" in result and i == 0:
                return result
            
            time.sleep(0.02)  # Small delay between stop commands
        
        return {"success": True}


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
  python3 primitives/control_wheels.py --mode real --target-yaw 0.0
  python3 primitives/control_wheels.py --mode real --target-yaw 90.0 --angular-speed 0.4
        """
    )
    
    default_mode = get_default_mode()
    parser.add_argument(
        '--mode',
        choices=['real', 'sim'],
        default=default_mode,
        help=f'Hardware mode: real for real hardware, sim for simulation (default: {default_mode} from config)'
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
    
    # FIX: Changed duration default from None to 0.0 to match server.py schema fix.
    #      This ensures consistency: duration=0.0 means keep moving, duration>0 means move then stop.
    #      The server.py had schema validation errors with Optional[float] = None, so we changed to float = 0.0.
    parser.add_argument(
        '--duration',
        type=float,
        default=0.0,
        help='Duration in seconds. If 0.0, command is sent once and robot continues until stopped. If > 0, robot moves for that duration then stops.'
    )
    
    parser.add_argument(
        '--target-yaw',
        type=float,
        default=None,
        help='Target yaw angle in degrees. If specified, robot will rotate to this angle using IMU feedback. (default: None)'
    )
    
    parser.add_argument(
        '--angular-speed',
        type=float,
        default=0.4,
        help='Angular speed for rotation when using --target-yaw (default: 0.4)'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=3.0,
        help='Angular tolerance in degrees when using --target-yaw (default: 3.0)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create wheel controller
        controller = WheelController(mode=args.mode)
        
        # Give websocket time to establish connection
        time.sleep(0.5)
        
        # Execute the requested action
        if args.target_yaw is not None:
            # Move to target angle using IMU
            target_yaw_rad = math.radians(args.target_yaw)
            tolerance_rad = math.radians(args.tolerance)
            result = controller.move_to_angle(
                target_yaw=target_yaw_rad,
                angular_speed=args.angular_speed,
                tolerance=tolerance_rad
            )
        elif args.linear == 0.0 and args.angular == 0.0:
            result = controller.stop()
        else:
            result = controller.move_custom(linear=args.linear, angular=args.angular, duration=args.duration)
        
        if "error" in result:
            print(f"Error: {result.get('error')}")
            return 1
        
        # If duration was 0.0, we already sent the command and it will keep moving
        # If duration > 0, we already stopped in move_custom
        if args.duration == 0.0 and not (args.linear == 0.0 and args.angular == 0.0):
            # Keep the command active by publishing periodically
            # This allows the agent to control duration externally
            print(f"Command sent. Robot will continue until stopped. Use 'stop' command to halt.")
            # Give some time for message delivery
            time.sleep(0.5)
        
        return 0
        
    except KeyboardInterrupt:
        print('\nInterrupted by user')
        # Make sure to stop the robot
        try:
            controller.stop()
        except:
            pass
        return 1
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
