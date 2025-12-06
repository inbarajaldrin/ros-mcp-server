#!/usr/bin/env python3
"""
Visual Servo Lego Primitive
Moves the robot forward/backward using wheels until the object reaches the target x-position.

Uses objects_poses topic to:
1. Get object x-position
2. Move forward/backward until object x-position reaches target (default: -0.16m)

Usage:
    python3 primitives/move_to_lego.py --object_name lego_1 --mode real
    python3 primitives/move_to_lego.py --object_name lego_2 --mode sim
"""

import argparse
import time
import math
import sys
import os
import json
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_utils import get_default_mode
from utils.websocket_manager import WebSocketManager, parse_json

# Control parameters
LINEAR_GAIN = 10.0   # Gain for linear velocity from object x position
POSITION_TOLERANCE = 0.01  # Tolerance for reaching target position (1cm)
ALIGNMENT_TOLERANCE = 0.05  # Tolerance for alignment in x direction (5cm)
TARGET_X_MIN = -0.15  # Minimum acceptable object x-position (m)
TARGET_X_MAX = -0.14  # Maximum acceptable object x-position (m)
Y_ALIGNMENT_TOLERANCE = 0.01  # Tolerance for y alignment (1cm, 0 to 10mm)
Y_ANGULAR_SPEED = 0.2  # Angular velocity for y alignment (rad/s)
MAX_LINEAR_VELOCITY = 0.3  # Maximum linear velocity (m/s)
MIN_LINEAR_VELOCITY = 0.2  # Minimum linear velocity (m/s)
MAX_ANGULAR_VELOCITY = 0.5  # Maximum angular velocity (rad/s)
CONTROL_RATE = 30.0  # Control loop frequency (Hz)

# Wheel inversion compensation (matching control_wheels.py)
WHEELS_INVERTED = True

# Default websocket connection (can be overridden via environment or config)
ROSBRIDGE_IP = os.getenv("ROSBRIDGE_IP", "localhost")
ROSBRIDGE_PORT = int(os.getenv("ROSBRIDGE_PORT", "9090"))


def subscribe_once_websocket(
    ws_manager: WebSocketManager,
    topic: str,
    msg_type: str,
    timeout: float = 5.0
) -> Dict[str, Any]:
    """Subscribe to a ROS topic via websocket and return the first message"""
    subscribe_msg = {
        "op": "subscribe",
        "topic": topic,
        "type": msg_type,
    }
    
    send_error = ws_manager.send(subscribe_msg)
    if send_error:
        return {"error": f"Failed to subscribe: {send_error}"}
    
    end_time = time.time() + timeout
    while time.time() < end_time:
        response = ws_manager.receive(timeout=0.5)
        if response is None:
            continue
        
        msg_data = parse_json(response)
        if not msg_data:
            continue
        
        if msg_data.get("op") == "status" and msg_data.get("level") == "error":
            unsubscribe_msg = {"op": "unsubscribe", "topic": topic}
            ws_manager.send(unsubscribe_msg)
            return {"error": f"Rosbridge error: {msg_data.get('msg', 'Unknown error')}"}
        
        if msg_data.get("op") == "publish" and msg_data.get("topic") == topic:
            unsubscribe_msg = {"op": "unsubscribe", "topic": topic}
            ws_manager.send(unsubscribe_msg)
            return {"msg": msg_data.get("msg", {})}
    
    unsubscribe_msg = {"op": "unsubscribe", "topic": topic}
    ws_manager.send(unsubscribe_msg)
    return {"error": "Timeout waiting for message from topic"}


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


class VisualServoLegoController:
    """Controller for visual servoing to detected objects using wheel control"""
    
    def __init__(self, mode=None, object_name=None, linear_gain=LINEAR_GAIN, 
                 ws_manager=None):
        if mode is None:
            mode = get_default_mode()
        
        self.mode = mode
        self.use_real_hardware = (mode == 'real')
        self.object_name = object_name
        self.linear_gain = linear_gain
        self.ws_manager = ws_manager or WebSocketManager(ROSBRIDGE_IP, ROSBRIDGE_PORT, default_timeout=5.0)
        
        print(f'Visual servo Lego controller initialized in {mode} mode, targeting object {self.object_name}')
        print(f'Target range: {TARGET_X_MIN:.3f}m to {TARGET_X_MAX:.3f}m')
    
    def get_object_position(self, object_name: str, timeout: float = 5.0):
        """Get position of a specific object by name (returns position in meters)"""
        result = subscribe_once_websocket(
            self.ws_manager,
            "/objects_poses",
            "tf2_msgs/msg/TFMessage",
            timeout=timeout
        )
        
        if "error" in result:
            return None, result.get("error")
        
        transforms = result.get("msg", {}).get("transforms", [])
        
        for transform in transforms:
            if transform.get("child_frame_id") == object_name:
                translation = transform.get("transform", {}).get("translation", {})
                position = [
                    translation.get("x", 0.0),
                    translation.get("y", 0.0),
                    translation.get("z", 0.0)
                ]
                return position, None
        
        return None, f"Object {object_name} not found. Make sure the objects_poses topic is publishing."
    
    def get_end_effector_pose(self, timeout: float = 5.0):
        """Get current end-effector pose (returns position in meters)"""
        result = subscribe_once_websocket(
            self.ws_manager,
            "/ee_pose",
            "geometry_msgs/msg/PoseStamped",
            timeout=timeout
        )
        
        if "error" in result:
            return None, result.get("error")
        
        pose = result.get("msg", {}).get("pose", {})
        position = pose.get("position", {})
        
        ee_position = [
            position.get("x", 0.0),
            position.get("y", 0.0),
            position.get("z", 0.0)
        ]
        
        return ee_position, None
    
    def send_wheel_command_real(self, linear_x, angular_z, log=False):
        """Send wheel command to real hardware using Twist message (works with open connection)"""
        # Apply wheel inversion if needed
        if WHEELS_INVERTED:
            linear_x = -linear_x
            angular_z = -angular_z
        
        twist_msg = {
            "linear": {"x": float(linear_x), "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": float(angular_z)}
        }
        
        # Advertise topic (idempotent, safe to call multiple times)
        advertise_msg = {"op": "advertise", "topic": "cmd_vel", "type": "geometry_msgs/msg/Twist"}
        self.ws_manager.send(advertise_msg)
        
        # Publish message
        publish_msg = {"op": "publish", "topic": "cmd_vel", "msg": twist_msg}
        send_error = self.ws_manager.send(publish_msg)
        
        if send_error:
            if log:
                print(f'Error sending real hardware command: {send_error}')
            return {"error": send_error}
        
        return {"success": True}
    
    def send_wheel_command_sim(self, linear_speed, angular_speed, log=False):
        """Send wheel command to simulation using Float64MultiArray message (works with open connection)"""
        # Scale speeds to match GUI pattern (0.5 linear -> 5.0 sim speed)
        sim_linear = linear_speed * 10.0
        sim_angular = angular_speed * 10.0
        
        msg_data = []
        
        if abs(angular_speed) < 0.01:  # Pure linear motion
            msg_data = [sim_linear, -sim_linear, sim_linear, -sim_linear]
        elif abs(linear_speed) < 0.01:  # Pure rotation
            msg_data = [sim_angular, sim_angular, sim_angular, sim_angular]
        else:  # Combined motion (arc)
            left_speed = sim_linear + sim_angular
            right_speed = sim_linear - sim_angular
            msg_data = [left_speed, -right_speed, left_speed, -right_speed]
        
        # Apply wheel inversion compensation if needed
        if WHEELS_INVERTED:
            msg_data = [-x for x in msg_data]
        
        msg = {"data": msg_data}
        
        # Advertise topic (idempotent, safe to call multiple times)
        advertise_msg = {"op": "advertise", "topic": "/forward_velocity_controller/commands", "type": "std_msgs/msg/Float64MultiArray"}
        self.ws_manager.send(advertise_msg)
        
        # Publish message
        publish_msg = {"op": "publish", "topic": "/forward_velocity_controller/commands", "msg": msg}
        send_error = self.ws_manager.send(publish_msg)
        
        if send_error:
            if log:
                print(f'Error sending simulation command: {send_error}')
            return {"error": send_error}
        
        return {"success": True}
    
    def stop_wheels(self):
        """Stop all wheel motion (works with open connection)"""
        if self.use_real_hardware:
            return self.send_wheel_command_real(0.0, 0.0, log=False)
        else:
            # Use the same method as send_wheel_command_sim for consistency
            return self.send_wheel_command_sim(0.0, 0.0, log=False)
    
    def get_current_bearing_angle(self, timeout=1.0):
        """Get current base bearing joint angle (works with open connection)"""
        # Subscribe to joint_states
        subscribe_msg = {"op": "subscribe", "topic": "/joint_states", "type": "sensor_msgs/msg/JointState"}
        self.ws_manager.send(subscribe_msg)
        time.sleep(0.1)
        
        # Receive joint state message
        end_time = time.time() + timeout
        while time.time() < end_time:
            response = self.ws_manager.receive(timeout=0.1)
            if response:
                msg_data = parse_json(response)
                if msg_data and msg_data.get("op") == "publish" and msg_data.get("topic") == "/joint_states":
                    joint_state = msg_data.get("msg", {})
                    joint_names = joint_state.get("name", [])
                    joint_positions = joint_state.get("position", [])
                    
                    try:
                        bearing_idx = joint_names.index("revolute_BEARING")
                        angle = joint_positions[bearing_idx]
                        self.ws_manager.send({"op": "unsubscribe", "topic": "/joint_states"})
                        return angle
                    except (ValueError, IndexError):
                        pass
        
        self.ws_manager.send({"op": "unsubscribe", "topic": "/joint_states"})
        return None
    
    def send_bearing_command(self, angle):
        """Send base bearing joint command (works with open connection)"""
        # Advertise topic (idempotent, safe to call multiple times)
        advertise_msg = {"op": "advertise", "topic": "joint_commands", "type": "sensor_msgs/msg/JointState"}
        self.ws_manager.send(advertise_msg)
        
        # Send command
        joint_msg = {
            "header": {
                "stamp": {
                    "sec": int(time.time()),
                    "nanosec": int((time.time() % 1) * 1e9)
                }
            },
            "name": ["base_joint"],
            "position": [angle]
        }
        
        publish_msg = {"op": "publish", "topic": "joint_commands", "msg": joint_msg}
        send_error = self.ws_manager.send(publish_msg)
        
        if send_error:
            return {"error": send_error}
        
        return {"success": True}
    
    def send_bearing_trajectory_real(self, start_angle, target_angle, duration=1.0):
        """Send trajectory to real hardware (continuous commands at 50Hz)"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            t = min(1.0, elapsed / duration)
            
            # Smooth interpolation using cubic easing
            t_smooth = 3 * t**2 - 2 * t**3
            
            # Interpolate joint position
            current_angle = start_angle + (target_angle - start_angle) * t_smooth
            
            # Send command
            result = self.send_bearing_command(current_angle)
            if "error" in result:
                return result
            
            time.sleep(0.02)  # 50Hz update rate
        
        # Ensure final position is set
        return self.send_bearing_command(target_angle)
    
    def send_bearing_trajectory_sim(self, start_angle, target_angle, duration=1.0):
        """Send trajectory to simulation (continuous commands at 50Hz, matching move_to_aruco pattern)"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            t = min(1.0, elapsed / duration)
            
            # Smooth interpolation using cubic easing
            t_smooth = 3 * t**2 - 2 * t**3
            
            # Interpolate joint position
            current_angle = start_angle + (target_angle - start_angle) * t_smooth
            
            # Send command
            result = self.send_bearing_command(current_angle)
            if "error" in result:
                return result
            
            time.sleep(0.02)  # 50Hz update rate
        
        # Ensure final position is set
        return self.send_bearing_command(target_angle)
    
    def align_y_with_wheels(self, obj_position, max_iterations=200):
        """Align object y position using angular velocity control (wheel control)"""
        obj_x, obj_y, obj_z = obj_position
        print(f"Aligning y position using wheels. Object y={obj_y*1000:.1f}mm (target: 0mm, tolerance: ±{Y_ALIGNMENT_TOLERANCE*1000:.0f}mm)")
        
        control_period = 1.0 / CONTROL_RATE
        iteration = 0
        
        # Keep WebSocket connection open for the control loop
        with self.ws_manager:
            # Subscribe to object poses topic
            subscribe_obj_msg = {
                "op": "subscribe",
                "topic": "/objects_poses",
                "type": "tf2_msgs/msg/TFMessage",
            }
            
            self.ws_manager.send(subscribe_obj_msg)
            time.sleep(0.1)  # Give time for subscription to register
            
            current_y = obj_y
            last_obj_update = time.time()
            
            try:
                while iteration < max_iterations:
                    # Receive messages from subscribed topic
                    messages_processed = 0
                    while messages_processed < 5:
                        response = self.ws_manager.receive(timeout=0.01)
                        if response:
                            msg_data = parse_json(response)
                            if msg_data:
                                # Handle object position updates
                                if msg_data.get("op") == "publish" and msg_data.get("topic") == "/objects_poses":
                                    transforms = msg_data.get("msg", {}).get("transforms", [])
                                    for transform in transforms:
                                        if transform.get("child_frame_id") == self.object_name:
                                            translation = transform.get("transform", {}).get("translation", {})
                                            current_y = translation.get("y", 0.0)
                                            last_obj_update = time.time()
                                            messages_processed += 1
                                            break
                        else:
                            break
                    
                    # Check if we have recent data
                    current_time = time.time()
                    if (current_time - last_obj_update) > 0.5:
                        if iteration % 50 == 0:
                            print(f"Warning: No recent object position data")
                        time.sleep(control_period)
                        iteration += 1
                        continue
                    
                    # Check if y is within tolerance (0 to 10mm = 0 to 0.01m)
                    if abs(current_y) <= Y_ALIGNMENT_TOLERANCE:
                        self.stop_wheels()
                        print(f"Y alignment completed! Object y={current_y*1000:.1f}mm (within tolerance: ±{Y_ALIGNMENT_TOLERANCE*1000:.0f}mm)")
                        self.ws_manager.send({"op": "unsubscribe", "topic": "/objects_poses"})
                        return True, f"Successfully aligned y position. Final y: {current_y*1000:.1f}mm"
                    
                    # Calculate angular velocity based on y position
                    # If y is positive (object to the right), rotate left (negative angular velocity)
                    # If y is negative (object to the left), rotate right (positive angular velocity)
                    if current_y > 0:
                        angular_vel = -Y_ANGULAR_SPEED  # Rotate left
                    else:
                        angular_vel = Y_ANGULAR_SPEED  # Rotate right
                    
                    # Linear velocity is 0 (only angular movement)
                    linear_vel = 0.0
                    
                    # Send wheel command
                    if self.use_real_hardware:
                        result = self.send_wheel_command_real(linear_vel, angular_vel, log=(iteration % 50 == 0))
                    else:
                        result = self.send_wheel_command_sim(linear_vel, angular_vel, log=(iteration % 50 == 0))
                    
                    if "error" in result:
                        print(f"Error sending wheel command: {result.get('error')}")
                    
                    # Log status periodically
                    if iteration % 50 == 0:
                        print(f"Y alignment iteration {iteration}: y={current_y*1000:.1f}mm, angular_vel={angular_vel:.2f}rad/s")
                    
                    time.sleep(control_period)
                    iteration += 1
                
                # Timeout - stop wheels and unsubscribe
                self.stop_wheels()
                self.ws_manager.send({"op": "unsubscribe", "topic": "/objects_poses"})
                return False, f"Y alignment timeout after {max_iterations} iterations. Final y: {current_y*1000:.1f}mm"
                
            except KeyboardInterrupt:
                self.stop_wheels()
                self.ws_manager.send({"op": "unsubscribe", "topic": "/objects_poses"})
                print("\nInterrupted by user")
                return False, "Interrupted by user"
            except Exception as e:
                self.stop_wheels()
                self.ws_manager.send({"op": "unsubscribe", "topic": "/objects_poses"})
                print(f"Error during y alignment: {e}")
                import traceback
                traceback.print_exc()
                return False, f"Error: {str(e)}"
    
    def align_and_move_to_object(self, max_iterations=500):
        """First align y position, then move forward/backward until object reaches target x-position"""
        print(f"Moving to object {self.object_name}")
        print(f"Step 1: Y alignment (target: 0mm, tolerance: ±{Y_ALIGNMENT_TOLERANCE*1000:.0f}mm)")
        print(f"Step 2: X alignment (target: {TARGET_X_MAX:.3f}m, acceptable range: {TARGET_X_MIN:.3f}m to {TARGET_X_MAX:.3f}m)")
        
        # Step 1: Y alignment first
        # Get initial object position for y alignment
        obj_position, error = self.get_object_position(self.object_name, timeout=5.0)
        if obj_position is None:
            return False, error or "Failed to get initial object position for y alignment"
        
        # Perform y alignment using angular velocity control (wheels)
        print("Starting y alignment...")
        align_success, align_message = self.align_y_with_wheels(obj_position)
        if not align_success:
            print(f"Warning: Y alignment failed: {align_message}")
            # Continue with x alignment anyway
        
        # Step 2: X alignment
        print("Starting x alignment...")
        control_period = 1.0 / CONTROL_RATE
        iteration = 0
        
        # Keep WebSocket connection open for the entire control loop
        with self.ws_manager:
            # Subscribe to object poses topic only
            subscribe_obj_msg = {
                "op": "subscribe",
                "topic": "/objects_poses",
                "type": "tf2_msgs/msg/TFMessage",
            }
            
            self.ws_manager.send(subscribe_obj_msg)
            time.sleep(0.1)  # Give time for subscription to register
            
            obj_position = None  # [x, y, z] in meters
            last_obj_update = 0
            
            try:
                while iteration < max_iterations:
                    # Receive messages from subscribed topic (check multiple times per iteration)
                    # Process multiple messages to catch all updates
                    messages_processed = 0
                    while messages_processed < 5:  # Process up to 5 messages per iteration
                        response = self.ws_manager.receive(timeout=0.01)  # Shorter timeout for faster checking
                        if response:
                            msg_data = parse_json(response)
                            if msg_data:
                                # Handle object position updates
                                if msg_data.get("op") == "publish" and msg_data.get("topic") == "/objects_poses":
                                    transforms = msg_data.get("msg", {}).get("transforms", [])
                                    for transform in transforms:
                                        if transform.get("child_frame_id") == self.object_name:
                                            translation = transform.get("transform", {}).get("translation", {})
                                            obj_position = [
                                                translation.get("x", 0.0),
                                                translation.get("y", 0.0),
                                                translation.get("z", 0.0)
                                            ]
                                            last_obj_update = time.time()
                                            messages_processed += 1
                                            break
                        else:
                            break  # No more messages available
                    
                    # Check if we have recent data (within 0.5 seconds)
                    current_time = time.time()
                    if obj_position is None or (current_time - last_obj_update) > 0.5:
                        if iteration % 50 == 0:
                            print(f"Warning: No recent object position data")
                        time.sleep(control_period)
                        iteration += 1
                        continue
                    
                    obj_x = obj_position[0]
                    
                    # Calculate error: object x position vs target x position (target is upper bound)
                    target_x = TARGET_X_MAX  # Target is -0.14 (upper bound)
                    x_error = obj_x - target_x
                    
                    # Check if we've reached the target (acceptable range)
                    if TARGET_X_MIN <= obj_x <= TARGET_X_MAX:
                        self.stop_wheels()
                        print(f"X alignment completed! Object x-position: {obj_x:.4f}m (acceptable range: {TARGET_X_MIN:.2f} to {TARGET_X_MAX:.2f})")
                        
                        # Unsubscribe before returning
                        self.ws_manager.send({"op": "unsubscribe", "topic": "/objects_poses"})
                        return True, f"Successfully reached object. Object x-position: {obj_x:.4f}m. {align_message}"
                    
                    # Calculate control velocities
                    # Angular: set to 0 - no rotation, only linear movement
                    angular_vel = 0.0
                    
                    # Linear: move forward/backward to reach target x position (-0.15)
                    # Pattern: obj_x < target (e.g., -0.16 < -0.15) → move forward
                    #         obj_x > target (e.g., -0.14 > -0.15) → move backward
                    if obj_x < target_x:  # Object is more negative (farther from base)
                        linear_vel = self.linear_gain * abs(x_error)  # Move forward
                    else:  # Object is less negative (closer to base)
                        linear_vel = -self.linear_gain * abs(x_error)  # Move backward
                    
                    # Clamp to max velocity
                    linear_vel = max(-MAX_LINEAR_VELOCITY, min(MAX_LINEAR_VELOCITY, linear_vel))
                    
                    # Apply minimum speed (preserve direction)
                    if abs(linear_vel) > 0 and abs(linear_vel) < MIN_LINEAR_VELOCITY:
                        linear_vel = MIN_LINEAR_VELOCITY if linear_vel > 0 else -MIN_LINEAR_VELOCITY
                    
                    # If we're close to target, reduce linear velocity (but not below minimum)
                    if abs(x_error) < 0.1:
                        reduced_vel = linear_vel * 0.5
                        # Only apply reduction if it doesn't go below minimum
                        if abs(reduced_vel) >= MIN_LINEAR_VELOCITY:
                            linear_vel = reduced_vel
                    
                    # Send wheel command (publish doesn't need subscription, just advertise)
                    if self.use_real_hardware:
                        result = self.send_wheel_command_real(linear_vel, angular_vel, log=(iteration % 50 == 0))
                    else:
                        result = self.send_wheel_command_sim(linear_vel, angular_vel, log=(iteration % 50 == 0))
                    
                    if "error" in result:
                        print(f"Error sending wheel command: {result.get('error')}")
                    
                    # Log status periodically
                    if iteration % 50 == 0:
                        print(f"Iteration {iteration}: Object x={obj_x:.4f}m (error={x_error:.4f}m, target={target_x:.4f}m), "
                              f"linear={linear_vel:.3f}m/s")
                    
                    time.sleep(control_period)
                    iteration += 1
                
                # Timeout - stop wheels and unsubscribe
                self.stop_wheels()
                self.ws_manager.send({"op": "unsubscribe", "topic": "/objects_poses"})
                return False, f"Timeout after {max_iterations} iterations"
                
            except KeyboardInterrupt:
                self.stop_wheels()
                self.ws_manager.send({"op": "unsubscribe", "topic": "/objects_poses"})
                print("\nInterrupted by user")
                return False, "Interrupted by user"
            except Exception as e:
                self.stop_wheels()
                self.ws_manager.send({"op": "unsubscribe", "topic": "/objects_poses"})
                print(f"Error during control loop: {e}")
                import traceback
                traceback.print_exc()
                return False, f"Error: {str(e)}"


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Visual servo to move robot forward/backward until object reaches target x-position (no rotation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 primitives/move_to_lego.py --object_name lego_1 --mode real
  python3 primitives/move_to_lego.py --object_name lego_2 --mode sim
        """
    )
    
    parser.add_argument(
        '--object_name',
        type=str,
        required=True,
        help='Object name to align to (e.g., lego_1, lego_2)'
    )
    
    default_mode = get_default_mode()
    parser.add_argument(
        '--mode',
        choices=['real', 'sim'],
        default=default_mode,
        help=f'Hardware mode: real for real hardware, sim for simulation (default: {default_mode} from config)'
    )
    
    parser.add_argument(
        '--linear_gain',
        type=float,
        default=LINEAR_GAIN,
        help=f'Linear velocity control gain (default: {LINEAR_GAIN})'
    )
    
    args = parser.parse_args()
    
    try:
        # Create controller
        controller = VisualServoLegoController(
            mode=args.mode, 
            object_name=args.object_name,
            linear_gain=args.linear_gain
        )
        
        # Give websocket time to establish connection
        time.sleep(0.5)
        
        # Perform alignment and movement
        success, message = controller.align_and_move_to_object()
        
        if success:
            print(f"Success: {message}")
            return 0
        else:
            print(f"Error: {message}")
            return 1
        
    except KeyboardInterrupt:
        print('\nInterrupted by user')
        return 1
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
