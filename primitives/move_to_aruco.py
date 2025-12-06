#!/usr/bin/env python3
"""
Visual Servo ArUco Primitive
Aligns the camera center to an ArUco marker by rotating the base via websocket.

Uses aruco_poses topic to get marker position and rotates base bearing joint
to center the marker in the camera view.

Usage:
    python3 primitives/move_to_aruco.py --aruco_id 1 --mode real
    python3 primitives/move_to_aruco.py --aruco_id 5 --mode sim
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

# Trajectory duration in seconds
TRAJECTORY_DURATION = 1.0

# Control parameters
ROTATION_GAIN = 1.0  # Gain for calculating target angles from marker position
Y_ALIGNMENT_TOLERANCE = 0.01  # Tolerance for y alignment (1cm)
TARGET_X_MIN = -0.3  # Minimum acceptable marker x-position (m)
TARGET_X_MAX = -0.27  # Maximum acceptable marker x-position (m)
Y_ANGULAR_SPEED = 0.2  # Angular velocity for y alignment (rad/s)
LINEAR_GAIN = 10.0  # Gain for linear velocity from marker x position
MAX_LINEAR_VELOCITY = 0.3  # Maximum linear velocity (m/s)
MIN_LINEAR_VELOCITY = 0.2  # Minimum linear velocity (m/s)
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


class VisualServoArUcoController:
    """Controller for visual servoing to ArUco markers using websocket"""
    
    def __init__(self, mode=None, aruco_id=1, rotation_gain=ROTATION_GAIN, ws_manager=None):
        if mode is None:
            mode = get_default_mode()
        
        self.mode = mode
        self.use_real_hardware = (mode == 'real')
        self.aruco_id = aruco_id
        self.target_marker_name = f"aruco_{aruco_id}"
        self.rotation_gain = rotation_gain
        self.ws_manager = ws_manager or WebSocketManager(ROSBRIDGE_IP, ROSBRIDGE_PORT, default_timeout=5.0)
        
        print(f'Visual servo ArUco controller initialized in {mode} mode, targeting marker {self.target_marker_name}')
    
    def get_marker_position(self, marker_name: str, timeout: float = 5.0):
        """Get position of a specific marker by name (returns position in meters)"""
        result = subscribe_once_websocket(
            self.ws_manager,
            "/aruco_poses",
            "tf2_msgs/msg/TFMessage",
            timeout=timeout
        )
        
        if "error" in result:
            return None, result.get("error")
        
        transforms = result.get("msg", {}).get("transforms", [])
        
        for transform in transforms:
            if transform.get("child_frame_id") == marker_name:
                translation = transform.get("transform", {}).get("translation", {})
                position = [
                    translation.get("x", 0.0),
                    translation.get("y", 0.0),
                    translation.get("z", 0.0)
                ]
                return position, None
        
        return None, f"Marker {marker_name} not found. Make sure the aruco_poses topic is publishing."
    
    def get_current_bearing_angle(self, timeout: float = 5.0):
        """Get current base bearing joint angle"""
        result = subscribe_once_websocket(
            self.ws_manager,
            "/joint_states",
            "sensor_msgs/msg/JointState",
            timeout=timeout
        )
        
        if "error" in result:
            return None
        
        joint_state = result.get("msg", {})
        joint_names = joint_state.get("name", [])
        joint_positions = joint_state.get("position", [])
        
        try:
            bearing_idx = joint_names.index("revolute_BEARING")
            return joint_positions[bearing_idx]
        except (ValueError, IndexError):
            return None
    
    def get_current_camera_angle(self, timeout: float = 5.0):
        """Get current camera tilt angle"""
        result = subscribe_once_websocket(
            self.ws_manager,
            "/joint_states",
            "sensor_msgs/msg/JointState",
            timeout=timeout
        )
        
        if "error" in result:
            return None
        
        joint_state = result.get("msg", {})
        joint_names = joint_state.get("name", [])
        joint_positions = joint_state.get("position", [])
        
        try:
            camera_idx = joint_names.index("revolute_CAMERA_HOLDER_ARM_LOWER")
            return joint_positions[camera_idx]
        except (ValueError, IndexError):
            return None
    
    def send_bearing_command(self, angle):
        """Send base bearing joint command"""
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
        
        return publish_once_websocket(
            self.ws_manager,
            "joint_commands",
            "sensor_msgs/msg/JointState",
            joint_msg
        )
    
    def send_camera_command(self, angle):
        """Send camera joint command"""
        joint_msg = {
            "header": {
                "stamp": {
                    "sec": int(time.time()),
                    "nanosec": int((time.time() % 1) * 1e9)
                }
            },
            "name": ["camera_joint"],
            "position": [angle]
        }
        
        return publish_once_websocket(
            self.ws_manager,
            "joint_commands",
            "sensor_msgs/msg/JointState",
            joint_msg
        )
    
    def send_joint_trajectory_real(self, start_bearing, start_camera, target_bearing, target_camera, duration):
        """Send trajectory to real hardware (continuous commands at 50Hz)"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            t = min(1.0, elapsed / duration)
            
            # Smooth interpolation using cubic easing
            t_smooth = 3 * t**2 - 2 * t**3
            
            # Interpolate joint positions
            current_bearing = start_bearing + (target_bearing - start_bearing) * t_smooth
            current_camera = start_camera + (target_camera - start_camera) * t_smooth
            
            # Send commands
            result1 = self.send_bearing_command(current_bearing)
            result2 = self.send_camera_command(current_camera)
            if "error" in result1:
                return result1
            if "error" in result2:
                return result2
            
            time.sleep(0.02)  # 50Hz update rate
        
        # Ensure final positions are set
        result1 = self.send_bearing_command(target_bearing)
        result2 = self.send_camera_command(target_camera)
        if "error" in result1:
            return result1
        return result2
    
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
            return self.send_wheel_command_sim(0.0, 0.0, log=False)
    
    def align_y_with_wheels(self, marker_position, max_iterations=200):
        """Align marker y position using angular velocity control (wheel control)"""
        marker_x, marker_y, marker_z = marker_position
        print(f"Aligning y position using wheels. Marker y={marker_y*1000:.1f}mm (target: 0mm, tolerance: ±{Y_ALIGNMENT_TOLERANCE*1000:.0f}mm)")
        
        control_period = 1.0 / CONTROL_RATE
        iteration = 0
        
        # Keep WebSocket connection open for the control loop
        with self.ws_manager:
            # Subscribe to aruco poses topic
            subscribe_msg = {
                "op": "subscribe",
                "topic": "/aruco_poses",
                "type": "tf2_msgs/msg/TFMessage",
            }
            
            self.ws_manager.send(subscribe_msg)
            time.sleep(0.1)  # Give time for subscription to register
            
            current_y = marker_y
            last_marker_update = time.time()
            
            try:
                while iteration < max_iterations:
                    # Receive messages from subscribed topic
                    messages_processed = 0
                    while messages_processed < 5:
                        response = self.ws_manager.receive(timeout=0.01)
                        if response:
                            msg_data = parse_json(response)
                            if msg_data:
                                # Handle marker position updates
                                if msg_data.get("op") == "publish" and msg_data.get("topic") == "/aruco_poses":
                                    transforms = msg_data.get("msg", {}).get("transforms", [])
                                    for transform in transforms:
                                        if transform.get("child_frame_id") == self.target_marker_name:
                                            translation = transform.get("transform", {}).get("translation", {})
                                            current_y = translation.get("y", 0.0)
                                            last_marker_update = time.time()
                                            messages_processed += 1
                                            break
                        else:
                            break
                    
                    # Check if we have recent data
                    current_time = time.time()
                    if (current_time - last_marker_update) > 0.5:
                        if iteration % 50 == 0:
                            print(f"Warning: No recent marker position data")
                        time.sleep(control_period)
                        iteration += 1
                        continue
                    
                    # Check if y is within tolerance
                    if abs(current_y) <= Y_ALIGNMENT_TOLERANCE:
                        self.stop_wheels()
                        print(f"Y alignment completed! Marker y={current_y*1000:.1f}mm (within tolerance: ±{Y_ALIGNMENT_TOLERANCE*1000:.0f}mm)")
                        self.ws_manager.send({"op": "unsubscribe", "topic": "/aruco_poses"})
                        return True, f"Successfully aligned y position. Final y: {current_y*1000:.1f}mm"
                    
                    # Calculate angular velocity based on y position
                    # If y is positive (marker to the right), rotate left (negative angular velocity)
                    # If y is negative (marker to the left), rotate right (positive angular velocity)
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
                self.ws_manager.send({"op": "unsubscribe", "topic": "/aruco_poses"})
                return False, f"Y alignment timeout after {max_iterations} iterations. Final y: {current_y*1000:.1f}mm"
                
            except KeyboardInterrupt:
                self.stop_wheels()
                self.ws_manager.send({"op": "unsubscribe", "topic": "/aruco_poses"})
                print("\nInterrupted by user")
                return False, "Interrupted by user"
            except Exception as e:
                self.stop_wheels()
                self.ws_manager.send({"op": "unsubscribe", "topic": "/aruco_poses"})
                print(f"Error during y alignment: {e}")
                import traceback
                traceback.print_exc()
                return False, f"Error: {str(e)}"
    
    def move_forward_to_marker(self, marker_position, max_iterations=500):
        """Move forward/backward using wheels until marker reaches target x-position (like move_to_lego)"""
        marker_x, marker_y, marker_z = marker_position
        print(f"Moving forward/backward to marker. Marker x={marker_x*1000:.1f}mm (target range: {TARGET_X_MIN*1000:.0f}mm to {TARGET_X_MAX*1000:.0f}mm)")
        
        control_period = 1.0 / CONTROL_RATE
        iteration = 0
        
        # Keep WebSocket connection open for the control loop
        with self.ws_manager:
            # Subscribe to aruco poses topic
            subscribe_msg = {
                "op": "subscribe",
                "topic": "/aruco_poses",
                "type": "tf2_msgs/msg/TFMessage",
            }
            
            self.ws_manager.send(subscribe_msg)
            time.sleep(0.1)  # Give time for subscription to register
            
            current_x = marker_x
            last_marker_update = time.time()
            
            try:
                while iteration < max_iterations:
                    # Receive messages from subscribed topic
                    messages_processed = 0
                    while messages_processed < 5:
                        response = self.ws_manager.receive(timeout=0.01)
                        if response:
                            msg_data = parse_json(response)
                            if msg_data:
                                # Handle marker position updates
                                if msg_data.get("op") == "publish" and msg_data.get("topic") == "/aruco_poses":
                                    transforms = msg_data.get("msg", {}).get("transforms", [])
                                    for transform in transforms:
                                        if transform.get("child_frame_id") == self.target_marker_name:
                                            translation = transform.get("transform", {}).get("translation", {})
                                            current_x = translation.get("x", 0.0)
                                            last_marker_update = time.time()
                                            messages_processed += 1
                                            break
                        else:
                            break
                    
                    # Check if we have recent data
                    current_time = time.time()
                    if (current_time - last_marker_update) > 0.5:
                        if iteration % 50 == 0:
                            print(f"Warning: No recent marker position data")
                        time.sleep(control_period)
                        iteration += 1
                        continue
                    
                    # Calculate error: marker x position vs target x position (target is upper bound)
                    target_x = TARGET_X_MAX  # Target is -0.14 (upper bound)
                    x_error = current_x - target_x
                    
                    # Check if we've reached the target (acceptable range)
                    if TARGET_X_MIN <= current_x <= TARGET_X_MAX:
                        self.stop_wheels()
                        print(f"X alignment completed! Marker x-position: {current_x:.4f}m (acceptable range: {TARGET_X_MIN:.3f} to {TARGET_X_MAX:.3f})")
                        self.ws_manager.send({"op": "unsubscribe", "topic": "/aruco_poses"})
                        return True, f"Successfully reached marker. Marker x-position: {current_x:.4f}m"
                    
                    # Calculate control velocities
                    # Angular: set to 0 - no rotation, only linear movement
                    angular_vel = 0.0
                    
                    # Linear: move forward/backward to reach target x position
                    # Pattern: current_x < target (e.g., -0.16 < -0.14) → move forward
                    #         current_x > target (e.g., -0.13 > -0.14) → move backward
                    if current_x < target_x:  # Marker is more negative (farther from base)
                        linear_vel = LINEAR_GAIN * abs(x_error)  # Move forward
                    else:  # Marker is less negative (closer to base)
                        linear_vel = -LINEAR_GAIN * abs(x_error)  # Move backward
                    
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
                    
                    # Send wheel command
                    if self.use_real_hardware:
                        result = self.send_wheel_command_real(linear_vel, angular_vel, log=(iteration % 50 == 0))
                    else:
                        result = self.send_wheel_command_sim(linear_vel, angular_vel, log=(iteration % 50 == 0))
                    
                    if "error" in result:
                        print(f"Error sending wheel command: {result.get('error')}")
                    
                    # Log status periodically
                    if iteration % 50 == 0:
                        print(f"Iteration {iteration}: Marker x={current_x:.4f}m (error={x_error:.4f}m, target={target_x:.4f}m), "
                              f"linear={linear_vel:.3f}m/s")
                    
                    time.sleep(control_period)
                    iteration += 1
                
                # Timeout - stop wheels and unsubscribe
                self.stop_wheels()
                self.ws_manager.send({"op": "unsubscribe", "topic": "/aruco_poses"})
                return False, f"X alignment timeout after {max_iterations} iterations. Final x: {current_x*1000:.1f}mm"
                
            except KeyboardInterrupt:
                self.stop_wheels()
                self.ws_manager.send({"op": "unsubscribe", "topic": "/aruco_poses"})
                print("\nInterrupted by user")
                return False, "Interrupted by user"
            except Exception as e:
                self.stop_wheels()
                self.ws_manager.send({"op": "unsubscribe", "topic": "/aruco_poses"})
                print(f"Error during x alignment: {e}")
                import traceback
                traceback.print_exc()
                return False, f"Error: {str(e)}"
    
    def align_to_marker(self, duration=TRAJECTORY_DURATION):
        """Align to marker: y alignment (wheels), then x alignment (wheels) - same order as move_to_lego"""
        print(f"Aligning to marker {self.target_marker_name}")
        print(f"Step 1: Y alignment using wheels (target: 0mm, tolerance: ±{Y_ALIGNMENT_TOLERANCE*1000:.0f}mm)")
        print(f"Step 2: X alignment using wheels (target range: {TARGET_X_MIN*1000:.0f}mm to {TARGET_X_MAX*1000:.0f}mm)")
        
        # Step 1: Y alignment using wheels
        print("Starting y alignment using wheels...")
        # Get initial marker position for y alignment
        position, error = self.get_marker_position(self.target_marker_name, timeout=5.0)
        if position is None:
            return False, error or "Failed to get initial marker position for y alignment"
        
        # Perform y alignment using angular velocity control (wheels)
        align_success, align_message = self.align_y_with_wheels(position)
        if not align_success:
            print(f"Warning: Y alignment failed: {align_message}")
            # Continue with x alignment anyway
        
        # Step 2: X alignment using wheels (forward/backward movement)
        print("Starting x alignment using wheels (forward/backward movement)...")
        # Get updated marker position after y alignment
        position, error = self.get_marker_position(self.target_marker_name, timeout=5.0)
        if position is None:
            return False, error or "Failed to get marker position for x alignment"
        
        move_success, move_message = self.move_forward_to_marker(position)
        if not move_success:
            return False, move_message
        
        print(f"All alignment steps completed successfully!")
        return True, f"Successfully aligned to marker. {align_message if align_success else ''} {move_message}"


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Visual servo to align camera center to ArUco marker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 primitives/move_to_aruco.py --aruco_id 1 --mode real
  python3 primitives/move_to_aruco.py --aruco_id 5 --mode sim
  python3 primitives/move_to_aruco.py --aruco_id 8 --mode real --gain 1.5
        """
    )
    
    parser.add_argument(
        '--aruco_id',
        type=int,
        required=True,
        help='ArUco marker ID to align to (e.g., 1, 5, 8)'
    )
    
    default_mode = get_default_mode()
    parser.add_argument(
        '--mode',
        choices=['real', 'sim'],
        default=default_mode,
        help=f'Hardware mode: real for real hardware, sim for simulation (default: {default_mode} from config)'
    )
    
    parser.add_argument(
        '--gain',
        type=float,
        default=ROTATION_GAIN,
        help=f'Rotation control gain (default: {ROTATION_GAIN})'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=TRAJECTORY_DURATION,
        help=f'Trajectory duration in seconds (default: {TRAJECTORY_DURATION})'
    )
    
    args = parser.parse_args()
    
    try:
        # Create controller
        controller = VisualServoArUcoController(
            mode=args.mode, 
            aruco_id=args.aruco_id,
            rotation_gain=args.gain
        )
        
        # Give websocket time to establish connection
        time.sleep(0.5)
        
        # Perform alignment
        success, message = controller.align_to_marker(duration=args.duration)
        
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
