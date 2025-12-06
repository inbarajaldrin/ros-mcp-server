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
    
    def align_to_marker(self, duration=TRAJECTORY_DURATION):
        """Align to marker by moving base and camera once"""
        print(f"Aligning to marker {self.target_marker_name}")
        
        # Get marker position
        position, error = self.get_marker_position(self.target_marker_name, timeout=5.0)
        if position is None:
            return False, error or "Failed to get marker position"
        
        x, y, z = position
        
        print(f"Marker position: x={x:.4f}m, y={y:.4f}m, z={z:.4f}m")
        
        # Get current angles
        current_bearing = self.get_current_bearing_angle(timeout=5.0)
        if current_bearing is None:
            current_bearing = 0.0
            print("Warning: Current bearing angle unknown, assuming 0.0")
        
        current_camera = self.get_current_camera_angle(timeout=5.0)
        if current_camera is None:
            current_camera = 0.0
            print("Warning: Current camera angle unknown, assuming 0.0")
        
        # Calculate target angles based on marker position
        # x < 0 means marker is to the left, so rotate base RIGHT (positive) to center it
        # x > 0 means marker is to the right, so rotate base LEFT (negative) to center it
        # y < 0 means marker is below, so tilt camera DOWN (negative) to center it
        # y > 0 means marker is above, so tilt camera UP (positive) to center it
        # Use proportional control with gain
        bearing_delta = -self.rotation_gain * x  # Negative x (left) -> positive rotation (right), positive x (right) -> negative rotation (left)
        camera_delta = -self.rotation_gain * y  # Negative y (below) -> positive tilt (up), positive y (above) -> negative tilt (down)
        
        target_bearing = current_bearing + bearing_delta
        target_camera = current_camera + camera_delta
        
        # Clamp to joint limits
        # Base bearing: -1.5708 to 1.5708 rad
        target_bearing = max(-1.5708, min(1.5708, target_bearing))
        # Camera: -0.785398 to 0.785398 rad (±45 degrees)
        target_camera = max(-0.785398, min(0.785398, target_camera))
        
        print(f"Moving base: {current_bearing:.4f} -> {target_bearing:.4f} rad")
        print(f"Moving camera: {current_camera:.4f} -> {target_camera:.4f} rad")
        
        # Send trajectory
        if self.use_real_hardware:
            result = self.send_joint_trajectory_real(current_bearing, current_camera, target_bearing, target_camera, duration)
            if "error" in result:
                return False, result.get("error", "Failed to send trajectory")
        else:
            # For simulation, send commands continuously
            start_time = time.time()
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                t = min(1.0, elapsed / duration)
                
                # Smooth interpolation
                t_smooth = 3 * t**2 - 2 * t**3
                
                current_bearing_interp = current_bearing + (target_bearing - current_bearing) * t_smooth
                current_camera_interp = current_camera + (target_camera - current_camera) * t_smooth
                
                result1 = self.send_bearing_command(current_bearing_interp)
                result2 = self.send_camera_command(current_camera_interp)
                if "error" in result1:
                    return False, result1.get("error", "Failed to send bearing command")
                if "error" in result2:
                    return False, result2.get("error", "Failed to send camera command")
                
                time.sleep(0.02)  # 50Hz update rate
            
            # Ensure final positions are set
            result1 = self.send_bearing_command(target_bearing)
            result2 = self.send_camera_command(target_camera)
            if "error" in result1:
                return False, result1.get("error", "Failed to send final bearing command")
            if "error" in result2:
                return False, result2.get("error", "Failed to send final camera command")
        
        # Wait for trajectory to complete
        time.sleep(duration + 0.2)
        
        print(f"Alignment completed. Base: {target_bearing:.4f} rad, Camera: {target_camera:.4f} rad")
        return True, f"Successfully aligned to marker. Base: {math.degrees(target_bearing):.1f}°, Camera: {math.degrees(target_camera):.1f}°"


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
