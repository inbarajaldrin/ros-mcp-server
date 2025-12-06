#!/usr/bin/env python3
"""
Move Camera Primitive
Moves the JETANK camera to a specified angle via websocket.

Usage:
    python3 primitives/move_camera.py --angle -45 --mode real
    python3 primitives/move_camera.py --angle 0 --mode sim
    python3 primitives/move_camera.py down --mode real
    python3 primitives/move_camera.py reset --mode sim
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
CAMERA_TRAJECTORY_DURATION = 1.0

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


class MoveCameraController:
    """Controller for moving camera using websocket"""
    
    def __init__(self, mode=None, ws_manager=None):
        if mode is None:
            mode = get_default_mode()
        
        self.mode = mode
        self.use_real_hardware = (mode == 'real')
        self.ws_manager = ws_manager or WebSocketManager(ROSBRIDGE_IP, ROSBRIDGE_PORT, default_timeout=5.0)
        
        print(f'Move camera controller initialized in {mode} mode')
    
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
                result = self.send_camera_command(current_angle)
                if "error" in result:
                    return result
            
            iteration_count += 1
            time.sleep(0.02)  # 50Hz update rate
        
        # Ensure final position is set
        return self.send_camera_command(target_angle)
    
    def move_camera(self, target_angle, duration=CAMERA_TRAJECTORY_DURATION):
        """Move camera to target angle (in radians)"""
        # Get current camera angle
        current_angle = self.get_current_camera_angle(timeout=5.0)
        if current_angle is None:
            current_angle = 0.0
            print("Warning: Current camera angle unknown, assuming 0.0")
        
        target_degrees = math.degrees(target_angle)
        current_degrees = math.degrees(current_angle)
        
        print(f"Moving camera from {current_degrees:.1f}° to {target_degrees:.1f}°")
        
        # Send trajectory (always use trajectory for camera, matching GUI)
        if self.use_real_hardware:
            result = self.send_camera_trajectory_real(current_angle, target_angle, duration)
            if "error" in result:
                return False, result.get("error", "Failed to send trajectory")
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
                result = self.send_camera_command(current_angle_interp)
                if "error" in result:
                    return False, result.get("error", "Failed to send command")
                
                time.sleep(0.02)  # 50Hz update rate
            
            # Ensure final position is set
            result = self.send_camera_command(target_angle)
            if "error" in result:
                return False, result.get("error", "Failed to send final command")
        
        # Wait for trajectory to complete
        time.sleep(duration + 0.2)
        
        print(f"Camera moved to {target_degrees:.1f}°")
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
    
    try:
        # Create controller
        controller = MoveCameraController(mode=args.mode)
        
        # Give websocket time to establish connection
        time.sleep(0.5)
        
        # Move camera
        success, message = controller.move_camera(target_angle, duration=args.duration)
        
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
