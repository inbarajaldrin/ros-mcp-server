#!/usr/bin/env python3
"""
Reset Joints Primitive
Resets the JETANK arm joints to home position (0, 0, 0) via websocket.

Usage:
    python3 primitives/reset_joints.py --mode real
    python3 primitives/reset_joints.py --mode sim --duration 2.0
"""

import argparse
import time
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


class ResetJointsController:
    """Controller for resetting arm joints to home position using websocket"""
    
    def __init__(self, mode=None, ws_manager=None):
        if mode is None:
            mode = get_default_mode()
        
        self.mode = mode
        self.use_real_hardware = (mode == 'real')
        self.ws_manager = ws_manager or WebSocketManager(ROSBRIDGE_IP, ROSBRIDGE_PORT, default_timeout=5.0)
        
        # Arm joint names (matching GUI order)
        self.arm_joint_names = [
            'revolute_BEARING',           # 0
            'Revolute_SERVO_LOWER',        # 1
            'Revolute_SERVO_UPPER'         # 2
        ]
        
        print(f'Reset joints controller initialized in {mode} mode')
    
    def get_current_arm_joints(self, timeout: float = 5.0):
        """Get current arm joint positions"""
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
            servo_lower_idx = joint_names.index("Revolute_SERVO_LOWER")
            servo_upper_idx = joint_names.index("Revolute_SERVO_UPPER")
            
            return [
                joint_positions[bearing_idx],
                joint_positions[servo_lower_idx],
                joint_positions[servo_upper_idx]
            ]
        except (ValueError, IndexError):
            return None
    
    def send_arm_command_real(self, theta0, theta1, theta3):
        """Send arm command to real hardware (sends each joint separately, matching GUI behavior)"""
        joint_names_map = ['base_joint', 'shoulder_joint', 'elbow_joint']
        joint_values = [theta0, theta1, theta3]
        
        for joint_name, position in zip(joint_names_map, joint_values):
            joint_msg = {
                "header": {
                    "stamp": {
                        "sec": int(time.time()),
                        "nanosec": int((time.time() % 1) * 1e9)
                    }
                },
                "name": [joint_name],
                "position": [position]
            }
            
            result = publish_once_websocket(
                self.ws_manager,
                "joint_commands",
                "sensor_msgs/msg/JointState",
                joint_msg
            )
            
            if "error" in result:
                return result
        
        return {"success": True}
    
    def send_trajectory_sim(self, start_joints, target_joints, duration):
        """Send trajectory to simulation"""
        # Calculate number of steps (50 steps per second for smooth motion)
        steps = max(10, int(duration * 50))
        
        # Create trajectory points
        trajectory_points = []
        
        for i in range(steps + 1):
            # Calculate interpolation factor (0 to 1)
            t = i / steps if steps > 0 else 1.0
            
            # Smooth interpolation using cubic easing
            t_smooth = 3 * t**2 - 2 * t**3  # Smooth start and end
            
            # Interpolate joint positions
            current_positions = []
            for j in range(3):
                pos = start_joints[j] + (target_joints[j] - start_joints[j]) * t_smooth
                current_positions.append(pos)
            
            # Create trajectory point
            point_time = t * duration
            point = {
                "positions": current_positions,
                "time_from_start": {
                    "sec": int(point_time),
                    "nanosec": int(((point_time % 1) * 1e9))
                }
            }
            trajectory_points.append(point)
        
        # Create and publish trajectory
        trajectory_msg = {
            "joint_names": self.arm_joint_names,
            "points": trajectory_points
        }
        
        return publish_once_websocket(
            self.ws_manager,
            "arm_trajectory",
            "trajectory_msgs/msg/JointTrajectory",
            trajectory_msg
        )
    
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
            result = self.send_arm_command_real(current_positions[0], current_positions[1], current_positions[2])
            if "error" in result:
                return result
            
            time.sleep(0.02)  # 50Hz update rate
        
        # Ensure final position is set
        return self.send_arm_command_real(target_joints[0], target_joints[1], target_joints[2])
    
    def reset_joints(self, duration=TRAJECTORY_DURATION):
        """Reset arm joints to home position (0, 0, 0)"""
        # Get current joint positions
        current_joints = self.get_current_arm_joints(timeout=5.0)
        if current_joints is None:
            # Use default if current position unknown
            current_joints = [0.0, 0.785, -1.57]
            print("Warning: Current joint positions unknown, using default")
        
        # Target joint positions (home position)
        target_joints = [0.0, 0.0, 0.0]
        
        print(f"Resetting arm joints from [{current_joints[0]:.3f}, {current_joints[1]:.3f}, {current_joints[2]:.3f}] to [0.000, 0.000, 0.000]")
        
        # Send trajectory
        if self.use_real_hardware:
            result = self.send_trajectory_real(current_joints, target_joints, duration)
            if "error" in result:
                return False, result.get("error", "Failed to send trajectory")
        else:
            result = self.send_trajectory_sim(current_joints, target_joints, duration)
            if "error" in result:
                return False, result.get("error", "Failed to send trajectory")
        
        # Wait for trajectory to complete
        time.sleep(duration + 1.0)  # Wait for trajectory duration + 1 second for hardware settling
        
        print("Arm joints reset to home position (camera unchanged)")
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
    
    try:
        # Create controller
        controller = ResetJointsController(mode=args.mode)
        
        # Give websocket time to establish connection
        time.sleep(0.5)
        
        # Reset joints
        success, message = controller.reset_joints(duration=args.duration)
        
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
