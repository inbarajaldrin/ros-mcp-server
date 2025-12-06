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

# Wheel inversion compensation (set to True if wheels are inverted, False otherwise)
WHEELS_INVERTED = True

# Default websocket connection (can be overridden via environment or config)
ROSBRIDGE_IP = os.getenv("ROSBRIDGE_IP", "localhost")
ROSBRIDGE_PORT = int(os.getenv("ROSBRIDGE_PORT", "9090"))


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
    
    args = parser.parse_args()
    
    try:
        # Create wheel controller
        controller = WheelController(mode=args.mode)
        
        # Give websocket time to establish connection
        time.sleep(0.5)
        
        # Execute the requested action
        if args.linear == 0.0 and args.angular == 0.0:
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
