#!/usr/bin/env python3
"""
Gripper Control Script
Controls the JETANK gripper in both simulation and real hardware modes via websocket.

Usage:
    python3 primitives/control_gripper.py open --mode real
    python3 primitives/control_gripper.py close --mode sim
"""

import argparse
import time
import sys
import os
import json
import threading
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_utils import get_default_mode
from utils.websocket_manager import WebSocketManager, parse_json

# Trajectory duration in seconds
TRAJECTORY_DURATION = 1.0

# Force monitoring configuration
FORCE_THRESHOLD = 0.5  # Stop when both R2 and L2 exceed this value (N)
MAX_GRIPPER_RETRIES = 3  # Maximum retry attempts when force threshold is reached

# Gripper position limits (wrist joint angle in radians)
GRIPPER_MIN_ANGLE = 0.0  # Fully closed
GRIPPER_MAX_ANGLE = 1.22  # Fully open

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


def subscribe_for_duration_websocket(
    ws_manager: WebSocketManager,
    topic: str,
    msg_type: str,
    duration: float,
    callback_func=None
) -> Dict[str, Any]:
    """Subscribe to a ROS topic for a duration and collect messages"""
    subscribe_msg = {
        "op": "subscribe",
        "topic": topic,
        "type": msg_type,
    }
    
    collected_messages = []
    send_error = ws_manager.send(subscribe_msg)
    if send_error:
        return {"error": f"Failed to subscribe: {send_error}"}
    
    end_time = time.time() + duration
    while time.time() < end_time:
        response = ws_manager.receive(timeout=0.1)
        if response is None:
            continue
        
        msg_data = parse_json(response)
        if not msg_data:
            continue
        
        if msg_data.get("op") == "publish" and msg_data.get("topic") == topic:
            msg = msg_data.get("msg", {})
            collected_messages.append(msg)
            if callback_func:
                callback_func(msg)
    
    unsubscribe_msg = {"op": "unsubscribe", "topic": topic}
    ws_manager.send(unsubscribe_msg)
    
    return {"messages": collected_messages}


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


class GripperController:
    """Controller for controlling gripper using websocket"""
    
    def __init__(self, mode=None, ws_manager=None):
        if mode is None:
            mode = get_default_mode()
        
        self.mode = mode
        self.use_real_hardware = (mode == 'real')
        self.ws_manager = ws_manager or WebSocketManager(ROSBRIDGE_IP, ROSBRIDGE_PORT, default_timeout=5.0)
        
        # Gripper limits from URDF
        self.gripper_max_angle = 1.047198  # URDF gripper joint limit
        self.wrist_max_angle = 1.22  # Maximum wrist servo angle (fully open)
        
        # Force monitoring for gripper (simulation mode only)
        self.gripper_force_r2 = 0.0
        self.gripper_force_l2 = 0.0
        self.force_lock = threading.Lock()
        self.force_threshold = FORCE_THRESHOLD
        self.gripper_retry_count = 0
        self.max_gripper_retries = MAX_GRIPPER_RETRIES
        
        print(f'Gripper controller initialized in {mode} mode')
    
    def gripper_to_wrist_angle(self, gripper_angle):
        """Convert gripper finger angle to wrist servo angle"""
        # Reverse of wrist_to_gripper_angle
        # Actual wrist servo max range: 0.0 (closed) to 1.22 (fully open) radians
        wrist_angle = gripper_angle * self.wrist_max_angle / self.gripper_max_angle
        return wrist_angle
    
    def get_gripper_forces(self, timeout: float = 0.1):
        """Get current gripper forces (for simulation mode)"""
        if self.use_real_hardware:
            return 0.0, 0.0
        
        # Subscribe to both force topics quickly
        force_r2 = 0.0
        force_l2 = 0.0
        
        # Try to get latest force values
        result_r2 = subscribe_once_websocket(
            self.ws_manager,
            "/gripper_r2/contact_force",
            "std_msgs/msg/Float32",
            timeout=timeout
        )
        if "msg" in result_r2:
            force_r2 = abs(result_r2["msg"].get("data", 0.0))
        
        result_l2 = subscribe_once_websocket(
            self.ws_manager,
            "/gripper_l2/contact_force",
            "std_msgs/msg/Float32",
            timeout=timeout
        )
        if "msg" in result_l2:
            force_l2 = abs(result_l2["msg"].get("data", 0.0))
        
        with self.force_lock:
            self.gripper_force_r2 = force_r2
            self.gripper_force_l2 = force_l2
        
        return force_r2, force_l2
    
    def get_wrist_joint_position(self, timeout: float = 5.0):
        """Get current wrist joint position"""
        topic = "real_joint_states" if self.use_real_hardware else "joint_states"
        
        result = subscribe_once_websocket(
            self.ws_manager,
            f"/{topic}",
            "sensor_msgs/msg/JointState",
            timeout=timeout
        )
        
        if "error" in result:
            return None
        
        joint_state = result.get("msg", {})
        joint_names = joint_state.get("name", [])
        joint_positions = joint_state.get("position", [])
        
        # Find wrist_joint in real mode, or calculate from gripper joints in sim mode
        if self.use_real_hardware:
            # Real mode: look for wrist_joint
            try:
                wrist_idx = joint_names.index("wrist_joint")
                if wrist_idx < len(joint_positions):
                    return joint_positions[wrist_idx]
            except ValueError:
                return None
        else:
            # Sim mode: calculate from gripper R1 joint (which represents opening)
            # R1 angle directly represents the gripper opening (0 = closed, max_angle = open)
            try:
                r1_idx = joint_names.index("Revolute_GRIPPER_R1")
                if r1_idx < len(joint_positions):
                    gripper_angle = abs(joint_positions[r1_idx])
                    # Convert gripper angle to wrist angle (0-1.047198 -> 0-1.22)
                    return self.gripper_to_wrist_angle(gripper_angle)
            except ValueError:
                return None
        
        return None
    
    def send_gripper_command_real(self, target_wrist_angle, duration=TRAJECTORY_DURATION, current_wrist_angle=None):
        """Send gripper (wrist) command to real hardware with trajectory over duration"""
        if current_wrist_angle is None:
            current_wrist_angle = 0.0
        
        start_time = time.time()
        
        # Send commands continuously at 50Hz during trajectory execution (matching GUI)
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            t = min(1.0, elapsed / duration)
            
            # Smooth interpolation using cubic easing (matching GUI)
            t_smooth = 3 * t**2 - 2 * t**3  # Smooth start and end
            
            # Interpolate wrist angle
            current_angle = current_wrist_angle + (target_wrist_angle - current_wrist_angle) * t_smooth
            
            # Send command
            joint_msg = {
                "header": {
                    "stamp": {
                        "sec": int(time.time()),
                        "nanosec": int((time.time() % 1) * 1e9)
                    }
                },
                "name": ["wrist_joint"],
                "position": [current_angle]
            }
            
            result = publish_once_websocket(
                self.ws_manager,
                "joint_commands",
                "sensor_msgs/msg/JointState",
                joint_msg
            )
            
            if "error" in result:
                return result
            
            time.sleep(0.02)  # 50Hz update rate (matching GUI)
        
        # Ensure final position is set
        joint_msg = {
            "header": {
                "stamp": {
                    "sec": int(time.time()),
                    "nanosec": int((time.time() % 1) * 1e9)
                }
            },
            "name": ["wrist_joint"],
            "position": [target_wrist_angle]
        }
        
        return publish_once_websocket(
            self.ws_manager,
            "joint_commands",
            "sensor_msgs/msg/JointState",
            joint_msg
        )
    
    def send_gripper_command_sim(self, target_gripper_angles, duration=TRAJECTORY_DURATION, current_gripper_angles=None, is_closing=False):
        """Send gripper command to simulation via trajectory with force monitoring"""
        gripper_joint_names = [
            'revolute_GRIPPER_L1',
            'revolute_GRIPPER_L2',
            'Revolute_GRIPPER_R1',
            'Revolute_GRIPPER_R2'
        ]
        
        if current_gripper_angles is None:
            current_gripper_angles = [0.0, 0.0, 0.0, 0.0]
        
        # Reset retry counter for new operation
        self.gripper_retry_count = 0
        force_exceeded = False
        previous_above_threshold = False
        
        # Calculate number of steps (50 steps per second for smooth motion) - matching GUI
        steps = max(10, int(duration * 50))
        
        # Create trajectory points (matching GUI implementation)
        trajectory_points = []
        
        for i in range(steps + 1):
            # Calculate interpolation factor (0 to 1)
            t = i / steps if steps > 0 else 1.0
            
            # Smooth interpolation using cubic easing (matching GUI)
            t_smooth = 3 * t**2 - 2 * t**3  # Smooth start and end
            
            # Interpolate joint positions
            current_positions = []
            for j in range(4):  # 4 gripper joints
                pos = current_gripper_angles[j] + (target_gripper_angles[j] - current_gripper_angles[j]) * t_smooth
                current_positions.append(pos)
            
            # Create trajectory point (matching GUI format)
            point_time = t * duration
            point = {
                "positions": current_positions,
                "time_from_start": {
                    "sec": int(point_time),
                    "nanosec": int(((point_time % 1) * 1e9))
                }
            }
            trajectory_points.append(point)
        
        # Create and publish trajectory (matching GUI)
        trajectory_msg = {
            "joint_names": gripper_joint_names,
            "points": trajectory_points
        }
        
        result = publish_once_websocket(
            self.ws_manager,
            "arm_trajectory",
            "trajectory_msgs/msg/JointTrajectory",
            trajectory_msg
        )
        
        if "error" in result:
            return result
        
        # Also send joint_commands continuously during trajectory execution (like GUI does)
        start_time = time.time()
        start_joints = current_gripper_angles.copy()
        final_positions = target_gripper_angles  # Default to target
        
        while time.time() - start_time < duration and not force_exceeded:
            elapsed = time.time() - start_time
            t = min(1.0, elapsed / duration)
            
            # Smooth interpolation using cubic easing (matching GUI)
            t_smooth = 3 * t**2 - 2 * t**3
            
            # Calculate current positions
            current_positions = []
            for j in range(4):  # 4 gripper joints
                pos = start_joints[j] + (target_gripper_angles[j] - start_joints[j]) * t_smooth
                current_positions.append(pos)
            
            # Check forces if closing (matching GUI behavior)
            if is_closing:
                force_r2, force_l2 = self.get_gripper_forces(timeout=0.05)
                
                # Check if both forces exceed threshold
                current_above_threshold = (force_r2 >= self.force_threshold and force_l2 >= self.force_threshold)
                
                # Detect threshold crossing (went from below to above threshold)
                if current_above_threshold and not previous_above_threshold:
                    # Threshold just crossed - increment retry count
                    self.gripper_retry_count += 1
                    
                    print(f'Force threshold reached (retry {self.gripper_retry_count}/{self.max_gripper_retries}): R2={force_r2:.3f}N, L2={force_l2:.3f}N (threshold={self.force_threshold}N)')
                    
                    # If we've reached max retries, stop completely
                    if self.gripper_retry_count > self.max_gripper_retries:
                        force_exceeded = True
                        final_positions = current_positions
                        print(f'Max retries reached. Stopping gripper at current position.')
                        break
                    
                    # Otherwise, wait 1 second before continuing (retry)
                    time.sleep(1.0)
                    
                    # Check if forces are still above threshold after delay
                    force_r2_check, force_l2_check = self.get_gripper_forces(timeout=0.05)
                    if force_r2_check >= self.force_threshold and force_l2_check >= self.force_threshold:
                        # Forces still above threshold - stop instead of continuing
                        force_exceeded = True
                        final_positions = current_positions
                        print(f'Forces still above threshold after delay. Stopping gripper.')
                        break
                    
                    # Forces dropped below threshold - continue closing from current position
                    start_joints = current_positions.copy()
                    start_time = time.time()  # Reset timer for next retry
                    previous_above_threshold = False  # Reset to detect next threshold crossing
                    continue
                
                previous_above_threshold = current_above_threshold
            
            # Send joint command (GUI subscribes to this in sim mode)
            joint_msg = {
                "header": {
                    "stamp": {
                        "sec": int(time.time()),
                        "nanosec": int((time.time() % 1) * 1e9)
                    }
                },
                "name": gripper_joint_names,
                "position": current_positions
            }
            
            result = publish_once_websocket(
                self.ws_manager,
                "joint_commands",
                "sensor_msgs/msg/JointState",
                joint_msg
            )
            
            if "error" in result:
                return result
            
            time.sleep(0.02)  # 50Hz update rate (matching GUI)
        
        # Ensure final position is set (or current position if force stopped)
        if force_exceeded:
            print(f'Gripper stopped due to force threshold. Final position: L1={final_positions[0]:.3f}, L2={final_positions[1]:.3f}, R1={final_positions[2]:.3f}, R2={final_positions[3]:.3f}')
        
        joint_msg = {
            "header": {
                "stamp": {
                    "sec": int(time.time()),
                    "nanosec": int((time.time() % 1) * 1e9)
                }
            },
            "name": gripper_joint_names,
            "position": final_positions
        }
        
        return publish_once_websocket(
            self.ws_manager,
            "joint_commands",
            "sensor_msgs/msg/JointState",
            joint_msg
        )
    
    def wait_for_joint_state_settle(self, timeout=5.0, stability_threshold=0.01, required_stable_readings=5):
        """Wait for joint state to settle and return the stable value"""
        stable_value = None
        stable_count = 0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            current_value = self.get_wrist_joint_position(timeout=0.1)
            
            if current_value is not None:
                if stable_value is None:
                    stable_value = current_value
                    stable_count = 1
                elif abs(current_value - stable_value) < stability_threshold:
                    stable_count += 1
                    if stable_count >= required_stable_readings:
                        return current_value
                else:
                    # Value changed, reset stability check
                    stable_value = current_value
                    stable_count = 1
            
            time.sleep(0.1)
        
        # Return the last stable value we got, or None if we never got one
        return stable_value
    
    def log_gripper_position(self, initial_value=None, final_value=None):
        """Log gripper position in the specified format"""
        print(f"Gripper range: {GRIPPER_MIN_ANGLE:.2f} - {GRIPPER_MAX_ANGLE:.2f} rad")
        
        if initial_value is not None:
            print(f"Initial angle: {initial_value:.2f} rad")
        
        if final_value is not None:
            print(f"Current angle: {final_value:.2f} rad")
    
    def open_gripper(self, duration=TRAJECTORY_DURATION):
        """Open the gripper with trajectory over specified duration"""
        # Read initial position
        initial_value = None
        for _ in range(30):  # Try to get initial position (up to 3 seconds)
            initial_value = self.get_wrist_joint_position(timeout=0.1)
            if initial_value is not None:
                break
            time.sleep(0.1)
        
        if self.use_real_hardware:
            # Real hardware: send wrist joint trajectory
            target_wrist_angle = self.wrist_max_angle  # 1.22 rad = fully open
            result = self.send_gripper_command_real(target_wrist_angle, duration=duration, current_wrist_angle=initial_value)
            if "error" in result:
                print(f"Error: {result.get('error')}")
                return
        else:
            # Simulation: send individual gripper joint trajectory
            max_angle = self.gripper_max_angle
            target_gripper_angles = [
                -max_angle,  # L1 (negative)
                -max_angle,  # L2 (negative)
                max_angle,   # R1 (positive)
                -max_angle   # R2 (negative)
            ]
            result = self.send_gripper_command_sim(target_gripper_angles, duration=duration)
            if "error" in result:
                print(f"Error: {result.get('error')}")
                return
        
        # Wait for trajectory to complete
        time.sleep(duration + 0.2)
        
        # Wait for joint state to settle after trajectory completion
        final_value = self.wait_for_joint_state_settle(timeout=1.0)
        
        # Log gripper position
        self.log_gripper_position(initial_value, final_value)
    
    def close_gripper(self, duration=TRAJECTORY_DURATION):
        """Close the gripper with trajectory over specified duration"""
        # Read initial position
        initial_value = None
        for _ in range(30):  # Try to get initial position (up to 3 seconds)
            initial_value = self.get_wrist_joint_position(timeout=0.1)
            if initial_value is not None:
                break
            time.sleep(0.1)
        
        if self.use_real_hardware:
            # Real hardware: send wrist joint trajectory
            target_wrist_angle = 0.0  # 0 = closed
            result = self.send_gripper_command_real(target_wrist_angle, duration=duration, current_wrist_angle=initial_value)
            if "error" in result:
                print(f"Error: {result.get('error')}")
                return
        else:
            # Simulation: send individual gripper joint trajectory with force monitoring
            target_gripper_angles = [0.0, 0.0, 0.0, 0.0]  # All closed
            result = self.send_gripper_command_sim(target_gripper_angles, duration=duration, is_closing=True)
            if "error" in result:
                print(f"Error: {result.get('error')}")
                return
        
        # Wait for trajectory to complete
        time.sleep(duration + 0.2)
        
        # Wait for joint state to settle after trajectory completion
        final_value = self.wait_for_joint_state_settle(timeout=3.0)
        
        # Log gripper position
        self.log_gripper_position(initial_value, final_value)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Control JETANK gripper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 primitives/control_gripper.py open --mode real
  python3 primitives/control_gripper.py close --mode sim
        """
    )
    
    parser.add_argument(
        'action',
        choices=['open', 'close'],
        help='Gripper action: open or close'
    )
    
    default_mode = get_default_mode()
    parser.add_argument(
        '--mode',
        choices=['real', 'sim'],
        default=default_mode,
        help=f'Hardware mode: real for real hardware, sim for simulation (default: {default_mode} from config)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create gripper controller
        controller = GripperController(mode=args.mode)
        
        # Give websocket time to establish connection
        time.sleep(0.5)
        
        # Execute the requested action with trajectory
        trajectory_duration = TRAJECTORY_DURATION
        if args.action == 'open':
            controller.open_gripper(duration=trajectory_duration)
        elif args.action == 'close':
            controller.close_gripper(duration=trajectory_duration)
        
        return 0
        
    except KeyboardInterrupt:
        print('\nInterrupted by user')
        return 1
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
