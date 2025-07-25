from mcp.server.fastmcp import FastMCP, Image
from typing import List, Any, Optional, Union
from pathlib import Path
import json
from utils.websocket_manager import WebSocketManager
from msgs.geometry_msgs import Twist
from msgs.sensor_msgs import Image as RosImage, JointState
import subprocess

#camera
import time
import os
from datetime import datetime
import io
import numpy as np
import cv2
from PIL import Image as PILImage
import threading
import signal

#ik
import tempfile
import os
from typing import List, Dict, Any
import numpy as np
from scipy.spatial.transform import Rotation as R

LOCAL_IP = "192.168.1.164"  # Replace with your local IP address
ROSBRIDGE_IP = "localhost"  # Replace with your rosbridge server IP address
ROSBRIDGE_PORT = 9090

mcp = FastMCP("ros-mcp-server")

# Global WebSocket manager - don't close it after every operation
ws_manager = WebSocketManager(ROSBRIDGE_IP, ROSBRIDGE_PORT, LOCAL_IP)

@mcp.tool()
def get_topics():
    topic_info = ws_manager.get_topics()
    # Don't close the connection here - keep it alive for other operations
    
    if topic_info:
        topics, types = zip(*topic_info)
        return {
            "topics": list(topics),
            "types": list(types)
        }
    else:
        return "No topics found"

@mcp.tool()
def pub_twist(linear: List[Any], angular: List[Any]):
    twist = Twist(ws_manager, topic="/cmd_vel")
    msg = twist.publish(linear, angular)
    # Don't close the main connection, just this specific publisher if needed
    
    if msg is not None:
        return "Twist message published successfully"
    else:
        return "No message published"

@mcp.tool()
def pub_twist_seq(linear: List[Any], angular: List[Any], duration: List[Any]):
    twist = Twist(ws_manager, topic="/cmd_vel")
    twist.publish_sequence(linear, angular, duration)

@mcp.tool()
def pub_jointstate(name: list[str], position: list[float], velocity: list[float], effort: list[float]):
    jointstate = JointState(ws_manager, topic="/joint_states")
    msg = jointstate.publish(name, position, velocity, effort)
    # Don't close the main connection
    
    if msg is not None:
        return "JointState message published successfully"
    else:
        return "No message published"

@mcp.tool()
def sub_jointstate():
    jointstate = JointState(ws_manager, topic="/joint_states")
    msg = jointstate.subscribe()
    # Don't close the main connection
    
    if msg is not None:
        return msg
    else:
        return "No JointState data received"

def _np_to_mcp_image(arr_rgb):
    """Convert numpy array to MCP Image format."""
    # Convert numpy array to PIL Image
    pil_image = PILImage.fromarray(arr_rgb)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Return MCP Image
    return Image(data=img_byte_arr, format="jpeg")

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

@mcp.tool(description="Capture camera image from any topic and return it so the agent can see and analyze it. Works with any camera topic including isometric cameras.")
def capture_camera_image(topic_name: str, timeout: int = 10):
    """
    Capture camera image using any camera topic.
    Works with intel cameras, isometric cameras, and any other image topics.
    Returns a list with status info and the image that the agent can see.
    
    Args:
        topic_name: The ROS topic to subscribe to (e.g., "/intel_camera_rgb", "/isometric_camera/image_raw")
        timeout: Timeout in seconds for image capture
    """
    result_json = {
        "timestamp": datetime.now().isoformat(),
        "topic": topic_name,
        "status": "attempting"
    }
    
    try:
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        # Create dynamic image subscriber for the specified topic
        image_subscriber = RosImage(ws_manager, topic=topic_name)
        
        # Subscribe and get image data
        msg = image_subscriber.subscribe()
        
        # Cancel timeout
        signal.alarm(0)
        
        result_json["status"] = "success"
        
        if msg is not None and 'data' in msg:
            image_data = msg['data']
            
            # Convert the image data to numpy array (RGB format)
            # Handle different data types and formats
            if isinstance(image_data, np.ndarray):
                # Ensure proper format for RGB conversion
                if len(image_data.shape) == 3:
                    if image_data.shape[2] == 3:
                        # Already RGB or BGR - assume RGB
                        arr_rgb = image_data.astype(np.uint8)
                    elif image_data.shape[2] == 4:
                        # RGBA to RGB
                        arr_rgb = image_data[:, :, :3].astype(np.uint8)
                    else:
                        raise Exception(f"Unsupported number of channels: {image_data.shape[2]}")
                elif len(image_data.shape) == 2:
                    # Grayscale - convert to RGB
                    gray = image_data.astype(np.uint8)
                    arr_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                else:
                    raise Exception(f"Unsupported image shape: {image_data.shape}")
            else:
                raise Exception(f"Unexpected data type: {type(image_data)}. Expected numpy array.")
            
            # Use the working conversion function
            mcp_image = _np_to_mcp_image(arr_rgb)
            
            # Save backup screenshot with topic-specific naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Clean topic name for filename
            topic_clean = topic_name.replace("/", "_").replace(":", "_")
            filename = f"screenshots/{timestamp}_{topic_clean}.jpg"
            os.makedirs("screenshots", exist_ok=True)
            bgr_image = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, bgr_image)
            
            result_json["message"] = f"Image captured from {topic_name}"
            result_json["saved_to"] = filename
            
            # Add metadata from the message if available
            if 'metadata' in msg:
                result_json["image_metadata"] = msg['metadata']
            
            # Return in robot_MCP pattern: [json, image]
            return [result_json, mcp_image]
            
        else:
            # No live data, try to use latest screenshot
            result_json["status"] = "fallback"
            result_json["message"] = f"No live data from {topic_name}, using latest screenshot"
            
            try:
                screenshot_path = "screenshots"
                if os.path.exists(screenshot_path):
                    files = sorted([f for f in os.listdir(screenshot_path) if f.endswith('.jpg') or f.endswith('.png')])
                    if files:
                        latest_file = os.path.join(screenshot_path, files[-1])
                        with open(latest_file, 'rb') as f:
                            raw_data = f.read()
                        mcp_image = Image(data=raw_data, format="jpeg")
                        result_json["used_screenshot"] = files[-1]
                        return [result_json, mcp_image]
            except Exception as e:
                result_json["screenshot_error"] = str(e)
            
            raise Exception(f"No image data received from {topic_name} and no screenshots available")
            
    except TimeoutError:
        signal.alarm(0)  # Cancel alarm
        error_result = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic_name,
            "status": "timeout",
            "error": f"Image capture timed out after {timeout} seconds"
        }
        
        # Try fallback to screenshot
        try:
            screenshot_path = "screenshots"
            if os.path.exists(screenshot_path):
                files = sorted([f for f in os.listdir(screenshot_path) if f.endswith('.jpg') or f.endswith('.png')])
                if files:
                    latest_file = os.path.join(screenshot_path, files[-1])
                    with open(latest_file, 'rb') as f:
                        raw_data = f.read()
                    mcp_image = Image(data=raw_data, format="jpeg")
                    error_result["status"] = "timeout_fallback"
                    error_result["message"] = "Timed out, using latest screenshot"
                    error_result["used_screenshot"] = files[-1]
                    return [error_result, mcp_image]
        except:
            pass
            
        return [error_result]
        
    except Exception as e:
        signal.alarm(0)  # Cancel alarm
        error_result = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic_name,
            "status": "error",
            "error": str(e)
        }
        return [error_result]

@mcp.tool()
def test_screenshot_loading():
    """
    Test loading the latest screenshot using the robot_MCP pattern.
    This should work even if live camera data isn't available.
    """
    try:
        result_json = {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "method": "screenshot_loading"
        }
        
        screenshot_path = "screenshots"
        if os.path.exists(screenshot_path):
            files = sorted([f for f in os.listdir(screenshot_path) if f.endswith('.jpg') or f.endswith('.png')])
            if files:
                latest_file = os.path.join(screenshot_path, files[-1])
                result_json["screenshot_used"] = files[-1]
                result_json["file_size"] = os.path.getsize(latest_file)
                
                with open(latest_file, 'rb') as f:
                    raw_data = f.read()
                mcp_image = Image(data=raw_data, format="jpeg")
                
                return [result_json, mcp_image]
            else:
                result_json["status"] = "error"
                result_json["error"] = "No screenshot files found"
        else:
            result_json["status"] = "error"
            result_json["error"] = "Screenshots folder does not exist"
            
        return [result_json]
        
    except Exception as e:
        error_result = {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }
        return [error_result]

@mcp.tool()
def read_topic(topic_name: str, timeout: int = 5):
    """
    Read data from any ROS topic using ros2 topic echo --once command.
    Works with standard ROS2 message types like PoseStamped, Twist, JointState, etc.
    
    Args:
        topic_name: The ROS topic to subscribe to (e.g., "/object_poses/jenga_2", "/cmd_vel")
        timeout: Timeout in seconds for message capture (default: 5)
    
    Returns:
        Dictionary containing the topic data or error information
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "topic": topic_name,
        "status": "attempting"
    }
    
    try:
        # Source ROS2 and run the command in bash
        cmd = f"source /opt/ros/humble/setup.bash && timeout {timeout} ros2 topic echo {topic_name} --once"
        
        process_result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',  # Explicitly use bash
            capture_output=True,
            text=True,
            timeout=timeout + 2  # Add buffer for subprocess timeout
        )
        
        if process_result.returncode == 0:
            result["status"] = "success"
            result["message_data"] = process_result.stdout.strip()
            result["message"] = f"Successfully read data from {topic_name}"
            return result
        elif process_result.returncode == 124:  # timeout command exit code
            result["status"] = "timeout"
            result["error"] = f"No message received from {topic_name} within {timeout} seconds"
            if process_result.stderr:
                result["stderr"] = process_result.stderr.strip()
            return result
        else:
            result["status"] = "error"
            result["error"] = f"Command failed with return code {process_result.returncode}"
            if process_result.stderr:
                result["stderr"] = process_result.stderr.strip()
            return result
            
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = f"Command timed out after {timeout} seconds"
        return result
        
    except FileNotFoundError:
        result["status"] = "error"
        result["error"] = "ros2 command not found. Make sure ROS2 is properly installed and sourced."
        return result
        
    except Exception as e:
        import traceback
        result["status"] = "error"
        result["error"] = f"Failed to read topic {topic_name}: {str(e)}"
        result["traceback"] = traceback.format_exc()
        return result

@mcp.tool()
def perform_ik(target_position: List[float], target_rpy: List[float], 
               duration: float = 3.0, 
               custom_lib_path: str = "/home/aaugus11/Desktop/ros2_ws/src/ur_asu-main/ur_asu/custom_libraries") -> Dict[str, Any]:
    """
    Perform inverse kinematics and execute smooth trajectory movement using ROS2.
    
    Args:
        target_position: [x, y, z] target position in meters
        target_rpy: [roll, pitch, yaw] target orientation in degrees
        duration: Time to complete the movement in seconds (default: 3.0)
        custom_lib_path: Path to your custom IK solver library
        
    Returns:
        Dictionary with execution result including joint angles and trajectory execution status.
    """
    try:
        import sys
        
        # Add custom libraries to Python path if not already there
        if custom_lib_path not in sys.path:
            sys.path.append(custom_lib_path)
        
        # Import the IK solver
        try:
            from ik_solver import compute_ik
        except ImportError as e:
            return {
                "status": "error",
                "message": f"Failed to import ik_solver: {str(e)}. Check if {custom_lib_path}/ik_solver.py exists."
            }
        
        # Solve IK
        joint_angles = compute_ik(position=target_position, rpy=target_rpy)
        
        if joint_angles is not None:
            joint_angles_deg = np.degrees(joint_angles)
            
            # Execute joint trajectory using ROS2 action directly
            trajectory_result = execute_joint_trajectory(joint_angles.tolist(), duration)
            
            if trajectory_result.get("status") == "success":
                return {
                    "status": "success",
                    "message": "IK solved and trajectory executed successfully",
                    "target_position": target_position,
                    "target_rpy": target_rpy,
                    "joint_angles_rad": joint_angles.tolist(),
                    "joint_angles_deg": joint_angles_deg.tolist(),
                    "duration": duration,
                    "trajectory_status": "executed"
                }
            else:
                return {
                    "status": "partial_success",
                    "message": f"IK solved but trajectory execution failed: {trajectory_result.get('message', 'Unknown error')}",
                    "target_position": target_position,
                    "target_rpy": target_rpy,
                    "joint_angles_rad": joint_angles.tolist(),
                    "joint_angles_deg": joint_angles_deg.tolist(),
                    "duration": duration,
                    "trajectory_status": "failed"
                }
        else:
            return {
                "status": "error",
                "message": "IK solver failed to find a solution for the given target pose",
                "target_position": target_position,
                "target_rpy": target_rpy,
                "duration": duration
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error in IK computation: {str(e)}"
        }

@mcp.tool()
def execute_joint_trajectory(joint_angles: List[float], duration: float = 3.0) -> Dict[str, Any]:
    """
    Execute joint trajectory using ROS2 FollowJointTrajectory action.
    Simple tool that uses ros2 action send_goal command directly.
    
    Args:
        joint_angles: Target joint angles in radians [j1, j2, j3, j4, j5, j6]
        duration: Time to complete the movement in seconds (default: 3.0)
        
    Returns:
        Dictionary with execution result.
    """
    try:
        # Validate inputs
        if len(joint_angles) != 6:
            return {
                "status": "error",
                "message": f"Expected 6 joint angles, got {len(joint_angles)}"
            }
        
        # Format joint angles for ROS2 action command
        positions_str = "[" + ", ".join([f"{angle:.6f}" for angle in joint_angles]) + "]"
        
        # Create the action goal message
        action_goal = f'''{{
  trajectory: {{
    joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
    points: [{{
      positions: {positions_str},
      velocities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      time_from_start: {{sec: {int(duration)}, nanosec: {int((duration % 1) * 1e9)}}}
    }}]
  }},
  goal_time_tolerance: {{sec: 1, nanosec: 0}}
}}'''
        
        # Use ros2 action send_goal (same pattern as your read_topic tool)
        cmd = f"source /opt/ros/humble/setup.bash && source ~/Desktop/ros2_ws/install/setup.bash && ros2 action send_goal /scaled_joint_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory '{action_goal}' --feedback"
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=duration + 10
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Joint trajectory executed successfully",
                "joint_angles": joint_angles,
                "duration": duration,
                "ros_output": result.stdout.strip() if result.stdout else None
            }
        else:
            return {
                "status": "error",
                "message": f"ROS2 action failed: {result.stderr.strip()}",
                "joint_angles": joint_angles,
                "ros_output": result.stdout.strip() if result.stdout else None
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": f"ROS2 action timed out after {duration + 10} seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error in joint trajectory execution: {str(e)}"
        }

@mcp.tool()
def get_ee_pose(joint_angles: List[float] = None,
                custom_lib_path: str = "/home/aaugus11/Desktop/ros2_ws/src/ur_asu-main/ur_asu/custom_libraries") -> Dict[str, Any]:
    """
    Get end-effector pose using forward kinematics from specified or current joint angles.
    
    Args:
        joint_angles: Optional joint angles in radians. If None, gets current from ROS2
        custom_lib_path: Path to your custom IK solver library
        
    Returns:
        Dictionary with end-effector position, orientation, and joint angles used.
    """
    try:
        import sys
        
        # Add custom libraries to Python path
        if custom_lib_path not in sys.path:
            sys.path.append(custom_lib_path)
        
        # Import the IK solver module
        try:
            from ik_solver import forward_kinematics, dh_params
        except ImportError as e:
            return {
                "status": "error",
                "message": f"Failed to import from ik_solver: {str(e)}"
            }
        
        # Get joint angles (either provided or current from ROS2)
        if joint_angles is None:
            # Use your existing read_topic function to get current joint states
            joint_result = read_topic("/joint_states", timeout=5)
            
            if joint_result.get("status") != "success":
                return {
                    "status": "error",
                    "message": f"Failed to get current joint states: {joint_result.get('error', 'Unknown error')}"
                }
            
            # Parse joint positions from the topic data
            # This is a simplified parser - you might need to improve it based on your message format
            try:
                import re
                output = joint_result.get("message_data", "")
                positions_match = re.search(r'position:\s*\[(.*?)\]', output, re.DOTALL)
                if positions_match:
                    positions_str = positions_match.group(1)
                    joint_angles = [float(x.strip()) for x in positions_str.split(',')]
                    source = "current_ros2_joint_states"
                else:
                    return {
                        "status": "error",
                        "message": "Could not parse joint positions from ROS2 joint_states topic"
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to parse joint states: {str(e)}"
                }
        else:
            joint_angles = np.array(joint_angles)
            source = "provided_joint_angles"
        
        # Convert to numpy array
        joint_angles = np.array(joint_angles)
        
        # Compute forward kinematics
        try:
            T_ee = forward_kinematics(dh_params, joint_angles)
            
            # Extract position and orientation
            ee_position = T_ee[:3, 3]
            ee_rotation_matrix = T_ee[:3, :3]
            
            # Convert rotation matrix to RPY
            rotation = R.from_matrix(ee_rotation_matrix)
            ee_rpy_rad = rotation.as_euler('xyz', degrees=False)
            ee_rpy_deg = rotation.as_euler('xyz', degrees=True)
            ee_quaternion = rotation.as_quat()  # [x, y, z, w]
            
            # Convert joint angles to degrees
            joint_angles_deg = np.degrees(joint_angles)
            
            return {
                "status": "success",
                "message": "Forward kinematics computed successfully",
                "joint_angles_source": source,
                "joint_angles_rad": joint_angles.tolist(),
                "joint_angles_deg": joint_angles_deg.tolist(),
                "ee_position": ee_position.tolist(),  # [x, y, z] in meters
                "ee_rpy_rad": ee_rpy_rad.tolist(),   # [roll, pitch, yaw] in radians
                "ee_rpy_deg": ee_rpy_deg.tolist(),   # [roll, pitch, yaw] in degrees
                "ee_quaternion_xyzw": ee_quaternion.tolist(),  # [x, y, z, w]
                "transformation_matrix": T_ee.tolist()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to compute forward kinematics: {str(e)}",
                "joint_angles_rad": joint_angles.tolist() if joint_angles is not None else None
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error in forward kinematics computation: {str(e)}"
        }

@mcp.tool()
def close_connections():
    """
    Manually close WebSocket connections when needed.
    """
    try:
        ws_manager.close()
        return {"status": "success", "message": "WebSocket connections closed"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Clean up connections on exit
        try:
            ws_manager.close()
        except:
            pass