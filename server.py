from mcp.server.fastmcp import FastMCP, Image
from typing import List, Any, Optional, Union
from pathlib import Path
import json
from utils.websocket_manager import WebSocketManager
from msgs.geometry_msgs import Twist, PoseStamped
from msgs.sensor_msgs import Image as RosImage, JointState
import subprocess

#camera
import time
import os
from datetime import datetime, timedelta
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
import traceback
import re

LOCAL_IP = "192.168.56.1"  # Replace with your local IP address
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
        cmd = f"source /opt/ros/humble/setup.bash && source ~/Desktop/ros2_ws/install/setup.bash && timeout {timeout} ros2 topic echo {topic_name} --once"
        
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
    Perform ros2 topic echo --once /joint_states to get current joint angles.
    
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

def _run_primitive(script_name: str, args: str = "", timeout: int = 60, error_prefix: str = "Primitive execution") -> Dict[str, Any]:
    """
    Helper function to run primitive scripts from the primitives directory.
    Parses JSON output if present (scripts that output __RESULT_JSON__ markers).
    
    Args:
        script_name: Name of the script file (e.g., "control_gripper.py")
        args: Command line arguments to pass to the script
        timeout: Maximum execution time in seconds
        error_prefix: Prefix for error messages
        
    Returns:
        Dictionary with execution results, including parsed JSON if available
    """
    try:
        import subprocess
        import os
        import re
        
        # Path to primitives directory
        primitives_dir = "/home/aaugus11/Documents/ros-mcp-server/primitives"
        script_path = os.path.join(primitives_dir, script_name)
        
        # Check if script exists
        if not os.path.exists(script_path):
            return {
                "status": "error",
                "message": f"{error_prefix}: Script not found: {script_path}"
            }
        
        # Build command with ROS2 environment sourcing
        cmd = f"""
source /opt/ros/humble/setup.bash
source ~/Desktop/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
cd {primitives_dir}
/usr/bin/python3 {script_name} {args}
"""
        
        # Run with bash shell to source environment
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Try to parse JSON output if present (for scripts like control_gripper.py)
        json_result = None
        if result.stdout:
            # Look for JSON markers: __RESULT_JSON__ ... __END_RESULT_JSON__
            json_match = re.search(r'__RESULT_JSON__\s*\n(.*?)\n__END_RESULT_JSON__', result.stdout, re.DOTALL)
            if json_match:
                try:
                    json_result = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
        
        # Build response
        if result.returncode == 0:
            response = {
                "status": "success",
                "message": f"{error_prefix} completed successfully",
                "script_path": script_path
            }
            
            # Add parsed JSON result if available
            if json_result:
                response.update(json_result)
                # Map script's result format to our status format
                if "result" in json_result:
                    if json_result["result"] == "success":
                        response["status"] = "success"
                    elif json_result["result"] == "failure":
                        response["status"] = "error"
                        response["message"] = json_result.get("error", f"{error_prefix} failed")
            else:
                # No JSON output, include stdout/stderr
                if result.stdout:
                    response["output"] = result.stdout.strip()
                if result.stderr:
                    response["stderr"] = result.stderr.strip()
            
            return response
        else:
            # Script failed
            error_msg = f"{error_prefix} failed"
            if json_result and "error" in json_result:
                error_msg = json_result["error"]
            elif result.stderr:
                error_msg = result.stderr.strip()
            
            response = {
                "status": "error",
                "message": error_msg,
                "script_path": script_path,
                "return_code": result.returncode
            }
            
            # Include parsed JSON if available
            if json_result:
                response.update(json_result)
            else:
                if result.stdout:
                    response["output"] = result.stdout.strip()
                if result.stderr:
                    response["stderr"] = result.stderr.strip()
            
            return response
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": f"{error_prefix} timed out after {timeout} seconds",
            "script_name": script_name
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"{error_prefix} error: {str(e)}",
            "script_name": script_name,
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def control_gripper(command: str, mode: str = "sim") -> Dict[str, Any]:
    """Control gripper

    Supports "open", "close", "half-open" (30mm), or numeric values 0-110 (range of width in mm).

    Args:
        command: Gripper command - "open", "close", "half-open" (30mm), or numeric value 0-110 (width in mm)
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")

    """
    return _run_primitive("control_gripper.py", f"{command} --mode {mode}", timeout=60, error_prefix="Gripper control")

def _get_all_mcp_tools():
    """
    Dynamically discover all available MCP tools in the current module.
    Returns a dictionary of tool_name -> function mappings.
    """
    import sys
    import inspect
    
    current_module = sys.modules[__name__]
    available_tools = {}
    
    # Get all functions that could be MCP tools
    for name, obj in inspect.getmembers(current_module, inspect.isfunction):
        if (not name.startswith('_') and 
            callable(obj) and 
            obj.__module__ == current_module.__name__ and
            # Exclude the execution tools themselves to avoid recursion
            name not in ['execute_python_code', 'execute_code_with_server_access', 'start_background_task']):
            available_tools[name] = obj
    
    return available_tools

@mcp.tool()
def execute_python_code(code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute Python code dynamically with access to all server functions and libraries.
    Claude can write custom logic on-the-fly based on conversation context.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary with execution results including output, errors, and status
    """
    try:
        import tempfile
        import subprocess
        import sys
        import os
        
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Get all available tools dynamically
            available_tools = _get_all_mcp_tools()
            tool_imports = ", ".join(available_tools.keys())
            
            # Add imports and context that Claude might need
            code_with_context = f"""
import sys
import os
import traceback
sys.path.append('/home/aaugus11/Documents/ros-mcp-server')

# Set up ROS2 environment
os.environ['ROS_DOMAIN_ID'] = '0'
os.environ['ROS_VERSION'] = '2'
os.environ['ROS_DISTRO'] = 'humble'

# Source ROS2 setup in subprocess calls
def run_ros2_command(cmd_list, **kwargs):
    \"\"\"Run ROS2 commands with proper environment sourcing\"\"\"
    import subprocess
    if isinstance(cmd_list, str):
        cmd = f"source /opt/ros/humble/setup.bash && source ~/Desktop/ros2_ws/install/setup.bash && source ~/ros2/install/setup.bash && {{cmd_list}}"
        return subprocess.run(cmd, shell=True, executable='/bin/bash', **kwargs)
    else:
        cmd = f"source /opt/ros/humble/setup.bash && source ~/Desktop/ros2_ws/install/setup.bash && source ~/ros2/install/setup.bash && {{' '.join(cmd_list)}}"
        return subprocess.run(cmd, shell=True, executable='/bin/bash', **kwargs)

# Import all the existing functions Claude can use (dynamically discovered)
try:
    from server import {tool_imports}
    from msgs.geometry_msgs import Twist, PoseStamped
    from msgs.sensor_msgs import JointState, Image as RosImage
except ImportError:
    # Fallback imports if running standalone
    pass

import json
import time
import numpy as np
from datetime import datetime, timedelta
import threading
import subprocess
import re

# User's dynamic code starts here:
{code}
"""
            f.write(code_with_context)
            temp_file = f.name
        
        # Execute the code
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd='/home/aaugus11/Documents/ros-mcp-server'
        )
        
        # Clean up
        os.unlink(temp_file)
        
        if result.returncode == 0:
            return {
                "status": "success",
                "output": result.stdout,
                "stderr": result.stderr if result.stderr else None,
                "execution_time": f"Completed within {timeout}s timeout"
            }
        else:
            return {
                "status": "error",
                "output": result.stdout if result.stdout else None,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        # Clean up the temp file
        try:
            os.unlink(temp_file)
        except:
            pass
        return {
            "status": "timeout",
            "error": f"Code execution timed out after {timeout} seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def execute_code_with_server_access(code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute code with direct access to server's WebSocket manager and all tools.
    Allows Claude to write sophisticated real-time behaviors with full server context.
    
    Args:
        code: Python code to execute with server access
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary with execution results
    """
    try:
        import signal
        
        # Set up timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Code execution timed out after {timeout} seconds")
        
        # Get all available tools dynamically
        available_tools = _get_all_mcp_tools()
        
        # ROS2 helper function for proper environment sourcing
        def run_ros2_command(cmd_list, **kwargs):
            """Run ROS2 commands with proper environment sourcing"""
            import subprocess
            if isinstance(cmd_list, str):
                cmd = f"source /opt/ros/humble/setup.bash && source ~/Desktop/ros2_ws/install/setup.bash && {cmd_list}"
                return subprocess.run(cmd, shell=True, executable='/bin/bash', **kwargs)
            else:
                cmd = f"source /opt/ros/humble/setup.bash && source ~/Desktop/ros2_ws/install/setup.bash && {' '.join(cmd_list)}"
                return subprocess.run(cmd, shell=True, executable='/bin/bash', **kwargs)
        
        # Create execution context with access to server internals
        exec_globals = {
            # Server components
            'ws_manager': ws_manager,
            'mcp': mcp,
            
            # ROS2 helper
            'run_ros2_command': run_ros2_command,
            
            # Standard libraries Claude might need
            'time': time,
            'datetime': datetime,
            'timedelta': timedelta,
            'np': np,
            'json': json,
            'threading': threading,
            'os': os,
            'cv2': cv2,
            'subprocess': subprocess,
            're': re,
            
            # Message classes
            'Twist': Twist,
            'PoseStamped': PoseStamped,
            'JointState': JointState,
            'RosImage': RosImage,
            
            # Utility functions
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
        }
        
        # Add all discovered tools to execution context
        exec_globals.update(available_tools)
        
        exec_locals = {}
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            # Execute the code in this context
            exec(code, exec_globals, exec_locals)
            
            # Cancel timeout
            signal.alarm(0)
            
            # Capture any return values or variables created
            result_vars = {k: str(v) for k, v in exec_locals.items() 
                          if not k.startswith('_') and not callable(v)}
            
            return {
                "status": "success",
                "message": "Code executed successfully with server access",
                "variables_created": result_vars if result_vars else None,
                "available_tools": list(available_tools.keys())
            }
            
        except Exception as e:
            signal.alarm(0)
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to set up execution environment: {str(e)}",
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def start_background_task(code: str, task_name: str = "background_task") -> Dict[str, Any]:
    """
    Start a background task that runs independently of the main conversation.
    Perfect for real-time tracking, monitoring, or continuous operations.
    
    Args:
        code: Python code to run in background
        task_name: Name identifier for the task
        
    Returns:
        Dictionary with task startup status
    """
    try:
        import threading
        import time
        
        # Get all available tools dynamically
        available_tools = _get_all_mcp_tools()
        
        # ROS2 helper function for proper environment sourcing
        def run_ros2_command(cmd_list, **kwargs):
            """Run ROS2 commands with proper environment sourcing"""
            import subprocess
            if isinstance(cmd_list, str):
                cmd = f"source /opt/ros/humble/setup.bash && source ~/Desktop/ros2_ws/install/setup.bash && {cmd_list}"
                return subprocess.run(cmd, shell=True, executable='/bin/bash', **kwargs)
            else:
                cmd = f"source /opt/ros/humble/setup.bash && source ~/Desktop/ros2_ws/install/setup.bash && {' '.join(cmd_list)}"
                return subprocess.run(cmd, shell=True, executable='/bin/bash', **kwargs)
        
        # Create execution context similar to execute_code_with_server_access
        exec_globals = {
            'ws_manager': ws_manager,
            'run_ros2_command': run_ros2_command,
            'time': time,
            'datetime': datetime,
            'np': np,
            'json': json,
            'threading': threading,
            'subprocess': subprocess,
            're': re,
            'print': print,
            'task_name': task_name,
            
            # Message classes
            'Twist': Twist,
            'PoseStamped': PoseStamped,
            'JointState': JointState,
            'RosImage': RosImage,
        }
        
        # Add all discovered tools to execution context
        exec_globals.update(available_tools)
        
        def run_background_code():
            try:
                exec(code, exec_globals)
            except Exception as e:
                print(f"Background task '{task_name}' error: {e}")
                traceback.print_exc()
        
        # Start the background thread
        thread = threading.Thread(target=run_background_code, name=task_name, daemon=True)
        thread.start()
        
        return {
            "status": "success",
            "message": f"Background task '{task_name}' started successfully",
            "task_name": task_name,
            "thread_id": thread.ident,
            "available_tools": list(available_tools.keys())
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to start background task: {str(e)}",
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def start_topic_stream(topic_name: str, message_type: str = "geometry_msgs/PoseStamped"):
    """
    Start a continuous WebSocket stream for a ROS topic.
    Returns a stream ID that can be used to read from the stream.
    
    Args:
        topic_name: ROS topic to stream (e.g., "/object_poses/jenga_2")
        message_type: ROS message type (e.g., "geometry_msgs/PoseStamped", "geometry_msgs/PoseArray")
        
    Returns:
        Dictionary with stream status and ID
    """
    try:
        # Create subscription message
        subscribe_msg = {
            "op": "subscribe",
            "topic": topic_name,
            "type": message_type
        }
        
        # Send subscription via WebSocket
        ws_manager.send(subscribe_msg)
        
        return {
            "status": "success",
            "message": f"Started streaming {topic_name}",
            "topic": topic_name,
            "message_type": message_type,
            "stream_id": f"stream_{topic_name.replace('/', '_')}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "topic": topic_name,
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def read_from_stream(topic_name: str, timeout: int = 1):
    """
    Read the latest message from a streaming topic.
    Much faster than ros2 topic echo since it uses persistent WebSocket connection.
    
    Args:
        topic_name: ROS topic to read from (must be already streaming)
        timeout: Timeout in seconds
        
    Returns:
        Parsed message data or error information
    """
    try:
        import re
        import math
        
        # Read from WebSocket stream
        raw_data = ws_manager.receive_binary()
        
        if not raw_data:
            return {
                "status": "no_data",
                "topic": topic_name,
                "message": "No data received from stream"
            }
        
        # Parse WebSocket message
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode("utf-8")
            
        msg_data = json.loads(raw_data)
        
        # Extract the ROS message
        if "msg" in msg_data:
            ros_msg = msg_data["msg"]
        else:
            ros_msg = msg_data
            
        parsed_result = {
            "status": "success",
            "topic": topic_name,
            "timestamp": datetime.now().isoformat(),
            "raw_message": ros_msg
        }
        
        # Parse common message structures
        
        # Handle PoseStamped messages
        if "pose" in ros_msg:
            pose = ros_msg["pose"]
            
            # Extract position
            if "position" in pose:
                parsed_result["position"] = pose["position"]
                
            # Extract orientation and convert to RPY
            if "orientation" in pose:
                quat = pose["orientation"]
                parsed_result["orientation_quaternion"] = quat
                
                # Convert to RPY
                qx, qy, qz, qw = quat["x"], quat["y"], quat["z"], quat["w"]
                
                # Quaternion to Euler conversion
                sinr_cosp = 2 * (qw * qx + qy * qz)
                cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
                roll = math.atan2(sinr_cosp, cosr_cosp)
                
                sinp = 2 * (qw * qy - qz * qx)
                if abs(sinp) >= 1:
                    pitch = math.copysign(math.pi / 2, sinp)
                else:
                    pitch = math.asin(sinp)
                
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                
                parsed_result["orientation_rpy_degrees"] = {
                    "roll": math.degrees(roll),
                    "pitch": math.degrees(pitch),
                    "yaw": math.degrees(yaw)
                }
                
        # Handle PoseArray messages
        elif "poses" in ros_msg:
            poses = ros_msg["poses"]
            parsed_result["poses"] = []
            
            for i, pose in enumerate(poses):
                pose_data = {"index": i}
                
                if "position" in pose:
                    pose_data["position"] = pose["position"]
                    
                if "orientation" in pose:
                    quat = pose["orientation"]
                    pose_data["orientation_quaternion"] = quat
                    
                    # Convert to RPY
                    qx, qy, qz, qw = quat["x"], quat["y"], quat["z"], quat["w"]
                    
                    sinr_cosp = 2 * (qw * qx + qy * qz)
                    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
                    roll = math.atan2(sinr_cosp, cosr_cosp)
                    
                    sinp = 2 * (qw * qy - qz * qx)
                    if abs(sinp) >= 1:
                        pitch = math.copysign(math.pi / 2, sinp)
                    else:
                        pitch = math.asin(sinp)
                    
                    siny_cosp = 2 * (qw * qz + qx * qy)
                    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                    yaw = math.atan2(siny_cosp, cosy_cosp)
                    
                    pose_data["orientation_rpy_degrees"] = {
                        "roll": math.degrees(roll),
                        "pitch": math.degrees(pitch),
                        "yaw": math.degrees(yaw)
                    }
                
                parsed_result["poses"].append(pose_data)
                
            parsed_result["pose_count"] = len(poses)
        
        return parsed_result
        
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "topic": topic_name,
            "error": f"JSON decode error: {str(e)}",
            "raw_data": raw_data if 'raw_data' in locals() else None
        }
    except Exception as e:
        return {
            "status": "error", 
            "topic": topic_name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def stop_topic_stream(topic_name: str):
    """
    Stop streaming a ROS topic.
    
    Args:
        topic_name: ROS topic to stop streaming
        
    Returns:
        Dictionary with stop status
    """
    try:
        # Send unsubscribe message
        unsubscribe_msg = {
            "op": "unsubscribe",
            "topic": topic_name
        }
        
        ws_manager.send(unsubscribe_msg)
        
        return {
            "status": "success",
            "message": f"Stopped streaming {topic_name}",
            "topic": topic_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "topic": topic_name,
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def fast_topic_read(topic_name: str, timeout: int = 2):
    """
    Fast ROS2 topic reading with proper environment sourcing.
    Optimized for Claude's dynamic code execution to avoid environment issues.
    
    Args:
        topic_name: ROS topic to read from
        timeout: Timeout in seconds
        
    Returns:
        Parsed topic data or error information
    """
    try:
        import subprocess
        import re
        
        # Use properly sourced environment
        cmd = f"source /opt/ros/humble/setup.bash && source ~/Desktop/ros2_ws/install/setup.bash && source ~/ros2/install/setup.bash && timeout {timeout} ros2 topic echo {topic_name} --once"
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=timeout + 2
        )
        
        if result.returncode == 0:
            # Parse common message types automatically
            message_data = result.stdout.strip()
            
            # Try to parse position data
            x_match = re.search(r'position:\s*\n\s*x: ([\d\.-]+)', message_data)
            y_match = re.search(r'y: ([\d\.-]+)', message_data)
            z_match = re.search(r'z: ([\d\.-]+)', message_data)
            
            # Try to parse orientation data (quaternion)
            orientation_match = re.search(
                r'orientation:\s*\n\s*x: ([\d\.-]+)\s*\n\s*y: ([\d\.-]+)\s*\n\s*z: ([\d\.-]+)\s*\n\s*w: ([\d\.-]+)', 
                message_data
            )
            
            parsed_data = {
                "status": "success",
                "topic": topic_name,
                "raw_message": message_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add parsed position if found
            if all([x_match, y_match, z_match]):
                parsed_data["position"] = {
                    "x": float(x_match.group(1)),
                    "y": float(y_match.group(1)),
                    "z": float(z_match.group(1))
                }
            
            # Add parsed orientation if found
            if orientation_match:
                parsed_data["orientation_quaternion"] = {
                    "x": float(orientation_match.group(1)),
                    "y": float(orientation_match.group(2)),
                    "z": float(orientation_match.group(3)),
                    "w": float(orientation_match.group(4))
                }
                
                # Convert to RPY for convenience
                import math
                qx, qy, qz, qw = [float(orientation_match.group(i)) for i in range(1, 5)]
                
                # Convert quaternion to Euler angles
                sinr_cosp = 2 * (qw * qx + qy * qz)
                cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
                roll = math.atan2(sinr_cosp, cosr_cosp)
                
                sinp = 2 * (qw * qy - qz * qx)
                if abs(sinp) >= 1:
                    pitch = math.copysign(math.pi / 2, sinp)
                else:
                    pitch = math.asin(sinp)
                
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                
                parsed_data["orientation_rpy_degrees"] = {
                    "roll": math.degrees(roll),
                    "pitch": math.degrees(pitch),
                    "yaw": math.degrees(yaw)
                }
            
            return parsed_data
            
        else:
            return {
                "status": "error",
                "topic": topic_name,
                "error": f"Failed to read topic: {result.stderr}",
                "return_code": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "topic": topic_name,
            "error": f"Topic read timed out after {timeout} seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "topic": topic_name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def pub_pose_stamped(position: List[float], orientation: List[float], topic: str = "/target_pose", frame_id: str = "base_link"):
    """
    Publish a PoseStamped message to a ROS topic.
    
    Args:
        position: [x, y, z] position in meters
        orientation: [x, y, z, w] quaternion OR [roll, pitch, yaw] in degrees  
        topic: ROS topic to publish to
        frame_id: Reference frame (default: "base_link")
        
    Returns:
        Dictionary with publish status
    """
    try:
        pose_stamped = PoseStamped(ws_manager, topic=topic)
        msg = pose_stamped.publish(position, orientation, frame_id)
        
        return {
            "status": "success",
            "message": f"PoseStamped published to {topic}",
            "topic": topic,
            "position": position,
            "orientation": orientation,
            "frame_id": frame_id
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "topic": topic,
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def sub_pose_stamped(topic: str = "/object_pose"):
    """
    Subscribe to a PoseStamped topic and get parsed pose data.
    
    Args:
        topic: ROS topic to subscribe to
        
    Returns:
        Dictionary with parsed pose data including position, quaternion, and RPY
    """
    try:
        pose_stamped = PoseStamped(ws_manager, topic=topic)
        msg = pose_stamped.subscribe()
        
        if msg is not None:
            return {
                "status": "success",
                "data": msg
            }
        else:
            return {
                "status": "no_data",
                "message": f"No PoseStamped data received from {topic}"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "topic": topic,
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def create_topic_listener(topic_name: str, message_type: str = "geometry_msgs/PoseStamped", 
                         callback_code: str = "", frequency_hz: float = 10.0):
    """
    Create a continuous topic listener that executes custom code for each message.
    Perfect for real-time tracking and reactive behaviors.
    
    Args:
        topic_name: ROS topic to listen to (e.g., "/object_poses/jenga_2")
        message_type: Type of ROS message ("geometry_msgs/PoseStamped", "geometry_msgs/PoseArray", etc.)
        callback_code: Python code to execute for each received message (has access to 'msg_data')
        frequency_hz: Listening frequency in Hz
        
    Returns:
        Dictionary with listener status and control info
    """
    try:
        import threading
        import time
        
        # Create the appropriate message class
        if message_type == "geometry_msgs/PoseStamped":
            subscriber = PoseStamped(ws_manager, topic=topic_name)
        elif message_type == "geometry_msgs/PoseArray":
            # Could add PoseArray class here
            return {"status": "error", "error": "PoseArray not yet implemented"}
        else:
            return {"status": "error", "error": f"Unsupported message type: {message_type}"}
        
        listener_active = True
        
        def listener_thread():
            """Background thread that continuously listens to the topic"""
            nonlocal listener_active
            
            # Get all available tools for the callback code
            available_tools = _get_all_mcp_tools()
            
            # Create execution context for callback code
            exec_globals = {
                'ws_manager': ws_manager,
                'time': time,
                'datetime': datetime,
                'np': np,
                'json': json,
                'print': print,
                'subscriber': subscriber,
                # Add all available tools
                **available_tools
            }
            
            while listener_active:
                try:
                    # Get message from topic
                    msg_data = subscriber.subscribe()
                    
                    if msg_data:
                        # Make message data available to callback code
                        exec_globals['msg_data'] = msg_data
                        
                        # Execute user's callback code
                        if callback_code.strip():
                            try:
                                exec(callback_code, exec_globals)
                            except Exception as e:
                                print(f"[TopicListener] Callback error: {e}")
                        else:
                            # Default behavior - just print the data
                            if "position" in msg_data:
                                pos = msg_data["position"]
                                print(f"[{topic_name}] Position: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
                            
                            if "orientation_rpy_degrees" in msg_data:
                                rpy = msg_data["orientation_rpy_degrees"]
                                print(f"[{topic_name}] RPY: ({rpy['roll']:.1f}°, {rpy['pitch']:.1f}°, {rpy['yaw']:.1f}°)")
                    
                    # Control frequency
                    time.sleep(1.0 / frequency_hz)
                    
                except Exception as e:
                    print(f"[TopicListener] Error: {e}")
                    time.sleep(0.1)
        
        # Start the listener thread
        thread = threading.Thread(target=listener_thread, name=f"listener_{topic_name.replace('/', '_')}", daemon=True)
        thread.start()
        
        return {
            "status": "success",
            "message": f"Topic listener started for {topic_name}",
            "topic": topic_name,
            "message_type": message_type,
            "frequency_hz": frequency_hz,
            "thread_id": thread.ident,
            "listener_id": f"listener_{topic_name.replace('/', '_')}"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def test_ros2_environment():
    """
    Test if ROS2 environment is properly set up in dynamic code execution.
    This will verify that ros2 commands work in the execution context.
    """
    try:
        # Test using execute_code_with_server_access
        test_code = '''
# Test ROS2 environment
result = run_ros2_command("ros2 topic list", capture_output=True, text=True, timeout=5)

if result.returncode == 0:
    topics = result.stdout.strip().split('\\n')
    print(f"✅ ROS2 environment working! Found {len(topics)} topics:")
    for topic in topics[:5]:  # Show first 5 topics
        print(f"  - {topic}")
    if len(topics) > 5:
        print(f"  ... and {len(topics) - 5} more")
else:
    print(f"❌ ROS2 environment issue: {result.stderr}")
    
# Test publishing a simple message
pub_result = run_ros2_command(
    "ros2 topic pub /test_topic std_msgs/String \\"data: 'Environment test from MCP server'\\" --once",
    capture_output=True, text=True, timeout=10
)

if pub_result.returncode == 0:
    print("✅ Successfully published test message to /test_topic")
else:
    print(f"❌ Failed to publish: {pub_result.stderr}")
'''
        
        # Get all available tools dynamically
        available_tools = _get_all_mcp_tools()
        
        # ROS2 helper function
        def run_ros2_command(cmd_list, **kwargs):
            import subprocess
            if isinstance(cmd_list, str):
                cmd = f"source /opt/ros/humble/setup.bash && source ~/Desktop/ros2_ws/install/setup.bash && {cmd_list}"
                return subprocess.run(cmd, shell=True, executable='/bin/bash', **kwargs)
            else:
                cmd = f"source /opt/ros/humble/setup.bash && source ~/Desktop/ros2_ws/install/setup.bash && {' '.join(cmd_list)}"
                return subprocess.run(cmd, shell=True, executable='/bin/bash', **kwargs)
        
        # Create execution context
        exec_globals = {
            'ws_manager': ws_manager,
            'run_ros2_command': run_ros2_command,
            'time': time,
            'datetime': datetime,
            'timedelta': timedelta,
            'print': print,
            **available_tools
        }
        
        # Execute the test code
        exec(test_code, exec_globals)
        
        return {
            "status": "success",
            "message": "ROS2 environment test completed - check output above"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def run_primitive_script(script_name: str, timeout: int = 30):
    """
    Run a primitive script from the ur_asu package with proper ROS2 environment.
    
    Args:
        script_name: Name of the script (e.g., "movea2b.py", "move_home.py", "pick_place.py")
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary with execution results
    """
    try:
        import subprocess
        import os
        
        # Path to primitive scripts
        primitives_dir = "/home/aaugus11/Desktop/ros2_ws/src/ur_asu-main/ur_asu/scripts/primitives"
        script_path = os.path.join(primitives_dir, script_name)
        
        # Check if script exists
        if not os.path.exists(script_path):
            return {
                "status": "error",
                "error": f"Script not found: {script_path}",
                "available_scripts": [f for f in os.listdir(primitives_dir) if f.endswith('.py')] if os.path.exists(primitives_dir) else []
            }
        
        # Source ROS2 environment and run script
        cmd = f"""
source /opt/ros/humble/setup.bash
source ~/Desktop/ros2_ws/install/setup.bash
source ~/ros2/install/setup.bash
export ROS_DOMAIN_ID=0
cd {primitives_dir}
/usr/bin/python3 {script_name}
"""
        
        # Run with bash shell to source environment
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": f"Primitive script '{script_name}' executed successfully",
                "script_path": script_path,
                "output": result.stdout,
                "stderr": result.stderr if result.stderr else None
            }
        else:
            return {
                "status": "error",
                "message": f"Script execution failed with return code {result.returncode}",
                "script_path": script_path,
                "output": result.stdout if result.stdout else None,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "message": f"Script execution timed out after {timeout} seconds",
            "script_name": script_name
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "script_name": script_name,
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def list_primitive_scripts():
    """
    List all available primitive scripts in the ur_asu package.
    
    Returns:
        Dictionary with list of available scripts
    """
    try:
        import os
        
        primitives_dir = "/home/aaugus11/Desktop/ros2_ws/src/ur_asu-main/ur_asu/scripts/primitives"
        
        if not os.path.exists(primitives_dir):
            return {
                "status": "error",
                "error": f"Primitives directory not found: {primitives_dir}"
            }
        
        scripts = [f for f in os.listdir(primitives_dir) if f.endswith('.py')]
        
        return {
            "status": "success",
            "primitives_directory": primitives_dir,
            "available_scripts": scripts,
            "script_count": len(scripts)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def move_down(height: float = 0.148, position: list = None, orientation: list = None, quaternion: list = None):
    """
    Execute the move_down.py primitive script.
    This tool performs a two-step movement:
    1. First moves to target X,Y position with specified orientation
    2. Then moves down in Z-axis with force monitoring
    
    The robot will stop when either the force threshold is reached OR the target height is reached, whichever comes first.
    
    Args:
        height: Target Z position in meters (default: 0.148)
        position: Target position [x, y, z] in meters (optional, uses current position if not provided)
        orientation: Target orientation [roll, pitch, yaw] in degrees (optional, uses [0, 180, 0] if not provided)
        quaternion: Target orientation [x, y, z, w] quaternion (optional, overrides orientation if provided)
    
    Returns:
        Dictionary with execution status and results
    """
    try:
        import subprocess
        import os
        
        # Path to the move_down.py script
        script_path = "/home/aaugus11/Documents/ros-mcp-server/primitives/move_down.py"
        
        # Check if script exists
        if not os.path.exists(script_path):
            return {
                "status": "error",
                "error": f"Move down script not found: {script_path}"
            }
        
        # Build command with optional parameters
        cmd_parts = [
            "source /opt/ros/humble/setup.bash",
            "source ~/Desktop/ros2_ws/install/setup.bash", 
            "export ROS_DOMAIN_ID=0",
            "cd /home/aaugus11/Documents/ros-mcp-server/primitives",
            f"/usr/bin/python3 move_down.py --height {height}"
        ]
        
        # Build the python command with all parameters
        python_cmd = f"/usr/bin/python3 move_down.py --height {height}"
        
        # Add position parameter if provided
        if position and len(position) == 3:
            python_cmd += f" --position {' '.join(map(str, position))}"
        
        # Add orientation parameter if provided  
        if orientation and len(orientation) == 3:
            python_cmd += f" --orientation {' '.join(map(str, orientation))}"
        
        # Add quaternion parameter if provided (overrides orientation)
        if quaternion and len(quaternion) == 4:
            python_cmd += f" --quaternion {' '.join(map(str, quaternion))}"
        
        # Build final command
        cmd_parts = [
            "source /opt/ros/humble/setup.bash",
            "source ~/Desktop/ros2_ws/install/setup.bash", 
            "export ROS_DOMAIN_ID=0",
            "cd /home/aaugus11/Documents/ros-mcp-server/primitives",
            python_cmd
        ]
        
        cmd = "\n".join(cmd_parts)
        
        # Run with bash shell to source environment
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=30  # Reduced timeout to 30 seconds for faster feedback
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Move down operation completed successfully",
                "script_path": script_path,
                "output": result.stdout,
                "stderr": result.stderr if result.stderr else None
            }
        else:
            return {
                "status": "error",
                "message": f"Move down operation failed with return code {result.returncode}",
                "script_path": script_path,
                "output": result.stdout if result.stdout else None,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "message": "Move down operation timed out after 30 seconds. The script may be waiting for ROS2 services or hanging.",
            "script_name": "move_down.py",
            "debug_info": "Check if ROS2 services are running: ros2 topic list, ros2 service list"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "script_name": "move_down.py",
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def visual_servo_pick(topic_name: str = "/object_poses/jenga_4", hover_height: float = 0.15, duration: int = 30):
    """
    Execute visual servo pick alignment for Jenga blocks.
    This tool uses visual servoing to align the robot end-effector above a Jenga block for picking.
    
    Args:
        topic_name: ROS topic for Jenga pose (e.g., "/object_poses/jenga_4")
        hover_height: Height above the block to hover (meters, default: 0.15)
        duration: How long to run the visual servo in seconds (default: 30)
    
    Returns:
        Dictionary with execution status and results
    """
    try:
        import subprocess
        import os
        import time
        
        # Path to the visual_servo.py script
        script_path = "/home/aaugus11/Documents/ros-mcp-server/primitives/visual_servo.py"
        
        # Check if script exists
        if not os.path.exists(script_path):
            return {
                "status": "error",
                "error": f"Visual servo script not found: {script_path}"
            }
        
        # Source ROS2 environment and run script with timeout
        cmd = f"""
source /opt/ros/humble/setup.bash
source ~/Desktop/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
cd /home/aaugus11/Documents/ros-mcp-server/primitives
timeout {duration + 5} /usr/bin/python3 visual_servo.py --topic {topic_name} --height {hover_height} --duration {duration}
"""
        
        # Run with bash shell to source environment
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=duration + 10  # Add buffer for startup/shutdown
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": f"Visual servo pick completed successfully for {duration} seconds",
                "topic_name": topic_name,
                "hover_height": hover_height,
                "duration": duration,
                "output": result.stdout,
                "stderr": result.stderr if result.stderr else None
            }
        else:
            return {
                "status": "error",
                "message": f"Visual servo pick failed with return code {result.returncode}",
                "topic_name": topic_name,
                "hover_height": hover_height,
                "duration": duration,
                "output": result.stdout if result.stdout else None,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "message": f"Visual servo pick timed out after {duration + 10} seconds",
            "topic_name": topic_name,
            "hover_height": hover_height,
            "duration": duration
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "topic_name": topic_name,
            "hover_height": hover_height,
            "duration": duration,
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def visual_servo_yoloe(topic_name: str = "/objects_poses", object_name: str = "blue_dot_0", hover_height: float = 0.15, duration: int = 30, movement_duration: float = 7.0, target_xyz: Optional[List[float]] = None, target_xyzw: Optional[List[float]] = None):
    """
    Execute visual servo YOLO for object detection and robot movement.
    This tool uses YOLO-based visual servoing to move the robot to a specific object.
    
    IMPORTANT: When providing target_xyz, you MUST also provide target_xyzw for proper orientation.
    Both position and orientation are required together for accurate robot movement.
    
    Args:
        topic_name: ROS topic for object poses (default: "/objects_poses")
        object_name: Name of the object to move to (default: "blue_dot_0")
        hover_height: Height above the object to hover in meters (default: 0.15)
        duration: Maximum duration in seconds (default: 30)
        movement_duration: Duration for the movement in seconds (default: 7.0)
        target_xyz: Target position [x, y, z] in meters - REQUIRED with target_xyzw (default: None)
        target_xyzw: Target orientation [x, y, z, w] quaternion - REQUIRED with target_xyz (default: None)
    
    Usage:
        - For object detection: Don't provide target_xyz or target_xyzw
        - For manual targeting: Provide BOTH target_xyz AND target_xyzw together
    
    Returns:
        Dictionary with execution status and results
    """
    try:
        import subprocess
        import os
        
        # Path to the visual_servo_yoloe.py script
        script_path = "/home/aaugus11/Documents/ros-mcp-server/primitives/visual_servo_yoloe.py"
        
        # Check if script exists
        if not os.path.exists(script_path):
            return {
                "status": "error",
                "error": f"Visual servo YOLO script not found: {script_path}"
            }
        
        # Validate that both target_xyz and target_xyzw are provided together
        if (target_xyz is not None and target_xyzw is None) or (target_xyz is None and target_xyzw is not None):
            return {
                "status": "error",
                "error": "Both target_xyz and target_xyzw must be provided together for manual targeting. Use either both parameters or neither (for object detection mode).",
                "target_xyz_provided": target_xyz is not None,
                "target_xyzw_provided": target_xyzw is not None
            }
        
        # Build command with optional parameters
        cmd_parts = [
            "source /opt/ros/humble/setup.bash",
            "source ~/Desktop/ros2_ws/install/setup.bash",
            "export ROS_DOMAIN_ID=0",
            "cd /home/aaugus11/Documents/ros-mcp-server/primitives",
            f"timeout {duration + 5} /usr/bin/python3 visual_servo_yoloe.py --topic {topic_name} --object-name \"{object_name}\" --height {hover_height} --duration {duration} --movement-duration {movement_duration}"
        ]
        
        # Add optional target position if provided
        if target_xyz is not None and len(target_xyz) == 3:
            cmd_parts[-1] += f" --target-xyz {' '.join(map(str, target_xyz))}"
        
        # Add optional target orientation if provided
        if target_xyzw is not None and len(target_xyzw) == 4:
            cmd_parts[-1] += f" --target-xyzw {' '.join(map(str, target_xyzw))}"
        
        cmd = "\n".join(cmd_parts)
        
        # Run with bash shell to source environment
        # subprocess.run() is SYNCHRONOUS - it waits for completion before returning
        # This ensures the MCP tool doesn't proceed until visual servo is done
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=duration + 10  # Add buffer for startup/shutdown
        )
        
        if result.returncode == 0:
            # Check if the script actually completed successfully by looking for completion message
            output = result.stdout.strip() if result.stdout else ""
            success_indicators = ["Direct movement completed", "Trajectory sent successfully", "Exiting"]
            is_completed = any(indicator in output for indicator in success_indicators)
            
            return {
                "status": "success",
                "message": "Visual servo YOLO completed successfully",
                "topic_name": topic_name,
                "object_name": object_name,
                "hover_height": hover_height,
                "duration": duration,
                "movement_duration": movement_duration,
                "target_xyz": target_xyz,
                "target_xyzw": target_xyzw,
                "output": output,
                "movement_completed": is_completed
            }
        else:
            return {
                "status": "error",
                "message": "Visual servo YOLO failed",
                "error": result.stderr.strip() if result.stderr else "Unknown error",
                "output": result.stdout.strip() if result.stdout else None,
                "topic_name": topic_name,
                "object_name": object_name,
                "hover_height": hover_height,
                "duration": duration,
                "movement_duration": movement_duration,
                "target_xyz": target_xyz,
                "target_xyzw": target_xyzw,
                "return_code": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "message": f"Visual servo YOLO timed out after {duration + 10} seconds",
            "topic_name": topic_name,
            "object_name": object_name,
            "hover_height": hover_height,
            "duration": duration,
            "movement_duration": movement_duration,
            "target_xyz": target_xyz,
            "target_xyzw": target_xyzw
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "topic_name": topic_name,
            "object_name": object_name,
            "hover_height": hover_height,
            "duration": duration,
            "movement_duration": movement_duration,
            "target_xyz": target_xyz,
            "target_xyzw": target_xyzw,
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def run_prompt_free_detection():
    """
    Run the prompt-free YOLOE detection test and return both original and annotated images.
    This tool executes the prompt_free_test.py script and returns the images it creates.
    
    Returns:
        Dictionary with detection results, original image, and annotated image
    """
    try:
        import subprocess
        import os
        import glob
        from datetime import datetime
        
        # Path to the prompt free test script
        script_path = "/home/aaugus11/Documents/ros-mcp-server/yoloe/prompt_free_test.py"
        screenshots_dir = "/home/aaugus11/Documents/ros-mcp-server/yoloe/screenshots"
        
        # Check if script exists
        if not os.path.exists(script_path):
            return {
                "status": "error",
                "error": f"Prompt free test script not found: {script_path}"
            }
        
        # Run the script with proper environment setup
        cmd = [
            "bash", "-c",
            "source /opt/ros/humble/setup.bash && "
            "source ~/Desktop/ros2_ws/install/setup.bash && "
            "export ROS_DOMAIN_ID=0 && "
            f"cd /home/aaugus11/Documents/ros-mcp-server/yoloe && "
            "python3 prompt_free_test.py"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # Increased timeout since the script now works properly
        )
        
        # Find the images that were created
        annotated_files = glob.glob(os.path.join(screenshots_dir, "annotated_photo_*.jpg"))
        original_files = glob.glob(os.path.join(screenshots_dir, "original_photo_*.jpg"))
        
        if not annotated_files:
            return {
                "status": "error",
                "message": "No annotated images found after running script",
                "script_output": result.stdout if result.stdout else None,
                "script_stderr": result.stderr if result.stderr else None
            }
        
        # Get the most recent files
        latest_annotated = max(annotated_files, key=os.path.getctime)
        latest_original = max(original_files, key=os.path.getctime) if original_files else None
        
        # Read the images
        try:
            # Read annotated image
            with open(latest_annotated, 'rb') as f:
                annotated_data = f.read()
            annotated_mcp_image = Image(data=annotated_data, format="jpeg")
            
            # Read original image if it exists
            if latest_original and os.path.exists(latest_original):
                with open(latest_original, 'rb') as f:
                    original_data = f.read()
                original_mcp_image = Image(data=original_data, format="jpeg")
            else:
                # Use annotated image as fallback
                original_mcp_image = annotated_mcp_image
                latest_original = latest_annotated
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to read images: {str(e)}",
                "annotated_image": latest_annotated,
                "original_image": latest_original
            }
        
        # Parse detection results from script output
        detected_objects = []
        if result and result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if '✅' in line and 'conf:' in line:
                    try:
                        parts = line.split('✅')[1].strip()
                        if '(' in parts and 'conf:' in parts:
                            class_part = parts.split('(')[0].strip()
                            conf_part = parts.split('conf:')[1].split(')')[0].strip()
                            detected_objects.append({
                                "class_name": class_part,
                                "confidence": float(conf_part)
                            })
                    except (ValueError, IndexError):
                        continue
        
        # If no objects parsed, provide generic info
        if not detected_objects:
            detected_objects = [{
                "class_name": "detected_objects",
                "confidence": 0.8,
                "note": "Objects detected by YOLOE prompt-free model (see annotated image for details)"
            }]
        
        return {
            "status": "success",
            "message": "Prompt-free detection completed successfully",
            "annotated_image": latest_annotated,
            "original_image": latest_original,
            "detected_objects": detected_objects,
            "object_count": len(detected_objects),
            "script_output": result.stdout if result and result.stdout else None,
            "script_stderr": result.stderr if result and result.stderr else None,
            "timestamp": datetime.now().isoformat()
        }, original_mcp_image, annotated_mcp_image
        
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "message": "Script execution timed out after 60 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def update_yolo_prompts(color_map: dict):
    """
    Update YOLO detection prompts using the UpdateYoloPrompts ROS2 service.
    This CORRECTS mislabeled objects from the prompt-free YOLO detector.
    
    ⚠️ CRITICAL: The color_map maps FROM what YOLO wrongly detected TO what it actually is!
    The prompts are automatically derived from the color_map keys.
    
    Args:
        color_map: Dictionary that CORRECTS wrong labels
                   ⚠️ KEY = What YOLO WRONGLY detected (look at annotated image labels)
                   ⚠️ VALUE = What the object ACTUALLY is (what you want it called)
                   
                   Format: {
                       "YOLO_WRONG_LABEL": "actual_object_name",
                       "YOLO_WRONG_LABEL": "actual_object_name"
                   }
                   
                   ⚠️ CRITICAL FOR COLORED BLOCKS: When mapping colored blocks, you MUST use the pattern:
                   "{color} object" → "{color} block" (e.g., "red object" → "red block", "blue object" → "blue block")
                   Do NOT use the wrong YOLO label directly. Extract the color and use "{color} object" as the key.
    
    Returns:
        Dictionary with service call results
        
    Examples:
        # Example 1: Correct format for colored blocks
        # YOLO detected "lip balm" but it's actually a red block
        # You MUST use "red object" → "red block" (not "lip balm" → "red block")
        update_yolo_prompts({
            "document": "white box",
            "red object": "red block",      # ✅ CORRECT: Use "{color} object" pattern
            "blue object": "blue block"     # ✅ CORRECT: Use "{color} object" pattern
        })
        
        # Example 2: WRONG - Don't use the wrong YOLO label for colored blocks
        # ❌ WRONG: {"lip balm": "red block", "bookmark": "blue block"}
        # ✅ CORRECT: {"red object": "red block", "blue object": "blue block"}
        
        # Example 3: Mixed mappings (colored blocks vs other objects)
        update_yolo_prompts({
            "document": "white box",         # ✅ OK: Not a colored block, use YOLO label
            "cork": "jenga block",           # ✅ OK: Not a colored block, use YOLO label
            "red object": "red block",       # ✅ CORRECT: Colored block, use "{color} object"
            "blue object": "blue block"     # ✅ CORRECT: Colored block, use "{color} object"
        })
        
        # Step-by-step for VLM:
        # 1. Look at ANNOTATED image - read the wrong labels YOLO put on each object
        # 2. Look at ORIGINAL image - identify what each object actually is
        # 3. For colored blocks (red block, blue block, green block, etc.):
        #    - Extract the color from the actual object name
        #    - Use "{color} object" as the KEY (e.g., "red object", "blue object")
        #    - Use "{color} block" as the VALUE (e.g., "red block", "blue block")
        # 4. For non-colored objects, use the YOLO wrong label directly
    """
    try:
        import subprocess
        import os
        
        # Derive prompts from color_map keys
        prompts = list(color_map.keys())
        
        # Path to the update service script
        script_path = "/home/aaugus11/Documents/ros-mcp-server/yoloe/update_yolo_prompts_service.py"
        
        # Check if script exists
        if not os.path.exists(script_path):
            return {
                "status": "error",
                "error": f"Update service script not found: {script_path}"
            }
        
        # Build the command arguments
        cmd_parts = [
            "bash", "-c",
            "source /opt/ros/humble/setup.bash && "
            "source ~/Desktop/ros2_ws/install/setup.bash && "
            "export ROS_DOMAIN_ID=0 && "
            f"cd /home/aaugus11/Documents/ros-mcp-server/yoloe && "
            f"python3 update_yolo_prompts_service.py"
        ]
        
        # Add prompts to the command (derived from color_map keys)
        for prompt in prompts:
            cmd_parts[2] += f" '{prompt}'"
        
        # Add color map
        cmd_parts[2] += " --color-map"
        for prompt, color in color_map.items():
            cmd_parts[2] += f" '{prompt}:{color}'"
        
        # Execute the service call
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "YOLO prompts updated successfully",
                "prompts": prompts,
                "color_map": color_map,
                "service_output": result.stdout.strip() if result.stdout else None
            }
        else:
            return {
                "status": "error",
                "message": "Failed to update YOLO prompts",
                "error": result.stderr.strip() if result.stderr else "Unknown error",
                "service_output": result.stdout.strip() if result.stdout else None,
                "return_code": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "message": "Service call timed out after 10 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def move_to_safe_height(safe_height: float = 0.481) -> Dict[str, Any]:
    """
    Move robot to safe height while maintaining current position.
    Reads current end-effector pose and moves to same x,y coordinates with specified z height.
    Uses fixed orientation [0, 180, 0] for consistent downward-pointing end-effector.
    
    Args:
        safe_height: Target Z height in meters (default: 0.481)
        
    Returns:
        Dictionary with execution status and results
    """
    try:
        import subprocess
        import yaml
        import re
        import sys
        import os
        
        # Add custom libraries to Python path
        custom_lib_path = "/home/aaugus11/Desktop/ros2_ws/src/ur_asu-main/ur_asu/custom_libraries"
        if custom_lib_path not in sys.path:
            sys.path.append(custom_lib_path)
        
        from ik_solver import compute_ik
        
        def quaternion_to_rpy(x, y, z, w):
            """Convert quaternion to roll, pitch, yaw in degrees"""
            import math
            
            # Roll
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
            
            # Pitch
            sinp = 2 * (w * y - z * x)
            if abs(sinp) >= 1:
                pitch = math.degrees(math.copysign(math.pi / 2, sinp))
            else:
                pitch = math.degrees(math.asin(sinp))
            
            # Yaw
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
            
            return [roll, pitch, yaw]
        
        # Read current end-effector pose (same method as move_down.py)
        result = subprocess.run([
            'bash', '-c', 
            'source /opt/ros/humble/setup.bash && ros2 topic echo /tcp_pose_broadcaster/pose --once'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"Failed to read pose topic: {result.stderr}"
            }
        
        # Clean the output to remove special characters (same as standalone script)
        cleaned_output = result.stdout
        
        # Remove ANSI escape sequences and other special characters
        cleaned_output = re.sub(r'\x1b\[[0-9;]*m', '', cleaned_output)  # Remove ANSI color codes
        cleaned_output = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', cleaned_output)  # Remove other escape sequences
        cleaned_output = re.sub(r'[^\x20-\x7E\n\r]', '', cleaned_output)  # Keep only printable ASCII and newlines
        
        # Parse the YAML output - handle multiple documents by taking the last one
        data = None
        try:
            # Try loading all YAML documents and take the last one (the actual message)
            documents = list(yaml.safe_load_all(cleaned_output))
            if documents:
                # Find the last document that has pose data
                for doc in reversed(documents):
                    if doc and isinstance(doc, dict) and 'pose' in doc:
                        data = doc
                        break
                # If no document with pose found, use the last document
                if data is None:
                    data = documents[-1]
        except yaml.YAMLError as e:
            # If safe_load_all fails, try to extract YAML content manually
            # Find the last YAML document between --- markers
            yaml_parts = cleaned_output.split('---')
            if len(yaml_parts) > 1:
                # Take the last part that looks like YAML
                for part in reversed(yaml_parts):
                    part = part.strip()
                    if part and ('pose:' in part or 'position:' in part):
                        try:
                            data = yaml.safe_load(part)
                            if data and 'pose' in data:
                                break
                        except yaml.YAMLError:
                            continue
            # Final fallback: try safe_load on entire output
            if data is None:
                try:
                    data = yaml.safe_load(cleaned_output)
                except yaml.YAMLError:
                    return {
                        "status": "error",
                        "message": f"Failed to parse YAML output: {str(e)}. Output preview: {cleaned_output[:200]}"
                    }
        
        if 'pose' not in data:
            return {
                "status": "error",
                "message": "No pose data found in command output"
            }
        
        pose_data = data['pose']
        current_pos = [
            pose_data['position']['x'],
            pose_data['position']['y'], 
            pose_data['position']['z']
        ]
        
        current_quat = [
            pose_data['orientation']['x'],
            pose_data['orientation']['y'],
            pose_data['orientation']['z'],
            pose_data['orientation']['w']
        ]
        
        # Convert to RPY
        current_rpy = quaternion_to_rpy(
            current_quat[0], current_quat[1], 
            current_quat[2], current_quat[3]
        )
        
        # Create target pose
        target_position = current_pos.copy()
        target_position[2] = safe_height
        
        target_rpy = [0, 180, 0]  # Fixed orientation [0, 180, 0]
        
        # Compute IK
        joint_angles = compute_ik(target_position, target_rpy)
        
        if joint_angles is None:
            return {
                "status": "error",
                "message": "IK failed: couldn't compute safe height position"
            }
        
        # Execute movement using existing execute_joint_trajectory function
        return execute_joint_trajectory(joint_angles.tolist(), 3.0)
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to move to safe height: {str(e)}"
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