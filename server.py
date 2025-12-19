from mcp.server.fastmcp import FastMCP, Image
from typing import List, Any, Optional, Union
from pathlib import Path
import json
import base64
from utils.websocket_manager import WebSocketManager
from msgs.geometry_msgs import Twist, PoseStamped
from msgs.sensor_msgs import Image as RosImage, JointState
import subprocess
import sys

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


# Configuration using environment variables with defaults (similar to newer version)
# ROS Bridge connection settings
LOCAL_IP = os.getenv("ROSBRIDGE_LOCAL_IP", "127.0.0.1")  # Default: localhost
ROSBRIDGE_IP = os.getenv("ROSBRIDGE_IP", "127.0.0.1")  # Default: localhost
ROSBRIDGE_PORT = int(os.getenv("ROSBRIDGE_PORT", "9090"))  # Default: rosbridge port

# This is Global WebSocket manager - don't close it after every operation
ws_manager = WebSocketManager(ROSBRIDGE_IP, ROSBRIDGE_PORT, LOCAL_IP)

# Output directories - use MCP_CLIENT_OUTPUT_DIR if set, otherwise use relative paths
# Directories are created lazily when needed by tools
BASE_OUTPUT_DIR = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
import sys
if BASE_OUTPUT_DIR:
    SCREENSHOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "screenshots")
    PYTHON_EXECUTIONS_DIR = os.path.join(BASE_OUTPUT_DIR, "python_executions")
else:
    SCREENSHOTS_DIR = "screenshots"
    PYTHON_EXECUTIONS_DIR = "python_executions"

# Initialize MCP 
mcp = FastMCP("ros-mcp-server")


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
        
        # First, unsubscribe from the topic to clear any buffered/stale messages
        # This ensures we get a fresh image on the next subscription
        try:
            unsubscribe_msg = {
                "op": "unsubscribe",
                "topic": topic_name
            }
            ws_manager.send(unsubscribe_msg)
            # Small delay to ensure unsubscribe is processed
            time.sleep(0.1)
            
            # Flush any pending messages from the WebSocket buffer
            # This ensures we don't get stale messages from previous subscriptions
            ws_manager.flush_pending_messages(timeout=0.1)
        except Exception as e:
            # If unsubscribe fails, continue anyway - might not be subscribed
            pass
        
        # Create dynamic image subscriber for the specified topic
        image_subscriber = RosImage(ws_manager, topic=topic_name)
        
        # Subscribe and get image data with timeout
        msg = image_subscriber.subscribe(timeout=timeout)
        
        # Cancel timeout
        signal.alarm(0)
        
        # Unsubscribe after getting the image to prevent message buffering
        try:
            unsubscribe_msg = {
                "op": "unsubscribe",
                "topic": topic_name
            }
            ws_manager.send(unsubscribe_msg)
        except Exception as e:
            # If unsubscribe fails, continue anyway
            pass
        
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
            os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
            filename = os.path.join(SCREENSHOTS_DIR, f"{timestamp}_{topic_clean}.jpg")
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
                if os.path.exists(SCREENSHOTS_DIR):
                    files = sorted([f for f in os.listdir(SCREENSHOTS_DIR) if f.endswith('.jpg') or f.endswith('.png')])
                    if files:
                        latest_file = os.path.join(SCREENSHOTS_DIR, files[-1])
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
            if os.path.exists(SCREENSHOTS_DIR):
                files = sorted([f for f in os.listdir(SCREENSHOTS_DIR) if f.endswith('.jpg') or f.endswith('.png')])
                if files:
                    latest_file = os.path.join(SCREENSHOTS_DIR, files[-1])
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
        # Run command - ROS2 environment should already be sourced
        cmd = f"timeout {timeout} ros2 topic echo {topic_name} --once"
        
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

# @mcp.tool()
# def perform_ik(target_position: List[float], target_rpy: List[float], 
#                duration: float = 5.0) -> Dict[str, Any]:
#     """
#     Perform inverse kinematics and execute smooth trajectory movement using ROS2.
#     
#     Args:
#         target_position: [x, y, z] target position in meters
#         target_rpy: [roll, pitch, yaw] target orientation in degrees
#         duration: Time to complete the movement in seconds (default: 5.0)
#         
#     Returns:
#         Raw output from the perform IK primitive script
#     """
#     timeout_seconds = int(duration) + 10  # Add buffer for communication overhead
#     args = f"--target-position {target_position[0]} {target_position[1]} {target_position[2]} --target-rpy {target_rpy[0]} {target_rpy[1]} {target_rpy[2]} --duration {duration}"
#     return _run_primitive("perform_ik.py", args, timeout=timeout_seconds, error_prefix="Perform IK")

# @mcp.tool()
# def get_ee_pose(joint_angles: List[float] = None) -> Dict[str, Any]:
#     """
#     Get end-effector pose from ROS topic /tcp_pose_broadcaster/pose.
#     
#     Args:
#         joint_angles: This parameter is ignored. The pose is read directly from ROS topic.
#         
#     Returns:
#         Raw output from the get EE pose primitive script
#     """
#     return _run_primitive("get_ee_pose.py", timeout=10, error_prefix="Get EE pose")

@mcp.tool()
def execute_python_code(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute Python code for calculations and math operations.
    
    This tool allows the agent to execute Python code for performing calculations,
    math operations, data processing, or any other Python computations.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30)
        
    Returns:
        Dictionary with output from the executed Python code (stdout + stderr) and file information
    """
    import subprocess
    import tempfile
    import os
    import sys
    import shutil
    
    try:
        # Create python_executions directory if it doesn't exist
        os.makedirs(PYTHON_EXECUTIONS_DIR, exist_ok=True)
        
        # Get list of files before execution
        files_before = set(os.listdir(PYTHON_EXECUTIONS_DIR))
        
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Wrap code with common imports and print result if it's an expression
            code_with_imports = f"""import math
import numpy as np
from datetime import datetime, timedelta
import json
import sys

# User's code:
{code}
"""
            f.write(code_with_imports)
            temp_file = f.name
        
        # Execute the code in the python_executions directory
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PYTHON_EXECUTIONS_DIR
        )
        
        # Clean up
        try:
            os.unlink(temp_file)
        except:
            pass
        
        # Get list of files after execution
        files_after = set(os.listdir(PYTHON_EXECUTIONS_DIR))
        created_files = files_after - files_before
        
        # Return combined stdout and stderr
        output = result.stdout if result.stdout else ""
        if result.stderr:
            output += result.stderr
        
        result_dict = {"output": output}
        
        # Files are already in the correct location (PYTHON_EXECUTIONS_DIR uses MCP_CLIENT_OUTPUT_DIR if set)
        if created_files:
            result_dict["files_created"] = list(created_files)
            result_dict["files_location"] = PYTHON_EXECUTIONS_DIR
        
        return result_dict
        
    except subprocess.TimeoutExpired:
        # Clean up the temp file
        try:
            os.unlink(temp_file)
        except:
            pass
        return {"output": f"Error: Code execution timed out after {timeout} seconds"}
    except Exception as e:
        return {"output": f"Error: Failed to execute Python code: {str(e)}"}

def _run_primitive(script_name: str, command_args: str = "", timeout: int = 60, error_prefix: str = "Primitive") -> Dict[str, Any]:
    """Helper function to run primitive scripts and return raw output.
    
    Args:
        script_name: Name of the primitive script (e.g., "move_home.py", "control_gripper.py")
        command_args: Optional command-line arguments to pass to the script
        timeout: Timeout for the subprocess (default: 60 seconds)
        error_prefix: Prefix for error messages (default: "Primitive")
    
    Returns:
        Dictionary with output from the primitive script (stdout + stderr)
    """
    import subprocess
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    cmd_parts = [
        f"cd {script_dir}/primitives",
        f"timeout {timeout} /usr/bin/python3 {script_name} {command_args}".strip()
    ]
    
    cmd = "\n".join(cmd_parts)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=timeout + 10  # Add buffer for subprocess timeout
        )
        
        # Return combined stdout and stderr (primitive handles its own output formatting)
        output = result.stdout if result.stdout else ""
        if result.stderr:
            output += result.stderr
        
        return {"output": output}
        
    except subprocess.TimeoutExpired:
        return {"output": f"Error: {error_prefix} timed out after {timeout} seconds"}
    except Exception as e:
        return {"output": f"Error: Failed to execute {error_prefix.lower()}: {str(e)}"}

@mcp.tool()
def move_home() -> Dict[str, Any]:
    """Move robot to home position."""
    return _run_primitive("move_home.py", timeout=45, error_prefix="Move home")

@mcp.tool()
def move_to_grasp(object_name: str, grasp_id: int, mode: str = "sim", move_to_object: bool = False, move_to_safe_height: bool = False) -> Dict[str, Any]:
    """Move to grasp position.
    This tool is used to move to the grasp an object. And once the object is grasped, you can move to the safe height.
    REQUIRED: At least one flag must be set to True.

    Args:
        object_name: Name of the object to grasp
        grasp_id: ID of the grasp point to use
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
        move_to_object: Moves to the specified grasp point
        move_to_safe_height: After closing gripper move to safe height 
    """
    # Validate that at least one flag is set
    if not (move_to_object or move_to_safe_height):
        return {"output": "Error: At least one of move_to_object or move_to_safe_height must be set to True"}
    
    cmd = f"--object-name \"{object_name}\" --grasp-id {grasp_id} --mode {mode}"
    if move_to_object:
        cmd += " --move-to-object"
    if move_to_safe_height:
        cmd += " --move-to-safe-height"
    
    return _run_primitive("move_to_grasp.py", cmd, timeout=60, error_prefix="Move to grasp")

# @mcp.tool()
# def reorient_for_assembly(object_name: str, base_name: str, mode: str = "sim", current_object_orientation: Optional[List[float]] = None, target_base_orientation: Optional[List[float]] = None) -> Dict[str, Any]:
#     """Reorient object for assembly.
# 
#     Args:
#         object_name: Name of the object to reorient
#         base_name: Name of the base object
#         mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
#         current_object_orientation: Current object orientation quaternion [x, y, z, w] (required in real mode, optional in sim mode)
#         target_base_orientation: Target base orientation quaternion [x, y, z, w] (required in real mode, optional in sim mode)
# 
#     """
#     cmd = f"--mode {mode} --object-name \"{object_name}\" --base-name \"{base_name}\""
#     if current_object_orientation is not None:
#         cmd += f" --current-object-orientation {' '.join(str(x) for x in current_object_orientation)}"
#     if target_base_orientation is not None:
#         cmd += f" --target-base-orientation {' '.join(str(x) for x in target_base_orientation)}"
#     return _run_primitive("reorient_for_assembly.py", cmd, timeout=90, error_prefix="Reorient for assembly")

@mcp.tool()
def reorient_object(object_name: str, base_name: str, mode: str = "sim", current_object_orientation: Optional[List[float]] = None, target_base_orientation: Optional[List[float]] = None) -> Dict[str, Any]:
    """Reorient object for assembly.

    Args:
        object_name: Name of the object to reorient
        base_name: Name of the base object
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
        current_object_orientation: Current object orientation quaternion [x, y, z, w] (required in real mode, optional in sim mode)
        target_base_orientation: Target base orientation quaternion [x, y, z, w] (required in real mode, optional in sim mode)

    """
    cmd = f"--mode {mode} --object-name \"{object_name}\" --base-name \"{base_name}\""
    if current_object_orientation is not None:
        cmd += f" --current-object-orientation {' '.join(str(x) for x in current_object_orientation)}"
    if target_base_orientation is not None:
        cmd += f" --target-base-orientation {' '.join(str(x) for x in target_base_orientation)}"
    return _run_primitive("reorient_for_assembly.py", cmd, timeout=90, error_prefix="Reorient for assembly")

# @mcp.tool()
# def translate_for_assembly(object_name: str, base_name: str, mode: str = "sim", final_base_pos: Optional[List[float]] = None, final_base_orientation: Optional[List[float]] = None, use_default_base: bool = False) -> Dict[str, Any]:
#     """Translate object to target position for assembly.
#     
#     Args:
#         object_name: Name of the object being held
#         base_name: Name of the base object
#         mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
#         final_base_pos: Final base position [x, y, z] in meters (required in real mode unless use_default_base is True)
#         final_base_orientation: Final base orientation quaternion [x, y, z, w] (required in real mode unless use_default_base is True)
#         use_default_base: Use default base position and orientation (for real mode)
#     
#     Returns:
#         Raw output from the translate for assembly primitive script
#     """
#     cmd = f"--mode {mode} --object-name \"{object_name}\" --base-name \"{base_name}\""
#     if final_base_pos is not None:
#         cmd += f" --final-base-pos {' '.join(str(x) for x in final_base_pos)}"
#     if final_base_orientation is not None:
#         cmd += f" --final-base-orientation {' '.join(str(x) for x in final_base_orientation)}"
#     if use_default_base:
#         cmd += " --use-default-base"
#     return _run_primitive("translate_for_assembly.py", cmd, timeout=90, error_prefix="Translate for assembly")

# @mcp.tool()
# def perform_insert(mode: str, object_name: Optional[str] = None, base_name: Optional[str] = None) -> Dict[str, Any]:
#     """Perform insert operation with force compliance (sim mode) or force-compliant movement (real mode).
#     
#     Args:
#         mode: Mode to use - "sim" for simulation or "real" for real robot (required)
#         object_name: Name of the object being held (required in sim mode)
#         base_name: Name of the base object (required in sim mode)
#     
#     Returns:
#         Raw output from the perform insert primitive script
#     """
#     # Build command based on mode
#     if mode == "sim":
#         args = f"--mode {mode} --object-name \"{object_name}\" --base-name \"{base_name}\""
#         timeout = 90
#     else:  # real mode
#         args = f"--mode {mode}"
#         timeout = 300
#     
#     return _run_primitive("perform_insert.py", args, timeout=timeout, error_prefix="Perform insert")

@mcp.tool()
def verify_final_assembly_pose(object_name: str, base_name: str) -> Dict[str, Any]:
    """Verify if object is in correct final assembly pose relative to base.
    
    Args:
        object_name: Name of the object
        base_name: Name of the base object
    
    Returns:
        Raw output from the verify final assembly pose primitive script
    """
    return _run_primitive("verify_final_assembly_pose.py", f"--object-name \"{object_name}\" --base-name \"{base_name}\"", timeout=30, error_prefix="Verify final assembly pose")

# @mcp.tool()
# def move_down(mode: str = "real") -> Dict[str, Any]:
#     """Move robot down with force monitoring.
#     
#     Args:
#         mode: Mode to use - "sim" for simulation or "real" for real robot (default: "real")
#     
#     Returns:
#         Raw output from the move down primitive script
#     """
#     return _run_primitive("move_down.py", f"--mode {mode}", timeout=300, error_prefix="Move down")

@mcp.tool()
def control_gripper(command: str, mode: str = "sim") -> Dict[str, Any]:
    """Control gripper with verification.
    
    Supports "open", "close", "half-open" (30mm), or numeric values 0-110 (width in mm).
    
    Args:
        command: Gripper command - "open", "close", "half-open" (30mm), or numeric value 0-110 (width in mm)
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")

    """
    return _run_primitive("control_gripper.py", f"{command} --mode {mode}", timeout=60, error_prefix="Gripper control")

# @mcp.tool()
# def move_to_safe_height() -> Dict[str, Any]:
#     """Move robot to safe height.
#     
#     Returns:
#         Raw output from the move to safe height primitive script
#     """
#     return _run_primitive("move_to_safe_height.py", timeout=30, error_prefix="Move to safe height")

# @mcp.tool()
# def move_to_clear_area(mode: str = "move") -> Dict[str, Any]:
#     """Move robot to clear area position.
#     
#     Args:
#         mode: Mode to use - "move" to keep current end-effector orientation (default) or "hover" for top-down (face-down) orientation
#     
#     Returns:
#         Raw output from the move to clear area primitive script
#     """
#     return _run_primitive("move_to_clear_area.py", f"--{mode}", timeout=45, error_prefix="Move to clear area")

# @mcp.tool()
# def get_target_ee_pose(object_name: str, base_name: str, mode: str) -> Dict[str, Any]:
#     """Get target end-effector pose (position and orientation) from assembly configuration.
#     
#     Args:
#         object_name: Name of the object
#         base_name: Name of the base object
#         mode: Mode to use - "sim" for simulation (reads base pose from topic) or "real" for real robot (uses default base pose)
#     
#     Returns:
#         Raw output from the get target EE pose primitive script
#     """
#     return _run_primitive("get_target_ee_pose.py", f"--object-name \"{object_name}\" --base-name \"{base_name}\" --mode {mode}", timeout=10, error_prefix="Get target EE pose")

# @mcp.tool()
# def get_target_object_pose(object_name: str, base_name: str, mode: str) -> Dict[str, Any]:
#     """Get target object pose (position and orientation) in world frame from assembly configuration.
#     
#     Args:
#         object_name: Name of the object
#         base_name: Name of the base object
#         mode: Mode to use - "sim" for simulation (reads base pose from topic) or "real" for real robot (uses default base pose)
#     
#     Returns:
#         Raw output from the get target object pose primitive script (JSON with target_object_pose)
#     """
#     return _run_primitive("get_target_object_pose.py", f"--object-name \"{object_name}\" --base-name \"{base_name}\" --mode {mode}", timeout=10, error_prefix="Get target object pose")

@mcp.tool()
def move_to_regrasp(mode: str, move_to_clear_space: bool = False, move_down: bool = False, move_to_safe_height: bool = False) -> Dict[str, Any]:
    """Move to regrasp position.
    This tool is used to aid in reorienting the current object if the by placing it down on clear space and then moving to safe height so the object can be grasped again.
    
    IMPORTANT: Only ONE flag can be set to True at a time. These flags must be called in sequence one by one to complete the move to regrasp sequence.
    
    Args:
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
        move_to_clear_space: This is to move above a clear space maintaining the current orientation of the object.
        move_down: This is a force compliant move down to place the object on the clear space.
        move_to_safe_height: This is to move to the safe height position after having opened the gripper. Now you are ready to grasp the object again.
    
    Returns:
        Raw output from the move to regrasp primitive script
    """
    # Count how many flags are set
    flags_set = sum([move_to_clear_space, move_down, move_to_safe_height])
    
    # Validate that exactly one flag is set
    if flags_set == 0:
        return {"output": "Error: Exactly one of move_to_clear_space, move_down, or move_to_safe_height must be set to True"}
    elif flags_set > 1:
        return {"output": "Error: Only one flag can be set at a time. Set exactly one of move_to_clear_space, move_down, or move_to_safe_height to True"}
    
    cmd = f"--mode {mode}"
    if move_to_clear_space:
        cmd += " --move-to-clear-space"
    if move_down:
        cmd += " --move-down"
    if move_to_safe_height:
        cmd += " --move-to-safe-height"
    
    return _run_primitive("move_to_regrasp.py", cmd, timeout=60, error_prefix="Move to regrasp")

@mcp.tool()
def translate_object(mode: str, object_name: Optional[str] = None, base_name: Optional[str] = None, move_to_base: bool = False, move_down: bool = False, final_base_pos: Optional[List[float]] = None, final_base_orientation: Optional[List[float]] = None, use_default_base: bool = False) -> Dict[str, Any]:
    """Translate object to target position.
    
    REQUIRED: Exactly one of move_to_base or move_down must be set to True (they are mutually exclusive).
    
    Args:
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
        object_name: Name of the object being held (required in sim mode)
        base_name: Name of the base object (required in sim mode when using move_to_base or move_down; required in real mode when using move_to_base)
        move_to_base: Moves to the specified base position (exactly one flag must be True)
        move_down: Moves down to the specified target object position (exactly one flag must be True)
        final_base_pos: Final base position [x, y, z] in meters (for translate_for_assembly in real mode)
        final_base_orientation: Final base orientation quaternion [x, y, z, w] (for translate_for_assembly in real mode)
        use_default_base: Use default base position and orientation (for translate_for_assembly in real mode)
    """
    # Validate that exactly one flag is set
    flags_set = sum([move_to_base, move_down])
    if flags_set == 0:
        return {"output": "Error: Exactly one of move_to_base or move_down must be set to True"}
    elif flags_set > 1:
        return {"output": "Error: move_to_base and move_down are mutually exclusive. Set exactly one to True"}
    
    # Validate sim mode requirements
    if mode == "sim":
        if object_name is None:
            return {"output": "Error: object_name is required in sim mode"}
        if base_name is None:
            return {"output": "Error: base_name is required in sim mode when using move_to_base or move_down"}
    
    # Validate real mode requirements for move_to_base
    if mode == "real" and move_to_base:
        if base_name is None:
            return {"output": "Error: base_name is required in real mode when using move_to_base"}
        if not use_default_base and final_base_pos is None:
            return {"output": "Error: In real mode with move_to_base, either final_base_pos or use_default_base is required"}
    
    cmd = f"--mode {mode}"
    if object_name is not None:
        cmd += f" --object-name \"{object_name}\""
    if base_name is not None:
        cmd += f" --base-name \"{base_name}\""
    if move_to_base:
        cmd += " --move-to-base"
    if move_down:
        cmd += " --move-down"
    if final_base_pos is not None:
        cmd += f" --final-base-pos {' '.join(str(x) for x in final_base_pos)}"
    if final_base_orientation is not None:
        cmd += f" --final-base-orientation {' '.join(str(x) for x in final_base_orientation)}"
    if use_default_base:
        cmd += " --use-default-base"
    
    # Adjust timeout based on operation
    if move_down:
        timeout = 300  # Force compliance can take longer
    else:
        timeout = 90
    
    return _run_primitive("translate_object.py", cmd, timeout=timeout, error_prefix="Translate object")


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