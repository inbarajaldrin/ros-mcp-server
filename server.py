from mcp.server.fastmcp import FastMCP, Image
from typing import List, Any, Optional, Union
from pathlib import Path
import json
import base64
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


# Configuration using environment variables with defaults (similar to newer version)
# ROS Bridge connection settings
LOCAL_IP = os.getenv("ROSBRIDGE_LOCAL_IP", "127.0.0.1")  # Default: localhost
ROSBRIDGE_IP = os.getenv("ROSBRIDGE_IP", "127.0.0.1")  # Default: localhost
ROSBRIDGE_PORT = int(os.getenv("ROSBRIDGE_PORT", "9090"))  # Default: rosbridge port

# This is Global WebSocket manager - don't close it after every operation
ws_manager = WebSocketManager(ROSBRIDGE_IP, ROSBRIDGE_PORT, LOCAL_IP)

# MCP Directory - use script directory
MCP_SRV_DIR = str(Path(__file__).parent.absolute())
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

@mcp.tool()
def perform_ik(target_position: List[float], target_rpy: List[float], 
               duration: float = 5.0) -> Dict[str, Any]:
    """
    Perform inverse kinematics and execute smooth trajectory movement using ROS2.
    
    Args:
        target_position: [x, y, z] target position in meters
        target_rpy: [roll, pitch, yaw] target orientation in degrees
        duration: Time to complete the movement in seconds (default: 5.0)
        
    Returns:
        Raw output from the perform IK primitive script
    """
    timeout_seconds = int(duration) + 10  # Add buffer for communication overhead
    args = f"--target-position {target_position[0]} {target_position[1]} {target_position[2]} --target-rpy {target_rpy[0]} {target_rpy[1]} {target_rpy[2]} --duration {duration}"
    return _run_primitive("perform_ik.py", args, timeout=timeout_seconds, error_prefix="Perform IK")

@mcp.tool()
def get_ee_pose(joint_angles: List[float] = None) -> Dict[str, Any]:
    """
    Get end-effector pose from ROS topic /tcp_pose_broadcaster/pose.
    
    Args:
        joint_angles: This parameter is ignored. The pose is read directly from ROS topic.
        
    Returns:
        Raw output from the get EE pose primitive script
    """
    return _run_primitive("get_ee_pose.py", timeout=10, error_prefix="Get EE pose")

@mcp.tool()
def execute_python_code(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute Python code for calculations and math operations.
    
    This tool allows the agent to execute Python code for performing calculations,
    math operations, data processing, or any other Python computations.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30)
        
    Returns:
        Dictionary with output from the executed Python code (stdout + stderr)
    """
    import subprocess
    import tempfile
    import os
    import sys
    
    try:
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
        
        # Execute the code
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Clean up
        try:
            os.unlink(temp_file)
        except:
            pass
        
        # Return combined stdout and stderr
        output = result.stdout if result.stdout else ""
        if result.stderr:
            output += result.stderr
        
        return {"output": output}
        
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
def move_to_grasp(object_name: str, grasp_id: int, mode: str = "sim") -> Dict[str, Any]:
    """Move to grasp position.
    
    The grasp_id should be obtained from either:
    - The captured image space (visual analysis of the image)
    - The /grasp_points_sim topic (sim mode) or /grasp_points_real topic (real mode) (ROS topics that publish available grasp points)
    
    Args:
        object_name: Name of the object to grasp
        grasp_id: ID of the grasp point to use
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
    
    Returns:
        Raw output from the move to grasp primitive script
    """
    return _run_primitive("move_to_grasp.py", f"--object-name \"{object_name}\" --grasp-id {grasp_id} --mode {mode}", timeout=60, error_prefix="Move to grasp")

@mcp.tool()
def reorient_for_assembly(object_name: str, base_name: str, mode: str = "sim") -> Dict[str, Any]:
    """Reorient object for assembly.
    
    Args:
        object_name: Name of the object to reorient
        base_name: Name of the base object
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
    
    Returns:
        Raw output from the reorient for assembly primitive script
    """
    return _run_primitive("reorient_for_assembly.py", f"--mode {mode} --object-name \"{object_name}\" --base-name \"{base_name}\"", timeout=90, error_prefix="Reorient for assembly")

@mcp.tool()
def translate_for_assembly(object_name: str, base_name: str, mode: str = "sim") -> Dict[str, Any]:
    """Translate object to target position for assembly.
    
    Args:
        object_name: Name of the object being held
        base_name: Name of the base object
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
    
    Returns:
        Raw output from the translate for assembly primitive script
    """
    return _run_primitive("translate_for_assembly.py", f"--mode {mode} --object-name \"{object_name}\" --base-name \"{base_name}\"", timeout=90, error_prefix="Translate for assembly")

@mcp.tool()
def perform_insert(mode: str, object_name: Optional[str] = None, base_name: Optional[str] = None) -> Dict[str, Any]:
    """Perform insert operation with force compliance (sim mode) or force-compliant movement (real mode).
    
    Args:
        mode: Mode to use - "sim" for simulation or "real" for real robot (required)
        object_name: Name of the object being held (required in sim mode)
        base_name: Name of the base object (required in sim mode)
    
    Returns:
        Raw output from the perform insert primitive script
    """
    # Build command based on mode
    if mode == "sim":
        args = f"--mode {mode} --object-name \"{object_name}\" --base-name \"{base_name}\""
        timeout = 90
    else:  # real mode
        args = f"--mode {mode}"
        timeout = 300
    
    return _run_primitive("perform_insert.py", args, timeout=timeout, error_prefix="Perform insert")

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

@mcp.tool()
def move_down(mode: str = "real") -> Dict[str, Any]:
    """Move robot down with force monitoring.
    
    Args:
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "real")
    
    Returns:
        Raw output from the move down primitive script
    """
    return _run_primitive("move_down.py", f"--mode {mode}", timeout=300, error_prefix="Move down")

@mcp.tool()
def control_gripper(command: str, mode: str = "sim") -> Dict[str, Any]:
    """Control gripper with verification.
    
    Supports "open", "close", or numeric values 0-110 (representing 0-11cm width).
    
    Args:
        command: Gripper command - "open", "close", or numeric value 0-110 (width in mm, representing 0-11cm)
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
    
    Returns:
        Raw output from the gripper control primitive script
    """
    return _run_primitive("control_gripper.py", f"{command} --mode {mode}", timeout=60, error_prefix="Gripper control")

@mcp.tool()
def move_to_safe_height() -> Dict[str, Any]:
    """Move robot to safe height.
    
    Returns:
        Raw output from the move to safe height primitive script
    """
    return _run_primitive("move_to_safe_height.py", timeout=30, error_prefix="Move to safe height")

@mcp.tool()
def move_to_clear_area(mode: str = "move") -> Dict[str, Any]:
    """Move robot to clear area position.
    
    Args:
        mode: Mode to use - "move" to keep current end-effector orientation (default) or "hover" for top-down (face-down) orientation
    
    Returns:
        Raw output from the move to clear area primitive script
    """
    return _run_primitive("move_to_clear_area.py", f"--{mode}", timeout=45, error_prefix="Move to clear area")


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