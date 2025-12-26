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


## ############################################################################################## ##
##
##                      ROS TOPIC TOOLS
##
## ############################################################################################## ##

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

@mcp.tool(description="Capture camera image from any topic and return it so the agent can see and analyze it.")
def capture_camera_image(topic_name: str, timeout: int = 10):
    """
    Capture camera image using any camera topic.
    Returns a list with status info and the image that the agent can see.
    
    Args:
        topic_name: The ROS topic to subscribe to
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

## ############################################################################################## ##
##
##                      CODE EXECUTION TOOLS
##
## ############################################################################################## ##

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

@mcp.tool()
def execute_policy_code(code: str, timeout: int = 3600) -> Dict[str, Any]:
    """Execute Python code with direct access to this server's primitives API.

    Allows executable Python code that calls server primitives with complex control flow
    (loops, conditionals, result-based branching, etc.).

    Available API: Import with 'from primitives.utils.primitives_api import *'
    All primitives return dictionaries with results for decision-making.

    Example:
    ```python
    from primitives.utils.primitives_api import *

    # Use any available primitives
    result = some_primitive(param1, param2)

    # Make decisions based on results
    if result["status"] == "success":
        another_primitive()
    ```

    Args:
        code: Python code (must import from primitives.utils.primitives_api)
        timeout: Maximum execution time in seconds (default: 3600)

    Returns:
        Dictionary with output from code execution (stdout + stderr)
    """
    import subprocess
    import tempfile
    import os
    import sys

    try:
        # Create python_executions directory if it doesn't exist
        os.makedirs(PYTHON_EXECUTIONS_DIR, exist_ok=True)

        # Get the script directory for PYTHONPATH (need this before building template)
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add standard imports that policies typically need
            code_with_imports = f"""import sys
import os
from typing import Dict, Any, List, Optional

# Add the parent directory to path so we can import primitives_api
sys.path.insert(0, '{script_dir}')

# Standard imports for policies
import math
import numpy as np
from datetime import datetime, timedelta
import json

# User's policy code:
{code}
"""
            f.write(code_with_imports)
            temp_file = f.name

        # Set up environment with PYTHONPATH
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{script_dir}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = script_dir

        # Execute the code in the python_executions directory
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PYTHON_EXECUTIONS_DIR,
            env=env
        )

        # Clean up
        try:
            os.unlink(temp_file)
        except:
            pass

        # Return combined stdout and stderr
        output = result.stdout if result.stdout else ""
        if result.stderr:
            output += "\n" + result.stderr

        result_dict = {
            "output": output,
            "returncode": result.returncode
        }

        # Add success indicator
        if result.returncode == 0:
            result_dict["status"] = "success"
        else:
            result_dict["status"] = "failed"
            result_dict["message"] = f"Policy execution failed with return code {result.returncode}"

        return result_dict

    except subprocess.TimeoutExpired:
        # Clean up the temp file
        try:
            os.unlink(temp_file)
        except:
            pass
        return {
            "output": f"Error: Policy execution timed out after {timeout} seconds",
            "status": "timeout"
        }
    except Exception as e:
        return {
            "output": f"Error: Failed to execute policy code: {str(e)}",
            "status": "error"
        }

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
    
    # Set PYTHONPATH to include project root for imports
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{script_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = script_dir
    
    cmd_parts = [
        f"cd {script_dir}/primitives",
        f"timeout {timeout} /usr/bin/python3 -u {script_name} {command_args}".strip()
    ]
    
    cmd = "\n".join(cmd_parts)
    
    try:
        # Use Popen with threading to capture output in real-time
        process = subprocess.Popen(
            cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            env=env
        )
        
        # Read output in a separate thread to avoid blocking
        output_lines = []
        output_lock = threading.Lock()
        read_complete = threading.Event()
        
        def read_output():
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        with output_lock:
                            output_lines.append(line)
                    if process.poll() is not None:
                        break
                # Read any remaining output
                remaining = process.stdout.read()
                if remaining:
                    with output_lock:
                        output_lines.append(remaining)
            except Exception:
                pass
            finally:
                read_complete.set()
        
        # Start reading thread
        read_thread = threading.Thread(target=read_output, daemon=True)
        read_thread.start()
        
        # Wait for process to complete or timeout
        start_time = time.time()
        while process.poll() is None:
            if time.time() - start_time > timeout + 10:
                # Process timed out, kill it
                process.kill()
                try:
                    process.wait()
                except:
                    pass
                break
            time.sleep(0.1)
        
        # Ensure process is terminated
        if process.poll() is None:
            process.kill()
            try:
                process.wait()
            except:
                pass
        
        # Wait a bit for output thread to finish reading
        read_complete.wait(timeout=1)
        
        # Get return code
        returncode = process.returncode
        
        # Combine all output
        with output_lock:
            output = "".join(output_lines)
        
        # If process was killed by timeout command (exit code 124), add timeout message
        if returncode == 124:
            if output:
                return {"output": f"{output}\n\nError: {error_prefix} timed out after {timeout} seconds", "returncode": returncode}
            else:
                return {"output": f"Error: {error_prefix} timed out after {timeout} seconds", "returncode": returncode}
        
        # If process was killed by us (returncode -9 or None), it timed out
        if returncode is None or returncode == -9:
            if output:
                return {"output": f"{output}\n\nError: {error_prefix} timed out after {timeout} seconds", "returncode": returncode or -9}
            else:
                return {"output": f"Error: {error_prefix} timed out after {timeout} seconds", "returncode": returncode or -9}
        
        return {"output": output if output else "", "returncode": returncode}
        
    except subprocess.TimeoutExpired as e:
        # Fallback: try to get any output from the exception
        output = ""
        if hasattr(e, 'stdout') and e.stdout:
            output += e.stdout
        if hasattr(e, 'stderr') and e.stderr:
            output += e.stderr
        if output:
            return {"output": f"{output}\n\nError: {error_prefix} timed out after {timeout} seconds", "returncode": None}
        else:
            return {"output": f"Error: {error_prefix} timed out after {timeout} seconds", "returncode": None}
    except Exception as e:
        return {"output": f"Error: Failed to execute {error_prefix.lower()}: {str(e)}", "returncode": None}

def _parse_verify_result(primitive_result: Dict[str, Any]) -> str:
    """Parse verification result from primitive output.
    
    Args:
        primitive_result: Dictionary returned from _run_primitive with "output" and "returncode" keys
    
    Returns:
        "SUCCESS" or "FAILURE" based on return code and output parsing
    """
    returncode = primitive_result.get("returncode")
    output = primitive_result.get("output", "").upper()
    
    # Primary method: check return code
    # 0 = success, non-zero = failure
    if returncode == 0:
        return "SUCCESS"
    elif returncode is not None and returncode != 0:
        return "FAILURE"
    
    # Fallback: parse output for keywords if returncode is ambiguous
    # Check for explicit success/failure messages
    if "SUCCESS" in output and "VERIFICATION" in output:
        return "SUCCESS"
    elif "FAILED" in output or "FAILURE" in output:
        return "FAILURE"
    
    # Default to FAILURE if we can't determine
    return "FAILURE"

def _run_query(script_name: str, command_args: str = "", timeout: int = 10, error_prefix: str = "Query") -> Dict[str, Any]:
    """Helper function to run query scripts and return raw output.
    
    Args:
        script_name: Name of the query script (e.g., "get_available_objects.py", "get_available_grasp_ids.py")
        command_args: Optional command-line arguments to pass to the script
        timeout: Timeout for the subprocess (default: 10 seconds)
        error_prefix: Prefix for error messages (default: "Query")
    
    Returns:
        Dictionary with output from the query script (stdout + stderr)
    """
    import subprocess
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    cmd_parts = [
        f"cd {script_dir}/queries",
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
            timeout=timeout + 5  # Add buffer for subprocess timeout
        )
        
        # Return combined stdout and stderr (query handles its own output formatting)
        output = result.stdout if result.stdout else ""
        if result.stderr:
            output += result.stderr
        
        return {"output": output}
        
    except subprocess.TimeoutExpired:
        return {"output": f"Error: {error_prefix} timed out after {timeout} seconds"}
    except Exception as e:
        return {"output": f"Error: Failed to execute {error_prefix.lower()}: {str(e)}"}

## ############################################################################################## ##
##
##                      QUERIES
##
## ############################################################################################## ##

@mcp.tool()
def get_available_objects(mode: str = "sim") -> Dict[str, Any]:
    """Get list of available object names from ROS topic.
    
    Args:
        mode: Mode to use - "sim" for simulation (reads from /objects_poses_sim) or "real" for real robot (reads from /objects_poses_real) (default: "sim")
    
    Returns:
        JSON output containing list of available object names
    """
    return _run_query("get_available_objects.py", f"--mode {mode}", timeout=10, error_prefix="Get available objects")

@mcp.tool()
def get_available_grasp_ids(mode: str = "sim") -> Dict[str, Any]:
    """Get available grasp IDs per object from ROS topic.
    
    Args:
        mode: Mode to use - "sim" for simulation (reads from /grasp_points_sim) or "real" for real robot (reads from /grasp_points_real) (default: "sim")
    
    Returns:
        JSON output containing available grasp IDs per object
    """
    return _run_query("get_available_grasp_ids.py", f"--mode {mode}", timeout=10, error_prefix="Get available grasp IDs")

# @mcp.tool()
# def get_current_object_pose(object_name: Optional[str] = None, mode: str = "sim") -> Dict[str, Any]:
#     """Get current object pose(s) from ROS topic.
#     
#     Args:
#         object_name: Optional name of the object to get pose for. If not provided, returns poses for all objects.
#         mode: Mode to use - "sim" for simulation (reads from /objects_poses_sim) or "real" for real robot (reads from /objects_poses_real) (default: "sim")
#     
#     Returns:
#         JSON output containing the current object pose(s) (position and orientation)
#     """
#     if object_name:
#         cmd = f"--object-name \"{object_name}\" --mode {mode}"
#     else:
#         cmd = f"--all --mode {mode}"
#     return _run_query("get_current_object_pose.py", cmd, timeout=10, error_prefix="Get current object pose")

## ############################################################################################## ##
##
##                      PRIMITIVES
##
## ############################################################################################## ##

@mcp.tool()
def move_home() -> Dict[str, Any]:
    """Move robot to home position."""
    return _run_primitive("move_home.py", timeout=45, error_prefix="Move home")

@mcp.tool()
def control_gripper(command: str, mode: str = "sim") -> Dict[str, Any]:
    """Control gripper.
    
    Supports "open", "close", "half-open" (30mm), or numeric values 0-110 (width in mm).
    
    Args:
        command: Gripper command - "open", "close", "half-open" (30mm), or numeric value 0-110 (width in mm)
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")

    """
    return _run_primitive("control_gripper.py", f"{command} --mode {mode}", timeout=60, error_prefix="Gripper control")

@mcp.tool()
def scan_workspace(object_name: str) -> Dict[str, Any]:
    """Scan workspace at fixed height to locate object.
    
    This tool scans the workspace by following a predefined path across x,y at a fixed z height.
    The robot moves along the path and stops as soon as the object is detected.
    Only works in real mode (not available for simulation).
    
    Args:
        object_name: Name of the object to locate
    
    Returns:
        Raw output from the scan workspace primitive script
    """
    return _run_primitive("scan_workspace.py", f"--object-name \"{object_name}\" --mode real", timeout=300, error_prefix="Scan workspace")

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
        Raw output from the move to regrasp primitive script. Notw down the object position and orientation for future use.
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
def translate_object(mode: str, base_name: Optional[str] = None, object_name: Optional[str] = None, move_to_base: bool = False, move_down: bool = False, move_to_safe_height: bool = False, use_default_base_position: bool = False) -> Dict[str, Any]:
    """Translate object to target position.
    Moves object to target position relative to base.
    REQUIRED: Exactly one of move_to_base, move_down, or move_to_safe_height must be set to True (they are mutually exclusive).
    
    Args:
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
        base_name: Name of the base object (required)
        object_name: Name of the object being held (required in sim mode)
        move_to_base: Moves to the specified base position in safe height (exactly one flag must be True)
        move_down: Moves down to the specified target object position (exactly one flag must be True)
        move_to_safe_height: After closing gripper move to safe height (exactly one flag must be True)
        use_default_base_position: Use default base position and orientation (for real mode)
    """
    # Validate that exactly one flag is set
    flags_set = sum([move_to_base, move_down, move_to_safe_height])
    if flags_set == 0:
        return {"output": "Error: Exactly one of move_to_base, move_down, or move_to_safe_height must be set to True"}
    elif flags_set > 1:
        return {"output": "Error: move_to_base, move_down, and move_to_safe_height are mutually exclusive. Set exactly one to True"}
    
    if mode == "sim" and object_name is None:
        return {"output": "Error: object_name is required in sim mode"}
    
    cmd = f"--mode {mode}"
    if object_name is not None:
        cmd += f" --object-name \"{object_name}\""
    if base_name is not None:
        cmd += f" --base-name \"{base_name}\""
    if move_to_base:
        cmd += " --move-to-base"
    if move_down:
        cmd += " --move-down"
    if move_to_safe_height:
        cmd += " --move-to-safe-height"
    if use_default_base_position:
        cmd += " --use-default-base-position"
    
    # Adjust timeout based on operation
    # For move_down (perform_insert), use a long timeout to let it complete naturally
    # The alignment_stop_timeout (10s) in perform_insert will handle stopping the alignment phase
    # 900s (15 min) allows for: search phase + alignment phase (max 10s) + any delays
    if move_down:
        timeout = 300  
    elif move_to_safe_height:
        timeout = 40  # Safe height movement is quick
    else:
        timeout = 90
    
    return _run_primitive("translate_object.py", cmd, timeout=timeout, error_prefix="Translate object")

@mcp.tool()
def reorient_object(object_name: str, base_name: str, mode: str = "sim", current_object_orientation: Optional[List[float]] = None, target_base_orientation: Optional[List[float]] = None, use_default_base_orientation: bool = False) -> Dict[str, Any]:
    """Reorient object for assembly.
    Reorients object to target base orientation relative to base.
    Args:
        object_name: Name of the object to reorient
        base_name: Name of the base object
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
        current_object_orientation: Current object orientation quaternion [x, y, z, w] (required in real mode and always use the orientation of the object you got after moving to grasp the object because the object might not be visible in the camera after moving to grasp the object.)
        target_base_orientation: Target base orientation quaternion [x, y, z, w] (required in real mode unless use_default_base_orientation is True, optional in sim mode)
        use_default_base_orientation: Use default base orientation [0.0, 0.0, 0.0, 1.0] (for real mode, mutually exclusive with target_base_orientation)

    """
    cmd = f"--mode {mode} --object-name \"{object_name}\" --base-name \"{base_name}\""
    if current_object_orientation is not None:
        cmd += f" --current-object-orientation {' '.join(str(x) for x in current_object_orientation)}"
    if use_default_base_orientation:
        cmd += " --use-default-base-orientation"
    elif target_base_orientation is not None:
        cmd += f" --target-base-orientation {' '.join(str(x) for x in target_base_orientation)}"
    return _run_primitive("reorient_object.py", cmd, timeout=90, error_prefix="Reorient for assembly")

## ############################################################################################## ##
##
##                      VERIFICATION TOOLS
##
## ############################################################################################## ##


@mcp.tool()
def verify_grasp(object_name: str, mode: str = "sim") -> Dict[str, Any]:
    """Verify if object is within grasp radius from gripper center.
    
    This tool checks if an object is successfully grasped by verifying that the object position
    is within a 6cm radius from the gripper center position in all directions.
    Only call this tool after moving to safe height.
    
    Args:
        object_name: Name of the object to verify
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
    
    Returns:
        Dictionary with "output" (raw output from script) and "result" ("SUCCESS" or "FAILURE")
    """
    primitive_result = _run_primitive("verify_grasp.py", f"--object-name \"{object_name}\" --mode {mode} --radius 0.06", timeout=30, error_prefix="Verify grasp")
    result = _parse_verify_result(primitive_result)
    return {
        "output": primitive_result.get("output", ""),
        "result": result
    }

@mcp.tool()
def verify_assembly(object_name: str, base_name: str) -> Dict[str, Any]:
    """Verify if object is in correct assembly pose relative to base.
    
    Args:
        object_name: Name of the object
        base_name: Name of the base object
    
    Returns:
        Dictionary with "output" (raw output from script) and "result" ("SUCCESS" or "FAILURE")
    """
    primitive_result = _run_primitive("verify_assembly.py", f"--object-name \"{object_name}\" --base-name \"{base_name}\"", timeout=30, error_prefix="Verify assembly")
    result = _parse_verify_result(primitive_result)
    return {
        "output": primitive_result.get("output", ""),
        "result": result
    }

@mcp.tool()
def verify_disassembly(object_name: str, base_name: str) -> Dict[str, Any]:
    """Verify if object is NOT in assembly position relative to base.
    
    This tool checks if an object has been successfully disassembled by verifying it is NOT in the 
    target assembly position. Returns success if the object is away from the assembly position.
    
    Args:
        object_name: Name of the object to verify
        base_name: Name of the base object
    
    Returns:
        Dictionary with "output" (raw output from script) and "result" ("SUCCESS" or "FAILURE")
    """
    primitive_result = _run_primitive("verify_disassembly.py", f"--object-name \"{object_name}\" --base-name \"{base_name}\"", timeout=30, error_prefix="Verify disassembly")
    result = _parse_verify_result(primitive_result)
    return {
        "output": primitive_result.get("output", ""),
        "result": result
    }

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