from mcp.server.fastmcp import FastMCP, Image, Context
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field
from typing import List, Any, Optional, Union, Literal, Dict
from pathlib import Path
import json
import base64
from utils.websocket_manager import WebSocketManager
from msgs.geometry_msgs import Twist, PoseStamped
from msgs.sensor_msgs import Image as RosImage, JointState
import subprocess
import sys
import logging

#camera
import time
import os
from datetime import datetime, timedelta
import io
import numpy as np
import cv2
from PIL import Image as PILImage
import threading

#ik
import tempfile
import os
from typing import List, Dict, Any
import numpy as np
from scipy.spatial.transform import Rotation as R
import traceback
import re

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RosMCPServer")

# Type aliases for consistent parameter types
Mode = Literal["sim", "real"]
GripperCommand = Literal["open", "close", "half-open"]
MoveToGraspAction = Literal["move_to_object", "move_to_safe_height"]
MoveToRegraspAction = Literal["move_to_clear_space", "move_down", "move_ee_top_down"]
TranslateAction = Literal["move_to_base", "perform_insert", "move_to_safe_height", "move_away_from_base"]

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
    PYTHON_EXECUTIONS_DIR = os.path.join(BASE_OUTPUT_DIR, "python_executions")
else:
    PYTHON_EXECUTIONS_DIR = "python_executions"

# Use Cyclone DDS for faster discovery (3-7x faster than FastDDS UDP-only).
# Shared memory disabled to prevent /dev/shm exhaustion from zombie processes.
# Note: ROS_LOCALHOST_ONLY=1 still works but disables multicast (slightly slower).
_cyclonedds_profile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cyclonedds_local.xml")
if os.path.exists(_cyclonedds_profile):
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    os.environ.setdefault("CYCLONEDDS_URI", f"file://{_cyclonedds_profile}")

def _start_services():
    """Start rosbridge and grasp publisher as detached processes."""
    logger.info("RosMCP server starting up")

    # Kill stale processes from previous sessions
    for pattern in ["rosbridge_websocket", "rosapi_node", "grasp_points_publisher.py"]:
        result = subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
        if result.returncode == 0:
            logger.info(f"Cleaned up stale {pattern} processes")
    time.sleep(1)

    # Start rosbridge (start_new_session detaches from parent, DEVNULL fully disconnects)
    rosbridge_process = subprocess.Popen(
        ["ros2", "launch", "rosbridge_server", "rosbridge_websocket_launch.xml"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    logger.info(f"Started rosbridge server (PID: {rosbridge_process.pid})")

    # Start grasp publisher
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grasp_publisher_process = subprocess.Popen(
        ["/usr/bin/python3", f"{script_dir}/primitives/utils/grasp_points_publisher.py", "--mode", "default"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    logger.info(f"Started grasp points publisher (PID: {grasp_publisher_process.pid})")

    time.sleep(2)


# Initialize MCP (no lifespan - services started in __main__ before mcp.run())
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

    File saving: Use relative paths (e.g., "output.txt") instead of absolute paths like "/tmp/output.txt".

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

# @mcp.tool()
# def execute_policy_code(code: str, timeout: int = 3600) -> Dict[str, Any]:
#     """Execute Python code with direct access to this server's primitives API.
#
#     Allows executable Python code that calls server primitives with complex control flow
#     (loops, conditionals, result-based branching, etc.).
#
#     Available API: Import with 'from primitives.utils.primitives_api import *'
#     All primitives return dictionaries with results for decision-making.
#
#     Example:
#     ```python
#     from primitives.utils.primitives_api import *
#
#     # Use any available primitives
#     result = some_primitive(param1, param2)
#
#     # Make decisions based on results
#     if result["status"] == "success":
#         another_primitive()
#     ```
#
#     Args:
#         code: Python code (must import from primitives.utils.primitives_api)
#         timeout: Maximum execution time in seconds (default: 3600)
#
#     Returns:
#         Dictionary with output from code execution (stdout + stderr)
#     """
#     import subprocess
#     import tempfile
#     import os
#     import sys
#
#     try:
#         # Create python_executions directory if it doesn't exist
#         os.makedirs(PYTHON_EXECUTIONS_DIR, exist_ok=True)
#
#         # Get the script directory for PYTHONPATH (need this before building template)
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#
#         # Create a temporary Python file
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
#             # Add standard imports that policies typically need
#             code_with_imports = f"""import sys
# import os
# from typing import Dict, Any, List, Optional
#
# # Add the parent directory to path so we can import primitives_api
# sys.path.insert(0, '{script_dir}')
#
# # Standard imports for policies
# import math
# import numpy as np
# from datetime import datetime, timedelta
# import json
#
# # User's policy code:
# {code}
# """
#             f.write(code_with_imports)
#             temp_file = f.name
#
#         # Set up environment with PYTHONPATH
#         env = os.environ.copy()
#         if 'PYTHONPATH' in env:
#             env['PYTHONPATH'] = f"{script_dir}:{env['PYTHONPATH']}"
#         else:
#             env['PYTHONPATH'] = script_dir
#
#         # Execute the code in the python_executions directory
#         result = subprocess.run(
#             [sys.executable, temp_file],
#             capture_output=True,
#             text=True,
#             timeout=timeout,
#             cwd=PYTHON_EXECUTIONS_DIR,
#             env=env
#         )
#
#         # Clean up
#         try:
#             os.unlink(temp_file)
#         except:
#             pass
#
#         # Return combined stdout and stderr
#         output = result.stdout if result.stdout else ""
#         if result.stderr:
#             output += "\n" + result.stderr
#
#         result_dict = {
#             "output": output,
#             "returncode": result.returncode
#         }
#
#         # Add success indicator
#         if result.returncode == 0:
#             result_dict["status"] = "success"
#         else:
#             result_dict["status"] = "failed"
#             result_dict["message"] = f"Policy execution failed with return code {result.returncode}"
#
#         return result_dict
#
#     except subprocess.TimeoutExpired:
#         # Clean up the temp file
#         try:
#             os.unlink(temp_file)
#         except:
#             pass
#         return {
#             "output": f"Error: Policy execution timed out after {timeout} seconds",
#             "status": "timeout"
#         }
#     except Exception as e:
#         return {
#             "output": f"Error: Failed to execute policy code: {str(e)}",
#             "status": "error"
#         }

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
    
    # Use sys.executable to ensure we use the same Python interpreter as the MCP server
    # This preserves conda/virtualenv environment and ROS2 DDS configuration
    import sys
    python_executable = sys.executable

    cmd_parts = [
        f"cd {script_dir}/primitives",
        f"timeout {timeout} {python_executable} -u {script_name} {command_args}".strip()
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
        
        # Check if output contains JSON markers (parse even on timeout)
        if output and "__RESULT_JSON__" in output and "__END_RESULT_JSON__" in output:
            # Extract JSON portion - use rfind to get the LAST occurrence
            # This handles cases where subprocess output also contains markers with ROS logger prefixes
            start_marker = "__RESULT_JSON__"
            end_marker = "__END_RESULT_JSON__"
            start_idx = output.rfind(start_marker) + len(start_marker)
            end_idx = output.rfind(end_marker)
            json_str = output[start_idx:end_idx].strip()

            try:
                # Parse and return the JSON directly (no extra fields)
                import json
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, fall through to old format handling
                pass

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

        # Old format (backward compatible)
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

def _run_query(script_name: str, command_args: str = "", timeout: int = 10, error_prefix: str = "Query") -> Dict[str, Any]:
    """Helper function to run query scripts and return raw output.
    
    Args:
        script_name: Name of the query script (e.g., "get_scene_info.py", "get_current_grasp_points_pose.py")
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

        # Return combined stdout and stderr
        output = result.stdout if result.stdout else ""
        if result.stderr:
            output += result.stderr

        # Check if output contains JSON markers (parse JSON if present)
        if output and "__RESULT_JSON__" in output and "__END_RESULT_JSON__" in output:
            # Extract JSON portion - use rfind to get the LAST occurrence
            # This handles cases where subprocess output also contains markers with ROS logger prefixes
            start_marker = "__RESULT_JSON__"
            end_marker = "__END_RESULT_JSON__"
            start_idx = output.rfind(start_marker) + len(start_marker)
            end_idx = output.rfind(end_marker)
            json_str = output[start_idx:end_idx].strip()

            try:
                # Parse and return the JSON directly (no extra fields)
                import json
                result_json = json.loads(json_str)
                return result_json
            except json.JSONDecodeError:
                # If JSON parsing fails, fall through to returning raw output
                pass

        # Fallback: return raw output if no JSON markers or parsing failed
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
def get_scene_info(mode: Mode = "sim") -> Dict[str, Any]:
    """Get scene information with objects and their available grasp points.

    All objects are included; those without grasp points have an empty array.

    Args:
        mode: Robot mode

    Returns:
        JSON with format:
        {
            "object_name": [
                {"id": 0, "gripper_states": ["open", "half-open"]},
                {"id": 1, "gripper_states": ["open"]}
            ],
            "another_object": []
        }
    """
    return _run_query("get_scene_info.py", f"--mode {mode}", timeout=10, error_prefix="Get scene info")

@mcp.tool()
def get_target_object_pose(object_name: str, base_name: str, mode: Mode = "sim") -> Dict[str, Any]:
    """Get target object pose in world frame from assembly configuration.

    Calculates the target object pose in world frame by:
    1. Loading assembly configuration from JSON
    2. Reading base pose (from ROS topic in sim mode, or using default in real mode)
    3. Extracting target position and orientation from JSON (relative to base)
    4. Transforming target pose from base frame to world frame

    Args:
        object_name: Name of the object
        base_name: Name of the base object
        mode: Robot mode

    Returns:
        JSON output containing the target object pose in world frame (position and orientation)
    """
    cmd = f"--object-name \"{object_name}\" --base-name \"{base_name}\" --mode {mode}"
    return _run_query("get_target_object_pose.py", cmd, timeout=10, error_prefix="Get target object pose")

@mcp.tool()
def get_current_object_pose(object_name: Optional[str] = None, mode: Mode = "sim", all: bool = False) -> Dict[str, Any]:
    """Get current object pose(s) from ROS topic.

    Args:
        object_name: Name of the object to get pose for (ignored if all=True)
        mode: Robot mode
        all: If True, returns poses for all objects

    Returns:
        JSON output containing the current object pose(s) (position and orientation)
    """
    if all:
        cmd = f"--all --mode {mode}"
    elif object_name:
        cmd = f"--object-name \"{object_name}\" --mode {mode}"
    else:
        return {"output": "Error: Either object_name or all=True must be provided"}
    return _run_query("get_current_object_pose.py", cmd, timeout=10, error_prefix="Get current object pose")

@mcp.tool()
def verify_grasp(object_name: str, mode: Mode = "sim") -> Dict[str, Any]:
    """Verify if object is within grasp radius from gripper center.

    Checks if object position is within 6cm radius from gripper center.
    Only call this tool after moving to safe height.

    Args:
        object_name: Name of the object to verify
        mode: Robot mode

    Returns:
        Dictionary with result (success or failure)
    """
    return _run_query("verify_grasp.py", f"--object-name \"{object_name}\" --mode {mode} --radius 0.06", timeout=30, error_prefix="Verify grasp")

@mcp.tool()
def verify_assembly(base_name: str, object_name: str = None, check_all: bool = False) -> Dict[str, Any]:
    """Verify if object(s) are in correct assembly pose relative to base.

    Args:
        base_name: Name of the base object
        object_name: Name of the object to verify (optional if check_all is True)
        check_all: If True, check all objects in the assembly instead of a specific one

    Returns:
        When check_all=False (single object):
        - "result": "success" or "failure" for the verified object
        - "object_name": Name of the verified object
        - "base_name": Name of the base object
        - "position_error_m": Position error metrics (x, y, z)
        - "orientation_error_deg": Orientation error metrics (roll, pitch, yaw)
        - "within_tolerance": Boolean indicating if within tolerance
        - "unassembled_objects": List of other objects in the same assembly that are NOT assembled

        When check_all=True:
        - "result": "success" if all assembled, "failure" otherwise
        - "base_name": Name of the base object
        - "all_assembled": Boolean indicating if all objects are assembled
        - "assembled_objects": List of objects that are correctly assembled
        - "unassembled_objects": List of objects that are NOT assembled
    """
    if check_all:
        return _run_query("verify_assembly.py", f"--base-name \"{base_name}\" --check-all", timeout=30, error_prefix="Verify assembly")
    elif object_name:
        return _run_query("verify_assembly.py", f"--object-name \"{object_name}\" --base-name \"{base_name}\"", timeout=30, error_prefix="Verify assembly")
    else:
        return {"result": "failure", "error": "Either object_name or check_all=True must be specified"}

@mcp.tool()
def verify_disassembly(base_name: str, object_name: str = None, check_all: bool = False) -> Dict[str, Any]:
    """Verify if object(s) are NOT in assembly position relative to base.

    This tool checks if object(s) have been successfully disassembled by verifying they are NOT in the
    target assembly position. Returns success if the object(s) are away from the assembly position.

    Args:
        base_name: Name of the base object
        object_name: Name of the object to verify (optional if check_all is True)
        check_all: If True, check all objects in the assembly instead of a specific one

    Returns:
        When check_all=False (single object):
        - "result": "success" or "failure" for the verified object
        - "object_name": Name of the verified object
        - "base_name": Name of the base object
        - "position_error_m": Position error metrics (x, y, z)
        - "orientation_error_deg": Orientation error metrics (roll, pitch, yaw)

        When check_all=True:
        - "result": "success" if all disassembled, "failure" otherwise
        - "base_name": Name of the base object
        - "all_disassembled": Boolean indicating if all objects are disassembled
        - "disassembled_objects": List of objects that are disassembled
        - "still_assembled_objects": List of objects still in assembly position
    """
    if check_all:
        return _run_query("verify_disassembly.py", f"--base-name \"{base_name}\" --check-all", timeout=30, error_prefix="Verify disassembly")
    elif object_name:
        return _run_query("verify_disassembly.py", f"--object-name \"{object_name}\" --base-name \"{base_name}\"", timeout=30, error_prefix="Verify disassembly")
    else:
        return {"result": "failure", "error": "Either object_name or check_all=True must be specified"}

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
def control_gripper(command: GripperCommand | int, mode: Mode = "sim") -> Dict[str, Any]:
    """Control gripper.

    Args:
        command: Gripper action or numeric width 0-110 mm
        mode: Robot mode
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
        Dictionary containing:
        - "result": "success" or "failure"
        - "mode": "real"
        - "object_name": Name of the object that was scanned for
        - "error": Error message (only present if result is "failure")
    """
    return _run_primitive("scan_workspace.py", f"--object-name \"{object_name}\" --mode real", timeout=300, error_prefix="Scan workspace")

@mcp.tool()
def move_to_grasp(object_name: str, grasp_id: int, action: MoveToGraspAction, mode: Mode = "sim") -> Dict[str, Any]:
    """Move to grasp position.

    Workflow:
    1. Call move_to_object to move to the grasp point (make sure the gripper is open before this call)
    2. Call control_gripper to grasp the object
    3. Call move_to_safe_height to move to the safe height (z=0.3m)

    Args:
        object_name: Name of the object to grasp
        grasp_id: ID of the grasp point to use
        action: Which step to perform (move_to_object or move_to_safe_height)
        mode: Robot mode

    Returns:
        Dictionary containing:
        - "result": "success" or "failure"
        - "object_name": Name of the object
        - "grasp_id": ID of the grasp point used
        - "mode": Robot mode ("sim" or "real")
        - "movement_type": The action that was performed (move_to_object or move_to_safe_height)
        - "current_object_position": Object position after movement (x, y, z) - only on success
        - "current_object_orientation": Object orientation quaternion (x, y, z, w) - only on success
        - "current_object_orientation_rpy_deg": Object orientation in degrees (roll, pitch, yaw) - only on success
        - "error": Error message (only present if result is "failure")
    """
    cmd = f"--object-name \"{object_name}\" --grasp-id {grasp_id} --mode {mode}"
    cmd += f" --{action.replace('_', '-')}"
    return _run_primitive("move_to_grasp.py", cmd, timeout=60, error_prefix="Move to grasp")

@mcp.tool()
def move_to_regrasp(action: MoveToRegraspAction, mode: Mode = "sim") -> Dict[str, Any]:
    """Move to regrasp position.

    Purpose: After rotating an object, the EE may be in a non-optimal orientation causing IK failures.
    This tool places the rotated object down and enables regrasping with a top-down EE orientation.

    Workflow: 
    1. Call rotate_object to move the object to the target orientation relative to base orientation (which results in a non-optimal end effector orientation)
    2. Call move_to_clear_space to move to the clear space (make sure the objects is already grasped and rotated)
    3. Call move_down to place the object on the table
    4. Call control_gripper to release the object
    5. Call move_ee_top_down to move the EE to the top-down orientation at z=0.3m
    6. Call move_to_grasp to grasp the object again
    7. IMPORTANT: Call rotate_object again to restore the object's orientation. Opening the gripper to drop the object typically causes minor orientation changes, so this step is REQUIRED to correct the orientation back to the target before continuing.
    8. Continue with what you were doing

    Args:
        action: Which step to perform (move_to_clear_space, move_down, or move_ee_top_down)
        mode: Robot mode

    Returns:
        Dictionary containing:
        - "result": "success" or "failure"
        - "mode": Robot mode ("sim" or "real")
        - "movement_type": The action that was performed (move_to_clear_space, move_down, or move_ee_top_down)
        - "error": Error message (only present if result is "failure")
    """
    cmd = f"--mode {mode} --{action.replace('_', '-')}"
    return _run_primitive("move_to_regrasp.py", cmd, timeout=60, error_prefix="Move to regrasp")

@mcp.tool()
def translate_object(action: TranslateAction, mode: Mode = "sim", base_name: Optional[str] = None, object_name: Optional[str] = None, use_default_base_position: bool = False) -> Dict[str, Any]:
    """Translate object to target position.

    Call this tool only if the object is already grasped.
    Moves object to target position for assembly. Maintains object's current orientation. 

    Workflow for assembly:
    1. Call move_to_base to move to the base
    2. Call perform_insert to move down to the final position (Make sure the object orientation is correct before this call)
    3. Call control_gripper to release the object
    4. Call move_to_safe_height to move to the safe height (z=0.3m)

    Workflow for disassembly:
    1. Call move_away_from_base once you are holding the object and in safe height to move the object away from the base
    2. Call control_gripper to release the object

    Args:
        action: Which step to perform (move_to_base, perform_insert, move_to_safe_height, or move_away_from_base)
        mode: Robot mode
        base_name: Required for move_to_base/perform_insert in sim mode
        object_name: Required for move_to_base/perform_insert in sim mode
        use_default_base_position: Use default base position (for real mode)

    Returns:
        Dictionary containing:
        - "result": "success" or "failure"
        - "mode": Robot mode ("sim" or "real")
        - "movement_type": The action that was performed (move_to_base, perform_insert, move_to_safe_height, or move_away_from_base)
        - "object_name": Name of the object (present in sim mode for certain actions)
        - "base_name": Name of the base (present in sim mode for certain actions)
        - "error": Error message (only present if result is "failure")
    """
    # Validate sim mode requirements for certain actions
    if mode == "sim" and action in ["move_to_base", "perform_insert"]:
        if object_name is None:
            return {"output": "Error: object_name is required in sim mode for this action"}
        if base_name is None:
            return {"output": "Error: base_name is required in sim mode for this action"}

    cmd = f"--mode {mode}"
    if object_name is not None:
        cmd += f" --object-name \"{object_name}\""
    if base_name is not None:
        cmd += f" --base-name \"{base_name}\""
    cmd += f" --{action.replace('_', '-')}"
    if use_default_base_position:
        cmd += " --use-default-base-position"

    # Adjust timeout based on action
    if action == "perform_insert":
        timeout = 300
    elif action in ["move_to_safe_height", "move_away_from_base"]:
        timeout = 60
    else:
        timeout = 90

    return _run_primitive("translate_object.py", cmd, timeout=timeout, error_prefix="Translate object")

@mcp.tool()
def rotate_object(object_name: str, base_name: str, mode: Mode = "sim", current_object_orientation: Optional[List[float]] = None, target_base_orientation: Optional[List[float]] = None, use_default_base_orientation: bool = False) -> Dict[str, Any]:
    """Rotate object for assembly.

    Call this tool only if the object is already grasped.
    Rotates object from current to target orientation relative to base orientation.

    Args:
        object_name: Name of the object to rotate (required)
        base_name: Name of the base object (required)
        mode: Robot mode (default: "sim")
        current_object_orientation: Current object orientation quaternion [x, y, z, w]. Required in real mode, optional in sim mode (reads from ROS topic if not provided)
        target_base_orientation: Target base orientation quaternion [x, y, z, w]. Optional in sim mode (reads from ROS topic if not provided), required in real mode unless use_default_base_orientation is True
        use_default_base_orientation: Use default base orientation [0, 0, 0, 1]. Optional, mainly for real mode

    Returns:
        Dictionary containing:
        - "result": "success" or "failure"
        - "object_name": Name of the object
        - "base_name": Name of the base object
        - "mode": Robot mode ("sim" or "real")
        - "movement_type": "rotate_object"
        - "initial_object_orientation": Initial object orientation quaternion (x, y, z, w) - only on success
        - "initial_object_orientation_rpy_deg": Initial object orientation in degrees (roll, pitch, yaw) - only on success
        - "final_object_orientation": Final object orientation quaternion (x, y, z, w) - only on success
        - "final_object_orientation_rpy_deg": Final object orientation in degrees (roll, pitch, yaw) - only on success
        - "initial_end_effector_orientation": Initial end-effector orientation quaternion (x, y, z, w) - only on success
        - "initial_end_effector_orientation_rpy_deg": Initial end-effector orientation in degrees (roll, pitch, yaw) - only on success
        - "final_end_effector_orientation": Final end-effector orientation quaternion (x, y, z, w) - only on success
        - "final_end_effector_orientation_rpy_deg": Final end-effector orientation in degrees (roll, pitch, yaw) - only on success
        - "error": Error message (only present if result is "failure")
    """
    cmd = f"--mode {mode} --object-name \"{object_name}\" --base-name \"{base_name}\""
    if current_object_orientation is not None:
        # Format numbers to avoid scientific notation which can confuse argument parser
        cmd += f" --current-object-orientation {' '.join(f'{x:.10f}'.rstrip('0').rstrip('.') for x in current_object_orientation)}"
    if use_default_base_orientation:
        cmd += " --use-default-base-orientation"
    elif target_base_orientation is not None:
        # Format numbers to avoid scientific notation which can confuse argument parser
        cmd += f" --target-base-orientation {' '.join(f'{x:.10f}'.rstrip('0').rstrip('.') for x in target_base_orientation)}"
    return _run_primitive("rotate_object.py", cmd, timeout=90, error_prefix="Rotate for assembly")

## ############################################################################################## ##
##
##                      ELICITATION TOOL
##
## ############################################################################################## ##

# Elicitation allows MCP servers to pause execution and request structured input from the user.
# This is useful for:
# 1. Confirming potentially dangerous operations before executing
# 2. Collecting additional parameters that weren't provided initially
# 3. Offering choices when multiple valid options exist
# 4. Human-in-the-loop workflows where user decisions are needed mid-execution
#
# Define schemas using Pydantic models - these define what fields the user will be asked to fill
# Note: MCP elicitation supports primitive types: str, int, float, bool, or list[str]
# For enum-style selection, use json_schema_extra={"enum": [...]} on str fields

# class GripperConfirmation(BaseModel):
#     """Schema for confirming gripper operation."""
#     confirm: bool = Field(
#         default=True,
#         description="Confirm you want to proceed with this gripper operation?"
#     )
#     action: str = Field(
#         default="open",
#         description="Select gripper action",
#         json_schema_extra={"enum": ["open", "close", "half-open", "cancel"]}
#     )
#     width_mm: int = Field(
#         default=50,
#         description="Gripper width in mm (0-110), only used for custom width"
#     )

# @mcp.tool()
# async def demo_elicitation_gripper(ctx: Context[ServerSession, None]) -> Dict[str, Any]:
#     """
#     DEMO: Demonstrates MCP elicitation by asking user to confirm gripper operation.
#
#     This tool shows how elicitation works:
#     1. Server pauses and sends a schema to the client
#     2. Client presents a form/dialog to the user
#     3. User fills in the form and submits
#     4. Server receives the structured response and continues
#
#     NOTE: Elicitation requires client support. Claude Desktop/Claude Code may not
#     support elicitation yet (will return "Method not found" error).
#     """
#     try:
#         # Request user input using elicitation
#         # The schema defines what fields the user will see
#         result = await ctx.elicit(
#             message="Please confirm the gripper operation you want to perform:",
#             schema=GripperConfirmation
#         )
#
#         # Handle the response
#         # result.action can be: "accept", "decline", or "cancel"
#         if result.action == "accept" and result.data:
#             data = result.data
#             if not data.confirm:
#                 return {
#                     "status": "cancelled",
#                     "message": "User declined to proceed with gripper operation"
#                 }
#
#             # Literal type ensures action is valid - no runtime validation needed
#             if data.action == "cancel":
#                 return {
#                     "status": "cancelled",
#                     "message": "User selected cancel"
#                 }
#
#             # In a real implementation, you would execute the gripper command here
#             # For demo purposes, we just return what would happen
#             return {
#                 "status": "success",
#                 "message": f"Would execute gripper action: {data.action}",
#                 "parameters": {
#                     "action": data.action,
#                     "width_mm": data.width_mm
#                 },
#                 "note": "This is a demo - no actual gripper movement occurred"
#             }
#
#         elif result.action == "decline":
#             return {
#                 "status": "declined",
#                 "message": "User declined the elicitation request"
#             }
#
#         else:  # cancelled
#             return {
#                 "status": "cancelled",
#                 "message": "Elicitation was cancelled"
#             }
#
#     except Exception as e:
#         error_msg = str(e)
#         if "Method not found" in error_msg:
#             return {
#                 "status": "error",
#                 "message": "Elicitation not supported by this client",
#                 "details": "Claude Desktop and Claude Code don't currently support MCP elicitation. Try using an MCP client that supports elicitation (like the MCP Inspector or custom clients).",
#                 "error": error_msg
#             }
#         return {
#             "status": "error",
#             "message": f"Elicitation failed: {error_msg}"
#         }

if __name__ == "__main__":
    # Start services BEFORE MCP server runs (outside async context)
    _start_services()

    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Close WebSocket connection (rosbridge processes are left running intentionally)
        try:
            ws_manager.close()
        except:
            pass
        logger.info("RosMCP server shut down")