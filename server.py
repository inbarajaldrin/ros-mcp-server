from mcp.server.fastmcp import FastMCP, Image, Context
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field
from typing import Annotated, List, Any, Optional, Union, Literal, Dict
from pathlib import Path
import json
import base64
from utils.websocket_manager import WebSocketManager
import subprocess
import sys
import logging
import re

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

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
TaskType = Literal["assembly", "disassembly"]
GripperCommand = Literal["open", "close", "half-open"]
MoveToGraspAction = Literal["move_to_object", "move_to_safe_height"]
TranslateAction = Literal["move_to_base", "perform_insert", "move_to_safe_height", "place_down"]
PhaseNumber = Literal[0, 1, 2, 3]
PhaseStatus = Literal["success", "failure"]

# Configuration using environment variables with defaults (similar to newer version)
# ROS Bridge connection settings
LOCAL_IP = os.getenv("ROSBRIDGE_LOCAL_IP", "127.0.0.1")  # Default: localhost
ROSBRIDGE_IP = os.getenv("ROSBRIDGE_IP", "127.0.0.1")  # Default: localhost
ROSBRIDGE_PORT = int(os.getenv("ROSBRIDGE_PORT", "9090"))  # Default: rosbridge port

# This is Global WebSocket manager - don't close it after every operation
ws_manager = WebSocketManager(ROSBRIDGE_IP, ROSBRIDGE_PORT, LOCAL_IP)

# Process handles for health monitoring (set by _start_services, checked by _ensure_services_healthy)
_rosbridge_process = None
_grasp_publisher_process = None

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
    global _rosbridge_process, _grasp_publisher_process
    logger.info("RosMCP server starting up")

    # Kill stale processes from previous sessions
    for pattern in ["rosbridge_websocket", "rosapi_node", "grasp_points_publisher.py"]:
        result = subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
        if result.returncode == 0:
            logger.info(f"Cleaned up stale {pattern} processes")
    time.sleep(1)

    # Start rosbridge (start_new_session detaches from parent, DEVNULL fully disconnects)
    _rosbridge_process = subprocess.Popen(
        ["ros2", "launch", "rosbridge_server", "rosbridge_websocket_launch.xml"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    logger.info(f"Started rosbridge server (PID: {_rosbridge_process.pid})")

    # Start grasp publisher
    script_dir = os.path.dirname(os.path.abspath(__file__))
    _grasp_publisher_process = subprocess.Popen(
        ["/usr/bin/python3", f"{script_dir}/utils/grasp_points_publisher.py", "--mode", "default"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    logger.info(f"Started grasp points publisher (PID: {_grasp_publisher_process.pid})")

    time.sleep(2)


def _ensure_services_healthy():
    """Check if rosbridge and grasp publisher are alive; restart if dead.

    Uses process.poll() which is non-blocking — no overhead when processes are healthy.
    """
    global _rosbridge_process, _grasp_publisher_process

    # Check rosbridge
    if _rosbridge_process is not None and _rosbridge_process.poll() is not None:
        exit_code = _rosbridge_process.returncode
        logger.warning(f"rosbridge died (exit code {exit_code}), restarting...")

        # Kill any orphaned child processes
        for pattern in ["rosbridge_websocket", "rosapi_node"]:
            subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)

        # Restart rosbridge
        _rosbridge_process = subprocess.Popen(
            ["ros2", "launch", "rosbridge_server", "rosbridge_websocket_launch.xml"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        logger.info(f"Restarted rosbridge server (PID: {_rosbridge_process.pid})")

        # Force ws_manager to reconnect on next use
        ws_manager.close()
        time.sleep(2)

    # Check grasp publisher
    if _grasp_publisher_process is not None and _grasp_publisher_process.poll() is not None:
        exit_code = _grasp_publisher_process.returncode
        logger.warning(f"grasp_points_publisher died (exit code {exit_code}), restarting...")

        subprocess.run(["pkill", "-9", "-f", "grasp_points_publisher.py"], capture_output=True)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        _grasp_publisher_process = subprocess.Popen(
            ["/usr/bin/python3", f"{script_dir}/utils/grasp_points_publisher.py", "--mode", "default"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        logger.info(f"Restarted grasp points publisher (PID: {_grasp_publisher_process.pid})")
        time.sleep(1)


# Patterns that indicate infrastructure failures (retry-able), not application errors
_CONNECTION_ERROR_PATTERNS = [
    "connection refused", "connection reset", "broken pipe",
    "timed out", "failed to discover", "could not contact",
]


def _is_connection_error(result: dict) -> bool:
    """Check if a tool result indicates a connection/infrastructure failure."""
    for field in ("output", "error"):
        value = result.get(field, "")
        if isinstance(value, str):
            lower = value.lower()
            if any(p in lower for p in _CONNECTION_ERROR_PATTERNS):
                return True
    return False


def _run_with_retry(func, *args, **kwargs) -> Dict[str, Any]:
    """Run a tool function with pre-flight health check and one retry on connection errors.

    1. Runs _ensure_services_healthy() before every call (lightweight poll()).
    2. Executes the tool function.
    3. If result matches a connection error pattern on the first attempt:
       restart services, close ws, sleep 2s, retry once.
    4. On success or application error: return immediately.
    """
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        _ensure_services_healthy()
        result = func(*args, **kwargs)

        if attempt < max_attempts and isinstance(result, dict) and _is_connection_error(result):
            logger.warning(f"Connection error detected (attempt {attempt}), restarting services and retrying...")
            # Force full restart
            _start_services()
            ws_manager.close()
            time.sleep(2)
            continue

        return result

    return result  # Should not reach here, but return last result as fallback


# Initialize MCP (no lifespan - services started in __main__ before mcp.run())
mcp = FastMCP("ros-mcp-server")


## ############################################################################################## ##
##
##                      ROS TOPIC TOOLS
##
## ############################################################################################## ##

@mcp.tool()
def get_topics():
    _ensure_services_healthy()
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
def read_topic(
    topic_name: Annotated[str, Field(description='e.g. /topic_name')],
    timeout: int = 5,
):
    """Read data from any ROS topic."""
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
def execute_python_code(
    code: str,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Execute Python code for calculations and math operations.

    File saving: Use relative paths (e.g., "output.txt") instead of absolute paths like "/tmp/output.txt"."""
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
        
        # Combine all output and strip ANSI color codes from ROS2 logging
        with output_lock:
            output = _ANSI_RE.sub("", "".join(output_lines))

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

        # Return combined stdout and stderr, strip ANSI color codes from ROS2 logging
        output = result.stdout if result.stdout else ""
        if result.stderr:
            output += result.stderr
        output = _ANSI_RE.sub("", output)

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

class GraspPoint(BaseModel):
    id: int
    gripper_states: List[Literal["open", "half-open"]] = Field(description="Gripper state required before grasping this grasp point")
    z_height: Optional[float] = Field(default=None, description="Z position of the grasp point in world frame relative to robot base")

class SceneObject(BaseModel):
    grasps: List[GraspPoint] = Field(description="Available grasp points. Empty if object has none")
    assembly_order: Optional[int] = Field(default=None, description="Present when task_type=assembly")
    disassembly_order: Optional[int] = Field(default=None, description="Present when task_type=disassembly. Boards excluded")

@mcp.tool()
def get_scene_info(
    task_type: TaskType = "assembly",
    mode: Mode = "sim",
) -> Dict[str, SceneObject]:
    """Get scene information with objects and their available grasp points."""
    return _run_with_retry(_run_query, "get_scene_info.py", f"--mode {mode} --task-type {task_type}", timeout=10, error_prefix="Get scene info")

@mcp.tool()
def get_target_object_pose(
    object_name: str,
    base_name: str,
    mode: Mode = "sim",
) -> Dict[str, Any]:
    """Get target object pose relative to the base."""
    cmd = f"--object-name \"{object_name}\" --base-name \"{base_name}\" --mode {mode}"
    return _run_with_retry(_run_query, "get_target_object_pose.py", cmd, timeout=10, error_prefix="Get target object pose")

@mcp.tool()
def get_current_object_pose(
    object_name: Annotated[Optional[str], Field(description="Ignored if all=True")] = None,
    mode: Mode = "sim",
    all: Annotated[bool, Field(description="If True, returns poses for all objects")] = False,
) -> Dict[str, Any]:
    """Get current object pose(s)."""
    if all:
        cmd = f"--all --mode {mode}"
    elif object_name:
        cmd = f"--object-name \"{object_name}\" --mode {mode}"
    else:
        return {"output": "Error: Either object_name or all=True must be provided"}
    return _run_with_retry(_run_query, "get_current_object_pose.py", cmd, timeout=10, error_prefix="Get current object pose")

class GraspVerificationResult(BaseModel):
    result: Literal["success", "failure"]
    object_name: str
    mode: Mode
    error: Optional[str] = None

@mcp.tool()
def verify_grasp(
    object_name: str,
    mode: Mode = "sim",
    grasp_id: Annotated[Optional[int], Field(description="Required for real mode")] = None,
    current_object_orientation: Annotated[Optional[List[float]], Field(description="Quaternion [x, y, z, w] (required for real mode)")] = None,
) -> GraspVerificationResult:
    """Verify if object is within grasp radius from gripper center. Call this tool after moving to safe height."""
    # Build command based on mode
    if mode == "real":
        missing = []
        if grasp_id is None:
            missing.append("grasp_id (get from get_scene_info)")
        if current_object_orientation is None:
            missing.append("current_object_orientation [x,y,z,w] (get from get_current_object_pose)")
        if missing:
            return {"result": "failure", "object_name": object_name, "mode": "real",
                   "error": f"Real mode requires: {', '.join(missing)}"}
        if not isinstance(current_object_orientation, list) or len(current_object_orientation) != 4:
            return {"result": "failure", "object_name": object_name, "mode": "real",
                   "error": "current_object_orientation must be a list of 4 floats [x, y, z, w]"}
        quat_str = " ".join(str(v) for v in current_object_orientation)
        cmd = f"--object-name \"{object_name}\" --mode real --grasp-id {grasp_id} --current-object-orientation {quat_str}"
    elif mode == "sim":
        cmd = f"--object-name \"{object_name}\" --mode sim --radius 0.06"
    else:
        return {"result": "failure", "object_name": object_name,
                "error": f"Invalid mode '{mode}'. Must be 'sim' or 'real'."}

    return _run_with_retry(_run_query, "verify_grasp.py", cmd, timeout=30, error_prefix="Verify grasp")


class PositionError(BaseModel):
    x: float
    y: float
    z: float

class OrientationError(BaseModel):
    roll: float
    pitch: float
    yaw: float

class ObjectOrderEntry(BaseModel):
    name: str
    assembly_order: int

class SingleAssemblyVerification(BaseModel):
    """Return schema when verifying a single object."""
    result: Literal["success", "failure"]
    object_name: str
    base_name: str
    assembly_order: Optional[int] = None
    position_error_m: Optional[PositionError] = None
    orientation_error_deg: Optional[OrientationError] = None
    within_tolerance: Optional[bool] = None
    unassembled_objects: Optional[List[ObjectOrderEntry]] = Field(default=None, description="Sim mode only")
    error: Optional[str] = None

class CheckAllAssemblyVerification(BaseModel):
    """Return schema when check_all=True (sim mode only)."""
    result: Literal["success", "failure"]
    base_name: str
    all_assembled: bool
    assembled_objects: Optional[List[ObjectOrderEntry]] = None
    unassembled_objects: Optional[List[ObjectOrderEntry]] = None
    error: Optional[str] = None

@mcp.tool()
def verify_assembly(
    base_name: str,
    object_name: Annotated[Optional[str], Field(description="Optional if check_all is True")] = None,
    check_all: Annotated[bool, Field(description="Check all objects instead of a specific one (sim mode only)")] = False,
    mode: Mode = "sim",
) -> SingleAssemblyVerification | CheckAllAssemblyVerification:
    """Verify if object(s) are assembled into the base."""
    if check_all and mode == "real":
        return {"result": "failure", "error": "check_all is not supported in real mode. Verify one object at a time."}
    if check_all:
        result = _run_with_retry(_run_query, "verify_assembly.py", f"--base-name \"{base_name}\" --mode {mode} --check-all", timeout=30, error_prefix="Verify assembly")
    elif object_name:
        result = _run_with_retry(_run_query, "verify_assembly.py", f"--object-name \"{object_name}\" --base-name \"{base_name}\" --mode {mode}", timeout=30, error_prefix="Verify assembly")
    else:
        return {"result": "failure", "error": "Either object_name or check_all=True must be specified"}

    # In real mode, don't return assembled_objects (only unassembled matters)
    if mode == "real" and isinstance(result, dict):
        result.pop("assembled_objects", None)

    return result

class DisassemblyViolations(BaseModel):
    """Order violations detected during single-object disassembly verification."""
    skipped_objects: Optional[List[str]] = Field(default=None, description="Objects that should have been disassembled first")
    disturbed_objects: Optional[List[str]] = Field(default=None, description="Lower-order objects knocked out of place")

class SingleDisassemblyVerification(BaseModel):
    """Return schema when verifying a single object's disassembly."""
    result: Literal["success", "failure"]
    object_name: str
    base_name: str
    disassembly_order: Optional[int] = Field(default=None, description="1 = first to remove, reverse of assembly order")
    position_error_m: Optional[PositionError] = None
    orientation_error_deg: Optional[OrientationError] = None
    error: Optional[DisassemblyViolations] = Field(default=None, description="Present only if order violations detected")

class CheckAllDisassemblyVerification(BaseModel):
    """Return schema when check_all=True."""
    result: Literal["success", "failure"]
    base_name: str
    all_disassembled: bool
    disassembled_objects: Optional[List[str]] = None
    still_assembled_objects: Optional[List[str]] = None
    error: Optional[str] = None

@mcp.tool()
def verify_disassembly(
    base_name: str,
    object_name: Annotated[Optional[str], Field(description="Optional if check_all is True")] = None,
    check_all: Annotated[bool, Field(description="Check all objects instead of a specific one")] = False,
    mode: Mode = "sim",
) -> SingleDisassemblyVerification | CheckAllDisassemblyVerification:
    """Verify if object(s) have been disassembled from the base."""
    if check_all:
        return _run_with_retry(_run_query, "verify_disassembly.py", f"--base-name \"{base_name}\" --mode {mode} --check-all", timeout=30, error_prefix="Verify disassembly")
    elif object_name:
        return _run_with_retry(_run_query, "verify_disassembly.py", f"--object-name \"{object_name}\" --base-name \"{base_name}\" --mode {mode}", timeout=30, error_prefix="Verify disassembly")
    else:
        return {"result": "failure", "error": "Either object_name or check_all=True must be specified"}

class ClearanceResult(BaseModel):
    result: Literal["success", "failure"]
    base_name: str
    ready_for_assembly: bool
    error: Optional[str] = None
    missing_objects: Optional[List[str]] = Field(default=None, description="Sim: never missing. Real: operator is prompted to fix.")
    objects_with_clearance_issues: Optional[List[str]] = Field(default=None, description="Sim: agent must call restore scene. Real: operator is prompted to fix.")

@mcp.tool()
async def verify_clearance(
    base_name: str,
    ctx: Context[ServerSession, None],
    mode: Mode = "sim",
) -> ClearanceResult:
    """Verify all objects have enough clearance for the gripper to operate."""
    # Real mode: ask human to confirm scene is ready before checking
    if mode == "real":
        scene_setup = await _invoke_scene_setup(ctx)
        if scene_setup.get("action") == "decline" or scene_setup.get("action") == "cancel":
            return {
                "result": "failure",
                "base_name": base_name,
                "ready_for_assembly": False,
                "error": "Scene setup was declined or cancelled by operator",
                "scene_setup": scene_setup,
            }
        if scene_setup.get("status") == "error":
            return {
                "result": "failure",
                "base_name": base_name,
                "ready_for_assembly": False,
                "error": f"Scene setup elicitation failed: {scene_setup.get('message', 'unknown error')}",
                "scene_setup": scene_setup,
            }

    # Run clearance verification query
    result = _run_with_retry(_run_query, "verify_clearance.py", f"--base-name \"{base_name}\" --mode {mode}", timeout=30, error_prefix="Verify clearance")

    # Real mode: attach scene setup response and offer elicitation to fix failures
    if mode == "real":
        if isinstance(result, dict):
            result["scene_setup"] = scene_setup

        if isinstance(result, dict) and result.get("result") == "failure":
            def retry_query(bn, m):
                return _run_with_retry(_run_query, "verify_clearance.py", f"--base-name \"{bn}\" --mode {m}", timeout=30, error_prefix="Verify clearance")

            result = await handle_clearance_failure(result, base_name, mode, ctx, _handle_elicitation, retry_query)

    return result


async def _invoke_scene_setup(ctx: Context[ServerSession, None]) -> Dict[str, Any]:
    """Invoke scene setup elicitation to confirm real workspace is ready.

    Loads the setup_real_scene elicitation module and presents a form
    asking the human to place objects and spawn the real robot.
    """
    try:
        import importlib.util as _ilu
        script_dir = os.path.dirname(os.path.abspath(__file__))
        setup_path = os.path.join(script_dir, "elicitations", "setup_real_scene.py")

        spec = _ilu.spec_from_file_location("elicitations.setup_real_scene", setup_path)
        module = _ilu.module_from_spec(spec)
        spec.loader.exec_module(module)

        message = module.build_elicitation_message({"phase": 3})
        return await _handle_elicitation(ctx, "setup_real_scene", message, {"phase": 3})

    except Exception as e:
        return {"status": "error", "message": f"Scene setup elicitation failed: {str(e)}"}


## ############################################################################################## ##
##
##                      PRIMITIVES
##
## ############################################################################################## ##

@mcp.tool()
def move_home() -> Dict[str, Any]:
    """Move robot to home position."""
    return _run_with_retry(_run_primitive, "move_home.py", timeout=45, error_prefix="Move home")

class GripperResult(BaseModel):
    result: Literal["success", "failure"]
    command: GripperCommand
    mode: Mode
    initial_width_mm: Optional[float] = None
    final_width_mm: Optional[float] = None
    change_mm: Optional[float] = None
    error: Optional[str] = None

@mcp.tool()
def control_gripper(
    command: Annotated[GripperCommand, Field(description="open = open jaws fully, half-open = 30 mm, close = close jaws to grasp")],
    mode: Mode = "sim",
) -> GripperResult:
    """Control gripper."""
    return _run_with_retry(_run_primitive, "control_gripper.py", f"{command} --mode {mode}", timeout=60, error_prefix="Gripper control")

@mcp.tool()
def scan_workspace(
    object_name: str,
) -> Dict[str, Any]:
    """Scan workspace at fixed height to locate object. Follows a predefined path across x,y and stops as soon as the object is detected."""
    return _run_with_retry(_run_primitive, "scan_workspace.py", f"--object-name \"{object_name}\" --mode real", timeout=300, error_prefix="Scan workspace")

class Quaternion(BaseModel):
    x: float
    y: float
    z: float
    w: float

class RPY(BaseModel):
    roll: float
    pitch: float
    yaw: float

class Orientation(BaseModel):
    quat: Quaternion
    rpy: RPY

class Position(BaseModel):
    x: float
    y: float
    z: float

class MoveToGraspResult(BaseModel):
    result: Literal["success", "failure"]
    object_name: str
    grasp_id: int
    mode: Mode
    movement_type: MoveToGraspAction
    current_object_position: Optional[Position] = None
    current_object_orientation: Optional[Orientation] = None
    error: Optional[str] = None

@mcp.tool()
def move_to_grasp(
    object_name: str,
    grasp_id: int,
    action: Annotated[MoveToGraspAction, Field(description=(
        "move_to_object: Move robotic arm down to the object's grasp point. Gripper must be open before calling. "
        "move_to_safe_height: Lift robotic arm to safe height (z=0.3m) after grasping. Call after closing the gripper."
    ))],
    mode: Mode = "sim",
) -> MoveToGraspResult:
    """Move to grasp position."""
    cmd = f"--object-name \"{object_name}\" --grasp-id {grasp_id} --mode {mode}"
    cmd += f" --{action.replace('_', '-')}"
    return _run_with_retry(_run_primitive, "move_to_grasp.py", cmd, timeout=60, error_prefix="Move to grasp")

class TranslateObjectResult(BaseModel):
    result: Literal["success", "failure"]
    mode: Mode
    movement_type: TranslateAction
    object_name: Optional[str] = None
    base_name: Optional[str] = None
    error: Optional[str] = None

@mcp.tool()
def translate_object(
    action: Annotated[TranslateAction, Field(description=(
        "move_to_base: Move grasped object to hover above the assembly base. Call before perform_insert. "
        "perform_insert: Insert the object downward into the base. Call after move_to_base. Verify object orientation is correct before calling. "
        "move_to_safe_height: Lift robotic arm to safe height (z=0.3m). Call after releasing the object post-insertion. "
        "place_down: Moves laterally to clear region, lowers the arm, and places the object on the table. Used during disassembly or regrasp. Will have to open gripper to release the object."
    ))],
    mode: Mode = "sim",
    object_name: Annotated[Optional[str], Field(description="The object being held. Required for move_to_base and perform_insert only.")] = None,
    base_name: Annotated[Optional[str], Field(description="The assembly base. Required for move_to_base and perform_insert only.")] = None,
    grasp_id: Optional[int] = None,
    current_object_orientation: Annotated[Optional[List[float]], Field(description="Quaternion [x, y, z, w]. Required for real mode only")] = None,
) -> TranslateObjectResult:
    """Translate object to target position. Maintains object's current orientation."""
    # Validate required fields per action
    if action in ["move_to_base", "perform_insert"]:
        missing = []
        if not object_name:
            missing.append("object_name")
        if not base_name:
            missing.append("base_name")
        if missing:
            return {"result": "failure",
                    "error": f"Action '{action}' requires: {', '.join(missing)}"}

    # Verify object is grasped before placing on clear area
    if action == "place_down":
        grasp_result = _run_query("verify_grasp.py", f"--object-name check --mode {mode} --width-only", timeout=15)
        if isinstance(grasp_result, dict) and grasp_result.get("result") == "failure":
            return {"result": "failure", "mode": mode, "movement_type": "place_down",
                    "error": f"Grasp check failed: {grasp_result.get('error', 'gripper not holding object')}"}

    if mode == "real" and action not in ["place_down", "move_to_safe_height"]:
        missing = []
        if grasp_id is None:
            missing.append("grasp_id")
        if current_object_orientation is None:
            missing.append("current_object_orientation")
        if missing:
            return {"result": "failure",
                    "error": f"Real mode requires: {', '.join(missing)}"}

    cmd = f"--mode {mode}"
    if object_name:
        cmd += f" --object-name \"{object_name}\""
    if base_name:
        cmd += f" --base-name \"{base_name}\""
    cmd += f" --{action.replace('_', '-')}"
    if grasp_id is not None:
        cmd += f" --grasp-id {grasp_id}"
    if current_object_orientation is not None:
        cmd += f" --current-object-orientation {' '.join(f'{x:.10f}'.rstrip('0').rstrip('.') for x in current_object_orientation)}"
    if mode == "real":
        cmd += " --use-default-base-position"

    # Adjust timeout based on action
    if action == "perform_insert":
        timeout = 300
    elif action in ["move_to_safe_height", "place_down"]:
        timeout = 60
    else:
        timeout = 90

    return _run_with_retry(_run_primitive, "translate_object.py", cmd, timeout=timeout, error_prefix="Translate object")

class RotateObjectResult(BaseModel):
    result: Literal["success", "failure"]
    object_name: str
    base_name: str
    mode: Mode
    movement_type: Literal["rotate_object"] = "rotate_object"
    initial_object_orientation: Optional[Orientation] = None
    final_object_orientation: Optional[Orientation] = None
    error: Optional[str] = None

@mcp.tool()
def rotate_object(
    object_name: Annotated[str, Field(description="The object being held by the gripper")],
    base_name: Annotated[str, Field(description="The assembly base to rotate relative to")],
    mode: Mode = "sim",
    current_object_orientation: Annotated[Optional[List[float]], Field(description="Quaternion [x, y, z, w]. Required for real mode only")] = None,
) -> RotateObjectResult:
    """Rotates object from current to target orientation relative to base orientation based on fold symmetry of the object."""
    cmd = f"--mode {mode} --object-name \"{object_name}\" --base-name \"{base_name}\""
    if current_object_orientation is not None:
        # Format numbers to avoid scientific notation which can confuse argument parser
        cmd += f" --current-object-orientation {' '.join(f'{x:.10f}'.rstrip('0').rstrip('.') for x in current_object_orientation)}"
    return _run_with_retry(_run_primitive, "rotate_object.py", cmd, timeout=90, error_prefix="Rotate for assembly")

## ############################################################################################## ##
##
##                      TRIGGER TOOLS
##
## ############################################################################################## ##

from triggers.signal_phase_complete import handle_phase_signal
from triggers.pre_assembly_check import handle_clearance_failure

def _verify_all_disassembled(base_name: str, mode: str) -> Dict[str, Any]:
    """Run verify_disassembly with check_all for gating phase completion."""
    return _run_with_retry(_run_query, "verify_disassembly.py", f"--base-name \"{base_name}\" --mode {mode} --check-all", timeout=30, error_prefix="Verify disassembly")

def _verify_all_assembled(base_name: str, mode: str) -> Dict[str, Any]:
    """Run verify_assembly with check_all for gating phase completion."""
    return _run_with_retry(_run_query, "verify_assembly.py", f"--base-name \"{base_name}\" --mode {mode} --check-all", timeout=30, error_prefix="Verify assembly")

@mcp.tool()
async def signal_phase_complete(
    phase: Annotated[PhaseNumber, Field(description="0=grasp point discovery, 1=disassembly discovery, 2=assembly discovery, 3=assembly execution (sim/real)")],
    status: PhaseStatus,
    ctx: Context[ServerSession, None],
    comment: Annotated[str, Field(description="Should explain failure reasons")] = "",
    mode: Mode = "sim",
    base_name: Annotated[str, Field(description="Assembly base name")] = "",
) -> Dict[str, Any]:
    """Signal completion of an assembly phase to the MCP client."""
    return await handle_phase_signal(
        phase=phase,
        status=status,
        comment=comment,
        ctx=ctx,
        elicit_user_fn=_handle_elicitation,
        mode=mode,
        base_name=base_name,
        verify_disassembly_fn=_verify_all_disassembled,
        verify_assembly_fn=_verify_all_assembled,
    )

class SignalOperatorResult(BaseModel):
    result: Literal["success", "failure"]
    action: str = Field(description="Operator decision: 'proceed' or 'abort'")
    reason: str
    feedback: Optional[str] = None

@mcp.tool()
async def signal_operator(
    message: Annotated[str, Field(description="Description of what the robot has done and what the operator needs to do before the robot can continue.")],
    ctx: Context[ServerSession, None],
    reason: Annotated[str, Field(description='Short tag for the client to categorize the signal.')] = "",
    mode: Mode = "sim",
) -> SignalOperatorResult:
    """Signal a human operator and wait for confirmation before proceeding."""
    if mode == "sim":
        return {
            "result": "success",
            "action": "proceed",
            "reason": reason,
        }

    response = await _handle_elicitation(ctx, "signal_operator", message, {"message": message, "reason": reason})

    if response.get("action") == "accept":
        user_action = response.get("user_action", "proceed")
        feedback = response.get("data", {}).get("feedback", "")
        return {
            "result": "success" if user_action == "proceed" else "failure",
            "action": user_action,
            "reason": reason,
            "feedback": feedback,
        }

    return {
        "result": "failure",
        "action": response.get("action", "cancel"),
        "reason": reason,
        "feedback": response.get("message", "Operator declined or cancelled"),
    }

## ############################################################################################## ##
##
##                      ELICITATION HANDLER
##
## ############################################################################################## ##

async def _handle_elicitation(ctx: Context[ServerSession, None], elicitation_script: str, message: str, context_data: dict = None) -> Dict[str, Any]:
    """Handle an elicitation by loading its schema and presenting a form to the user.

    Dynamically loads schema from elicitation script and presents user with a form to fill.

    Args:
        elicitation_script: Name of the elicitation script (e.g., "setup_real_scene")
        message: The message to display to the user
        context_data: Optional context passed to get_elicitation_schema for dynamic schema selection

    Returns:
        Dictionary with user's response (action: accept/decline/cancel, data: form fields if accepted)
    """
    try:
        # Import the elicitation script to get the schema class
        import importlib.util
        script_dir = os.path.dirname(os.path.abspath(__file__))
        elicitation_path = os.path.join(script_dir, f"elicitations/{elicitation_script}.py")

        if not os.path.exists(elicitation_path):
            return {
                "status": "error",
                "message": f"Elicitation script not found: {elicitation_script}",
                "error": f"Path: {elicitation_path}"
            }

        # Load the module
        spec = importlib.util.spec_from_file_location(f"elicitations.{elicitation_script}", elicitation_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the schema class
        if not hasattr(module, "get_elicitation_schema"):
            return {
                "status": "error",
                "message": f"Elicitation script missing get_elicitation_schema function: {elicitation_script}"
            }

        schema_class = module.get_elicitation_schema(context_data)

        # Request user input using elicitation
        result = await ctx.elicit(
            message=message,
            schema=schema_class
        )

        # Handle the response
        if result.action == "accept" and result.data:
            return {
                "action": "accept",
                "user_action": result.data.action if hasattr(result.data, 'action') else None,
                "data": result.data.model_dump() if hasattr(result.data, 'model_dump') else vars(result.data)
            }
        elif result.action == "decline":
            return {
                "action": "decline",
                "message": "User declined the elicitation"
            }
        else:  # cancelled
            return {
                "action": "cancel",
                "message": "Elicitation was cancelled"
            }

    except Exception as e:
        error_msg = str(e)
        if "Method not found" in error_msg:
            return {
                "status": "error",
                "message": "Elicitation not supported by this client",
                "details": "This client may not support MCP elicitation. Try using an MCP client that supports elicitation (like the MCP Inspector or custom clients).",
                "error": error_msg
            }
        return {
            "status": "error",
            "message": f"Elicitation failed: {error_msg}",
            "error": str(e)
        }

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