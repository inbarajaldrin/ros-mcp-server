"""
Robot Primitives API for Code as Policies

This module provides a Python API for robot primitives that can be imported
and used in policy code. It wraps the MCP tool implementations to make them
callable as regular Python functions.
"""

import subprocess
import os
import sys
import threading
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

# Get the project root directory (two levels up from primitives/utils/)
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    # Set PYTHONPATH to include project root for imports
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{SCRIPT_DIR}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = SCRIPT_DIR

    cmd_parts = [
        f"cd {SCRIPT_DIR}/primitives",
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
    cmd_parts = [
        f"cd {SCRIPT_DIR}/queries",
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
##                      QUERY FUNCTIONS
##
## ############################################################################################## ##

def get_available_objects(mode: str = "sim") -> Dict[str, Any]:
    """Get list of available object names from ROS topic.

    Args:
        mode: Mode to use - "sim" for simulation (reads from /objects_poses_sim) or "real" for real robot (reads from /objects_poses_real) (default: "sim")

    Returns:
        JSON output containing list of available object names
    """
    return _run_query("get_available_objects.py", f"--mode {mode}", timeout=10, error_prefix="Get available objects")


def get_available_grasp_ids(mode: str = "sim") -> Dict[str, Any]:
    """Get available grasp IDs per object from ROS topic.

    Args:
        mode: Mode to use - "sim" for simulation (reads from /grasp_points_sim) or "real" for real robot (reads from /grasp_points_real) (default: "sim")

    Returns:
        JSON output containing available grasp IDs per object
    """
    return _run_query("get_available_grasp_ids.py", f"--mode {mode}", timeout=10, error_prefix="Get available grasp IDs")


## ############################################################################################## ##
##
##                      PRIMITIVE FUNCTIONS
##
## ############################################################################################## ##

def move_home() -> Dict[str, Any]:
    """Move robot to home position.

    Returns:
        Dictionary with output from the primitive
    """
    return _run_primitive("move_home.py", timeout=45, error_prefix="Move home")


def control_gripper(command: str, mode: str = "sim") -> Dict[str, Any]:
    """Control gripper.

    Supports "open", "close", "half-open" (30mm), or numeric values 0-110 (width in mm).

    Args:
        command: Gripper command - "open", "close", "half-open" (30mm), or numeric value 0-110 (width in mm)
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")

    Returns:
        Dictionary with output from the primitive
    """
    return _run_primitive("control_gripper.py", f"{command} --mode {mode}", timeout=60, error_prefix="Gripper control")


def scan_workspace(object_name: str) -> Dict[str, Any]:
    """Scan workspace at fixed height to locate object.

    This tool scans the workspace by following a predefined path across x,y at a fixed z height.
    The robot moves along the path and stops as soon as the object is detected.
    Only works in real mode (not available for simulation).

    Args:
        object_name: Name of the object to locate

    Returns:
        Dictionary with output from the primitive
    """
    return _run_primitive("scan_workspace.py", f"--object-name \"{object_name}\" --mode real", timeout=300, error_prefix="Scan workspace")


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

    Returns:
        Dictionary with output from the primitive
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
        Dictionary with output from the primitive. Note down the object position and orientation for future use.
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

    Returns:
        Dictionary with output from the primitive
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
    if move_down:
        timeout = 300
    elif move_to_safe_height:
        timeout = 40
    else:
        timeout = 90

    return _run_primitive("translate_object.py", cmd, timeout=timeout, error_prefix="Translate object")


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

    Returns:
        Dictionary with output from the primitive
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
##                      VERIFICATION FUNCTIONS
##
## ############################################################################################## ##

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
