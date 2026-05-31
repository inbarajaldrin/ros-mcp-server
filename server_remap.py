from mcp.server.fastmcp import FastMCP, Image, Context
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field
from typing import Annotated, List, Any, Optional, Union, Literal, Dict
from pathlib import Path
import json
import base64
from utils.websocket_manager import WebSocketManager
# Observe-only telemetry (Track B baseline). Guarded so an import problem can
# never break the server. No-op unless MCP_CLIENT_OUTPUT_DIR is set (i.e. in a run).
try:
    from primitives.shared import instrument as _instr
except Exception:  # pragma: no cover - defensive
    class _InstrStub:
        @staticmethod
        def timed_primitive(fn):
            return fn

        @staticmethod
        def child_env():
            return {}
    _instr = _InstrStub()
import subprocess
import sys
import logging
import re

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

#camera
import time
import os
import glob
import functools
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
GripperCommand = str  # "open", "close", or numeric mm value (e.g. "35")
TranslateAction = Literal["insert", "place_down"]
PhaseNumber = Literal[1, 2, 3]
PhaseStatus = Literal["success", "failure"]

# ── Mode 2 real→sim redirect ────────────────────────────────────────────────
# All phases run in sim. Phase 3 YAML says mode='real' but execution is sim.
# Tool signatures use sim-mode parameters (no current_object_orientation).
def _remap_mode(mode: str) -> str:
    """Redirect real→sim. Agent sees real, primitives execute sim."""
    if mode == "real":
        return "sim"
    return mode

# ── Assembly-order precondition gate ─────────────────────────────────────────
# Rejects arm-committing actions on an object whose assembly-order predecessors
# are not yet seated, BEFORE any motion. The interlocking FMB geometry makes
# order a hard precondition (a later part blocks access to an earlier slot), so
# an out-of-order pick/insert wastes the whole run thrashing. The order is NOT
# something the agent discovers — it is injected via the assembly://{id}/objects
# resource — so enforcing it removes no measured signal; it only stops waste.
#
# Toggle: MCP_ORDER_GATE=0 disables (read at CALL time, so a relaunched server
# with the env unset/0 truly runs the ungated ablation arm).
# Scope: sim only — verify_assembly --check-all is unsupported in real mode
# (queries/verify_assembly.py rejects it), so the seated-set oracle is sim-only.
# The real-mode path no-ops here and is handled separately.
def _order_gate_on() -> bool:
    return os.getenv("MCP_ORDER_GATE", "1") != "0"

@functools.lru_cache(maxsize=1)
def _load_order_maps_raw():
    """Load order maps from disk ONCE (gate-independent -> safe to cache).

    Returns (by_base, by_obj, dups, n_files, errors):
      by_base = {base_name: [object_name in assembly order]}
      by_obj  = {object_name: base_name}   (base inference from object_name)
      dups    = sorted object names appearing under >1 base
      n_files = count of fmb*_assembly.json matched by glob
      errors  = list of "filename: reason" for files that failed to parse

    Source of truth: ros-mcp-server/ablations/eval_resources/fmb*_assembly.json
    (same files primitives/shared/config.py reads for gripper widths). This caches
    only the file load; it makes NO gate-policy decision, so a gate-off first call
    cannot poison a later gate-on call's fail-closed validation (done in _order_maps)."""
    by_base = {}
    by_obj = {}
    dups = set()
    errors = []
    pattern = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ablations", "eval_resources", "fmb*_assembly.json")
    files = sorted(glob.glob(pattern))
    for path in files:
        try:
            j = json.loads(open(path).read())
            base = j["base_name"]
            names = [s["object_name"] for s in sorted(j["assembly_order"],
                                                       key=lambda s: s["position"])]
        except Exception as e:
            errors.append(f"{os.path.basename(path)}: {e}")
            continue
        by_base[base] = names
        for n in names:
            if n in by_obj and by_obj[n] != base:
                dups.add(n)
            by_obj[n] = base
    return by_base, by_obj, sorted(dups), len(files), errors

def _order_maps():
    """Return (by_base, by_obj). Validates FAIL-CLOSED every call when the gate is
    on (validation is intentionally OUTSIDE the lru_cache so a prior gate-off call
    cannot skip it). A RuntimeError here is turned by FastMCP into a clean ToolError
    the agent sees; the server stays up. Fails closed when, with the gate on:
      - no maps loaded at all, OR
      - any matched fmb*_assembly.json failed to parse (partial load -> a base could
        be silently ungated), OR
      - an object name is ambiguous across assemblies (base inference unsafe)."""
    by_base, by_obj, dups, n_files, errors = _load_order_maps_raw()
    if _order_gate_on():
        if not by_base:
            raise RuntimeError("MCP_ORDER_GATE on but no assembly order maps loaded "
                               "from eval_resources/fmb*_assembly.json")
        if errors:
            raise RuntimeError(f"MCP_ORDER_GATE on but {len(errors)} of {n_files} "
                               f"assembly file(s) failed to load (a base could be "
                               f"silently ungated): {errors}")
        if dups:
            raise RuntimeError(f"MCP_ORDER_GATE on but object names are ambiguous "
                               f"across assemblies (base inference unsafe): {dups}")
    return by_base, by_obj

_ORDER_UNKNOWN_WARNED = set()  # one-time warn per unknown object (gate-on liveness audit)

def _seated_name(e):
    """Object name from a verify_assembly assembled_objects element, tolerating
    {'name':...}, {'object_name':...}, or a bare string. None if no name key."""
    if isinstance(e, dict):
        return e.get("name") or e.get("object_name")
    return e

def _assert_assembly_order(object_name, mode, base_name=None):
    """Precondition gate. Return a {result:failure,error} dict to BLOCK the action,
    or None to proceed. No-op when the gate is off or in real mode (real mode
    verify_assembly --check-all is unsupported, so the seated-set oracle is sim-only).

    base_name is inferred from object_name when not supplied (object names are
    disjoint across base1/2/3; ambiguity fails closed in _order_maps).

    Decision order:
      - gate off / real mode            -> None (proceed)
      - object/base unknown to the maps -> None (proceed) + one-time WARN (audit)
      - target is order 1               -> None (nothing precedes)
      - oracle unavailable              -> BLOCK with a distinct 'gate unavailable'
                                           error (fail CLOSED: never let an
                                           unverifiable action be recorded as gated)
      - target already seated           -> None (removal/regrasp, not an assembly pick;
                                           makes the gate correct in disassembly +
                                           dirty-scene recovery without a phase flag)
      - a predecessor not seated        -> BLOCK with an order-violation error
      - all predecessors seated         -> None (proceed)"""
    if not _order_gate_on() or _remap_mode(mode) != "sim":
        return None
    by_base, by_obj = _order_maps()
    base = base_name or by_obj.get(object_name)
    order = by_base.get(base)
    if not order or object_name not in order:
        # Unknown object/base -> proceed (liveness), but surface once so a gate-on
        # run that silently let actions through is auditable in the logs.
        if object_name not in _ORDER_UNKNOWN_WARNED:
            _ORDER_UNKNOWN_WARNED.add(object_name)
            logger.warning(f"[order-gate] object '{object_name}' (base={base}) not in any "
                           f"assembly order map -- NOT gated. Known bases: {list(by_base.keys())}")
        return None
    k = order.index(object_name)
    if k == 0:
        return None  # first object, nothing precedes
    res = _verify_all_assembled(base, "sim")
    if not isinstance(res, dict) or not isinstance(res.get("assembled_objects"), list):
        # Oracle unavailable (timeout/wedge/malformed) -> FAIL CLOSED. Blocking
        # visibly is correct for data integrity: an action we cannot verify must
        # not be silently recorded as having passed a live gate. Distinct error
        # so it is never confused with a real order violation.
        return {"result": "failure",
                "error": (f"Assembly order gate unavailable: verify_assembly returned no "
                          f"assembled_objects for base '{base}' (got: {str(res)[:200]}). "
                          f"Cannot confirm predecessors of '{object_name}' are seated; "
                          f"action blocked. Retry once the scene/oracle is responsive.")}
    seated = {n for n in (_seated_name(e) for e in res["assembled_objects"]) if n}
    if object_name in seated:
        return None  # already seated -> removal/regrasp, not an assembly pick
    missing = [o for o in order[:k] if o not in seated]
    if missing:
        return {"result": "failure",
                "error": (f"Assembly order violation: '{object_name}' is order {k + 1} "
                          f"but predecessor(s) {missing} are not seated yet. Seat "
                          f"'{missing[0]}' next. Required order for {base}: {order}.")}
    return None

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


def _wait_for_rosbridge(timeout: int = 10):
    """Poll until rosbridge accepts a websocket connection or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ws_manager.connect()
            logger.info("rosbridge ready (ws connected)")
            return
        except Exception:
            time.sleep(0.5)
    logger.warning(f"rosbridge not ready after {timeout}s — proceeding anyway")


def _run_with_retry(func, *args, **kwargs) -> Dict[str, Any]:
    """Run a tool function with pre-flight health check and one retry on connection errors.

    1. Runs _ensure_services_healthy() before every call (lightweight poll()).
    2. Executes the tool function.
    3. If result matches a connection error pattern on the first attempt:
       restart services, verify rosbridge readiness, retry once.
    4. On success or application error: return immediately.
    """
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        _ensure_services_healthy()
        result = func(*args, **kwargs)

        if attempt < max_attempts and isinstance(result, dict) and _is_connection_error(result):
            # Don't retry subprocess timeouts — they indicate a stuck process, not a connection error.
            # Retrying just doubles the wait (90s → 207s) with no chance of success.
            rc = result.get("returncode")
            if rc in (124, -9):
                logger.warning(f"Subprocess timed out (rc={rc}), skipping retry")
                return result
            logger.warning(f"Connection error detected (attempt {attempt}), restarting services and retrying...")
            # Force full restart
            _start_services()
            ws_manager.close()
            _wait_for_rosbridge(timeout=10)
            continue

        return result

    return result  # Should not reach here, but return last result as fallback


def _parse_result(model_cls, raw: dict):
    """Parse primitive output into a Pydantic result model.

    Safety net: if the primitive returned no 'result' field (e.g. empty dict
    after a failed retry), return a proper failure dict instead of crashing
    with a Pydantic validation error.
    """
    filtered = {k: v for k, v in raw.items() if k in model_cls.model_fields}
    if "result" not in filtered:
        error_msg = raw.get("output", raw.get("error", "Infrastructure error — primitive returned no result, retry the call"))
        logger.warning(f"Primitive returned no result field (raw keys: {list(raw.keys())}), returning failure")
        return {"result": "failure", "error": str(error_msg)[:500]}
    return model_cls(**filtered).model_dump(exclude_none=True)


# Initialize MCP (no lifespan - services started in __main__ before mcp.run())
mcp = FastMCP("ros-mcp-server")


## ############################################################################################## ##
##
##                      ROS TOPIC TOOLS
##
## ############################################################################################## ##

# TODO: Add MCP tool annotations (readOnlyHint, destructiveHint, etc.) to all tools.
# replay_verify.py can then filter out read-only tools during replay instead of
# hardcoding a skip-list. Query tools like get_scene_info, verify_clearance,
# verify_assembly, verify_grasp should be marked readOnlyHint=True.
#
# Example:
#   @mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False})
#   def get_scene_info(...):
#
# Then in replay_verify.py:
#   if tool_annotations.get("readOnlyHint"): continue  # skip queries during replay

@mcp.tool()
def get_topics():
    """List all active ROS topics.

    Returns:
        topics: list of topic names (e.g. ["/joint_states", "/tf"])
        types: list of message types (e.g. ["sensor_msgs/msg/JointState", "tf2_msgs/msg/TFMessage"])"""
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
    """Read data from any ROS topic.

    Returns:
        timestamp: ISO timestamp of the read attempt
        topic: the topic name requested
        status: "success", "timeout", or "error"
        message_data: raw topic message content (only on success)
        error: error description (only on failure)"""
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

class ExecutePythonResult(BaseModel):
    output: str = Field(description="Combined stdout and stderr from execution")
    files_created: Optional[List[str]] = Field(default=None, description="Filenames created during execution")
    files_location: Optional[str] = Field(default=None, description="Directory where created files are saved")

@mcp.tool()
def execute_python_code(
    code: Annotated[str, Field(description="Python code to execute. Has access to math, numpy, datetime, json, sys.")],
    timeout: Annotated[int, Field(description="Max execution time in seconds")] = 30,
) -> ExecutePythonResult:
    """Execute Python code for calculations, data processing, and math operations. Use relative paths for file saving.

    Returns:
        output: combined stdout and stderr from execution
        files_created: filenames created during execution (if any)
        files_location: directory where created files are saved (if any)"""
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

        # Return combined stdout and stderr (truncated to prevent context overflow)
        MAX_OUTPUT_CHARS = 8000
        output = result.stdout if result.stdout else ""
        if result.stderr:
            output += result.stderr

        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n     ...(truncated at {MAX_OUTPUT_CHARS} chars, total was {len(output)})"

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

@_instr.timed_primitive
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
    # Mode-aware config: surface the primitive's --mode to the subprocess as
    # ROS_MCP_MODE so config.py picks the sim/real GRIPPER_CENTER_TOOL_OFFSET and
    # gates real-only base calibration at import. (Primitives bind config
    # constants at import, so this must be in the env before the process starts.)
    import re as _re
    _mode_match = _re.search(r"--mode\s+(\w+)", command_args or "")
    if _mode_match:
        env['ROS_MCP_MODE'] = _mode_match.group(1).strip().lower()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{script_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = script_dir

    # Observe-only: push INSTRUMENT_CALL_ID + INSTRUMENT_SIDECAR into the subprocess
    # so primitive/IK telemetry is tagged with this call. No-op when disabled.
    env.update(_instr.child_env())

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


def _snapshot_robot_state(command_args):
    """Observe-only: read joints + gripper + EE pose for the before/after snapshot.

    Registered as instrument's snapshot hook so the timing decorator can bracket
    every tool call with robot state WITHOUT touching the primitives. Returns the
    parsed dict or None (fail-loud — instrument records "MISSING" on None).
    Reads the mode from the primitive's --mode arg; defaults to sim.
    """
    try:
        m = re.search(r"--mode\s+(\w+)", command_args or "")
        mode = (m.group(1).strip().lower() if m else "sim")
        if mode not in ("sim", "real"):
            mode = "sim"
        raw = _run_query("get_robot_state.py", f"--mode {mode}", timeout=6, error_prefix="Robot state")
        if isinstance(raw, dict) and "joints_rad" in raw:
            return raw
        return None
    except Exception:
        return None


# Wire the snapshot hook into the observe-only instrument (no-op if disabled).
try:
    _instr.register_snapshot_hook(_snapshot_robot_state)
except Exception:
    pass


## ############################################################################################## ##
##
##                      QUERIES
##
## ############################################################################################## ##

@mcp.tool()
def get_scene_info(
    object_name: str,
    base_name: str,
    mode: Mode,
) -> Dict[str, Any]:
    """Get complete info for one object: live grasp points, current pose, and target pose with all valid orientations.

    Call once per object before manipulating it. Re-call after rotation or placement to get updated z_heights and pose.

    Returns:
        grasps: list of grasp points, each with id (int, pass to move_to_grasp/verify_grasp/translate_object), position {x, y, z}, orientation {x, y, z, w} — live poses in world frame
        current_pose: position {x, y, z} and orientation {quat: {x, y, z, w}}
        target_pose: position {x, y, z} and valid_orientations (list of {quat: {x, y, z, w}}) — all fold-symmetric equivalents"""
    mode = _remap_mode(mode)
    cmd = f"--object-name \"{object_name}\" --base-name \"{base_name}\" --mode {mode}"
    return _run_with_retry(_run_query, "get_scene_info.py", cmd, timeout=10, error_prefix="Get scene info")

class GraspVerificationResult(BaseModel):
    result: Literal["success", "failure"]
    object_name: str
    mode: Mode
    error: Optional[str] = None

@mcp.tool()
def verify_grasp(
    object_name: str,
    mode: Mode,
    grasp_id: Annotated[Optional[int], Field(description="Grasp point ID from get_scene_info.")] = None,
) -> GraspVerificationResult:
    """Verify if the object is grasped.

    Returns:
        result: "success" or "failure"
        object_name: the object checked
        mode: "sim" or "real"
        error: failure reason (only on failure)"""
    mode = _remap_mode(mode)
    if grasp_id is None:
        return {"result": "failure", "object_name": object_name, "mode": mode,
               "error": "grasp_id is required — pass the grasp point ID used during move_to_grasp"}
    cmd = f"--object-name \"{object_name}\" --mode sim --radius 0.06 --grasp-id {grasp_id}"

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
    mode: Mode,
    object_name: Annotated[Optional[str], Field(description="Optional if check_all is True")] = None,
    check_all: Annotated[bool, Field(description="Check all objects instead of a specific one (sim mode only)")] = False,
) -> SingleAssemblyVerification | CheckAllAssemblyVerification:
    """Verify if object(s) are assembled into the base.

    Returns:
        Single object: result ("success"/"failure"), object_name, base_name, assembly_order, position_error_m {x,y,z}, orientation_error_deg {roll,pitch,yaw}, within_tolerance (bool), unassembled_objects (list), error
        check_all=True: result, base_name, all_assembled (bool), assembled_objects (list), unassembled_objects (list), error"""
    mode = _remap_mode(mode)
    if check_all:
        result = _run_with_retry(_run_query, "verify_assembly.py", f"--base-name \"{base_name}\" --mode {mode} --check-all", timeout=30, error_prefix="Verify assembly")
    elif object_name:
        result = _run_with_retry(_run_query, "verify_assembly.py", f"--object-name \"{object_name}\" --base-name \"{base_name}\" --mode {mode}", timeout=30, error_prefix="Verify assembly")
    else:
        return {"result": "failure", "error": "Either object_name or check_all=True must be specified"}

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
    mode: Mode,
    object_name: Annotated[Optional[str], Field(description="Optional if check_all is True")] = None,
    check_all: Annotated[bool, Field(description="Check all objects instead of a specific one")] = False,
) -> SingleDisassemblyVerification | CheckAllDisassemblyVerification:
    """Verify if object(s) have been disassembled from the base.

    Returns:
        Single object: result ("success"/"failure"), object_name, base_name, disassembly_order (1=first to remove), position_error_m {x,y,z}, orientation_error_deg {roll,pitch,yaw}, error (order violations with skipped_objects and disturbed_objects)
        check_all=True: result, base_name, all_disassembled (bool), disassembled_objects (list), still_assembled_objects (list), error"""
    mode = _remap_mode(mode)
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
    mode: Mode,
) -> ClearanceResult:
    """Verify all objects have enough clearance for the gripper to operate.

    Returns:
        result: "success" or "failure"
        base_name: the assembly base checked
        ready_for_assembly: bool indicating if all objects have clearance
        error: failure reason (only on failure)
        missing_objects: objects not found (real mode: operator prompted to fix)
        objects_with_clearance_issues: objects too close together (sim: restore scene; real: operator prompted)"""
    mode = _remap_mode(mode)
    result = _run_with_retry(_run_query, "verify_clearance.py", f"--base-name \"{base_name}\" --mode {mode}", timeout=30, error_prefix="Verify clearance")
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

class MoveHomeResult(BaseModel):
    result: Literal["success", "failure"]
    error: Optional[str] = None

@mcp.tool()
def move_home(mode: Mode) -> MoveHomeResult:
    """Move robot to home position.

Returns:
    result: "success" or "failure"
    error: failure reason (only on failure)"""
    mode = _remap_mode(mode)
    raw = _run_with_retry(_run_primitive, "move_home.py", f"--mode {mode}", timeout=45, error_prefix="Move home")
    return _parse_result(MoveHomeResult, raw)

class GripperResult(BaseModel):
    result: Literal["success", "failure"]
    error: Optional[str] = None

@mcp.tool()
def control_gripper(
    command: Annotated[GripperCommand, Field(description="-")],
    mode: Mode,
) -> GripperResult:
    """Control gripper finger width.

Commands:
- open = fully open gripper.
- close = close fingers on object.
- Numeric (e.g. 35) = open to that width in mm.

Returns:
    result: "success" or "failure"
    error (only on failure):
        "Gripper jammed: fingers are asymmetric." — indicate collisions between the gripper and the object meaning the grasp id is not reachable in the objects current orientation.
        "Gripper already holding an object."
        "Gripper blocked: pressing against object its holding, cannot reach target width." — open gripper to release the object"""
    mode = _remap_mode(mode)
    raw = _run_with_retry(_run_primitive, "control_gripper.py", f"{command} --mode {mode}", timeout=60, error_prefix="Gripper control")
    return _parse_result(GripperResult, raw)

class ScanWorkspaceResult(BaseModel):
    result: Literal["success", "failure"]
    error: Optional[str] = None

@mcp.tool()
def scan_workspace(
    object_name: Annotated[str, Field(description="-")],
    mode: Mode = "real",
) -> ScanWorkspaceResult:
    """Scan workspace at fixed height to locate an object. Follows a predefined path across x,y
and stops when the object is detected.

Returns:
    result: "success" or "failure"
    error: failure reason (only on failure)"""
    mode = _remap_mode(mode)
    raw = _run_with_retry(_run_primitive, "scan_workspace.py", f"--object-name \"{object_name}\" --mode {mode}", timeout=300, error_prefix="Scan workspace")
    return _parse_result(ScanWorkspaceResult, raw)

class Quaternion(BaseModel):
    x: float
    y: float
    z: float
    w: float

class Orientation(BaseModel):
    quat: Quaternion

class Position(BaseModel):
    x: float
    y: float
    z: float

class MoveToGraspResult(BaseModel):
    result: Literal["success", "failure"]
    current_object_position: Optional[Position] = None
    current_object_orientation: Optional[Orientation] = None
    error: Optional[str] = None

@mcp.tool()
def move_to_grasp(
    object_name: Annotated[str, Field(description="-")],
    grasp_id: Annotated[int, Field(description="-")],
    mode: Mode,
) -> MoveToGraspResult:
    """Move robotic arm to grasp an object.

Sequence:
1. "control_gripper" to gripper_width_mm before move_to_grasp.
2. "move_to_grasp" to move arm to grasp point.
3. "control_gripper" to grasp the object.
4. "move_to_safe_height" to lift arm after grasping.

Returns:
    result: "success" or "failure"
    current_object_position: {x, y, z} after movement
    current_object_orientation: {quat: {x, y, z, w}} after movement
    error: failure reason (only on failure)"""
    mode = _remap_mode(mode)
    gate_err = _assert_assembly_order(object_name, mode)  # base inferred from object_name
    if gate_err:
        return gate_err
    cmd = f"--object-name \"{object_name}\" --grasp-id {grasp_id} --mode {mode}"
    raw = _run_with_retry(_run_primitive, "move_to_grasp.py", cmd, timeout=60, error_prefix="Move to grasp")
    return _parse_result(MoveToGraspResult, raw)

class TranslateObjectResult(BaseModel):
    result: Literal["success", "failure"]
    error: Optional[str] = None

@mcp.tool()
def translate_object(
    action: Annotated[TranslateAction, Field(description="-")],
    object_name: Annotated[str, Field(description="-")],
    base_name: Annotated[str, Field(description="-")],
    mode: Mode,
) -> TranslateObjectResult:
    """Translate a grasped object to a target position.

Actions:
- "insert" — Move object above the base then insert object into the base.
- "place_down" — Move object laterally to clear region and place object on table.

Sequence:
- insert
  1. Object must be grasped and the robotic arm must be at safe height.
  2. Object must be at target orientation — call rotate_object to correct before inserting.
  3. Perform insert.
  4. Release object using control_gripper to gripper_width_mm.
  5. Call move_to_safe_height to retract the arm.
- place_down
  1. Object must be grasped and the robotic arm must be at safe height.
  2. Perform place_down.
  3. Release object using control_gripper to gripper_width_mm.
  4. Call move_to_safe_height to retract the arm.

Note: When performing insert, if rotate_object rotated the object by more than one axis,
there is a good chance the current orientation of the robotic arm would result in a motion
planning failure. In those cases a regrasp is required — place_down the object and regrasp
it, which retains the correct object orientation while resetting the robotic arm to a
top-down orientation.

Returns:
    result: "success" or "failure"
    error (only on failure):
        "Motion planning failed: no collision-free path to the target position"
            — No action was performed. Solver could not find a collision-free trajectory
              from current robotic arm pose. The arm is still holding the object.
        "Grasp check failed before {movement_type}: {err}"
            — Object is not grasped.
        "Object orientation error is {X} degrees (tolerance: {Y} degrees)."
            — Object needs rotation before insert."""
    mode = _remap_mode(mode)
    if action == "insert" and not base_name:
        return {"result": "failure",
                "error": f"Action '{action}' requires: base_name"}

    if action == "insert":
        gate_err = _assert_assembly_order(object_name, mode, base_name)
        if gate_err:
            return gate_err

    cmd = f"--mode sim --object-name \"{object_name}\""
    if base_name:
        cmd += f" --base-name \"{base_name}\""
    cmd += f" --{action.replace('_', '-')}"

    # Adjust timeout based on action
    if action == "insert":
        timeout = 300
    else:
        timeout = 60

    raw = _run_with_retry(_run_primitive, "translate_object.py", cmd, timeout=timeout, error_prefix="Translate object")
    return _parse_result(TranslateObjectResult, raw)

class RotateObjectResult(BaseModel):
    result: Literal["success", "failure"]
    initial_object_orientation: Optional[Orientation] = None
    final_object_orientation: Optional[Orientation] = None
    error: Optional[str] = None

@mcp.tool()
def rotate_object(
    object_name: Annotated[str, Field(description="-")],
    base_name: Annotated[str, Field(description="-")],
    mode: Mode,
) -> RotateObjectResult:
    """Rotates object from current to target orientation relative to current base orientation
based on fold symmetry of the object by moving the robotic arm.

Prerequisites:
- Object must be grasped and the robotic arm must be at safe height.

Returns:
    result: "success" or "failure"
    initial_object_orientation: {quat: {x, y, z, w}} before rotation
    final_object_orientation: {quat: {x, y, z, w}} after rotation
    error: failure reason (only on failure)"""
    mode = _remap_mode(mode)
    gate_err = _assert_assembly_order(object_name, mode, base_name)
    if gate_err:
        return gate_err
    # Verify grasp as a separate subprocess before rotate_object to avoid
    # nested DDS participant churn that causes discovery race conditions.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vg_script = os.path.join(script_dir, 'queries', 'verify_grasp.py')
    vg_cmd = [sys.executable, vg_script, '--object-name', object_name, '--mode', 'sim']
    try:
        vg_proc = subprocess.run(vg_cmd, capture_output=True, text=True, timeout=15,
                                 env={**os.environ, 'PYTHONPATH': f"{script_dir}:{os.environ.get('PYTHONPATH', '')}"})
    except subprocess.TimeoutExpired:
        return _parse_result(RotateObjectResult, {"result": "failure", "error": "Grasp verification timed out"})
    if vg_proc.returncode != 0:
        vg_out = (vg_proc.stdout or '') + (vg_proc.stderr or '')
        error = "Grasp verification failed - object may not be grasped"
        if '__RESULT_JSON__' in vg_out and '__END_RESULT_JSON__' in vg_out:
            try:
                json_str = vg_out[vg_out.rfind('__RESULT_JSON__') + len('__RESULT_JSON__'):vg_out.rfind('__END_RESULT_JSON__')].strip()
                vj = json.loads(json_str)
                error = vj.get('error', error)
            except (json.JSONDecodeError, ValueError):
                pass
        return _parse_result(RotateObjectResult, {"result": "failure", "error": error})

    cmd = f"--mode sim --object-name \"{object_name}\" --base-name \"{base_name}\" --skip-verify-grasp"
    raw = _run_with_retry(_run_primitive, "rotate_object.py", cmd, timeout=90, error_prefix="Rotate for assembly")
    return _parse_result(RotateObjectResult, raw)

class MoveToSafeHeightResult(BaseModel):
    result: Literal["success", "failure"]
    error: Optional[str] = None

@mcp.tool()
def move_to_safe_height(
    mode: Mode,
) -> MoveToSafeHeightResult:
    """Lift robotic arm to safe height (z=0.3m).

Returns:
    result: "success" or "failure"
    error: failure reason (only on failure)"""
    raw = _run_with_retry(_run_primitive, "move_to_safe_height.py", "", timeout=45, error_prefix="Move to safe height")
    return _parse_result(MoveToSafeHeightResult, raw)

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
    phase: Annotated[PhaseNumber, Field(description="Phase number")],
    status: PhaseStatus,
    ctx: Context[ServerSession, None],
    mode: Mode,
    comment: Annotated[str, Field(description="Should explain failure reasons")] = "",
    base_name: Annotated[str, Field(description="Assembly base name")] = "",
) -> Dict[str, Any]:
    """Signal completion of an assembly phase to the MCP client.

    Returns:
        phase: the phase number signaled
        status: "success" or "failure" (may be overridden by gate checks)
        requires_response: False when no further action needed from the LLM
        error: failure reason (only on gate check failure)"""
    mode = _remap_mode(mode)
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


@mcp.tool()
def signal_verify_results(
    phase: Annotated[Literal[1, 2], Field(description="1=disassembly, 2=assembly")],
    base_name: Annotated[str, Field(description="Assembly base object name")],
    mode: Mode,
    replay_data: Annotated[str, Field(description="JSON string with orchestrator replay outcome")],
) -> Dict[str, Any]:
    """Verify scene state after orchestrator replay of discovered sequences.

    Called by the orchestrator after replaying logged tool_sequences.
    Do not call directly. Runs verify_assembly (phase 2) or
    verify_disassembly (phase 1) on the current scene, then constructs
    a response combining replay outcome with verification result.

    replay_data JSON shape:
        replay: "success" or "failure"
        failed_object: object name (only on replay failure)
        failed_step: tool call string (only on replay failure)
        error: error message (only on replay failure)
        completed_objects: list of objects replayed successfully
        remaining_objects: list of objects not attempted (only on replay failure)

    Returns:
        result: "success" or "failure"
        message: agent-facing instructions (what to fix or confirmation)
        verification: full verify result from scene state check"""
    from triggers.signal_verify_results import handle_verify_results
    return handle_verify_results(
        phase=phase,
        base_name=base_name,
        mode=mode,
        replay_data=replay_data,
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
    mode: Mode,
    reason: Annotated[str, Field(description='Short tag for the client to categorize the signal.')] = "",
) -> SignalOperatorResult:
    """Signal a human operator and wait for confirmation before proceeding.

    Returns:
        result: "success" or "failure"
        action: operator decision ("proceed" or "abort")
        reason: the reason tag provided
        feedback: optional operator feedback text"""
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