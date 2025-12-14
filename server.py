from mcp.server.fastmcp import FastMCP, Image
from typing import List, Any, Optional, Union
from pathlib import Path
import json
import yaml
from box import Box
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


# Configs containing paths of ROS and other related filepaths
config_path = Path(__file__).parent / "SERVER_PATHS_CFGS.yaml"
with open(config_path, "r") as f:
    yaml_cfg = Box(yaml.safe_load(f))


# Set Up WebSocket Manager for ROSBRIDGE
LOCAL_IP = yaml_cfg.rosbridge.local_ip  # Replace with your local IP address
ROSBRIDGE_IP = yaml_cfg.rosbridge.rosbridge_ip  # Replace with your rosbridge server IP address
ROSBRIDGE_PORT = yaml_cfg.rosbridge.rosbridge_port

# This is Global WebSocket manager - don't close it after every operation
ws_manager = WebSocketManager(ROSBRIDGE_IP, ROSBRIDGE_PORT, LOCAL_IP)

# ROS Paths from YAML Config
ROS_SRC = yaml_cfg.ros_paths.ros_src_path
WS_SRC = yaml_cfg.ros_paths.ws_src_path
CUSTOM_LIBS_PATH = yaml_cfg.ros_paths.custom_lib_path
PRIMITIVE_LIBS_PATH = yaml_cfg.ros_paths.primitive_libs_path


# MCP Directory
MCP_SRV_DIR = yaml_cfg.mcp_wrkdir
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
        # Source ROS2 and run the command in bash
        cmd = f"source {ROS_SRC} && source {WS_SRC} && timeout {timeout} ros2 topic echo {topic_name} --once"
        
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
               duration: float = 5.0, 
               custom_lib_path: str = f"{MCP_SRV_DIR}/primitives") -> Dict[str, Any]:
    """
    Perform inverse kinematics and execute smooth trajectory movement using ROS2.
    
    Args:
        target_position: [x, y, z] target position in meters
        target_rpy: [roll, pitch, yaw] target orientation in degrees
        duration: Time to complete the movement in seconds (default: 5.0)
        custom_lib_path: Path to your custom IK solver library
        
    Returns:
        Dictionary with execution result including joint angles and trajectory execution status.
    """
    try:
        import sys
        
        # Import the IK solver from utils package
        try:
            from primitives.utils.ik_solver import compute_ik
        except ImportError as e:
            return {
                "status": "error",
                "message": f"Failed to import ik_solver: {str(e)}. Check if primitives.utils.ik_solver is available."
            }
        
        # Solve IK
        try:
            joint_angles = compute_ik(position=target_position, rpy=target_rpy)
        except Exception as ik_error:
            return {
                "status": "error",
                "message": f"IK solver raised an exception: {str(ik_error)}",
                "target_position": target_position,
                "target_rpy": target_rpy,
                "duration": duration
            }
        
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
                    "message": f"IK solved but trajectory execution failed: {trajectory_result.get('message', 'Unknown error')}. The action server may be temporarily unavailable or busy. Please try running the trajectory again with the same joint angles.",
                    "target_position": target_position,
                    "target_rpy": target_rpy,
                    "joint_angles_rad": joint_angles.tolist(),
                    "joint_angles_deg": joint_angles_deg.tolist(),
                    "duration": duration,
                    "trajectory_status": "failed",
                    "trajectory_error": trajectory_result.get("message"),
                    "trajectory_output": trajectory_result.get("ros_output"),
                    "trajectory_stderr": trajectory_result.get("stderr"),
                    "suggestion": "Try executing the trajectory again using execute_joint_trajectory with the same joint_angles_rad values. The IK solution is valid, only the trajectory execution failed."
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
        import traceback
        return {
            "status": "error",
            "message": f"Unexpected error in IK computation: {str(e)}",
            "traceback": traceback.format_exc()
        }

@mcp.tool()
def get_ee_pose(joint_angles: List[float] = None,
                custom_lib_path: str = f"{CUSTOM_LIBS_PATH}") -> Dict[str, Any]:
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
        
        # Import the IK solver module from utils package
        try:
            from primitives.utils.ik_solver import forward_kinematics, dh_params
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
def verify_grasp(timeout: int = 5) -> Dict[str, Any]:
    """
    Get gripper width information including fully open, fully closed, and current width.
    
    Args:
        timeout: Maximum time to wait for gripper reading (default: 5 seconds)
        
    Returns:
        Dictionary with gripper width information:
        - fully_open_width: Gripper width when fully open (110.0)
        - fully_closed_width: Gripper width when fully closed (9.0)
        - current_width: Current gripper width
        - status: "success" or "error"
    """
    try:
        import subprocess
        import re
        
        # Gripper width constants
        FULLY_OPEN_WIDTH = 110.0
        FULLY_CLOSED_WIDTH = 9.0
        
        # Run the verify_grasp.py script
        script_path = f"{MCP_SRV_DIR}/primitives/verify_grasp.py"
        cmd = f"source {ROS_SRC} && source {WS_SRC} && python {script_path} {timeout}"
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=timeout + 5
        )
        
        if result.returncode == 0:
            # Parse the output to extract current width
            output = result.stdout.strip()
            current_width = None
            
            # Look for the current width line
            for line in output.split('\n'):
                if 'Gripper current width:' in line:
                    try:
                        current_width = float(line.split(':')[1].strip())
                        break
                    except (ValueError, IndexError):
                        pass
            
            if current_width is not None:
                return {
                    "status": "success",
                    "fully_open_width": FULLY_OPEN_WIDTH,
                    "fully_closed_width": FULLY_CLOSED_WIDTH,
                    "current_width": current_width,
                    "raw_output": output
                }
            else:
                return {
                    "status": "error",
                    "message": f"Could not parse current width from output: {output}",
                    "raw_output": output
                }
        else:
            return {
                "status": "error",
                "message": f"Script failed with return code {result.returncode}",
                "stderr": result.stderr.strip() if result.stderr else None,
                "stdout": result.stdout.strip() if result.stdout else None
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": f"Script timed out after {timeout + 5} seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

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
sys.path.append('{MCP_SRV_DIR}')

# Set up ROS2 environment
os.environ['ROS_DOMAIN_ID'] = '0'
os.environ['ROS_VERSION'] = '2'
os.environ['ROS_DISTRO'] = 'humble'

# Source ROS2 setup in subprocess calls
def run_ros2_command(cmd_list, **kwargs):
    \"\"\"Run ROS2 commands with proper environment sourcing\"\"\"
    import subprocess
    if isinstance(cmd_list, str):
        cmd = f"source {ROS_SRC} && source {WS_SRC} && source ~/ros2/install/setup.bash && {{cmd_list}}"
        return subprocess.run(cmd, shell=True, executable='/bin/bash', **kwargs)
    else:
        cmd = f"source {ROS_SRC} && source {WS_SRC} && source ~/ros2/install/setup.bash && {{' '.join(cmd_list)}}"
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
            cwd='{MCP_SRV_DIR}'
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
def move_home() -> Dict[str, Any]:
    """Move robot to home position."""
    try:
        import subprocess
        import sys
        import os
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        move_home_path = os.path.join(script_dir, "primitives", "move_home.py")
        
        cmd_parts = [
            f"source {ROS_SRC}",
            f"source {WS_SRC}",
            "export ROS_DOMAIN_ID=0",
            f"cd {script_dir}/primitives",
            f"timeout 45 /usr/bin/python3 move_home.py"
        ]
        
        cmd = "\n".join(cmd_parts)
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=50
        )
        
        # Check for error messages in output even if returncode is 0
        output_lower = ((result.stdout or "") + (result.stderr or "")).lower()
        has_error = (
            "error" in output_lower or 
            "failed" in output_lower or 
            "❌" in output_lower or
            "timeout" in output_lower or
            "rejected" in output_lower or
            "aborted" in output_lower
        )
        
        if result.returncode == 0 and not has_error:
            return {
                "status": "success",
                "message": "Move home executed successfully",
                "output": result.stdout
            }
        else:
            return {
                "status": "error",
                "message": f"Move home failed with return code {result.returncode}" if result.returncode != 0 else "Move home failed (error detected in output)",
                "error": result.stderr,
                "output": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Move home timed out after 45 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to execute move home: {str(e)}"
        }

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
    """
    try:
        import subprocess
        import sys
        import os
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        move_to_grasp_path = os.path.join(script_dir, "primitives", "move_to_grasp.py")
        
        # Validate mode parameter
        if mode not in ["sim", "real"]:
            return {
                "status": "error",
                "message": f"Invalid mode '{mode}'. Must be 'sim' or 'real'"
            }
        
        cmd_parts = [
            f"source {ROS_SRC}",
            f"source {WS_SRC}",
            "export ROS_DOMAIN_ID=0",
            f"cd {script_dir}/primitives",
            f"timeout 60 /usr/bin/python3 move_to_grasp.py --object-name \"{object_name}\" --grasp-id {grasp_id} --mode {mode}"
        ]
        
        cmd = "\n".join(cmd_parts)
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=70
        )
        
        # Check for error messages in output even if returncode is 0
        output_lower = ((result.stdout or "") + (result.stderr or "")).lower()
        has_error = (
            "error" in output_lower or 
            "failed" in output_lower or 
            "❌" in output_lower or
            "timeout" in output_lower or
            "rejected" in output_lower or
            "aborted" in output_lower
        )
        
        if result.returncode == 0 and not has_error:
            return {
                "status": "success",
                "message": "Move to grasp executed successfully",
                "output": result.stdout,
                "parameters": {
                    "object_name": object_name,
                    "grasp_id": grasp_id,
                    "mode": mode
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Move to grasp failed with return code {result.returncode}" if result.returncode != 0 else "Move to grasp failed (error detected in output)",
                "error": result.stderr,
                "output": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Move to grasp timed out after 60 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to execute move to grasp: {str(e)}"
        }

@mcp.tool()
def reorient_for_assembly(object_name: str, base_name: str, mode: str = "sim") -> Dict[str, Any]:
    """Reorient object for assembly.
    
    Args:
        object_name: Name of the object to reorient
        base_name: Name of the base object
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
    """
    try:
        import subprocess
        import sys
        import os
        
        # Validate mode parameter
        if mode not in ["sim", "real"]:
            return {
                "status": "error",
                "message": f"Invalid mode '{mode}'. Must be 'sim' or 'real'"
            }
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reorient_path = os.path.join(script_dir, "primitives", "reorient_for_assembly.py")
        
        cmd_parts = [
            f"source {ROS_SRC}",
            f"source {WS_SRC}",
            "export ROS_DOMAIN_ID=0",
            f"cd {script_dir}/primitives",
            f"timeout 90 /usr/bin/python3 reorient_for_assembly.py --mode {mode} --object-name \"{object_name}\" --base-name \"{base_name}\""
        ]
        
        cmd = "\n".join(cmd_parts)
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=100
        )
        
        # Check for error messages in output even if returncode is 0
        output_lower = (result.stdout + result.stderr).lower()
        has_error = (
            "error" in output_lower or 
            "failed" in output_lower or 
            "no pose data" in output_lower or
            "not found" in output_lower
        )
        
        if result.returncode == 0 and not has_error:
            return {
                "status": "success",
                "message": "Reorient for assembly executed successfully",
                "parameters": {
                    "object_name": object_name,
                    "base_name": base_name,
                    "mode": mode
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Reorient for assembly failed with return code {result.returncode}" if result.returncode != 0 else "Reorient for assembly failed (error detected in output)",
                "error": result.stderr,
                "output": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Reorient for assembly timed out after 90 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to execute reorient for assembly: {str(e)}"
        }

@mcp.tool()
def translate_for_assembly(object_name: str, base_name: str, mode: str = "sim") -> Dict[str, Any]:
    """Translate object to target position for assembly.
    
    Args:
        object_name: Name of the object being held
        base_name: Name of the base object
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
    """
    try:
        import subprocess
        import sys
        import os
        
        # Validate mode parameter
        if mode not in ["sim", "real"]:
            return {
                "status": "error",
                "message": f"Invalid mode '{mode}'. Must be 'sim' or 'real'"
            }
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        translate_path = os.path.join(script_dir, "primitives", "translate_for_assembly.py")
        
        cmd_parts = [
            f"source {ROS_SRC}",
            f"source {WS_SRC}",
            "export ROS_DOMAIN_ID=0",
            f"cd {script_dir}/primitives",
            f"timeout 90 /usr/bin/python3 translate_for_assembly.py --mode {mode} --object-name \"{object_name}\" --base-name \"{base_name}\""
        ]
        
        cmd = "\n".join(cmd_parts)
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=100
        )
        
        # Check for error messages in output even if returncode is 0
        output_lower = (result.stdout + result.stderr).lower()
        has_error = (
            "error" in output_lower or 
            "failed" in output_lower or 
            "no pose data" in output_lower or
            "not found" in output_lower
        )
        
        if result.returncode == 0 and not has_error:
            return {
                "status": "success",
                "message": "Translate for assembly executed successfully",
                "parameters": {
                    "object_name": object_name,
                    "base_name": base_name,
                    "mode": mode
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Translate for assembly failed with return code {result.returncode}" if result.returncode != 0 else "Translate for assembly failed (error detected in output)",
                "error": result.stderr,
                "output": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Translate for assembly timed out after 90 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to execute translate for assembly: {str(e)}"
        }

@mcp.tool()
def perform_insert(mode: str, object_name: Optional[str] = None, base_name: Optional[str] = None) -> Dict[str, Any]:
    """Perform insert operation with force compliance (sim mode) or force-compliant movement (real mode).
    
    Args:
        mode: Mode to use - "sim" for simulation or "real" for real robot (required)
        object_name: Name of the object being held (required in sim mode)
        base_name: Name of the base object (required in sim mode)
    """
    try:
        import subprocess
        import sys
        import os
        
        # Validate mode parameter
        if mode not in ["sim", "real"]:
            return {
                "status": "error",
                "message": f"Invalid mode '{mode}'. Must be 'sim' or 'real'"
            }
        
        # Validate sim mode requirements
        if mode == "sim":
            if object_name is None or base_name is None:
                return {
                    "status": "error",
                    "message": "In sim mode, object_name and base_name are required"
                }
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        perform_insert_path = os.path.join(script_dir, "primitives", "perform_insert.py")
        
        # Build command based on mode
        if mode == "sim":
            cmd_parts = [
                f"source {ROS_SRC}",
                f"source {WS_SRC}",
                "export ROS_DOMAIN_ID=0",
                f"cd {script_dir}/primitives",
                f"timeout 90 /usr/bin/python3 perform_insert.py --mode {mode} --object-name \"{object_name}\" --base-name \"{base_name}\""
            ]
        else:  # real mode - only mode parameter, use defaults for all others
            cmd_parts = [
                f"source {ROS_SRC}",
                f"source {WS_SRC}",
                "export ROS_DOMAIN_ID=0",
                f"cd {script_dir}/primitives",
                f"timeout 300 /usr/bin/python3 perform_insert.py --mode {mode}"
            ]
        
        cmd = "\n".join(cmd_parts)
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=310 if mode == "real" else 100  # Real mode can take longer
        )
        
        # Check for error messages in output even if returncode is 0
        output_lower = (result.stdout + result.stderr).lower()
        has_error = (
            "error" in output_lower or 
            "failed" in output_lower or 
            "no pose data" in output_lower or
            "not found" in output_lower
        )
        
        if result.returncode == 0 and not has_error:
            return {
                "status": "success",
                "message": "Force compliant move down executed successfully",
                "output": result.stdout,
                "parameters": {
                    "mode": mode,
                    "object_name": object_name if mode == "sim" else None,
                    "base_name": base_name if mode == "sim" else None
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Force compliant move down failed with return code {result.returncode}" if result.returncode != 0 else "Force compliant move down failed (error detected in output)",
                "error": result.stderr,
                "output": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": f"Force compliant move down timed out after {'5 minutes' if mode == 'real' else '90 seconds'}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to execute force compliant move down: {str(e)}"
        }

@mcp.tool()
def verify_final_assembly_pose(object_name: str, base_name: str) -> Dict[str, Any]:
    """Verify if object is in correct final assembly pose relative to base."""
    try:
        import subprocess
        import sys
        import os
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        verify_path = os.path.join(script_dir, "primitives", "verify_final_assembly_pose.py")
        
        cmd_parts = [
            f"source {ROS_SRC}",
            f"source {WS_SRC}",
            "export ROS_DOMAIN_ID=0",
            f"cd {script_dir}/primitives",
            f"timeout 30 /usr/bin/python3 verify_final_assembly_pose.py --object-name \"{object_name}\" --base-name \"{base_name}\""
        ]
        
        cmd = "\n".join(cmd_parts)
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=40
        )
        
        # Check for error messages in output even if returncode is 0
        output_lower = (result.stdout + result.stderr).lower()
        has_error = (
            "error" in output_lower or 
            "failed" in output_lower or 
            "no pose data" in output_lower or
            "not found" in output_lower or
            "verification failed" in output_lower or
            "placement failed" in output_lower
        )
        
        # Check for success indicators
        has_success = (
            "verification successful" in output_lower or
            "verification: success" in output_lower
        )
        
        if result.returncode == 0 and has_success and not has_error:
            return {
                "status": "success",
                "message": "Object is in correct final assembly pose",
                "output": result.stdout,
                "parameters": {
                    "object_name": object_name,
                    "base_name": base_name
                }
            }
        else:
            # If returncode is 1 or verification failed, return error
            return {
                "status": "error",
                "message": "Placement failed - Object is NOT in correct final assembly pose" if (result.returncode == 1 or "verification failed" in output_lower) else f"Verify final assembly pose failed with return code {result.returncode}",
                "error": result.stderr,
                "output": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Verify final assembly pose timed out after 30 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to execute verify final assembly pose: {str(e)}"
        }

@mcp.tool()
def move_down(mode: str = "real") -> Dict[str, Any]:
    """Move robot down with force monitoring."""
    try:
        import subprocess
        import sys
        import os
        
        if mode not in ["sim", "real"]:
            return {
                "status": "error",
                "message": f"Invalid mode '{mode}'. Must be 'sim' or 'real'"
            }
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        move_down_path = os.path.join(script_dir, "primitives", "move_down.py")
        
        cmd_parts = [
            f"source {ROS_SRC}",
            f"source {WS_SRC}",
            "export ROS_DOMAIN_ID=0",
            f"cd {script_dir}/primitives",
            f"/usr/bin/python3 move_down.py --mode {mode}"
        ]
        
        cmd = "\n".join(cmd_parts)
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes - move_down can run incrementally until force threshold
        )
        
        # Check for error messages in output even if returncode is 0
        output_lower = ((result.stdout or "") + (result.stderr or "")).lower()
        has_error = (
            "error" in output_lower or 
            "failed" in output_lower or 
            "❌" in output_lower or
            "timeout" in output_lower or
            "rejected" in output_lower or
            "aborted" in output_lower
        )
        
        if result.returncode == 0 and not has_error:
            return {
                "status": "success",
                "message": "Move down executed successfully",
                "output": result.stdout,
                "parameters": {
                    "mode": mode
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Move down failed with return code {result.returncode}" if result.returncode != 0 else "Move down failed (error detected in output)",
                "error": result.stderr,
                "output": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Move down timed out after 5 minutes. The robot may still be moving - check robot status."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to execute move down: {str(e)}"
        }

@mcp.tool()
def control_gripper(command: str, mode: str = "sim") -> Dict[str, Any]:
    """Control gripper with verification.
    
    Supports "open", "close", or numeric values 0-110 (representing 0-11cm width).
    
    Args:
        command: Gripper command - "open", "close", or numeric value 0-110 (width in mm, representing 0-11cm)
        mode: Mode to use - "sim" for simulation or "real" for real robot (default: "sim")
    
    Returns:
        Dictionary with execution status and results
    """
    try:
        import subprocess
        import sys
        import os
        
        # Validate mode
        if mode not in ["sim", "real"]:
            return {
                "status": "error",
                "message": f"Invalid mode '{mode}'. Must be 'sim' or 'real'"
            }
        
        # Validate command: "open", "close", or numeric value 0-110
        command_lower = command.lower()
        command_value = None
        
        if command_lower in ["open", "close"]:
            command_value = command_lower
        else:
            # Try to parse as numeric value
            try:
                numeric_value = float(command)
                if 0 <= numeric_value <= 110:
                    command_value = str(numeric_value)
                else:
                    return {
                        "status": "error",
                        "message": f"Invalid numeric value '{command}'. Use 0-110 (representing 0-11cm width), 'open', or 'close'."
                    }
            except ValueError:
                return {
                    "status": "error",
                    "message": f"Invalid command '{command}'. Use 'open', 'close', or numeric value 0-110 (representing 0-11cm width)."
                }
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        control_gripper_path = os.path.join(script_dir, "primitives", "control_gripper.py")
        
        cmd_parts = [
            f"source {ROS_SRC}",
            f"source {WS_SRC}",
            "export ROS_DOMAIN_ID=0",
            f"cd {script_dir}/primitives",
            f"timeout 60 /usr/bin/python3 control_gripper.py {command_value} --mode {mode}"
        ]
        
        cmd = "\n".join(cmd_parts)
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=70
        )
        
        # Check for error messages in output even if returncode is 0
        output_lower = ((result.stdout or "") + (result.stderr or "")).lower()
        has_error = (
            "error" in output_lower or 
            "failed" in output_lower or 
            "❌" in output_lower or
            "timeout" in output_lower or
            "rejected" in output_lower or
            "aborted" in output_lower
        )
        
        if result.returncode == 0 and not has_error:
            return {
                "status": "success",
                "message": "Gripper control executed successfully",
                "output": result.stdout,
                "parameters": {
                    "command": command_value,
                    "mode": mode
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Gripper control failed with return code {result.returncode}" if result.returncode != 0 else "Gripper control failed (error detected in output)",
                "error": result.stderr,
                "output": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Gripper control timed out after 60 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to execute gripper control: {str(e)}"
        }

@mcp.tool()
def move_to_safe_height() -> Dict[str, Any]:
    """Move robot to safe height."""
    try:
        import subprocess
        import sys
        import os
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        move_to_safe_height_path = os.path.join(script_dir, "primitives", "move_to_safe_height.py")
        
        cmd_parts = [
            f"source {ROS_SRC}",
            f"source {WS_SRC}",
            "export ROS_DOMAIN_ID=0",
            f"cd {script_dir}/primitives",
            f"timeout 30 /usr/bin/python3 move_to_safe_height.py"
        ]
        
        cmd = "\n".join(cmd_parts)
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=35
        )
        
        # Check for error messages in output even if returncode is 0
        output_lower = ((result.stdout or "") + (result.stderr or "")).lower()
        has_error = (
            "error" in output_lower or 
            "failed" in output_lower or 
            "❌" in output_lower or
            "timeout" in output_lower or
            "rejected" in output_lower or
            "aborted" in output_lower
        )
        
        if result.returncode == 0 and not has_error:
            return {
                "status": "success",
                "message": "Move to safe height executed successfully",
                "output": result.stdout
            }
        else:
            return {
                "status": "error",
                "message": f"Move to safe height failed with return code {result.returncode}" if result.returncode != 0 else "Move to safe height failed (error detected in output)",
                "error": result.stderr,
                "output": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Move to safe height timed out after 30 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to execute move to safe height: {str(e)}"
        }

@mcp.tool()
def move_to_clear_area(mode: str = "move") -> Dict[str, Any]:
    """Move robot to clear area position.
    
    Args:
        mode: Mode to use - "move" to keep current end-effector orientation (default) or "hover" for top-down (face-down) orientation
    """
    try:
        import subprocess
        import sys
        import os
        
        # Validate mode parameter
        if mode not in ["move", "hover"]:
            return {
                "status": "error",
                "message": f"Invalid mode '{mode}'. Must be 'move' or 'hover'"
            }
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        move_to_clear_area_path = os.path.join(script_dir, "primitives", "move_to_clear_area.py")
        
        # Build command based on mode
        if mode == "hover":
            cmd_parts = [
                f"source {ROS_SRC}",
                f"source {WS_SRC}",
                "export ROS_DOMAIN_ID=0",
                f"cd {script_dir}/primitives",
                f"timeout 45 /usr/bin/python3 move_to_clear_area.py --hover"
            ]
        else:  # move mode (default)
            cmd_parts = [
                f"source {ROS_SRC}",
                f"source {WS_SRC}",
                "export ROS_DOMAIN_ID=0",
                f"cd {script_dir}/primitives",
                f"timeout 45 /usr/bin/python3 move_to_clear_area.py --move"
            ]
        
        cmd = "\n".join(cmd_parts)
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=50
        )
        
        # Check for error messages in output even if returncode is 0
        output_lower = ((result.stdout or "") + (result.stderr or "")).lower()
        has_error = (
            "error" in output_lower or 
            "failed" in output_lower or 
            "❌" in output_lower or
            "timeout" in output_lower or
            "rejected" in output_lower or
            "aborted" in output_lower
        )
        
        if result.returncode == 0 and not has_error:
            return {
                "status": "success",
                "message": "Move to place down executed successfully",
                "output": result.stdout,
                "parameters": {
                    "mode": mode
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Move to place down failed with return code {result.returncode}" if result.returncode != 0 else "Move to place down failed (error detected in output)",
                "error": result.stderr,
                "output": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Move to place down timed out after 45 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to execute move to place down: {str(e)}"
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