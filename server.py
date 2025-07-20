from mcp.server.fastmcp import FastMCP, Image
from typing import List, Any, Optional, Union
from pathlib import Path
import json
from utils.websocket_manager import WebSocketManager
from msgs.geometry_msgs import Twist
from msgs.sensor_msgs import Image as RosImage, JointState

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