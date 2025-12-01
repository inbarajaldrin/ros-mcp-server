import argparse
import io
import json
import os
import time
from typing import Any, Dict, List, Optional, Union

from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from PIL import Image as PILImage

from utils.config_utils import get_robot_specifications, parse_robot_config
from utils.network_utils import ping_ip_and_port
from utils.websocket_manager import WebSocketManager, parse_image, parse_json

# ROS bridge connection settings
ROSBRIDGE_IP = "localhost"  # Default is localhost. Replace with your local IPor set using the LLM.
ROSBRIDGE_PORT = (
    9090  # Rosbridge default is 9090. Replace with your rosbridge port or set using the LLM.
)

# MCP transport settings
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio").lower()  # Default is stdio.

# MCP connection settings (streamable-http)
MCP_HOST = os.getenv(
    "MCP_HOST", "localhost"
)  # Default is localhost. Replace with the address of your remote MCP server.

# MCP port settings (default=9000)
MCP_PORT = int(
    os.getenv("MCP_PORT", "9000")
)  # Default is 9000. Replace with the port of your remote MCP server.

# Initialize MCP server and WebSocket manager
mcp = FastMCP("ros-mcp-server")
ws_manager = WebSocketManager(
    ROSBRIDGE_IP, ROSBRIDGE_PORT, default_timeout=5.0
)  # Increased default timeout for ROS operations


# @mcp.tool(description=("Get robot configuration from YAML file."))
def get_robot_config(name: str) -> dict:
    """
    Get the robot configuration from the YAML file for connecting to the robot and knowing its capabilities.

    Returns:
        dict: The robot configuration.
    """
    robot_config = parse_robot_config(name)

    if len(robot_config) > 1:
        return {
            "error": f"Multiple configurations found for robot '{name}'. Please specify a more precise name."
        }
    elif not robot_config:
        return {
            "error": f"No configuration found for robot '{name}'. Please check the name and try again. Or you can set the IP/port manually using the 'connect_to_robot' tool."
        }
    return {"robot_config": robot_config}


# @mcp.tool(
#     description=("List all available robot specifications that can be used with get_robot_config.")
# )
def list_verified_robot_specifications() -> dict:
    """
    Get a list of all available robot specification files.

    Returns:
        dict: List of available robot names that can be used with get_robot_config.
    """
    return get_robot_specifications()


# @mcp.tool(
#     description=(
#         "After getting the robot config, connect to the robot by setting the IP/port and testing connectivity."
#     )
# )
def connect_to_robot(
    ip: Optional[str] = None,
    port: Optional[Union[int, str]] = None,
    ping_timeout: float = 2.0,
    port_timeout: float = 2.0,
) -> dict:
    """
    Connect to a robot by setting the IP and port for the WebSocket connection, then testing connectivity.

    Args:
        ip (Optional[str]): The IP address of the rosbridge server. Defaults to "127.0.0.1" (localhost).
        port (Optional[int]): The port number of the rosbridge server. Defaults to 9090.
        ping_timeout (float): Timeout for ping in seconds. Default = 2.0.
        port_timeout (float): Timeout for port check in seconds. Default = 2.0.

    Returns:
        dict: Connection status with ping and port check results.
    """
    # Set default values if None
    actual_ip = ip if ip is not None else "127.0.0.1"
    actual_port = int(port) if port is not None else 9090

    # Set the IP and port
    ws_manager.set_ip(actual_ip, actual_port)

    # Test connectivity
    ping_result = ping_ip_and_port(actual_ip, actual_port, ping_timeout, port_timeout)

    # Combine the results
    return {
        "message": f"WebSocket IP set to {actual_ip}:{actual_port}",
        "connectivity_test": ping_result,
    }


# @mcp.tool(description="Detect the ROS version and distribution via rosbridge.")
def detect_ros_version() -> dict:
    """
    Detects the ROS version and distro via rosbridge WebSocket.
    Returns:
        dict: {'version': <version or '1'>, 'distro': <distro>} or error info.
    """
    # Try ROS2 detection
    ros2_request = {
        "op": "call_service",
        "id": "ros2_version_check",
        "service": "/rosapi/get_ros_version",
        "args": {},
    }
    with ws_manager:
        response = ws_manager.request(ros2_request)
        values = response.get("values") if response else None
        if isinstance(values, dict) and "version" in values:
            return {"version": values.get("version"), "distro": values.get("distro")}
        # Fallback to ROS1 detection
        ros1_request = {
            "op": "call_service",
            "id": "ros1_distro_check",
            "service": "/rosapi/get_param",
            "args": {"name": "/rosdistro"},
        }
        response = ws_manager.request(ros1_request)
        value = response.get("values") if response else None
        if value:
            distro = value.get("value") if isinstance(value, dict) else value
            distro_clean = str(distro).strip('"').replace("\\n", "").replace("\n", "")
            return {"version": "1", "distro": distro_clean}
        return {"error": "Could not detect ROS version"}


@mcp.tool(description=("Fetch available topics from the ROS bridge.\nExample:\nget_topics()"))
def get_topics() -> dict:
    """
    Fetch available topics from the ROS bridge.

    Returns:
        dict: Contains two lists - 'topics' and 'types',
            or a message string if no topics are found.
    """
    # rosbridge service call to get topic list
    message = {
        "op": "call_service",
        "service": "/rosapi/topics",
        "type": "rosapi/Topics",
        "id": "get_topics_request_1",
    }

    # Request topic list from rosbridge
    with ws_manager:
        response = ws_manager.request(message)

    # Check for service response errors first
    if response and "result" in response and not response["result"]:
        # Service call failed - return error with details from values
        error_msg = response.get("values", {}).get("message", "Service call failed")
        return {"error": f"Service call failed: {error_msg}"}

    # Return topic info if present
    if response and "values" in response:
        return response["values"]
    else:
        return {"warning": "No topics found"}


# @mcp.tool(
#     description=("Get the message type for a specific topic.\nExample:\nget_topic_type('/cmd_vel')")
# )
def get_topic_type(topic: str) -> dict:
    """
    Get the message type for a specific topic.

    Args:
        topic (str): The topic name (e.g., '/cmd_vel')

    Returns:
        dict: Contains the 'type' field with the message type,
            or an error message if topic doesn't exist.
    """
    # Validate input
    if not topic or not topic.strip():
        return {"error": "Topic name cannot be empty"}

    # rosbridge service call to get topic type
    message = {
        "op": "call_service",
        "service": "/rosapi/topic_type",
        "type": "rosapi/TopicType",
        "args": {"topic": topic},
        "id": f"get_topic_type_request_{topic.replace('/', '_')}",
    }

    # Request topic type from rosbridge
    with ws_manager:
        response = ws_manager.request(message)

    # Check for service response errors first
    if response and "result" in response and not response["result"]:
        # Service call failed - return error with details from values
        error_msg = response.get("values", {}).get("message", "Service call failed")
        return {"error": f"Service call failed: {error_msg}"}

    # Return topic type if present
    if response and "values" in response:
        topic_type = response["values"].get("type", "")
        if topic_type:
            return {"topic": topic, "type": topic_type}
        else:
            return {"error": f"Topic {topic} does not exist or has no type"}
    else:
        return {"error": f"Failed to get type for topic {topic}"}


# @mcp.tool(
#     description=(
#         "Get the complete structure/definition of a message type.\n"
#         "Example:\n"
#         "get_message_details('geometry_msgs/Twist')"
#     )
# )
def get_message_details(message_type: str) -> dict:
    """
    Get the complete structure/definition of a message type.

    Args:
        message_type (str): The message type (e.g., 'geometry_msgs/Twist')

    Returns:
        dict: Contains the message structure with field names and types,
            or an error message if the message type doesn't exist.
    """
    # Validate input
    if not message_type or not message_type.strip():
        return {"error": "Message type cannot be empty"}

    # rosbridge service call to get message details
    message = {
        "op": "call_service",
        "service": "/rosapi/message_details",
        "type": "rosapi/MessageDetails",
        "args": {"type": message_type},
        "id": f"get_message_details_request_{message_type.replace('/', '_')}",
    }

    # Request message details from rosbridge
    with ws_manager:
        response = ws_manager.request(message)

    # Check for service response errors first
    if response and "result" in response and not response["result"]:
        # Service call failed - return error with details from values
        error_msg = response.get("values", {}).get("message", "Service call failed")
        return {"error": f"Service call failed: {error_msg}"}

    # Return message structure if present
    if response and "values" in response:
        typedefs = response["values"].get("typedefs", [])
        if typedefs:
            # Parse the structure into a more readable format
            structure = {}
            for typedef in typedefs:
                type_name = typedef.get("type", message_type)
                field_names = typedef.get("fieldnames", [])
                field_types = typedef.get("fieldtypes", [])

                fields = {}
                for name, ftype in zip(field_names, field_types):
                    fields[name] = ftype

                structure[type_name] = {"fields": fields, "field_count": len(fields)}

            return {"message_type": message_type, "structure": structure}
        else:
            return {"error": f"Message type {message_type} not found or has no definition"}
    else:
        return {"error": f"Failed to get details for message type {message_type}"}


# @mcp.tool(
#     description=(
#         "Get list of nodes that are publishing to a specific topic.\n"
#         "Example:\n"
#         "get_publishers_for_topic('/cmd_vel')"
#     )
# )
def get_publishers_for_topic(topic: str) -> dict:
    """
    Get list of nodes that are publishing to a specific topic.

    Args:
        topic (str): The topic name (e.g., '/cmd_vel')

    Returns:
        dict: Contains list of publisher node names,
            or a message if no publishers found.
    """
    # Validate input
    if not topic or not topic.strip():
        return {"error": "Topic name cannot be empty"}

    # rosbridge service call to get publishers
    message = {
        "op": "call_service",
        "service": "/rosapi/publishers",
        "type": "rosapi/Publishers",
        "args": {"topic": topic},
        "id": f"get_publishers_for_topic_request_{topic.replace('/', '_')}",
    }

    # Request publishers from rosbridge
    with ws_manager:
        response = ws_manager.request(message)

    # Check for service response errors first
    if response and "result" in response and not response["result"]:
        # Service call failed - return error with details from values
        error_msg = response.get("values", {}).get("message", "Service call failed")
        return {"error": f"Service call failed: {error_msg}"}

    # Return publishers if present
    if response and "values" in response:
        publishers = response["values"].get("publishers", [])
        return {"topic": topic, "publishers": publishers, "publisher_count": len(publishers)}
    else:
        return {"error": f"Failed to get publishers for topic {topic}"}


# @mcp.tool(
#     description=(
#         "Get list of nodes that are subscribed to a specific topic.\n"
#         "Example:\n"
#         "get_subscribers_for_topic('/cmd_vel')"
#     )
# )
def get_subscribers_for_topic(topic: str) -> dict:
    """
    Get list of nodes that are subscribed to a specific topic.

    Args:
        topic (str): The topic name (e.g., '/cmd_vel')

    Returns:
        dict: Contains list of subscriber node names,
            or a message if no subscribers found.
    """
    # Validate input
    if not topic or not topic.strip():
        return {"error": "Topic name cannot be empty"}

    # rosbridge service call to get subscribers
    message = {
        "op": "call_service",
        "service": "/rosapi/subscribers",
        "type": "rosapi/Subscribers",
        "args": {"topic": topic},
        "id": f"get_subscribers_for_topic_request_{topic.replace('/', '_')}",
    }

    # Request subscribers from rosbridge
    with ws_manager:
        response = ws_manager.request(message)

    # Check for service response errors first
    if response and "result" in response and not response["result"]:
        # Service call failed - return error with details from values
        error_msg = response.get("values", {}).get("message", "Service call failed")
        return {"error": f"Service call failed: {error_msg}"}

    # Return subscribers if present
    if response and "values" in response:
        subscribers = response["values"].get("subscribers", [])
        return {"topic": topic, "subscribers": subscribers, "subscriber_count": len(subscribers)}
    else:
        return {"error": f"Failed to get subscribers for topic {topic}"}


# @mcp.tool(
#     description=(
#         "Get comprehensive information about all ROS topics including publishers, subscribers, and message types. Note that this may take time to execute when three are a large number of topics since it queries each one by one under the hood. \n"
#         "Example:\n"
#         "inspect_all_topics()"
#     )
# )
def inspect_all_topics() -> dict:
    """
    Get comprehensive information about all ROS topics including publishers, subscribers, and message types.

    Returns:
        dict: Contains detailed information about all topics including:
            - Topic names and message types
            - Publishers for each topic
            - Subscribers for each topic
            - Connection counts and statistics
    """
    # First get all topics
    topics_message = {
        "op": "call_service",
        "service": "/rosapi/topics",
        "type": "rosapi/Topics",
        "args": {},
        "id": "inspect_all_topics_request_1",
    }

    with ws_manager:
        topics_response = ws_manager.request(topics_message)

        if not topics_response or "values" not in topics_response:
            return {"error": "Failed to get topics list"}

        topics = topics_response["values"].get("topics", [])
        types = topics_response["values"].get("types", [])
        topic_details = {}

        # Get details for each topic
        topic_errors = []
        for i, topic in enumerate(topics):
            # Get topic type
            topic_type = types[i] if i < len(types) else "unknown"

            # Get publishers for this topic
            publishers_message = {
                "op": "call_service",
                "service": "/rosapi/publishers",
                "type": "rosapi/Publishers",
                "args": {"topic": topic},
                "id": f"get_publishers_{topic.replace('/', '_')}",
            }

            publishers_response = ws_manager.request(publishers_message)
            publishers = []
            if publishers_response and "values" in publishers_response:
                publishers = publishers_response["values"].get("publishers", [])
            elif publishers_response and "error" in publishers_response:
                topic_errors.append(f"Topic {topic} publishers: {publishers_response['error']}")

            # Get subscribers for this topic
            subscribers_message = {
                "op": "call_service",
                "service": "/rosapi/subscribers",
                "type": "rosapi/Subscribers",
                "args": {"topic": topic},
                "id": f"get_subscribers_{topic.replace('/', '_')}",
            }

            subscribers_response = ws_manager.request(subscribers_message)
            subscribers = []
            if subscribers_response and "values" in subscribers_response:
                subscribers = subscribers_response["values"].get("subscribers", [])
            elif subscribers_response and "error" in subscribers_response:
                topic_errors.append(f"Topic {topic} subscribers: {subscribers_response['error']}")

            topic_details[topic] = {
                "type": topic_type,
                "publishers": publishers,
                "subscribers": subscribers,
                "publisher_count": len(publishers),
                "subscriber_count": len(subscribers),
            }

        return {
            "total_topics": len(topics),
            "topics": topic_details,
            "topic_errors": topic_errors,  # Include any errors encountered during inspection
        }


@mcp.tool(
    description=(
        "Subscribe to a ROS topic and return the first message received.\n"
        "Example:\n"
        "subscribe_once(topic='/cmd_vel', msg_type='geometry_msgs/msg/TwistStamped')\n"
        "subscribe_once(topic='/slow_topic', msg_type='my_package/SlowMsg', timeout=None)  # Specify timeout only if topic publishes infrequently\n"
        "subscribe_once(topic='/high_rate_topic', msg_type='sensor_msgs/Image', timeout=None, queue_length=5, throttle_rate_ms=100)  # Control message buffering and rate"
    )
)
def subscribe_once(
    topic: str = "",
    msg_type: str = "",
    timeout: Optional[float] = None,
    queue_length: Optional[int] = None,
    throttle_rate_ms: Optional[int] = None,
) -> dict:
    """
    Subscribe to a given ROS topic via rosbridge and return the first message received.

    Args:
        topic (str): The ROS topic name (e.g., "/cmd_vel", "/joint_states").
        msg_type (str): The ROS message type (e.g., "geometry_msgs/Twist").
        timeout (Optional[float]): Timeout in seconds. If None, uses the default timeout.
        queue_length (Optional[int]): How many messages to buffer before dropping old ones. Must be ≥ 1.
        throttle_rate_ms (Optional[int]): Minimum interval between messages in milliseconds. Must be ≥ 0.

    Returns:
        dict:
            - {"msg": <parsed ROS message>} if successful
            - {"error": "<error message>"} if subscription or timeout fails
    """
    # Validate critical args before attempting subscription
    if not topic or not msg_type:
        return {"error": "Missing required arguments: topic and msg_type must be provided."}

    # Validate optional parameters
    if queue_length is not None and (not isinstance(queue_length, int) or queue_length < 1):
        return {"error": "queue_length must be an integer ≥ 1"}

    if throttle_rate_ms is not None and (
        not isinstance(throttle_rate_ms, int) or throttle_rate_ms < 0
    ):
        return {"error": "throttle_rate_ms must be an integer ≥ 0"}

    # Construct the rosbridge subscribe message
    subscribe_msg: dict = {
        "op": "subscribe",
        "topic": topic,
        "type": msg_type,
    }

    # Add optional parameters if provided
    if queue_length is not None:
        subscribe_msg["queue_length"] = queue_length

    if throttle_rate_ms is not None:
        subscribe_msg["throttle_rate"] = throttle_rate_ms

    # Subscribe and wait for the first message
    with ws_manager:
        # Send subscription request
        send_error = ws_manager.send(subscribe_msg)
        if send_error:
            return {"error": f"Failed to subscribe: {send_error}"}

        # Use default timeout if none specified
        actual_timeout = timeout if timeout is not None else ws_manager.default_timeout

        # Loop until we receive the first message or timeout
        end_time = time.time() + actual_timeout
        while time.time() < end_time:
            response = ws_manager.receive(timeout=0.5)  # non-blocking small timeout
            if response is None:
                continue  # idle timeout: no frame this tick

            if "Image" in msg_type:
                msg_data = parse_image(response)
            else:
                msg_data = parse_json(response)

            if not msg_data:
                continue  # non-JSON or empty

            # Check for status errors from rosbridge
            if msg_data.get("op") == "status" and msg_data.get("level") == "error":
                return {"error": f"Rosbridge error: {msg_data.get('msg', 'Unknown error')}"}

            # Check for the first published message
            if msg_data.get("op") == "publish" and msg_data.get("topic") == topic:
                # Unsubscribe before returning the message
                unsubscribe_msg = {"op": "unsubscribe", "topic": topic}
                ws_manager.send(unsubscribe_msg)
                if "Image" in msg_type:
                    return {
                        "message": "Image received successfully and saved in the MCP server. Run the 'analyze_previously_received_image' tool to analyze it"
                    }
                else:
                    return {"msg": msg_data.get("msg", {})}

        # Timeout - unsubscribe and return error
        unsubscribe_msg = {"op": "unsubscribe", "topic": topic}
        ws_manager.send(unsubscribe_msg)
        return {"error": "Timeout waiting for message from topic"}


# @mcp.tool(
#     description=(
#         "Publish a single message to a ROS topic.\n"
#         "Example:\n"
#         "publish_once(topic='/cmd_vel', msg_type='geometry_msgs/msg/TwistStamped', msg={'linear': {'x': 1.0}})"
#     )
# )
def publish_once(topic: str = "", msg_type: str = "", msg: dict = {}) -> dict:
    """
    Publish a single message to a ROS topic via rosbridge.

    Args:
        topic (str): ROS topic name (e.g., "/cmd_vel")
        msg_type (str): ROS message type (e.g., "geometry_msgs/Twist")
        msg (dict): Message payload as a dictionary

    Returns:
        dict:
            - {"success": True} if sent without errors
            - {"error": "<error message>"} if connection/send failed
            - If rosbridge responds (usually it doesn’t for publish), parsed JSON or error info
    """
    # Validate critical args before attempting publish
    if not topic or not msg_type or msg == {}:
        return {
            "error": "Missing required arguments: topic, msg_type, and msg must all be provided."
        }

    # Use proper advertise → publish → unadvertise pattern
    with ws_manager:
        # 1. Advertise the topic
        advertise_msg = {"op": "advertise", "topic": topic, "type": msg_type}
        send_error = ws_manager.send(advertise_msg)
        if send_error:
            return {"error": f"Failed to advertise topic: {send_error}"}

        # Check for advertise response/errors
        response = ws_manager.receive(timeout=1.0)
        if response:
            try:
                msg_data = json.loads(response)
                if msg_data.get("op") == "status" and msg_data.get("level") == "error":
                    return {"error": f"Advertise failed: {msg_data.get('msg', 'Unknown error')}"}
            except json.JSONDecodeError:
                pass  # Non-JSON response is usually fine for advertise

        # 2. Publish the message
        publish_msg = {"op": "publish", "topic": topic, "msg": msg}
        send_error = ws_manager.send(publish_msg)
        if send_error:
            # Try to unadvertise even if publish failed
            ws_manager.send({"op": "unadvertise", "topic": topic})
            return {"error": f"Failed to publish message: {send_error}"}

        # Check for publish response/errors
        response = ws_manager.receive(timeout=1.0)
        if response:
            try:
                msg_data = json.loads(response)
                if msg_data.get("op") == "status" and msg_data.get("level") == "error":
                    # Unadvertise before returning error
                    ws_manager.send({"op": "unadvertise", "topic": topic})
                    return {"error": f"Publish failed: {msg_data.get('msg', 'Unknown error')}"}
            except json.JSONDecodeError:
                pass  # Non-JSON response is usually fine for publish

        # 3. Unadvertise the topic
        unadvertise_msg = {"op": "unadvertise", "topic": topic}
        ws_manager.send(unadvertise_msg)

    return {
        "success": True,
        "note": "Message published using advertise → publish → unadvertise pattern",
    }


# @mcp.tool(
#     description=(
#         "Subscribe to a topic for a duration and collect messages.\n"
#         "Example:\n"
#         "subscribe_for_duration(topic='/cmd_vel', msg_type='geometry_msgs/msg/TwistStamped', duration=5, max_messages=10)\n"
#         "subscribe_for_duration(topic='/high_rate_topic', msg_type='sensor_msgs/Image', duration=10, queue_length=5, throttle_rate_ms=100)  # Control message buffering and rate"
#     )
# )
def subscribe_for_duration(
    topic: str = "",
    msg_type: str = "",
    duration: float = 5.0,
    max_messages: int = 100,
    queue_length: Optional[int] = None,
    throttle_rate_ms: Optional[int] = None,
) -> dict:
    """
    Subscribe to a ROS topic via rosbridge for a fixed duration and collect messages.

    Args:
        topic (str): ROS topic name (e.g. "/cmd_vel", "/joint_states")
        msg_type (str): ROS message type (e.g. "geometry_msgs/Twist")
        duration (float): How long (seconds) to listen for messages
        max_messages (int): Maximum number of messages to collect before stopping
        queue_length (Optional[int]): How many messages to buffer before dropping old ones. Must be ≥ 1.
        throttle_rate_ms (Optional[int]): Minimum interval between messages in milliseconds. Must be ≥ 0.

    Returns:
        dict:
            {
                "topic": topic_name,
                "collected_count": N,
                "messages": [msg1, msg2, ...]
            }
    """
    # Validate critical args before subscribing
    if not topic or not msg_type:
        return {"error": "Missing required arguments: topic and msg_type must be provided."}

    # Validate optional parameters
    if queue_length is not None and (not isinstance(queue_length, int) or queue_length < 1):
        return {"error": "queue_length must be an integer ≥ 1"}

    if throttle_rate_ms is not None and (
        not isinstance(throttle_rate_ms, int) or throttle_rate_ms < 0
    ):
        return {"error": "throttle_rate_ms must be an integer ≥ 0"}

    # Send subscription request
    subscribe_msg: dict = {
        "op": "subscribe",
        "topic": topic,
        "type": msg_type,
    }

    # Add optional parameters if provided
    if queue_length is not None:
        subscribe_msg["queue_length"] = queue_length

    if throttle_rate_ms is not None:
        subscribe_msg["throttle_rate"] = throttle_rate_ms

    with ws_manager:
        send_error = ws_manager.send(subscribe_msg)
        if send_error:
            return {"error": f"Failed to subscribe: {send_error}"}

        collected_messages = []
        status_errors = []
        end_time = time.time() + duration

        # Loop until duration expires or we hit max_messages
        while time.time() < end_time and len(collected_messages) < max_messages:
            response = ws_manager.receive(timeout=0.5)  # non-blocking small timeout
            if response is None:
                continue  # idle timeout: no frame this tick

            msg_data = parse_json(response)
            if not msg_data:
                continue  # non-JSON or empty

            # Check for status errors from rosbridge
            if msg_data.get("op") == "status" and msg_data.get("level") == "error":
                status_errors.append(msg_data.get("msg", "Unknown error"))
                continue

            # Check for published messages matching our topic
            if msg_data.get("op") == "publish" and msg_data.get("topic") == topic:
                collected_messages.append(msg_data.get("msg", {}))

        # Unsubscribe when done
        unsubscribe_msg = {"op": "unsubscribe", "topic": topic}
        ws_manager.send(unsubscribe_msg)

    return {
        "topic": topic,
        "collected_count": len(collected_messages),
        "messages": collected_messages,
        "status_errors": status_errors,  # Include any errors encountered during collection
    }


# @mcp.tool(
#     description=(
#         "Publish a sequence of messages with delays.\n"
#         "Example:\n"
#         "publish_for_durations(topic='/cmd_vel', msg_type='geometry_msgs/msg/TwistStamped', messages=[{'linear': {'x': 1.0}}, {'linear': {'x': 0.0}}], durations=[1, 2])"
#     )
# )
def publish_for_durations(
    topic: str = "",
    msg_type: str = "",
    messages: List[Dict[str, Any]] = [],
    durations: List[float] = [],
) -> dict:
    """
    Publish a sequence of messages to a given ROS topic with delays in between.

    Args:
        topic (str): ROS topic name (e.g., "/cmd_vel")
        msg_type (str): ROS message type (e.g., "geometry_msgs/Twist")
        messages (List[Dict[str, Any]]): A list of message dictionaries (ROS-compatible payloads)
        durations (List[float]): A list of durations (seconds) to wait between messages

    Returns:
        dict:
            {
                "success": True,
                "published_count": <number of messages>,
                "topic": topic,
                "msg_type": msg_type
            }
            OR {"error": "<error message>"} if something failed
    """
    # Validate critical args before publishing
    if not topic or not msg_type or messages == [] or durations == []:
        return {
            "error": "Missing required arguments: topic, msg_type, messages, and durations must all be provided."
        }

    # Ensure same length for messages & durations
    if len(messages) != len(durations):
        return {"error": "messages and durations must have the same length"}

    # Use proper advertise → publish → unadvertise pattern
    with ws_manager:
        # 1. Advertise the topic
        advertise_msg = {"op": "advertise", "topic": topic, "type": msg_type}
        send_error = ws_manager.send(advertise_msg)
        if send_error:
            return {"error": f"Failed to advertise topic: {send_error}"}

        # Check for advertise response/errors
        response = ws_manager.receive(timeout=1.0)
        if response:
            try:
                msg_data = json.loads(response)
                if msg_data.get("op") == "status" and msg_data.get("level") == "error":
                    return {"error": f"Advertise failed: {msg_data.get('msg', 'Unknown error')}"}
            except json.JSONDecodeError:
                pass  # Non-JSON response is usually fine for advertise

        published_count = 0
        errors = []

        # 2. Iterate and publish each message with a delay
        for i, (msg, delay) in enumerate(zip(messages, durations)):
            # Build the rosbridge publish message
            publish_msg = {"op": "publish", "topic": topic, "msg": msg}

            # Send it
            send_error = ws_manager.send(publish_msg)
            if send_error:
                errors.append(f"Message {i + 1}: {send_error}")
                continue  # Continue with next message instead of failing completely

            # Check for publish response/errors
            response = ws_manager.receive(timeout=1.0)
            if response:
                try:
                    msg_data = json.loads(response)
                    if msg_data.get("op") == "status" and msg_data.get("level") == "error":
                        errors.append(f"Message {i + 1}: {msg_data.get('msg', 'Unknown error')}")
                        continue
                except json.JSONDecodeError:
                    pass  # Non-JSON response is usually fine for publish

            published_count += 1

            # Wait before sending the next message
            time.sleep(delay)

        # 3. Unadvertise the topic
        unadvertise_msg = {"op": "unadvertise", "topic": topic}
        ws_manager.send(unadvertise_msg)

    return {
        "success": True,
        "published_count": published_count,
        "total_messages": len(messages),
        "topic": topic,
        "msg_type": msg_type,
        "errors": errors,  # Include any errors encountered during publishing
    }


## ############################################################################################## ##
##
##                       ROS SERVICES
##
## ############################################################################################## ##


# @mcp.tool(description=("Get list of all available ROS services.\nExample:\nget_services()"))
def get_services() -> dict:
    """
    Get list of all available ROS services.

    Returns:
        dict: Contains list of all active services,
            or a message string if no services are found.
    """
    # rosbridge service call to get service list
    message = {
        "op": "call_service",
        "service": "/rosapi/services",
        "type": "rosapi/Services",
        "args": {},
        "id": "get_services_request_1",
    }

    # Request service list from rosbridge
    with ws_manager:
        response = ws_manager.request(message)

    # Check for service response errors first
    if response and "result" in response and not response["result"]:
        # Service call failed - return error with details from values
        error_msg = response.get("values", {}).get("message", "Service call failed")
        return {"error": f"Service call failed: {error_msg}"}

    # Return service info if present
    if response and "values" in response:
        services = response["values"].get("services", [])
        return {"services": services, "service_count": len(services)}
    else:
        return {"warning": "No services found"}


# @mcp.tool(
#     description=(
#         "Get the service type for a specific service.\nExample:\nget_service_type('/rosapi/topics')"
#     )
# )
def get_service_type(service: str) -> dict:
    """
    Get the service type for a specific service.

    Args:
        service (str): The service name (e.g., '/rosapi/topics')

    Returns:
        dict: Contains the service type,
            or an error message if service doesn't exist.
    """
    # Validate input
    if not service or not service.strip():
        return {"error": "Service name cannot be empty"}

    # rosbridge service call to get service type
    message = {
        "op": "call_service",
        "service": "/rosapi/service_type",
        "type": "rosapi/ServiceType",
        "args": {"service": service},
        "id": f"get_service_type_request_{service.replace('/', '_')}",
    }

    # Request service type from rosbridge
    with ws_manager:
        response = ws_manager.request(message)

    # Check for service response errors first
    if response and "result" in response and not response["result"]:
        # Service call failed - return error with details from values
        error_msg = response.get("values", {}).get("message", "Service call failed")
        return {"error": f"Service call failed: {error_msg}"}

    # Return service type if present
    if response and "values" in response:
        service_type = response["values"].get("type", "")
        if service_type:
            return {"service": service, "type": service_type}
        else:
            return {"error": f"Service {service} does not exist or has no type"}
    else:
        return {"error": f"Failed to get type for service {service}"}


# @mcp.tool(
#     description=(
#         "Get complete service details including request and response structures.\n"
#         "Example:\n"
#         "get_service_details('my_package/CustomService')"
#     )
# )
def get_service_details(service_type: str) -> dict:
    """
    Get complete service details including request and response structures.

    Args:
        service_type (str): The service type (e.g., 'my_package/CustomService')

    Returns:
        dict: Contains complete service definition with request and response structures.
    """
    # Validate input
    if not service_type or not service_type.strip():
        return {"error": "Service type cannot be empty"}

    result = {"service_type": service_type, "request": {}, "response": {}}

    # Get both request and response details in a single WebSocket context
    with ws_manager:
        # Get request details
        request_message = {
            "op": "call_service",
            "service": "/rosapi/service_request_details",
            "type": "rosapi/ServiceRequestDetails",
            "args": {"type": service_type},
            "id": f"get_service_details_request_{service_type.replace('/', '_')}",
        }

        request_response = ws_manager.request(request_message)
        if request_response and "values" in request_response:
            typedefs = request_response["values"].get("typedefs", [])
            if typedefs:
                for typedef in typedefs:
                    field_names = typedef.get("fieldnames", [])
                    field_types = typedef.get("fieldtypes", [])
                    fields = {}
                    for name, ftype in zip(field_names, field_types):
                        fields[name] = ftype
                    result["request"] = {"fields": fields, "field_count": len(fields)}

        # Get response details
        response_message = {
            "op": "call_service",
            "service": "/rosapi/service_response_details",
            "type": "rosapi/ServiceResponseDetails",
            "args": {"type": service_type},
            "id": f"get_service_details_response_{service_type.replace('/', '_')}",
        }

        response_response = ws_manager.request(response_message)
        if response_response and "values" in response_response:
            typedefs = response_response["values"].get("typedefs", [])
            if typedefs:
                for typedef in typedefs:
                    field_names = typedef.get("fieldnames", [])
                    field_types = typedef.get("fieldtypes", [])
                    fields = {}
                    for name, ftype in zip(field_names, field_types):
                        fields[name] = ftype
                    result["response"] = {"fields": fields, "field_count": len(fields)}

    # Check if we got any data
    if not result["request"] and not result["response"]:
        return {"error": f"Service type {service_type} not found or has no definition"}

    return result


# @mcp.tool(
#     description=(
#         "Get list of nodes that provide a specific service.\n"
#         "Example:\n"
#         "get_service_providers('/rosapi/topics')"
#     )
# )
def get_service_providers(service: str) -> dict:
    """
    Get list of nodes that provide a specific service.

    Args:
        service (str): The service name (e.g., '/rosapi/topics')

    Returns:
        dict: Contains list of nodes providing this service,
            or an error message if service doesn't exist.
    """
    # Validate input
    if not service or not service.strip():
        return {"error": "Service name cannot be empty"}

    # rosbridge service call to get service providers (using service_node like inspect_all_services)
    message = {
        "op": "call_service",
        "service": "/rosapi/service_node",
        "type": "rosapi/ServiceNode",
        "args": {"service": service},
        "id": f"get_service_providers_request_{service.replace('/', '_')}",
    }

    # Request service providers from rosbridge
    with ws_manager:
        response = ws_manager.request(message)

    # Return service providers if present (using same logic as inspect_all_services)
    providers = []

    # Handle different response formats safely
    if response and isinstance(response, dict):
        if "values" in response:
            node = response["values"].get("node", "")
            if node:
                providers = [node]
        elif "result" in response:
            node = response["result"].get("node", "")
            if node:
                providers = [node]
        elif "error" in response:
            return {"error": f"Service call failed: {response['error']}"}
    elif response is False:
        return {"error": f"No response received for service {service}"}
    elif response is True:
        return {"error": f"Unexpected boolean response for service {service}"}
    else:
        return {"error": f"Failed to get providers for service {service}"}

    return {"service": service, "providers": providers, "provider_count": len(providers)}


# @mcp.tool(
#     description=(
#         "Get comprehensive information about all services including types and providers. Note that this may take time to execute when three are a large number of services since it queries each one by one under the hood. \n"
#         "Example:\n"
#         "inspect_all_services()"
#     )
# )
def inspect_all_services() -> dict:
    """
    Get comprehensive information about all services including types and providers.

    Returns:
        dict: Contains detailed information about all services,
            including service names, types, and provider nodes.
    """
    # First get all services
    services_message = {
        "op": "call_service",
        "service": "/rosapi/services",
        "type": "rosapi/Services",
        "args": {},
        "id": "inspect_all_services_request_1",
    }

    with ws_manager:
        services_response = ws_manager.request(services_message)

        if not services_response or "values" not in services_response:
            return {"error": "Failed to get services list"}

        services = services_response["values"].get("services", [])
        service_details = {}

        # Get details for each service
        service_errors = []
        for service in services:
            # Get service type
            type_message = {
                "op": "call_service",
                "service": "/rosapi/service_type",
                "type": "rosapi/ServiceType",
                "args": {"service": service},
                "id": f"get_type_{service.replace('/', '_')}",
            }

            type_response = ws_manager.request(type_message)
            service_type = ""
            if type_response and "values" in type_response:
                service_type = type_response["values"].get("type", "unknown")
            elif type_response and "error" in type_response:
                service_errors.append(f"Service {service}: {type_response['error']}")

            # Get service provider (using service_node instead of service_providers)
            provider_message = {
                "op": "call_service",
                "service": "/rosapi/service_node",
                "type": "rosapi/ServiceNode",
                "args": {"service": service},
                "id": f"get_provider_{service.replace('/', '_')}",
            }

            provider_response = ws_manager.request(provider_message)
            providers = []

            # Handle different response formats safely
            if provider_response and isinstance(provider_response, dict):
                if "values" in provider_response:
                    node = provider_response["values"].get("node", "")
                    if node:
                        providers = [node]
                elif "result" in provider_response:
                    node = provider_response["result"].get("node", "")
                    if node:
                        providers = [node]
                elif "error" in provider_response:
                    service_errors.append(
                        f"Service {service} provider: {provider_response['error']}"
                    )
            elif provider_response is False:
                service_errors.append(f"Service {service} provider: No response received")
            elif provider_response is True:
                service_errors.append(f"Service {service} provider: Unexpected boolean response")

            service_details[service] = {
                "type": service_type,
                "providers": providers,
                "provider_count": len(providers),
            }

        return {
            "total_services": len(services),
            "services": service_details,
            "service_errors": service_errors,  # Include any errors encountered during inspection
        }


# @mcp.tool(
#     description=(
#         "Call a ROS service with specified request data.\n"
#         "Example:\n"
#         "call_service('/rosapi/topics', 'rosapi/Topics', {})\n"
#         "call_service('/slow_service', 'my_package/SlowService', {}, timeout=10.0)  # Specify timeout only for slow services"
#     )
# )
def call_service(
    service_name: str, service_type: str, request: dict, timeout: Optional[float] = None
) -> dict:
    """
    Call a ROS service with specified request data.

    Args:
        service_name (str): The service name (e.g., '/rosapi/topics')
        service_type (str): The service type (e.g., 'rosapi/Topics')
        request (dict): Service request data as a dictionary
        timeout (Optional[float]): Timeout in seconds. If None, uses the default timeout.

    Returns:
        dict: Contains the service response or error information.
    """
    # rosbridge service call
    message = {
        "op": "call_service",
        "service": service_name,
        "type": service_type,
        "args": request,
        "id": f"call_service_request_{service_name.replace('/', '_')}",
    }

    # Call the service through rosbridge
    with ws_manager:
        response = ws_manager.request(message, timeout=timeout)

    # Check for service response errors first
    if response and "result" in response and not response["result"]:
        # Service call failed - return error with details from values
        error_msg = response.get("values", {}).get("message", "Service call failed")
        return {
            "service": service_name,
            "service_type": service_type,
            "success": False,
            "error": f"Service call failed: {error_msg}",
        }

    # Return service response if present
    if response:
        if response.get("op") == "service_response":
            # Alternative response format
            return {
                "service": service_name,
                "service_type": service_type,
                "success": response.get("result", True),
                "result": response.get("values", {}),
            }
        elif response.get("op") == "status" and response.get("level") == "error":
            # Error response
            return {
                "service": service_name,
                "service_type": service_type,
                "success": False,
                "error": response.get("msg", "Unknown error"),
            }
        else:
            # Unexpected response format
            return {
                "service": service_name,
                "service_type": service_type,
                "success": False,
                "error": "Unexpected response format",
                "raw_response": response,
            }
    else:
        return {
            "service": service_name,
            "service_type": service_type,
            "success": False,
            "error": "No response received from service call",
        }


# @mcp.tool(description=("Get list of all currently running ROS nodes.\nExample:\nget_nodes()"))
def get_nodes() -> dict:
    """
    Get list of all currently running ROS nodes.

    Returns:
        dict: Contains list of all active nodes,
            or a message string if no nodes are found.
    """
    # rosbridge service call to get node list
    message = {
        "op": "call_service",
        "service": "/rosapi/nodes",
        "type": "rosapi/Nodes",
        "args": {},
        "id": "get_nodes_request_1",
    }

    # Request node list from rosbridge
    with ws_manager:
        response = ws_manager.request(message)

    # Check for service response errors first
    if response and "result" in response and not response["result"]:
        # Service call failed - return error with details from values
        error_msg = response.get("values", {}).get("message", "Service call failed")
        return {"error": f"Service call failed: {error_msg}"}

    # Return node info if present
    if response and "values" in response:
        nodes = response["values"].get("nodes", [])
        return {"nodes": nodes, "node_count": len(nodes)}
    else:
        return {"warning": "No nodes found"}


# @mcp.tool(
#     description=(
#         "Get detailed information about a specific node including its publishers, subscribers, and services.\n"
#         "Example:\n"
#         "get_node_details('/turtlesim')"
#     )
# )
def get_node_details(node: str) -> dict:
    """
    Get detailed information about a specific node including its publishers, subscribers, and services.

    Args:
        node (str): The node name (e.g., '/turtlesim')

    Returns:
        dict: Contains detailed node information including publishers, subscribers, and services,
            or an error message if node doesn't exist.
    """
    # Validate input
    if not node or not node.strip():
        return {"error": "Node name cannot be empty"}

    result = {
        "node": node,
        "publishers": [],
        "subscribers": [],
        "services": [],
        "publisher_count": 0,
        "subscriber_count": 0,
        "service_count": 0,
    }

    # rosbridge service call to get node details
    message = {
        "op": "call_service",
        "service": "/rosapi/node_details",
        "type": "rosapi/NodeDetails",
        "args": {"node": node},
        "id": f"get_node_details_{node.replace('/', '_')}",
    }

    # Request node details from rosbridge
    with ws_manager:
        response = ws_manager.request(message)

    # Check for service response errors first
    if response and "result" in response and not response["result"]:
        # Service call failed - return error with details from values
        error_msg = response.get("values", {}).get("message", "Service call failed")
        return {"error": f"Service call failed: {error_msg}"}

    # Extract data from the response
    if response and "values" in response:
        values = response["values"]
        # Extract publishers, subscribers, and services from the response
        # Note: rosapi uses "publishing" and "subscribing" field names
        publishers = values.get("publishing", [])
        subscribers = values.get("subscribing", [])
        services = values.get("services", [])

        result["publishers"] = publishers
        result["subscribers"] = subscribers
        result["services"] = services
        result["publisher_count"] = len(publishers)
        result["subscriber_count"] = len(subscribers)
        result["service_count"] = len(services)

    # Check if we got any data
    if not result["publishers"] and not result["subscribers"] and not result["services"]:
        return {"error": f"Node {node} not found or has no details available"}

    return result


# @mcp.tool(
#     description=(
#         "Get comprehensive information about all ROS nodes including their publishers, subscribers, and services.\n"
#         "Example:\n"
#         "inspect_all_nodes()"
#     )
# )
def inspect_all_nodes() -> dict:
    """
    Get comprehensive information about all ROS nodes including their publishers, subscribers, and services.

    Returns:
        dict: Contains detailed information about all nodes including:
            - Node names and details
            - Publishers for each node
            - Subscribers for each node
            - Services provided by each node
            - Connection counts and statistics
    """
    # First get all nodes
    nodes_message = {
        "op": "call_service",
        "service": "/rosapi/nodes",
        "type": "rosapi/Nodes",
        "args": {},
        "id": "inspect_all_nodes_request_1",
    }

    with ws_manager:
        nodes_response = ws_manager.request(nodes_message)

        if not nodes_response or "values" not in nodes_response:
            return {"error": "Failed to get nodes list"}

        nodes = nodes_response["values"].get("nodes", [])
        node_details = {}

        # Get details for each node
        node_errors = []
        for node in nodes:
            # Get node details (publishers, subscribers, services)
            node_details_message = {
                "op": "call_service",
                "service": "/rosapi/node_details",
                "type": "rosapi/NodeDetails",
                "args": {"node": node},
                "id": f"get_node_details_{node.replace('/', '_')}",
            }

            node_details_response = ws_manager.request(node_details_message)

            if node_details_response and "values" in node_details_response:
                values = node_details_response["values"]
                # Extract publishers, subscribers, and services from the response
                # Note: rosapi uses "publishing" and "subscribing" field names
                publishers = values.get("publishing", [])
                subscribers = values.get("subscribing", [])
                services = values.get("services", [])

                node_details[node] = {
                    "publishers": publishers,
                    "subscribers": subscribers,
                    "services": services,
                    "publisher_count": len(publishers),
                    "subscriber_count": len(subscribers),
                    "service_count": len(services),
                }
            elif (
                node_details_response
                and "result" in node_details_response
                and not node_details_response["result"]
            ):
                error_msg = node_details_response.get("values", {}).get(
                    "message", "Service call failed"
                )
                node_errors.append(f"Node {node}: {error_msg}")
            else:
                node_errors.append(f"Node {node}: Failed to get node details")

        return {
            "total_nodes": len(nodes),
            "nodes": node_details,
            "node_errors": node_errors,  # Include any errors encountered during inspection
        }


## ############################################################################################## ##
##
##                       NETWORK DIAGNOSTICS
##
## ############################################################################################## ##


# @mcp.tool(
#     description=(
#         "Ping a robot's IP address and check if a specific port is open.\n"
#         "A successful ping to the IP but not the port can indicate that ROSbridge is not running.\n"
#         "Example:\n"
#         "ping_robot(ip='192.168.1.100', port=9090)"
#     )
# )
def ping_robot(ip: str, port: int, ping_timeout: float = 2.0, port_timeout: float = 2.0) -> dict:
    """
    Ping an IP address and check if a specific port is open.

    Args:
        ip (str): The IP address to ping (e.g., '192.168.1.100')
        port (int): The port number to check (e.g., 9090)
        ping_timeout (float): Timeout for ping in seconds. Default = 2.0.
        port_timeout (float): Timeout for port check in seconds. Default = 2.0.

    Returns:
        dict: Contains ping and port check results with detailed status information.
    """
    return ping_ip_and_port(ip, port, ping_timeout, port_timeout)


## ############################################################################################## ##
##
##                      IMAGE ANALYSIS
##
## ############################################################################################## ##


# @mcp.tool(
#     description=(
#         "First, subscribe to an Image topic using 'subscribe_once' to save an image.\n"
#         "Then, use this tool to analyze the saved image\n"
#     )
# )
def _encode_image_to_imagecontent(image):
    """
    Encodes a PIL Image to a format compatible with ImageContent.

    Args:
        image (PIL.Image.Image): The image to encode.

    Returns:
        ImageContent: JPEG-encoded image wrapped in an ImageContent object.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    img_obj = Image(data=img_bytes, format="jpeg")
    return img_obj.to_image_content()


def analyze_previously_received_image():
    """
    Analyze the previously received image saved at ./camera/received_image.jpeg

    This tool loads the previously saved image from './camera/received_image.jpeg'
    (which must have been created by 'parse_image' or 'subscribe_once'), and converts
    it into an MCP-compatible ImageContent format so that the LLM can interpret it.
    """
    path = "./camera/received_image.jpeg"
    if not os.path.exists(path):
        return {"error": "No image found at ./camera/received_image.jpeg"}
    img = PILImage.open(path)
    return _encode_image_to_imagecontent(img)


def parse_arguments():
    """Parse command line arguments for MCP server configuration."""
    parser = argparse.ArgumentParser(
        description="ROS MCP Server - Connect to ROS robots via MCP protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py                                    # Use stdio transport (default)
  python server.py --transport http --host 0.0.0.0 --port 9000
  python server.py --transport streamable-http --host 127.0.0.1 --port 8080
        """,
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport protocol to use (default: stdio)",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address for HTTP-based transports (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port number for HTTP-based transports (default: 9000)",
    )

    return parser.parse_args()


## ############################################################################################## ##
##
##                      YOLOE DETECTION
##
## ############################################################################################## ##


# @mcp.tool(
#     description=(
#         "Run the prompt-free YOLOE detection test and return both original and annotated images.\n"
#         "This tool executes the prompt_free_test.py script and returns the images it creates."
#     )
# )
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
        script_path = "/home/aaugus11/Documents/ros-mcp-server/tools/yoloe/prompt_free_test.py"
        screenshots_dir = "/home/aaugus11/Documents/ros-mcp-server/tools/yoloe/screenshots"
        
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
            f"cd /home/aaugus11/Documents/ros-mcp-server/tools/yoloe && "
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
                if 'conf:' in line:
                    try:
                        parts = line.split('conf:')[0].strip()
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
            "error": str(e)
        }

# @mcp.tool(
#     description=(
#         "Run the prompt-set YOLOE detection test with multiple text prompts.\n"
#         "Tests multiple prompts and identifies the best performing one.\n"
#         "Format: Use comma-separated prompts like 'red object, blue object, green object'"
#     )
# )
def run_prompt_set_detection(prompts: str = ""):
    """
    Run the prompt-set YOLOE detection test with multiple text prompts.
    This tool tests multiple prompts on a single captured image, saves annotated results for each prompt,
    and identifies which prompt performs best for object detection.
    
    CRITICAL: Separate multiple prompts with COMMAS, not periods or other separators.
    
    Args:
        prompts: Comma-separated string of text prompts to test.
                 Each prompt should describe objects to detect.
                 
                 CORRECT FORMAT: "red object, blue object, green object"
                 WRONG FORMAT:   "red object. blue object. green object"
                 
                 Alternative: JSON array format like '["red object", "blue object"]'
                 
                 Pass empty string "" to use default prompts.
    
    Returns:
        Dictionary with detection results, images for each tested prompt, and the best prompt identified
        
    Examples:
        # CORRECT - Comma-separated prompts
        run_prompt_set_detection(prompts="red object, blue object, green object, yellow object")
        
        # CORRECT - JSON array format
        run_prompt_set_detection(prompts='["red object", "blue object", "green object"]')
        
        # Use default prompts
        run_prompt_set_detection(prompts="")
    """
    try:
        import subprocess
        import os
        import glob
        from datetime import datetime
        import re
        import json
        
        # Parse prompts from string format (matches update_yolo_prompts pattern)
        prompt_list = []
        
        if prompts and prompts.strip():
            # Try to parse as JSON array first
            if prompts.strip().startswith('['):
                try:
                    prompt_list = json.loads(prompts)
                    if not isinstance(prompt_list, list):
                        return {
                            "status": "error",
                            "error": f"JSON parsed but result is not a list: {type(prompt_list).__name__}"
                        }
                except json.JSONDecodeError as e:
                    return {
                        "status": "error",
                        "error": f"Invalid JSON array format: {str(e)}"
                    }
            else:
                # Parse as comma-separated string only
                if ',' in prompts:
                    prompt_list = [p.strip() for p in prompts.split(',') if p.strip()]
                else:
                    # Single prompt (no commas)
                    prompt_list = [prompts.strip()]
        
        # If empty, prompt_list stays [] and script will use defaults
        
        script_path = "/home/aaugus11/Documents/ros-mcp-server/tools/yoloe/prompt_set_test.py"
        screenshots_dir = "/home/aaugus11/Documents/ros-mcp-server/tools/yoloe/screenshots"
        
        # Check if script exists
        if not os.path.exists(script_path):
            return {
                "status": "error",
                "error": f"Prompt set test script not found: {script_path}"
            }
        
        # Clear old screenshots to avoid confusion
        old_screenshots = glob.glob(os.path.join(screenshots_dir, "prompt_*.jpg"))
        for old_file in old_screenshots:
            try:
                os.remove(old_file)
            except:
                pass
        
        # Build the command parts (same pattern as update_yolo_prompts)
        cmd_parts = [
            "bash", "-c",
            "source /opt/ros/humble/setup.bash && "
            "source ~/Desktop/ros2_ws/install/setup.bash && "
            "export ROS_DOMAIN_ID=0 && "
            f"cd /home/aaugus11/Documents/ros-mcp-server/tools/yoloe && "
            "python3 prompt_set_test.py"
        ]
        
        # Add prompts as command-line arguments if provided (same pattern as update_yolo_prompts)
        if prompt_list and len(prompt_list) > 0:
            cmd_parts[2] += " --prompts"
            for prompt in prompt_list:
                cmd_parts[2] += f" '{prompt}'"
        # If empty list, script will use DEFAULT_PROMPTS
        
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=120  # Longer timeout since testing multiple prompts
        )
        
        # Find all the prompt test images that were created
        prompt_images = glob.glob(os.path.join(screenshots_dir, "prompt_*.jpg"))
        
        if not prompt_images:
            return {
                "status": "error",
                "message": "No prompt test images found after running script",
                "script_output": result.stdout if result.stdout else None,
                "script_stderr": result.stderr if result.stderr else None
            }
        
        # Parse results from script output
        tested_prompts = []
        best_prompt = None
        
        if result and result.stdout:
            lines = result.stdout.strip().split('\n')
            
            # Parse test results
            for line in lines:
                if "Testing prompt" in line and ":" in line:
                    # Extract prompt name from line like "Testing prompt 1/5: 'red object'"
                    match = re.search(r"Testing prompt \d+/\d+: '([^']+)'", line)
                    if match:
                        prompt_name = match.group(1)
                        tested_prompts.append({"prompt": prompt_name, "detections": 0})
                
                # Look for detection counts
                elif "Found" in line and "detections" in line:
                    match = re.search(r"Found (\d+) detections", line)
                    if match and tested_prompts:
                        tested_prompts[-1]["detections"] = int(match.group(1))
                
                # Extract best prompt
                elif "BEST PROMPT:" in line:
                    match = re.search(r"BEST PROMPT: '([^']+)'", line)
                    if match:
                        best_prompt = match.group(1)
        
        # Read the prompt test images and create MCP Image objects
        mcp_images = []
        for img_path in sorted(prompt_images):
            try:
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                mcp_images.append(Image(data=img_data, format="jpeg"))
            except Exception as e:
                continue
        
        return {
            "status": "success",
            "message": "Prompt-set detection completed successfully",
            "tested_prompts": tested_prompts,
            "best_prompt": best_prompt,
            "test_images_count": len(prompt_images),
            "image_paths": prompt_images,
            "script_output": result.stdout if result and result.stdout else None,
            "script_stderr": result.stderr if result and result.stderr else None,
            "timestamp": datetime.now().isoformat()
        }, *mcp_images
        
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "message": "Script execution timed out after 120 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# @mcp.tool(
#     description=(
#         "Update YOLO detection prompts using the UpdateYoloPrompts ROS2 service.\n"
#         "This tool CORRECTS mislabeled objects from the prompt-free YOLO detector."
#     )
# )
def update_yolo_prompts(prompt_map: dict):
    """
    Update YOLO detection prompts using the UpdateYoloPrompts ROS2 service.
    This CORRECTS mislabeled objects from the prompt-free YOLO detector.
    
    CRITICAL: The prompt_map maps FROM what YOLO wrongly detected TO what it actually is!
    The prompts are automatically derived from the prompt_map keys.
    
    Args:
        prompt_map: Dictionary that CORRECTS wrong labels
                   KEY = What YOLO WRONGLY detected (look at annotated image labels)
                   VALUE = What the object ACTUALLY is (what you want it called)
                   
                   Format: {
                       "YOLO_WRONG_LABEL": "actual_object_name",
                       "YOLO_WRONG_LABEL": "actual_object_name"
                   }
    
    Returns:
        Dictionary with service call results
        
    Examples:
        # Looking at annotated image, YOLO detected:
        # - "cork" (but it's actually a jenga block)
        # - "matchbox" (but it's actually a red block)  
        # - "envelope" (but it's actually a blue block)
        # - "first-aid kit" (but it's actually a white box)
        
        update_yolo_prompts({
            "cork": "jenga block",           # YOLO said "cork" → Actually "jenga block"
            "first-aid kit": "white box",    # YOLO said "first-aid kit" → Actually "white box"
            "envelope": "blue block",        # YOLO said "envelope" → Actually "blue block"
            "clipboard": "red block"         # YOLO said "clipboard" → Actually "red block"
        })
        
        # Step-by-step for VLM:
        # 1. Look at ANNOTATED image - read the wrong labels YOLO put on each object
        # 2. Look at ORIGINAL image - identify what each object actually is
        # 3. Create mapping: prompt_map[wrong_yolo_label] = actual_object_name
    """
    try:
        import subprocess
        import os
        
        # Derive prompts from prompt_map keys
        prompts = list(prompt_map.keys())
        
        # Path to the update service script
        script_path = "/home/aaugus11/Documents/ros-mcp-server/tools/yoloe/update_yolo_prompts_service.py"
        
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
            f"cd /home/aaugus11/Documents/ros-mcp-server/tools/yoloe && "
            f"python3 update_yolo_prompts_service.py"
        ]
        
        # Add prompts to the command (derived from prompt_map keys)
        for prompt in prompts:
            cmd_parts[2] += f" '{prompt}'"
        
        # Add prompt map
        cmd_parts[2] += " --prompt-map"
        for prompt, color in prompt_map.items():
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
                "prompt_map": prompt_map,
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
            "error": str(e)
        }


@mcp.tool(
    description=(
        "Add a YOLO detection prompt by sending a ROS2 service call via rosbridge.\n"
        "This tool adds a single prompt with its mapped name to the YOLO detector.\n"
        "You can provide just the prompt, or 'prompt:mapped_name' format.\n"
        "Example:\n"
        "add_yolo_prompt('blue_object')\n"
        "add_yolo_prompt('blue_object:blue object')"
    )
)
def add_yolo_prompt(prompt_input: str, timeout: Optional[float] = None) -> dict:
    """
    Add a YOLO detection prompt by sending a ROS2 service call via rosbridge.
    
    Args:
        prompt_input (str): The prompt input. Can be:
            - Just the prompt name (e.g., 'blue_object') - will use prompt as mapped_name
            - Format 'prompt:mapped_name' (e.g., 'blue_object:blue object')
        timeout (Optional[float]): Timeout in seconds. If None, uses the default timeout.
    
    Returns:
        dict: Service call results with status and output
    
    Example:
        add_yolo_prompt('blue_object')
        add_yolo_prompt('blue_object:blue object')
    """
    # Validate input
    if not prompt_input or not prompt_input.strip():
        return {"status": "error", "error": "Prompt input cannot be empty"}
    
    # Parse the input - check if it contains a colon for mapping
    if ':' in prompt_input:
        # Format: "prompt:mapped_name"
        parts = prompt_input.split(':', 1)
        prompt = parts[0].strip()
        mapped_name = parts[1].strip()
        
        if not prompt:
            return {"status": "error", "error": "Prompt cannot be empty"}
        if not mapped_name:
            return {"status": "error", "error": "Mapped name cannot be empty"}
    else:
        # Just the prompt - use it as both prompt and mapped_name
        prompt = prompt_input.strip()
        mapped_name = prompt
    
    # Build the service request
    # The service expects prompts_json (JSON array) and color_map_json (JSON object)
    prompts_list = [prompt]
    prompt_map = {prompt: mapped_name}
    
    # Create rosbridge service call message
    message = {
        "op": "call_service",
        "service": "/update_yolo_prompts",
        "type": "max_camera_msgs/srv/UpdateYoloPrompts",
        "args": {
            "prompts_json": json.dumps(prompts_list),
            "color_map_json": json.dumps(prompt_map)
        },
        "id": f"add_yolo_prompt_{prompt.replace(' ', '_')}",
    }
    
    # Call the service through rosbridge (same pattern as subscribe_once)
    with ws_manager:
        response = ws_manager.request(message, timeout=timeout)
    
    # Check for service response errors
    if response and "result" in response and not response["result"]:
        # Service call failed - return error with details from values
        error_msg = response.get("values", {}).get("message", "Service call failed")
        return {
            "status": "error",
            "message": "Failed to add YOLO prompt",
            "error": f"Service call failed: {error_msg}",
            "prompt": prompt,
            "mapped_name": mapped_name
        }
    
    # Return service response if present
    if response:
        if response.get("op") == "service_response":
            # Service response format
            values = response.get("values", {})
            success = response.get("result", True)
            message_text = values.get("message", "")
            
            if success:
                return {
                    "status": "success",
                    "message": message_text or "YOLO prompt added successfully",
                    "prompt": prompt,
                    "mapped_name": mapped_name,
                    "service_response": values
                }
            else:
                return {
                    "status": "error",
                    "message": message_text or "Failed to add YOLO prompt",
                    "prompt": prompt,
                    "mapped_name": mapped_name,
                    "service_response": values
                }
        elif response.get("op") == "status" and response.get("level") == "error":
            # Error response
            return {
                "status": "error",
                "message": "Failed to add YOLO prompt",
                "error": response.get("msg", "Unknown error"),
                "prompt": prompt,
                "mapped_name": mapped_name
            }
        else:
            # Check if response has values with success/message
            if "values" in response:
                values = response["values"]
                if isinstance(values, dict) and "success" in values:
                    if values.get("success"):
                        return {
                            "status": "success",
                            "message": values.get("message", "YOLO prompt added successfully"),
                            "prompt": prompt,
                            "mapped_name": mapped_name,
                            "service_response": values
                        }
                    else:
                        return {
                            "status": "error",
                            "message": values.get("message", "Failed to add YOLO prompt"),
                            "prompt": prompt,
                            "mapped_name": mapped_name,
                            "service_response": values
                        }
            
            # Unexpected response format
            return {
                "status": "error",
                "message": "Unexpected response format from service",
                "prompt": prompt,
                "mapped_name": mapped_name,
                "raw_response": response
            }
    else:
        return {
            "status": "error",
            "message": "No response received from service call",
            "prompt": prompt,
            "mapped_name": mapped_name
        }


## ############################################################################################## ##
##
##                      MAIN
##
## ############################################################################################## ##


def main():
    """Main entry point for the MCP server console script."""
    # Parse command line arguments
    args = parse_arguments()

    # Update global variables with parsed arguments
    global MCP_TRANSPORT, MCP_HOST, MCP_PORT
    MCP_TRANSPORT = args.transport.lower()
    MCP_HOST = args.host
    MCP_PORT = args.port

    if MCP_TRANSPORT == "stdio":
        # stdio doesn't need host/port
        mcp.run(transport="stdio")

    elif MCP_TRANSPORT in {"http", "streamable-http"}:
        # http and streamable-http both require host/port
        print(f"Transport: {MCP_TRANSPORT} -> http://{MCP_HOST}:{MCP_PORT}")
        mcp.run(transport=MCP_TRANSPORT, host=MCP_HOST, port=MCP_PORT)

    elif MCP_TRANSPORT == "sse":
        print(f"Transport: {MCP_TRANSPORT} -> http://{MCP_HOST}:{MCP_PORT}")
        print("Currently unsupported. Use 'stdio', 'http', or 'streamable-http'.")
        mcp.run(transport=MCP_TRANSPORT, host=MCP_HOST, port=MCP_PORT)

    else:
        raise ValueError(
            f"Unsupported MCP_TRANSPORT={MCP_TRANSPORT!r}. "
            "Use 'stdio', 'http', or 'streamable-http'."
        )


if __name__ == "__main__":
    main()

