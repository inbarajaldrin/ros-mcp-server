#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sys
import os
import time
import subprocess
import signal
import psutil
from std_msgs.msg import String, Float32, Bool
from std_srvs.srv import Trigger

onrobot_path = "/home/aaugus11/Desktop/ros2_ws/src/onrobot_ros/"
if onrobot_path not in sys.path:
    sys.path.append(onrobot_path)

from onrobot_ros.custom_libraries.onrobot import RG

class RGGripperNode(Node):
    def __init__(self):
        super().__init__('rg_gripper')

        self.declare_parameter('gripper', 'rg2')
        self.declare_parameter('device', '/tmp/ttyUR')
        self.declare_parameter('status_publish_rate', 10.0)

        self.gripper_type = self.get_parameter('gripper').value
        self.device = self.get_parameter('device').value
        self.status_rate = self.get_parameter('status_publish_rate').value

        self.get_logger().info(f"Connecting to {self.gripper_type} gripper at {self.device}")

        # Track socat process
        self.socat_process = None
        self.connection_healthy = False
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5

        # Ensure `socat` is running before trying to connect
        self.ensure_socat()

        # Initialize gripper using serial device
        self.rg = None
        self.initialize_gripper()

        # Subscribe to `/gripper_command` topic
        self.subscription = self.create_subscription(
            String,
            'gripper_command',
            self.command_callback,
            10)

        # Publishers for gripper state feedback
        self.width_pub = self.create_publisher(Float32, 'gripper_width', 10)
        self.width_offset_pub = self.create_publisher(Float32, 'gripper_width_offset', 10)
        self.grasp_detected_pub = self.create_publisher(Bool, 'gripper_grasp_detected', 10)
        self.motion_status_pub = self.create_publisher(Bool, 'gripper_motion_ongoing', 10)
        self.status_pub = self.create_publisher(String, 'gripper_status', 10)

        # Service to manually reconnect
        self.reconnect_srv = self.create_service(
            Trigger,
            'restart_gripper_connection',
            self.handle_reconnect_request)

        # Timer to periodically publish gripper state
        self.status_timer = self.create_timer(1.0/self.status_rate, self.publish_gripper_status)

        # Timer for connection health check (every 5 seconds)
        self.health_check_timer = self.create_timer(5.0, self.check_connection_health)

        self.get_logger().info("Gripper node ready")

    def initialize_gripper(self):
        """Initialize the gripper connection with retries."""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self.rg = RG(gripper=self.gripper_type, device=self.device)
                self.connection_healthy = True
                self.consecutive_errors = 0
                self.get_logger().info("Gripper initialized successfully")
                return
            except Exception as e:
                self.get_logger().warn(f"Failed to initialize gripper (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self.get_logger().info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)

        self.get_logger().error(f"Failed to initialize gripper after {max_retries} attempts")
        self.connection_healthy = False

    def ensure_socat(self):
        """Start socat to forward gripper TCP connection to virtual serial port."""
        self.kill_socat()
        time.sleep(1)

        socat_cmd = f"socat -d -d pty,link={self.device},raw,ignoreeof,waitslave tcp:192.168.1.111:54321"
        self.get_logger().info(f"Starting socat")

        try:
            self.socat_process = subprocess.Popen(
                socat_cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid)
            time.sleep(4)
        except Exception as e:
            self.get_logger().error(f"Failed to start socat: {e}")
            self.socat_process = None

    def kill_socat(self):
        """Safely kill the socat process."""
        try:
            # Kill existing socat processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'socat' in proc.name() and self.device in ' '.join(proc.cmdline()):
                        self.get_logger().info(f"Killing socat process {proc.pid}")
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        time.sleep(0.5)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            self.get_logger().warn(f"Error killing socat: {e}")

    def check_connection_health(self):
        """Periodically check if connection is healthy and attempt recovery if not."""
        if not self.connection_healthy or self.rg is None:
            return

        try:
            # Try a simple read to check if connection is alive
            _ = self.rg.get_width()
            self.consecutive_errors = 0
        except Exception as e:
            self.consecutive_errors += 1
            self.get_logger().warn(f"Connection error ({self.consecutive_errors}/{self.max_consecutive_errors}): {e}")

            if self.consecutive_errors >= self.max_consecutive_errors:
                self.get_logger().error("Connection lost! Attempting automatic recovery...")
                self.reconnect_gripper()

    def reconnect_gripper(self):
        """Reconnect to the gripper by restarting socat and reinitializing the client."""
        try:
            self.get_logger().info("Attempting to reconnect gripper...")

            # Close existing connection
            if self.rg:
                try:
                    self.rg.close_connection()
                except:
                    pass

            # Restart socat
            self.ensure_socat()

            # Reinitialize gripper
            self.initialize_gripper()

            if self.connection_healthy:
                self.get_logger().info("✓ Gripper reconnected successfully!")
            else:
                self.get_logger().error("✗ Failed to reconnect gripper")
        except Exception as e:
            self.get_logger().error(f"Reconnection failed: {e}")
            self.connection_healthy = False

    def handle_reconnect_request(self, request, response):
        """Service handler for manual reconnection."""
        self.get_logger().info("Received manual reconnection request")
        self.reconnect_gripper()

        if self.connection_healthy:
            response.success = True
            response.message = "Gripper reconnected successfully"
        else:
            response.success = False
            response.message = "Failed to reconnect gripper"

        return response


    def command_callback(self, msg):
        """Handles incoming ROS 2 commands for the gripper."""
        if not self.connection_healthy or self.rg is None:
            self.get_logger().warn("Gripper not connected, cannot execute command")
            return

        command = msg.data.lower()

        try:
            if command == "open":
                self.rg.open_gripper()
            elif command == "close":
                self.rg.close_gripper()
            elif command.isdigit():
                width = int(command)
                self.rg.move_gripper(width)
            else:
                self.get_logger().warn("Unknown command received.")
        except Exception as e:
            self.get_logger().error(f"Command execution failed: {e}")
            self.consecutive_errors += 1

    def publish_gripper_status(self):
        """Periodically publish gripper state information."""
        if not self.connection_healthy or self.rg is None:
            return

        try:
            # Get current width
            width = self.rg.get_width()
            if width is not None:
                width_msg = Float32()
                width_msg.data = width
                self.width_pub.publish(width_msg)

            # Get width with offset
            width_offset = self.rg.get_width_with_offset()
            if width_offset is not None:
                width_offset_msg = Float32()
                width_offset_msg.data = width_offset
                self.width_offset_pub.publish(width_offset_msg)

            # Use dedicated functions for cleaner status checking
            # Check grasp detection using dedicated function
            grasp_detected = self.rg.is_grasped()
            grasp_msg = Bool()
            grasp_msg.data = grasp_detected
            self.grasp_detected_pub.publish(grasp_msg)

            # Check motion status using dedicated function
            motion_ongoing = self.rg.is_moving()
            motion_msg = Bool()
            motion_msg.data = motion_ongoing
            self.motion_status_pub.publish(motion_msg)

            # Get detailed status for comprehensive information
            detailed_status = self.rg.get_detailed_status()
            if detailed_status:
                status_msg = String()
                # Surface safety_circuit_1/2 explicitly — when these latch,
                # the firmware silently refuses motor commands. Hiding them
                # behind only Safety1/Safety2 (which are the SWITCH bits, not
                # the CIRCUIT bits) made the "stuck gripper" bug invisible.
                status_text = (
                    f"Motion:{detailed_status['motion_ongoing']}, "
                    f"Grasp:{detailed_status['grasp_detected']}, "
                    f"Width:{detailed_status['current_width']:.1f}mm, "
                    f"Safety1:{detailed_status['safety_switch_1']}, "
                    f"Safety2:{detailed_status['safety_switch_2']}, "
                    f"Circuit1:{detailed_status['safety_circuit_1']}, "
                    f"Circuit2:{detailed_status['safety_circuit_2']}"
                )
                status_msg.data = status_text
                self.status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().debug(f"Error publishing gripper status: {e}")
            self.consecutive_errors += 1

    def cleanup(self):
        """Clean up resources when node is shutting down."""
        self.get_logger().info("Shutting down gripper node")
        if self.rg:
            try:
                self.rg.close_connection()
            except:
                pass
        self.kill_socat()

def main(args=None):
    rclpy.init(args=args)
    node = RGGripperNode()

    def signal_handler(sig, frame):
        node.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
