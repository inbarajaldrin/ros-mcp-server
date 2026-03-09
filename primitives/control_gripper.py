#!/usr/bin/env python3
"""
Gripper Control with Verification

Controls gripper and verifies movement using gripper width readings (both sim and real).
Supports "open", "close", "half-open" (35mm), or numeric values 0-100 (width in mm).

Usage:
    python3 control_gripper.py open [--mode sim|real]
    python3 control_gripper.py close [--mode sim|real]
    python3 control_gripper.py half-open [--mode sim|real]
    python3 control_gripper.py 55 [--mode sim|real]  # 55mm width
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64, Float32
import argparse
import time
import sys
import json
import threading
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Gripper range: 0.0 - 100.0mm
GRIPPER_MIN_WIDTH = 0.0
GRIPPER_MAX_WIDTH = 100.0
GRIPPER_HALF_OPEN_WIDTH = 35.0  # mm

MAX_RETRIES = 3
RETRY_DELAY = 0.5  # Seconds to wait between retries

class GripperController(Node):
    def __init__(self, command, mode='sim'):
        super().__init__('gripper_controller')

        self.command = command
        self.mode = mode
        self.current_width = None
        self.width_received = False

        # Threading synchronization
        self.monitoring_lock = threading.Lock()
        self.verification_complete = False
        self.verification_result = None
        self.final_stabilized_width = None  # Store the final stabilized width
        self.asymmetry = 0.0  # Asymmetry between fingers in mm
        self.asymmetry_threshold = 15.0  # mm - above this means fingers are asymmetric
        self.blocked = False  # Gripper pressing against object (effort detected)
        self.current_effort = 0.0  # Total measured joint effort
        self.effort_threshold = 0.05  # Effort above this means gripper is pressing against something

        # Publisher for gripper commands
        self.gripper_pub = self.create_publisher(String, '/gripper_command', 10)

        # Subscriber based on mode - both use width now
        if self.mode == 'sim':
            # Sim mode: use gripper width sim
            self.width_sub = self.create_subscription(
                Float64,
                '/gripper_width_sim',
                self.width_callback,
                10
            )
            # Effort topic for blocked detection
            self.effort_sub = self.create_subscription(
                Float64,
                '/gripper_effort_sim',
                self.effort_callback,
                10
            )
            # Asymmetry topic for jam detection
            self.asymmetry_sub = self.create_subscription(
                Float64,
                '/gripper_asymmetry_sim',
                self.asymmetry_callback,
                10
            )
            self.get_logger().info("Using SIM mode: monitoring /gripper_width_sim + /gripper_effort_sim + /gripper_asymmetry_sim")
        else:
            # Real mode: use gripper width with fingertip offset (Float32)
            self.width_sub = self.create_subscription(
                Float32,
                '/gripper_width_offset',
                self.width_callback,
                10
            )
            self.get_logger().info("Using REAL mode: monitoring /gripper_width_offset")
        
        # Determine target state
        if command.lower() == "open":
            self.target_state = "open"
            self.ros_command = "open"
            self.numeric_value = 1000
        elif command.lower() == "close":
            self.target_state = "close"
            self.ros_command = "close"
            self.numeric_value = 0
        elif command.lower() == "half-open":
            # Half-open: uses GRIPPER_HALF_OPEN_WIDTH
            self.target_state = "numeric"
            self.numeric_value = int(GRIPPER_HALF_OPEN_WIDTH * 10)  # Convert mm to 0-1000 range
            self.ros_command = str(self.numeric_value)
        else:
            # Numeric value 0-100 (convert to 0-1000 for ROS)
            try:
                value = float(command)
                if not (0 <= value <= 100):
                    self.get_logger().error(f"Value {value} out of range. Use 0-100.")
                    sys.exit(1)
                self.target_state = "numeric"
                self.numeric_value = int(value * 10)  # Convert 0-100 to 0-1000
                self.ros_command = str(self.numeric_value)
            except ValueError:
                self.get_logger().error(f"Invalid command '{command}'. Use 'open', 'close', 'half-open', or 0-100.")
                sys.exit(1)
        
        self.get_logger().info(f"Target state: {self.target_state}, ROS command: {self.ros_command}, Mode: {self.mode}")
    
    def width_callback(self, msg):
        """Callback for gripper width readings (always real average width)."""
        self.current_width = msg.data
        self.width_received = True

    def asymmetry_callback(self, msg):
        """Callback for gripper asymmetry readings (mm difference between fingers)."""
        self.asymmetry = msg.data

    def effort_callback(self, msg):
        """Callback for gripper effort readings. High effort = pressing against object."""
        self.current_effort = msg.data
        self.blocked = self.current_effort > self.effort_threshold
    
    def check_topic_available(self, topic_name, timeout=3.0):
        """Check if topic exists and has publishers"""
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < timeout:
            try:
                # Get topic names and types
                topic_names_and_types = self.get_topic_names_and_types()
                topic_exists = any(topic[0] == topic_name for topic in topic_names_and_types)
                
                if topic_exists:
                    # Check if there are publishers
                    publishers_info = self.get_publishers_info_by_topic(topic_name)
                    if len(publishers_info) > 0:
                        self.get_logger().info(f"Topic {topic_name} exists with {len(publishers_info)} publisher(s)")
                        return True
                    else:
                        self.get_logger().warn(f"Topic {topic_name} exists but has no publishers yet")
                else:
                    self.get_logger().debug(f"Topic {topic_name} not found yet")
                
                # Spin to allow discovery
                rclpy.spin_once(self, timeout_sec=0.2)
            except Exception as e:
                self.get_logger().debug(f"Error checking topic: {e}")
                rclpy.spin_once(self, timeout_sec=0.2)
        
        return False
    
    def get_current_width(self, timeout=5.0):
        """Get current gripper width reading"""
        self.width_received = False
        start_time = time.time()
        
        while rclpy.ok() and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.width_received:
                return self.current_width
        return None
    
    def has_moved(self, initial_width, current_width, movement_threshold=0.5):
        """Check if gripper has moved from initial position"""
        if initial_width is None or current_width is None:
            return False
        return abs(current_width - initial_width) > movement_threshold
    
    def is_at_target_state(self, width, tolerance=3.0, close_tolerance=10.0):
        """Check if gripper is already at target state"""
        if width is None or width < 0:
            return False
        
        if self.target_state == "open":
            # Open: should be near max width (100mm)
            return width >= (GRIPPER_MAX_WIDTH - tolerance)
        elif self.target_state == "close":
            # Close: should be near min width (0mm), tolerance 0-10mm
            return width <= (GRIPPER_MIN_WIDTH + close_tolerance)
        else:  # numeric
            # Numeric: should be close to target value
            target_width = self.numeric_value / 10.0  # Convert from 0-1000 to 0-100
            return abs(width - target_width) <= tolerance
    
    def verify_gripper_state(self, initial_value=None):
        """Wait for gripper movement to complete by monitoring topic for stability"""
        if initial_value is None:
            initial_value = self.get_current_width(timeout=0.5)

        # Check if already at target state (but not if effort is high — means pressing against object)
        if initial_value is not None and self.is_at_target_state(initial_value):
            # Spin a few times to ensure effort/asymmetry callbacks have fired
            if self.mode == 'sim':
                for _ in range(5):
                    rclpy.spin_once(self, timeout_sec=0.05)
            is_asymmetric = self.asymmetry > self.asymmetry_threshold
            if is_asymmetric:
                self.get_logger().warn(f"✗ Gripper jammed: fingers are asymmetric (width: {initial_value:.2f}mm)")
                self.final_stabilized_width = initial_value
                return False
            if self.blocked and self.target_state != "close":
                self.get_logger().warn(f"Width near target ({initial_value:.2f}mm) but effort high ({self.current_effort:.3f}) — gripper is blocked")
                self.final_stabilized_width = initial_value
                return False
            state_str = "open" if self.target_state == "open" else "closed" if self.target_state == "close" else f"{self.numeric_value/10.0:.1f}mm"
            self.get_logger().info(f"✓ Gripper already at target state ({state_str}, width: {initial_value:.2f}mm)")
            return True

        # Start monitoring immediately
        self.get_logger().info("Monitoring gripper movement...")
        if initial_value is not None:
            self.get_logger().info(f"Starting from width: {initial_value:.2f}, target: {self.target_state}")

        # Monitoring parameters - purely based on topic changes
        check_interval = 0.1  # How often to check for new readings
        no_change_threshold = 0.3  # mm - values within this range are considered stable
        movement_threshold = 0.5  # mm - minimum change to consider as movement
        required_stable_readings = 5  # Number of consecutive stable readings to confirm stopped
        max_no_movement_readings = 20  # If no movement for this many readings, retry command

        last_value = initial_value
        no_change_count = 0
        no_movement_count = 0
        movement_detected = False
        baseline_value = initial_value
        readings_count = 0
        grace_readings = 10  # Wait this many readings before checking jammed (let command take effect)

        # Monitor indefinitely until gripper stabilizes
        while rclpy.ok():
            current_value = self.get_current_width(timeout=0.3)
            readings_count += 1

            # Check for asymmetry and blocked — only after grace period so command has time to take effect
            is_asymmetric = self.asymmetry > self.asymmetry_threshold
            # Asymmetry = jammed
            if is_asymmetric and readings_count > grace_readings and no_change_count >= 5:
                self.get_logger().warn(f"✗ Gripper jammed: fingers are asymmetric")
                self.final_stabilized_width = current_value if current_value is not None else initial_value
                return False

            # Blocked (effort detected)
            if self.blocked and readings_count > grace_readings and no_change_count >= 5:
                if self.target_state == "close":
                    movement_from_start = abs(initial_value - current_value) if (initial_value is not None and current_value is not None) else 0
                    if movement_from_start < 3.0:
                        # Already gripping — gripper didn't close further
                        self.get_logger().warn(f"✗ Gripper already closed on object (effort: {self.current_effort:.3f})")
                        self.final_stabilized_width = current_value if current_value is not None else initial_value
                        return False
                    # Significant movement + effort = freshly gripping
                    self.get_logger().info(f"✓ Gripper gripping object (effort: {self.current_effort:.3f})")
                    self.final_stabilized_width = current_value if current_value is not None else initial_value
                    return True
                else:
                    # Opening/half-open + effort = blocked by object
                    self.get_logger().error(f"✗ Gripper blocked: pressing against object (effort: {self.current_effort:.3f})")
                    self.final_stabilized_width = current_value if current_value is not None else initial_value
                    return False

            # Set baseline if we don't have one yet
            if baseline_value is None and current_value is not None:
                baseline_value = current_value
                self.get_logger().info(f"Baseline width established: {baseline_value:.2f}")

            if current_value is not None:
                # Check if gripper is moving (comparing to previous reading)
                if last_value is not None:
                    change = abs(current_value - last_value)

                    if change > movement_threshold:
                        # Gripper is moving
                        if not movement_detected:
                            self.get_logger().info(f"Gripper movement detected: {current_value:.2f}mm (was {last_value:.2f}mm)")
                            movement_detected = True
                        no_change_count = 0
                        no_movement_count = 0
                    elif change <= no_change_threshold:
                        # Gripper appears stable
                        no_change_count += 1

                        # Check if we haven't seen any movement from baseline
                        if baseline_value is not None and not movement_detected:
                            baseline_change = abs(current_value - baseline_value)
                            if baseline_change < movement_threshold:
                                no_movement_count += 1

                                # If no movement detected after many readings, command may have failed
                                if no_movement_count >= max_no_movement_readings:
                                    self.get_logger().warn(f"No gripper movement detected after {no_movement_count} readings (width: {current_value:.2f}mm). Retrying...")
                                    return False

                        # If stable for required number of readings, gripper has stopped
                        if no_change_count >= required_stable_readings:
                            # Verify movement occurred from baseline
                            check_value = baseline_value if baseline_value is not None else initial_value
                            if movement_detected or (check_value is not None and self.has_moved(check_value, current_value, movement_threshold)):
                                # Check if movement is in the correct direction
                                if check_value is not None:
                                    actual_change = current_value - check_value

                                    # Determine expected direction based on target
                                    if self.target_state == "open":
                                        # Opening: width should increase
                                        if actual_change <= 0:
                                            self.get_logger().warn(f"✗ Gripper moved in wrong direction for 'open': {check_value:.2f}mm → {current_value:.2f}mm (change: {actual_change:+.2f}mm). Expected positive change.")
                                            return False
                                    elif self.target_state == "close":
                                        # Closing: width should decrease
                                        if actual_change >= 0:
                                            self.get_logger().warn(f"✗ Gripper moved in wrong direction for 'close': {check_value:.2f}mm → {current_value:.2f}mm (change: {actual_change:+.2f}mm). Expected negative change.")
                                            return False
                                    else:  # numeric target
                                        # Check if we reached the target or moved toward it
                                        target_width = self.numeric_value / 10.0
                                        initial_distance = abs(check_value - target_width)
                                        final_distance = abs(current_value - target_width)

                                        if not self.is_at_target_state(current_value) and final_distance >= initial_distance:
                                            self.get_logger().warn(f"✗ Gripper moved away from target {target_width:.1f}mm: {check_value:.2f}mm → {current_value:.2f}mm")
                                            return False

                                self.final_stabilized_width = current_value
                                self.get_logger().info(f"✓ Gripper stopped and stabilized at {current_value:.2f}mm (change: {current_value - check_value:+.2f}mm)")
                                return True
                            else:
                                # Stable but no movement detected
                                self.get_logger().warn(f"✗ Gripper stable but no movement from baseline ({check_value:.2f}mm to {current_value:.2f}mm)")
                                return False
                    else:
                        # Small change, reset counter
                        no_change_count = 0

                last_value = current_value

            time.sleep(check_interval)
    
    def send_gripper_command(self):
        """Send gripper command via ROS2 topic"""
        msg = String()
        msg.data = self.ros_command
        self.gripper_pub.publish(msg)
        self.get_logger().info(f"Sent gripper command: {self.ros_command}")
        time.sleep(0.1)  # Small delay to ensure message is sent
    
    def verify_gripper_state_threaded(self, initial_value):
        """Verify gripper state in a separate thread - starts monitoring immediately"""
        result = self.verify_gripper_state(initial_value)
        with self.monitoring_lock:
            self.verification_result = result
            self.verification_complete = True
        return result
    
    def control_with_verification(self, initial_value=None):
        """Control gripper with verification and retry logic"""
        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                self.get_logger().info(f"Retry attempt {attempt}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY)

            # Reset verification state
            with self.monitoring_lock:
                self.verification_complete = False
                self.verification_result = None

            # Update initial_value to current width for retries
            if attempt > 1:
                current = self.get_current_width(timeout=0.5)
                if current is not None:
                    initial_value = current

            # Start monitoring thread before sending command (parallel execution)
            monitoring_thread = threading.Thread(
                target=self.verify_gripper_state_threaded,
                args=(initial_value,),
                daemon=True
            )
            monitoring_thread.start()
            time.sleep(0.01)  # Small delay to ensure thread starts

            # Send command (monitoring is already active)
            self.send_gripper_command()

            # Wait for verification to complete (no timeout - relies on topic monitoring)
            monitoring_thread.join()

            # Check result
            with self.monitoring_lock:
                if self.verification_complete and self.verification_result:
                    self.get_logger().info(f"Gripper control successful after {attempt} attempt(s)!")
                    return True

            if attempt < MAX_RETRIES:
                self.get_logger().warn(f"Verification failed, retrying... (attempt {attempt}/{MAX_RETRIES})")

        self.get_logger().error(f"Gripper control failed after {MAX_RETRIES} attempts")
        return False


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


def main(args=None):
    parser = argparse.ArgumentParser(description='Control gripper with verification')
    parser.add_argument('command', type=str, help='Gripper command: "open", "close", "half-open" (35mm), or 0-100 (width in mm)')
    parser.add_argument('--mode', type=str, default='sim', choices=['sim', 'real'],
                       help='Mode: "sim" for simulation (uses /gripper_width_sim), "real" for real robot (uses /gripper_width). Default: sim')

    # Parse known args to avoid conflicts with ROS2
    known_args, unknown_args = parser.parse_known_args()

    # Initialize variables for result
    success = False
    initial_value = None
    final_value = None
    error = None
    controller = None

    try:
        rclpy.init(args=args)

        controller = GripperController(known_args.command, known_args.mode)

        # Brief spin for subscriptions to establish (replaces fixed sleep)
        for _ in range(3):
            rclpy.spin_once(controller, timeout_sec=0.05)

        # Check if topic exists and has publishers
        topic_name = '/gripper_width' if known_args.mode == 'real' else '/gripper_width_sim'
        controller.get_logger().info(f"Checking if topic {topic_name} is available...")
        
        # First check if topic exists and has publishers
        if not controller.check_topic_available(topic_name, timeout=0.5):
            error_msg = f"Topic {topic_name} not found or has no publishers. Cannot proceed with gripper control."
            controller.get_logger().error(error_msg)
            error = error_msg
            success = False
            # Build result and exit immediately
            result = {
                "result": "failure",
                "command": known_args.command,
                "mode": known_args.mode,
                "initial_width_mm": None,
                "final_width_mm": None,
                "change_mm": None,
                "error": error
            }
            output_result(result)
            sys.exit(1)
        
        # Topic exists and has publishers, now wait for a message (with longer timeout)
        controller.get_logger().info(f"Topic {topic_name} is available, waiting for first message...")
        initial_value = controller.get_current_width(timeout=5.0)
        
        if initial_value is None:
            error_msg = f"Topic {topic_name} exists but no message received within 5 seconds. The publisher may be publishing too slowly or there may be a QoS mismatch."
            controller.get_logger().error(error_msg)
            error = error_msg
            success = False
            # Build result and exit immediately
            result = {
                "result": "failure",
                "command": known_args.command,
                "mode": known_args.mode,
                "initial_width_mm": None,
                "final_width_mm": None,
                "change_mm": None,
                "error": error
            }
            output_result(result)
            sys.exit(1)
        
        controller.get_logger().info(f"Initial gripper width: {initial_value:.2f}")

        # Control with verification
        success = controller.control_with_verification(initial_value)

        if controller.asymmetry > controller.asymmetry_threshold:
            success = False
            error = "Gripper jammed: fingers are asymmetric."
        elif not success:
            if controller.blocked:
                if known_args.command == "close":
                    error = "Gripper already holding an object."
                else:
                    error = "Gripper blocked: pressing against object its holding, cannot reach target width."
            else:
                error = "Gripper verification failed after retries"

        # Get final reading - use stored stabilized width from monitoring
        # If None (e.g., already at target), use initial value as final
        final_value = controller.final_stabilized_width if controller.final_stabilized_width is not None else initial_value

        if final_value is not None:
            controller.get_logger().info(f"Final gripper width: {final_value:.2f}mm")

        # Output gripper range and width change (skip change for jammed/blocked/recovery — misleading)
        controller.get_logger().info(f"Gripper range: {GRIPPER_MIN_WIDTH:.1f} - {GRIPPER_MAX_WIDTH:.1f}mm")
        is_clean_success = success and not (controller.asymmetry > controller.asymmetry_threshold)
        if is_clean_success and initial_value is not None and final_value is not None:
            controller.get_logger().info(f"Gripper width: {initial_value:.2f}mm → {final_value:.2f}mm (change: {final_value - initial_value:+.2f}mm)")

    except Exception as e:
        success = False
        error = str(e)
    finally:
        # Clean up ROS
        try:
            if controller is not None:
                controller.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass  # Ignore cleanup errors

    # Build structured result
    result_str = "success" if success else "failure"
    result = {
        "result": result_str,
        "command": known_args.command,
        "mode": known_args.mode,
    }

    # Width fields: only final_width for jammed/blocked (initial and change are misleading)
    is_blocked = controller is not None and controller.blocked and not success
    if is_blocked:
        result["final_width_mm"] = round(final_value, 2) if final_value is not None else None
    else:
        result["initial_width_mm"] = round(initial_value, 2) if initial_value is not None else None
        result["final_width_mm"] = round(final_value, 2) if final_value is not None else None
        result["change_mm"] = round(final_value - initial_value, 2) if (initial_value is not None and final_value is not None) else None

    # Add error if failed
    if not success and error:
        result["error"] = error

    # Output JSON markers
    output_result(result)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

