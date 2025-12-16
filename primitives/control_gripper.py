#!/usr/bin/env python3
"""
Gripper Control with Verification

Controls gripper and verifies movement using gripper width readings (both sim and real).
Supports "open", "close", "half-open" (30mm), or numeric values 0-110 (width in mm).

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
import threading

# Gripper range: 0.0 - 110.0mm
GRIPPER_MIN_WIDTH = 0.0
GRIPPER_MAX_WIDTH = 110.0

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
            self.get_logger().info("Using SIM mode: monitoring /gripper_width_sim")
        else:
            # Real mode: use gripper width (Float32)
            self.width_sub = self.create_subscription(
                Float32,
                '/gripper_width',
                self.width_callback,
                10
            )
            self.get_logger().info("Using REAL mode: monitoring /gripper_width")
        
        # Determine target state
        if command.lower() == "open":
            self.target_state = "open"
            self.ros_command = "open"
            self.numeric_value = 1100
        elif command.lower() == "close":
            self.target_state = "close"
            self.ros_command = "close"
            self.numeric_value = 0
        elif command.lower() == "half-open":
            # Half-open: hardcoded to 30mm
            self.target_state = "numeric"
            self.numeric_value = int(30.0 * 10)  # Convert 30mm to 300 (0-1100 range)
            self.ros_command = str(self.numeric_value)
        else:
            # Numeric value 0-110 (convert to 0-1100 for ROS)
            try:
                value = float(command)
                if not (0 <= value <= 110):
                    self.get_logger().error(f"Value {value} out of range. Use 0-110.")
                    sys.exit(1)
                self.target_state = "numeric"
                self.numeric_value = int(value * 10)  # Convert 0-110 to 0-1100
                self.ros_command = str(self.numeric_value)
            except ValueError:
                self.get_logger().error(f"Invalid command '{command}'. Use 'open', 'close', 'half-open', or 0-110.")
                sys.exit(1)
        
        self.get_logger().info(f"Target state: {self.target_state}, ROS command: {self.ros_command}, Mode: {self.mode}")
    
    def width_callback(self, msg):
        """Callback for gripper width readings"""
        self.current_width = msg.data
        self.width_received = True
    
    def get_current_width(self, timeout=2.0):
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
    
    def is_at_target_state(self, width, tolerance=2.0, close_tolerance=10.0):
        """Check if gripper is already at target state"""
        if width is None:
            return False
        
        if self.target_state == "open":
            # Open: should be near max width (110mm)
            return width >= (GRIPPER_MAX_WIDTH - tolerance)
        elif self.target_state == "close":
            # Close: should be near min width (0mm), tolerance 0-10mm
            return width <= (GRIPPER_MIN_WIDTH + close_tolerance)
        else:  # numeric
            # Numeric: should be close to target value
            target_width = self.numeric_value / 10.0  # Convert from 0-1100 to 0-110
            return abs(width - target_width) <= tolerance
    
    def verify_gripper_state(self, initial_value=None):
        """Wait for gripper movement to complete - if it moved from initial, it worked"""
        if initial_value is None:
            initial_value = self.get_current_width(timeout=0.5)
        
        # Check if already at target state
        if initial_value is not None and self.is_at_target_state(initial_value):
            state_str = "open" if self.target_state == "open" else "closed" if self.target_state == "close" else f"{self.numeric_value/10.0:.1f}mm"
            self.get_logger().info(f"✓ Gripper already at target state ({state_str}, width: {initial_value:.2f}mm)")
            return True
        
        # Start monitoring immediately
        self.get_logger().info("Monitoring gripper movement...")
        if initial_value is not None:
            self.get_logger().info(f"Starting from width: {initial_value:.2f}, target: {self.target_state}")
        
        # Monitoring parameters
        max_wait_time = 15.0
        early_retry_time = 2.0  # Retry after 2 seconds if no movement detected
        check_interval = 0.2
        no_change_threshold = 0.3
        movement_threshold = 0.5
        required_stable_no_change = 5
        
        start_time = time.time()
        last_value = initial_value
        no_change_count = 0
        last_change_time = None
        movement_detected = False
        baseline_value = initial_value  # Use first reading as baseline for movement detection
        
        while (time.time() - start_time) < max_wait_time:
            elapsed = time.time() - start_time
            current_value = self.get_current_width(timeout=0.3)
            
            # Set baseline if we don't have one yet
            if baseline_value is None and current_value is not None:
                baseline_value = current_value
                self.get_logger().info(f"Baseline width established: {baseline_value:.2f}")
            
            # Early retry check: if no movement detected within 2 seconds
            if elapsed >= early_retry_time and baseline_value is not None and not movement_detected:
                if current_value is not None and abs(current_value - baseline_value) < movement_threshold:
                    self.get_logger().warn(f"No gripper movement detected within {early_retry_time}s (width unchanged: {current_value:.2f}). Retrying...")
                    return False
            
            if current_value is not None:
                # Detect movement
                if last_value is not None and abs(current_value - last_value) > movement_threshold:
                    if not movement_detected:
                        self.get_logger().info(f"Gripper movement detected: {current_value:.2f} (was {last_value:.2f})")
                        movement_detected = True
                    no_change_count = 0
                    last_change_time = time.time()
                elif last_value is None:
                    self.get_logger().info(f"Monitoring width... current: {current_value:.2f}")
                elif abs(current_value - last_value) <= no_change_threshold:
                    # Value is stable
                    no_change_count += 1
                    if no_change_count >= required_stable_no_change:
                        # Movement completed and stabilized
                        check_value = baseline_value if baseline_value is not None else initial_value
                        if movement_detected or (check_value is not None and self.has_moved(check_value, current_value, movement_threshold)):
                            # Wait a bit more to ensure we have the true final value
                            time.sleep(0.5)
                            # Get one more reading to confirm final value
                            final_confirmed = self.get_current_width(timeout=0.5)
                            if final_confirmed is not None:
                                current_value = final_confirmed
                            self.final_stabilized_width = current_value
                            self.get_logger().info(f"✓ Gripper movement completed and stabilized (width: {current_value:.2f}mm)")
                            return True
                else:
                    no_change_count = 0
                
                last_value = current_value
            
            time.sleep(check_interval)
        
        # Timeout reached - check if movement occurred
        self.get_logger().info("Timeout reached, checking final state...")
        time.sleep(1.0)
        
        # Get final reading with extra wait to ensure it's the true final value
        time.sleep(0.5)
        final_value = self.get_current_width(timeout=1.0)
        
        # Use baseline_value if initial_value was None
        check_value = baseline_value if baseline_value is not None else initial_value
        
        # Store final value
        if final_value is not None:
            self.final_stabilized_width = final_value
        
        # If gripper moved from baseline, consider it successful
        if final_value is not None and check_value is not None and self.has_moved(check_value, final_value, movement_threshold):
            self.get_logger().info(f"✓ Gripper movement detected (baseline: {check_value:.2f}mm, final: {final_value:.2f}mm)")
            return True
        
        # If no movement detected, return False for retry
        baseline_value_str = f"{check_value:.2f}" if check_value is not None else "N/A"
        final_value_str = f"{final_value:.2f}" if final_value is not None else "N/A"
        self.get_logger().warn(f"✗ No gripper movement detected (baseline: {baseline_value_str}, final: {final_value_str})")
        return False
    
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
            
            # Wait for verification to complete (increased timeout to ensure completion)
            monitoring_thread.join(timeout=25.0)
            
            # Check result
            with self.monitoring_lock:
                if self.verification_complete and self.verification_result:
                    # Wait a bit more to ensure final value is truly stable
                    time.sleep(0.3)
                    self.get_logger().info(f"Gripper control successful after {attempt} attempt(s)!")
                    return True
                elif not self.verification_complete:
                    self.get_logger().warn("Verification thread did not complete in time")
            
            if attempt < MAX_RETRIES:
                self.get_logger().warn(f"Verification failed, retrying... (attempt {attempt}/{MAX_RETRIES})")
        
        self.get_logger().error(f"Gripper control failed after {MAX_RETRIES} attempts")
        return False


def main(args=None):
    parser = argparse.ArgumentParser(description='Control gripper with verification')
    parser.add_argument('command', type=str, help='Gripper command: "open", "close", "half-open" (30mm), or 0-110 (width in mm)')
    parser.add_argument('--mode', type=str, default='sim', choices=['sim', 'real'],
                       help='Mode: "sim" for simulation (uses /gripper_width_sim), "real" for real robot (uses /gripper_width). Default: sim')
    
    # Parse known args to avoid conflicts with ROS2
    known_args, unknown_args = parser.parse_known_args()
    
    rclpy.init(args=args)
    
    controller = GripperController(known_args.command, known_args.mode)
    
    # Wait a moment for subscriptions to establish
    time.sleep(0.5)
    
    # Get initial reading (both modes use width now)
    initial_value = controller.get_current_width(timeout=2.0)
    if initial_value is not None:
        controller.get_logger().info(f"Initial gripper width: {initial_value:.2f}")
    
    # Control with verification
    success = controller.control_with_verification(initial_value)
    
    # Wait a bit more to ensure gripper has fully stopped and final value is stable
    time.sleep(0.5)
    
    # Get final reading (both modes use width now)
    # Use stored stabilized width if available, otherwise get current reading
    if controller.final_stabilized_width is not None:
        final_value = controller.final_stabilized_width
        controller.get_logger().info(f"Using stabilized final gripper width: {final_value:.2f}mm")
    else:
        # Get a few readings to ensure we have the true final value
        final_readings = []
        for _ in range(3):
            reading = controller.get_current_width(timeout=0.5)
            if reading is not None:
                final_readings.append(reading)
            time.sleep(0.2)
        
        if final_readings:
            # Use the most recent reading, or average if they're close
            final_value = final_readings[-1]
            if len(final_readings) >= 2:
                # Check if readings are stable (within 0.5mm)
                if abs(final_readings[-1] - final_readings[-2]) <= 0.5:
                    final_value = final_readings[-1]
        else:
            final_value = controller.get_current_width(timeout=2.0)
    
    if final_value is not None:
        controller.get_logger().info(f"Final gripper width: {final_value:.2f}mm")
    
    # Output gripper range and current width
    controller.get_logger().info(f"Gripper range: {GRIPPER_MIN_WIDTH:.1f} - {GRIPPER_MAX_WIDTH:.1f}mm")
    if initial_value is not None and final_value is not None:
        controller.get_logger().info(f"Gripper width: {initial_value:.2f}mm → {final_value:.2f}mm (change: {final_value - initial_value:+.2f}mm)")
    elif final_value is not None:
        controller.get_logger().info(f"Current gripper width: {final_value:.2f}mm")
    
    controller.destroy_node()
    rclpy.shutdown()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

