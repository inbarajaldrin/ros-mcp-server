#!/usr/bin/env python3
"""
Gripper Control Script
Controls the JETANK gripper in both simulation and real hardware modes.

Usage:
    python3 primitives/control_gripper.py open --mode real
    python3 primitives/control_gripper.py close --mode sim
"""

import argparse
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float32
import time
import threading

# Trajectory duration in seconds
TRAJECTORY_DURATION = 1.0

# Force monitoring configuration
FORCE_THRESHOLD = 0.5  # Stop when both R2 and L2 exceed this value (N)
MAX_GRIPPER_RETRIES = 3  # Maximum retry attempts when force threshold is reached

# Gripper position limits (wrist joint angle in radians)
GRIPPER_MIN_ANGLE = 0.0  # Fully closed
GRIPPER_MAX_ANGLE = 1.22  # Fully open


class GripperController(Node):
    """ROS2 node for controlling the gripper"""
    
    def __init__(self, mode='sim'):
        super().__init__('gripper_controller')
        
        self.mode = mode
        self.use_real_hardware = (mode == 'real')
        
        # Create publisher for joint commands (used for both real and sim)
        # In simulation, the GUI subscribes to this and updates internal state
        self.joint_command_pub = self.create_publisher(
            JointState, 
            'joint_commands', 
            10
        )
        
        # Create publisher for trajectory (simulation mode)
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            'arm_trajectory',
            10
        )
        
        # All joint names for simulation (must match GUI order)
        self.all_joint_names = [
            'revolute_BEARING',                    # 0
            'revolute_FREE_WHEEL_LEFT',            # 1
            'revolute_FREE_WHEEL_RIGHT',           # 2
            'revolute_GRIPPER_L1',                 # 3
            'revolute_GRIPPER_L2',                 # 4
            'Revolute_SERVO_UPPER',                 # 5
            'Revolute_SERVO_LOWER',                 # 6
            'Revolute_DRIVING_WHEEL_R',             # 7
            'Revolute_DRIVING_WHEEL_L',             # 8
            'Revolute_GRIPPER_R2',                  # 9
            'Revolute_GRIPPER_R1',                  # 10
            'revolute_CAMERA_HOLDER_ARM_LOWER'     # 11
        ]
        
        # Gripper joint indices in the full joint state array
        self.gripper_joint_indices = {
            'L1': 3,  # revolute_GRIPPER_L1
            'L2': 4,  # revolute_GRIPPER_L2
            'R1': 10, # Revolute_GRIPPER_R1
            'R2': 9   # Revolute_GRIPPER_R2
        }
        
        # Gripper limits from URDF
        self.gripper_max_angle = 1.047198  # URDF gripper joint limit
        self.wrist_max_angle = 1.22  # Maximum wrist servo angle (fully open)
        
        # Force monitoring for gripper (simulation mode only)
        self.gripper_force_r2 = 0.0
        self.gripper_force_l2 = 0.0
        self.force_lock = threading.Lock()
        self.force_threshold = FORCE_THRESHOLD
        self.gripper_retry_count = 0
        self.max_gripper_retries = MAX_GRIPPER_RETRIES
        
        # Create subscribers for gripper force topics (simulation only)
        if not self.use_real_hardware:
            self.force_r2_sub = self.create_subscription(
                Float32,
                '/gripper_r2/contact_force',
                self.force_r2_callback,
                10
            )
            
            self.force_l2_sub = self.create_subscription(
                Float32,
                '/gripper_l2/contact_force',
                self.force_l2_callback,
                10
            )
        
        # Joint state storage for reading wrist joint position
        self.joint_state_lock = threading.Lock()
        self.current_joint_state = None
        
        # Subscribe to joint states to read wrist joint position
        if self.use_real_hardware:
            # Real mode: subscribe to real_joint_states from servo driver
            self.joint_state_sub = self.create_subscription(
                JointState,
                'real_joint_states',
                self.joint_state_callback,
                10
            )
        else:
            # Sim mode: subscribe to joint_states
            self.joint_state_sub = self.create_subscription(
                JointState,
                'joint_states',
                self.joint_state_callback,
                10
            )
    
    def gripper_to_wrist_angle(self, gripper_angle):
        """Convert gripper finger angle to wrist servo angle"""
        # Reverse of wrist_to_gripper_angle
        # Actual wrist servo max range: 0.0 (closed) to 1.22 (fully open) radians
        wrist_angle = gripper_angle * self.wrist_max_angle / self.gripper_max_angle
        return wrist_angle
    
    def force_r2_callback(self, msg):
        """Callback for gripper R2 force topic"""
        with self.force_lock:
            self.gripper_force_r2 = abs(msg.data)  # Use absolute value for force
    
    def force_l2_callback(self, msg):
        """Callback for gripper L2 force topic"""
        with self.force_lock:
            self.gripper_force_l2 = abs(msg.data)  # Use absolute value for force
    
    def get_gripper_forces(self):
        """Get current gripper forces (thread-safe)"""
        with self.force_lock:
            return self.gripper_force_r2, self.gripper_force_l2
    
    def joint_state_callback(self, msg):
        """Callback for receiving joint states"""
        with self.joint_state_lock:
            self.current_joint_state = msg
    
    def get_wrist_joint_position(self):
        """Get current wrist joint position (thread-safe)"""
        with self.joint_state_lock:
            if self.current_joint_state is None:
                return None
            
            # Find wrist_joint in real mode, or calculate from gripper joints in sim mode
            if self.use_real_hardware:
                # Real mode: look for wrist_joint
                try:
                    wrist_idx = self.current_joint_state.name.index('wrist_joint')
                    if wrist_idx < len(self.current_joint_state.position):
                        return self.current_joint_state.position[wrist_idx]
                except ValueError:
                    return None
            else:
                # Sim mode: calculate from gripper R1 joint (which represents opening)
                # R1 angle directly represents the gripper opening (0 = closed, max_angle = open)
                try:
                    r1_idx = self.current_joint_state.name.index('Revolute_GRIPPER_R1')
                    if r1_idx < len(self.current_joint_state.position):
                        gripper_angle = abs(self.current_joint_state.position[r1_idx])
                        # Convert gripper angle to wrist angle (0-1.047198 -> 0-1.22)
                        return self.gripper_to_wrist_angle(gripper_angle)
                except ValueError:
                    return None
            
            return None
    
    def send_gripper_command_real(self, target_wrist_angle, duration=TRAJECTORY_DURATION, current_wrist_angle=None):
        """Send gripper (wrist) command to real hardware with trajectory over duration
        Matches GUI behavior: sends commands continuously at 50Hz during trajectory execution
        """
        if current_wrist_angle is None:
            # Assume starting from current position (0.0 if unknown)
            current_wrist_angle = 0.0
        
        start_time = time.time()
        
        # Send commands continuously at 50Hz during trajectory execution (matching GUI)
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            t = min(1.0, elapsed / duration)
            
            # Smooth interpolation using cubic easing (matching GUI)
            t_smooth = 3 * t**2 - 2 * t**3  # Smooth start and end
            
            # Interpolate wrist angle
            current_angle = current_wrist_angle + (target_wrist_angle - current_wrist_angle) * t_smooth
            
            # Send command
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ['wrist_joint']
            msg.position = [current_angle]
            
            self.joint_command_pub.publish(msg)
            time.sleep(0.02)  # 50Hz update rate (matching GUI)
        
        # Ensure final position is set
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['wrist_joint']
        msg.position = [target_wrist_angle]
        self.joint_command_pub.publish(msg)
    
    def send_gripper_command_sim(self, target_gripper_angles, duration=TRAJECTORY_DURATION, current_gripper_angles=None, is_closing=False):
        """Send gripper command to simulation via trajectory
        Matches GUI behavior: publishes trajectory once, then sends joint_commands continuously
        Monitors forces when closing and stops if threshold is reached
        """
        gripper_joint_names = [
            'revolute_GRIPPER_L1',
            'revolute_GRIPPER_L2',
            'Revolute_GRIPPER_R1',
            'Revolute_GRIPPER_R2'
        ]
        
        if current_gripper_angles is None:
            # Assume starting from current position (0.0 if unknown)
            current_gripper_angles = [0.0, 0.0, 0.0, 0.0]
        
        # Reset retry counter for new operation
        self.gripper_retry_count = 0
        force_exceeded = False
        previous_above_threshold = False
        
        # Calculate number of steps (50 steps per second for smooth motion) - matching GUI
        steps = max(10, int(duration * 50))
        
        # Create trajectory points (matching GUI implementation)
        trajectory_points = []
        
        for i in range(steps + 1):
            # Calculate interpolation factor (0 to 1)
            t = i / steps
            
            # Smooth interpolation using cubic easing (matching GUI)
            t_smooth = 3 * t**2 - 2 * t**3  # Smooth start and end
            
            # Interpolate joint positions
            current_positions = []
            for j in range(4):  # 4 gripper joints
                pos = current_gripper_angles[j] + (target_gripper_angles[j] - current_gripper_angles[j]) * t_smooth
                current_positions.append(pos)
            
            # Create trajectory point (matching GUI format)
            point = JointTrajectoryPoint()
            point.positions = current_positions
            point.time_from_start = Duration(sec=int(t * duration), nanosec=int(((t * duration) % 1) * 1e9))
            
            trajectory_points.append(point)
        
        # Create and publish trajectory (matching GUI)
        traj = JointTrajectory()
        traj.joint_names = gripper_joint_names
        traj.points = trajectory_points
        
        # Publish trajectory once (matching GUI)
        self.trajectory_pub.publish(traj)
        
        # Also send joint_commands continuously during trajectory execution (like GUI does)
        # The GUI subscribes to joint_commands in sim mode and updates internal state
        start_time = time.time()
        start_joints = current_gripper_angles.copy()
        final_positions = target_gripper_angles  # Default to target
        
        while time.time() - start_time < duration and not force_exceeded:
            elapsed = time.time() - start_time
            t = min(1.0, elapsed / duration)
            
            # Smooth interpolation using cubic easing (matching GUI)
            t_smooth = 3 * t**2 - 2 * t**3
            
            # Calculate current positions
            current_positions = []
            for j in range(4):  # 4 gripper joints
                pos = start_joints[j] + (target_gripper_angles[j] - start_joints[j]) * t_smooth
                current_positions.append(pos)
            
            # Check forces if closing (matching GUI behavior)
            if is_closing:
                force_r2, force_l2 = self.get_gripper_forces()
                
                # Check if both forces exceed threshold
                current_above_threshold = (force_r2 >= self.force_threshold and force_l2 >= self.force_threshold)
                
                # Detect threshold crossing (went from below to above threshold)
                if current_above_threshold and not previous_above_threshold:
                    # Threshold just crossed - increment retry count
                    self.gripper_retry_count += 1
                    
                    self.get_logger().info(f'Force threshold reached (retry {self.gripper_retry_count}/{self.max_gripper_retries}): R2={force_r2:.3f}N, L2={force_l2:.3f}N (threshold={self.force_threshold}N)')
                    
                    # If we've reached max retries, stop completely
                    if self.gripper_retry_count > self.max_gripper_retries:
                        force_exceeded = True
                        final_positions = current_positions
                        self.get_logger().info(f'Max retries reached. Stopping gripper at current position.')
                        break
                    
                    # Otherwise, wait 1 second before continuing (retry)
                    # Retry mechanism helps handle slip in simulation and ensures object is securely held
                    time.sleep(1.0)
                    
                    # Check if forces are still above threshold after delay
                    force_r2_check, force_l2_check = self.get_gripper_forces()
                    if force_r2_check >= self.force_threshold and force_l2_check >= self.force_threshold:
                        # Forces still above threshold - stop instead of continuing
                        force_exceeded = True
                        final_positions = current_positions
                        self.get_logger().info(f'Forces still above threshold after delay. Stopping gripper.')
                        break
                    
                    # Forces dropped below threshold - continue closing from current position
                    # Update start positions to current position for next retry attempt
                    start_joints = current_positions.copy()
                    start_time = time.time()  # Reset timer for next retry
                    previous_above_threshold = False  # Reset to detect next threshold crossing
                    continue
                
                previous_above_threshold = current_above_threshold
            
            # Send joint command (GUI subscribes to this in sim mode)
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = gripper_joint_names
            msg.position = current_positions
            
            self.joint_command_pub.publish(msg)
            time.sleep(0.02)  # 50Hz update rate (matching GUI)
        
        # Ensure final position is set (or current position if force stopped)
        if force_exceeded:
            self.get_logger().info(f'Gripper stopped due to force threshold. Final position: L1={final_positions[0]:.3f}, L2={final_positions[1]:.3f}, R1={final_positions[2]:.3f}, R2={final_positions[3]:.3f}')
        
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = gripper_joint_names
        msg.position = final_positions
        self.joint_command_pub.publish(msg)
    
    def wait_for_joint_state_settle(self, timeout=5.0, stability_threshold=0.01, required_stable_readings=5):
        """Wait for joint state to settle and return the stable value"""
        stable_value = None
        stable_count = 0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            current_value = self.get_wrist_joint_position()
            
            if current_value is not None:
                if stable_value is None:
                    stable_value = current_value
                    stable_count = 1
                elif abs(current_value - stable_value) < stability_threshold:
                    stable_count += 1
                    if stable_count >= required_stable_readings:
                        return current_value
                else:
                    # Value changed, reset stability check
                    stable_value = current_value
                    stable_count = 1
            
            time.sleep(0.1)
        
        # Return the last stable value we got, or None if we never got one
        return stable_value
    
    def log_gripper_position(self, initial_value=None, final_value=None):
        """Log gripper position in the specified format"""
        self.get_logger().info(f"Gripper range: {GRIPPER_MIN_ANGLE:.2f} - {GRIPPER_MAX_ANGLE:.2f} rad")
        
        if initial_value is not None:
            self.get_logger().info(f"Initial angle: {initial_value:.2f} rad")
        
        if final_value is not None:
            self.get_logger().info(f"Current angle: {final_value:.2f} rad")
    
    def open_gripper(self, duration=TRAJECTORY_DURATION):
        """Open the gripper with trajectory over specified duration"""
        # Read initial position - wait longer to ensure we get joint state
        initial_value = None
        for _ in range(30):  # Try to get initial position (up to 3 seconds)
            rclpy.spin_once(self, timeout_sec=0.1)
            initial_value = self.get_wrist_joint_position()
            if initial_value is not None:
                break
            time.sleep(0.1)
        
        if self.use_real_hardware:
            # Real hardware: send wrist joint trajectory
            target_wrist_angle = self.wrist_max_angle  # 1.22 rad = fully open
            self.send_gripper_command_real(target_wrist_angle, duration=duration)
        else:
            # Simulation: send individual gripper joint trajectory
            max_angle = self.gripper_max_angle
            target_gripper_angles = [
                -max_angle,  # L1 (negative)
                -max_angle,  # L2 (negative)
                max_angle,   # R1 (positive)
                -max_angle   # R2 (negative)
            ]
            self.send_gripper_command_sim(target_gripper_angles, duration=duration)
        
        # Wait for trajectory to complete
        time.sleep(duration + 0.2)
        
        # Wait for joint state to settle after trajectory completion
        final_value = self.wait_for_joint_state_settle(timeout=1.0)
        
        # Log gripper position
        self.log_gripper_position(initial_value, final_value)
    
    def close_gripper(self, duration=TRAJECTORY_DURATION):
        """Close the gripper with trajectory over specified duration"""
        # Read initial position - wait longer to ensure we get joint state
        initial_value = None
        for _ in range(30):  # Try to get initial position (up to 3 seconds)
            rclpy.spin_once(self, timeout_sec=0.1)
            initial_value = self.get_wrist_joint_position()
            if initial_value is not None:
                break
            time.sleep(0.1)
        
        if self.use_real_hardware:
            # Real hardware: send wrist joint trajectory
            target_wrist_angle = 0.0  # 0 = closed
            self.send_gripper_command_real(target_wrist_angle, duration=duration)
        else:
            # Simulation: send individual gripper joint trajectory with force monitoring
            target_gripper_angles = [0.0, 0.0, 0.0, 0.0]  # All closed
            self.send_gripper_command_sim(target_gripper_angles, duration=duration, is_closing=True)
        
        # Wait for trajectory to complete
        time.sleep(duration + 0.2)
        
        # Wait for joint state to settle after trajectory completion
        final_value = self.wait_for_joint_state_settle(timeout=3.0)
        
        # Log gripper position
        self.log_gripper_position(initial_value, final_value)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Control JETANK gripper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 primitives/control_gripper.py open --mode real
  python3 primitives/control_gripper.py close --mode sim
        """
    )
    
    parser.add_argument(
        'action',
        choices=['open', 'close'],
        help='Gripper action: open or close'
    )
    
    parser.add_argument(
        '--mode',
        choices=['real', 'sim'],
        default='sim',
        help='Hardware mode: real for real hardware, sim for simulation (default: sim)'
    )
    
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create gripper controller node
        controller = GripperController(mode=args.mode)
        
        # Give publisher time to establish connection (DDS discovery)
        # This is critical for the message to actually be sent
        for _ in range(20):  # Wait up to 2 seconds for discovery
            rclpy.spin_once(controller, timeout_sec=0.1)
        time.sleep(0.5)  # Additional buffer for DDS discovery
        
        # Execute the requested action with trajectory
        trajectory_duration = TRAJECTORY_DURATION
        if args.action == 'open':
            controller.open_gripper(duration=trajectory_duration)
        elif args.action == 'close':
            controller.close_gripper(duration=trajectory_duration)
        
        # Give some time for the trajectory to complete and ensure it's sent
        # Spin a few times to process callbacks and ensure message delivery
        for _ in range(10):
            rclpy.spin_once(controller, timeout_sec=0.1)
        
        # Clean shutdown
        controller.destroy_node()
        rclpy.shutdown()
        
        return 0
        
    except KeyboardInterrupt:
        print('\nInterrupted by user')
        return 1
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())

