#!/usr/bin/env python3
"""
Visual Servo ArUco Primitive
Aligns the camera center to an ArUco marker by rotating the base.

Uses aruco_poses topic to get marker position and rotates base bearing joint
to center the marker in the camera view.

Usage:
    python3 primitives/visual_servo_aruco.py --aruco_id 1 --mode real
    python3 primitives/visual_servo_aruco.py --aruco_id 5 --mode sim
"""

import argparse
import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
import time
import threading
import math

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_utils import get_default_mode

# Trajectory duration in seconds
TRAJECTORY_DURATION = 1.0

# Control parameters
ROTATION_GAIN = 1.0  # Gain for calculating target angles from marker position


class VisualServoArUcoController(Node):
    """ROS2 node for visual servoing to ArUco markers"""
    
    def __init__(self, mode=None, aruco_id=1, rotation_gain=ROTATION_GAIN):
        if mode is None:
            mode = get_default_mode()
        super().__init__('visual_servo_aruco_controller')
        
        self.mode = mode
        self.use_real_hardware = (mode == 'real')
        self.aruco_id = aruco_id
        self.target_marker_name = f"aruco_{aruco_id}"
        
        # Control parameters
        self.rotation_gain = rotation_gain
        
        # ArUco poses data storage
        self.aruco_poses_data = {}
        self.aruco_poses_lock = threading.Lock()
        
        # Create subscriber for ArUco poses
        self.aruco_poses_sub = self.create_subscription(
            TFMessage,
            '/aruco_poses',
            self.aruco_poses_callback,
            10
        )
        
        # Create publisher for joint commands
        self.joint_command_pub = self.create_publisher(
            JointState, 
            'joint_commands', 
            10
        )
        
        # Current joint state
        self.current_joint_state = None
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )
        
        
        self.get_logger().info(f'Visual servo ArUco controller initialized in {mode} mode, targeting marker {self.target_marker_name}')
    
    def aruco_poses_callback(self, msg):
        """Callback for aruco_poses topic (TFMessage format)"""
        with self.aruco_poses_lock:
            # Store the latest ArUco poses data
            self.aruco_poses_data = {}
            for transform in msg.transforms:
                # Use child_frame_id as marker name
                marker_name = transform.child_frame_id
                
                # Extract position from transform (already in correct frame, no transformation needed)
                pos = [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ]
                
                # Store position (in meters)
                self.aruco_poses_data[marker_name] = {
                    'position': pos,  # Position in meters
                    'header': transform.header
                }
    
    def joint_state_callback(self, msg):
        """Callback for joint states"""
        self.current_joint_state = msg
    
    def get_marker_position(self, marker_name):
        """Get position of a specific marker by name (returns position in meters)"""
        with self.aruco_poses_lock:
            if marker_name in self.aruco_poses_data:
                return self.aruco_poses_data[marker_name]['position']
            return None
    
    def get_current_bearing_angle(self):
        """Get current base bearing joint angle"""
        if self.current_joint_state is None:
            return None
        
        try:
            bearing_idx = self.current_joint_state.name.index('revolute_BEARING')
            return self.current_joint_state.position[bearing_idx]
        except (ValueError, IndexError):
            return None
    
    def get_current_camera_angle(self):
        """Get current camera tilt angle"""
        if self.current_joint_state is None:
            return None
        
        try:
            camera_idx = self.current_joint_state.name.index('revolute_CAMERA_HOLDER_ARM_LOWER')
            return self.current_joint_state.position[camera_idx]
        except (ValueError, IndexError):
            return None
    
    def send_real_hardware_command(self, joint_name, position, velocity=None):
        """Send joint command to real hardware (matching GUI behavior)"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [joint_name]
        msg.position = [position]
        if velocity is not None:
            msg.velocity = [velocity]
        
        self.joint_command_pub.publish(msg)
    
    def send_bearing_command(self, angle):
        """Send base bearing joint command"""
        self.send_real_hardware_command('base_joint', angle)
    
    def send_camera_command(self, angle):
        """Send camera joint command"""
        self.send_real_hardware_command('camera_joint', angle)
    
    
    def send_joint_trajectory_real(self, start_bearing, start_camera, target_bearing, target_camera, duration):
        """Send trajectory to real hardware (continuous commands at 50Hz)"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            t = min(1.0, elapsed / duration)
            
            # Smooth interpolation using cubic easing
            t_smooth = 3 * t**2 - 2 * t**3
            
            # Interpolate joint positions
            current_bearing = start_bearing + (target_bearing - start_bearing) * t_smooth
            current_camera = start_camera + (target_camera - start_camera) * t_smooth
            
            # Send commands
            self.send_bearing_command(current_bearing)
            self.send_camera_command(current_camera)
            time.sleep(0.02)  # 50Hz update rate
        
        # Ensure final positions are set
        self.send_bearing_command(target_bearing)
        self.send_camera_command(target_camera)
    
    def align_to_marker(self, duration=TRAJECTORY_DURATION):
        """Align to marker by moving base and camera once"""
        self.get_logger().info(f"Aligning to marker {self.target_marker_name}")
        
        # Wait for marker data
        position = None
        for _ in range(50):  # Wait up to 5 seconds
            rclpy.spin_once(self, timeout_sec=0.1)
            position = self.get_marker_position(self.target_marker_name)
            if position is not None:
                break
            time.sleep(0.1)
        
        if position is None:
            error_msg = f"Marker {self.target_marker_name} not found. Make sure the aruco_poses topic is publishing."
            self.get_logger().error(error_msg)
            return False, error_msg
        
        x, y, z = position
        
        self.get_logger().info(f"Marker position: x={x:.4f}m, y={y:.4f}m, z={z:.4f}m")
        
        # Get current angles
        current_bearing = self.get_current_bearing_angle()
        if current_bearing is None:
            current_bearing = 0.0
            self.get_logger().warn("Current bearing angle unknown, assuming 0.0")
        
        current_camera = self.get_current_camera_angle()
        if current_camera is None:
            current_camera = 0.0
            self.get_logger().warn("Current camera angle unknown, assuming 0.0")
        
        # Calculate target angles based on marker position
        # x < 0 means marker is to the left, so rotate base RIGHT (negative) to center it
        # x > 0 means marker is to the right, so rotate base LEFT (positive) to center it
        # y < 0 means marker is below, so tilt camera DOWN (negative) to center it
        # y > 0 means marker is above, so tilt camera UP (positive) to center it
        # Use proportional control with gain
        bearing_delta = self.rotation_gain * x  # Positive x (right) -> positive rotation (left), negative x (left) -> negative rotation (right)
        camera_delta = -self.rotation_gain * y  # Positive y (above) -> negative tilt (down), negative y (below) -> positive tilt (up)
        
        target_bearing = current_bearing + bearing_delta
        target_camera = current_camera + camera_delta
        
        # Clamp to joint limits
        # Base bearing: -1.5708 to 1.5708 rad
        target_bearing = max(-1.5708, min(1.5708, target_bearing))
        # Camera: -0.785398 to 0.785398 rad (±45 degrees)
        target_camera = max(-0.785398, min(0.785398, target_camera))
        
        self.get_logger().info(f"Moving base: {current_bearing:.4f} -> {target_bearing:.4f} rad")
        self.get_logger().info(f"Moving camera: {current_camera:.4f} -> {target_camera:.4f} rad")
        
        # Send trajectory
        if self.use_real_hardware:
            self.send_joint_trajectory_real(current_bearing, current_camera, target_bearing, target_camera, duration)
        else:
            # For simulation, send commands continuously
            start_time = time.time()
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                t = min(1.0, elapsed / duration)
                
                # Smooth interpolation
                t_smooth = 3 * t**2 - 2 * t**3
                
                current_bearing_interp = current_bearing + (target_bearing - current_bearing) * t_smooth
                current_camera_interp = current_camera + (target_camera - current_camera) * t_smooth
                
                self.send_bearing_command(current_bearing_interp)
                self.send_camera_command(current_camera_interp)
                time.sleep(0.02)  # 50Hz update rate
            
            # Ensure final positions are set
            self.send_bearing_command(target_bearing)
            self.send_camera_command(target_camera)
        
        # Wait for trajectory to complete
        time.sleep(duration + 0.2)
        
        self.get_logger().info(f"Alignment completed. Base: {target_bearing:.4f} rad, Camera: {target_camera:.4f} rad")
        return True, f"Successfully aligned to marker. Base: {math.degrees(target_bearing):.1f}°, Camera: {math.degrees(target_camera):.1f}°"


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Visual servo to align camera center to ArUco marker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 primitives/visual_servo_aruco.py --aruco_id 1 --mode real
  python3 primitives/visual_servo_aruco.py --aruco_id 5 --mode sim
  python3 primitives/visual_servo_aruco.py --aruco_id 8 --mode real --gain 1.5
        """
    )
    
    parser.add_argument(
        '--aruco_id',
        type=int,
        required=True,
        help='ArUco marker ID to align to (e.g., 1, 5, 8)'
    )
    
    default_mode = get_default_mode()
    parser.add_argument(
        '--mode',
        choices=['real', 'sim'],
        default=default_mode,
        help=f'Hardware mode: real for real hardware, sim for simulation (default: {default_mode} from config)'
    )
    
    parser.add_argument(
        '--gain',
        type=float,
        default=ROTATION_GAIN,
        help=f'Rotation control gain (default: {ROTATION_GAIN})'
    )
    
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create controller node with parameters
        controller = VisualServoArUcoController(
            mode=args.mode, 
            aruco_id=args.aruco_id,
            rotation_gain=args.gain
        )
        
        # Give subscribers time to establish connection (DDS discovery)
        for _ in range(20):  # Wait up to 2 seconds for discovery
            rclpy.spin_once(controller, timeout_sec=0.1)
        time.sleep(0.5)  # Additional buffer for DDS discovery
        
        # Perform alignment (blocking call)
        success, message = controller.align_to_marker()
        
        # Clean shutdown
        controller.destroy_node()
        rclpy.shutdown()
        
        if success:
            print(f"Success: {message}")
            return 0
        else:
            print(f"Error: {message}")
            return 1
        
    except KeyboardInterrupt:
        print('\nInterrupted by user')
        try:
            controller.running = False
            controller.destroy_node()
            rclpy.shutdown()
        except:
            pass
        return 1
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        try:
            controller.running = False
            controller.destroy_node()
            rclpy.shutdown()
        except:
            pass
        return 1


if __name__ == '__main__':
    exit(main())
