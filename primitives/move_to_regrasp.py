#!/usr/bin/env python3
"""
Move to Regrasp Primitive

This primitive supports different behaviors via flags (at least one must be specified):
- --move-to-clear-space: Only moves to clear area (no move down)
- --move-down: Only moves down (skips clear area)
- --move-to-safe-height: Moves to safe height (hover) position only

Requires --mode argument (sim or real) to be specified.
"""

import sys
import os
import subprocess
import argparse

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node


class MoveToRegrasp(Node):
    def __init__(self, mode, move_to_clear_space=False, move_down=False, move_to_safe_height=False):
        super().__init__('move_to_regrasp')
        self.mode = mode
        self.move_to_clear_space = move_to_clear_space
        self.move_down = move_down
        self.move_to_safe_height = move_to_safe_height
        self.sequence_complete = False
        self.get_logger().info(f"Using {self.mode.upper()} mode")
        if self.move_to_safe_height:
            self.get_logger().info("Starting move to regrasp sequence: move to safe height (hover)")
        elif self.move_down:
            self.get_logger().info("Starting move to regrasp sequence: move down")
        elif self.move_to_clear_space:
            self.get_logger().info("Starting move to regrasp sequence: move to clear area")
        else:
            # This should not happen if validation is correct, but keep for safety
            self.get_logger().info("Starting move to regrasp sequence: move to clear area -> move down")
    
    def execute_sequence(self):
        """Execute the move to regrasp sequence"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Handle move_to_safe_height: only move to safe height (hover)
        if self.move_to_safe_height:
            self.get_logger().info("Moving to safe height (hover position)")
            move_mode = "--hover"
            
            try:
                cmd_parts = [
                    f"cd {script_dir}",
                    f"timeout 45 /usr/bin/python3 move_to_clear_area.py {move_mode}"
                ]
                cmd = "\n".join(cmd_parts)
                
                result = subprocess.run(
                    cmd,
                    shell=True,
                    executable='/bin/bash',
                    capture_output=True,
                    text=True,
                    timeout=55
                )
                
                # Log output
                if result.stdout:
                    self.get_logger().info(f"Move to clear area output: {result.stdout}")
                if result.stderr:
                    self.get_logger().warn(f"Move to clear area stderr: {result.stderr}")
                
                if result.returncode != 0:
                    self.get_logger().error(f"Move to clear area failed with return code: {result.returncode}")
                else:
                    self.get_logger().info("Move to regrasp sequence completed successfully (safe height only)")
                    
            except subprocess.TimeoutExpired:
                self.get_logger().error("Move to clear area timed out")
            except Exception as e:
                self.get_logger().error(f"Failed to execute move to clear area: {e}")
            finally:
                self.sequence_complete = True
            return
        
        # Handle move_to_clear_space: only move to clear area (no move down)
        if self.move_to_clear_space:
            self.get_logger().info("Moving to clear area")
            move_mode = "--move"
            
            try:
                cmd_parts = [
                    f"cd {script_dir}",
                    f"timeout 45 /usr/bin/python3 move_to_clear_area.py {move_mode}"
                ]
                cmd = "\n".join(cmd_parts)
                
                result = subprocess.run(
                    cmd,
                    shell=True,
                    executable='/bin/bash',
                    capture_output=True,
                    text=True,
                    timeout=55
                )
                
                # Log output
                if result.stdout:
                    self.get_logger().info(f"Move to clear area output: {result.stdout}")
                if result.stderr:
                    self.get_logger().warn(f"Move to clear area stderr: {result.stderr}")
                
                if result.returncode != 0:
                    self.get_logger().error(f"Move to clear area failed with return code: {result.returncode}")
                else:
                    self.get_logger().info("Move to regrasp sequence completed successfully (clear area only)")
                    
            except subprocess.TimeoutExpired:
                self.get_logger().error("Move to clear area timed out")
            except Exception as e:
                self.get_logger().error(f"Failed to execute move to clear area: {e}")
            finally:
                self.sequence_complete = True
            return
        
        # Handle move_down: only move down (skip clear area)
        if self.move_down:
            self.get_logger().info("Calling move down")
            try:
                cmd_parts = [
                    f"cd {script_dir}",
                    f"timeout 300 /usr/bin/python3 move_down.py --mode {self.mode}"
                ]
                cmd = "\n".join(cmd_parts)
                
                result = subprocess.run(
                    cmd,
                    shell=True,
                    executable='/bin/bash',
                    capture_output=True,
                    text=True,
                    timeout=310
                )
                
                # Log output
                if result.stdout:
                    self.get_logger().info(f"Move down output: {result.stdout}")
                if result.stderr:
                    self.get_logger().warn(f"Move down stderr: {result.stderr}")
                
                if result.returncode != 0:
                    self.get_logger().error(f"Move down failed with return code: {result.returncode}")
                else:
                    self.get_logger().info("Move to regrasp sequence completed successfully")
                    
            except subprocess.TimeoutExpired:
                self.get_logger().error("Move down timed out")
            except Exception as e:
                self.get_logger().error(f"Failed to execute move down: {e}")
            finally:
                self.sequence_complete = True
            return
        
        # Default: move to clear area -> move down
        # Step 1: Move to clear area
        self.get_logger().info("Moving to clear area")
        move_mode = "--move"
        
        try:
            cmd_parts = [
                f"cd {script_dir}",
                f"timeout 45 /usr/bin/python3 move_to_clear_area.py {move_mode}"
            ]
            cmd = "\n".join(cmd_parts)
            
            result = subprocess.run(
                cmd,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True,
                timeout=55
            )
            
            # Log output
            if result.stdout:
                self.get_logger().info(f"Move to clear area output: {result.stdout}")
            if result.stderr:
                self.get_logger().warn(f"Move to clear area stderr: {result.stderr}")
            
            if result.returncode != 0:
                self.get_logger().error(f"Move to clear area failed with return code: {result.returncode}")
                self.sequence_complete = True
                return
                
        except subprocess.TimeoutExpired:
            self.get_logger().error("Move to clear area timed out")
            self.sequence_complete = True
            return
        except Exception as e:
            self.get_logger().error(f"Failed to execute move to clear area: {e}")
            self.sequence_complete = True
            return
        
        # Step 2: Move down
        self.get_logger().info("Calling move down")
        try:
            cmd_parts = [
                f"cd {script_dir}",
                f"timeout 300 /usr/bin/python3 move_down.py --mode {self.mode}"
            ]
            cmd = "\n".join(cmd_parts)
            
            result = subprocess.run(
                cmd,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True,
                timeout=310
            )
            
            # Log output
            if result.stdout:
                self.get_logger().info(f"Move down output: {result.stdout}")
            if result.stderr:
                self.get_logger().warn(f"Move down stderr: {result.stderr}")
            
            if result.returncode != 0:
                self.get_logger().error(f"Move down failed with return code: {result.returncode}")
            else:
                self.get_logger().info("Move to regrasp sequence completed successfully")
                
        except subprocess.TimeoutExpired:
            self.get_logger().error("Move down timed out")
        except Exception as e:
            self.get_logger().error(f"Failed to execute move down: {e}")
        finally:
            self.sequence_complete = True


def main(args=None):
    parser = argparse.ArgumentParser(description='Move to Regrasp Primitive')
    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: "sim" for simulation, "real" for real robot (required)')
    parser.add_argument('--move-to-clear-space', action='store_true',
                       help='Only move to clear area (no move down)')
    parser.add_argument('--move-down', action='store_true',
                       help='Only move down (skip clear area)')
    parser.add_argument('--move-to-safe-height', action='store_true',
                       help='Move to safe height (hover) position only')
    
    # Parse known args to avoid conflicts with ROS2
    known_args, unknown_args = parser.parse_known_args()
    
    # Validate that at least one action flag is provided
    if not (known_args.move_to_clear_space or known_args.move_down or known_args.move_to_safe_height):
        parser.error('At least one of --move-to-clear-space, --move-down, or --move-to-safe-height must be specified')
    
    rclpy.init(args=args)
    node = MoveToRegrasp(
        known_args.mode, 
        known_args.move_to_clear_space,
        known_args.move_down,
        known_args.move_to_safe_height
    )
    
    try:
        # Execute sequence synchronously (no need to spin since we're just using logger)
        node.execute_sequence()
    except KeyboardInterrupt:
        node.get_logger().info("Move to regrasp interrupted by user")
    except Exception as e:
        node.get_logger().error(f"Error in move to regrasp: {e}")
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except (RuntimeError, AttributeError):
            pass  # Context already shut down or node already destroyed


if __name__ == '__main__':
    main()

