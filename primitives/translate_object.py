#!/usr/bin/env python3
"""
Translate Object Primitive - Combines translate_for_assembly and perform_insert

This primitive combines functionality from translate_for_assembly and perform_insert:
- --move-to-base: Calls translate_for_assembly to move to safe height
- --move-down: Calls perform_insert to move down to final position

Note: --move-to-base and --move-down are mutually exclusive (cannot be used together).

Usage:
    # Sim mode - safe height only
    python3 translate_object.py --mode sim --object-name fork_orange --base-name base --move-to-base

    # Sim mode - move down only
    python3 translate_object.py --mode sim --object-name fork_orange --base-name base --move-down

    # Real mode - safe height only
    python3 translate_object.py --mode real --base-name base --move-to-base --final-base-pos 0.5 -0.37 0.1882 --final-base-orientation 0.0 0.0 0.0 1.0

    # Real mode - move down only with force compliance
    python3 translate_object.py --mode real --move-down --speed 0.01 --gain 2.0
"""

import sys
import os
import subprocess
import argparse
import threading

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node


def stream_output(pipe, logger, prefix=""):
    """Stream subprocess output line by line"""
    for line in iter(pipe.readline, ''):
        if line:
            # Remove trailing newline and log
            line = line.rstrip()
            if line:
                logger.info(f"{prefix}{line}")
    pipe.close()


def run_translate_for_assembly(args, logger):
    """Run translate_for_assembly script"""
    script_path = os.path.join(os.path.dirname(__file__), 'translate_for_assembly.py')
    
    cmd = [sys.executable, script_path,
           '--mode', args.mode,
           '--object-name', args.object_name,
           '--base-name', args.base_name]
    
    # Add real mode arguments if provided
    if args.mode == 'real':
        if args.final_base_pos:
            cmd.extend(['--final-base-pos'] + [str(x) for x in args.final_base_pos])
        if args.final_base_orientation:
            cmd.extend(['--final-base-orientation'] + [str(x) for x in args.final_base_orientation])
        if args.use_default_base:
            cmd.append('--use-default-base')
    
    logger.info(f"Translating {args.object_name} relative to {args.base_name}")
    
    # Run subprocess and stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output in a separate thread
    output_thread = threading.Thread(
        target=stream_output,
        args=(process.stdout, logger),
        daemon=True
    )
    output_thread.start()
    
    # Wait for process to complete
    returncode = process.wait()
    output_thread.join(timeout=1.0)
    
    if returncode != 0:
        logger.error(f"translate_for_assembly failed with return code {returncode}")
        return False
    
    logger.info("translate_for_assembly completed successfully")
    return True


def run_perform_insert(args, logger):
    """Run perform_insert script"""
    script_path = os.path.join(os.path.dirname(__file__), 'perform_insert.py')
    
    cmd = [sys.executable, script_path, '--mode', args.mode]
    
    # Add sim mode arguments if provided
    if args.mode == 'sim':
        if args.object_name:
            cmd.extend(['--object-name', args.object_name])
        if args.base_name:
            cmd.extend(['--base-name', args.base_name])
    
    # Add real mode force compliance parameters
    if args.mode == 'real':
        cmd.extend(['--speed', str(args.speed)])
        cmd.extend(['--gain', str(args.gain)])
        cmd.extend(['--deadband', str(args.deadband)])
        cmd.extend(['--max-vel', str(args.max_vel)])
        if args.reverse:
            cmd.append('--reverse')
        cmd.extend(['--z-threshold', str(args.z_threshold)])
        cmd.extend(['--xy-threshold', str(args.xy_threshold)])
    
    if args.mode == 'sim':
        logger.info(f"Moving down {args.object_name} to {args.base_name}")
    else:
        logger.info("Moving down with force compliance")
    
    # Run subprocess and stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output in a separate thread
    output_thread = threading.Thread(
        target=stream_output,
        args=(process.stdout, logger),
        daemon=True
    )
    output_thread.start()
    
    # Wait for process to complete
    returncode = process.wait()
    output_thread.join(timeout=1.0)
    
    if returncode != 0:
        logger.error(f"perform_insert failed with return code {returncode}")
        return False
    
    logger.info("perform_insert completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Translate Object - Combines translate_for_assembly and perform_insert',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sim mode - safe height only
  python3 translate_object.py --mode sim --object-name fork_orange --base-name base --move-to-base

  # Sim mode - move down only
  python3 translate_object.py --mode sim --object-name fork_orange --base-name base --move-down

  # Real mode - safe height only
  python3 translate_object.py --mode real --base-name base --move-to-base --final-base-pos 0.5 -0.37 0.1882 --final-base-orientation 0.0 0.0 0.0 1.0

  # Real mode - safe height only (using defaults)
  python3 translate_object.py --mode real --base-name base --move-to-base --use-default-base

  # Real mode - move down only with force compliance
  python3 translate_object.py --mode real --move-down --speed 0.01 --gain 2.0
        """
    )

    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'],
                       help='Mode: sim or real')
    
    # Common arguments
    parser.add_argument('--object-name', type=str,
                       help='Name of the object being held (required in sim mode)')
    parser.add_argument('--base-name', type=str,
                       help='Name of the base object (required for translate_for_assembly)')

    # Movement mode flags
    parser.add_argument('--move-to-base', action='store_true',
                       help='Call translate_for_assembly to move to safe height')
    parser.add_argument('--move-down', action='store_true',
                       help='Call perform_insert to move down to final position')

    # Real mode arguments for translate_for_assembly
    parser.add_argument('--final-base-pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                       help='Final base position [x, y, z] in meters (for translate_for_assembly in real mode)')
    parser.add_argument('--final-base-orientation', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'),
                       help='Final base orientation quaternion [x, y, z, w] (for translate_for_assembly in real mode)')
    parser.add_argument('--use-default-base', action='store_true',
                       help='Use default base position and orientation (for translate_for_assembly in real mode)')

    # Real mode force compliance parameters (for perform_insert)
    parser.add_argument('--speed', type=float, default=0.005,
                       help='Downward speed in m/s (default: 0.005, real mode only)')
    parser.add_argument('--gain', type=float, default=1.67,
                       help='X/Y compliance gain in mm/s per Newton (default: 1.67, real mode only)')
    parser.add_argument('--deadband', type=float, default=1.0,
                       help='X/Y force deadband in N (default: 1.0, real mode only)')
    parser.add_argument('--max-vel', type=float, default=15.0,
                       help='X/Y max compliance velocity in mm/s (default: 15.0, real mode only)')
    parser.add_argument('--reverse', action='store_true',
                       help='Reverse X/Y force response directions (real mode only)')
    parser.add_argument('--z-threshold', type=float, default=-10.0,
                       help='Z force threshold in N to detect contact (default: -10.0, real mode only)')
    parser.add_argument('--xy-threshold', type=float, default=1.0,
                       help='Minimum X/Y force magnitude for alignment mode (default: 1.0, real mode only)')

    args = parser.parse_args()

    # Validate arguments
    if not args.move_to_base and not args.move_down:
        parser.error("Either --move-to-base or --move-down must be specified")
    
    if args.move_to_base and args.move_down:
        parser.error("Cannot use both --move-to-base and --move-down")

    # Validate sim mode requirements
    if args.mode == 'sim':
        if args.object_name is None:
            parser.error("--object-name is required in sim mode")
        if args.base_name is None and args.move_to_base:
            parser.error("--base-name is required when using --move-to-base in sim mode")
        if args.base_name is None and args.move_down:
            parser.error("--base-name is required when using --move-down in sim mode")

    # Validate real mode requirements for translate_for_assembly
    if args.mode == 'real' and args.move_to_base:
        if args.base_name is None:
            parser.error("--base-name is required when using --move-to-base in real mode")
        if not args.use_default_base and args.final_base_pos is None:
            parser.error("In real mode with --move-to-base, either --final-base-pos or --use-default-base is required")

    # Initialize ROS
    rclpy.init()
    node = Node('translate_object')
    logger = node.get_logger()
    
    logger.info(f"Using {args.mode.upper()} mode")

    try:
        # Call translate_for_assembly if --move-to-base is specified
        if args.move_to_base:
            if not run_translate_for_assembly(args, logger):
                logger.error("translate_for_assembly failed, aborting")
                node.destroy_node()
                rclpy.shutdown()
                sys.exit(1)
            logger.info("Operation completed successfully!")
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(0)

        # Call perform_insert if --move-down is specified
        if args.move_down:
            if not run_perform_insert(args, logger):
                logger.error("perform_insert failed, aborting")
                node.destroy_node()
                rclpy.shutdown()
                sys.exit(1)
            logger.info("Operation completed successfully!")
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()

