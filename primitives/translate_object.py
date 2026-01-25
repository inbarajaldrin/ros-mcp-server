#!/usr/bin/env python3
"""
Translate Object Primitive - Combines translate_for_assembly and perform_insert

This primitive combines functionality from translate_for_assembly and perform_insert:
- --move-to-base: Calls translate_for_assembly to move to safe height
- --move-down: Calls perform_insert to move down to final position
- --move-to-safe-height: Moves to safe height (after closing gripper)
- --move-away-from-base: Calls move_to_clear_area to move object away from base to a clear area

Note: --move-to-base, --move-down, --move-to-safe-height, and --move-away-from-base are mutually exclusive (cannot be used together).

Usage:
    # Sim mode - safe height only
    python3 translate_object.py --mode sim --object-name fork_orange --base-name base --move-to-base

    # Sim mode - move down only
    python3 translate_object.py --mode sim --object-name fork_orange --base-name base --move-down

    # Sim mode - move to safe height only
    python3 translate_object.py --mode sim --move-to-safe-height

    # Real mode - safe height only
    python3 translate_object.py --mode real --base-name base --move-to-base --final-base-pos 0.5 -0.37 0.1882 --final-base-orientation 0.0 0.0 0.0 1.0

    # Real mode - move down only with force compliance
    python3 translate_object.py --mode real --move-down --speed 0.01 --gain 2.0

    # Real mode - move to safe height only
    python3 translate_object.py --mode real --move-to-safe-height

    # Move away from base (to clear area)
    python3 translate_object.py --mode real --move-away-from-base
"""

import sys
import os
import subprocess
import argparse
import threading
import json

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node


def output_result(result):
    """Output JSON result with markers for MCP server parsing"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


def extract_json_from_output(output_text):
    """Extract JSON result from subprocess output"""
    if "__RESULT_JSON__" in output_text and "__END_RESULT_JSON__" in output_text:
        start = output_text.find("__RESULT_JSON__") + len("__RESULT_JSON__")
        end = output_text.find("__END_RESULT_JSON__")
        json_str = output_text[start:end].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None


def stream_output(pipe, logger, output_lines, prefix=""):
    """Stream subprocess output line by line and capture to output_lines"""
    for line in iter(pipe.readline, ''):
        if line:
            # Remove trailing newline and log
            line = line.rstrip()
            if line:
                output_lines.append(line)
                logger.info(f"{prefix}{line}")
    pipe.close()


def run_translate_for_assembly(args, logger):
    """Run translate_for_assembly script and return (success, output_text)"""
    script_path = os.path.join(os.path.dirname(__file__), 'legacy', 'translate_for_assembly.py')

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
        if args.use_default_base_position:
            cmd.append('--use-default-base')

    logger.info(f"Translating {args.object_name} relative to {args.base_name}")

    # Set PYTHONPATH to include project root for imports
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root

    # Capture output lines
    output_lines = []

    # Run subprocess and stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    # Stream output in a separate thread
    output_thread = threading.Thread(
        target=stream_output,
        args=(process.stdout, logger, output_lines),
        daemon=True
    )
    output_thread.start()

    # Wait for process to complete
    returncode = process.wait()
    output_thread.join(timeout=1.0)

    output_text = '\n'.join(output_lines)

    if returncode != 0:
        logger.error(f"translate_for_assembly failed with return code {returncode}")
        return False, output_text

    logger.info("translate_for_assembly completed successfully")
    return True, output_text


def run_perform_insert(args, logger):
    """Run perform_insert script and return (success, output_text)"""
    script_path = os.path.join(os.path.dirname(__file__), 'legacy', 'perform_insert.py')

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

    # Set PYTHONPATH to include project root for imports
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root

    # Capture output lines
    output_lines = []

    # Run subprocess and stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    # Stream output in a separate thread
    output_thread = threading.Thread(
        target=stream_output,
        args=(process.stdout, logger, output_lines),
        daemon=True
    )
    output_thread.start()

    # Wait for process to complete
    returncode = process.wait()
    output_thread.join(timeout=1.0)

    output_text = '\n'.join(output_lines)

    if returncode != 0:
        logger.error(f"perform_insert failed with return code {returncode}")
        return False, output_text

    logger.info("perform_insert completed successfully")
    return True, output_text


def run_move_to_clear_area(logger):
    """Run move_to_clear_area script and return (success, output_text)"""
    script_path = os.path.join(os.path.dirname(__file__), 'legacy', 'move_to_clear_area.py')

    cmd = [sys.executable, script_path]

    # Default to 'move' mode (keeps current EE orientation)
    cmd.append('--move')

    logger.info("Moving object away from base to clear area")

    # Set PYTHONPATH to include project root for imports
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root

    # Capture output lines
    output_lines = []

    # Run subprocess and stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    # Stream output in a separate thread
    output_thread = threading.Thread(
        target=stream_output,
        args=(process.stdout, logger, output_lines),
        daemon=True
    )
    output_thread.start()

    # Wait for process to complete
    returncode = process.wait()
    output_thread.join(timeout=1.0)

    output_text = '\n'.join(output_lines)

    if returncode != 0:
        logger.error(f"move_to_clear_area failed with return code {returncode}")
        return False, output_text

    logger.info("move_to_clear_area completed successfully")
    return True, output_text


def main():
    parser = argparse.ArgumentParser(
        description='Translate Object - Combines translate_for_assembly and perform_insert',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sim mode - safe height only
  python3 translate_object.py --mode sim --object-name fork_orange --base-name base --move-to-base

  # Sim mode - perform insert only
  python3 translate_object.py --mode sim --object-name fork_orange --base-name base --perform-insert

  # Sim mode - move to safe height only
  python3 translate_object.py --mode sim --move-to-safe-height

  # Real mode - safe height only
  python3 translate_object.py --mode real --base-name base --move-to-base --final-base-pos 0.5 -0.37 0.1882 --final-base-orientation 0.0 0.0 0.0 1.0

  # Real mode - safe height only (using defaults)
  python3 translate_object.py --mode real --base-name base --move-to-base --use-default-base-position

  # Real mode - perform insert only with force compliance
  python3 translate_object.py --mode real --perform-insert --speed 0.01 --gain 2.0

  # Real mode - move to safe height only
  python3 translate_object.py --mode real --move-to-safe-height

  # Move away from base (to clear area)
  python3 translate_object.py --mode real --move-away-from-base
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
    parser.add_argument('--perform-insert', action='store_true',
                       dest='perform_insert',
                       help='Call perform_insert to move down to final position')
    parser.add_argument('--move-to-safe-height', action='store_true',
                       help='Move to safe height (after closing gripper)')
    parser.add_argument('--move-away-from-base', action='store_true',
                       help='Move object away from base to a clear area (uses move_to_clear_area)')

    # Real mode arguments for translate_for_assembly
    parser.add_argument('--final-base-pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                       help='Final base position [x, y, z] in meters (for translate_for_assembly in real mode)')
    parser.add_argument('--final-base-orientation', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'),
                       help='Final base orientation quaternion [x, y, z, w] (for translate_for_assembly in real mode)')
    parser.add_argument('--use-default-base-position', action='store_true',
                       dest='use_default_base_position',
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
    flags_set = sum([args.move_to_base, args.perform_insert, args.move_to_safe_height, args.move_away_from_base])
    if flags_set == 0:
        parser.error("At least one of --move-to-base, --perform-insert, --move-to-safe-height, or --move-away-from-base must be specified")

    if flags_set > 1:
        parser.error("Cannot use multiple movement flags together. Use exactly one of --move-to-base, --perform-insert, --move-to-safe-height, or --move-away-from-base")

    # Validate sim mode requirements
    if args.mode == 'sim':
        if args.move_to_safe_height or args.move_away_from_base:
            # move_to_safe_height and move_away_from_base don't require object_name or base_name
            pass
        else:
            if args.object_name is None:
                parser.error("--object-name is required in sim mode")
            if args.base_name is None and args.move_to_base:
                parser.error("--base-name is required when using --move-to-base in sim mode")
            if args.base_name is None and args.perform_insert:
                parser.error("--base-name is required when using --perform-insert in sim mode")

    # Validate real mode requirements for translate_for_assembly
    if args.mode == 'real' and args.move_to_base:
        if args.base_name is None:
            parser.error("--base-name is required when using --move-to-base in real mode")
        if not args.use_default_base_position and args.final_base_pos is None:
            parser.error("In real mode with --move-to-base, either --final-base-pos or --use-default-base-position is required")

    # Initialize ROS
    rclpy.init()
    node = Node('translate_object')
    logger = node.get_logger()
    
    logger.info(f"Using {args.mode.upper()} mode")

    try:
        # Call move_to_safe_height if --move-to-safe-height is specified
        if args.move_to_safe_height:
            script_path = os.path.join(os.path.dirname(__file__), 'legacy', 'move_to_safe_height.py')

            logger.info("Moving to safe height...")

            # Set PYTHONPATH to include project root for imports
            env = os.environ.copy()
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = project_root

            # Capture output lines
            output_lines = []

            # Run subprocess with timeout
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )

            # Stream output in a separate thread
            output_thread = threading.Thread(
                target=stream_output,
                args=(process.stdout, logger, output_lines),
                daemon=True
            )
            output_thread.start()

            # Wait for process to complete with timeout
            try:
                returncode = process.wait(timeout=40)
                output_thread.join(timeout=1.0)
            except subprocess.TimeoutExpired:
                logger.error("move_to_safe_height timed out")
                process.kill()
                output_result({
                    "result": "failure",
                    "mode": args.mode,
                    "movement_type": "move_to_safe_height",
                    "error": "move_to_safe_height timed out"
                })
                node.destroy_node()
                rclpy.shutdown()
                sys.exit(1)

            output_text = '\n'.join(output_lines)
            subprocess_json = extract_json_from_output(output_text)

            if returncode != 0:
                logger.error(f"move_to_safe_height failed with return code {returncode}")
                # Use subprocess error if available, otherwise generic error
                if subprocess_json:
                    subprocess_json["movement_type"] = "move_to_safe_height"
                    output_result(subprocess_json)
                else:
                    output_result({
                        "result": "failure",
                        "mode": args.mode,
                        "movement_type": "move_to_safe_height",
                        "error": f"move_to_safe_height failed with return code {returncode}"
                    })
                node.destroy_node()
                rclpy.shutdown()
                sys.exit(1)

            logger.info("move_to_safe_height completed successfully")
            # Add movement_type to subprocess JSON or create success JSON
            if subprocess_json:
                subprocess_json["movement_type"] = "move_to_safe_height"
                output_result(subprocess_json)
            else:
                output_result({
                    "result": "success",
                    "mode": args.mode,
                    "movement_type": "move_to_safe_height"
                })
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(0)

        # Call translate_for_assembly if --move-to-base is specified
        if args.move_to_base:
            success, output_text = run_translate_for_assembly(args, logger)
            subprocess_json = extract_json_from_output(output_text)

            if not success:
                logger.error("translate_for_assembly failed, aborting")
                # Use subprocess error if available, otherwise generic error
                if subprocess_json:
                    subprocess_json["movement_type"] = "move_to_base"
                    output_result(subprocess_json)
                else:
                    output_result({
                        "result": "failure",
                        "mode": args.mode,
                        "movement_type": "move_to_base",
                        "object_name": args.object_name,
                        "base_name": args.base_name,
                        "error": "translate_for_assembly failed"
                    })
                node.destroy_node()
                rclpy.shutdown()
                sys.exit(1)

            logger.info("Operation completed successfully!")
            # Add movement_type to subprocess JSON or create success JSON
            if subprocess_json:
                subprocess_json["movement_type"] = "move_to_base"
                output_result(subprocess_json)
            else:
                output_result({
                    "result": "success",
                    "mode": args.mode,
                    "movement_type": "move_to_base",
                    "object_name": args.object_name,
                    "base_name": args.base_name
                })
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(0)

        # Call perform_insert if --perform-insert is specified
        if args.perform_insert:
            success, output_text = run_perform_insert(args, logger)
            subprocess_json = extract_json_from_output(output_text)

            if not success:
                logger.error("perform_insert failed, aborting")
                # Use subprocess error if available, otherwise generic error
                if subprocess_json:
                    subprocess_json["movement_type"] = "perform_insert"
                    output_result(subprocess_json)
                else:
                    result = {
                        "result": "failure",
                        "mode": args.mode,
                        "movement_type": "perform_insert",
                        "error": "perform_insert failed"
                    }
                    if args.mode == 'sim':
                        result["object_name"] = args.object_name
                        result["base_name"] = args.base_name
                    output_result(result)
                node.destroy_node()
                rclpy.shutdown()
                sys.exit(1)

            logger.info("Operation completed successfully!")
            # Add movement_type to subprocess JSON or create success JSON
            if subprocess_json:
                subprocess_json["movement_type"] = "perform_insert"
                output_result(subprocess_json)
            else:
                result = {
                    "result": "success",
                    "mode": args.mode,
                    "movement_type": "perform_insert"
                }
                if args.mode == 'sim':
                    result["object_name"] = args.object_name
                    result["base_name"] = args.base_name
                output_result(result)
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(0)

        # Call move_to_clear_area if --move-away-from-base is specified
        if args.move_away_from_base:
            success, output_text = run_move_to_clear_area(logger)
            subprocess_json = extract_json_from_output(output_text)

            if not success:
                logger.error("move_to_clear_area failed, aborting")
                # Use subprocess error if available, otherwise generic error
                if subprocess_json:
                    subprocess_json["movement_type"] = "move_away_from_base"
                    output_result(subprocess_json)
                else:
                    output_result({
                        "result": "failure",
                        "mode": args.mode,
                        "movement_type": "move_away_from_base",
                        "error": "move_to_clear_area failed"
                    })
                node.destroy_node()
                rclpy.shutdown()
                sys.exit(1)

            logger.info("Operation completed successfully!")
            # Add movement_type to subprocess JSON or create success JSON
            if subprocess_json:
                subprocess_json["movement_type"] = "move_away_from_base"
                output_result(subprocess_json)
            else:
                output_result({
                    "result": "success",
                    "mode": args.mode,
                    "movement_type": "move_away_from_base"
                })
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

