#!/usr/bin/env python3
"""
Interactive CLI tool for controlling URSim using dashboard services.

Run the script and enter commands interactively.
"""

import sys
import os
import json
import readline  # Enables arrow keys and history in input

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class URSimCLI(Node):
    """ROS2 node for URSim dashboard control."""

    TRIGGER_SERVICES = {
        'play': '/dashboard_client/play',
        'stop': '/dashboard_client/stop',
        'pause': '/dashboard_client/pause',
        'power_on': '/dashboard_client/power_on',
        'power_off': '/dashboard_client/power_off',
        'brake_release': '/dashboard_client/brake_release',
        'unlock_protective_stop': '/dashboard_client/unlock_protective_stop',
        'connect': '/dashboard_client/connect',
        'quit': '/dashboard_client/quit',
        'shutdown': '/dashboard_client/shutdown',
        'restart_safety': '/dashboard_client/restart_safety',
        'clear_operational_mode': '/dashboard_client/clear_operational_mode',
        'close_popup': '/dashboard_client/close_popup',
        'close_safety_popup': '/dashboard_client/close_safety_popup',
    }

    QUERY_SERVICES = {
        'status': '/dashboard_client/get_loaded_program',
        'mode': '/dashboard_client/get_robot_mode',
        'safety': '/dashboard_client/get_safety_mode',
        'program': '/dashboard_client/get_loaded_program',
        'running': '/dashboard_client/program_running',
        'saved': '/dashboard_client/program_saved',
        'state': '/dashboard_client/program_state',
    }

    PARAM_SERVICES = {
        'load': '/dashboard_client/load_program',
        'log': '/dashboard_client/add_to_log',
        'popup': '/dashboard_client/popup',
        'raw': '/dashboard_client/raw_request',
    }

    # Shorthand aliases
    ALIASES = {
        'unlock': 'unlock_protective_stop',
        'clear_mode': 'clear_operational_mode',
        'release': 'brake_release',
        'on': 'power_on',
        'off': 'power_off',
    }

    def __init__(self):
        super().__init__('ursim_cli')

    def call_trigger(self, command: str) -> dict:
        """Call a trigger service (no parameters)."""
        service_name = self.TRIGGER_SERVICES.get(command)
        if not service_name:
            return {'success': False, 'error': f"Unknown command: {command}"}

        client = self.create_client(Trigger, service_name)
        if not client.wait_for_service(timeout_sec=5.0):
            return {'success': False, 'error': f"Service {service_name} not available"}

        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done():
            return {'success': False, 'error': 'Service call timed out'}

        response = future.result()
        return {
            'success': response.success,
            'message': response.message
        }

    def call_query(self, query_type: str) -> dict:
        """Call a query service to get robot state."""
        service_name = self.QUERY_SERVICES.get(query_type)
        if not service_name:
            return {'success': False, 'error': f"Unknown query: {query_type}"}

        try:
            if query_type in ['status', 'program']:
                from ur_dashboard_msgs.srv import GetLoadedProgram
                srv_class = GetLoadedProgram
            elif query_type == 'mode':
                from ur_dashboard_msgs.srv import GetRobotMode
                srv_class = GetRobotMode
            elif query_type == 'safety':
                from ur_dashboard_msgs.srv import GetSafetyMode
                srv_class = GetSafetyMode
            elif query_type == 'running':
                from ur_dashboard_msgs.srv import IsProgramRunning
                srv_class = IsProgramRunning
            elif query_type == 'saved':
                from ur_dashboard_msgs.srv import IsProgramSaved
                srv_class = IsProgramSaved
            elif query_type == 'state':
                from ur_dashboard_msgs.srv import GetProgramState
                srv_class = GetProgramState
            else:
                return {'success': False, 'error': f"Unknown query type: {query_type}"}
        except ImportError as e:
            return {'success': False, 'error': f"Missing ur_dashboard_msgs: {e}"}

        client = self.create_client(srv_class, service_name)
        if not client.wait_for_service(timeout_sec=5.0):
            return {'success': False, 'error': f"Service {service_name} not available"}

        future = client.call_async(srv_class.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done():
            return {'success': False, 'error': 'Service call timed out'}

        response = future.result()
        result = {'success': True}

        if query_type in ['status', 'program']:
            result['program_name'] = response.program_name
            result['answer'] = response.answer
        elif query_type == 'mode':
            result['robot_mode'] = response.robot_mode.mode
            result['answer'] = response.answer
        elif query_type == 'safety':
            result['safety_mode'] = response.safety_mode.mode
            result['answer'] = response.answer
        elif query_type == 'running':
            result['running'] = response.program_running
            result['answer'] = response.answer
        elif query_type == 'saved':
            result['saved'] = response.program_saved
            result['program_name'] = response.program_name
            result['answer'] = response.answer
        elif query_type == 'state':
            result['state'] = response.state.state
            result['program_name'] = response.program_name
            result['answer'] = response.answer

        return result

    def call_param_service(self, command: str, param: str) -> dict:
        """Call a service that requires a parameter."""
        service_name = self.PARAM_SERVICES.get(command)
        if not service_name:
            return {'success': False, 'error': f"Unknown command: {command}"}

        try:
            if command == 'load':
                from ur_dashboard_msgs.srv import Load
                srv_class = Load
                field = 'filename'
            elif command == 'log':
                from ur_dashboard_msgs.srv import AddToLog
                srv_class = AddToLog
                field = 'message'
            elif command == 'popup':
                from ur_dashboard_msgs.srv import Popup
                srv_class = Popup
                field = 'message'
            elif command == 'raw':
                from ur_dashboard_msgs.srv import RawRequest
                srv_class = RawRequest
                field = 'query'
            else:
                return {'success': False, 'error': f"Unknown param command: {command}"}
        except ImportError as e:
            return {'success': False, 'error': f"Missing ur_dashboard_msgs: {e}"}

        client = self.create_client(srv_class, service_name)
        if not client.wait_for_service(timeout_sec=5.0):
            return {'success': False, 'error': f"Service {service_name} not available"}

        request = srv_class.Request()
        setattr(request, field, param)

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done():
            return {'success': False, 'error': 'Service call timed out'}

        response = future.result()
        return {
            'success': response.success,
            'answer': response.answer
        }

    def recover(self) -> dict:
        """Unlock protective stop, wait for release, then play. Returns combined result."""
        import time
        r = self.call_trigger('unlock_protective_stop')
        if not r.get('success'):
            return r
        time.sleep(2)
        r = self.call_trigger('play')
        if not r.get('success'):
            return r
        return {'success': True, 'message': 'Protective stop unlocked and program started'}

    def execute(self, user_input: str) -> dict:
        """Parse and execute a command."""
        parts = user_input.strip().split(maxsplit=1)
        if not parts:
            return None

        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None

        # Resolve aliases
        cmd = self.ALIASES.get(cmd, cmd)

        # Combo commands
        if cmd == 'recover':
            return self.recover()

        # Execute based on command type
        if cmd in self.TRIGGER_SERVICES:
            return self.call_trigger(cmd)
        elif cmd in self.QUERY_SERVICES:
            return self.call_query(cmd)
        elif cmd in self.PARAM_SERVICES:
            if not arg:
                return {'success': False, 'error': f"'{cmd}' requires an argument"}
            return self.call_param_service(cmd, arg)
        else:
            return {'success': False, 'error': f"Unknown command: {cmd}"}


def print_help():
    """Print available commands."""
    print("""
Commands:
  Control:
    play                  Start the loaded program
    stop                  Stop the running program
    pause                 Pause the running program
    on / power_on         Power on the robot
    off / power_off       Power off the robot
    release / brake_release  Release the brakes
    connect               Connect to the robot
    quit                  Disconnect from the robot
    shutdown              Shutdown the robot controller

  Safety:
    unlock                Unlock protective stop
    recover               Unlock protective stop + play (one-shot fix)
    restart_safety        Restart the safety controller
    close_popup           Close popup dialog (like clicking "Continue")
    close_safety_popup    Close safety popup
    clear_mode            Clear operational mode

  Query:
    status                Get loaded program info
    mode                  Get robot mode (RUNNING, IDLE, etc.)
    safety                Get safety mode
    program               Get loaded program name
    running               Check if a program is running
    saved                 Check if program is saved
    state                 Get program state

  Program:
    load <file>           Load a program file
                          Example: load my_program.urp
    log <message>         Add message to robot log
                          Example: log "Robot started task"
    popup <message>       Show popup on teach pendant
                          Example: popup "Waiting for operator"
    raw <query>           Send raw dashboard command
                          Example: raw "robotmode"
                          Example: raw "get serial number"

  Other:
    help / ?              Show this help
    exit / q              Exit the CLI

Workflow Examples:
  # Start robot from power-off state:
    on -> release -> play

  # Handle "External Control error" dialog:
    stop -> close_popup -> on -> release -> play

  # Handle "Protective Stop" (safety_mode=3):
  # Occurs when UR rejects a trajectory position too close to joint
  # limits. Robot freezes, program stops, subsequent play fails.
    unlock -> play
""")


def print_result(result: dict):
    """Print command result."""
    if result is None:
        return
    if result.get('success'):
        msg = result.get('message') or result.get('answer') or 'OK'
        print(f"  ✓ {msg}")
        for key, val in result.items():
            if key not in ['success', 'message', 'answer']:
                print(f"    {key}: {val}")
    else:
        print(f"  ✗ {result.get('error') or result.get('message') or result.get('answer') or 'Unknown error'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='URSim dashboard CLI')
    parser.add_argument('--recover', action='store_true',
                        help='Unlock protective stop and play (non-interactive)')
    cli_args = parser.parse_args()

    rclpy.init()
    node = URSimCLI()

    if cli_args.recover:
        result = node.recover()
        print_result(result)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0 if result.get('success') else 1)

    print("URSim CLI - Interactive dashboard control")
    print("Type 'help' for commands, 'exit' to quit\n")

    try:
        while True:
            try:
                user_input = input("ursim> ").strip()
            except EOFError:
                print()
                break

            if not user_input:
                continue

            cmd = user_input.lower().split()[0]

            if cmd in ['exit', 'q']:
                break
            elif cmd in ['help', '?']:
                print_help()
                continue

            result = node.execute(user_input)
            print_result(result)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
