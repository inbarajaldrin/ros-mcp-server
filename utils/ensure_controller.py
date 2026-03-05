#!/usr/bin/env python3
"""
Ensure the scaled_joint_trajectory_controller is active.

Checks current controller state and activates it if inactive.
Also diagnoses common reasons for deactivation (protective stop,
external control program not running, etc.).

Usage:
    python3 utils/ensure_controller.py              # activate scaled controller
    python3 utils/ensure_controller.py --status      # just print status, don't switch
    python3 utils/ensure_controller.py --diagnose    # full diagnostic of all controllers
"""

import sys
import os
import argparse
import time

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import ListControllers, SwitchController


SCALED_CONTROLLER = 'scaled_joint_trajectory_controller'

# Controllers that conflict with scaled (only one trajectory controller can be active)
CONFLICTING_CONTROLLERS = [
    'joint_trajectory_controller',
    'passthrough_trajectory_controller',
    'forward_position_controller',
    'forward_velocity_controller',
    'forward_effort_controller',
    'freedrive_mode_controller',
]


class ControllerEnsurer(Node):
    def __init__(self):
        super().__init__('ensure_controller')

    def list_controllers(self) -> list:
        """Get all controllers and their states."""
        cli = self.create_client(ListControllers, '/controller_manager/list_controllers')
        if not cli.wait_for_service(timeout_sec=5.0):
            print("ERROR: /controller_manager/list_controllers service not available")
            return []

        future = cli.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            print("ERROR: list_controllers call timed out")
            return []

        return list(future.result().controller)

    def switch_controllers(self, activate: list, deactivate: list, strict: bool = True) -> bool:
        """Switch controllers via the controller_manager service."""
        cli = self.create_client(SwitchController, '/controller_manager/switch_controller')
        if not cli.wait_for_service(timeout_sec=5.0):
            print("ERROR: /controller_manager/switch_controller service not available")
            return False

        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.STRICT if strict else SwitchController.Request.BEST_EFFORT
        req.activate_asap = False
        req.timeout.sec = 5

        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            print("ERROR: switch_controller call timed out")
            return False

        return future.result().ok

    def get_dashboard_info(self) -> dict:
        """Query dashboard for robot state diagnostics."""
        info = {}
        try:
            from ur_dashboard_msgs.srv import GetRobotMode, GetSafetyMode, IsProgramRunning

            for name, srv_cls, key in [
                ('/dashboard_client/get_robot_mode', GetRobotMode, 'robot_mode'),
                ('/dashboard_client/get_safety_mode', GetSafetyMode, 'safety_mode'),
                ('/dashboard_client/program_running', IsProgramRunning, 'program_running'),
            ]:
                cli = self.create_client(srv_cls, name)
                if not cli.wait_for_service(timeout_sec=3.0):
                    info[key] = 'service unavailable'
                    continue
                future = cli.call_async(srv_cls.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
                if future.done() and future.result() is not None:
                    info[key] = future.result().answer
                else:
                    info[key] = 'timeout'
        except ImportError:
            info['error'] = 'ur_dashboard_msgs not installed'

        return info

    def diagnose(self, controllers: list) -> list:
        """Return list of diagnostic messages."""
        msgs = []
        scaled = None
        active_conflicting = []

        for c in controllers:
            if c.name == SCALED_CONTROLLER:
                scaled = c
            elif c.name in CONFLICTING_CONTROLLERS and c.state == 'active':
                active_conflicting.append(c.name)

        if scaled is None:
            msgs.append(f"CRITICAL: {SCALED_CONTROLLER} not loaded in controller_manager")
            return msgs

        msgs.append(f"{SCALED_CONTROLLER} state: {scaled.state}")

        if scaled.state == 'active':
            msgs.append("Controller is active - no action needed")
        elif scaled.state == 'inactive':
            if active_conflicting:
                msgs.append(f"Conflicting active controllers: {', '.join(active_conflicting)}")
                msgs.append("These must be deactivated before activating scaled controller")
            else:
                msgs.append("No conflicting controllers active - can activate directly")

            # Check dashboard state
            dash = self.get_dashboard_info()
            for k, v in dash.items():
                msgs.append(f"Dashboard {k}: {v}")

            if 'PROTECTIVE' in str(dash.get('safety_mode', '')).upper():
                msgs.append("CAUSE: Robot is in protective stop - unlock first")
            elif 'false' in str(dash.get('program_running', '')).lower():
                msgs.append("CAUSE: External control program not running - press Play on pendant")
        elif scaled.state == 'unconfigured':
            msgs.append("Controller is unconfigured - may need to reload UR driver")

        return msgs

    def ensure_active(self) -> bool:
        """Activate scaled_joint_trajectory_controller if not already active."""
        controllers = self.list_controllers()
        if not controllers:
            return False

        scaled = None
        active_conflicting = []

        for c in controllers:
            if c.name == SCALED_CONTROLLER:
                scaled = c
            elif c.name in CONFLICTING_CONTROLLERS and c.state == 'active':
                active_conflicting.append(c.name)

        if scaled is None:
            print(f"ERROR: {SCALED_CONTROLLER} not found in controller_manager")
            return False

        if scaled.state == 'active':
            print(f"{SCALED_CONTROLLER} is already active")
            return True

        print(f"{SCALED_CONTROLLER} is {scaled.state}")

        # Deactivate conflicting controllers first
        if active_conflicting:
            print(f"Deactivating conflicting: {', '.join(active_conflicting)}")
            ok = self.switch_controllers([], active_conflicting, strict=False)
            if not ok:
                print("WARNING: Failed to deactivate conflicting controllers, trying anyway")
            time.sleep(0.5)

        # Activate
        print(f"Activating {SCALED_CONTROLLER}...")
        ok = self.switch_controllers([SCALED_CONTROLLER], [])
        if ok:
            print(f"OK: {SCALED_CONTROLLER} activated")
        else:
            print(f"FAILED to activate {SCALED_CONTROLLER}")
            # Run diagnostics
            for msg in self.diagnose(self.list_controllers()):
                print(f"  {msg}")

        return ok


def main():
    parser = argparse.ArgumentParser(description='Ensure scaled_joint_trajectory_controller is active')
    parser.add_argument('--status', action='store_true', help='Print status only, do not switch')
    parser.add_argument('--diagnose', action='store_true', help='Full diagnostic of all controllers')
    args = parser.parse_args()

    rclpy.init()
    node = ControllerEnsurer()

    try:
        if args.diagnose:
            controllers = node.list_controllers()
            print("=== Controller States ===")
            for c in sorted(controllers, key=lambda x: x.name):
                marker = 'ACTIVE' if c.state == 'active' else c.state
                print(f"  {c.name:40s} {marker}")
            print()
            print("=== Diagnostics ===")
            for msg in node.diagnose(controllers):
                print(f"  {msg}")
            print()
            print("=== Dashboard ===")
            for k, v in node.get_dashboard_info().items():
                print(f"  {k}: {v}")
        elif args.status:
            controllers = node.list_controllers()
            for c in controllers:
                if c.name == SCALED_CONTROLLER:
                    print(f"{SCALED_CONTROLLER}: {c.state}")
                    sys.exit(0 if c.state == 'active' else 1)
            print(f"{SCALED_CONTROLLER}: NOT FOUND")
            sys.exit(1)
        else:
            ok = node.ensure_active()
            sys.exit(0 if ok else 1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
