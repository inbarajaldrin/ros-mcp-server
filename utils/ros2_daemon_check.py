#!/usr/bin/env python3
"""
Check if the ROS2 daemon is running. Optionally ensure/start it if not.

Usage:
    python utils/ros2_daemon_check.py

Exit code:
    0 if daemon is running
    1 if daemon is not running or check failed
"""

import subprocess
import sys
import time


def is_ros2_daemon_working() -> tuple[bool, str]:
    """
    Check if the ROS2 daemon is running and responsive.

    Returns:
        (ok, message): ok is True if daemon is working, False otherwise.
                       message describes the status.
    """
    try:
        result = subprocess.run(
            ["ros2", "daemon", "status"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode == 0:
            if "running" in stdout.lower():
                return True, stdout or "The daemon is running"
            return True, stdout or "Daemon check succeeded"

        # Non-zero exit: daemon not running or error
        if "not running" in stdout.lower() or "not running" in stderr.lower():
            return False, stdout or stderr or "The daemon is not running"
        if stderr:
            return False, stderr
        return False, stdout or f"Daemon check failed (exit {result.returncode})"

    except subprocess.TimeoutExpired:
        return False, "Daemon check timed out after 5 seconds"
    except FileNotFoundError:
        return False, "ros2 command not found — ROS2 may not be sourced"
    except PermissionError as e:
        return False, f"Permission error checking daemon: {e}"
    except OSError as e:
        return False, f"OS error: {e}"


def ensure_ros2_daemon_running() -> tuple[bool, str]:
    """
    Ensure the ROS2 daemon is running. If not, start it and re-check.

    Returns:
        (ok, message): ok is True if daemon is working (was already running or started successfully).
                       message describes the status.
    """
    ok, msg = is_ros2_daemon_working()
    if ok:
        return True, msg

    try:
        subprocess.run(
            ["ros2", "daemon", "start"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        time.sleep(1)
        ok, msg = is_ros2_daemon_working()
        if ok:
            return True, "Daemon was not running; started successfully"
        return False, msg
    except subprocess.TimeoutExpired:
        return False, "Daemon start timed out"
    except FileNotFoundError:
        return False, "ros2 command not found — ROS2 may not be sourced"
    except Exception as e:
        return False, f"Failed to start daemon: {e}"


def main() -> int:
    ok, msg = is_ros2_daemon_working()
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
