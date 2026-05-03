#!/usr/bin/env python3
"""
Visual axis-direction verification — robot moves slowly in each horizontal axis.

Companion to verify_ft_signs.py. Where that script measures FORCE direction,
this one drives MOTION direction. For each axis (+X, -X, +Y, -Y), the robot:
  1. Moves home (clean starting pose)
  2. Enters force mode with selection_vector enabling only the test axis
     (other axes held — robot moves ONLY in the commanded direction)
  3. Commands a small wrench (~2 N) in the test direction
  4. Streams live TCP pose for 3 s while operator watches the robot
  5. Stops force mode + asks operator to confirm motion direction

Safety:
  - Commanded force 2.0 N (well under 5 N CONVENTIONS cap)
  - Speed cap 0.02 m/s linear → ~6 cm displacement over 3 s window
  - Single-axis compliance per test (other axes held in position)
  - Ctrl-C anytime → stop force mode + restore position controller

Usage:
  cd /home/aaugus11/Documents/ros-mcp-server
  python3 compliant_insertion_studio/scripts/verify_axis_motion.py
"""

import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from ur_msgs.srv import SetForceMode


POS_CTRL = "scaled_joint_trajectory_controller"
FORCE_CTRL = "force_mode_controller"

_LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "diagnostics"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_PATH = _LOG_DIR / f"verify_axis_motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try: s.write(data); s.flush()
            except Exception: pass
    def flush(self):
        for s in self._streams:
            try: s.flush()
            except Exception: pass


class AxisMotionTester(Node):
    def __init__(self):
        super().__init__("verify_axis_motion")
        self.tcp: PoseStamped | None = None
        self.in_force_mode = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PoseStamped, "/tcp_pose_broadcaster/pose",
                                 self._tcp_cb, sensor_qos)
        self.start_fm = self.create_client(SetForceMode, "/force_mode_controller/start_force_mode")
        self.stop_fm = self.create_client(Trigger, "/force_mode_controller/stop_force_mode")

    def _tcp_cb(self, msg):
        self.tcp = msg

    def wait_for_tcp(self, timeout=5.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tcp is not None:
                return True
        return False

    def start_axis_motion(self, axis: str, direction: int, force_n: float = 2.0) -> bool:
        """Enter force mode with single-axis compliance + commanded force.

        axis: 'x' | 'y' | 'z'
        direction: +1 or -1
        force_n: magnitude in N (default 2.0)
        """
        if axis not in ("x", "y", "z") or direction not in (1, -1):
            return False
        if not self.start_fm.wait_for_service(timeout_sec=3.0):
            print("  ERROR: start_force_mode service unavailable")
            return False

        req = SetForceMode.Request()
        req.task_frame.header.frame_id = "base_link"
        req.task_frame.pose.orientation.w = 1.0   # identity

        # Only the test axis is compliant; others are held in position
        req.selection_vector_x = (axis == "x")
        req.selection_vector_y = (axis == "y")
        req.selection_vector_z = (axis == "z")
        req.selection_vector_rx = False
        req.selection_vector_ry = False
        req.selection_vector_rz = False

        # Commanded wrench in base_link frame
        req.wrench.force.x = float(direction * force_n) if axis == "x" else 0.0
        req.wrench.force.y = float(direction * force_n) if axis == "y" else 0.0
        req.wrench.force.z = float(direction * force_n) if axis == "z" else 0.0
        req.wrench.torque.x = 0.0
        req.wrench.torque.y = 0.0
        req.wrench.torque.z = 0.0

        req.type = 2   # NO_TRANSFORM (task_frame = base_link as-is)

        # Conservative speed caps (~6 cm over 3 s)
        req.speed_limits.linear.x = 0.02
        req.speed_limits.linear.y = 0.02
        req.speed_limits.linear.z = 0.02
        req.speed_limits.angular.x = 0.05
        req.speed_limits.angular.y = 0.05
        req.speed_limits.angular.z = 0.05

        req.gain_scaling = 0.5
        req.damping_factor = 0.7

        future = self.start_fm.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None or not result.success:
            print("  ERROR: start_force_mode RPC failed")
            return False
        self.in_force_mode = True
        return True

    def stop_axis_motion(self):
        if not self.in_force_mode:
            return
        if self.stop_fm.wait_for_service(timeout_sec=2.0):
            future = self.stop_fm.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        self.in_force_mode = False

    def stream_tcp_pose(self, duration_s: float, label: str, start_pose: PoseStamped):
        """Print TCP pose + delta-from-start at 5 Hz for `duration_s`."""
        sx, sy, sz = (start_pose.pose.position.x,
                      start_pose.pose.position.y,
                      start_pose.pose.position.z)
        print(f"\n  [{label}] streaming TCP pose for {duration_s:.0f}s "
              f"(start: x={sx:+.4f} y={sy:+.4f} z={sz:+.4f})...")
        print(f"  {'t':>5}  | "
              f"{'x':>9} {'y':>9} {'z':>9} | "
              f"{'dx':>+9} {'dy':>+9} {'dz':>+9}")
        t_end = time.time() + duration_s
        last_print = 0.0
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.tcp is None:
                continue
            now = time.time()
            if now - last_print < 0.2:
                continue
            last_print = now
            p = self.tcp.pose.position
            elapsed = duration_s - (t_end - now)
            print(f"  {elapsed:5.1f}  | "
                  f"{p.x:+9.4f} {p.y:+9.4f} {p.z:+9.4f} | "
                  f"{p.x - sx:+9.4f} {p.y - sy:+9.4f} {p.z - sz:+9.4f}")


def switch_controllers(activate, deactivate):
    cmd = ["ros2", "control", "switch_controllers"]
    if deactivate: cmd += ["--deactivate"] + deactivate
    if activate: cmd += ["--activate"] + activate
    print(f"  switch_controllers: -{deactivate} +{activate}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr.strip()}")
        return False
    return True


def is_program_running() -> bool:
    try:
        r = subprocess.run(
            ["ros2", "service", "call", "/dashboard_client/program_running",
             "ur_dashboard_msgs/srv/IsProgramRunning"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return False
    return "program_running=True" in r.stdout


def go_home_then_force(node: AxisMotionTester) -> bool:
    if not is_program_running():
        print("  ERROR: pendant program not running. Press Play and re-run.")
        return False

    print("  [reset] Stop force mode...")
    try: node.stop_axis_motion()
    except Exception as e: print(f"  WARN: stop error (non-fatal): {e}")
    time.sleep(0.5)

    print("  [reset] Switch to position controller...")
    subprocess.run(["ros2", "control", "switch_controllers", "--deactivate", FORCE_CTRL],
                   capture_output=True, text=True, timeout=10)
    time.sleep(0.3)
    r = subprocess.run(["ros2", "control", "switch_controllers", "--activate", POS_CTRL],
                       capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        print(f"  ERROR: position controller activate failed: {r.stderr.strip()}")
        return False

    print("  [reset] Move home...")
    repo_root = Path(__file__).resolve().parents[2]
    try:
        r = subprocess.run([sys.executable, str(repo_root / "primitives" / "move_home.py")],
                           capture_output=True, text=True, timeout=60, cwd=str(repo_root))
    except subprocess.TimeoutExpired:
        print("  ERROR: move_home timeout (>60 s). Pendant paused?")
        return False
    if '"result": "success"' not in r.stdout:
        print(f"  ERROR: move_home failed. stdout tail: {r.stdout[-300:]!r}")
        return False

    print("  [reset] Switch back to force mode...")
    subprocess.run(["ros2", "control", "switch_controllers", "--deactivate", POS_CTRL],
                   capture_output=True, text=True, timeout=10)
    time.sleep(0.3)
    r = subprocess.run(["ros2", "control", "switch_controllers", "--activate", FORCE_CTRL],
                       capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        print(f"  ERROR: force controller activate failed: {r.stderr.strip()}")
        return False
    time.sleep(1.0)
    print("  [reset] At home, ready for axis motion.\n")
    return True


def prompt(msg: str) -> str:
    try: return input(msg)
    except EOFError: return ""


def main():
    logfile = open(_LOG_PATH, "w", buffering=1)
    sys.stdout = _TeeStream(sys.__stdout__, logfile)
    print(f"[LOG] Tee'd output to: {_LOG_PATH}")

    rclpy.init()
    node = AxisMotionTester()

    # Same Ctrl-C policy as verify_ft_signs.py post-fix: don't override SIGINT.
    state = {"interrupted": False}
    def _on_sigterm(signum, frame):
        print(f"\n[!] SIGTERM received — cleanup starting")
        state["interrupted"] = True
    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        print("=" * 70)
        print("  Visual axis-direction verification — robot moves in each direction")
        print("=" * 70)
        print()
        print("  Per axis test: home -> force mode (single-axis sel_vec)")
        print("                 -> commanded 2 N for 3 s -> stream TCP pose")
        print("                 -> stop -> ask you to confirm direction")
        print()
        print("  EXPECTED MOTION (base_link, verified empirically 2026-05-03):")
        print("    +X push -> robot moves to its LEFT")
        print("    -X push -> robot moves to its RIGHT")
        print("    +Y push -> robot moves FORWARD (away from base)")
        print("    -Y push -> robot moves BACK toward base")
        print()
        print("  Speed cap: 0.02 m/s (~6 cm over 3 s window).")
        print("  Force magnitude: 2.0 N. Ctrl-C anytime for clean cleanup.")
        print()
        prompt("  Press Enter when ready to start... ")

        if not node.wait_for_tcp(timeout=5.0):
            print("  ERROR: /tcp_pose_broadcaster/pose silent. Bringup running?")
            return

        # Initial switch into force mode (subsequent axes use go_home_then_force)
        print("\n[init] Switching to force_mode_controller...")
        if not switch_controllers(activate=[FORCE_CTRL], deactivate=[POS_CTRL]):
            print("  ERROR: initial switch failed.")
            return
        time.sleep(1.0)

        axis_tests = [
            ("+X (LEFT)", "x", +1, "Robot should move to its LEFT (your right facing front)"),
            ("-X (RIGHT)", "x", -1, "Robot should move to its RIGHT"),
            ("+Y (FORWARD)", "y", +1, "Robot should move FORWARD (away from base)"),
            ("-Y (BACK)", "y", -1, "Robot should move BACK toward base"),
        ]

        for label, axis, direction, expected in axis_tests:
            if state["interrupted"]:
                return
            print(f"\n  ===== Axis test: {label} =====")
            if not go_home_then_force(node):
                print(f"  Aborting — go_home_then_force failed before {label}")
                return

            # Snapshot start pose
            node.tcp = None
            t0 = time.time()
            while node.tcp is None and time.time() - t0 < 2.0:
                rclpy.spin_once(node, timeout_sec=0.1)
            if node.tcp is None:
                print("  ERROR: no TCP pose after home. Skipping this axis.")
                continue
            start_pose = node.tcp

            print(f"  Expected: {expected}")
            prompt("  Press Enter to drive the robot... ")

            if not node.start_axis_motion(axis=axis, direction=direction, force_n=2.0):
                print(f"  ERROR: start_axis_motion failed for {label}")
                return

            node.stream_tcp_pose(3.0, label, start_pose)
            node.stop_axis_motion()
            time.sleep(0.5)   # let robot settle

            ans = prompt(f"  Did robot move correctly for {label}? [y/N]: ").strip().lower()
            if ans != "y":
                print(f"  ⚠️  Operator marked {label} as INCORRECT. Continuing for full report.")
            else:
                print(f"  ✓ {label} confirmed correct.")

        print("\n" + "=" * 70)
        print("  AXIS MOTION VERIFICATION COMPLETE")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n[!] Ctrl-C received — cleanup starting")

    finally:
        print("\n" + "=" * 70)
        print("[CLEANUP] Step 1/3: Stop force mode...")
        print("=" * 70)
        try: node.stop_axis_motion(); print("  ✓ stop complete")
        except Exception as e: print(f"  WARN: {e}")
        time.sleep(0.5)

        print("[CLEANUP] Step 2/3: Deactivate force_mode_controller...")
        try:
            r = subprocess.run(["ros2", "control", "switch_controllers", "--deactivate", FORCE_CTRL],
                              capture_output=True, text=True, timeout=10)
            print(f"  {'✓' if r.returncode == 0 else 'WARN'}: {r.stderr.strip() or 'ok'}")
        except Exception as e: print(f"  WARN: {e}")
        time.sleep(0.5)

        print(f"[CLEANUP] Step 3/3: Activate {POS_CTRL}...")
        try:
            r = subprocess.run(["ros2", "control", "switch_controllers", "--activate", POS_CTRL],
                              capture_output=True, text=True, timeout=10)
            print(f"  {'✓' if r.returncode == 0 else 'ERROR'}: {r.stderr.strip() or 'ok'}")
        except Exception as e: print(f"  ERROR: {e}")

        try: node.destroy_node(); rclpy.shutdown()
        except Exception: pass

        print("=" * 70)
        print("[CLEANUP] Done. Robot back in position-controlled mode.")
        print(f"[LOG] {_LOG_PATH}")
        print("=" * 70)
        try: logfile.close()
        except Exception: pass


if __name__ == "__main__":
    main()
