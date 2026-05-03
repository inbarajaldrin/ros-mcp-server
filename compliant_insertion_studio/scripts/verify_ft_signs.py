#!/usr/bin/env python3
"""
F/T sign-convention verification — interactive floppy-mode test.

Puts UR5e into 6-DOF compliant force mode with ZERO commanded wrench, then
walks the operator through a 6-direction push test (push +X, -X, +Y, -Y, +Z, -Z)
while continuously printing the wrench reading. Operator confirms each axis
sign matches SCHEMA.md (compliant_insertion_studio/docs/SCHEMA.md).

Safety:
- Commanded force/torque = 0 on all 6 axes (operator drives motion entirely)
- Speed caps: 0.02 m/s linear, 0.2 rad/s angular
- Damping 0.7, gain 0.5 (research-validated low-force defaults)
- All 6 axes compliant — robot yields freely to gentle pushes
- Ctrl-C / SIGTERM → stop force mode → switch back to position controller

DOES NOT zero the F/T sensor — operator wants to see baseline gravity bias
+ change-on-push, not just post-zero residual.

Usage:
  source /opt/ros/humble/setup.bash
  source ~/Desktop/ros2_ws/install/setup.bash   # or wherever ur_msgs lives
  python3 compliant_insertion_studio/scripts/verify_ft_signs.py
"""

import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger
from ur_msgs.srv import SetForceMode
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


POS_CTRL = "scaled_joint_trajectory_controller"
FORCE_CTRL = "force_mode_controller"

# Per CONVENTIONS: diagnostic scripts tee to timestamped logfile.
_LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "diagnostics"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_PATH = _LOG_DIR / f"verify_ft_signs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class _TeeStream:
    """Write to both terminal and logfile. Replaces sys.stdout."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def switch_controllers(activate, deactivate):
    cmd = ["ros2", "control", "switch_controllers"]
    if deactivate:
        cmd += ["--deactivate"] + deactivate
    if activate:
        cmd += ["--activate"] + activate
    print(f"  switch_controllers: -{deactivate} +{activate}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr.strip()}")
        return False
    return True


class FloppyTester(Node):
    def __init__(self):
        super().__init__("verify_ft_signs")
        self.wrench: WrenchStamped | None = None
        self.in_force_mode = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            WrenchStamped, "/force_torque_sensor_broadcaster/wrench",
            self._wrench_cb, sensor_qos,
        )
        self.start_fm = self.create_client(SetForceMode, "/force_mode_controller/start_force_mode")
        self.stop_fm = self.create_client(Trigger, "/force_mode_controller/stop_force_mode")

        # TF for wrench frame transform (tool0_controller -> base_link)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _wrench_cb(self, msg):
        self.wrench = msg

    def transform_wrench_to_base(self, wrench: WrenchStamped) -> tuple[tuple[float, float, float], tuple[float, float, float], bool]:
        """Transform wrench from its native frame (tool0_controller) to base_link.

        Returns (force_base, torque_base, transform_succeeded).
        On failure, returns the raw (sensor-frame) values with succeeded=False.
        """
        f_raw = (wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z)
        t_raw = (wrench.wrench.torque.x, wrench.wrench.torque.y, wrench.wrench.torque.z)
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", wrench.header.frame_id or "tool0_controller",
                rclpy.time.Time(),   # latest
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return f_raw, t_raw, False
        q = tf.transform.rotation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w])
        f_base = R.apply(np.asarray(f_raw))
        t_base = R.apply(np.asarray(t_raw))
        return tuple(f_base), tuple(t_base), True

    def wait_for_wrench(self, timeout=5.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.wrench is not None:
                return True
        return False

    def start_floppy(self) -> bool:
        if not self.start_fm.wait_for_service(timeout_sec=3.0):
            print("  ERROR: start_force_mode service unavailable")
            return False
        req = SetForceMode.Request()
        req.task_frame.header.frame_id = "base_link"
        req.task_frame.pose.orientation.w = 1.0   # identity

        # All 6 axes compliant
        req.selection_vector_x = True
        req.selection_vector_y = True
        req.selection_vector_z = True
        req.selection_vector_rx = True
        req.selection_vector_ry = True
        req.selection_vector_rz = True

        # ZERO commanded wrench — operator drives all motion
        req.wrench.force.x = 0.0
        req.wrench.force.y = 0.0
        req.wrench.force.z = 0.0
        req.wrench.torque.x = 0.0
        req.wrench.torque.y = 0.0
        req.wrench.torque.z = 0.0

        req.type = 2   # NO_TRANSFORM (task_frame = base_link as-is)

        # Conservative speed caps
        req.speed_limits.linear.x = 0.02
        req.speed_limits.linear.y = 0.02
        req.speed_limits.linear.z = 0.02
        req.speed_limits.angular.x = 0.2
        req.speed_limits.angular.y = 0.2
        req.speed_limits.angular.z = 0.2

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

    def stop_floppy(self):
        if not self.in_force_mode:
            return
        if self.stop_fm.wait_for_service(timeout_sec=2.0):
            future = self.stop_fm.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        self.in_force_mode = False

    def stream_wrench(self, duration_s: float, label: str):
        """Print wrench at ~5 Hz for `duration_s`, with a label header.

        Prints BOTH raw (sensor-native, typically tool0_controller) and
        base_link-transformed values so operator can see the difference and
        confirm the transform is doing what it should.
        """
        print(f"\n  [{label}] streaming wrench for {duration_s:.0f}s...")
        print(f"  {'t':>5}  | "
              f"{'fx_raw':>9} {'fy_raw':>9} {'fz_raw':>9} | "
              f"{'fx_base':>9} {'fy_base':>9} {'fz_base':>9} | "
              f"{'tf':>4}")
        t_end = time.time() + duration_s
        last_print = 0.0
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.wrench is None:
                continue
            now = time.time()
            if now - last_print < 0.2:   # 5 Hz
                continue
            last_print = now
            f = self.wrench.wrench.force
            tq = self.wrench.wrench.torque
            elapsed = duration_s - (t_end - now)
            f_base, _t_base, ok = self.transform_wrench_to_base(self.wrench)
            tf_marker = "OK" if ok else "MISS"
            print(f"  {elapsed:5.1f}  | "
                  f"{f.x:+9.3f} {f.y:+9.3f} {f.z:+9.3f} | "
                  f"{f_base[0]:+9.3f} {f_base[1]:+9.3f} {f_base[2]:+9.3f} | "
                  f"{tf_marker:>4}")


def prompt(msg: str) -> str:
    """Read a line from stdin, raising KeyboardInterrupt naturally on Ctrl-C.

    Important: do NOT install a custom SIGINT handler in this script — Python's
    default SIGINT handler raises KeyboardInterrupt from blocking input(), which
    is what we want. A custom SIGINT handler that just sets a flag leaves input()
    blocking forever (the bug the operator hit).
    """
    try:
        return input(msg)
    except EOFError:
        return ""


def _is_program_running() -> bool:
    """Check whether external_control.urp (or whatever) is running on the pendant."""
    try:
        r = subprocess.run(
            ["ros2", "service", "call", "/dashboard_client/program_running",
             "ur_dashboard_msgs/srv/IsProgramRunning"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return False
    return "program_running=True" in r.stdout


def _ros_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def go_home_then_floppy(node: "FloppyTester") -> bool:
    """Reset robot to home then re-enter floppy force mode.

    Sequence: stop force mode -> switch to position controller -> move_home ->
    switch back to force mode -> start floppy. Each step prints status; failures
    are reported with detail and the function returns False so the caller can
    abort the test cleanly.
    """
    if not _is_program_running():
        print("  ERROR: pendant program not running. Press Play on the pendant and re-run this script.")
        return False

    print("  [reset] Stop force mode...")
    try:
        node.stop_floppy()
    except Exception as e:
        print(f"  WARN: stop_force_mode raised (non-fatal): {e}")
    time.sleep(0.5)

    print("  [reset] Switch to position controller (deactivate force, activate position)...")
    subprocess.run(
        ["ros2", "control", "switch_controllers", "--deactivate", FORCE_CTRL],
        capture_output=True, text=True, timeout=10,
    )
    time.sleep(0.3)
    r = subprocess.run(
        ["ros2", "control", "switch_controllers", "--activate", POS_CTRL],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0:
        print(f"  ERROR: failed to activate position controller: {r.stderr.strip()}")
        return False

    print("  [reset] Move home...")
    repo_root = _ros_repo_root()
    try:
        r = subprocess.run(
            [sys.executable, str(repo_root / "primitives" / "move_home.py")],
            capture_output=True, text=True, timeout=60, cwd=str(repo_root),
        )
    except subprocess.TimeoutExpired:
        print("  ERROR: move_home timed out (>60 s). Pendant program paused?")
        return False
    if '"result": "success"' not in r.stdout:
        print(f"  ERROR: move_home failed.")
        print(f"    stdout tail: {r.stdout[-300:]!r}")
        print(f"    stderr tail: {r.stderr[-200:]!r}")
        return False

    print("  [reset] Switch back to force mode (deactivate position, activate force)...")
    subprocess.run(
        ["ros2", "control", "switch_controllers", "--deactivate", POS_CTRL],
        capture_output=True, text=True, timeout=10,
    )
    time.sleep(0.3)
    r = subprocess.run(
        ["ros2", "control", "switch_controllers", "--activate", FORCE_CTRL],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0:
        print(f"  ERROR: failed to activate force_mode_controller: {r.stderr.strip()}")
        return False

    time.sleep(1.0)   # post-switch settle to avoid force spikes contaminating start
    if not node.start_floppy():
        print("  ERROR: start_force_mode failed after re-switch.")
        return False
    print("  [reset] Floppy mode re-active. Robot at home and yielding.\n")
    return True


def main():
    # Open logfile and tee stdout so every print() lands in both terminal + log.
    logfile = open(_LOG_PATH, "w", buffering=1)
    sys.stdout = _TeeStream(sys.__stdout__, logfile)
    print(f"[LOG] Tee'd output to: {_LOG_PATH}")

    rclpy.init()
    node = FloppyTester()

    # SIGTERM -> set flag (rare; operator-issued kill). SIGINT is intentionally
    # NOT overridden so Ctrl-C raises KeyboardInterrupt from blocking input(),
    # which the outer try/except handles cleanly.
    state = {"interrupted": False}
    def _on_sigterm(signum, frame):
        print(f"\n[!] SIGTERM received — cleanup starting")
        state["interrupted"] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        print("=" * 70)
        print("  F/T sign-convention verification — interactive floppy-mode test")
        print("=" * 70)
        print()
        print("  This will:")
        print("    1. Switch into force_mode_controller, start floppy (zero commanded wrench)")
        print("    2. Stream a BASELINE wrench (don't touch)")
        print("    3. For each of 6 push directions: home -> floppy -> stream during your push")
        print("    4. Cleanup: stop force mode, switch back to position controller")
        print("    Ctrl-C at any time -> clean cleanup")
        print()
        print("  SAFETY:")
        print("    - Robot yields to gentle hand pressure on the gripper.")
        print("    - Speed limits 0.02 m/s linear, 0.2 rad/s angular.")
        print("    - Damping 0.7, gain 0.5 — robot won't drift far on its own.")
        print()
        print("  WORKSPACE AXES (operator-confirmed):")
        print("    +X = robot's LEFT       -X = robot's RIGHT")
        print("    +Y = BACK toward base   -Y = FORWARD away from base")
        print("    +Z = UP (ceiling)       -Z = DOWN (floor)")
        print()
        print("  Reads wrench in tool0_controller (sensor frame), transforms to base_link")
        print("  via TF. The fx_base/fy_base/fz_base columns are the ones that should")
        print("  match the directions above.")
        print()
        prompt("  Stand near the robot. Press Enter when ready to enter force mode... ")

        print("\n[1/3] Waiting for wrench topic...")
        if not node.wait_for_wrench(timeout=5.0):
            print("  ERROR: /force_torque_sensor_broadcaster/wrench is silent. Bringup running?")
            return

        print("[2/3] Initial switch into force_mode_controller for the BASELINE window...")
        if not switch_controllers(activate=[FORCE_CTRL], deactivate=[POS_CTRL]):
            print("  ERROR: controller switch failed.")
            return
        time.sleep(1.0)   # post-switch settle

        print("[3/3] Starting force mode (zero commanded wrench, 6-DOF compliant)...")
        if not node.start_floppy():
            print("  ERROR: start_force_mode failed.")
            return
        print("  Force mode ACTIVE. Floppy.\n")

        # ---- BASELINE (no home reset needed; we just entered force mode) ----
        node.stream_wrench(5.0, "BASELINE — DO NOT TOUCH the robot")
        if state["interrupted"]: return

        # ---- Per-push pattern: home + floppy + push window ----
        push_steps = [
            ("Step 1: push +X (robot's LEFT)",
             "Push the gripper GENTLY to the robot's LEFT.",
             "Expected: fx_base goes POSITIVE."),
            ("Step 2: push -X (robot's RIGHT)",
             "Push the gripper GENTLY to the robot's RIGHT.",
             "Expected: fx_base goes NEGATIVE."),
            ("Step 3: push +Y (BACK toward base)",
             "Push the gripper GENTLY BACKWARD toward the robot base.",
             "Expected: fy_base goes POSITIVE."),
            ("Step 4: push -Y (FORWARD away from base)",
             "Push the gripper GENTLY FORWARD into the workspace, away from the robot base.",
             "Expected: fy_base goes NEGATIVE."),
            ("Step 5: push +Z (LIFT UP toward ceiling)",
             "LIFT the gripper gently UPWARD toward the ceiling.",
             "Expected: fz_base goes POSITIVE.   (Critical post-fix check.)"),
            ("Step 6: push -Z (PUSH DOWN toward floor)",
             "PUSH DOWN gently on the gripper.",
             "Expected: fz_base goes NEGATIVE.   (May be small — robot yields.)"),
        ]

        for label, instruction, expected in push_steps:
            if state["interrupted"]:
                return
            print(f"\n  ----- {label} -----")
            print(f"  [pre-push] Resetting to home before push...")
            if not go_home_then_floppy(node):
                print(f"  Aborting test — go_home_then_floppy failed before {label}")
                return
            print(f"  {instruction}")
            print(f"  {expected}")
            prompt("  Press Enter when you're touching the gripper, ready to push... ")
            node.stream_wrench(5.0, label.split(":")[1].strip() if ":" in label else label)

        print("\n" + "=" * 70)
        print("  VERIFICATION COMPLETE")
        print("=" * 70)
        print("  Sign convention summary (look at the *_base columns, not the *_raw):")
        print("    +X push (robot's LEFT)         -> fx_base POSITIVE")
        print("    -X push (robot's RIGHT)        -> fx_base NEGATIVE")
        print("    +Y push (BACK toward base)     -> fy_base POSITIVE")
        print("    -Y push (FORWARD into workspace) -> fy_base NEGATIVE")
        print("    +Z lift (UP)                   -> fz_base POSITIVE")
        print("    -Z push (DOWN)                 -> fz_base NEGATIVE")
        print()

    except KeyboardInterrupt:
        print("\n[!] Ctrl-C received — cleanup starting")

    finally:
        # Cleanup pattern matches the proven _real_mode_stash/peg_in_hole_insertion.py
        # `restore_controllers()` flow: stop force mode FIRST → settle → 2-step switch
        # (deactivate, settle, activate) to avoid transients per PITFALLS.md.
        print("\n" + "=" * 70)
        print("[CLEANUP] Step 1/3: Stopping force mode (idempotent)...")
        print("=" * 70)
        try:
            node.stop_floppy()
            print("  ✓ stop_force_mode complete")
        except Exception as e:
            print(f"  WARNING: stop_force_mode error (non-fatal): {e}")

        time.sleep(0.5)   # settle to avoid transient when switching

        print("[CLEANUP] Step 2/3: Deactivating force_mode_controller...")
        try:
            r = subprocess.run(
                ["ros2", "control", "switch_controllers", "--deactivate", FORCE_CTRL],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                print(f"  ✓ Deactivated {FORCE_CTRL}")
            else:
                print(f"  WARNING: deactivate failed (non-fatal): {r.stderr.strip()}")
        except Exception as e:
            print(f"  WARNING: deactivate error (non-fatal): {e}")

        time.sleep(0.5)

        print(f"[CLEANUP] Step 3/3: Activating {POS_CTRL}...")
        try:
            r = subprocess.run(
                ["ros2", "control", "switch_controllers", "--activate", POS_CTRL],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                print(f"  ✓ Activated {POS_CTRL}")
            else:
                print(f"  ERROR: activate failed: {r.stderr.strip()}")
                print(f"  Robot may be in an indeterminate state — check with 'ros2 control list_controllers'")
        except Exception as e:
            print(f"  ERROR: activate error: {e}")

        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

        print("=" * 70)
        print("[CLEANUP] Done. Robot back in position-controlled mode.")
        print(f"[LOG] Full session log: {_LOG_PATH}")
        print("=" * 70)
        try:
            logfile.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
