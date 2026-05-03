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
    try:
        return input(msg)
    except EOFError:
        return ""


def main():
    # Open logfile and replace stdout with a tee so every print() lands in both
    # the terminal and the timestamped log under logs/diagnostics/.
    logfile = open(_LOG_PATH, "w", buffering=1)
    sys.stdout = _TeeStream(sys.__stdout__, logfile)
    print(f"[LOG] Tee'd output to: {_LOG_PATH}")

    rclpy.init()
    node = FloppyTester()

    state = {"interrupted": False}
    def _exit(signum, frame):
        print(f"\n[!] Signal {signum} — cleanup starting")
        state["interrupted"] = True

    signal.signal(signal.SIGINT, _exit)
    signal.signal(signal.SIGTERM, _exit)

    try:
        print("=" * 70)
        print("  F/T sign-convention verification — interactive floppy-mode test")
        print("=" * 70)
        print()
        print("  This will:")
        print("    1. Switch the robot into force_mode_controller")
        print("    2. Start force mode with ZERO commanded wrench (all 6 axes loose)")
        print("    3. Walk you through 6 push directions, printing fx/fy/fz")
        print("    4. On Ctrl-C: stop force mode, switch back to position controller")
        print()
        print("  SAFETY:")
        print("    - The robot will yield to gentle hand pressure on the gripper.")
        print("    - Speed limits are 0.02 m/s linear, 0.2 rad/s angular.")
        print("    - Damping 0.7, gain 0.5 — robot won't drift far on its own.")
        print()
        prompt("  Stand near the robot. Press Enter when ready to enter force mode... ")

        print("\n[1/4] Waiting for wrench topic...")
        if not node.wait_for_wrench(timeout=5.0):
            print("  ERROR: /force_torque_sensor_broadcaster/wrench is silent. Bringup running?")
            return

        print("[2/4] Switching to force_mode_controller...")
        if not switch_controllers(activate=[FORCE_CTRL], deactivate=[POS_CTRL]):
            print("  ERROR: controller switch failed.")
            return
        time.sleep(1.0)   # post-switch settle to avoid force spikes

        print("[3/4] Starting force mode (zero commanded wrench, 6-DOF compliant)...")
        if not node.start_floppy():
            print("  ERROR: start_force_mode failed.")
            return

        print("[4/4] Force mode ACTIVE. Robot is now floppy.\n")

        # ---- Step 0: baseline ----
        node.stream_wrench(5.0, "BASELINE — DO NOT TOUCH the robot")
        if state["interrupted"]: return

        # ---- Step 1: +X ----
        print("\n  ----- Step 1: push +X -----")
        print("  Push the gripper GENTLY in the +X direction (away from robot base, toward the front of the workspace).")
        print("  Hold for ~5 seconds.")
        prompt("  Press Enter when ready to start streaming...")
        node.stream_wrench(5.0, "PUSH +X")
        if state["interrupted"]: return

        # ---- Step 2: -X ----
        print("\n  ----- Step 2: push -X -----")
        print("  Now push GENTLY in the -X direction (back toward the robot base).")
        prompt("  Press Enter when ready...")
        node.stream_wrench(5.0, "PUSH -X")
        if state["interrupted"]: return

        # ---- Step 3: +Y ----
        print("\n  ----- Step 3: push +Y -----")
        print("  Push GENTLY in the +Y direction (your LEFT if you face the robot from the front).")
        prompt("  Press Enter when ready...")
        node.stream_wrench(5.0, "PUSH +Y")
        if state["interrupted"]: return

        # ---- Step 4: -Y ----
        print("\n  ----- Step 4: push -Y -----")
        print("  Push GENTLY in the -Y direction (your RIGHT).")
        prompt("  Press Enter when ready...")
        node.stream_wrench(5.0, "PUSH -Y")
        if state["interrupted"]: return

        # ---- Step 5: +Z ----
        print("\n  ----- Step 5: push +Z (LIFT UP) -----")
        print("  LIFT the gripper gently UPWARD (toward the ceiling).")
        print("  This is the critical sign-convention check — SCHEMA.md says fz should go MORE POSITIVE.")
        prompt("  Press Enter when ready...")
        node.stream_wrench(5.0, "LIFT UP (+Z)")
        if state["interrupted"]: return

        # ---- Step 6: -Z ----
        print("\n  ----- Step 6: push -Z (PUSH DOWN) -----")
        print("  Now PUSH DOWN gently on the gripper (toward the floor).")
        print("  SCHEMA.md says fz should go MORE NEGATIVE.")
        prompt("  Press Enter when ready...")
        node.stream_wrench(5.0, "PUSH DOWN (-Z)")
        if state["interrupted"]: return

        print("\n" + "=" * 70)
        print("  VERIFICATION COMPLETE")
        print("=" * 70)
        print("  Review the streams above and report back to me which axes match")
        print("  SCHEMA.md's sign conventions. Cleanup will run on Ctrl-C.")
        print()
        print("  Press Ctrl-C when done reviewing to stop force mode and switch back.")
        print()

        # Hold force mode until operator Ctrl-Cs
        while not state["interrupted"]:
            rclpy.spin_once(node, timeout_sec=0.5)

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
