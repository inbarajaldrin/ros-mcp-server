#!/usr/bin/env python3
"""
Verify the wrapper's mental model: command motion in base_link, observe TCP move in base_link.

For each of 6 cardinal directions in base_link (+X, -X, +Y, -Y, +Z, -Z):
  1. Reset to home
  2. Snapshot TCP pose (transformed to base_link)
  3. Compute the SetForceMode command that produces motion in commanded base_link direction
     (accounting for the base ↔ base_link 180° rotation discovered while debugging
     verify_axis_motion.py)
  4. Drive the robot for 3 s
  5. Stream TCP pose (in base_link) so operator + Claude both see the trajectory
  6. Stop force mode, report the TCP delta vector in base_link
  7. Operator confirms visually that motion direction matches expectation

The script and operator BOTH speak base_link — operator never has to think about base or
tool0_controller. All frame conversion happens internally per:

  base ↔ base_link rotation matrix (verified live 2026-05-03):
    R = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    => base +X = base_link -X
    => base +Y = base_link -Y
    => base +Z = base_link +Z (preserved)

Live frame facts (also verified):
  - /tcp_pose_broadcaster/pose publishes in `base` frame (not base_link)
  - /force_torque_sensor_broadcaster/wrench publishes in `tool0_controller`
  - force_mode_controller transforms task_frame from header.frame_id to `base` internally,
    then URScript interprets the wrench in that transformed task_frame

Strategy: send task_frame with header.frame_id="base" + identity orientation, so what we
write IS what URScript receives — no auto-transform surprise. Then convert base_link
direction -> base direction (flip X, Y signs) before populating the wrench.

Safety:
  - 2 N commanded (well under 5 N CONVENTIONS cap)
  - 0.02 m/s linear speed cap (~6 cm displacement over 3 s)
  - Single-axis compliance (others held in position)
  - Ctrl-C anytime -> KeyboardInterrupt -> proven 2-step cleanup pattern
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
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from ur_msgs.srv import SetForceMode
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


POS_CTRL = "scaled_joint_trajectory_controller"
FORCE_CTRL = "force_mode_controller"

_LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "diagnostics"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_PATH = _LOG_DIR / f"verify_baselink_motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class _TeeStream:
    def __init__(self, *streams): self._streams = streams
    def write(self, data):
        for s in self._streams:
            try: s.write(data); s.flush()
            except Exception: pass
    def flush(self):
        for s in self._streams:
            try: s.flush()
            except Exception: pass


class BaselinkMotionTester(Node):
    def __init__(self):
        super().__init__("verify_baselink_motion")
        self.tcp_in_base: PoseStamped | None = None
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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _tcp_cb(self, msg):
        # /tcp_pose_broadcaster/pose publishes in `base` frame natively.
        self.tcp_in_base = msg

    def wait_for_tcp(self, timeout=5.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tcp_in_base is not None:
                return True
        return False

    def _tcp_in_base_link(self) -> tuple[float, float, float] | None:
        """Read latest TCP pose, transform position into base_link.

        Returns (x, y, z) in base_link, or None if no TCP sample yet.
        """
        if self.tcp_in_base is None:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link",
                self.tcp_in_base.header.frame_id or "base",
                rclpy.time.Time(),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        # Apply rotation + translation
        q = tf.transform.rotation
        t = tf.transform.translation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w])
        p = self.tcp_in_base.pose.position
        p_xyz = np.asarray([p.x, p.y, p.z])
        p_base_link = R.apply(p_xyz) + np.asarray([t.x, t.y, t.z])
        return tuple(float(v) for v in p_base_link)

    def drive_baselink(self, axis_baselink: str, direction: int) -> bool:
        """Drive the robot in a base_link cardinal direction (visible ~6 cm in 3 s).

        Internally uses force_mode as the drive mechanism, with low damping +
        high gain tuned so the robot reaches the speed cap quickly. Converts
        base_link -> base (X and Y sign-flipped, Z preserved) before populating
        the request, and sends task_frame.header.frame_id="base" so there's no
        auto-transform surprise.

        axis_baselink: 'x' | 'y' | 'z' (cardinal in base_link)
        direction: +1 or -1
        """
        if axis_baselink not in ("x", "y", "z") or direction not in (1, -1):
            return False
        if not self.start_fm.wait_for_service(timeout_sec=3.0):
            print("  ERROR: start_force_mode service unavailable")
            return False

        # base ↔ base_link conversion: X and Y signs flipped, Z preserved
        base_direction = -direction if axis_baselink in ("x", "y") else direction

        # Drive params — picked so the robot actually moves visibly:
        #  - 3 N command (within 5 N CONVENTIONS cap)
        #  - damping_factor=0.025 (low — robot doesn't fight its own velocity)
        #  - gain_scaling=1.0 (responsive)
        # With these, the robot reaches the 0.02 m/s speed cap quickly →
        # ~6 cm displacement over the 3 s drive window.
        DRIVE_FORCE_N = 3.0
        DAMPING = 0.025
        GAIN = 1.0

        req = SetForceMode.Request()
        req.task_frame.header.frame_id = "base"
        req.task_frame.pose.orientation.w = 1.0   # identity in base

        # Single-axis compliance (same axis line label in base and base_link)
        req.selection_vector_x = (axis_baselink == "x")
        req.selection_vector_y = (axis_baselink == "y")
        req.selection_vector_z = (axis_baselink == "z")
        req.selection_vector_rx = False
        req.selection_vector_ry = False
        req.selection_vector_rz = False

        # Internal drive vector — wrench in `base` frame
        req.wrench.force.x = float(base_direction * DRIVE_FORCE_N) if axis_baselink == "x" else 0.0
        req.wrench.force.y = float(base_direction * DRIVE_FORCE_N) if axis_baselink == "y" else 0.0
        req.wrench.force.z = float(base_direction * DRIVE_FORCE_N) if axis_baselink == "z" else 0.0

        req.type = SetForceMode.Request.NO_TRANSFORM   # = 2

        req.speed_limits.linear.x = 0.02
        req.speed_limits.linear.y = 0.02
        req.speed_limits.linear.z = 0.02
        req.speed_limits.angular.x = 0.05
        req.speed_limits.angular.y = 0.05
        req.speed_limits.angular.z = 0.05

        req.gain_scaling = GAIN
        req.damping_factor = DAMPING

        future = self.start_fm.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None or not result.success:
            print("  ERROR: drive RPC failed")
            return False
        self.in_force_mode = True
        return True

    def stop(self):
        if not self.in_force_mode:
            return
        if self.stop_fm.wait_for_service(timeout_sec=2.0):
            future = self.stop_fm.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        self.in_force_mode = False

    def stream_tcp_in_baselink(self, duration_s: float, label: str,
                                start_xyz_bl: tuple[float, float, float]):
        """Print TCP pose in BASE_LINK + delta from start at 5 Hz for `duration_s`."""
        sx, sy, sz = start_xyz_bl
        print(f"\n  [{label}] streaming base_link TCP for {duration_s:.0f}s "
              f"(start: x={sx:+.4f}, y={sy:+.4f}, z={sz:+.4f})")
        print(f"  {'t':>5} | {'x_bl':>9} {'y_bl':>9} {'z_bl':>9} | "
              f"{'dx_bl':>9} {'dy_bl':>9} {'dz_bl':>9}")
        t_end = time.time() + duration_s
        last_print = 0.0
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.05)
            now = time.time()
            if now - last_print < 0.2:
                continue
            last_print = now
            xyz_bl = self._tcp_in_base_link()
            if xyz_bl is None:
                continue
            elapsed = duration_s - (t_end - now)
            dx, dy, dz = xyz_bl[0] - sx, xyz_bl[1] - sy, xyz_bl[2] - sz
            print(f"  {elapsed:5.1f} | {xyz_bl[0]:+9.4f} {xyz_bl[1]:+9.4f} {xyz_bl[2]:+9.4f} | "
                  f"{dx:+9.4f} {dy:+9.4f} {dz:+9.4f}")


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


def go_home_then_force(node: BaselinkMotionTester) -> bool:
    if not is_program_running():
        print("  ERROR: pendant program not running. Press Play and re-run.")
        return False
    print("  [reset] Stop force mode...")
    try: node.stop()
    except Exception as e: print(f"  WARN: {e}")
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
        print("  ERROR: move_home timeout (>60 s).")
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
    return True


def prompt(msg: str) -> str:
    try: return input(msg)
    except EOFError: return ""


def main():
    logfile = open(_LOG_PATH, "w", buffering=1)
    sys.stdout = _TeeStream(sys.__stdout__, logfile)
    print(f"[LOG] Tee'd output to: {_LOG_PATH}")

    rclpy.init()
    node = BaselinkMotionTester()

    state = {"interrupted": False}
    def _on_sigterm(signum, frame):
        print(f"\n[!] SIGTERM received — cleanup starting")
        state["interrupted"] = True
    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        print("=" * 70)
        print("  base_link motion verification — operator and script both speak base_link")
        print("=" * 70)
        print()
        print("  Per axis: home -> drive in commanded base_link direction for 3 s")
        print("           -> stream TCP pose in BASE_LINK -> stop -> verdict")
        print()
        print("  All numbers shown are in base_link frame.")
        print("  (Internal frame conversion to base, 180-deg Z flip, is hidden.)")
        print()
        print("  YOUR conventions (operator-confirmed):")
        print("    +X = robot's RIGHT            -X = robot's LEFT")
        print("    +Y = FORWARD (away from base) -Y = BACK toward base")
        print("    +Z = UP                       -Z = DOWN")
        print()
        print("  Speed cap 0.02 m/s -> ~6 cm displacement per 3 s window.")
        print("  Ctrl-C anytime for clean cleanup.")
        print()
        prompt("  Press Enter when ready to start (you should be near robot)... ")

        if not node.wait_for_tcp(timeout=5.0):
            print("  ERROR: /tcp_pose_broadcaster/pose silent. Bringup running?")
            return

        # Initial switch into force mode (subsequent axes use go_home_then_force)
        print("\n[init] Initial switch into force_mode_controller...")
        if not switch_controllers(activate=[FORCE_CTRL], deactivate=[POS_CTRL]):
            print("  ERROR: initial switch failed.")
            return
        time.sleep(1.0)

        axis_tests = [
            ("+X (robot's RIGHT)",     "x", +1),
            ("-X (robot's LEFT)",      "x", -1),
            ("+Y (FORWARD, away)",     "y", +1),
            ("-Y (BACK, toward base)", "y", -1),
            ("+Z (UP)",                "z", +1),
            ("-Z (DOWN)",              "z", -1),
        ]

        results = []
        for label, axis_bl, direction in axis_tests:
            if state["interrupted"]:
                return
            print(f"\n  ===== {label} =====")
            if not go_home_then_force(node):
                print(f"  Aborting — go_home_then_force failed before {label}")
                return

            # Snapshot start pose in base_link. Drop any cached TCP first +
            # spin until a FRESH sample arrives, otherwise we'd snapshot a
            # pre-home-motion pose and the start would be stale (caused the +Z
            # auto-verdict false-positive in the 2026-05-03 run).
            node.tcp_in_base = None
            t0 = time.time()
            while node.tcp_in_base is None and time.time() - t0 < 3.0:
                rclpy.spin_once(node, timeout_sec=0.05)
            # Tiny extra settle after first fresh sample, in case home motion is
            # still ringing down
            time.sleep(0.2)
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=0.05)
            start_xyz_bl = node._tcp_in_base_link()
            if start_xyz_bl is None:
                print("  ERROR: could not get start pose in base_link. Skipping axis.")
                continue

            print(f"  Driving base_link {label}...")
            if not node.drive_baselink(axis_baselink=axis_bl, direction=direction):
                print(f"  ERROR: drive_baselink failed")
                return

            node.stream_tcp_in_baselink(3.0, label, start_xyz_bl)
            node.stop()
            time.sleep(0.5)

            # Final delta
            end_xyz_bl = node._tcp_in_base_link()
            if end_xyz_bl is not None:
                dx, dy, dz = (end_xyz_bl[0] - start_xyz_bl[0],
                              end_xyz_bl[1] - start_xyz_bl[1],
                              end_xyz_bl[2] - start_xyz_bl[2])
                # Determine which axis dominated
                deltas = {"x": dx, "y": dy, "z": dz}
                dominant = max(deltas, key=lambda k: abs(deltas[k]))
                expected_axis = axis_bl
                expected_sign = direction
                actual_sign = 1 if deltas[dominant] > 0 else -1
                match = (dominant == expected_axis and actual_sign == expected_sign)
                marker = "✓ MATCH" if match else "✗ MISMATCH"
                print(f"\n  TCP delta in base_link: dx={dx:+.4f}  dy={dy:+.4f}  dz={dz:+.4f}")
                print(f"  Dominant axis: {dominant}{'+' if actual_sign > 0 else '-'}  "
                      f"Expected: {expected_axis}{'+' if expected_sign > 0 else '-'}  =>  {marker}")
                results.append((label, dx, dy, dz, dominant, actual_sign, match))
            ans = prompt(f"  Did robot move correctly for {label}? [y/N]: ").strip().lower()
            print(f"  Operator: {'YES' if ans == 'y' else 'NO/skip'}")

        print("\n" + "=" * 70)
        print("  SUMMARY (base_link frame)")
        print("=" * 70)
        for label, dx, dy, dz, dom, sign, match in results:
            marker = "OK " if match else "BAD"
            print(f"  {marker} {label:40s} dx={dx:+.4f} dy={dy:+.4f} dz={dz:+.4f}  "
                  f"dominant={dom}{'+' if sign > 0 else '-'}")
        any_bad = any(not r[6] for r in results)
        if any_bad:
            print("\n  ⚠️  At least one mismatch above means my base_link mental model is still")
            print("     wrong. Check the dominant axis vs commanded axis to see what's flipped.")
        else:
            print("\n  All axes match: TCP moved in the commanded base_link direction.")
            print("  Mental model verified. Wrapper can use the same base->base_link conversion.")

    except KeyboardInterrupt:
        print("\n[!] Ctrl-C — cleanup")

    finally:
        print("\n" + "=" * 70)
        print("[CLEANUP] Stop force mode...")
        try: node.stop(); print("  ✓ stop complete")
        except Exception as e: print(f"  WARN: {e}")
        time.sleep(0.5)
        print("[CLEANUP] Deactivate force_mode_controller...")
        subprocess.run(["ros2", "control", "switch_controllers", "--deactivate", FORCE_CTRL],
                      capture_output=True, text=True, timeout=10)
        time.sleep(0.5)
        print(f"[CLEANUP] Activate {POS_CTRL}...")
        r = subprocess.run(["ros2", "control", "switch_controllers", "--activate", POS_CTRL],
                          capture_output=True, text=True, timeout=10)
        print(f"  {'✓' if r.returncode == 0 else 'ERROR'}: {r.stderr.strip() or 'ok'}")
        try: node.destroy_node(); rclpy.shutdown()
        except Exception: pass
        print("=" * 70)
        print(f"[CLEANUP] Done. Robot in position controller. Log: {_LOG_PATH}")
        print("=" * 70)
        try: logfile.close()
        except Exception: pass


if __name__ == "__main__":
    main()
