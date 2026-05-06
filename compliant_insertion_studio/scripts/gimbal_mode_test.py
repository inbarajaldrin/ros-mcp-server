#!/usr/bin/env python3
"""
Standalone gimbal-stabilization test for the UR5e force_mode controller.

Puts the robot into XY-compliant, Z-locked, rotation-locked force mode. The
operator can push the EE laterally and XY yields; Z height + TCP orientation
remain locked. The controller actively counter-rotates wrist joints as XY
translates so the canonical face-down orientation is preserved (this is the
"gimbal" effect — TCP orientation is anchored even as the operator drags the
arm around in the workspace).

Selection vector:
  X = compliant (operator's hand drives motion)
  Y = compliant
  Z = LOCKED (cannot be pushed up/down)
  Rx, Ry, Rz = LOCKED (orientation held at current value)

Usage:
  # Robot must be at a sensible starting pose (e.g. hover, post-rotate,
  # or any pose with EE at canonical face-down).
  python3 -m compliant_insertion_studio.scripts.gimbal_mode_test

  # Operator pushes EE around. Press Ctrl+C when done — script stops force
  # mode and switches back to scaled_joint_trajectory_controller cleanly.

Why this exists: validates the gimbal mode in isolation before integrating
it into the GUIDED state of the data-collection wrapper. If this script
behaves correctly (XY moves freely, Z + orientation locked), we know the
GUIDED state will too.
"""
import argparse
import re
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from ur_msgs.srv import SetForceMode


SERVICE_START = "/force_mode_controller/start_force_mode"
SERVICE_STOP  = "/force_mode_controller/stop_force_mode"
POS_CTRL = "scaled_joint_trajectory_controller"
FORCE_CTRL = "force_mode_controller"


def _switch_controllers(activate, deactivate):
    cmd = ["ros2", "control", "switch_controllers"]
    for c in activate:   cmd += ["--activate", c]
    for c in deactivate: cmd += ["--deactivate", c]
    print(f"  switch_controllers: -{deactivate} +{activate}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if res.returncode != 0:
        print(f"  switch_controllers FAILED: {res.stderr.strip() or res.stdout.strip()}")
        return False
    # Strip ANSI escapes (per CLAUDE.md note) before parsing
    out = re.sub(r"\x1b\[[0-9;]*m", "", res.stdout)
    return True


def _read_tcp_pose():
    """One-shot TCP pose read via ros2 topic echo. Returns (xyz, quat_xyzw) or None."""
    try:
        out = subprocess.run(
            ["timeout", "2", "ros2", "topic", "echo", "--once", "/tcp_pose_broadcaster/pose"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return None
    mp = re.search(r"position:\s*\n\s*x:\s*([-\d.eE+]+)\s*\n\s*y:\s*([-\d.eE+]+)\s*\n\s*z:\s*([-\d.eE+]+)", out)
    mq = re.search(r"orientation:\s*\n\s*x:\s*([-\d.eE+]+)\s*\n\s*y:\s*([-\d.eE+]+)\s*\n\s*z:\s*([-\d.eE+]+)\s*\n\s*w:\s*([-\d.eE+]+)", out)
    if not (mp and mq):
        return None
    xyz = tuple(float(v) for v in mp.groups())
    quat = tuple(float(v) for v in mq.groups())
    return xyz, quat


class GimbalNode(Node):
    def __init__(self):
        super().__init__("gimbal_mode_test")
        self.start_fm = self.create_client(SetForceMode, SERVICE_START)
        self.stop_fm  = self.create_client(Trigger, SERVICE_STOP)

    def start_gimbal(self, gain=1.0, damping=0.5, lin_speed=0.10, ang_speed=0.5):
        if not self.start_fm.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"{SERVICE_START} not available")
            return False
        req = SetForceMode.Request()
        req.task_frame.header.frame_id = "base"
        req.task_frame.pose.orientation.w = 1.0  # identity in base
        # GIMBAL: XY compliant (True), Z locked (False), rotation locked (False)
        req.selection_vector_x = True
        req.selection_vector_y = True
        req.selection_vector_z = False
        req.selection_vector_rx = False
        req.selection_vector_ry = False
        req.selection_vector_rz = False
        # Zero wrench — we want pure compliance, no commanded direction
        req.wrench.force.x = 0.0
        req.wrench.force.y = 0.0
        req.wrench.force.z = 0.0
        req.wrench.torque.x = 0.0
        req.wrench.torque.y = 0.0
        req.wrench.torque.z = 0.0
        req.type = SetForceMode.Request.NO_TRANSFORM  # = 2
        # Speed limits (cap how fast operator can drag the EE)
        req.speed_limits.linear.x = float(lin_speed)
        req.speed_limits.linear.y = float(lin_speed)
        req.speed_limits.linear.z = float(lin_speed)
        req.speed_limits.angular.x = float(ang_speed)
        req.speed_limits.angular.y = float(ang_speed)
        req.speed_limits.angular.z = float(ang_speed)
        req.damping_factor = float(damping)
        req.gain_scaling   = float(gain)

        future = self.start_fm.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        resp = future.result()
        if resp is None:
            self.get_logger().error("start_force_mode service call timed out")
            return False
        if not getattr(resp, "success", False):
            self.get_logger().error(f"start_force_mode returned success=False: {resp}")
            return False
        return True

    def stop_gimbal(self):
        if not self.stop_fm.wait_for_service(timeout_sec=3.0):
            return False
        future = self.stop_fm.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        resp = future.result()
        return resp is not None and getattr(resp, "success", False)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    p.add_argument("--gain", type=float, default=1.0,
                   help="force_mode gain_scaling (higher = more compliant) [default 1.0]")
    p.add_argument("--damping", type=float, default=0.5,
                   help="force_mode damping_factor (lower = easier to push) [default 0.5]")
    p.add_argument("--lin-speed", type=float, default=0.10,
                   help="max linear speed cap (m/s) [default 0.10 = 10cm/s]")
    p.add_argument("--ang-speed", type=float, default=0.5,
                   help="max angular speed cap (rad/s) [default 0.5]")
    p.add_argument("--print-pose-hz", type=float, default=2.0,
                   help="how often to print TCP pose during gimbal mode (Hz) [default 2.0]")
    args = p.parse_args()

    print("=" * 72)
    print("  GIMBAL MODE TEST")
    print("=" * 72)
    print(f"  Selection: X=COMPLIANT  Y=COMPLIANT  Z=LOCKED  Rx=LOCKED  Ry=LOCKED  Rz=LOCKED")
    print(f"  gain={args.gain}  damping={args.damping}  "
          f"lin_speed={args.lin_speed}m/s  ang_speed={args.ang_speed}rad/s")
    print()
    print(f"  Robot will yield to your lateral pushes (X, Y).")
    print(f"  Robot will RESIST any push in Z or any rotation — orientation is locked.")
    print()

    # Capture starting pose for reference
    start = _read_tcp_pose()
    if start:
        sx, sy, sz = start[0]
        qx, qy, qz, qw = start[1]
        print(f"  Starting TCP: ({sx*1000:+.1f}, {sy*1000:+.1f}, {sz*1000:+.1f}) mm  "
              f"quat=({qx:+.4f}, {qy:+.4f}, {qz:+.4f}, {qw:+.4f})")
    print()
    print("  Press Ctrl+C when done. Robot will return to position-control mode cleanly.")
    print("=" * 72)
    print()

    rclpy.init()
    node = GimbalNode()

    print("  [1/3] Switching to force_mode_controller...")
    if not _switch_controllers(activate=[FORCE_CTRL], deactivate=[POS_CTRL]):
        print("  ERROR: controller switch failed. Aborting.")
        rclpy.shutdown()
        return 2

    print("  [2/3] Starting force mode (gimbal config)...")
    ok = node.start_gimbal(gain=args.gain, damping=args.damping,
                            lin_speed=args.lin_speed, ang_speed=args.ang_speed)
    if not ok:
        print("  ERROR: start_force_mode failed. Reverting controller switch.")
        _switch_controllers(activate=[POS_CTRL], deactivate=[FORCE_CTRL])
        rclpy.shutdown()
        return 3

    print("  [3/3] Gimbal mode ACTIVE.  Push the EE around. Ctrl+C to exit.")
    print()

    # Periodically print TCP pose so operator sees how it moves
    interval = 1.0 / max(args.print_pose_hz, 0.1)
    last_print = 0.0
    interrupted = {"flag": False}
    def handle_sigint(signum, frame):
        interrupted["flag"] = True
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)  # so external 'kill' triggers clean shutdown

    try:
        while not interrupted["flag"]:
            now = time.time()
            if now - last_print >= interval:
                last_print = now
                pose = _read_tcp_pose()
                if pose:
                    (x, y, z), (qx, qy, qz, qw) = pose
                    # Quick tilt computation
                    ee_z_world = (
                        2*(qx*qz + qy*qw),
                        2*(qy*qz - qx*qw),
                        1 - 2*(qx*qx + qy*qy)
                    )
                    import math
                    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -ee_z_world[2]))))
                    if start:
                        dx = (x - start[0][0]) * 1000
                        dy = (y - start[0][1]) * 1000
                        dz = (z - start[0][2]) * 1000
                        print(f"    TCP=({x*1000:+.1f}, {y*1000:+.1f}, {z*1000:+.1f})mm  "
                              f"Δ=({dx:+.1f}, {dy:+.1f}, {dz:+.1f})mm  tilt={tilt:.3f}°")
                    else:
                        print(f"    TCP=({x*1000:+.1f}, {y*1000:+.1f}, {z*1000:+.1f})mm  tilt={tilt:.3f}°")
            time.sleep(0.05)
    finally:
        print()
        print("  Stopping force mode...")
        node.stop_gimbal()
        time.sleep(0.5)
        print("  Switching back to scaled_joint_trajectory_controller...")
        _switch_controllers(activate=[POS_CTRL], deactivate=[FORCE_CTRL])
        end = _read_tcp_pose()
        if end and start:
            (ex, ey, ez), _ = end
            sx, sy, sz = start[0]
            print(f"  Final TCP: ({ex*1000:+.1f}, {ey*1000:+.1f}, {ez*1000:+.1f}) mm  "
                  f"Δfrom start: ({(ex-sx)*1000:+.1f}, {(ey-sy)*1000:+.1f}, {(ez-sz)*1000:+.1f}) mm")
        rclpy.shutdown()
    print("  Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
