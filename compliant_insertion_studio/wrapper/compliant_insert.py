#!/usr/bin/env python3
"""
Compliant Insert (TDD scaffold)
================================

Minimal force-mode descent for peg-in-hole testing on UR5e.

Behaviour:
  - Switches controller stack: scaled_joint_trajectory_controller -> force_mode_controller
  - Activates 6-DOF compliant force mode in base frame
      selection_vector = [1, 1, 1, 1, 1, 1]   (XYZ + RxRyRz all compliant)
      wrench           = [0, 0, -Fz, 0, 0, 0] (gentle steady push down)
      type             = 2 (task_frame is base_link, no transform)
  - Logs TCP pose + wrench at ~100 Hz to logs/insert_<obj>_<ts>.csv
  - Runs until --timeout OR SIGINT/SIGTERM (then cleans up: stop force mode,
    switch back to scaled_joint_trajectory_controller)
  - No success/jam exit logic — gather data first, tune later

Usage:
  python3 compliant_insert.py --object-name u_brown --fz 5.0 --timeout 60
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, WrenchStamped
from ur_msgs.srv import SetForceMode
from std_srvs.srv import Trigger


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def switch_controllers(activate, deactivate, logger=None):
    """Shell out to `ros2 control switch_controllers` (matches existing primitive style)."""
    cmd = ['ros2', 'control', 'switch_controllers']
    if deactivate:
        cmd += ['--deactivate'] + deactivate
    if activate:
        cmd += ['--activate'] + activate
    msg = f"Switching controllers: -{deactivate} +{activate}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    ok = result.returncode == 0
    if not ok and logger:
        logger.error(f"switch_controllers failed: {result.stderr.strip()}")
    return ok


class CompliantInsert(Node):
    def __init__(self, args):
        super().__init__('compliant_insert')
        self.args = args
        self.tcp = None
        self.wrench = None
        self.in_force_mode = False
        self.start_t = None
        self.csv_file = None
        self.csv_writer = None

        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self._tcp_cb, 10)
        self.create_subscription(
            WrenchStamped, '/force_torque_sensor_broadcaster/wrench', self._wrench_cb, 10
        )

        self.start_fm = self.create_client(SetForceMode, '/force_mode_controller/start_force_mode')
        self.stop_fm = self.create_client(Trigger, '/force_mode_controller/stop_force_mode')
        self.zero_ft = self.create_client(Trigger, '/io_and_status_controller/zero_ftsensor')

    def _tcp_cb(self, msg):
        self.tcp = msg

    def _wrench_cb(self, msg):
        self.wrench = msg

    def wait_for_topics(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tcp is not None and self.wrench is not None:
                return True
        return False

    def open_csv(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"insert_{self.args.object_name or 'unknown'}_{ts}.csv"
        path = LOG_DIR / name
        self.csv_file = open(path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            't_s',
            'tcp_x', 'tcp_y', 'tcp_z',
            'tcp_qx', 'tcp_qy', 'tcp_qz', 'tcp_qw',
            'fx', 'fy', 'fz',
            'tx', 'ty', 'tz',
        ])
        self.get_logger().info(f"Logging to {path}")
        return path

    def log_sample(self):
        if self.tcp is None or self.wrench is None:
            return
        t = time.time() - self.start_t
        p = self.tcp.pose.position
        q = self.tcp.pose.orientation
        f = self.wrench.wrench.force
        tq = self.wrench.wrench.torque
        self.csv_writer.writerow([
            f"{t:.4f}",
            f"{p.x:.6f}", f"{p.y:.6f}", f"{p.z:.6f}",
            f"{q.x:.6f}", f"{q.y:.6f}", f"{q.z:.6f}", f"{q.w:.6f}",
            f"{f.x:.4f}", f"{f.y:.4f}", f"{f.z:.4f}",
            f"{tq.x:.4f}", f"{tq.y:.4f}", f"{tq.z:.4f}",
        ])

    def call_zero_ftsensor(self, settle_s=0.5, bias_warn_n=2.0):
        """Zero the F/T sensor so force mode chases contact, not gravity bias.

        Reference: stash uses CLI `ros2 service call ... zero_ftsensor` then sleep.
        After zero, we wait `settle_s` for the broadcaster to publish fresh readings,
        then sample /wrench to verify residual bias is below `bias_warn_n` (warn only).
        """
        result = subprocess.run(
            ['ros2', 'service', 'call', '/io_and_status_controller/zero_ftsensor',
             'std_srvs/srv/Trigger'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or 'success=True' not in result.stdout:
            self.get_logger().error(f"zero_ftsensor failed: rc={result.returncode} out={result.stdout.strip()}")
            return False
        self.get_logger().info("zero_ftsensor: success — settling")

        # Drop any cached pre-zero wrench, then settle and sample fresh
        self.wrench = None
        t_settle_end = time.time() + settle_s
        while time.time() < t_settle_end:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.wrench is None:
            self.get_logger().warn("No /wrench sample arrived during settle window — proceeding anyway")
            return True

        f = self.wrench.wrench.force
        t = self.wrench.wrench.torque
        max_f = max(abs(f.x), abs(f.y), abs(f.z))
        max_t = max(abs(t.x), abs(t.y), abs(t.z))
        self.get_logger().info(
            f"Post-zero bias: F=({f.x:+.2f},{f.y:+.2f},{f.z:+.2f})N  "
            f"T=({t.x:+.3f},{t.y:+.3f},{t.z:+.3f})Nm"
        )
        if max_f > bias_warn_n:
            self.get_logger().warn(
                f"Residual force bias {max_f:.2f}N > {bias_warn_n}N — XY drift likely under 6-DOF compliance"
            )
        return True

    def call_start_force_mode(self):
        if not self.start_fm.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("start_force_mode service unavailable")
            return False

        req = SetForceMode.Request()
        req.task_frame.header.frame_id = 'base_link'
        req.task_frame.pose.orientation.w = 1.0  # identity

        # Full 6-DOF compliance
        req.selection_vector_x = True
        req.selection_vector_y = True
        req.selection_vector_z = True
        req.selection_vector_rx = True
        req.selection_vector_ry = True
        req.selection_vector_rz = True

        # Gentle steady downward push (base frame, type=2)
        req.wrench.force.x = 0.0
        req.wrench.force.y = 0.0
        req.wrench.force.z = -float(self.args.fz)
        req.wrench.torque.x = 0.0
        req.wrench.torque.y = 0.0
        req.wrench.torque.z = 0.0

        req.type = 2  # NO_TRANSFORM — task frame is base_link as-is

        # Speed caps for compliant axes (tune via CLI for slower/faster descent)
        req.speed_limits.linear.x = float(self.args.lin_speed)
        req.speed_limits.linear.y = float(self.args.lin_speed)
        req.speed_limits.linear.z = float(self.args.lin_speed)
        req.speed_limits.angular.x = float(self.args.ang_speed)
        req.speed_limits.angular.y = float(self.args.ang_speed)
        req.speed_limits.angular.z = float(self.args.ang_speed)

        # Higher gain_scaling = more responsive to applied force differential.
        # Lower damping_factor = less resistance to motion (yields easier when pushed).
        req.gain_scaling = float(self.args.gain)
        req.damping_factor = float(self.args.damping)

        future = self.start_fm.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None or not result.success:
            self.get_logger().error("start_force_mode call failed")
            return False
        self.in_force_mode = True
        self.get_logger().info(
            f"Force mode active: Fz=-{self.args.fz}N, all 6 axes compliant, base frame"
        )
        return True

    def call_stop_force_mode(self):
        if not self.in_force_mode:
            return
        if self.stop_fm.wait_for_service(timeout_sec=2.0):
            future = self.stop_fm.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            r = future.result()
            if r is not None and r.success:
                self.get_logger().info("Force mode stopped")
            else:
                self.get_logger().warn(f"stop_force_mode reported: {getattr(r, 'message', 'no response')}")
        self.in_force_mode = False

    def cleanup(self):
        try:
            self.call_stop_force_mode()
        except Exception as e:
            self.get_logger().error(f"stop_force_mode cleanup error: {e}")
        try:
            switch_controllers(
                activate=['scaled_joint_trajectory_controller'],
                deactivate=['force_mode_controller'],
                logger=self.get_logger(),
            )
        except Exception as e:
            self.get_logger().error(f"switch back error: {e}")
        try:
            if self.csv_file is not None:
                self.csv_file.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Compliant insert (TDD scaffold)")
    parser.add_argument('--object-name', type=str, default=None)
    parser.add_argument('--fz', type=float, default=3.0,
                        help='Downward force in N (gentle default 3.0)')
    parser.add_argument('--timeout', type=float, default=120.0,
                        help='Max runtime in seconds before clean exit')
    parser.add_argument('--rate-hz', type=float, default=100.0,
                        help='Logging rate in Hz')
    parser.add_argument('--lin-speed', type=float, default=0.02,
                        help='Max linear speed cap in m/s for compliant axes (smaller = slower descent)')
    parser.add_argument('--ang-speed', type=float, default=0.2,
                        help='Max angular speed cap in rad/s for compliant axes')
    parser.add_argument('--gain', type=float, default=1.0,
                        help='Force-mode gain_scaling [0..2]. Higher = more responsive to push (default 1.0)')
    parser.add_argument('--damping', type=float, default=0.005,
                        help='Force-mode damping_factor [0..1]. Lower = less resistance to motion (default 0.005)')
    args = parser.parse_args()

    rclpy.init()
    node = CompliantInsert(args)

    # Signals:
    #   SIGINT / SIGTERM -> full teardown (stop force mode + switch back to position controller)
    #   SIGUSR1          -> pause: stop force mode, hold position. Robot is frozen but
    #                       force_mode_controller stays loaded so resume is instant.
    #   SIGUSR2          -> resume: re-zero F/T sensor + restart force mode with same params
    state = {'interrupted': False, 'pause_request': False, 'resume_request': False, 'paused': False}

    def _exit_handler(signum, frame):
        node.get_logger().warn(f"Signal {signum} — full teardown requested")
        state['interrupted'] = True

    def _pause_handler(signum, frame):
        node.get_logger().warn("SIGUSR1 — pause requested (robot will freeze)")
        state['pause_request'] = True

    def _resume_handler(signum, frame):
        node.get_logger().warn("SIGUSR2 — resume requested (re-zero F/T + restart force mode)")
        state['resume_request'] = True

    signal.signal(signal.SIGINT, _exit_handler)
    signal.signal(signal.SIGTERM, _exit_handler)
    signal.signal(signal.SIGUSR1, _pause_handler)
    signal.signal(signal.SIGUSR2, _resume_handler)

    result = {"result": "failure", "error": "unknown"}

    try:
        if not node.wait_for_topics(timeout=5.0):
            node.get_logger().error("Timed out waiting for /tcp_pose or /wrench")
            result = {"result": "failure", "error": "topics not live"}
            return

        # Switch into force mode
        ok = switch_controllers(
            activate=['force_mode_controller'],
            deactivate=['scaled_joint_trajectory_controller'],
            logger=node.get_logger(),
        )
        if not ok:
            result = {"result": "failure", "error": "switch to force_mode_controller failed"}
            return

        # Settle after controller switch — switching can produce force spikes that
        # contaminate the zero call if done too soon (lesson from stash move_down).
        time.sleep(1.0)

        # Zero F/T sensor BEFORE entering force mode — otherwise the gravity load of
        # the gripper + payload biases the wrench reading and force mode chases the
        # bias instead of contact (causes wrong-direction drift).
        if not node.call_zero_ftsensor(settle_s=0.5):
            result = {"result": "failure", "error": "zero_ftsensor failed"}
            return

        if not node.call_start_force_mode():
            result = {"result": "failure", "error": "start_force_mode RPC failed"}
            return

        log_path = node.open_csv()
        node.start_t = time.time()
        period = 1.0 / max(args.rate_hz, 1.0)
        next_tick = node.start_t

        sample_count = 0
        pause_events = 0

        while not state['interrupted']:
            elapsed = time.time() - node.start_t
            if elapsed >= args.timeout:
                node.get_logger().info(f"Timeout {args.timeout}s reached — exiting")
                break

            # Handle pause/resume edges
            if state['pause_request'] and not state['paused']:
                node.call_stop_force_mode()
                state['paused'] = True
                pause_events += 1
                node.get_logger().info("Paused — robot frozen, force_mode_controller still loaded")
                state['pause_request'] = False
            if state['resume_request'] and state['paused']:
                node.call_zero_ftsensor(settle_s=0.5)
                if node.call_start_force_mode():
                    state['paused'] = False
                    node.get_logger().info("Resumed")
                else:
                    node.get_logger().error("Resume failed — restart_force_mode rejected")
                state['resume_request'] = False

            rclpy.spin_once(node, timeout_sec=0.005)
            if time.time() >= next_tick:
                # Mark paused samples with a flag column so post-mortem can split runs
                node.log_sample()
                sample_count += 1
                next_tick += period

        result = {
            "result": "success",
            "object_name": args.object_name,
            "fz_commanded_N": args.fz,
            "duration_s": round(time.time() - node.start_t, 2),
            "samples_logged": sample_count,
            "pause_events": pause_events,
            "csv_path": str(log_path),
            "interrupted": state['interrupted'],
        }

    finally:
        node.cleanup()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


if __name__ == '__main__':
    main()
