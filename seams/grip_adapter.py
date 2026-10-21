#!/usr/bin/env python3
"""The real RG2's adapter onto prompt2assembly's parallel_jaw contract: pa_gripper_msgs/action/Grip (2026-10-20, M3a).

    python3 -m seams.grip_adapter            # serves /ur5e/rg2/grip

The contract (pa_gripper_msgs/action/Grip.action, unchanged): {width_m, force, expect, expect_span_m} ->
{reached, holding} + evidence. The twin's own RG2 adapter (pa_rg2_adapter) is a ros2_control-style node; this cell's
RG2 runs through onrobot_ros over the tool serial bridge, so per the seam rule ("one thin adapter per gripper") this is
the cell's adapter. It holds no width: every number in the result is read back from the gripper.

  motion     the cell's calibrated gripper primitive (primitives.control_gripper: open / close / width in mm, with its
             settle detection), so the command path is the one every real run used
  width      /gripper_width_offset (tip-to-tip gap, mm; plate spacing minus 2 x 4.6 mm, prompt2assembly's
             rg2.grasp-geometry fingertip_offset_mm)
  holding    /gripper_grasp_detected (the RG2's own stall/grip flag), or stalled > STALL_MM short of a closing command
  force      STUB (seam left open). goal.force is recorded, not applied: onrobot_ros's rg_gripper node passes no force,
             so every command runs at the driver default 400 (= 40 N, the RG2 maximum). The driver itself can
             (custom_libraries/onrobot.py move_gripper(width, force_val): register 0, units 0.1 N, 0..400); opening
             it = rg_gripper.command_callback accepting "width,force" + passing FORCE_N here. Not needed yet.
             The RG2 reports no effort, so the evidence carries NaN for it
  identity   expect_span_m > 0 -> width_vs_span: |measured gap - declared span| <= SPAN_TOL_M
"""
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

from pa_gripper_msgs.action import Grip

REPO = Path(__file__).resolve().parents[1]
ACTION = '/ur5e/rg2/grip'
OPEN_M = 0.100                 # RG2 full opening (tip gap ~100 mm)
REACH_TOL_M = 0.002
STALL_M = 0.002
SPAN_TOL_M = 0.003
FORCE_APPLIED_N = 40.0         # what the RG2 actually closes at today (driver default 400 x 0.1 N)


class GripAdapter(Node):
    def __init__(self):
        super().__init__('rg2_grip_adapter')
        self.gap = None; self.grasped = None; self.moving = None
        self.create_subscription(Float32, '/gripper_width_offset', lambda m: setattr(self, 'gap', m.data / 1000.0), 10)
        self.create_subscription(Bool, '/gripper_grasp_detected', lambda m: setattr(self, 'grasped', bool(m.data)), 10)
        self.create_subscription(Bool, '/gripper_motion_ongoing', lambda m: setattr(self, 'moving', bool(m.data)), 10)
        self.events = self.create_publisher(String, '/ur5e/rg2/grip_events', 10)
        ActionServer(self, Grip, ACTION, self.execute, callback_group=ReentrantCallbackGroup())
        self.get_logger().info(f'Grip contract served at {ACTION}')

    def execute(self, gh):
        g = gh.request
        res = Grip.Result()
        res.width_commanded_m = float(g.width_m)
        res.expected_part = g.expect
        res.discovered_part = ''                      # a rig has no discovery
        if not (0.0 <= g.width_m <= OPEN_M + 0.005):
            res.verdict = 'refused'; gh.abort(); return res
        cmd = 'open' if g.width_m >= OPEN_M - 0.001 else ('close' if g.width_m <= 0.001 else f"{1000 * g.width_m:.1f}")
        p = subprocess.run(['python3', '-u', '-m', 'primitives.control_gripper', cmd, '--mode', 'real'],
                           cwd=str(REPO), capture_output=True, text=True, timeout=60)
        time.sleep(0.3)
        w = self.gap
        res.width_measured_m = float('nan') if w is None else float(w)
        res.theta_measured_rad = float('nan')          # onrobot_ros reports no joint angle
        res.effort_measured_nm = float('nan')          # nor an effort
        res.stall_margin_m = float('nan') if w is None else float(w - g.width_m)
        closing = cmd == 'close' or (w is not None and g.width_m < w)
        res.reached = w is not None and abs(w - g.width_m) <= REACH_TOL_M
        res.holding = bool(self.grasped) or (w is not None and cmd != 'open' and w - g.width_m > STALL_M)
        if g.expect_span_m > 0 and w is not None:
            res.identity_verified = abs(w - g.expect_span_m) <= SPAN_TOL_M
            res.identity_authority = 'width_vs_span'
        else:
            res.identity_verified = False; res.identity_authority = 'none'
        res.attach_applied = False; res.attached_body = ''; res.attach_drift_mm = float('nan'); res.attach_refusal = ''
        res.verdict = 'holding' if res.holding else ('reached' if res.reached else ('refused' if p.returncode else 'unreachable'))
        ev = {'goal': {'width_m': g.width_m, 'force': g.force, 'expect': g.expect, 'expect_span_m': g.expect_span_m},
              'cmd': cmd, 'rc': p.returncode, 'width_measured_m': res.width_measured_m, 'holding': res.holding,
              'reached': res.reached, 'identity_verified': res.identity_verified, 'verdict': res.verdict,
              'force_requested_n': g.force, 'force_applied_n': FORCE_APPLIED_N,
              'force_note': 'stub: requested force not applied (onrobot_ros passes none; seam open)'}
        self.events.publish(String(data=json.dumps(ev)))
        gh.succeed()
        return res


def main():
    rclpy.init()
    n = GripAdapter()
    ex = MultiThreadedExecutor(); ex.add_node(n)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
