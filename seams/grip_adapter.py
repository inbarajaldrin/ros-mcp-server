#!/usr/bin/env python3
"""The real RG2's adapter onto prompt2assembly's parallel_jaw contract: pa_gripper_msgs/action/Grip (2026-10-20, M3a).

    python3 -m seams.grip_adapter            # serves /ur5e/rg2/grip

The contract (pa_gripper_msgs/action/Grip.action, prompt2assembly's current one): {width_m, force, expect, expect_span_m} ->
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
  holding    (Grip.action as of prompt2assembly robot-integration 3b9eeaf, plan47 2026-10-21) a WITNESSED hold of the
             part `expect` names; a rig has no contact witness, so: a closing move that stalled (the RG2's grip flag,
             or stopped > STALL_M short of the command) at a width within SPAN_TOL_M of `expect_span_m`.  Otherwise
             holding is false with not_holding_reason (judge(), below):
               not_a_close     the goal closed nothing (an open, a no-op, a goal refused unmoved)
               empty           a close that did not stall (nothing between the pads, or a full close)
               width_off_band  stalled, but at a width outside the declared span's band
               unknown         stalled with no span declared (or expect "none"), or no width reading: nothing
                               witnessed what is between the pads; never read as a hold
             and not_holding_detail the measured sentence.  verdict: holding | not_holding | reached | refused |
             unreachable.
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
REACH_TOL_M = 0.003            # the RG2 has no precise positioning mode (SETUP.md): 2.1 mm overshoot seen 10-21
STALL_M = 0.002
SPAN_TOL_M = 0.003
FORCE_APPLIED_N = 40.0         # what the RG2 actually closes at today (driver default 400 x 0.1 N)


def judge(cmd, width_m, w_before, w, grasped, expect, span_m, rc):
    """The verdict fields from the readings (no ROS).  cmd: 'open' | 'close' | '<mm>'; w_before / w: the measured gap
    (m) before / after, None when unread; grasped: the RG2's grip flag; span_m: the goal's expect_span_m."""
    closing = cmd == 'close' or (cmd != 'open' and w_before is not None and width_m < w_before)
    reached = w is not None and abs(w - width_m) <= REACH_TOL_M
    stalled = closing and w is not None and (bool(grasped) or w - width_m > STALL_M)
    has_span = bool(span_m and span_m > 0) and (expect or 'none') != 'none'
    identity = has_span and w is not None and abs(w - span_m) <= SPAN_TOL_M
    out = dict(reached=reached, identity_verified=bool(identity),
               identity_authority='width_vs_span' if (has_span and w is not None) else 'none')
    w_txt = 'unread' if w is None else f'{1000 * w:.1f} mm'
    if rc and not closing:
        reason, detail = 'not_a_close', f'the goal was refused (control_gripper rc {rc})'
    elif not closing:
        reason, detail = 'not_a_close', f'an opening / no-op move to {1000 * width_m:.1f} mm (measured {w_txt})'
    elif w is None:
        reason, detail = 'unknown', 'no width reading after the close'
    elif not stalled:
        reason, detail = 'empty', (f'the close to {1000 * width_m:.1f} mm did not stall (measured {w_txt}, grip flag '
                                   f'{bool(grasped)}): nothing between the pads')
    elif not has_span:
        reason, detail = 'unknown', (f'stalled at {w_txt} with no declared span (expect {expect!r}): nothing '
                                     f'witnessed what is between the pads')
    elif not identity:
        reason, detail = 'width_off_band', (f'stalled at {w_txt}, {1000 * abs(w - span_m):.1f} mm from the declared '
                                            f'span {1000 * span_m:.1f} mm (band +-{1000 * SPAN_TOL_M:.0f})')
    else:
        reason, detail = '', ''
    holding = closing and stalled and identity
    out.update(holding=bool(holding), not_holding_reason=reason, not_holding_detail=detail, closing=closing)
    out['verdict'] = ('holding' if holding else 'not_holding' if (closing and w is not None and stalled)
                      else 'reached' if reached else 'refused' if rc else 'not_holding' if closing else 'unreachable')
    return out


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
        w0 = self.gap                                   # before the move: decides closing vs opening
        p = subprocess.run(['python3', '-u', '-m', 'primitives.control_gripper', cmd, '--mode', 'real'],
                           cwd=str(REPO), capture_output=True, text=True, timeout=60)
        time.sleep(0.3)
        w = self.gap
        res.width_measured_m = float('nan') if w is None else float(w)
        res.theta_measured_rad = float('nan')          # onrobot_ros reports no joint angle
        res.effort_measured_nm = float('nan')          # nor an effort
        res.stall_margin_m = float('nan') if w is None else float(w - g.width_m)
        j = judge(cmd, float(g.width_m), w0, w, self.grasped, g.expect, float(g.expect_span_m), p.returncode)
        closing = j['closing']
        res.reached, res.holding = j['reached'], j['holding']
        res.identity_verified, res.identity_authority = j['identity_verified'], j['identity_authority']
        res.not_holding_reason, res.not_holding_detail = j['not_holding_reason'], j['not_holding_detail']
        res.attach_applied = False; res.attached_body = ''; res.attach_drift_mm = float('nan'); res.attach_refusal = ''
        res.verdict = j['verdict']
        ev = {'goal': {'width_m': g.width_m, 'force': g.force, 'expect': g.expect, 'expect_span_m': g.expect_span_m},
              'cmd': cmd, 'closing': closing, 'width_before_m': w0, 'rc': p.returncode, 'width_measured_m': res.width_measured_m, 'holding': res.holding,
              'reached': res.reached, 'identity_verified': res.identity_verified, 'verdict': res.verdict,
              'not_holding_reason': res.not_holding_reason, 'not_holding_detail': res.not_holding_detail,
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
