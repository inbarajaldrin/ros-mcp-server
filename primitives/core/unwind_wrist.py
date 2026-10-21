#!/usr/bin/env python3
"""Unwind wrist_3 by whole turns when it has wound up (2026-10-20).

Views, pushes and inserts each turn the wrist within their own step limits, but nothing bounded the total: over a
day wrist_3 reached 6.30 rad (past its 2*pi limit: two protective stops) with the wrist camera's worn cable a full
turn around the wrist. Run this where it is safe - gripper empty (or the part well clear of everything), at safe
height: it rotates wrist_3 alone by -sign(q)*2*pi*k toward 0 (the same physical pose), slowly.

    python3 -m primitives.core.unwind_wrist --mode real [--threshold 3.5] [--seconds-per-turn 30] [--dry-run]
Prints __RESULT_JSON__ like the other primitives.
"""
import argparse
import json
import sys
import time

import numpy as np

ORDER = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


def result(d):
    print("__RESULT_JSON__"); print(json.dumps(d)); print("__END_RESULT_JSON__", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='real', choices=['real'])
    ap.add_argument('--threshold', type=float, default=3.5, help='unwind when |wrist_3| exceeds this (rad)')
    ap.add_argument('--seconds-per-turn', type=float, default=30.0)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    import rclpy
    from rclpy.action import ActionClient
    from sensor_msgs.msg import JointState
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    rclpy.init(); n = rclpy.create_node('unwind_wrist'); st = {}
    n.create_subscription(JointState, '/joint_states', lambda m: st.__setitem__('j', m), 10)
    t = time.time()
    while 'j' not in st and time.time() - t < 5:
        rclpy.spin_once(n, timeout_sec=0.1)
    if 'j' not in st:
        result({'result': 'failure', 'error': 'no joint states'}); return 1
    J = dict(zip(st['j'].name, st['j'].position))
    q0 = [J[k] for k in ORDER]; w3 = q0[5]
    if abs(w3) <= a.threshold:
        result({'result': 'success', 'wrist_3': round(w3, 3), 'unwound_turns': 0}); return 0
    k = int(np.round(w3 / (2 * np.pi)))
    target = w3 - 2 * np.pi * k
    if abs(target) > np.pi:
        result({'result': 'failure', 'error': f'wrist_3 {w3:.2f}: no whole-turn target within +-pi'}); return 1
    print(f"[unwind] wrist_3 {w3:+.3f} -> {target:+.3f} rad ({-k:+d} turn)", flush=True)
    if a.dry_run:
        result({'result': 'success', 'dry_run': True, 'wrist_3': round(w3, 3), 'target': round(target, 3)}); return 0
    ac = ActionClient(n, FollowJointTrajectory, '/scaled_joint_trajectory_controller/follow_joint_trajectory')
    if not ac.wait_for_server(timeout_sec=5):
        result({'result': 'failure', 'error': 'no trajectory controller'}); return 1
    g = FollowJointTrajectory.Goal(); g.trajectory.joint_names = ORDER
    T = a.seconds_per_turn * abs(k); N = 20 * abs(k)
    for i in range(1, N + 1):
        s = i / N; s_ = (1 - np.cos(np.pi * s)) / 2
        p = JointTrajectoryPoint(); q = list(q0); q[5] = w3 + (target - w3) * s_; p.positions = q; tt = T * s
        p.time_from_start = Duration(sec=int(tt), nanosec=int((tt - int(tt)) * 1e9)); g.trajectory.points.append(p)
    f = ac.send_goal_async(g); rclpy.spin_until_future_complete(n, f, timeout_sec=10)
    gh = f.result()
    if gh is None or not gh.accepted:
        result({'result': 'failure', 'error': 'goal rejected'}); return 1
    r = gh.get_result_async(); rclpy.spin_until_future_complete(n, r, timeout_sec=T + 15)
    ok = r.result() is not None and r.result().result.error_code == 0
    result({'result': 'success' if ok else 'failure', 'wrist_3_from': round(w3, 3), 'wrist_3_to': round(target, 3),
            'unwound_turns': -k})
    n.destroy_node(); rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
