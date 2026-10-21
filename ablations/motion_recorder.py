#!/usr/bin/env python3
"""Record when the arm is actually moving (2026-10-21, Speed track T0).

    python3 -u ablations/motion_recorder.py OUT.jsonl

One line per transition: {"t": wall time, "moving": true|false, "xyz": [...]}. Moving = TCP speed above
MOVING_MPS over a SPEED_WINDOW_S window. Independent of what any primitive reports, so cycle_time.py can split a run's
wall time into arm-moving and arm-still exactly. Ends on SIGINT/SIGTERM.
"""
import json
import signal
import sys
import time
from collections import deque

import rclpy
from geometry_msgs.msg import PoseStamped

MOVING_MPS = 0.002
SPEED_WINDOW_S = 0.1


def main():
    out = open(sys.argv[1], 'a', buffering=1)
    rclpy.init(); n = rclpy.create_node('motion_recorder')
    buf = deque(); st = {'moving': None}

    def cb(m):
        t = time.time(); p = (m.pose.position.x, m.pose.position.y, m.pose.position.z)
        buf.append((t, p))
        while buf and buf[0][0] < t - SPEED_WINDOW_S:
            buf.popleft()
        if len(buf) < 2 or buf[-1][0] - buf[0][0] < 0.5 * SPEED_WINDOW_S:
            return
        d = sum((a - b) ** 2 for a, b in zip(buf[-1][1], buf[0][1])) ** 0.5
        moving = d / (buf[-1][0] - buf[0][0]) > MOVING_MPS
        if moving != st['moving']:
            st['moving'] = moving
            out.write(json.dumps({'t': t, 'moving': moving, 'xyz': [round(v, 4) for v in p]}) + '\n')

    n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', cb, 50)
    stop = {'v': False}
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.__setitem__('v', True))
    while not stop['v']:
        rclpy.spin_once(n, timeout_sec=0.1)
    out.write(json.dumps({'t': time.time(), 'end': True}) + '\n')
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
