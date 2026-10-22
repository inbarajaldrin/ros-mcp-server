#!/usr/bin/env python3
"""Latest robot state on disk, for the replay's per-step snapshots (2026-10-21, Speed T1).

    python3 -u ablations/state_tap.py [--out ablations/logs/robot_state_latest.json] [--hz 20]

The replay took two snapshots per step, each a fresh python + rclpy process (~0.8-1 s apiece: the median 2.24 s
between steps, ASSEMBLY_STATE_MACHINE.md item 3). This one long-lived subscriber keeps the newest /joint_states and
/tcp_pose_broadcaster/pose and rewrites one JSON file atomically at --hz: {joints, tcp, t} in the snapshot's format,
t = wall time of the write. A reader trusts it only while t is fresh (replay_real_assembly.snapshot_robot_state).
"""
import argparse
import json
import os
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / 'ablations' / 'logs' / 'robot_state_latest.json'


class Tap(Node):
    def __init__(self):
        super().__init__('replay_state_tap')
        self.joints = None
        self.tcp = None
        self.create_subscription(JointState, '/joint_states', self._js, 10)
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self._tcp, 10)

    def _js(self, m):
        self.joints = {'name': list(m.name), 'position': [float(x) for x in m.position],
                       'stamp_s': m.header.stamp.sec + m.header.stamp.nanosec * 1e-9}

    def _tcp(self, m):
        p, q = m.pose.position, m.pose.orientation
        self.tcp = {'xyz': [p.x, p.y, p.z], 'quat_xyzw': [q.x, q.y, q.z, q.w],
                    'stamp_s': m.header.stamp.sec + m.header.stamp.nanosec * 1e-9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--hz', type=float, default=20.0)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix('.tmp')
    rclpy.init(); n = Tap(); period = 1.0 / a.hz; nxt = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(n, timeout_sec=period / 4)
            if time.time() >= nxt and n.joints is not None and n.tcp is not None:
                tmp.write_text(json.dumps({'joints': n.joints, 'tcp': n.tcp, 't': time.time()}))
                os.replace(tmp, out)
                nxt = time.time() + period
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
