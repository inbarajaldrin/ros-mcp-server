#!/usr/bin/env python3
"""The vision seam on ROS (2026-10-20, M3a; contract: ARCHITECTURE_SEAMS.md, fmb_seam_msgs).

    python3 -m seams.vision_server

  /vision/get_pose        fmb_seam_msgs/srv/GetVisionPose   {name, kind: base | part | held, timeout_s} -> VisionPose
  /vision/certify_seat    fmb_seam_msgs/srv/CertifySeat     {mate_id: fmb1/mate:00} -> inside, error, tilt, evidence
  /objects/poses          tf2_msgs/TFMessage                the twin lab's name for /objects_poses_real (relayed)

No motion: every answer is read from wherever the camera is (the planner picks and flies the view). Each request
runs seams.vision_ops in its own process - the readers start their own rclpy contexts - so one request at a time.
"""
import json
import math
import subprocess
import sys
import threading
from pathlib import Path

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from builtin_interfaces.msg import Time
from tf2_msgs.msg import TFMessage

from fmb_seam_msgs.msg import VisionPose
from fmb_seam_msgs.srv import CertifySeat, GetVisionPose

REPO = Path(__file__).resolve().parents[1]


def _t(s):
    return Time(sec=int(s), nanosec=int((s - int(s)) * 1e9))


def _run(args, timeout):
    p = subprocess.run(['python3', '-u', '-m', 'seams.vision_ops', *args], cwd=str(REPO), capture_output=True,
                       text=True, timeout=timeout)
    line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith('__VISION__ ')), None)
    if line is None:
        return {'ok': False, 'error': f'vision_ops rc={p.returncode}: ' + (p.stderr.strip().splitlines() or [''])[-1][:300]}
    return json.loads(line[len('__VISION__ '):])


class VisionServer(Node):
    def __init__(self):
        super().__init__('fmb_vision')
        one = MutuallyExclusiveCallbackGroup()                 # one read at a time: they share the camera
        self.create_service(GetVisionPose, '/vision/get_pose', self.get_pose, callback_group=one)
        self.create_service(CertifySeat, '/vision/certify_seat', self.certify, callback_group=one)
        relay = self.create_publisher(TFMessage, '/objects/poses', 10)
        self.create_subscription(TFMessage, '/objects_poses_real', relay.publish, 10,
                                 callback_group=ReentrantCallbackGroup())
        self.get_logger().info('vision seam: /vision/get_pose, /vision/certify_seat, /objects/poses')

    def get_pose(self, req, res):
        t = req.timeout_s if req.timeout_s > 0 else 15.0
        d = _run(['pose', '--name', req.name, '--kind', req.kind, '--timeout', str(t)], timeout=t + 120)
        res.ok = bool(d.get('ok')); res.error = d.get('error', '')
        if res.ok:
            v = VisionPose()
            v.header.frame_id = d['frame_id']; v.header.stamp = _t(d['stamp'])
            v.name = d['name']
            (v.pose.position.x, v.pose.position.y, v.pose.position.z) = d['xyz']
            (v.pose.orientation.x, v.pose.orientation.y, v.pose.orientation.z, v.pose.orientation.w) = d['quat_xyzw']
            v.sigma = [float('nan') if s is None else float(s) for s in d['sigma']]
            v.stationary_since = _t(d['stationary_since'])
            v.source = d['source']; v.view_z = float(d['view_z']); v.detection_set = d['detection_set']
            v.n = int(d['n']); v.symmetry = d['symmetry']
            res.pose = v
        self.get_logger().info(f"get_pose {req.name}/{req.kind}: {'ok ' + res.pose.source if res.ok else res.error}")
        return res

    def certify(self, req, res):
        d = _run(['certify', '--mate', req.mate_id], timeout=240)
        res.inside = bool(d.get('inside', False))
        e = d.get('error_mate_frame_m') or [math.nan] * 3
        res.error_mate_frame_m = [float('nan') if v is None else float(v) for v in e]
        res.tilt_deg = float('nan') if d.get('tilt_deg') is None else float(d['tilt_deg'])
        res.method = d.get('method', 'none')
        res.evidence_json = json.dumps(d.get('evidence') or d, default=float)
        self.get_logger().info(f"certify {req.mate_id}: inside={res.inside} via {res.method}")
        return res


def main():
    rclpy.init()
    n = VisionServer()
    ex = MultiThreadedExecutor(); ex.add_node(n)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
