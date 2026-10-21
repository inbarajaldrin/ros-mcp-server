#!/usr/bin/env python3
"""The planner's side of the seams (2026-10-20, M3a): call drive-mate, ExecuteMate, vision and Grip by mate id.

    python3 -m seams.client drive   --mate fmb1/mate:00 --target X Y Z QX QY QZ QW --held X Y Z QX QY QZ QW
    python3 -m seams.client execute --mate fmb1/mate:00
    python3 -m seams.client pose    --name base1 --kind base
    python3 -m seams.client grip    --width 0.021 [--expect u_brown --span 0.021]
    python3 -m seams.client mate-for --part u_brown --base base1

Each prints `__SEAM__ {json}` last. The replay (the reference planner) runs these as subprocesses, like every
primitive, so it never holds an rclpy context of its own. Part names are allowed here: this is the planner layer.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
B = yaml.safe_load(open(REPO / 'seams' / 'bindings.yaml'))


def mate_for(part, base):
    for asm, A in B.items():
        if asm == 'sigma' or A.get('reference') != base:
            continue
        for addr, m in A['mates'].items():
            if m['mover'] == part:
                return f'{asm}/{addr}'
    return None


def _nan(v):
    return None if isinstance(v, float) and math.isnan(v) else v


def _vp(VisionPose, frame, xyz, q):
    v = VisionPose(); v.header.frame_id = frame
    v.pose.position.x, v.pose.position.y, v.pose.position.z = map(float, xyz)
    v.pose.orientation.x, v.pose.orientation.y, v.pose.orientation.z, v.pose.orientation.w = map(float, q)
    return v


def _action(node, ActionT, name, goal, timeout):
    import rclpy
    from rclpy.action import ActionClient
    ac = ActionClient(node, ActionT, name)
    if not ac.wait_for_server(timeout_sec=5.0):
        return None, f'{name}: no server'
    f = ac.send_goal_async(goal); rclpy.spin_until_future_complete(node, f, timeout_sec=10.0)
    gh = f.result()
    if gh is None or not gh.accepted:
        return None, f'{name}: goal rejected'
    r = gh.get_result_async(); rclpy.spin_until_future_complete(node, r, timeout_sec=timeout)
    if r.result() is None:
        return None, f'{name}: no result in {timeout:.0f} s'
    return r.result().result, ''


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='op', required=True)
    d = sp.add_parser('drive'); d.add_argument('--mate', required=True)
    d.add_argument('--target', type=float, nargs=7, required=True); d.add_argument('--held', type=float, nargs=7, required=True)
    d.add_argument('--timeout', type=float, default=650.0)
    e = sp.add_parser('execute'); e.add_argument('--mate', required=True); e.add_argument('--timeout', type=float, default=400.0)
    p = sp.add_parser('pose'); p.add_argument('--name', required=True); p.add_argument('--kind', default='base')
    p.add_argument('--timeout', type=float, default=15.0)
    g = sp.add_parser('grip'); g.add_argument('--width', type=float, required=True); g.add_argument('--force', type=float, default=0.0)
    g.add_argument('--expect', default=''); g.add_argument('--span', type=float, default=0.0)
    m = sp.add_parser('mate-for'); m.add_argument('--part', required=True); m.add_argument('--base', required=True)
    a = ap.parse_args()
    if a.op == 'mate-for':
        print('__SEAM__ ' + json.dumps({'mate': mate_for(a.part, a.base)})); return 0
    import rclpy
    rclpy.init(); node = rclpy.create_node('seams_client'); out = {}
    try:
        if a.op == 'drive':
            from fmb_seam_msgs.action import DriveMate
            from fmb_seam_msgs.msg import VisionPose
            goal = DriveMate.Goal(); goal.mate_id = a.mate
            goal.target = _vp(VisionPose, 'base', a.target[:3], a.target[3:])
            goal.held = _vp(VisionPose, 'tool0', a.held[:3], a.held[3:])
            r, err = _action(node, DriveMate, '/ur5e/drive_mate', goal, a.timeout)
            out = {'error': err} if r is None else {
                'seated': r.seated, 'stage_reached': r.stage_reached, 'slide': [_nan(v) for v in r.slide],
                'measured_correction': [_nan(v) for v in r.measured_correction], 'trace_path': r.trace_path,
                'refusal': r.refusal}
        elif a.op == 'execute':
            from pa_assembly_msgs.action import ExecuteMate
            goal = ExecuteMate.Goal(); goal.mate = a.mate
            r, err = _action(node, ExecuteMate, '/ur5e/execute_mate', goal, a.timeout)
            out = {'error': err} if r is None else {
                'placed': r.placed, 'fastened': r.fastened, 'refusal': r.refusal, 'refusal_detail': r.refusal_detail,
                'order_position': r.order_position, 'granted_on': r.granted_on, 'next_move': r.next_move,
                'engagement_depth_m': _nan(r.engagement_depth_m), 'journalled': r.journalled}
        elif a.op == 'pose':
            from fmb_seam_msgs.srv import GetVisionPose
            c = node.create_client(GetVisionPose, '/vision/get_pose')
            if not c.wait_for_service(timeout_sec=5.0):
                out = {'error': '/vision/get_pose: no server'}
            else:
                req = GetVisionPose.Request(); req.name = a.name; req.kind = a.kind; req.timeout_s = a.timeout
                f = c.call_async(req); rclpy.spin_until_future_complete(node, f, timeout_sec=a.timeout + 150)
                r = f.result()
                if r is None:
                    out = {'error': 'no answer'}
                else:
                    pp = r.pose.pose
                    out = {'ok': r.ok, 'error': r.error, 'xyz': [pp.position.x, pp.position.y, pp.position.z],
                           'quat_xyzw': [pp.orientation.x, pp.orientation.y, pp.orientation.z, pp.orientation.w],
                           'sigma': [_nan(float(v)) for v in r.pose.sigma], 'source': r.pose.source}
        elif a.op == 'grip':
            from pa_gripper_msgs.action import Grip
            goal = Grip.Goal(); goal.width_m = a.width; goal.force = a.force; goal.expect = a.expect
            goal.expect_span_m = a.span
            r, err = _action(node, Grip, '/ur5e/rg2/grip', goal, 90.0)
            out = {'error': err} if r is None else {
                'verdict': r.verdict, 'reached': r.reached, 'holding': r.holding,
                'width_measured_m': _nan(r.width_measured_m), 'identity_verified': r.identity_verified}
    finally:
        node.destroy_node(); rclpy.shutdown()
    print('__SEAM__ ' + json.dumps(out, default=float), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
