#!/usr/bin/env python3
"""Render a demo bag (record_demo_bag.sh) to RGB frames + localizer-annotated frames (2026-10-17).

    python3 render_bag.py BAG_DIR OUT_DIR --localizer-src PKG_DIR --wireframes DIR [--workers N]
    bash encode_frames.sh OUT_DIR NAME        # -> NAME_{rgb,annotated,side_by_side}.mp4

The overlay is drawn with the localizer's OWN code (aruco_camera_localizer.drawing_functions.draw_text /
draw_object_lines, its world-frame wireframe path via geometric_functions.transform_points_world_to_img,
its RobotConfig for intrinsics and the hand-eye offset) from the poses it PUBLISHED during the run —
nothing is re-detected. Per RGB frame (capture stamp t): camera pose = TCP pose interpolated at t composed
with the hand-eye offset (as localizer_bridge.get_camera_pose); parts = the latest /objects_poses_real
message recorded <= t and < 1 s old. OUT_DIR/rgb/<t>.jpg are the camera JPEGs as recorded (not
re-encoded); OUT_DIR/annotated/<t>.jpg carry the overlay. Meant for dual-a4500, not the robot PC.
"""
import argparse
import bisect
import datetime
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

STALE_NS = 1_000_000_000
G = {}


def read_bag(bag, out_rgb):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}
    tcp, objs, n = [], [], 0
    while r.has_next():
        topic, data, t_rec = r.read_next()
        m = deserialize_message(data, types[topic])
        if topic.endswith('/compressed'):
            ns = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
            with open(os.path.join(out_rgb, f'{ns}.jpg'), 'wb') as f:
                f.write(bytes(m.data))
            n += 1
        elif topic == '/tcp_pose_broadcaster/pose':
            p, q = m.pose.position, m.pose.orientation
            tcp.append((m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec,
                        [p.x, p.y, p.z, q.x, q.y, q.z, q.w]))
        elif topic == '/objects_poses_real':
            objs.append((t_rec, [{'name': x.child_frame_id,
                                  'position': np.array([x.transform.translation.x, x.transform.translation.y,
                                                        x.transform.translation.z]),
                                  'quaternion': np.array([x.transform.rotation.x, x.transform.rotation.y,
                                                          x.transform.rotation.z, x.transform.rotation.w])}
                                 for x in m.transforms]))
    tcp.sort(key=lambda x: x[0]); objs.sort(key=lambda x: x[0])
    return n, tcp, objs


def init(state):
    G.update(state)
    sys.path.insert(0, G['src'])
    from aruco_camera_localizer.robot_config import RobotConfig
    cfg = RobotConfig(os.path.join(G['src'], 'config', 'robot_config.yaml'))
    fx, fy, cx, cy = cfg.camera_matrix
    G['K'] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    G['off_p'], G['off_q'] = np.array(cfg.cam_offset_position), np.array(cfg.cam_offset_quat)
    G['tcp_t'] = [x[0] for x in G['tcp']]
    G['obj_t'] = [x[0] for x in G['objs']]
    import json
    G['wf'] = {}
    for f in os.listdir(G['wfdir']):
        if f.endswith('_wireframe.json'):
            d = json.load(open(os.path.join(G['wfdir'], f)))
            G['wf'][f[:-len('_wireframe.json')]] = (d['vertices'], d['edges'])


def tcp_at(t):
    ts, tcp = G['tcp_t'], G['tcp']
    i = bisect.bisect_left(ts, t)
    if i <= 0 or i >= len(ts):
        return tcp[min(max(i, 0), len(ts) - 1)][1]
    (t0, a), (t1, b) = tcp[i - 1], tcp[i]
    f = (t - t0) / max(t1 - t0, 1)
    p = np.array(a[:3]) * (1 - f) + np.array(b[:3]) * f
    return list(p) + list(Slerp([0, 1], R.from_quat([a[3:], b[3:]]))([f]).as_quat()[0])


def render(name):
    from aruco_camera_localizer.drawing_functions import draw_text, draw_object_lines
    from aruco_camera_localizer.geometric_functions import transform_points_world_to_img
    t = int(name.split('.')[0])
    frame = cv2.imread(os.path.join(G['out'], 'rgb', name))
    ee = tcp_at(t)
    r_ee = R.from_quat(ee[3:])
    cam_pos = np.array(ee[:3]) + r_ee.apply(G['off_p'])           # localizer_bridge.get_camera_pose
    cam_quat = (r_ee * R.from_quat(G['off_q'])).as_quat()
    i = bisect.bisect_right(G['obj_t'], t) - 1
    objs = G['objs'][i][1] if i >= 0 and t - G['obj_t'][i] <= STALE_NS else []
    stamp = datetime.datetime.fromtimestamp(t / 1e9).strftime('%H:%M:%S.%f')[:-4]
    # Same calls and order as merged_localization's "Publish and draw" block
    draw_text(frame, cam_pos, cam_quat, objs, stamp, np.array(ee[:3]), np.array(ee[3:]), euler_convention='extrinsic')
    draw_object_lines(frame, G['K'], cam_pos, cam_quat, objs, [])
    for obj in objs:                                               # world-frame wireframe path
        if obj['name'] not in G['wf']:
            continue
        verts, edges = G['wf'][obj['name']]
        rot = R.from_quat(obj['quaternion']).as_matrix()
        world_pts = [obj['position'] + rot @ np.array(v) for v in verts]
        img_pts = transform_points_world_to_img(world_pts, cam_pos, cam_quat, G['K'])
        if len(img_pts) != len(verts):
            continue
        projected = np.array(img_pts, dtype=int)
        for e in edges:
            if len(e) >= 2 and e[0] < len(projected) and e[1] < len(projected):
                p1, p2 = tuple(map(int, projected[e[0]])), tuple(map(int, projected[e[1]]))
                if all(-2000 < c < 4000 for c in p1 + p2):
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(G['out'], 'annotated', name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag'); ap.add_argument('out')
    ap.add_argument('--localizer-src', required=True, help='aruco_camera_localizer package dir (has config/)')
    ap.add_argument('--wireframes', required=True, help='aruco-grasp-annotator/data/wireframe')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    a = ap.parse_args()
    out = os.path.realpath(a.out)
    for k in ('rgb', 'annotated'):
        os.makedirs(os.path.join(out, k), exist_ok=True)
    n, tcp, objs = read_bag(a.bag, os.path.join(out, 'rgb'))
    print(f"bag: {n} RGB frames, {len(tcp)} TCP poses, {len(objs)} part-pose messages", flush=True)
    state = dict(out=out, src=os.path.realpath(a.localizer_src), wfdir=a.wireframes, tcp=tcp, objs=objs)
    names = sorted(os.listdir(os.path.join(out, 'rgb')))
    with Pool(a.workers, initializer=init, initargs=(state,)) as p:
        p.map(render, names, chunksize=32)
    print(f"rendered {len(names)} annotated frames -> {out}/annotated")


if __name__ == '__main__':
    main()
