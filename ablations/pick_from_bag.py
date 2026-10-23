#!/usr/bin/env python3
"""Was a pick off because the camera read was off, or because the grasp moved the part?

From one part_loop diagnostic bag (record_demo_bag DIAG=1) and the insert meta of the same iteration:

  close        the gripper-width drop of the PICK at the clear area (the first close after the regrasp hover)
  read         the part's grasp point from /grasp_points_real in the 3 s before the close, while the TCP is still
               (median + spread), and the part pose from /objects_poses_real (+ _raw)
  tcp@close    where the gripper actually closed (pad centre = flange + grasp depth along tool z)
  pick err     grasp point (camera) - gripper centre at the close, in the gripper's frame: along the closing axis
               the pads centre the part, across it (finger width) nothing does
  seat err     the insert's force-found seat - its aim (meta), world mm

Run after a part_loop: python3 ablations/pick_from_bag.py ~/Documents/fmb1_frames/part_loop/u_brown_*_it1 [...]
(source /opt/ros/humble/setup.bash first - rosbag2_py)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def read_bag(path, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    out = {t: [] for t in topics}
    while r.has_next():
        topic, data, t_ns = r.read_next()
        out[topic].append((t_ns * 1e-9, deserialize_message(data, get_message(types[topic]))))
    return out


def analyse(bag, part='u_brown', gid=1):
    d = read_bag(bag, ['/gripper_width', '/gripper_width_offset', '/tcp_pose_broadcaster/pose', '/grasp_points_real',
                       '/objects_poses_real', '/objects_poses_real_raw'])
    # tip-to-tip gap: /gripper_width_offset when recorded, else the plate spacing minus the RG2 fingertips
    # (2 x 4.6 mm, the fingertip_offset_mm of prompt2assembly's rg2.grasp-geometry.yaml)
    src = d['/gripper_width_offset'] or d['/gripper_width']
    gw = np.array([(t, m.data - (0.0 if d['/gripper_width_offset'] else 9.2)) for t, m in src])
    if len(gw) < 10:
        return {'bag': bag.name, 'error': 'no gripper width'}
    # closes: tip gap falling by > 8 mm within 1.5 s and ending below 36 mm
    closes = []
    for i in range(len(gw)):
        j = np.searchsorted(gw[:, 0], gw[i, 0] + 1.5)
        if j < len(gw) and gw[i, 1] - gw[j, 1] > 8 and gw[j, 1] < 36 and (not closes or gw[i, 0] - closes[-1] > 5):
            closes.append(gw[i, 0])
    if not closes:
        return {'bag': bag.name, 'error': 'no close found'}
    tcp = [(t, m.pose) for t, m in d['/tcp_pose_broadcaster/pose']]
    tt = np.array([t for t, _ in tcp])
    rows = []
    for tc in closes:
        k = np.searchsorted(tt, tc)
        P = tcp[min(k, len(tcp) - 1)][1]
        flange = np.array([P.position.x, P.position.y, P.position.z])
        Rf = R.from_quat([P.orientation.x, P.orientation.y, P.orientation.z, P.orientation.w])
        # grasp-point samples of this part in the 3 s before the close
        gp = []
        for t, m in d['/grasp_points_real']:
            if tc - 3.0 <= t <= tc:
                for mk in m.markers:
                    if mk.ns == part and mk.id == gid:      # one grasp point, not the part's three
                        gp.append([mk.pose.position.x, mk.pose.position.y, mk.pose.position.z])
        if not gp:
            rows.append(dict(close_t=tc, error='no grasp point in the 3 s before the close'))
            continue
        gp = np.array(gp); g = np.median(gp, 0)
        # gripper centre at the close: flange + 0.2286 m along tool z (GRIPPER_CENTER_TOOL_OFFSET)
        gc = flange + Rf.apply([0, 0, 0.2286])
        e_w = 1000 * (g - gc)
        e_t = Rf.inv().apply(e_w)              # tool frame: x = closing axis of the RG2, y = across the fingers
        rows.append(dict(close_t=tc, n=len(gp), spread=float(1000 * np.median(np.linalg.norm(gp - g, axis=1))),
                         pick_err_tool=[round(v, 1) for v in e_t[:2]], pick_err_world=[round(v, 1) for v in e_w[:2]],
                         flange=[round(1000 * v, 1) for v in flange]))
    return {'bag': bag.name, 'closes': rows}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('bags', nargs='+'); ap.add_argument('--part', default='u_brown')
    ap.add_argument('--grasp-id', type=int, default=1)
    a = ap.parse_args()
    for b in a.bags:
        res = analyse(Path(b), a.part, a.grasp_id)
        if 'error' in res:
            print(f"{res['bag']}: {res['error']}"); continue
        for r in res['closes']:
            if 'error' in r:
                print(f"{res['bag']} close@{r['close_t']:.0f}: {r['error']}"); continue
            print(f"{res['bag']} close@{r['close_t']:.0f}: grasp-point samples {r['n']} (spread {r['spread']:.1f} mm); "
                  f"camera grasp point - gripper centre: tool (closing x, across y) {r['pick_err_tool']} mm, "
                  f"world {r['pick_err_world']} mm")


if __name__ == '__main__':
    main()
