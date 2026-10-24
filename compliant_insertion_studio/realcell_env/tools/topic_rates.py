#!/usr/bin/env python3
"""Read-only: subscribe to the cell's topics for WINDOW_S seconds and report the receive rate, inter-arrival spread,
and (for TFMessage pose streams) the parent frame and the set of children seen. Publishes nothing, calls nothing.

    source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash
    python3 topic_rates.py [WINDOW_S] > ../evidence/topic_rates_<stamp>.json
"""
import json
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from rosidl_runtime_py.utilities import get_message

TOPICS = {
    '/objects_poses_real': 'tf2_msgs/msg/TFMessage',
    '/objects/poses': 'tf2_msgs/msg/TFMessage',
    '/objects_poses_real_raw': 'tf2_msgs/msg/TFMessage',
    '/joint_states': 'sensor_msgs/msg/JointState',
    '/tcp_pose_broadcaster/pose': 'geometry_msgs/msg/PoseStamped',
    '/camera/camera/color/image_raw': 'sensor_msgs/msg/Image',
    '/camera/camera/aligned_depth_to_color/image_raw': 'sensor_msgs/msg/Image',
    '/grasp_points_real': None,
    '/gripper_width_offset': 'std_msgs/msg/Float32',
    '/gripper_width': 'std_msgs/msg/Float32',
    '/gripper_grasp_detected': 'std_msgs/msg/Bool',
    '/force_torque_sensor_broadcaster/wrench': 'geometry_msgs/msg/WrenchStamped',
    '/camera_pose': 'geometry_msgs/msg/PoseStamped',
}


def main():
    window = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    rclpy.init()
    node = rclpy.create_node('realcell_env_rate_probe')
    names = dict(node.get_topic_names_and_types())
    rec = {}
    for t, ty in TOPICS.items():
        if t not in names:
            rec[t] = {'present': False}
            continue
        ty = ty or names[t][0]
        rec[t] = {'present': True, 'type': ty, 'stamps': [], 'frames': set(), 'children': set(), 'joint_names': None,
                  'last': None}
        msg_t = get_message(ty)

        def cb(m, t=t):
            r = rec[t]; r['stamps'].append(time.monotonic())
            if hasattr(m, 'transforms'):
                for tr in m.transforms:
                    r['frames'].add(tr.header.frame_id); r['children'].add(tr.child_frame_id)
            if hasattr(m, 'name') and isinstance(getattr(m, 'name'), (list, tuple)) and r['joint_names'] is None:
                r['joint_names'] = list(m.name)
            if hasattr(m, 'data') and isinstance(m.data, (bool, int, float)):
                r['last'] = m.data
            if hasattr(m, 'header') and hasattr(m.header, 'frame_id'):
                r['frames'].add(m.header.frame_id)
        qos = qos_profile_sensor_data if 'camera' in t else QoSProfile(depth=50)
        node.create_subscription(msg_t, t, cb, qos)
    t_end = time.monotonic() + window
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
    out = {'window_s': window, 'measured_at_unix': time.time(), 'topics': {}}
    for t, r in rec.items():
        if not r.get('present'):
            out['topics'][t] = r; continue
        s = np.array(r['stamps'])
        d = np.diff(s) if len(s) > 1 else np.array([])
        out['topics'][t] = {
            'type': r['type'], 'n': int(len(s)),
            'rate_hz': round(float((len(s) - 1) / (s[-1] - s[0])), 2) if len(s) > 2 else 0.0,
            'gap_s': {'median': round(float(np.median(d)), 4), 'max': round(float(d.max()), 4)} if len(d) else None,
            'frames': sorted(r['frames']), 'children': sorted(r['children']), 'joint_names': r['joint_names'],
            'last_value': r['last']}
    node.destroy_node(); rclpy.shutdown()
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
