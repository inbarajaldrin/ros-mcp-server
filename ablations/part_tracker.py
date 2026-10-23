#!/usr/bin/env python3
"""Where every part was last seen, from ANY view (2026-10-21).

    python3 -u ablations/part_tracker.py [OUT.json]        # default ablations/logs/part_tracker.json

Each read in the replay looks for its own target and throws away every other part in the frame: u_orange was in
the base view (the marker verify printed its pose) and the run then failed with "not seen from above its last
position nor from the staging view". This keeps one table of last sightings for all of them: a localizer pose counts
only if the TCP was still for STILL_WINDOW_S before it was published (poses from a moving camera are ~10 mm off);
per name, the median of its last few still samples, with the time it was seen. Rewritten about once a second.
Ends on SIGINT/SIGTERM.
"""
import json
import os
import signal
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage

STILL_WINDOW_S = 0.8
KEEP = 5
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'logs' / 'part_tracker.json'


def main():
    rclpy.init(); n = rclpy.create_node('part_tracker')
    tcp = deque(maxlen=3000); pending = []; seen = defaultdict(lambda: deque(maxlen=KEEP))
    try:
        table = json.load(open(OUT))
    except Exception:
        table = {}

    def on_tf(m):
        t = time.time()
        for tr in m.transforms:
            p, q = tr.transform.translation, tr.transform.rotation
            pending.append((t, tr.child_frame_id, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))

    n.create_subscription(TFMessage, '/objects_poses_real', on_tf, 10)
    n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose',
                          lambda m: tcp.append((time.time(), m.pose.position.x, m.pose.position.y, m.pose.position.z)), 100)
    stop = {'v': False}
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.__setitem__('v', True))
    last_write = 0.0
    while not stop['v']:
        rclpy.spin_once(n, timeout_sec=0.05)
        keep = []
        for item in pending:
            t_pub = item[0]
            if not tcp or tcp[-1][0] < t_pub:
                keep.append(item); continue
            win = [c for c in tcp if t_pub - STILL_WINDOW_S <= c[0] <= t_pub]
            if len(win) >= 10 and max(np.linalg.norm(np.subtract(c[1:], win[-1][1:])) for c in win) <= 0.0005:
                seen[item[1]].append(item)
        pending[:] = [k for k in keep if time.time() - k[0] < 5.0]
        if time.time() - last_write > 1.0 and seen:
            for name, S in seen.items():
                P = np.median(np.array([s[2] for s in S]), axis=0)
                table[name] = {'xyz': [round(float(v), 4) for v in P], 'quat_xyzw': list(S[-1][3]), 't': S[-1][0],
                               'n': len(S), 'tcp_xyz': [round(v, 3) for v in tcp[-1][1:]] if tcp else None}
            tmp = OUT.with_suffix('.tmp')
            json.dump(table, open(tmp, 'w'), indent=1); os.replace(tmp, OUT)
            last_write = time.time()
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
