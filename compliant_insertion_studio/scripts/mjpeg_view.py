#!/usr/bin/env python3
"""Camera in any browser, no installs: serve ROS image topics as MJPEG (2026-10-17; several topics 2026-10-20).

    python3 compliant_insertion_studio/scripts/mjpeg_view.py [--topic T ...] [--fps 5] [--port 8080]
    then open  http://<this PC's Tailscale IP>:8080/   (e.g. from the Mac)

Default page (2026-10-20, operator): the annotated stream (localizer overlay + the robot +-x/+-y compass),
the depth height map (depth_view.py: what the depth checks see, height above the table in colour) and the
raw colour camera, side by side. /stream/<i> is one topic alone.

Topics ending in /compressed are passed through as JPEG (no re-encode); raw sensor_msgs/Image topics are
JPEG-encoded here, at most --fps per topic (the OptiPlex is CPU-bound; the localizer comes first).
Foxglove needs its app/layout set up and the web app could not open ws:// from https; this is a plain page.
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from sensor_msgs.msg import CompressedImage, Image

ap = argparse.ArgumentParser()
ap.add_argument('--topic', action='append', default=None,
                help='repeatable; default: /annotated_stream, /depth_view, /camera/camera/color/image_raw/compressed')
ap.add_argument('--fps', type=float, default=5.0)
ap.add_argument('--port', type=int, default=8080)
ap.add_argument('--bind', default='0.0.0.0')
a = ap.parse_args()
TOPICS = a.topic or ['/annotated_stream', '/depth_view', '/camera/camera/color/image_raw/compressed']

latest = [{'jpg': None, 't': 0.0, 'raw': None} for _ in TOPICS]
rclpy.init(); node = rclpy.create_node('mjpeg_view')


def _raw_cb(i):
    def cb(m):
        latest[i]['raw'] = m          # encoded lazily, only while someone is watching
    return cb


for i, t in enumerate(TOPICS):
    if t.endswith('/compressed'):
        node.create_subscription(CompressedImage, t, lambda m, i=i: latest[i].update(jpg=bytes(m.data), t=time.time()), 2)
    else:
        node.create_subscription(Image, t, _raw_cb(i), 2)
threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()


def frame(i):
    L = latest[i]
    if L['raw'] is not None:
        import cv2
        import numpy as np
        m, L['raw'] = L['raw'], None
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        if m.encoding == 'rgb8':
            img = img[:, :, ::-1]
        if img.shape[1] > 960:
            img = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            L['jpg'] = buf.tobytes(); L['t'] = time.time()
    return L['jpg']


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith('/stream'):
            try:
                i = int(self.path.split('/')[2]) if self.path.count('/') >= 2 and self.path.split('/')[2] else 0
            except ValueError:
                i = 0
            i = min(max(i, 0), len(TOPICS) - 1)
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    jpg = frame(i)
                    if jpg:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                         + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                    time.sleep(1.0 / a.fps)
            except (BrokenPipeError, ConnectionResetError):
                return
        cells = ''.join(
            f"<div style='flex:1 1 30%;min-width:300px'><img src='/stream/{i}' style='width:100%;height:auto'>"
            f"<div style='color:#aaa;font:12px sans-serif;padding:4px'>{t} - <a style='color:#8af' href='/stream/{i}'>alone</a></div></div>"
            for i, t in enumerate(TOPICS))
        page = (f"<html><head><title>Cell cameras</title><meta name='viewport' content='width=device-width'></head>"
                f"<body style='margin:0;background:#111'><div style='display:flex;flex-wrap:wrap;gap:4px'>{cells}</div>"
                f"<div style='color:#777;font:12px sans-serif;padding:4px'>@ {a.fps:g} fps per stream</div></body></html>").encode()
        self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
        self.wfile.write(page)


ThreadingHTTPServer((a.bind, a.port), H).serve_forever()
