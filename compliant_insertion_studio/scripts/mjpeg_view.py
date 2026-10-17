#!/usr/bin/env python3
"""Camera in any browser, no installs: serve a ROS CompressedImage topic as MJPEG (2026-10-17).

    python3 compliant_insertion_studio/scripts/mjpeg_view.py [--topic T] [--fps 5] [--port 8080]
    then open  http://<this PC's Tailscale IP>:8080/   (e.g. from the Mac)

Foxglove needs its app/layout set up and the web app could not open ws:// from https; this is a
plain page. Frames are passed through as JPEG (no re-encode), throttled to --fps for slow links.
"""
import argparse, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from sensor_msgs.msg import CompressedImage

ap = argparse.ArgumentParser()
ap.add_argument('--topic', default='/camera/camera/color/image_raw/compressed')
ap.add_argument('--fps', type=float, default=5.0)
ap.add_argument('--port', type=int, default=8080)
ap.add_argument('--bind', default='0.0.0.0')
a = ap.parse_args()

latest = {'jpg': None, 't': 0.0}
rclpy.init(); node = rclpy.create_node('mjpeg_view')
node.create_subscription(CompressedImage, a.topic,
                         lambda m: latest.update(jpg=bytes(m.data), t=time.time()), 2)
threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith('/stream'):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    jpg = latest['jpg']
                    if jpg:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                         + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                    time.sleep(1.0 / a.fps)
            except (BrokenPipeError, ConnectionResetError):
                return
        page = (f"<html><head><title>FMB1 camera</title></head><body style='margin:0;background:#111'>"
                f"<img src='/stream' style='width:100%;height:auto'>"
                f"<div style='color:#aaa;font:12px sans-serif;padding:4px'>{a.topic} @ {a.fps:g} fps</div>"
                f"</body></html>").encode()
        self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
        self.wfile.write(page)


ThreadingHTTPServer((a.bind, a.port), H).serve_forever()
