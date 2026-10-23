#!/usr/bin/env python3
"""Publish what the depth camera sees as a picture: /depth_view (bgr8), height above the table in colour.

Operator 2026-10-20: "i dont have a visualization layer to see what the depth camera is seeing". Every seat
check, base outline and nudge decision reads this depth; this shows it. Each aligned depth pixel goes to the
robot frame through the wrist camera mount (the localizer's config, the same X_CAM the checks use) and the
TCP pose, and is coloured by its height above the table: table dark, base top (~35 mm) green, a seated bar's
top (~55 mm above the table) yellow/red; holes (no depth) black. A scale bar and the min/max sit at the bottom.
~0.5 Hz, quarter resolution (the OptiPlex is CPU-bound; the localizer comes first).

    python3 -u compliant_insertion_studio/scripts/depth_view.py      # then add /depth_view as an Image panel
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))
from seat_check import _load_x_cam   # noqa: E402

TABLE_Z = -0.0625 - 0.0175            # world z of the table (base origin -62.5 mm, base half-height 17.5 mm)
H_MAX_MM = 80.0                       # colour scale 0..80 mm above the table
PERIOD_S = 2.0


def main():
    rclpy.init()
    n = rclpy.create_node('depth_view')
    br = CvBridge(); st = {}
    X = _load_x_cam()
    n.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', lambda m: st.__setitem__('d', m), 2)
    n.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', lambda m: st.__setitem__('i', m), 2)
    n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: st.__setitem__('t', m), 10)
    pub = n.create_publisher(Image, 'depth_view', 2)
    last = 0.0
    while rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.05)
        if time.time() - last < PERIOD_S or not all(k in st for k in 'dit'):
            continue
        last = time.time()
        d = br.imgmsg_to_cv2(st['d'], 'passthrough').astype(np.float32)[::4, ::4] / 1000.0
        K = np.array(st['i'].k).reshape(3, 3).copy(); K[:2] /= 4.0
        p, o = st['t'].pose.position, st['t'].pose.orientation
        Tw = np.eye(4); Tw[:3, :3] = R.from_quat([o.x, o.y, o.z, o.w]).as_matrix(); Tw[:3, 3] = [p.x, p.y, p.z]
        Tc = Tw @ X
        hgt, wid = d.shape
        u, v = np.meshgrid(np.arange(wid), np.arange(hgt))
        Pc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
        zw = Pc @ Tc[2, :3] + Tc[2, 3]                                    # world z only
        h = 1000.0 * (zw - TABLE_Z)
        valid = d > 0.1
        g = np.clip(h / H_MAX_MM, 0, 1)
        img = cv2.applyColorMap((255 * g).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[~valid] = 0
        img[valid & (h < 3.0)] = (img[valid & (h < 3.0)] * 0.35).astype(np.uint8)   # table: dimmed
        # scale bar
        bar = cv2.applyColorMap(np.linspace(0, 255, wid - 20).astype(np.uint8)[None, :].repeat(10, 0), cv2.COLORMAP_TURBO)
        img[hgt - 22:hgt - 12, 10:wid - 10] = bar
        for mm in (0, 17.5, 35, 55, 80):
            x = int(10 + (wid - 20) * mm / H_MAX_MM)
            cv2.putText(img, f"{mm:g}", (max(0, x - 8), hgt - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(img, "height above table (mm): base top 35, bar top 54.5 when seated", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        m = br.cv2_to_imgmsg(img, 'bgr8'); m.header = st['d'].header
        pub.publish(m)
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
