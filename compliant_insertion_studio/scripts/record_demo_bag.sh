#!/bin/bash
# record_demo_bag.sh OUT_DIR — record what a demo video needs as a rosbag (mcap), lightly (2026-10-17).
#   compressed RGB (~15 Hz, JPEG from the RealSense node), part poses, grasp points, TCP pose, camera pose
# ~2.4 MB/s (~1 GB per 7-min assembly), ~12% of one core. Video + localizer overlay are made afterwards
# by render_bag.py on another machine — never live-encode on the OptiPlex: two live H.264 recorders
# pushed its load 5 -> 12 and starved the CPU-bound localizer (parts dropped / misread, runs stopped).
# Stop with SIGINT (finalizes the bag).
OUT=$1
[ -n "$OUT" ] || { echo "usage: $0 OUT_DIR"; exit 2; }
FREE_GB=$(df --output=avail -BG "$(dirname "$OUT")" | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge 10 ] || { echo "refusing: only ${FREE_GB} GB free"; exit 1; }
source /opt/ros/humble/setup.bash
exec nice -n 19 ros2 bag record -s mcap -o "$OUT" \
  /camera/camera/color/image_raw/compressed /objects_poses_real /grasp_points_real \
  /tcp_pose_broadcaster/pose /camera_pose
