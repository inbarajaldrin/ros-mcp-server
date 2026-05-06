#!/bin/bash
# Launch the aruco_camera_localizer node (publishes /objects_poses_real,
# /camera_pose, /annotated_stream) used to read held-part / base orientations
# before invoking the compliant_insert wrapper.
#
# Usage:
#   launch_camera.sh                    # foreground (Ctrl-C to stop)
#   launch_camera.sh --background       # nohup background, log to /tmp/aruco_logs/
#
# Side effects:
#   - Sources /opt/ros/humble + /home/aaugus11/Desktop/ros2_ws/install
#   - Runs `ros2 run aruco_camera_localizer localize --suppress-prints`
#   - In --background mode logs to /tmp/aruco_logs/aruco_<ts>.log
#
# After it's up, read u_brown's current quaternion with:
#   ros2 topic echo --once /objects_poses_real | grep -A8 "child_frame_id: u_brown"
# (or use the helper at /home/aaugus11/Desktop/ros2_ws/src/aruco_camera_localizer/utils/read_object_pose.py)

BACKGROUND=false
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --background|-b) BACKGROUND=true; shift ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

set +u
source /opt/ros/humble/setup.bash 2>/dev/null
source /home/aaugus11/Desktop/ros2_ws/install/setup.bash 2>/dev/null
set -u

if ! ros2 pkg list 2>/dev/null | grep -q '^aruco_camera_localizer$'; then
    echo "ERROR: aruco_camera_localizer not found on the ROS2 path." >&2
    echo "       Did you build /home/aaugus11/Desktop/ros2_ws? (colcon build)" >&2
    exit 1
fi

if pgrep -f "aruco_camera_localizer.*localize" >/dev/null 2>&1; then
    echo "ERROR: an aruco localizer process is already running:" >&2
    pgrep -af "aruco_camera_localizer.*localize" | head -3
    echo "       Stop it first (kill -SIGINT <pid>) before re-launching." >&2
    exit 2
fi

CMD=(ros2 run aruco_camera_localizer localize --suppress-prints "${EXTRA_ARGS[@]}")

if [[ "$BACKGROUND" == "true" ]]; then
    LOG_DIR="/tmp/aruco_logs"
    mkdir -p "$LOG_DIR"
    TS=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/aruco_${TS}.log"
    echo "[launch_camera] Background mode. Log: $LOG_FILE"
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "[launch_camera] PID: $PID"
    sleep 3
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "ERROR: aruco localizer exited within 3s. Tail of log:" >&2
        tail -20 "$LOG_FILE" | sed 's/^/  log: /' >&2
        exit 3
    fi
    echo "[launch_camera] Up. Verify topic with:"
    echo "  ros2 topic hz /objects_poses_real"
    echo "  ros2 topic echo --once /objects_poses_real"
    echo "Stop with: kill -SIGINT $PID   (or pkill -SIGINT -f aruco_camera_localizer)"
else
    echo "[launch_camera] Foreground. Ctrl-C to stop."
    exec "${CMD[@]}"
fi
