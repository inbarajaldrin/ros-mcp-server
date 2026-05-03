#!/bin/bash
# Launch UR5e bringup (real or fake hardware) with the OnRobot RG2 visualization.
#
# Usage:
#   launch_robot.sh real            # real robot at 192.168.1.111, no RViz
#   launch_robot.sh real --rviz     # real + RViz with dual-RobotModel config
#   launch_robot.sh fake            # fake hardware + RG2 publisher + static TF, no RViz
#   launch_robot.sh fake --rviz     # fake + RViz (Phase 7 visualization)
#   launch_robot.sh real --ip 192.168.1.222   # override robot IP
#
# Side effects:
#   - Spawns bringup in background with nohup
#   - Logs to /tmp/ur_bringup_logs/<mode>_bringup_<ts>.log
#   - Waits up to ~30s for controller_manager to come up
#   - Activates scaled_joint_trajectory_controller automatically
#   - Optionally launches RViz if --rviz flag is present
#
# Pendant note (real mode):
#   After this script returns, you still need to power-on + brake-release + load
#   external_control.urp + Press Play on the pendant (operator runs in Local mode
#   per CONVENTIONS — dashboard /play is blocked).
#
# To stop: ./close_robot.sh

set -u   # bail on undefined vars

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="/tmp/ur_bringup_logs"
mkdir -p "$LOG_DIR"

# --- arg parsing ---
MODE=""
ROBOT_IP="192.168.1.111"
USE_RVIZ=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        real|fake) MODE="$1"; shift ;;
        --rviz) USE_RVIZ=true; shift ;;
        --ip) ROBOT_IP="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "ERROR: missing mode (real|fake)" >&2
    echo "Usage: $0 real|fake [--rviz] [--ip <ROBOT_IP>]" >&2
    exit 1
fi

# --- safety: refuse to launch if a bringup is already running ---
if pgrep -f "ros2.*launch.*ur" >/dev/null 2>&1; then
    echo "ERROR: a UR bringup is already running. Run './close_robot.sh' first." >&2
    pgrep -af "ros2.*launch.*ur" | head -3
    exit 2
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${MODE}_bringup_${TS}.log"

# --- source ROS env ---
source_ros() {
    source /opt/ros/humble/setup.bash 2>/dev/null
    source /home/aaugus11/ros2_ws/install/setup.bash 2>/dev/null
    [[ -f /home/aaugus11/Desktop/ros2_ws/install/setup.bash ]] && \
        source /home/aaugus11/Desktop/ros2_ws/install/setup.bash 2>/dev/null
}

# --- launch ---
echo "[launch_robot] Mode: $MODE  RViz: $USE_RVIZ  Log: $LOG_FILE"

if [[ "$MODE" == "real" ]]; then
    echo "[launch_robot] Pinging robot at $ROBOT_IP ..."
    if ! timeout 2 ping -c 1 "$ROBOT_IP" >/dev/null 2>&1; then
        echo "ERROR: robot at $ROBOT_IP not reachable. Check cables/network." >&2
        exit 3
    fi
    echo "[launch_robot] Reachable. Launching real-hardware bringup..."
    nohup bash -c "$(declare -f source_ros); source_ros; \
        ros2 launch ur_bringup ur5e.launch.py ur_type:=ur5e robot_ip:=$ROBOT_IP" \
        > "$LOG_FILE" 2>&1 &
    LAUNCH_PID=$!
elif [[ "$MODE" == "fake" ]]; then
    echo "[launch_robot] Launching fake-hardware bringup with RG2 visualization..."
    RVIZ_ARG=""
    [[ "$USE_RVIZ" == "true" ]] && RVIZ_ARG="rviz:=true" || RVIZ_ARG="rviz:=false"
    nohup bash -c "$(declare -f source_ros); source_ros; \
        ros2 launch $REPO_ROOT/compliant_insertion_studio/launch/ur5e_with_rg2.launch.py $RVIZ_ARG" \
        > "$LOG_FILE" 2>&1 &
    LAUNCH_PID=$!
fi

echo "[launch_robot] Bringup PID: $LAUNCH_PID"

# --- wait for controller_manager to be live ---
echo "[launch_robot] Waiting for controller_manager + force_torque_sensor_broadcaster..."
for i in $(seq 1 15); do
    sleep 2
    CM_OK=$(timeout 1 ros2 control list_controllers 2>/dev/null | grep -c "force_torque_sensor_broadcaster.*active" || true)
    if [[ "$CM_OK" -ge "1" ]]; then
        echo "[launch_robot] Up (after ${i}x2s)."
        break
    fi
done
if [[ "$CM_OK" -lt "1" ]]; then
    echo "WARN: bringup didn't fully come up in 30s. Check log: $LOG_FILE" >&2
    tail -10 "$LOG_FILE" | sed 's/^/  log: /'
    exit 4
fi

# --- activate position controller ---
echo "[launch_robot] Activating scaled_joint_trajectory_controller..."
ros2 control switch_controllers --activate scaled_joint_trajectory_controller 2>&1 | tail -1
sleep 0.5

# --- launch RViz separately if real --rviz (fake mode bakes it into the launch) ---
if [[ "$USE_RVIZ" == "true" && "$MODE" == "real" ]]; then
    echo "[launch_robot] Launching RViz with dual-RobotModel config..."
    nohup bash -c "$(declare -f source_ros); source_ros; \
        rviz2 -d $REPO_ROOT/compliant_insertion_studio/rviz/ur5e_with_rg2.rviz" \
        > "$LOG_DIR/rviz_${TS}.log" 2>&1 &
    sleep 2
fi

# --- summary ---
echo ""
echo "============================================================"
echo "  Bringup ready. Mode: $MODE"
echo "============================================================"
ros2 control list_controllers 2>&1 | grep -E "force_mode|scaled_joint|force_torque|joint_state" | head -5
echo ""
if [[ "$MODE" == "real" ]]; then
    echo "  NEXT STEPS (real mode — pendant in Local):"
    echo "    1. Power on the robot (touch red panel, follow boot)"
    echo "    2. Release brakes (ON button)"
    echo "    3. Load external_control.urp"
    echo "    4. Press Play ▶️"
    echo ""
    echo "  Verify program running with:"
    echo "    ros2 service call /dashboard_client/program_running ur_dashboard_msgs/srv/IsProgramRunning"
fi
echo ""
echo "  Stop with: $(dirname "$0")/close_robot.sh"
echo "  Log: $LOG_FILE"
