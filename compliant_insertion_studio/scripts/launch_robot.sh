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
HEADLESS=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        real|fake) MODE="$1"; shift ;;
        --rviz) USE_RVIZ=true; shift ;;
        --headless) HEADLESS=true; shift ;;
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

# Both real and fake modes use the SAME launch file (Phase 7's
# ur5e_with_rg2.launch.py) so the RG2 robot_state_publisher + static TF
# tool0 -> rg2_base_link are present in BOTH modes — operator sees the
# gripper in RViz regardless. Only the use_fake_hardware + robot_ip args
# change.
RVIZ_ARG="rviz:=false"
[[ "$USE_RVIZ" == "true" ]] && RVIZ_ARG="rviz:=true"

if [[ "$MODE" == "real" ]]; then
    echo "[launch_robot] Pinging robot at $ROBOT_IP ..."
    if ! timeout 2 ping -c 1 "$ROBOT_IP" >/dev/null 2>&1; then
        echo "ERROR: robot at $ROBOT_IP not reachable. Check cables/network." >&2
        exit 3
    fi
    echo "[launch_robot] Reachable. Launching real-hardware bringup with RG2 visualization..."
    HW_ARGS="use_fake_hardware:=false robot_ip:=$ROBOT_IP"
    # --headless: the driver sends its own control script over the primary
    # interface, so no External Control .urp has to be loaded and played on the
    # pendant. Requires the pendant in REMOTE CONTROL mode. reverse_ip is this
    # host's address on the robot's subnet — derived from the route to the robot.
    if [[ "$HEADLESS" == "true" ]]; then
        REVERSE_IP=$(ip -4 route get "$ROBOT_IP" 2>/dev/null | grep -oE 'src [0-9.]+' | awk '{print $2}')
        if [[ -z "${REVERSE_IP:-}" ]]; then
            echo "ERROR: could not derive reverse_ip (this host's address on $ROBOT_IP's subnet)." >&2
            exit 3
        fi
        echo "[launch_robot] Headless external control — reverse_ip=$REVERSE_IP"
        HW_ARGS="$HW_ARGS headless_mode:=true reverse_ip:=$REVERSE_IP"
    fi
elif [[ "$MODE" == "fake" ]]; then
    echo "[launch_robot] Launching fake-hardware bringup with RG2 visualization..."
    HW_ARGS="use_fake_hardware:=true fake_sensor_commands:=true"
fi

nohup bash -c "$(declare -f source_ros); source_ros; \
    ros2 launch $REPO_ROOT/compliant_insertion_studio/launch/ur5e_with_rg2.launch.py \
        $RVIZ_ARG $HW_ARGS" \
    > "$LOG_FILE" 2>&1 &
LAUNCH_PID=$!

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

# --- set payload from foundational calibration (real mode only) ---
# CRITICAL: without this, force_mode controller has no gravity comp for the held
# part. UR pendant payload setting does NOT propagate to the ROS2 driver's
# force_mode — they're separate. With 0kg payload, ~21N of gravity load on the
# held u_orange goes uncompensated, force_mode yields, TCP drifts ~8mm during
# APPROACH (verified empirically 2026-05-06). With set_payload at the calibrated
# values, drift drops to <1mm. Source: ft_calibration_*.yaml in configs/.
if [[ "$MODE" == "real" ]]; then
    CAL_YAML=$(ls -t "$REPO_ROOT"/compliant_insertion_studio/configs/ft_calibration_*.yaml 2>/dev/null | head -1)
    if [[ -n "$CAL_YAML" ]]; then
        # Extract mass + cog from YAML using a one-liner python (avoid yq dependency)
        eval "$(python3 -c "
import yaml
with open('$CAL_YAML') as fh: d = yaml.safe_load(fh)
m = d['result']['mass_kg']
c = d['result']['cog_xyz_m']
print(f'PAYLOAD_MASS={m}')
print(f'PAYLOAD_COG_X={c[0]}')
print(f'PAYLOAD_COG_Y={c[1]}')
print(f'PAYLOAD_COG_Z={c[2]}')
" 2>/dev/null)"
        if [[ -n "${PAYLOAD_MASS:-}" ]]; then
            echo "[launch_robot] Setting driver payload from $(basename "$CAL_YAML"):"
            echo "    mass=${PAYLOAD_MASS} kg  cog=(${PAYLOAD_COG_X}, ${PAYLOAD_COG_Y}, ${PAYLOAD_COG_Z}) m"
            ros2 service call /io_and_status_controller/set_payload ur_msgs/srv/SetPayload \
                "{mass: ${PAYLOAD_MASS}, center_of_gravity: {x: ${PAYLOAD_COG_X}, y: ${PAYLOAD_COG_Y}, z: ${PAYLOAD_COG_Z}}}" \
                2>&1 | grep -E "success|response" | tail -2
        else
            echo "WARN: $(basename "$CAL_YAML") missing result.mass_kg / result.cog_xyz_m keys — payload NOT set." >&2
            echo "      Force_mode will have ~21N of unmodeled gravity. Drift WILL be excessive." >&2
        fi
    else
        echo "WARN: no ft_calibration_*.yaml in configs/ — payload NOT set." >&2
        echo "      Run ft_calibration.py first or paste calibrated values manually." >&2
    fi
fi

# RViz is now launched by the unified ur5e_with_rg2.launch.py via rviz:=true,
# so no separate RViz spawn is needed regardless of real/fake mode.

# --- summary ---
echo ""
echo "============================================================"
echo "  Bringup ready. Mode: $MODE"
echo "============================================================"
ros2 control list_controllers 2>&1 | grep -E "force_mode|scaled_joint|force_torque|joint_state" | head -5
echo ""
if [[ "$MODE" == "real" ]]; then
    echo "  NEXT STEPS (real mode — pendant in Local):"
    echo "    First-time / cold pendant:"
    echo "      1. Power on the robot (touch red panel, follow boot)"
    echo "      2. Release brakes (ON button)"
    echo "      3. Load external_control.urp"
    echo "      4. Press Play ▶️"
    echo ""
    echo "    If pendant was ALREADY running external_control.urp from before"
    echo "    this bringup launched (e.g. you ran close_robot.sh then this):"
    echo "      • Press STOP ⏹ on the pendant, then PLAY ▶️ again."
    echo "      • This re-links the External Control URCap to the new bringup."
    echo "      • Without this cycle, ros2_control trajectories will be"
    echo "        rejected with 'Velocity or acceleration limits exceeded.'"
    echo "        even though program_running reports true (the play is stale)."
    echo "      • Dashboard /play and /stop are blocked in Local mode, so"
    echo "        this must be done manually on the pendant."
    echo ""
    echo "  Verify program running with:"
    echo "    ros2 service call /dashboard_client/program_running ur_dashboard_msgs/srv/IsProgramRunning"
    echo ""
    echo "  Verify URCap link is fresh by sending a tiny joint move and"
    echo "  watching for 'Velocity or acceleration limits exceeded.'"
fi
echo ""
echo "  Stop with: $(dirname "$0")/close_robot.sh"
echo "  Log: $LOG_FILE"
